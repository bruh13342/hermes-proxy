#!/usr/bin/env python3
"""Best-effort OpenAI-compatible local proxy for the supplied upstream endpoint.

Install: pip install fastapi uvicorn httpx
Run (PowerShell): $env:PROXY_API_KEY="change-this-key"; python proxy.py
Run (macOS/Linux): PROXY_API_KEY="change-this-key" python proxy.py
"""
import json
import os
import time
import uuid
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM_URL = "https://allaitools-generate.freemodels.workers.dev/api/chat"
UPSTREAM_ORIGIN = "https://rusherai.com"
UPSTREAM_REFERER = "https://rusherai.com/"
LOCAL_API_KEY = os.getenv("PROXY_API_KEY", "change-this-key")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-6-astra")
# The upstream accepts any modelId, so this is just the list shown to clients
# (e.g. Hermes' model picker). Override with PROXY_MODELS="a,b,c".
MODELS = [m.strip() for m in os.getenv(
    "PROXY_MODELS",
    # Exact modelId slugs used by rusherai.com (display name in comment):
    "gpt-6-astra,"       # GPT 6 Astra
    "claude-opus-5.5,"   # Claude Opus 5.5
    "claude-sonnet-5,"   # Claude Sonnet 5.5
    "sol,"               # GPT 5.6 Sol
    "terra,"             # GPT 5.6 Terra
    "claude-fable-5,"    # Claude Fable 5
    "claude-fable-5.1,"  # Claude Fable 5.1
    "glm-5.2,"           # GLM 5.2
    "kimi-k3",           # Kimi K3
).split(",") if m.strip()]
app = FastAPI(title="Local OpenAI-Compatible Proxy", version="0.1.0")


def authorize(authorization: Optional[str]) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer API key")
    if authorization[7:].strip() != LOCAL_API_KEY:
        raise HTTPException(401, "Invalid API key")


def get_text(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return None
    choices = obj.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        first = choices[0]
        message = first.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        delta = first.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            return delta["content"]
    for key in ("content", "text", "response", "output", "answer"):
        if isinstance(obj.get(key), str):
            return obj[key]
    message = obj.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return None


def parse_sse(line: str) -> Optional[str]:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        return get_text(json.loads(data)) or None
    except json.JSONDecodeError:
        return data


def chunk(model: str, text: str, ident: str) -> str:
    item = {
        "id": ident, "object": "chat.completion.chunk", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/models")
async def models(authorization: Optional[str] = Header(default=None)):
    authorize(authorization)
    created = int(time.time())
    return {"object": "list", "data": [
        {"id": m, "object": "model", "created": created, "owned_by": "upstream"} for m in MODELS
    ]}


@app.post("/v1/chat/completions")
async def completions(request: Request, authorization: Optional[str] = Header(default=None)):
    authorize(authorization)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Body must be valid JSON")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages must be a non-empty list")

    model = body.get("model") or DEFAULT_MODEL
    stream = bool(body.get("stream", False))

    # Flatten OpenAI messages into {role, content-string} for the upstream.
    upstream_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        upstream_messages.append({
            "role": msg.get("role", "user"),
            "content": content if isinstance(content, str) else "",
        })

    payload = {
        "messages": upstream_messages,
        "modelId": model,
        "thinking": bool(body.get("thinking", False)),
        "deepSearch": bool(body.get("deepSearch", False)),
        "stream": True,  # always stream from upstream; we aggregate if needed
    }
    up_headers = {
        "Origin": UPSTREAM_ORIGIN,
        "Referer": UPSTREAM_REFERER,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    ident = "chatcmpl-" + uuid.uuid4().hex

    if stream:
        async def event_stream():
            # first chunk carries the assistant role (OpenAI convention)
            first = {
                "id": ident, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield "data: " + json.dumps(first) + "\n\n"
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", UPSTREAM_URL, headers=up_headers, json=payload) as resp:
                    if resp.status_code != 200:
                        raw = await resp.aread()
                        err = {"error": {"message": f"upstream {resp.status_code}: {raw.decode('utf-8', 'replace')[:500]}"}}
                        yield "data: " + json.dumps(err) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.strip() == "data: [DONE]":
                            break
                        text = parse_sse(line)
                        if text:
                            yield chunk(model, text, ident)
            final = {
                "id": ident, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield "data: " + json.dumps(final) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # non-streaming: aggregate the upstream stream into one answer
    parts = []
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", UPSTREAM_URL, headers=up_headers, json=payload) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                raise HTTPException(502, f"upstream {resp.status_code}: {raw.decode('utf-8', 'replace')[:500]}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.strip() == "data: [DONE]":
                    break
                text = parse_sse(line)
                if text:
                    parts.append(text)

    return JSONResponse({
        "id": ident, "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "".join(parts)},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8788"))
    print("=" * 66)
    print(" rusherai -> Hermes proxy")
    print("-" * 66)
    print(f" Base URL : http://{host}:{port}/v1")
    print(f" API key  : {LOCAL_API_KEY}   (set PROXY_API_KEY to change)")
    print(f" Models   : {', '.join(MODELS)}")
    print("=" * 66)
    uvicorn.run(app, host=host, port=port)