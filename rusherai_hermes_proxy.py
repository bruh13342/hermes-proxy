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


# --------------------------------------------------------------------------- #
# Tool-calling emulation                                                      #
#                                                                             #
# The upstream has no native function-calling. So when a client sends `tools`,#
# we (1) describe them in the prompt and ask for a JSON tool-call, then       #
# (2) parse the model's reply back into OpenAI `tool_calls`.                   #
# --------------------------------------------------------------------------- #

def tool_names(tools: Any) -> list:
    names = []
    for t in tools or []:
        if isinstance(t, dict) and t.get("type") == "function":
            fn = t.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                names.append(fn["name"])
    return names


def build_tool_prompt(tools: Any, tool_choice: Any) -> str:
    lines = ["You have access to the following tools (functions):", ""]
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function", {})
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        lines.append("- %s: %s" % (fn["name"], fn.get("description", "") or ""))
        params = fn.get("parameters")
        if params:
            try:
                lines.append("  parameters (JSON Schema): %s" % json.dumps(params))
            except Exception:
                pass
    lines += [
        "",
        "To use tools, reply with ONLY this JSON object and nothing else "
        "(no prose, no markdown fences):",
        '{"tool_calls": [{"name": "<tool name>", "arguments": {<arguments object>}}]}',
        "You may put more than one call in the array.",
    ]
    if isinstance(tool_choice, dict):
        forced = (tool_choice.get("function") or {}).get("name")
        if forced:
            lines.append('You MUST call the tool "%s" now.' % forced)
    elif tool_choice == "required":
        lines.append("You MUST call at least one tool now.")
    else:
        lines.append("If no tool is needed, reply normally in plain text instead of JSON.")
    return "\n".join(lines)


def first_json_object(s: str) -> Optional[str]:
    """Return the first balanced {...} block in s, respecting strings/escapes."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def extract_tool_calls(text: str, valid_names: set) -> Optional[list]:
    """Parse a model reply into OpenAI tool_calls, or None if it isn't one."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        block = first_json_object(s)
        if block:
            try:
                obj = json.loads(block)
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None

    calls = obj.get("tool_calls")
    if calls is None and isinstance(obj.get("tool_call"), dict):
        calls = [obj["tool_call"]]
    if calls is None and "name" in obj and "arguments" in obj:
        calls = [obj]
    if not isinstance(calls, list) or not calls:
        return None

    out = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        # only accept calls to tools the client actually offered
        if not name or (valid_names and name not in valid_names):
            continue
        args = c.get("arguments", {})
        if not isinstance(args, str):
            try:
                args = json.dumps(args)
            except Exception:
                args = "{}"
        out.append({
            "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": args},
        })
    return out or None


async def aggregate_upstream(payload: dict, up_headers: dict):
    """Collect the full upstream answer. Returns (status_code, text)."""
    parts = []
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", UPSTREAM_URL, headers=up_headers, json=payload) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                return resp.status_code, raw.decode("utf-8", "replace")[:500]
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.strip() == "data: [DONE]":
                    break
                text = parse_sse(line)
                if text:
                    parts.append(text)
    return 200, "".join(parts)


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
    tools = body.get("tools")
    tool_choice = body.get("tool_choice", "auto")
    # The upstream has no native function-calling; emulate it when tools are sent.
    tools_active = bool(tools) and tool_choice != "none"

    # Flatten OpenAI messages into {role, content-string} for the upstream.
    #
    # The rusherai upstream only honors "user"/"assistant" turns: it ignores the
    # "system" role and does not understand "tool"/"function". Clients like
    # Hermes put their agent instructions (and context) in the system prompt and
    # carry tool calls/results in the history, so we:
    #   - collect system/developer messages into a preamble folded into the
    #     first user turn (the upstream obeys instructions there),
    #   - render assistant tool_calls and tool results as text so the model can
    #     follow the tool conversation,
    #   - map any other role to a user turn so nothing is silently lost.
    system_preamble = []
    upstream_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if not isinstance(content, str):
            content = ""

        if role in ("system", "developer"):
            if content.strip():
                system_preamble.append(content)
            continue

        if role == "assistant":
            calls = msg.get("tool_calls")
            if isinstance(calls, list) and calls:
                rendered = []
                for call in calls:
                    fn = call.get("function", {}) if isinstance(call, dict) else {}
                    rendered.append("[called tool %s with arguments %s]" % (
                        fn.get("name", ""), fn.get("arguments", "{}")))
                content = (content + "\n" if content else "") + "\n".join(rendered)
            upstream_messages.append({"role": "assistant", "content": content})
            continue

        if role == "tool":
            name = msg.get("name") or msg.get("tool_call_id") or "tool"
            upstream_messages.append({
                "role": "user", "content": "[result from tool %s]:\n%s" % (name, content)})
            continue

        if role != "user":
            role = "user"
        upstream_messages.append({"role": role, "content": content})

    if tools_active:
        system_preamble.append(build_tool_prompt(tools, tool_choice))

    if system_preamble:
        preamble = "\n\n".join(system_preamble)
        for m in upstream_messages:
            if m["role"] == "user":
                m["content"] = preamble + "\n\n" + m["content"]
                break
        else:
            upstream_messages.insert(0, {"role": "user", "content": preamble})

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

    # ---- tool-calling path: aggregate, parse, return tool_calls or text ---- #
    if tools_active:
        status_code, full = await aggregate_upstream(payload, up_headers)
        if status_code != 200:
            raise HTTPException(502, f"upstream {status_code}: {full}")
        calls = extract_tool_calls(full, set(tool_names(tools)))
        created = int(time.time())

        if stream:
            async def tool_stream():
                yield "data: " + json.dumps({
                    "id": ident, "object": "chat.completion.chunk", "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }) + "\n\n"
                if calls:
                    delta_calls = [{
                        "index": i, "id": c["id"], "type": "function",
                        "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]},
                    } for i, c in enumerate(calls)]
                    yield "data: " + json.dumps({
                        "id": ident, "object": "chat.completion.chunk", "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"tool_calls": delta_calls}, "finish_reason": None}],
                    }) + "\n\n"
                    finish = "tool_calls"
                else:
                    if full:
                        yield chunk(model, full, ident)
                    finish = "stop"
                yield "data: " + json.dumps({
                    "id": ident, "object": "chat.completion.chunk", "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                }) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(tool_stream(), media_type="text/event-stream")

        message = {"role": "assistant", "content": None if calls else full}
        if calls:
            message["tool_calls"] = calls
        return JSONResponse({
            "id": ident, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

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
