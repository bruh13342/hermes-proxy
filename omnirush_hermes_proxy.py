#!/usr/bin/env python3
"""
omnirush -> Hermes proxy
========================

A tiny local proxy that lets you point Hermes Agent (or any OpenAI-compatible
client) at your omnirush.ai account.

It does three things:

  1. Sign-in once, via omnirush's device-code flow, and save the tokens.
  2. On every start, reuse the saved tokens. If the access token is expired it
     refreshes automatically; only if the refresh is dead does it ask you to
     sign in again.
  3. Run a local HTTP server that speaks the OpenAI API. It translates
     chat-completions requests into omnirush's Responses API (the only model
     format the gateway accepts) and translates the answer back. It also
     exposes /v1/models so Hermes can list the models in your account, and a
     /v1/responses passthrough for clients that speak the Responses API
     directly.

Only the Python standard library is used -- no pip install needed.

    python omnirush_hermes_proxy.py             # sign in if needed, then serve
    python omnirush_hermes_proxy.py --login     # force a fresh sign-in
    python omnirush_hermes_proxy.py --models    # just print the model list
    python omnirush_hermes_proxy.py --port 8787 # choose the port (default 8787)

Then in Hermes add a custom OpenAI-compatible provider:

    Base URL : http://127.0.0.1:8787/v1
    API key  : anything (e.g. "local") -- it is ignored
    Models   : use the ids printed on startup / from /v1/models

A note, same as before: omnirush's free models are meant to be used through
their own app ("your own provider keys are not supported"). Using the token in
another client may break their terms and usage is tied to your account. Use an
account you are willing to risk.
"""

import argparse
import json
import os
import sys
import time
import uuid
import threading
import webbrowser
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

DEFAULT_GATEWAY_URL = "https://omnirush.ai/omnirush/v1"

CONFIG_DIR = Path.home() / ".omnirush-hermes-proxy"
TOKENS_PATH = CONFIG_DIR / "tokens.json"

# The desktop app sends "X-OmniRush-Client: gui/<version>". The server treats a
# request with NO such header as "version unknown" and does not apply the
# client-update deadline, so by default we omit it. If you ever get an HTTP 426
# ("update required"), set this to the current app version, e.g. "3.0.1".
CLIENT_VERSION = None  # e.g. "3.0.1"

# omnirush.ai sits behind Cloudflare, which blocks non-browser user-agents with
# "Error 1010". The real app is Electron, so it sends a Chrome-like UA. We send
# one too; without it every request is refused with HTTP 403. If you ever hit a
# 1010/403 again, bump this to a current Chrome UA string.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

HTTP_TIMEOUT = 600  # seconds, generous for long generations

# in-memory copy of the saved tokens: {access_token, refresh_token, gateway_url}
TOKENS = None
REFRESH_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Small HTTP helpers (stdlib only)                                            #
# --------------------------------------------------------------------------- #

def _client_headers(extra=None):
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if CLIENT_VERSION:
        headers["X-OmniRush-Client"] = "gui/%s" % CLIENT_VERSION
    if extra:
        headers.update(extra)
    return headers


def open_upstream(url, method="GET", headers=None, data=None, stream=False):
    """
    Perform a request and return (status_code, response_object).

    The response object is the raw urllib file-like object; the caller reads or
    streams it. HTTP errors (4xx/5xx) are returned the same way, not raised, so
    the caller can forward the status and body.
    """
    req = urllib.request.Request(url, data=data, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        resp = urllib.request.urlopen(req, timeout=HTTP_TIMEOUT)
        return resp.status, resp
    except urllib.error.HTTPError as err:
        return err.code, err


def request_json(url, method="GET", headers=None, payload=None):
    """Request and parse a JSON body. Returns (status_code, parsed_or_bytes)."""
    data = None
    hdrs = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    status, resp = open_upstream(url, method=method, headers=hdrs, data=data)
    raw = resp.read()
    try:
        return status, json.loads(raw.decode("utf-8"))
    except Exception:
        return status, raw


def control_base(gateway_url):
    """The control-plane root: the gateway URL with a trailing '/' and '/v1' removed."""
    base = gateway_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def gateway_path(gateway_url, path):
    """<gateway>/<path>, e.g. .../omnirush/v1/responses."""
    return gateway_url.rstrip("/") + "/" + path.lstrip("/")


# --------------------------------------------------------------------------- #
# Token storage                                                               #
# --------------------------------------------------------------------------- #

def load_tokens():
    try:
        with open(TOKENS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("access_token") and data.get("refresh_token"):
            data.setdefault("gateway_url", DEFAULT_GATEWAY_URL)
            return data
    except FileNotFoundError:
        return None
    except Exception as exc:
        print("[warn] could not read saved tokens: %s" % exc, file=sys.stderr)
    return None


def save_tokens(tokens):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TOKENS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(tokens, handle, indent=2)
    os.replace(tmp, TOKENS_PATH)
    # Best-effort tighten permissions (no-op / limited on Windows).
    try:
        os.chmod(TOKENS_PATH, 0o600)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Device-code sign-in and refresh                                             #
# --------------------------------------------------------------------------- #

def platform_tag():
    if sys.platform.startswith("win"):
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def device_login(gateway_url=DEFAULT_GATEWAY_URL):
    """Run the full device-code flow and return a fresh token dict."""
    base = control_base(gateway_url)

    status, issued = request_json(
        base + "/device/authorize",
        method="POST",
        headers=_client_headers(),
        payload={"device_name": "hermes-proxy", "platform": platform_tag()},
    )
    if status != 200 or not isinstance(issued, dict):
        raise RuntimeError("could not start sign-in (HTTP %s): %s" % (status, issued))

    device_code = issued.get("device_code")
    verify_url = issued.get("verification_uri_complete")
    user_code = issued.get("user_code", "")
    if not device_code or not verify_url:
        raise RuntimeError("sign-in service returned an invalid device link")

    interval = max(2, int(issued.get("interval") or 3))
    expires_in = max(60, int(issued.get("expires_in") or 600))
    deadline = time.time() + expires_in

    print("\n==================================================================")
    print(" Sign in to omnirush.ai to authorize this proxy")
    print("------------------------------------------------------------------")
    print(" Open this URL in your browser and approve the request:\n")
    print("   %s" % verify_url)
    if user_code:
        print("\n Code: %s" % user_code)
    print("==================================================================\n")
    try:
        webbrowser.open(verify_url)
    except Exception:
        pass

    print("Waiting for approval", end="", flush=True)
    token_url = base + "/device/token"
    while time.time() < deadline:
        time.sleep(interval)
        print(".", end="", flush=True)
        status, payload = request_json(
            token_url,
            method="POST",
            headers=_client_headers(),
            payload={"device_code": device_code},
        )
        if status == 428:
            continue  # still waiting for the user to approve
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError("\nsign-in failed (HTTP %s): %s" % (status, payload))
        tokens = {
            "access_token": payload["access_token"],
            "refresh_token": payload["refresh_token"],
            "gateway_url": payload.get("gateway_url") or gateway_url,
        }
        save_tokens(tokens)
        print("\n[ok] signed in; tokens saved to %s\n" % TOKENS_PATH)
        return tokens

    raise RuntimeError("\nsign-in expired before it was approved")


def refresh_tokens(tokens):
    """
    Exchange the refresh token for a new pair. Returns the new token dict, or
    None if the session is dead (401/403) and a new sign-in is required.
    """
    base = control_base(tokens.get("gateway_url", DEFAULT_GATEWAY_URL))
    status, payload = request_json(
        base + "/device/refresh",
        method="POST",
        headers=_client_headers(),
        payload={"refresh_token": tokens["refresh_token"]},
    )
    if status in (401, 403):
        return None  # refresh token retired; must sign in again
    if status != 200 or not isinstance(payload, dict) or not payload.get("access_token"):
        raise RuntimeError("refresh failed (HTTP %s): %s" % (status, payload))
    new_tokens = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", tokens["refresh_token"]),
        "gateway_url": payload.get("gateway_url") or tokens.get("gateway_url", DEFAULT_GATEWAY_URL),
    }
    save_tokens(new_tokens)
    return new_tokens


def fetch_models(tokens):
    """GET <gateway>/models with the bearer. Returns (status, parsed)."""
    return request_json(
        gateway_path(tokens["gateway_url"], "models"),
        method="GET",
        headers=_client_headers({"Authorization": "Bearer %s" % tokens["access_token"]}),
    )


def ensure_session(force_login=False):
    """
    Make sure TOKENS holds a working session. Order of operations:

      - no saved tokens (or --login)  -> device sign-in
      - saved tokens valid            -> reuse as-is (skip)
      - saved tokens expired (401)    -> refresh; if refresh dead -> sign-in

    Returns the model-list payload so the caller can print it.
    """
    global TOKENS

    if force_login:
        TOKENS = device_login()
    else:
        TOKENS = load_tokens()
        if TOKENS is None:
            print("[info] no saved sign-in found.")
            TOKENS = device_login()

    # Probe /models; this both validates the token and gives us the catalog.
    status, payload = fetch_models(TOKENS)

    if status == 401:
        print("[info] access token expired; refreshing...")
        with REFRESH_LOCK:
            refreshed = refresh_tokens(TOKENS)
        if refreshed is None:
            print("[info] saved session is no longer valid; signing in again.")
            TOKENS = device_login()
        else:
            TOKENS = refreshed
            print("[ok] token refreshed.")
        status, payload = fetch_models(TOKENS)

    if status != 200:
        raise RuntimeError("could not reach omnirush models endpoint (HTTP %s): %s" % (status, payload))

    return payload


def do_refresh_locked():
    """Refresh TOKENS, guarding against several requests refreshing at once."""
    global TOKENS
    with REFRESH_LOCK:
        current = dict(TOKENS)
        refreshed = refresh_tokens(current)
        if refreshed is None:
            # Session is dead. We cannot run an interactive sign-in from inside
            # a request, so surface a clear error to the client instead.
            raise RuntimeError("omnirush session expired; run the proxy again to sign in")
        TOKENS = refreshed
        return TOKENS


# --------------------------------------------------------------------------- #
# Calling the omnirush gateway (with one automatic refresh on 401)            #
# --------------------------------------------------------------------------- #

def gateway_call(path, method="POST", body_bytes=None, stream=False):
    """
    Send a request to <gateway>/<path> with the device bearer. On 401, refresh
    once and retry. Returns (status_code, response_object). The caller reads or
    streams response_object and must close it.
    """
    def send():
        headers = _client_headers({
            "Authorization": "Bearer %s" % TOKENS["access_token"],
            "Content-Type": "application/json",
            "x-omnirush-client-request-id": uuid.uuid4().hex,
        })
        url = gateway_path(TOKENS["gateway_url"], path)
        return open_upstream(url, method=method, headers=headers, data=body_bytes)

    status, resp = send()
    if status == 401:
        try:
            resp.read()
            resp.close()
        except Exception:
            pass
        do_refresh_locked()
        status, resp = send()
    return status, resp


# --------------------------------------------------------------------------- #
# Translation: OpenAI chat-completions  <->  omnirush Responses API           #
# --------------------------------------------------------------------------- #

def _text_from_content(content):
    """Flatten a chat message 'content' (string or list of parts) to text parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    parts = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("text", "input_text", "output_text"):
                parts.append({"type": "input_text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url = part.get("image_url", {})
                url = url.get("url") if isinstance(url, dict) else url
                if url:
                    parts.append({"type": "input_image", "image_url": url})
    return parts


def chat_to_responses(chat):
    """Convert an OpenAI chat-completions body into a Responses API body."""
    messages = chat.get("messages", [])
    instructions = []
    input_items = []

    for msg in messages:
        role = msg.get("role")
        if role in ("system", "developer"):
            text = msg.get("content")
            if isinstance(text, list):
                text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
            if text:
                instructions.append(text)
            continue

        if role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": msg.get("content") if isinstance(msg.get("content"), str)
                          else json.dumps(msg.get("content")),
            })
            continue

        if role == "assistant":
            content = msg.get("content")
            if content:
                input_items.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content if isinstance(content, str)
                                 else " ".join(p.get("text", "") for p in content if isinstance(p, dict))}],
                })
            for call in msg.get("tool_calls", []) or []:
                fn = call.get("function", {})
                input_items.append({
                    "type": "function_call",
                    "call_id": call.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or "{}",
                })
            continue

        # user (and anything else) -> input item
        input_items.append({"role": role or "user", "content": _text_from_content(msg.get("content"))})

    # model id may carry an effort suffix, e.g. "gpt-6-astra:high"
    model = chat.get("model", "")
    effort = chat.get("reasoning_effort")
    if ":" in model:
        model, suffix = model.rsplit(":", 1)
        if suffix in ("low", "high", "xhigh", "max", "minimal", "medium"):
            effort = suffix

    body = {"model": model, "input": input_items}
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    if effort:
        body["reasoning"] = {"effort": effort}
    if chat.get("temperature") is not None:
        body["temperature"] = chat["temperature"]
    if chat.get("top_p") is not None:
        body["top_p"] = chat["top_p"]
    if chat.get("max_tokens") is not None:
        body["max_output_tokens"] = chat["max_tokens"]
    if chat.get("max_completion_tokens") is not None:
        body["max_output_tokens"] = chat["max_completion_tokens"]
    if chat.get("parallel_tool_calls") is not None:
        body["parallel_tool_calls"] = chat["parallel_tool_calls"]

    tools = chat.get("tools")
    if tools:
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                fn = tool["function"]
                converted.append({
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
        if converted:
            body["tools"] = converted

    tc = chat.get("tool_choice")
    if tc is not None:
        if isinstance(tc, dict) and tc.get("type") == "function":
            name = tc.get("function", {}).get("name")
            body["tool_choice"] = {"type": "function", "name": name}
        else:
            body["tool_choice"] = tc  # "auto" / "none" / "required"

    rf = chat.get("response_format")
    if isinstance(rf, dict):
        if rf.get("type") == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        elif rf.get("type") == "json_schema":
            body["text"] = {"format": dict({"type": "json_schema"}, **rf.get("json_schema", {}))}

    return model, body


def responses_to_chat(resp, model):
    """Convert a non-streaming Responses answer into a chat-completions answer."""
    text_chunks = []
    tool_calls = []
    for item in resp.get("output", []) or []:
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content", []) or []:
                if part.get("type") == "output_text":
                    text_chunks.append(part.get("text", ""))
        elif itype == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or ("call_" + uuid.uuid4().hex[:24]),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "") or "{}",
                },
            })

    # convenience field some gateways include
    if not text_chunks and isinstance(resp.get("output_text"), str):
        text_chunks.append(resp["output_text"])

    message = {"role": "assistant", "content": "".join(text_chunks) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = "tool_calls" if tool_calls else "stop"

    usage_in = resp.get("usage", {}) or {}
    usage = {
        "prompt_tokens": usage_in.get("input_tokens", 0),
        "completion_tokens": usage_in.get("output_tokens", 0),
        "total_tokens": usage_in.get("total_tokens",
                        usage_in.get("input_tokens", 0) + usage_in.get("output_tokens", 0)),
    }

    return {
        "id": resp.get("id") or ("chatcmpl-" + uuid.uuid4().hex),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.get("model") or model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    }


def sse_events(resp):
    """Yield parsed JSON data objects from an SSE stream (urllib response)."""
    data_lines = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except Exception:
                    pass
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        # 'event:' and other fields are ignored; each event carries a 'type'


def stream_responses_to_chat(resp, model, write):
    """
    Translate a streaming Responses SSE stream into chat.completion.chunk SSE,
    writing each chunk via write(bytes). Returns when the stream ends.
    """
    chat_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    def chunk(delta, finish=None, usage=None):
        obj = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            obj["usage"] = usage
        write(("data: " + json.dumps(obj) + "\n\n").encode("utf-8"))

    chunk({"role": "assistant"})

    tool_index = {}      # item_id -> tool_call index
    tool_meta = {}       # item_id -> {"id":..., "name":...}
    next_index = 0
    saw_tool = False

    for event in sse_events(resp):
        etype = event.get("type", "")

        if etype == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                chunk({"content": delta})

        elif etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            delta = event.get("delta", "")
            if delta:
                # non-standard but harmless; clients that understand it show it.
                chunk({"reasoning_content": delta})

        elif etype == "response.output_item.added":
            item = event.get("item", {}) or {}
            if item.get("type") == "function_call":
                saw_tool = True
                iid = item.get("id") or ("fc_" + uuid.uuid4().hex[:12])
                tool_index[iid] = next_index
                tool_meta[iid] = {
                    "id": item.get("call_id") or ("call_" + uuid.uuid4().hex[:24]),
                    "name": item.get("name", ""),
                }
                next_index += 1

        elif etype == "response.function_call_arguments.delta":
            iid = event.get("item_id")
            idx = tool_index.get(iid, 0)
            meta = tool_meta.get(iid, {})
            tool_delta = {"index": idx, "function": {"arguments": event.get("delta", "")}}
            # emit id + name once (on the first arguments delta for this call)
            if meta.get("name") is not None:
                tool_delta["id"] = meta.get("id")
                tool_delta["type"] = "function"
                tool_delta["function"]["name"] = meta.get("name", "")
                tool_meta[iid] = {"name": None}  # mark header already sent
            chunk({"tool_calls": [tool_delta]})

        elif etype == "response.completed":
            usage_in = (event.get("response", {}) or {}).get("usage", {}) or {}
            usage = {
                "prompt_tokens": usage_in.get("input_tokens", 0),
                "completion_tokens": usage_in.get("output_tokens", 0),
                "total_tokens": usage_in.get("total_tokens",
                                usage_in.get("input_tokens", 0) + usage_in.get("output_tokens", 0)),
            }
            chunk({}, finish=("tool_calls" if saw_tool else "stop"), usage=usage)
            write(b"data: [DONE]\n\n")
            return

        elif etype in ("response.failed", "error"):
            detail = json.dumps(event.get("response", event))
            chunk({"content": "\n[omnirush error] " + detail}, finish="stop")
            write(b"data: [DONE]\n\n")
            return

    # stream ended without an explicit completed event
    chunk({}, finish=("tool_calls" if saw_tool else "stop"))
    write(b"data: [DONE]\n\n")


def normalize_models(payload):
    """Return an OpenAI-shaped {object:list, data:[{id,object:model,...}]}."""
    data = []
    raw = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and entry.get("id"):
                item = {"id": entry["id"], "object": "model", "owned_by": entry.get("owned_by", "omnirush")}
                if entry.get("created"):
                    item["created"] = entry["created"]
                data.append(item)
    return {"object": "list", "data": data}


# --------------------------------------------------------------------------- #
# HTTP server                                                                 #
# --------------------------------------------------------------------------- #

class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers ---------------------------------------------------------- #

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length else b""

    def _path(self):
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    # -- routes ----------------------------------------------------------- #

    def do_GET(self):
        path = self._path()
        if path in ("/v1/models", "/models"):
            try:
                status, payload = fetch_models(TOKENS)
                if status == 401:
                    do_refresh_locked()
                    status, payload = fetch_models(TOKENS)
                if status != 200:
                    return self._send_json(status, {"error": {"message": payload}})
                return self._send_json(200, normalize_models(payload))
            except Exception as exc:
                return self._send_json(500, {"error": {"message": str(exc)}})
        if path in ("/", "/health"):
            return self._send_json(200, {"ok": True, "service": "omnirush-hermes-proxy"})
        return self._send_json(404, {"error": {"message": "not found: %s" % path}})

    def do_POST(self):
        path = self._path()
        if path in ("/v1/chat/completions", "/chat/completions"):
            return self._handle_chat()
        if path in ("/v1/responses", "/responses"):
            return self._handle_responses_passthrough(path)
        return self._send_json(404, {"error": {"message": "not found: %s" % path}})

    def _handle_chat(self):
        try:
            chat = json.loads(self._read_body() or b"{}")
        except Exception as exc:
            return self._send_json(400, {"error": {"message": "invalid JSON: %s" % exc}})

        stream = bool(chat.get("stream"))
        model, body = chat_to_responses(chat)
        body["stream"] = stream
        body_bytes = json.dumps(body).encode("utf-8")

        try:
            status, resp = gateway_call("responses", method="POST", body_bytes=body_bytes)
        except Exception as exc:
            return self._send_json(502, {"error": {"message": str(exc)}})

        if status != 200:
            raw = resp.read()
            try:
                err = json.loads(raw.decode("utf-8"))
            except Exception:
                err = {"error": {"message": raw.decode("utf-8", "replace")}}
            return self._send_json(status, err)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            def write(data):
                self.wfile.write(data)
                self.wfile.flush()

            try:
                stream_responses_to_chat(resp, model, write)
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                try:
                    resp.close()
                except Exception:
                    pass
            return

        raw = resp.read()
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            return self._send_json(502, {"error": {"message": "bad upstream body: %s" % exc}})
        return self._send_json(200, responses_to_chat(parsed, model))

    def _handle_responses_passthrough(self, path):
        """For clients that speak the Responses API directly."""
        body_bytes = self._read_body()
        try:
            is_stream = json.loads(body_bytes or b"{}").get("stream", False)
        except Exception:
            is_stream = False
        try:
            status, resp = gateway_call("responses", method="POST", body_bytes=body_bytes)
        except Exception as exc:
            return self._send_json(502, {"error": {"message": str(exc)}})

        if is_stream and status == 200:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for raw in resp:
                    self.wfile.write(raw)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                resp.close()
            return

        raw = resp.read()
        resp.close()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


# --------------------------------------------------------------------------- #
# Entrypoint                                                                   #
# --------------------------------------------------------------------------- #

def print_models(payload):
    models = normalize_models(payload).get("data", [])
    if not models:
        print("[warn] no models returned by the account.")
        return
    print("\nModels available in your omnirush.ai account:")
    for m in models:
        print("   - %s" % m["id"])
    print("")


def main():
    parser = argparse.ArgumentParser(description="omnirush -> Hermes OpenAI-compatible proxy")
    parser.add_argument("--port", type=int, default=8787, help="port to listen on (default 8787)")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind (default 127.0.0.1)")
    parser.add_argument("--login", action="store_true", help="force a fresh sign-in")
    parser.add_argument("--models", action="store_true", help="print the model list and exit")
    args = parser.parse_args()

    try:
        catalog = ensure_session(force_login=args.login)
    except Exception as exc:
        print("[error] %s" % exc, file=sys.stderr)
        sys.exit(1)

    print_models(catalog)

    if args.models:
        return

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    base = "http://%s:%d/v1" % (args.host, args.port)
    print("==================================================================")
    print(" omnirush -> Hermes proxy is running")
    print("------------------------------------------------------------------")
    print(" Point Hermes at an OpenAI-compatible provider:")
    print("   Base URL : %s" % base)
    print("   API key  : anything (ignored, e.g. 'local')")
    print(" Endpoints  : GET %s/models" % base)
    print("              POST %s/chat/completions" % base)
    print("              POST %s/responses  (passthrough)" % base)
    print("==================================================================")
    print(" Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
