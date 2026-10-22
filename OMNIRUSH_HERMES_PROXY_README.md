# omnirush → Hermes proxy

A single-file Python proxy (`omnirush_hermes_proxy.py`, stdlib only) that signs
in to your omnirush.ai account once, keeps the token refreshed, and exposes an
**OpenAI-compatible** server so Hermes Agent can use your omnirush models.

## Run it

```bash
python omnirush_hermes_proxy.py
```

- First run: a browser opens to omnirush.ai; approve the device request.
  Tokens are saved to `~/.omnirush-hermes-proxy/tokens.json`.
- Later runs: the saved token is reused. If it has expired it is refreshed
  automatically; only if the refresh is dead are you asked to sign in again.
- The model ids in your account are printed on startup.

Flags:

| Flag | Meaning |
| --- | --- |
| `--port 8787` | Port to listen on (default `8787`) |
| `--host 127.0.0.1` | Bind address (default localhost only) |
| `--login` | Force a fresh sign-in |
| `--models` | Print the model list and exit |

## Point Hermes at it

Add a custom **OpenAI-compatible** provider in Hermes:

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8787/v1` |
| API key | anything (e.g. `local`) — it is ignored |
| Model | one of the ids printed on startup (e.g. `gpt-6-astra`) |

To pick a reasoning effort, append it to the model id: `gpt-6-astra:high`
(`low` / `high` / `xhigh` / `max`). Or send OpenAI's `reasoning_effort` field.

## Endpoints

- `GET  /v1/models` — model catalog, OpenAI shape
- `POST /v1/chat/completions` — translated to omnirush's Responses API
  (streaming and non-streaming, tool calls, images)
- `POST /v1/responses` — raw passthrough for clients that speak the Responses API

## Notes / limits

- omnirush only accepts the **Responses API** upstream; this proxy translates
  chat-completions to/from it. Exotic chat fields may not map perfectly.
- If you ever get **HTTP 426 (update required)**, set `CLIENT_VERSION` near the
  top of the script to the current app version (e.g. `"3.0.1"`).
- Free omnirush access is intended for their own app ("your own provider keys
  are not supported"). Using the token elsewhere may breach their terms, and
  usage is tied to your account. Use an account you're willing to risk.
