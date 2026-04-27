# toolcall-middleware

OpenAI-compatible proxy that adds **tool calling emulation** to any LLM backend + a **ChatGPT web proxy** that lets you use ChatGPT Plus/Pro models (GPT-4o, o3, o4-mini, GPT-5 family, etc.) via API — using your browser session cookies.

## Architecture

```
Your App / Agent
       │
       ▼
┌─────────────────────────────┐
│  proxy.ts (Bun, port 1435)  │  ← OpenAI-compatible API
│  - Routes models             │
│  - Tool call emulation       │
├──────────┬──────────────────┤
│ chatgpt/*│  other models    │
│          │                  │
▼          ▼                  │
┌────────────────┐  ┌────────────────┐
│ chatgpt-http-  │  │   Upstream     │
│ helper.py      │  │   (any OpenAI  │
│ (curl_cffi)    │  │   compatible)  │
│ port 1436      │  │                │
└───────┬────────┘  └────────────────┘
        │
        ▼
   chatgpt.com
  (backend-api)
```

## Features

- **Tool call emulation** — Adds `<tool_call>` XML-based tool calling to models that don't support it natively
- **ChatGPT web proxy** — Use ChatGPT Plus/Pro models via OpenAI-compatible API
- **Session-based cookies** — `curl_cffi` Session object auto-captures every `Set-Cookie` from Cloudflare/ChatGPT, so cookies stay fresh
- **Auto retry on 403** — If Cloudflare cookies expire mid-request, automatically refreshes and retries
- **Cookie persistence** — Saves cookies to `.cookies.json` so they survive restarts
- **Proof of Work solver** — Handles ChatGPT's PoW challenges automatically
- **30+ ChatGPT models** — GPT-4o, o3, o4-mini, GPT-5 family, deep-research, agent mode, etc.

## Requirements

- [Bun](https://bun.sh/) (v1.0+)
- Python 3.10+
- `curl_cffi` Python package
- A ChatGPT Plus or Pro account (for ChatGPT proxy)

## Installation

```bash
# Clone the repo
git clone https://github.com/zhaefremedia/toolcall-middleware.git
cd toolcall-middleware

# Install Bun dependencies
bun install

# Install Python dependency
pip3 install curl_cffi

# Copy env file
cp .env.example .env
```

## Configuration

Edit `.env`:

```env
# Upstream LLM API (optional — for non-ChatGPT models)
UPSTREAM_URL=http://127.0.0.1:8080/v1
UPSTREAM_KEY=your-api-key-here

# Proxy port
PORT=1435

# ChatGPT session cookies (see below how to get them)
CHATGPT_COOKIES=paste_your_cookies_here
```

## How to Get ChatGPT Cookies

This is the most important step. You need cookies from a logged-in ChatGPT browser session.

### Step 1: Open ChatGPT in your browser

Go to [https://chatgpt.com](https://chatgpt.com) and make sure you're **logged in**.

### Step 2: Open DevTools

Press `F12` or `Ctrl+Shift+I` (Windows/Linux) / `Cmd+Option+I` (Mac) to open Developer Tools.

### Step 3: Copy cookies

**Option A: Via DevTools Console (easiest)**

Go to the **Console** tab and paste:

```javascript
document.cookie
```

Copy the entire output.

**Option B: Via DevTools Application tab**

1. Go to **Application** → **Cookies** → `https://chatgpt.com`
2. You need these cookies (copy name=value pairs, separated by `;`):

| Cookie | Required | Description |
|--------|----------|-------------|
| `__Secure-next-auth.session-token` | ✅ **Critical** | Your auth session (very long JWT) |
| `_puid` | ✅ Recommended | User identifier |
| `oai-sc` | ✅ Recommended | OpenAI session cookie |
| `__cf_bm` | Optional | Cloudflare bot management (auto-refreshed) |
| `_cfuvid` | Optional | Cloudflare visitor ID (auto-refreshed) |
| `__cflb` | Optional | Cloudflare load balancer |
| `__Host-next-auth.csrf-token` | Optional | CSRF token |
| `__Secure-next-auth.callback-url` | Optional | Auth callback URL |

> **Note:** Cloudflare cookies (`__cf_bm`, `_cfuvid`, etc.) expire every ~30 minutes, but the proxy **auto-refreshes them** using the Session object. You only need to provide them once — the proxy handles rotation.

### Step 4: Paste into `.env`

Paste the full cookie string as one line:

```env
CHATGPT_COOKIES=_puid=user-xxx;__Secure-next-auth.session-token=eyJhbG...very_long_token;oai-sc=0gAAAA...;__cf_bm=abc123...;_cfuvid=xyz789...
```

> **Important:** The `__Secure-next-auth.session-token` is the most critical cookie. Without it, nothing works. It typically lasts ~30 days before you need to re-copy from browser.

### Step 5: Verify

```bash
# Start the helper and check if token works
python3 chatgpt-http-helper.py
# Should show: "Access token obtained ✓"
```

## Usage

### Start everything

```bash
# Option 1: Start both helper + proxy together
./start.sh

# Option 2: Start separately
python3 chatgpt-http-helper.py &   # Start helper first (port 1436)
bun run proxy.ts                    # Start proxy (port 1435)
```

### Test it

```bash
# Health check
curl http://127.0.0.1:1435/health

# List models
curl http://127.0.0.1:1435/v1/models

# Chat (ChatGPT)
curl -X POST http://127.0.0.1:1435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatgpt/gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Chat with streaming
curl -X POST http://127.0.0.1:1435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatgpt/gpt-5.5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Helper endpoints

The HTTP helper (port 1436) has debug endpoints:

```bash
# Health + cookie status
curl http://127.0.0.1:1436/health

# List current cookie names
curl http://127.0.0.1:1436/cookies

# Manually refresh Cloudflare cookies
curl http://127.0.0.1:1436/refresh
```

## Available ChatGPT Models

Use these as the `model` parameter:

| Model ID | ChatGPT Model |
|----------|---------------|
| `chatgpt/auto` | Auto (default) |
| `chatgpt/gpt-4o` | GPT-4o |
| `chatgpt/gpt-4o-mini` | GPT-4o Mini |
| `chatgpt/gpt-4.1` | GPT-4.1 |
| `chatgpt/o1` | o1 |
| `chatgpt/o1-mini` | o1-mini |
| `chatgpt/o1-pro` | o1-pro |
| `chatgpt/o3` | o3 |
| `chatgpt/o3-mini` | o3-mini |
| `chatgpt/o3-mini-high` | o3-mini (high effort) |
| `chatgpt/o4-mini` | o4-mini |
| `chatgpt/o4-mini-high` | o4-mini (high effort) |
| `chatgpt/gpt-5` | GPT-5 |
| `chatgpt/gpt-5-mini` | GPT-5 Mini |
| `chatgpt/gpt-5.1` | GPT-5.1 |
| `chatgpt/gpt-5.2` | GPT-5.2 |
| `chatgpt/gpt-5.2-thinking` | GPT-5.2 Thinking |
| `chatgpt/gpt-5.2-pro` | GPT-5.2 Pro |
| `chatgpt/gpt-5.3` | GPT-5.3 |
| `chatgpt/gpt-5.4-thinking` | GPT-5.4 Thinking |
| `chatgpt/gpt-5.4-pro` | GPT-5.4 Pro |
| `chatgpt/gpt-5.5` | GPT-5.5 |
| `chatgpt/gpt-5.5-thinking` | GPT-5.5 Thinking |
| `chatgpt/gpt-5.5-pro` | GPT-5.5 Pro |
| `chatgpt/deep-research` | Deep Research |
| `chatgpt/agent` | Agent Mode |

## Use with OpenCode / Hermes / Other Agents

Point your agent's OpenAI-compatible provider to `http://127.0.0.1:1435/v1`:

**OpenCode** (`opencode.json`):
```json
{
  "provider": {
    "chatgpt-proxy": {
      "name": "ChatGPT Proxy",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:1435/v1"
      },
      "models": {
        "chatgpt/gpt-5.5": {
          "name": "GPT-5.5 (ChatGPT)",
          "limit": { "context": 1050000, "output": 128000 },
          "modalities": { "input": ["text"], "output": ["text"] }
        }
      }
    }
  }
}
```

**Hermes** (`config.yaml`):
```yaml
custom_providers:
  - name: chatgpt-proxy
    base_url: http://localhost:1435/v1
    api_key_env: ''
    models:
      chatgpt/gpt-5.5:
        context_length: 1050000
```

**Any OpenAI SDK**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1435/v1",
    api_key="not-needed"  # no API key required
)

response = client.chat.completions.create(
    model="chatgpt/gpt-5.5",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## How Cookie Auto-Refresh Works

The proxy uses `curl_cffi.Session` (not raw requests) which acts like a real browser tab:

1. **Every response** from chatgpt.com that contains `Set-Cookie` headers is automatically captured
2. **Cloudflare cookies** (`__cf_bm`, `_cfuvid`, etc.) rotate frequently — the Session handles this transparently
3. **On 403 errors**, the proxy automatically visits the homepage to get fresh Cloudflare cookies, then retries
4. **Cookies are persisted** to `.cookies.json` so they survive restarts
5. **On startup**, persisted cookies are merged with `.env` cookies (persisted takes priority since they're fresher)

You only need to manually update `CHATGPT_COOKIES` in `.env` when the `__Secure-next-auth.session-token` expires (~30 days).

## Troubleshooting

### "No session token found"
Your `CHATGPT_COOKIES` in `.env` is missing or doesn't contain `__Secure-next-auth.session-token`. Re-copy from browser.

### "Session fetch failed: 401"
Your session token has expired. Log into chatgpt.com in your browser and re-copy the cookies.

### "413 message_length_exceeds_limit"
ChatGPT web has a per-message size limit. This happens when agents send very large prompts (system prompt + tool definitions). Consider using ChatGPT models for simpler tasks, or use API-based models for agentic workloads.

### "Address already in use"
Another process is using port 1435 or 1436. Kill it:
```bash
fuser -k 1435/tcp 1436/tcp
```

### Cookie-related 403 errors
The proxy auto-handles most cookie issues. If you see persistent 403s:
```bash
# Manual refresh
curl http://127.0.0.1:1436/refresh

# Check cookie status
curl http://127.0.0.1:1436/health
```

## License

MIT
