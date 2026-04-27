/**
 * ChatGPT Web → OpenAI-compatible API provider
 * 
 * Uses chatgpt-http-helper.py (curl_cffi with Safari TLS fingerprint)
 * to bypass Cloudflare protection on chatgpt.com.
 * 
 * Architecture:
 *   Client → proxy.ts → chatgpt-provider.ts → chatgpt-http-helper.py (curl_cffi) → chatgpt.com
 * 
 * Pure fetch-based — no Playwright/browser needed.
 */

import { randomUUID } from "crypto";
import { spawn, type ChildProcess } from "child_process";
import { resolve } from "path";

// ── Config ──
const HELPER_PORT = parseInt(process.env.CHATGPT_BRIDGE_PORT || "1436");
const HELPER_URL = `http://127.0.0.1:${HELPER_PORT}`;

// ── Model mapping ──
const MODEL_MAP: Record<string, string> = {
  // Legacy / convenience aliases
  "chatgpt/auto":              "auto",
  "chatgpt/gpt-4o":            "gpt-4o",
  "chatgpt/gpt-4o-mini":       "gpt-4o-mini",
  "chatgpt/gpt-4":             "gpt-4",
  "chatgpt/gpt-4.1":           "gpt-4.1",
  "chatgpt/o1":                "o1",
  "chatgpt/o1-mini":           "o1-mini",
  "chatgpt/o1-pro":            "o1-pro",
  "chatgpt/o3":                "o3",
  "chatgpt/o3-mini":           "o3-mini",
  "chatgpt/o3-mini-high":      "o3-mini-high",
  "chatgpt/o4-mini":           "o4-mini",
  "chatgpt/o4-mini-high":      "o4-mini-high",
  "gpt-4o":                    "gpt-4o",
  "gpt-4o-mini":               "gpt-4o-mini",
  // GPT-5 family (business account slugs)
  "chatgpt/gpt-5":             "gpt-5",
  "chatgpt/gpt-5-mini":        "gpt-5-mini",
  "chatgpt/gpt-5.1":           "gpt-5-1",
  "chatgpt/gpt-5.2":           "gpt-5-2",
  "chatgpt/gpt-5.2-instant":   "gpt-5-2-instant",
  "chatgpt/gpt-5.2-thinking":  "gpt-5-2-thinking",
  "chatgpt/gpt-5.2-pro":       "gpt-5-2-pro",
  "chatgpt/gpt-5.3":           "gpt-5-3",
  "chatgpt/gpt-5.3-instant":   "gpt-5-3-instant",
  "chatgpt/gpt-5.3-mini":      "gpt-5-3-mini",
  "chatgpt/gpt-5.4-thinking":  "gpt-5-4-thinking",
  "chatgpt/gpt-5.4-pro":       "gpt-5-4-pro",
  "chatgpt/gpt-5.4-t-mini":    "gpt-5-4-t-mini",
  "chatgpt/gpt-5.5":           "gpt-5-5-thinking",
  "chatgpt/gpt-5.5-thinking":  "gpt-5-5-thinking",
  "chatgpt/gpt-5.5-pro":       "gpt-5-5-pro",
  "chatgpt/deep-research":     "research",
  "chatgpt/agent":             "agent-mode",
};

// ── Helper process management ──
let helperProcess: ChildProcess | null = null;
let helperReady = false;
let helperStarting: Promise<void> | null = null;

// ── Conversation Session Manager ──
const MAX_SESSION_AGE_MS = 30 * 60 * 1000; // 30 min timeout
const MAX_TURNS_BEFORE_REFRESH = 20;
const MAX_TOOL_RESULT_CHARS = 8000;

interface ConversationSession {
  conversationId: string;
  parentMessageId: string;
  setupHash: string;
  turnCount: number;
  createdAt: number;
  lastUsedAt: number;
}

const sessions = new Map<string, ConversationSession>();

function computeSetupHash(systemPrompt: string, tools: any[]): string {
  const toolNames = (tools || []).map((t: any) => (t.function || t).name).sort().join(",");
  const raw = systemPrompt.slice(0, 500) + "|" + toolNames;
  const encoder = new TextEncoder();
  const data = encoder.encode(raw);
  let hash = 0;
  for (const byte of data) {
    hash = ((hash << 5) - hash + byte) | 0;
  }
  return `sess_${Math.abs(hash).toString(36)}`;
}

function getSession(hash: string): ConversationSession | null {
  const s = sessions.get(hash);
  if (!s) return null;
  if (Date.now() - s.lastUsedAt > MAX_SESSION_AGE_MS) {
    void cleanupSession(hash);
    return null;
  }
  return s;
}

async function cleanupSession(hash: string): Promise<void> {
  const s = sessions.get(hash);
  if (s?.conversationId) {
    try {
      await fetch(`${HELPER_URL}/cleanup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: s.conversationId }),
        signal: AbortSignal.timeout(5000),
      });
      console.log(`[chatgpt] Session ${hash} cleaned up (conv: ${s.conversationId.slice(0, 8)}...)`);
    } catch {}
  }
  sessions.delete(hash);
}

// Export session count for health endpoint
export function getSessionCount(): number {
  return sessions.size;
}

export function buildCompressedToolPrompt(tools: any[]): string {
  if (!tools?.length) return "";

  const lines = tools.map((t: any) => {
    const fn = t.function || t;
    const params = fn.parameters;
    let paramStr = "";
    if (params?.properties) {
      const required = new Set(params.required || []);
      paramStr = Object.keys(params.properties)
        .map(k => required.has(k) ? k : `${k}?`)
        .join(", ");
    }
    const desc = (fn.description || "").split("\n")[0].slice(0, 80);
    return `- ${fn.name}(${paramStr}) — ${desc}`;
  }).join("\n");

  return `# Available Tools

You MUST use tools to accomplish tasks. Do NOT answer from memory when a tool can do it.

To call a tool, output EXACTLY this XML format (no markdown, no code blocks):

<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>

Rules:
- ALWAYS use tools when the task requires executing commands, reading files, searching, etc.
- Multiple tool calls = multiple separate <tool_call> blocks
- When calling tools, output ONLY <tool_call> blocks, nothing else
- Arguments must be valid JSON matching the tool's parameters

## Tool List

${lines}`;
}

async function sendToHelper(opts: {
  model: string;
  messages: any[];
  conversationId?: string;
  parentMessageId?: string;
  requestId: string;
}): Promise<any> {
  await ensureHelper();

  const helperRes = await fetch(`${HELPER_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: opts.model,
      messages: opts.messages,
      conversation_id: opts.conversationId,
      parent_message_id: opts.parentMessageId,
      request_id: opts.requestId,
    }),
    signal: AbortSignal.timeout(180000),
  });

  return helperRes.json();
}

function truncateContent(content: string, maxChars: number): string {
  if (content.length <= maxChars) return content;
  return content.slice(0, maxChars) + `\n\n[... truncated, showing first ${maxChars} chars of ${content.length} total]`;
}

function prepareMessages(messages: any[]): any[] {
  return messages.map((m: any) => {
    if (m.role === "tool" && typeof m.content === "string" && m.content.length > MAX_TOOL_RESULT_CHARS) {
      return { ...m, content: truncateContent(m.content, MAX_TOOL_RESULT_CHARS) };
    }
    return m;
  });
}

function buildResponse(requestId: string, created: number, model: string, text: string, wantStream: boolean): Response {
  const response = {
    id: requestId,
    object: "chat.completion",
    created,
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: text },
      finish_reason: "stop",
    }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  };

  if (wantStream) return sseFromResponse(response);
  return Response.json(response);
}

async function ensureHelper(): Promise<void> {
  if (helperReady) {
    try {
      const res = await fetch(`${HELPER_URL}/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return;
    } catch {
      helperReady = false;
    }
  }
  
  if (helperStarting) {
    await helperStarting;
    return;
  }
  
  helperStarting = startHelper();
  await helperStarting;
  helperStarting = null;
}

async function startHelper(): Promise<void> {
  if (helperProcess) {
    try { helperProcess.kill(); } catch {}
    helperProcess = null;
  }
  
  console.log("[chatgpt] Starting HTTP helper (curl_cffi)...");
  
  const scriptPath = resolve(import.meta.dir, "chatgpt-http-helper.py");
  
  helperProcess = spawn("python3", [scriptPath], {
    env: { ...process.env, CHATGPT_BRIDGE_PORT: String(HELPER_PORT) },
    stdio: ["ignore", "pipe", "pipe"],
  });
  
  helperProcess.stdout?.on("data", (data: Buffer) => {
    for (const line of data.toString().split("\n").filter(Boolean)) {
      console.log(line);
    }
  });
  
  helperProcess.stderr?.on("data", (data: Buffer) => {
    for (const line of data.toString().split("\n").filter(Boolean)) {
      console.error(line);
    }
  });
  
  helperProcess.on("exit", (code) => {
    console.log(`[chatgpt] HTTP helper exited with code ${code}`);
    helperReady = false;
    helperProcess = null;
  });
  
  // Wait for helper to be ready (much faster than Playwright — ~5s)
  const maxWait = 30000;
  const start = Date.now();
  
  while (Date.now() - start < maxWait) {
    await new Promise(r => setTimeout(r, 500));
    try {
      const res = await fetch(`${HELPER_URL}/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const data = await res.json() as any;
        if (data.has_token) {
          console.log("[chatgpt] HTTP helper ready ✓ (curl_cffi + Safari TLS)");
          helperReady = true;
          return;
        }
        // Helper is up but token not yet fetched — keep waiting
      }
    } catch {
      // Not ready yet
    }
  }
  
  // Even if token isn't ready, the helper might still work (lazy token fetch)
  try {
    const res = await fetch(`${HELPER_URL}/health`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      console.log("[chatgpt] HTTP helper ready ✓ (token will be fetched on first request)");
      helperReady = true;
      return;
    }
  } catch {}
  
  throw new Error("HTTP helper failed to start within 30 seconds");
}

// ── Main: handle chat completion request ──
export async function handleChatGPTRequest(body: any): Promise<Response> {
  const model = body.model || "chatgpt/auto";
  const slug = MODEL_MAP[model] || MODEL_MAP[`chatgpt/${model}`] || model.replace("chatgpt/", "");
  const wantStream = body.stream === true;
  const requestId = `chatcmpl-${randomUUID().replace(/-/g, "").slice(0, 24)}`;
  const created = Math.floor(Date.now() / 1000);

  try {
    await ensureHelper();

    console.log(`[chatgpt] → ${slug} (${body.messages?.length || 0} messages)`);
    
    const helperRes = await fetch(`${HELPER_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: slug,
        messages: body.messages || [],
        request_id: requestId,
      }),
      signal: AbortSignal.timeout(180000),
    });

    const result = await helperRes.json() as any;

    if (result.error) {
      console.error(`[chatgpt] ✗ Error: ${result.status} - ${(result.body || "").slice(0, 200)}`);
      return Response.json(
        { error: { message: `ChatGPT error: ${result.body || "Unknown error"}`, type: "chatgpt_error", code: result.status || 500 } },
        { status: 502 },
      );
    }

    const text = result.text || "";
    console.log(`[chatgpt] ✓ ${slug} → ${text.length} chars`);

    const response = {
      id: requestId,
      object: "chat.completion",
      created,
      model,
      choices: [{
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: "stop",
      }],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };

    if (wantStream) {
      return sseFromResponse(response);
    }

    return Response.json(response);

  } catch (err: any) {
    console.error("[chatgpt] error:", err.message);
    return Response.json(
      { error: { message: err.message, type: "chatgpt_provider_error" } },
      { status: 500 },
    );
  }
}

// ── Convert non-streaming response to SSE ──
function sseFromResponse(response: any): Response {
  const { id, model, created } = response;
  const content = response.choices[0].message.content || "";
  const chunks: string[] = [];

  chunks.push(JSON.stringify({
    id, object: "chat.completion.chunk", created, model,
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  }));
  
  for (let i = 0; i < content.length; i += 50) {
    chunks.push(JSON.stringify({
      id, object: "chat.completion.chunk", created, model,
      choices: [{ index: 0, delta: { content: content.slice(i, i + 50) }, finish_reason: null }],
    }));
  }
  
  chunks.push(JSON.stringify({
    id, object: "chat.completion.chunk", created, model,
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: response.usage,
  }));

  return new Response(
    chunks.map(c => `data: ${c}\n\n`).join("") + "data: [DONE]\n\n",
    { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } },
  );
}

// ── Check if a model should be routed to ChatGPT ──
export function isChatGPTModel(model: string): boolean {
  if (model.startsWith("chatgpt/")) return true;
  if (MODEL_MAP[model]) return true;
  return false;
}

// ── Export model list for /v1/models ──
export function getChatGPTModelObjects(): any[] {
  const models = [
    { id: "chatgpt/auto", context: 128000 },
    { id: "chatgpt/gpt-4o", context: 128000 },
    { id: "chatgpt/gpt-4o-mini", context: 128000 },
    { id: "chatgpt/gpt-4.1", context: 1047576 },
    { id: "chatgpt/o3", context: 200000 },
    { id: "chatgpt/o3-mini", context: 200000 },
    { id: "chatgpt/o3-mini-high", context: 200000 },
    { id: "chatgpt/o4-mini", context: 200000 },
    { id: "chatgpt/o4-mini-high", context: 200000 },
    { id: "chatgpt/o1", context: 200000 },
    { id: "chatgpt/o1-mini", context: 128000 },
    { id: "chatgpt/o1-pro", context: 200000 },
    // GPT-5 family
    { id: "chatgpt/gpt-5", context: 128000 },
    { id: "chatgpt/gpt-5-mini", context: 128000 },
    { id: "chatgpt/gpt-5.1", context: 128000 },
    { id: "chatgpt/gpt-5.2", context: 128000 },
    { id: "chatgpt/gpt-5.2-instant", context: 128000 },
    { id: "chatgpt/gpt-5.2-thinking", context: 262144 },
    { id: "chatgpt/gpt-5.2-pro", context: 262144 },
    { id: "chatgpt/gpt-5.3", context: 128000 },
    { id: "chatgpt/gpt-5.3-instant", context: 128000 },
    { id: "chatgpt/gpt-5.3-mini", context: 128000 },
    { id: "chatgpt/gpt-5.4-thinking", context: 262144 },
    { id: "chatgpt/gpt-5.4-pro", context: 410000 },
    { id: "chatgpt/gpt-5.4-t-mini", context: 262144 },
    { id: "chatgpt/gpt-5.5", context: 1050000 },
    { id: "chatgpt/gpt-5.5-thinking", context: 1050000 },
    { id: "chatgpt/gpt-5.5-pro", context: 1050000 },
    { id: "chatgpt/deep-research", context: 128000 },
    { id: "chatgpt/agent", context: 262144 },
  ];

  return models.map(m => ({
    id: m.id,
    object: "model",
    created: 1700000000,
    owned_by: "chatgpt-proxy",
  }));
}

export async function handleChatGPTSessionRequest(body: any): Promise<Response> {
  const model = body.model || "chatgpt/auto";
  const slug = MODEL_MAP[model] || MODEL_MAP[`chatgpt/${model}`] || model.replace("chatgpt/", "");
  const wantStream = body.stream === true;
  const requestId = `chatcmpl-${randomUUID().replace(/-/g, "").slice(0, 24)}`;
  const created = Math.floor(Date.now() / 1000);

  const tools = body._tools || [];
  const systemPrompt = body._systemPrompt || "";
  const messages = prepareMessages(body.messages || []);

  try {
    const setupHash = computeSetupHash(systemPrompt, tools);
    let session = getSession(setupHash);
    const compressedTools = buildCompressedToolPrompt(tools);
    const isNewSession = !session;

    if (!session) {
      session = {
        conversationId: "",
        parentMessageId: "",
        setupHash,
        turnCount: 0,
        createdAt: Date.now(),
        lastUsedAt: Date.now(),
      };
      sessions.set(setupHash, session);
      console.log(`[chatgpt] New session ${setupHash}`);
    }

    // ── Build message: system context + tools + actual content, every time ──
    const actualContent = messages.map((m: any) => {
      const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
      if (m.role === "assistant") return `[Assistant]: ${content}`;
      if (m.role === "tool") return `[Tool Result (${m.name || "unknown"})]: ${content}`;
      if (m.role === "system") return `[System]: ${content}`;
      return content;
    }).join("\n\n");

    const fullMessage = `You are an AI coding assistant with access to real tools that you can execute.
You MUST use tools to accomplish tasks. NEVER answer from memory or tell the user to do it themselves.

To call a tool, output EXACTLY:

<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>

Multiple tools = multiple <tool_call> blocks. When calling tools, output ONLY <tool_call> blocks.

${compressedTools}

---

${actualContent}`;

    const result = await sendToHelper({
      model: slug,
      messages: [
        ...(systemPrompt ? [{ role: "system", content: systemPrompt }] : []),
        { role: "user", content: fullMessage },
      ],
      conversationId: session.conversationId || undefined,
      parentMessageId: session.parentMessageId || undefined,
      requestId,
    });

    if (result.error) {
      console.error(`[chatgpt] ✗ Session ${setupHash} error: ${result.status}`);
      await cleanupSession(setupHash);

      if (result.status === 413) {
        console.log(`[chatgpt] 413 — retrying with truncated messages...`);
        const truncatedMessages = messages.filter((m: any) => m.role !== "system").slice(-5);
        const truncatedContent = truncatedMessages.map((m: any) => {
          const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
          return m.role === "user" ? content : `[${m.role}]: ${content}`;
        }).join("\n\n");

        const retryResult = await sendToHelper({
          model: slug,
          messages: [{ role: "user", content: `${compressedTools}\n\n---\n\n${truncatedContent}` }],
          requestId,
        });
        if (retryResult.error) {
          return Response.json(
            { error: { message: `ChatGPT error: ${retryResult.body}`, type: "chatgpt_error" } },
            { status: 502 },
          );
        }
        return buildResponse(requestId, created, model, retryResult.text || "", wantStream);
      }

      return Response.json(
        { error: { message: `ChatGPT error: ${result.body}`, type: "chatgpt_error" } },
        { status: 502 },
      );
    }

    // Update session state for conversation reuse
    if (result.message_id) session.parentMessageId = result.message_id;
    if (result.conversation_id) session.conversationId = result.conversation_id;
    session.turnCount++;
    session.lastUsedAt = Date.now();

    const text = result.text || "";
    console.log(`[chatgpt] ✓ ${slug} → ${text.length} chars (session ${setupHash}, turn ${session.turnCount})`);

    return buildResponse(requestId, created, model, text, wantStream);

  } catch (err: any) {
    console.error("[chatgpt] session error:", err.message);
    return Response.json(
      { error: { message: err.message, type: "chatgpt_provider_error" } },
      { status: 500 },
    );
  }
}

// ── Cleanup on exit ──
process.on("exit", () => {
  if (helperProcess) try { helperProcess.kill(); } catch {}
});
process.on("SIGINT", () => {
  if (helperProcess) try { helperProcess.kill(); } catch {}
  process.exit(0);
});
process.on("SIGTERM", () => {
  if (helperProcess) try { helperProcess.kill(); } catch {}
  process.exit(0);
});
