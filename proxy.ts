#!/usr/bin/env bun
import { handleChatGPTRequest, handleChatGPTSessionRequest, isChatGPTModel, getChatGPTModelObjects, getSessionCount } from "./chatgpt-provider";

const UPSTREAM = process.env.UPSTREAM_URL || "http://127.0.0.1:8080/v1";
const UPSTREAM_KEY = process.env.UPSTREAM_KEY || "";
const PORT = parseInt(process.env.PORT || "1435");

function upstreamHeaders(): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (UPSTREAM_KEY) h["Authorization"] = `Bearer ${UPSTREAM_KEY}`;
  return h;
}

function buildToolSystemPrompt(tools: any[]): string {
  if (!tools?.length) return "";

  const descs = tools.map((t: any) => {
    const fn = t.function || t;
    const params = fn.parameters ? JSON.stringify(fn.parameters, null, 2) : "{}";
    return `### ${fn.name}\n${fn.description || ""}\nParameters:\n\`\`\`json\n${params}\n\`\`\``;
  }).join("\n\n");

  return `\n\n# Available Tools

When you need to use a tool, output EXACTLY this format:

<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>

Rules:
- Use EXACT <tool_call> XML tags
- Valid JSON with "name" and "arguments" fields
- Multiple calls = multiple <tool_call> blocks
- No markdown code blocks around tool calls
- When calling tools, output ONLY tool_call blocks

## Tools

${descs}
`;
}

interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

function parseToolCalls(text: string): { toolCalls: ToolCall[]; textContent: string } | null {
  const re = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
  const calls: ToolCall[] = [];
  let m, i = 0;

  while ((m = re.exec(text)) !== null) {
    try {
      const raw = m[1].trim().replace(/,\s*}/g, "}").replace(/,\s*]/g, "]");
      const p = JSON.parse(raw);
      calls.push({
        id: `call_${Date.now()}_${i++}`,
        type: "function",
        function: {
          name: p.name,
          arguments: typeof p.arguments === "string" ? p.arguments : JSON.stringify(p.arguments),
        },
      });
    } catch {
      console.error("[toolcall-proxy] bad tool_call JSON:", m[1]);
    }
  }

  if (!calls.length) return null;
  return { toolCalls: calls, textContent: text.replace(re, "").trim() };
}

function sseFromToolCalls(response: any): Response {
  const msg = response.choices[0].message;
  const { id, model, created } = response;
  const chunks: string[] = [];

  chunks.push(JSON.stringify({
    id, object: "chat.completion.chunk", created, model,
    choices: [{ index: 0, delta: { role: "assistant", content: null }, finish_reason: null }],
  }));

  for (let i = 0; i < msg.tool_calls.length; i++) {
    const tc = msg.tool_calls[i];
    chunks.push(JSON.stringify({
      id, object: "chat.completion.chunk", created, model,
      choices: [{ index: 0, delta: { tool_calls: [{ index: i, id: tc.id, type: "function", function: { name: tc.function.name, arguments: "" } }] }, finish_reason: null }],
    }));
    const args = tc.function.arguments;
    for (let j = 0; j < args.length; j += 100) {
      chunks.push(JSON.stringify({
        id, object: "chat.completion.chunk", created, model,
        choices: [{ index: 0, delta: { tool_calls: [{ index: i, function: { arguments: args.slice(j, j + 100) } }] }, finish_reason: null }],
      }));
    }
  }

  chunks.push(JSON.stringify({
    id, object: "chat.completion.chunk", created, model,
    choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
    usage: response.usage,
  }));

  return new Response(
    chunks.map(c => `data: ${c}\n\n`).join("") + "data: [DONE]\n\n",
    { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" } },
  );
}

function sseFromText(response: any): Response {
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

// ── Smart Compression for ChatGPT web (which has message length limits) ──
const CHATGPT_MAX_CHARS = 50000; // safe limit for ChatGPT web per-conversation

function estimateChars(messages: any[]): number {
  return messages.reduce((sum: number, m: any) => {
    const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content || "");
    return sum + content.length;
  }, 0);
}

function compressToolsForChatGPT(tools: any[]): any[] {
  // Full JSON schema → compact: just name, description, param names+types
  return tools.map((t: any) => {
    const fn = t.function || t;
    const props = fn.parameters?.properties || {};
    const required = fn.parameters?.required || [];
    
    // Build compact param string: "city: string (required), unit: string"
    const paramParts = Object.entries(props).map(([name, schema]: [string, any]) => {
      const type = schema.type || "any";
      const isReq = required.includes(name);
      const enumVals = schema.enum ? ` [${schema.enum.join("|")}]` : "";
      return `${name}: ${type}${enumVals}${isReq ? " (required)" : ""}`;
    });
    
    return {
      type: "function",
      function: {
        name: fn.name,
        description: (fn.description || "").slice(0, 120),
        parameters: {
          type: "object",
          properties: Object.fromEntries(
            Object.entries(props).map(([k, v]: [string, any]) => [k, { type: v.type || "string" }])
          ),
          required,
        },
      },
    };
  });
}

function compressMessagesForChatGPT(messages: any[], maxChars: number): any[] {
  // Strategy: keep system (truncated) + first user + last N messages
  const result: any[] = [];
  let totalChars = 0;
  
  // 1. System prompt — cap at 15K chars
  const sysMsg = messages.find((m: any) => m.role === "system");
  if (sysMsg) {
    const content = typeof sysMsg.content === "string" ? sysMsg.content : JSON.stringify(sysMsg.content);
    const maxSys = 15000;
    if (content.length > maxSys) {
      result.push({ ...sysMsg, content: content.slice(0, maxSys) + "\n\n[... system prompt truncated for length ...]" });
      totalChars += maxSys;
    } else {
      result.push(sysMsg);
      totalChars += content.length;
    }
  }
  
  // 2. Non-system messages — work backwards from most recent, keep what fits
  const nonSystem = messages.filter((m: any) => m.role !== "system");
  const kept: any[] = [];
  
  for (let i = nonSystem.length - 1; i >= 0; i--) {
    const msg = nonSystem[i];
    let content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content || "");
    
    // Truncate individual tool results that are huge
    if (msg.role === "tool" && content.length > 3000) {
      content = content.slice(0, 3000) + "\n[... output truncated ...]";
    }
    
    // Truncate assistant messages with huge content
    if (msg.role === "assistant" && typeof msg.content === "string" && content.length > 5000) {
      content = content.slice(0, 5000) + "\n[... truncated ...]";
    }
    
    const msgChars = content.length;
    if (totalChars + msgChars > maxChars && kept.length >= 2) {
      // We have at least the last 2 messages, stop adding older ones
      break;
    }
    
    kept.unshift({ ...msg, content });
    totalChars += msgChars;
  }
  
  result.push(...kept);
  return result;
}

function compressBodyForChatGPT(body: any): any {
  const originalMessages = body.messages || [];
  const originalTools = body.tools || [];
  
  // Estimate original size
  const toolsJson = JSON.stringify(originalTools);
  const msgsChars = estimateChars(originalMessages);
  const totalOriginal = msgsChars + toolsJson.length;
  
  if (totalOriginal <= CHATGPT_MAX_CHARS) {
    // Already small enough, no compression needed
    return body;
  }
  
  console.log(`[compress] Original: ${(totalOriginal / 1000).toFixed(0)}K chars (${originalMessages.length} msgs, ${originalTools.length} tools)`);
  
  // Compress tools first (biggest win usually)
  const compressedTools = compressToolsForChatGPT(originalTools);
  const compressedToolsJson = JSON.stringify(compressedTools);
  const toolsSaved = toolsJson.length - compressedToolsJson.length;
  
  // Budget remaining for messages
  const msgBudget = CHATGPT_MAX_CHARS - compressedToolsJson.length;
  const compressedMessages = compressMessagesForChatGPT(originalMessages, Math.max(msgBudget, 10000));
  
  const newTotal = estimateChars(compressedMessages) + compressedToolsJson.length;
  console.log(`[compress] Compressed: ${(newTotal / 1000).toFixed(0)}K chars (${compressedMessages.length} msgs, ${compressedTools.length} tools) — saved ${((totalOriginal - newTotal) / 1000).toFixed(0)}K`);
  
  return {
    ...body,
    messages: compressedMessages,
    tools: compressedTools,
  };
}

// ── ChatGPT tool call emulation (same as upstream, but via ChatGPT provider) ──
async function handleChatGPTWithTools(body: any): Promise<Response> {
  const hasTools = body.tools?.length > 0;
  const wantStream = body.stream === true;

  if (!hasTools) {
    // Even without tools, compress messages if too large
    const msgsChars = estimateChars(body.messages || []);
    if (msgsChars > CHATGPT_MAX_CHARS) {
      console.log(`[compress] No tools but messages too large (${(msgsChars / 1000).toFixed(0)}K), compressing...`);
      const compressed = compressMessagesForChatGPT(body.messages || [], CHATGPT_MAX_CHARS);
      return handleChatGPTRequest({ ...body, messages: compressed });
    }
    return handleChatGPTRequest(body);
  }

  // ── Compress before sending to ChatGPT ──
  body = compressBodyForChatGPT(body);

  console.log(`[~] ${body.model} → emulating tool calls via ChatGPT (${body.tools.length} tools)...`);

  const toolPrompt = buildToolSystemPrompt(body.tools);
  const messages = [...(body.messages || [])];

  // ChatGPT web doesn't support system role well — merge system into first user message
  const sysIdx = messages.findIndex((m: any) => m.role === "system");
  let systemContent = "";
  if (sysIdx >= 0) {
    systemContent = messages[sysIdx].content || "";
    messages.splice(sysIdx, 1); // remove system message
  }

  // Find first user message and prepend system + tools
  const userIdx = messages.findIndex((m: any) => m.role === "user");
  if (userIdx >= 0) {
    const prefix = (systemContent ? systemContent + "\n\n" : "") + toolPrompt + "\n\n---\n\n";
    messages[userIdx] = { ...messages[userIdx], content: prefix + messages[userIdx].content };
  } else {
    // No user message — create one with system + tools
    messages.push({ role: "user", content: (systemContent ? systemContent + "\n\n" : "") + toolPrompt });
  }

  if (body.tool_choice === "required") {
    messages.push({ role: "user", content: "[SYSTEM: You MUST use at least one tool. Output <tool_call> blocks only.]" });
  } else if (body.tool_choice?.function) {
    messages.push({ role: "user", content: `[SYSTEM: You MUST call the "${body.tool_choice.function.name}" tool now.]` });
  }

  const emulatedBody = { ...body, messages, stream: false };
  delete emulatedBody.tools;
  delete emulatedBody.tool_choice;

  const emRes = await handleChatGPTRequest(emulatedBody);
  if (!emRes.ok) return emRes;

  const emResult = await emRes.json() as any;
  const emContent = emResult.choices?.[0]?.message?.content || "";
  const parsed = parseToolCalls(emContent);

  if (parsed?.toolCalls.length) {
    console.log(`[✓] ${body.model} → emulated ${parsed.toolCalls.length} tool call(s) via ChatGPT session`);
    const response = {
      ...emResult,
      choices: [{
        ...emResult.choices[0],
        finish_reason: "tool_calls",
        message: { role: "assistant", content: parsed.textContent || null, tool_calls: parsed.toolCalls },
      }],
    };
    if (wantStream) return sseFromToolCalls(response);
    return Response.json(response);
  }

  console.log(`[!] ${body.model} → no tool calls detected, returning text`);
  if (wantStream) return sseFromText(emResult);
  return Response.json(emResult);
}

// ── Upstream handler (original logic) ──
async function handleUpstreamChat(req: Request): Promise<Response> {
  const body = await req.json();
  const hasTools = body.tools?.length > 0;
  const wantStream = body.stream === true;

  if (!hasTools) {
    return fetch(`${UPSTREAM}/chat/completions`, {
      method: "POST",
      headers: upstreamHeaders(),
      body: JSON.stringify(body),
    });
  }

  // ── Try native first ──
  const nativeRes = await fetch(`${UPSTREAM}/chat/completions`, {
    method: "POST",
    headers: upstreamHeaders(),
    body: JSON.stringify({ ...body, stream: false }),
  });

  if (!nativeRes.ok) return nativeRes;

  const result = await nativeRes.json() as any;
  const msg = result.choices?.[0]?.message;

  if (msg?.tool_calls?.length) {
    console.log(`[✓] ${body.model} → native tool_calls`);
    if (wantStream) return sseFromToolCalls(result);
    return Response.json(result);
  }

  // ── Native didn't return tool_calls → emulate ──
  console.log(`[~] ${body.model} → emulating tool calls...`);

  const toolPrompt = buildToolSystemPrompt(body.tools);
  const messages = [...(body.messages || [])];

  const sysIdx = messages.findIndex((m: any) => m.role === "system");
  if (sysIdx >= 0) {
    messages[sysIdx] = { ...messages[sysIdx], content: messages[sysIdx].content + toolPrompt };
  } else {
    messages.unshift({ role: "system", content: toolPrompt.trim() });
  }

  if (body.tool_choice === "required") {
    messages.push({ role: "user", content: "[SYSTEM: You MUST use at least one tool. Output <tool_call> blocks only.]" });
  } else if (body.tool_choice?.function) {
    messages.push({ role: "user", content: `[SYSTEM: You MUST call the "${body.tool_choice.function.name}" tool now.]` });
  }

  const emulatedBody = { ...body, messages, stream: false };
  delete emulatedBody.tools;
  delete emulatedBody.tool_choice;

  const emRes = await fetch(`${UPSTREAM}/chat/completions`, {
    method: "POST",
    headers: upstreamHeaders(),
    body: JSON.stringify(emulatedBody),
  });

  if (!emRes.ok) return emRes;

  const emResult = await emRes.json() as any;
  const emContent = emResult.choices?.[0]?.message?.content || "";
  const parsed = parseToolCalls(emContent);

  if (parsed?.toolCalls.length) {
    console.log(`[✓] ${body.model} → emulated ${parsed.toolCalls.length} tool call(s)`);
    const response = {
      ...emResult,
      choices: [{
        ...emResult.choices[0],
        finish_reason: "tool_calls",
        message: { role: "assistant", content: parsed.textContent || null, tool_calls: parsed.toolCalls },
      }],
    };
    if (wantStream) return sseFromToolCalls(response);
    return Response.json(response);
  }

  console.log(`[!] ${body.model} → emulation failed, returning text`);
  if (wantStream) return sseFromText(emResult);
  return Response.json(emResult);
}

// ── Main router ──
async function handleChat(req: Request): Promise<Response> {
  const body = await req.json();
  const model = body.model || "";

  // Route to ChatGPT provider if model matches
  if (isChatGPTModel(model)) {
    return handleChatGPTWithTools(body);
  }

  // Otherwise, use upstream (original behavior)
  // Re-create request since we consumed the body
  const newReq = new Request(req.url, {
    method: req.method,
    headers: req.headers,
    body: JSON.stringify(body),
  });
  return handleUpstreamChat(newReq);
}

const server = Bun.serve({
  port: PORT,
  hostname: process.env.HOST || "127.0.0.1",
  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname;

    if (req.method === "OPTIONS") {
      return new Response(null, {
        headers: { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "*", "Access-Control-Allow-Headers": "*" },
      });
    }

    try {
      if (path === "/v1/chat/completions" && req.method === "POST") return handleChat(req);

      // ── /v1/models — merge upstream + ChatGPT models ──
      if (path === "/v1/models" && req.method === "GET") {
        const chatgptModels = getChatGPTModelObjects();
        let upstreamModels: any[] = [];

        try {
          const upRes = await fetch(`${UPSTREAM}/models`, { headers: upstreamHeaders() });
          if (upRes.ok) {
            const upData = await upRes.json() as any;
            upstreamModels = upData.data || upData.models || [];
          }
        } catch {
          // upstream might not be available
        }

        return Response.json({
          object: "list",
          data: [...upstreamModels, ...chatgptModels],
        });
      }

      if (path === "/health" || path === "/") {
        const hasChatGPT = !!process.env.CHATGPT_COOKIES;
        return Response.json({
          status: "ok",
          proxy: "toolcall-middleware",
          version: "3.0.0",
          port: PORT,
          upstream: UPSTREAM,
          providers: {
            upstream: { enabled: true, url: UPSTREAM },
            chatgpt: { enabled: hasChatGPT, models: hasChatGPT ? getChatGPTModelObjects().map(m => m.id) : [] },
          },
          sessions: { active: getSessionCount() },
        });
      }

      return fetch(`${UPSTREAM}${path}`, { method: req.method, headers: upstreamHeaders(), body: req.method !== "GET" ? await req.text() : undefined });
    } catch (err: any) {
      console.error("[toolcall-proxy]", err);
      return Response.json({ error: { message: err.message, type: "proxy_error" } }, { status: 500 });
    }
  },
});

const hasChatGPT = !!process.env.CHATGPT_COOKIES;
console.log(`\n  toolcall-middleware v3.0.0`);
console.log(`  → http://127.0.0.1:${PORT}`);
console.log(`  → upstream: ${UPSTREAM}`);
console.log(`  → chatgpt: ${hasChatGPT ? "✓ enabled" : "✗ disabled (no CHATGPT_COOKIES)"}`);
console.log(`  → mode: session-reuse, compressed tools\n`);
if (hasChatGPT) {
  console.log(`  ChatGPT models:`);
  getChatGPTModelObjects().forEach(m => console.log(`    • ${m.id}`));
  console.log();
}
