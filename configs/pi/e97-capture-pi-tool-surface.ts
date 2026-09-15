import { appendFileSync, chmodSync, renameSync, writeFileSync } from 'node:fs';
import { createAssistantMessageEventStream, type AssistantMessage } from '@earendil-works/pi-ai';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';

const PROVIDER = 'e97-tool-surface-capture';
const MODEL = 'capture-only';
const API = 'e97-tool-surface-capture-v1';
const PROMPT = 'E97_CAPTURE_ACTIVE_PI_TOOL_SURFACE_V1';

export default function captureToolSurface(pi: ExtensionAPI) {
  const output = process.env.E97_PI_TOOL_SURFACE_OUTPUT;
  const trace = process.env.E97_PI_TOOL_SURFACE_TRACE;
  if (!output || !trace) throw new Error('capture output and trace are required');
  appendFileSync(trace, 'extension-loaded\n', { mode: 0o600 });
  let captured = false;
  pi.registerProvider(PROVIDER, {
    api: API,
    apiKey: 'local-capture-no-credential',
    baseUrl: 'http://127.0.0.1/unused',
    models: [{ id: MODEL, name: 'Capture active Pi tool surface', reasoning: false,
      input: ['text'], contextWindow: 65536, maxTokens: 32,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
    streamSimple(model, context) {
      appendFileSync(trace, 'stream-invoked\n');
      const stream = createAssistantMessageEventStream();
      const message: AssistantMessage = { role: 'assistant', content: [], api: API,
        provider: PROVIDER, model: MODEL, timestamp: Date.now(), stopReason: 'pending',
        usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
      try {
        if (captured || model.id !== MODEL || model.provider !== PROVIDER ||
            context.messages.length !== 1 || context.messages[0].role !== 'user') {
          throw new Error('unexpected capture invocation');
        }
        const block = context.messages[0].content;
        const text = typeof block === 'string' ? block :
          Array.isArray(block) && block.length === 1 && block[0].type === 'text' ? block[0].text : '';
        if (text !== PROMPT) throw new Error('unexpected capture prompt');
        const names = context.tools.map((tool: any) => tool.name);
        if (names.length !== new Set(names).size) throw new Error('duplicate active tool name');
        const payload = { schema: 'emender-e97-active-pi-tool-surface-v1',
          provider: PROVIDER, model: MODEL, tools: context.tools };
        const temporary = output + '.tmp-' + process.pid;
        writeFileSync(temporary, JSON.stringify(payload, null, 2) + '\n', { mode: 0o600, flag: 'wx' });
        chmodSync(temporary, 0o600); renameSync(temporary, output); captured = true;
        const answer = 'captured'; message.content = [{ type: 'text', text: answer }]; message.stopReason = 'stop';
        stream.push({ type: 'start', partial: message });
        stream.push({ type: 'text_start', contentIndex: 0, partial: message });
        stream.push({ type: 'text_delta', contentIndex: 0, delta: answer, partial: message });
        stream.push({ type: 'text_end', contentIndex: 0, content: answer, partial: message });
        stream.push({ type: 'done', reason: 'stop', message });
      } catch (error) {
        message.stopReason = 'error'; message.errorMessage = String(error);
        stream.push({ type: 'error', reason: 'error', error: message });
      } finally { stream.end(); }
      return stream;
    },
  });
  appendFileSync(trace, 'provider-registered\n');
}
