/** Explicit opt-in, single-episode Pi front end. No native tools run on the host.
 * The private native transcript stays in the owner process behind a Unix socket.
 * Use only the isolated launcher; this is not a general installed Pi profile.
 */
import { readFileSync } from 'node:fs';
import { createConnection } from 'node:net';
import { createAssistantMessageEventStream, type AssistantMessage } from '@earendil-works/pi-ai';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { Type } from 'typebox';

const PROVIDER = 'e97-openhands-compat';
const MODEL = 'e97-4b-native-bridge';
const API = 'e97-native-unix-v1';
const MAX_FRAME = 16 * 1024 * 1024;

export default function (pi: ExtensionAPI) {
  const path = process.env.E97_PI_NATIVE_CONFIG;
  if (!path || process.env.PI_OFFLINE !== '1') throw new Error('isolated native launcher required');
  const config = JSON.parse(readFileSync(path, 'utf8'));
  const single = config.schema === 'emender-e97-pi-native-transport-v1';
  const multi = config.schema === 'emender-e97-pi-native-session-transport-v1';
  if (!single && !multi) throw new Error('native transport config');
  const tasks = single ? [{ task_id: 'single', prompt: config.prompt }] : config.tasks;
  if (!Array.isArray(tasks) || tasks.length < 1 || tasks.length > 4) throw new Error('native task config');
  let ready = false;
  let settled = false;
  let taskIndex = 0;

  function rpc(op: string, payload: object = {}, signal?: AbortSignal): Promise<any> {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) return reject(new Error('native transport aborted'));
      const socket = createConnection(config.socket);
      let data = '';
      let completed = false;
      function fail() {
        if (completed) return;
        completed = true;
        socket.destroy();
        reject(new Error('native transport stopped; inspect private owner receipt'));
      }
      const abort = () => fail();
      signal?.addEventListener('abort', abort, { once: true });
      socket.setEncoding('utf8');
      socket.setTimeout(config.timeout_ms, fail);
      socket.once('connect', () => {
        const wire = JSON.stringify({ op, ...payload }) + '\n';
        if (Buffer.byteLength(wire) > MAX_FRAME) return fail();
        socket.write(wire);
      });
      socket.on('data', (chunk: string) => {
        data += chunk;
        if (Buffer.byteLength(data) > MAX_FRAME) return fail();
        if (!data.endsWith('\n')) return;
        try {
          const response = JSON.parse(data);
          if (response.ok !== true) return fail();
          completed = true;
          socket.end();
          resolve(response.result);
        } catch { fail(); }
      });
      socket.once('error', fail);
      socket.once('close', () => {
        signal?.removeEventListener('abort', abort);
        if (!completed) fail();
      });
    });
  }

  for (const tool of config.tools) {
    pi.registerTool({
      name: tool.name,
      label: `OpenHands: ${tool.name}`,
      description: tool.description,
      // Validate only the transport. Original invalid arguments must reach the
      // original executor unchanged, including numeric spelling and huge ints.
      parameters: Type.Object({ call_ref: Type.String(), native_arguments_json: Type.Optional(Type.String()) },
                              { additionalProperties: false }),
      async execute(id, args, signal) {
        if (!ready || settled) throw new Error('native session not active');
        return rpc('execute', { id, name: tool.name, arguments: args }, signal);
      },
    });
  }
  pi.on('tool_result', (event) => {
    if (config.tools.some((t: any) => t.name === event.toolName) &&
        typeof (event.details as any)?.native_error === 'boolean') {
      return { isError: (event.details as any).native_error };
    }
  });
  pi.on('before_agent_start', async (event) => {
    const o = event.systemPromptOptions;
    const task = tasks[taskIndex];
    if (ready || settled || !task || o.customPrompt !== config.system || o.appendSystemPrompt ||
        o.contextFiles?.length || o.skills?.length || event.images?.length ||
        event.prompt !== task.prompt ||
        JSON.stringify([...pi.getActiveTools()].sort()) !== JSON.stringify(config.tools.map((t: any) => t.name).sort())) {
      ready = false;
      throw new Error('unsupported native session configuration');
    }
    if (multi) await rpc('begin_task', { task_id: task.task_id, prompt: task.prompt,
                                        acknowledge_fresh_record: true });
    ready = true;
    // Remove only Pi's known generated date/cwd scaffold in this explicitly
    // isolated profile, never loaded project instructions or arbitrary context.
    return { systemPrompt: config.system };
  });
  pi.on('session_before_compact', () => ({ cancel: true }));
  pi.on('session_before_switch', () => ({ cancel: true }));
  pi.on('session_before_fork', () => ({ cancel: true }));
  pi.on('session_before_tree', () => ({ cancel: true }));
  pi.on('user_bash', () => ({ result: { output: 'Host shell disabled in native compatibility sessions.',
                                       exitCode: 1, cancelled: true, truncated: false } }));

  pi.registerProvider(PROVIDER, {
    api: API, apiKey: 'local-unix-no-credential', baseUrl: 'http://127.0.0.1/unused',
    models: [{ id: MODEL, name: 'E97 unchanged weights / OpenHands compatibility', reasoning: false,
      input: ['text'], contextWindow: 65536, maxTokens: 4096,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
    streamSimple(model, context, options) {
      const stream = createAssistantMessageEventStream();
      const message: AssistantMessage = { role: 'assistant', content: [], api: API, provider: PROVIDER,
        model: MODEL, timestamp: Date.now(), stopReason: 'pending',
        usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
                 cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
      void (async () => {
        try {
          if (!ready || settled || model.id !== MODEL || model.provider !== PROVIDER) throw new Error('inactive');
          const result = await rpc('next', { systemPrompt: context.systemPrompt, messages: context.messages,
            tools: context.tools, model: model.id, provider: model.provider }, options?.signal);
          stream.push({ type: 'start', partial: message });
          for (const block of result.message.content) {
            const i = message.content.length;
            if (block.type === 'text') {
              message.content.push({ type: 'text', text: '' });
              stream.push({ type: 'text_start', contentIndex: i, partial: message });
              message.content[i] = block;
              stream.push({ type: 'text_delta', contentIndex: i, delta: block.text, partial: message });
              stream.push({ type: 'text_end', contentIndex: i, content: block.text, partial: message });
            } else {
              message.content.push({ ...block, arguments: {} });
              stream.push({ type: 'toolcall_start', contentIndex: i, partial: message });
              message.content[i] = block;
              stream.push({ type: 'toolcall_delta', contentIndex: i, delta: JSON.stringify(block.arguments), partial: message });
              stream.push({ type: 'toolcall_end', contentIndex: i, toolCall: block, partial: message });
            }
          }
          message.usage.input = result.input_tokens;
          message.usage.output = result.output_tokens;
          message.usage.totalTokens = result.input_tokens + result.output_tokens;
          message.stopReason = 'toolUse';
          stream.push({ type: 'done', reason: 'toolUse', message });
        } catch {
          message.stopReason = options?.signal?.aborted ? 'aborted' : 'error';
          message.errorMessage = 'Native compatibility transport stopped; inspect private owner receipt.';
          stream.push({ type: 'error', reason: message.stopReason, error: message });
        } finally { stream.end(); }
      })();
      return stream;
    },
  });
  pi.on('agent_settled', async (_event, ctx) => {
    ready = false;
    const entries = ctx.sessionManager.getBranch();
    if (entries.some((e) => ['compaction', 'branch_summary', 'custom_message'].includes(e.type))) {
      throw new Error('unsupported native session history');
    }
    const messages = entries.filter((e) => e.type === 'message').map((e: any) => e.message);
    if (single) {
      settled = true;
      await rpc('close', { messages });
      return;
    }
    await rpc('settle_task', { messages });
    taskIndex += 1;
    if (taskIndex === tasks.length) {
      settled = true;
      await rpc('close_session');
      ctx.shutdown();
    }
  });
}
