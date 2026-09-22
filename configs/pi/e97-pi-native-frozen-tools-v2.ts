/** Pi-native E97 provider, v2 frozen-tool-specs compat variant.

Byte-identical to configs/pi/e97-pi-native.ts except for the single marked
change: the model request carries the transport config's frozen tool specs
(config.tools, the sealed e97-active-tool-surface-v1 manifest specs) instead
of pi-core's current extension-registered specs. The owner-side bridge
(scripts/e97_pi_native_tool_bridge.py) hard-requires the request tools to be
JSON-identical to the frozen manifest; pi 0.87.0's bundled web-access
extension drifted three web-tool description strings, which broke that
contract (pi_contract_mismatch) while the core tool specs stayed identical.
This variant pins the wire contract to the sealed manifest so collection
records render with protocol headers byte-identical to the v1 family records
and the frozen chat-probe panel. The owner still executes tools through real
Pi; nothing about the verification machinery changes.
*/
import { readFileSync } from 'node:fs';
import { createConnection } from 'node:net';
import { createAssistantMessageEventStream, type AssistantMessage } from '@earendil-works/pi-ai';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
const PROVIDER='e97-pi-native',MODEL='e97-4b-pi-native',API='e97-pi-native-unix-v1',MAX_FRAME=16*1024*1024;
export default function(pi:ExtensionAPI){
 const path=process.env.E97_PI_NATIVE_TOOL_CONFIG;if(!path)throw new Error('isolated Pi-native launcher required');
 const config=JSON.parse(readFileSync(path,'utf8'));if(config.schema!=='emender-e97-pi-native-tool-transport-v1')throw new Error('Pi-native config');
 let ready=false,settled=false;
 function rpc(op:string,payload:object={},signal?:AbortSignal):Promise<any>{return new Promise((resolve,reject)=>{
  if(signal?.aborted)return reject(new Error('aborted'));const socket=createConnection(config.socket);let data='',done=false;
  const fail=()=>{if(done)return;done=true;socket.destroy();reject(new Error('Pi-native owner stopped'));};const abort=()=>fail();signal?.addEventListener('abort',abort,{once:true});socket.setEncoding('utf8');socket.setTimeout(config.timeout_ms,fail);
  socket.once('connect',()=>{const wire=JSON.stringify({op,...payload})+'\n';if(Buffer.byteLength(wire)>MAX_FRAME)return fail();socket.write(wire);});
  socket.on('data',(chunk:string)=>{data+=chunk;if(Buffer.byteLength(data)>MAX_FRAME)return fail();if(!data.endsWith('\n'))return;try{const r=JSON.parse(data);if(r.ok!==true)return fail();done=true;socket.end();resolve(r.result);}catch{fail();}});
  socket.once('error',fail);socket.once('close',()=>{signal?.removeEventListener('abort',abort);if(!done)fail();});
 });}
 pi.on('before_agent_start',(event)=>{const o=event.systemPromptOptions;if(ready||settled||event.prompt!==config.prompt||o.customPrompt!==config.system||o.appendSystemPrompt||o.contextFiles?.length||o.skills?.length||event.images?.length||JSON.stringify([...pi.getActiveTools()].sort())!==JSON.stringify(config.tools.map((t:any)=>t.name).sort()))throw new Error('unsupported Pi-native session');ready=true;return{systemPrompt:config.system};});
 pi.on('session_before_compact',()=>({cancel:true}));pi.on('session_before_switch',()=>({cancel:true}));pi.on('session_before_fork',()=>({cancel:true}));pi.on('session_before_tree',()=>({cancel:true}));pi.on('user_bash',()=>({result:{output:'Host shell disabled.',exitCode:1,cancelled:true,truncated:false}}));
 pi.registerProvider(PROVIDER,{api:API,apiKey:'local-unix-no-credential',baseUrl:'http://127.0.0.1/unused',models:[{id:MODEL,name:'E97 Pi-native',reasoning:false,input:['text'],contextWindow:65536,maxTokens:4096,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}],
  streamSimple(model,context,options){const stream=createAssistantMessageEventStream();const message:AssistantMessage={role:'assistant',content:[],api:API,provider:PROVIDER,model:MODEL,timestamp:Date.now(),stopReason:'pending',usage:{input:0,output:0,cacheRead:0,cacheWrite:0,totalTokens:0,cost:{input:0,output:0,cacheRead:0,cacheWrite:0,total:0}}};
   void(async()=>{try{if(!ready||settled||model.id!==MODEL||model.provider!==PROVIDER)throw new Error('inactive');const r=await rpc('next',{systemPrompt:context.systemPrompt,messages:context.messages,tools:config.tools /* v2 compat: frozen manifest tool specs, not pi-core drifted ones */,model:model.id,provider:model.provider},options?.signal);stream.push({type:'start',partial:message});
    for(const block of r.message.content){const i=message.content.length;if(block.type==='text'){message.content.push({type:'text',text:''});stream.push({type:'text_start',contentIndex:i,partial:message});message.content[i]=block;stream.push({type:'text_delta',contentIndex:i,delta:block.text,partial:message});stream.push({type:'text_end',contentIndex:i,content:block.text,partial:message});}else{message.content.push({...block,arguments:{}});stream.push({type:'toolcall_start',contentIndex:i,partial:message});message.content[i]=block;stream.push({type:'toolcall_delta',contentIndex:i,delta:JSON.stringify(block.arguments),partial:message});stream.push({type:'toolcall_end',contentIndex:i,toolCall:block,partial:message});}}
    message.usage.input=r.input_tokens;message.usage.output=r.output_tokens;message.usage.totalTokens=r.input_tokens+r.output_tokens;message.stopReason=r.stop_reason;stream.push({type:'done',reason:r.stop_reason,message});
   }catch{message.stopReason=options?.signal?.aborted?'aborted':'error';message.errorMessage='Pi-native transport stopped; inspect private owner receipt.';stream.push({type:'error',reason:message.stopReason,error:message});}finally{stream.end();}})();return stream;}});
 pi.on('agent_settled',async(_event,ctx)=>{ready=false;settled=true;const entries=ctx.sessionManager.getBranch();if(entries.some(e=>['compaction','branch_summary','custom_message'].includes(e.type)))throw new Error('unsupported Pi-native history');const messages=entries.filter(e=>e.type==='message').map((e:any)=>e.message);await rpc('close',{messages});ctx.shutdown();});
}
