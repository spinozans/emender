/** Credential-free collection tool surface: exact schemas, Unix-owner execution. */
import { readFileSync } from 'node:fs';
import { createConnection } from 'node:net';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
const MAX_FRAME=16*1024*1024;
export default function(pi:ExtensionAPI){
 const path=process.env.E97_PI_COLLECTION_PROXY_CONFIG;if(!path)throw new Error('collection proxy config required');
 const config=JSON.parse(readFileSync(path,'utf8'));if(config.schema!=='emender-e97-pi-native-collection-proxy-v1')throw new Error('collection proxy config');
 const manifest=JSON.parse(readFileSync(config.manifest,'utf8'));const selected=new Set(config.proxy_tools);const tools=manifest.model_visible_tools.filter((t:any)=>selected.has(t.name));
 if(tools.length!==selected.size)throw new Error('collection proxy tool coverage');
 function rpc(payload:object,signal?:AbortSignal):Promise<any>{return new Promise((resolve,reject)=>{
  if(signal?.aborted)return reject(new Error('aborted'));const socket=createConnection(config.socket);let data='',done=false;
  const fail=(message='collection owner stopped')=>{if(done)return;done=true;socket.destroy();reject(new Error(message));};const abort=()=>fail('aborted');signal?.addEventListener('abort',abort,{once:true});socket.setEncoding('utf8');socket.setTimeout(config.timeout_ms,()=>fail('collection owner timeout'));
  socket.once('connect',()=>{const wire=JSON.stringify(payload)+'\n';if(Buffer.byteLength(wire)>MAX_FRAME)return fail('request too large');socket.write(wire);});
  socket.on('data',(chunk:string)=>{data+=chunk;if(Buffer.byteLength(data)>MAX_FRAME)return fail('response too large');if(!data.endsWith('\n'))return;try{const value=JSON.parse(data);if(value.ok!==true)return fail(value.error||'collection owner rejected');done=true;socket.end();resolve(value.result);}catch(error){fail(String(error));}});
  socket.once('error',()=>fail());socket.once('close',()=>{signal?.removeEventListener('abort',abort);if(!done)fail();});
 });}
 for(const tool of tools)pi.registerTool({name:tool.name,label:tool.label,description:tool.description,parameters:tool.parameters,
  async execute(id:string,args:any,signal:AbortSignal,_update:any){const result=await rpc({op:'execute',id,name:tool.name,arguments:args},signal);return{content:[{type:'text',text:result.text}],details:{collection_proxy:true,receipt:result.receipt},isError:result.is_error};}});
 function webAllowed(name:string,input:any){const rules=config.web_allow?.[name];if(!Array.isArray(rules))return false;if(name==='fetch_content'){const urls=input.urls??(input.url?[input.url]:[]);return Array.isArray(urls)&&urls.length>0&&urls.every((u:any)=>rules.includes(u));}if(name==='get_search_content')return typeof input.responseId==='string'&&/^[A-Za-z0-9_-]{8,200}$/.test(input.responseId);const text=name==='source_check'?input.claim:(input.query??(Array.isArray(input.queries)?input.queries.join(' '):''));return typeof text==='string'&&rules.some((needle:string)=>text.toLowerCase().includes(needle.toLowerCase()));}
 pi.on('tool_call',(event:any)=>{if(selected.has(event.toolName))return;if(config.web_allow&&webAllowed(event.toolName,event.input))return;if(config.block_unproxied_tools)return{block:true,reason:'Tool is outside this collection task authority.',terminate:false};});
 pi.on('session_before_compact',()=>({cancel:true}));pi.on('session_before_switch',()=>({cancel:true}));pi.on('session_before_fork',()=>({cancel:true}));pi.on('session_before_tree',()=>({cancel:true}));pi.on('user_bash',()=>({result:{output:'Host shell disabled.',exitCode:1,cancelled:true,truncated:false}}));
}
