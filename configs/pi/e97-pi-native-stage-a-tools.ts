/** Stage-A safety surface: exact captured schemas; only read executes locally. */
import { readFileSync } from 'node:fs';
import { resolve, relative, isAbsolute } from 'node:path';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
export default function(pi:ExtensionAPI){
 const manifestPath=process.env.E97_PI_TOOL_MANIFEST;if(!manifestPath)throw new Error('tool manifest required');
 const manifest=JSON.parse(readFileSync(manifestPath,'utf8'));const wanted=['read','bash','edit','write','process'];
 const tools=manifest.model_visible_tools.filter((t:any)=>wanted.includes(t.name));if(tools.map((t:any)=>t.name).join(',')!==wanted.join(','))throw new Error('stage-A tool order');
 function confined(cwd:string,path:string){const p=resolve(cwd,path);const r=relative(cwd,p);if(isAbsolute(path)||r.startsWith('..')||isAbsolute(r))throw new Error('path outside Stage-A workspace');return p;}
 for(const tool of tools)pi.registerTool({name:tool.name,label:tool.label,description:tool.description,parameters:tool.parameters,
  async execute(_id:string,args:any,_signal:AbortSignal,_update:any,ctx:any){
   if(tool.name==='read'){
    const path=confined(ctx.cwd,args.path);const lines=readFileSync(path,'utf8').split(/\r?\n/);if(lines.at(-1)==='')lines.pop();const offset=args.offset??1,limit=args.limit??2000;
    const text=lines.slice(offset-1,offset-1+limit).map((v:string,i:number)=>`${offset+i}: ${v}`).join('\n');
    return{content:[{type:'text',text}],details:{stage_a_safe_read:true}};
   }
   return{content:[{type:'text',text:`ERROR: ${tool.name} execution is disabled in the Stage-A first-action measurement.`}],details:{stage_a_disabled:true,native_error:true}};
  }});
}
