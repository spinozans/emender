#!/usr/bin/env python3
"""Freeze exact active Pi tool schemas and installed extension identities."""
import argparse,hashlib,json,os
from pathlib import Path
EXPECTED={'core':['read','bash','edit','write'],'process':['process'],'fff':['ffgrep','fffind'],
          'web':['web_search','source_check','fetch_content','get_search_content']}
PACKAGES={
 'process':('/home/erikg/.pi/agent/npm/node_modules/@aliou/pi-processes','extensions/processes/index.ts'),
 'fff':('/home/erikg/.pi/agent/npm/node_modules/@ff-labs/pi-fff','src/index.ts'),
 'web':('/home/erikg/.pi/agent/npm/node_modules/pi-web-access','index.ts'),
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def tree_identity(root):
 root=Path(root); rows=[]
 for p in sorted(x for x in root.rglob('*') if x.is_file() and 'node_modules' not in x.relative_to(root).parts):
  rows.append((p.relative_to(root).as_posix(),sha(p)))
 text=''.join(f'{n}\0{h}\n' for n,h in rows).encode()
 return dict(file_count=len(rows),tree_sha256=hashlib.sha256(text).hexdigest())

def load_groups(root):
 root=Path(root);groups={};seen=set()
 for group,names in EXPECTED.items():
  path=root/group/'tools.json';raw=json.loads(path.read_text())
  if raw.get('schema')!='emender-e97-active-pi-tool-surface-v1':raise ValueError('capture schema')
  tools=raw.get('tools')
  if not isinstance(tools,list) or [t.get('name') for t in tools]!=names:raise ValueError(f'{group} tool coverage/order')
  for t in tools:
   if set(t)!={'name','label','description','parameters'} or t['name'] in seen:raise ValueError('tool shape/duplicate')
   if not isinstance(t['description'],str) or not isinstance(t['parameters'],dict):raise ValueError('tool schema types')
   seen.add(t['name'])
  groups[group]=dict(capture_sha256=sha(path),tools=tools)
 return groups

def freeze(args):
 groups=load_groups(args.capture_root);packages={}
 for name,(root,entry) in PACKAGES.items():
  root=Path(root);meta=json.loads((root/'package.json').read_text());identity=tree_identity(root)
  packages[name]=dict(name=meta['name'],version=meta['version'],package_json_sha256=sha(root/'package.json'),
                      entry=str(root/entry),entry_sha256=sha(root/entry),**identity)
 manifest=dict(schema='emender-e97-pi-tool-surface-authority-v1',pi_version='0.85.1',pi_bin=str(args.pi_bin.resolve()),
  pi_bin_sha256=sha(args.pi_bin),pi_inventory=str(args.pi_inventory.resolve()),pi_inventory_sha256=sha(args.pi_inventory),
  capture_extension_sha256=sha(args.capture_extension),capture_script_sha256=sha(args.capture_script),
  groups=groups,packages=packages,tool_order=[name for group in EXPECTED for name in EXPECTED[group]],
  model_visible_tools=sum((groups[g]['tools'] for g in EXPECTED),[]),training_eligible=False)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(manifest,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
 print('PI_TOOL_SURFACE_FROZEN',len(manifest['tool_order']),sha(args.output))

def main():
 p=argparse.ArgumentParser();p.add_argument('--capture-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--pi-bin',type=Path,required=True);p.add_argument('--pi-inventory',type=Path,required=True)
 p.add_argument('--capture-extension',type=Path,default=Path('configs/pi/e97-capture-pi-tool-surface.ts'))
 p.add_argument('--capture-script',type=Path,default=Path('scripts/capture_e97_pi_tool_surface.sh'));freeze(p.parse_args())
if __name__=='__main__':main()
