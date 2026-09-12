#!/usr/bin/env python3
"""CPU-only PTX inventory comparison, not a complete executable/runtime identity."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(data):return hashlib.sha256(data).hexdigest()


def normalized_ptx(text):
    lines=[];debug=False;opened=False;depth=0
    for line in text.splitlines():
        code=line.split('//',1)[0].strip()
        if re.match(r'^\.section\s+\.debug_\w+\s*$',code):
            if debug:raise ValueError('nested debug section')
            debug=True;opened=False;depth=0
            continue
        if debug:
            depth+=code.count('{')-code.count('}');opened=opened or '{' in code
            if depth<0:raise ValueError('debug section braces')
            if opened and depth==0:debug=False
            continue
        if not code or re.match(r'^\.(file|loc)\s',code):continue
        lines.append(line.strip())
    if debug:raise ValueError('unterminated debug section')
    if not lines:raise ValueError('empty PTX body')
    return '\n'.join(lines)


def inventory(root):
    files=sorted(root.rglob('*.ptx'))
    if not files or len(files)>10000:raise ValueError('cache inventory bound')
    result=[]
    for p in files:
        if p.stat().st_size>16*1024**2:raise ValueError('PTX size bound')
        raw=p.read_bytes();meta_path=p.with_suffix('.json');metadata=json.loads(meta_path.read_text())
        metadata.pop('hash',None)
        result.append(dict(path=str(p),name=p.stem,ptx_sha256=digest(raw),
            normalized_ptx_sha256=digest(normalized_ptx(raw.decode()).encode()),
            compiler_metadata_sha256=digest(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode()),
            cubin_sha256=digest(p.with_suffix('.cubin').read_bytes())))
    return result


def audit(left,right):
    a=inventory(left);b=inventory(right);counts={}
    for name in sorted({r['name'] for r in a+b}):
        x={r['normalized_ptx_sha256'] for r in a if r['name']==name};y={r['normalized_ptx_sha256'] for r in b if r['name']==name}
        u={(r['normalized_ptx_sha256'],r['compiler_metadata_sha256']) for r in a if r['name']==name}
        v={(r['normalized_ptx_sha256'],r['compiler_metadata_sha256']) for r in b if r['name']==name}
        counts[name]=dict(left_files=sum(r['name']==name for r in a),right_files=sum(r['name']==name for r in b),
                         left_unique_ptx=len(x),right_unique_ptx=len(y),common_unique_ptx=len(x&y),
                         right_ptx_subset_of_left=y<=x,right_ptx_and_metadata_subset_of_left=v<=u)
    return dict(schema='emender-triton-cache-overlap-v1',left=a,right=b,kernels=counts,
                limitation='Normalized PTX/config overlap only; not launch-order, operand, cuBLAS/PyTorch, SASS or complete runtime identity')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.left,args.right)
    with args.output.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result['kernels'],sort_keys=True))
