"""Documented xsa pipeline receipts; never changes the xsa checkout."""
import bz2
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part1d'
XSA = Path.home() / 'sxgc/xsa/target/release/xsa'
PYTHON = '/home/erikg/emender/.venv/bin/python'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def run(name, args):
    env = dict(os.environ, PATH=str(Path(PYTHON).parent)+':'+os.environ['PATH'])
    p = subprocess.run([str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    (OUT/f'{name}.stdout').write_bytes(p.stdout)
    (OUT/f'{name}.stderr').write_bytes(p.stderr)
    (OUT/f'{name}.command.json').write_text(json.dumps(dict(argv=list(map(str,args)),returncode=p.returncode),indent=2)+'\n')
    print(name,p.returncode,flush=True)
    return p

def prepare():
    OUT.mkdir(exist_ok=True)
    raw=(ROOT/'input.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    title='TinyShakespeare'
    # xsa copies XML text verbatim, including entities. A lexical wrapper is
    # necessary to keep source bytes unchanged; RESULT_PART1D.md records the
    # resulting XML well-formedness limitation (literal ampersands).
    xml=('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" version="0.11" xml:lang="en"><page><title>'+title+'</title><ns>0</ns><revision><text xml:space="preserve">'+raw.decode('utf-8')+'</text></revision></page></mediawiki>').encode()
    (OUT/'shakespeare.xml').write_bytes(xml)
    (OUT/'shakespeare.xml.bz2').write_bytes(bz2.compress(xml))
    assert run('corpus',[XSA,'corpus','--wikipedia',OUT/'shakespeare.xml.bz2','--out',OUT/'corpus','--separator','0x1e']).returncode==0
    corpus=OUT/'corpus/corpus.bin'
    assert corpus.read_bytes()==title.encode()+b'\n'+raw+b'\x1e'
    binding=dict(source_sha256=digest(ROOT/'input.txt'),source_bytes=len(raw),body_offset=len(title)+1,corpus_sha256=digest(corpus),corpus_bytes=corpus.stat().st_size,xsa_sha256=digest(XSA),reproduce_sha256=digest(Path.home()/'sxgc/REPRODUCE.md'),supervisor_tool_available=False)
    (OUT/'binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    args=[XSA,'build','--tape','--input',corpus,'--scratch',OUT/'build','--threads','1','--memory-gb','2','--timeout','3600']
    assert run('plan',args+['--plan-only']).returncode==0
    assert run('build',args).returncode==0

if __name__=='__main__':
    prepare()
