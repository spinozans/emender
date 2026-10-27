"""Resume the exact gated sources despite concurrent upstream working-tree edits.

Loads byte-verified source from the captured archive without changing any X31,
X34 or X36 files. Only source identity checks are redirected to the verified
manifest; numerical gates, data fingerprints and training code are unchanged.
"""
import hashlib
import importlib.abc
import importlib.util
import json
import linecache
from pathlib import Path
import sys
import tarfile

R=Path(__file__).resolve().parent
BASE=R.parents[1]
manifest=json.loads((R/'gate_sources.json').read_text())
entries={}
with tarfile.open(R/'gated_sources.tar.gz','r:gz') as archive:
    for name,expected in manifest.items():
        content=archive.extractfile(name).read()
        assert hashlib.sha256(content).hexdigest()==expected,name
        entries[name]=content
# Own experiment edits are never silently hidden by source pinning.
for name,expected in manifest.items():
    if name.startswith('x37_head_differentiation/'):
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==expected,name

modules={}
for name,content in entries.items():
    if not name.endswith('.py'):
        continue
    parts=name[:-3].split('/')
    package=parts[-1]=='__init__'
    if package:
        parts.pop()
    modules['experiments.expressivity_tasks.'+'.'.join(parts)]=(name,content,package)
    # Triton obtains JIT source through inspect/linecache, not only bytecode.
    # mtime=None keeps checkcache from replacing these verified source lines.
    filename=str(BASE/name)
    linecache.cache[filename]=(len(content),None,content.decode().splitlines(True),filename)

class Loader(importlib.abc.Loader):
    def __init__(self,name,content):
        self.name,self.content=name,content
    def get_source(self,fullname):
        return self.content.decode()
    def create_module(self,spec):
        return None
    def exec_module(self,module):
        exec(compile(self.content,str(BASE/self.name),'exec'),module.__dict__)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in modules:
            return None
        name,content,package=modules[fullname]
        return importlib.util.spec_from_file_location(fullname,BASE/name,
            loader=Loader(name,content),
            submodule_search_locations=[str((BASE/name).parent)] if package else None)

sys.path.insert(0,str(R.parents[3]))
sys.meta_path.insert(0,Finder())
from experiments.expressivity_tasks.x37_head_differentiation import cli
if sys.argv[1:] == ['--check']:
    import inspect
    from experiments.expressivity_tasks.x36_tape_perf import replay_setup
    # Exercise the real Triton source-discovery mechanism against the archive.
    count=0
    for value in vars(replay_setup).values():
        if hasattr(value,'fn') and hasattr(value,'src'):
            source=inspect.getsource(value.fn)
            assert source in entries['x36_tape_perf/replay_setup.py'].decode()
            assert value.src.strip() in source
            count+=1
    assert count>0
    print(f'{len(entries)} gated sources verified; {count} Triton source lookups pinned; own sources unchanged')
    raise SystemExit(0)
# All these bytes were individually checked above; the archive, not the
# concurrently changing filesystem, is the source authority for this process.
cli.sources=lambda:dict(manifest)
changed={name:dict(gated=expected,current=hashlib.sha256((BASE/name).read_bytes()).hexdigest())
         for name,expected in manifest.items()
         if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=expected}
cli.write(R/'pinned_execution.json',dict(archive='gated_sources.tar.gz',
    source_manifest='gate_sources.json',verified_files=len(entries),
    upstream_working_tree_differences=changed,
    reason='Continue the already parity-gated experiment without incorporating concurrent X36 edits.'))
if sys.argv[1:2] == ['--test-snapshot']:
    import pytest
    raise SystemExit(pytest.main([
        '-p','experiments.expressivity_tasks.x36_tape_perf.oracle_adapter',
        str(BASE/'x31_tape_cell_fused/tests'),
        str(BASE/'x37_head_differentiation/tests'),
        '-q','--junitxml='+str(R/'pinned_oracle.xml'),
    ]))
cli.main()
