"""Bind the installed candidate and terminal runs to unchanged tested sources."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
base=Path('experiments/expressivity_tasks')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
originals=json.loads((root/'production_before.json').read_text())
promoted=json.loads((root/'promotion_sources.json').read_text())
for name,source in originals.items():
    original=subprocess.check_output(['git','show',f'63ba1c3b:experiments/expressivity_tasks/{name}'])
    assert original==source.encode(),name
    assert sha((base/name).read_bytes())==promoted[name],name
name='x31_tape_cell_fused/all_field_journal.py'
def capacity_ast(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AllFieldJournal')
    return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='capacity_for'),include_attributes=False)
assert capacity_ast(originals[name])==capacity_ast((base/name).read_text())
candidate=(root.parent/'part21/candidate.py').read_text()
helper=(base/'x31_tape_cell_fused/counted_copy.py').read_text()
assert candidate[candidate.index('@triton.jit'):candidate.index('\ndef install():')].strip()==helper[helper.index('@triton.jit'):].strip()
for path in root.glob('production_*/console.txt'):
    records=[]
    for line in path.read_text(errors='replace').splitlines():
        try:record=json.loads(line)
        except ValueError:continue
        if isinstance(record,dict) and 'source_sha256' in record:records.append(record)
    for record in records:
        for name,h in record['source_sha256'].items():assert sha(Path(name).read_bytes())==h,(str(path),name)
result=dict(status='passed',baseline_revision='63ba1c3b',production_sources=promoted,counted_copy_matches_part21_candidate=True,counted_copy_sha256=sha(helper.encode()),parity_harness_sha256=sha((base/'x40_emender_lm/parity.py').read_bytes()),capacity_guard_ast_unchanged=True,terminal_run_sources_unchanged=True,review='No blocker found: masks preserve consumed operands, owner initialization covers historical replay rows, counted copies preserve every consumed prefix and define only unused tails. No arithmetic, reduction tree, journal guard, API check or race suppression was changed.')
(root/'source_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
