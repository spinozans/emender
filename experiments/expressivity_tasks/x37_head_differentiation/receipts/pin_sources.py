"""Archive the parity-gated runtime, with no edits to imported experiments."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

R=Path(__file__).resolve().parent
BASE=R.parents[1]
manifest=json.loads((R/'gate_sources.json').read_text())
recovered=[]
with tarfile.open(R/'gated_sources.tar.gz','w:gz') as archive:
    for name,expected in manifest.items():
        content=(BASE/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            content=subprocess.check_output(['git','show','b30edb8b:experiments/expressivity_tasks/'+name])
            recovered.append(name)
        assert hashlib.sha256(content).hexdigest()==expected,name
        info=tarfile.TarInfo(name)
        info.size=len(content)
        archive.addfile(info,io.BytesIO(content))
(R/'source_archive.json').write_text(json.dumps(dict(
    manifest='gate_sources.json',archive='gated_sources.tar.gz',
    sha256=hashlib.sha256((R/'gated_sources.tar.gz').read_bytes()).hexdigest(),
    files=len(manifest),recovered_from_baseline_git=recovered,
    baseline_commit='b30edb8b',reason='Pin parity-gated imports while concurrent X36 part-5 work edits the shared source tree.'),indent=2)+'\n')
