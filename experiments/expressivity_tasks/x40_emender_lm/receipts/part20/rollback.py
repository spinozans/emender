"""Restore only this candidate's exact edits; refuse concurrent changes."""
import hashlib
import json
from pathlib import Path
r = Path(__file__).resolve().parent
rows = json.loads((r/'candidate.json').read_text())
for name, row in rows.items():
    path = Path(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['candidate_sha256'], name
for name, row in rows.items():
    path = Path(name)
    source = path.read_text()
    for old, new in reversed(row['replacements']):
        assert source.count(new) == 1
        source = source.replace(new, old)
    assert hashlib.sha256(source.encode()).hexdigest() == row['before_sha256']
    path.write_text(source)
(r/'rollback.json').write_text(json.dumps(dict(status='passed', production_restored=True,
    files={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in rows}),indent=2)+'\n')
