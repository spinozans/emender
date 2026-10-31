"""Retain exact generated artifacts before launch; no code mutation."""
import gzip
import hashlib
import json
from pathlib import Path
from experiments.expressivity_tasks.x40_emender_lm import forward


def install():
    original = forward.scan_kernel
    out = Path(__file__).resolve().parent

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                kernel = original.warmup(*args, grid=grid, **kwargs)
                prefix = out / ('capture_' + kernel.hash)
                if not prefix.with_suffix('.json').exists():
                    records = {}
                    for name in ('ptx', 'cubin', 'ttir', 'ttgir'):
                        value = kernel.asm[name]
                        value = value.encode() if isinstance(value, str) else value
                        with gzip.open(str(prefix) + '.' + name + '.gz', 'wb') as f:
                            f.write(value)
                        records[name] = hashlib.sha256(value).hexdigest()
                    Path(str(prefix) + '.json').write_text(json.dumps(dict(
                        kernel_hash=kernel.hash, artifacts=records, source=original.src,
                        mutation=False), indent=2) + '\n')
                return original[grid](*args, **kwargs)
            return launch
    forward.scan_kernel = Capture()
