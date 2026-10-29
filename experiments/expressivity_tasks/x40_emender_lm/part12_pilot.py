"""Fresh 100k pilot measurement; inherited curve identified, never regenerated."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from . import part6_pilot_checkpoint as pilot
from .part12_measure import gate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/part12/pilot'


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1', '5', '6')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['GPU_LEASE_HELD']
    gate(512, {})
    OUT.mkdir(exist_ok=True)
    # These qualify the unchanged pilot batch/curve and integer pool builder.
    # New source-bound read-adjoint byte gates are checked above, separately.
    for name in ('production_T512.json', 'production_T512_expandable.json', 'fused_pool_parity.json'):
        shutil.copyfile(ROOT / 'receipts/part8' / name, OUT / name)
    (OUT / 'curve').mkdir(exist_ok=True)
    src = ROOT / 'receipts/part8/matched_part8_pilot_B128_bare_fused_sparse/summary.json'
    shutil.copyfile(src, OUT / 'curve/summary.json')
    (OUT / 'inherited_curve.json').write_text(json.dumps(dict(path=str(src.relative_to(ROOT)), sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        scope='Inherited 22M B128 unchanged T512 curve; not fresh 1.3B quality evidence'), indent=2) + '\n')
    pilot.ROOT = OUT
    sys.argv = [sys.argv[0], '--batch', '128', '--no-checkpoint', '--fused-pool', '--steps', '20', '--label', 'fresh_gate']
    pilot.main()


if __name__ == '__main__':
    main()
