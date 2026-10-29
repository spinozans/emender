"""Fresh unchanged B128 pilot; inherit explicitly identified matched curve."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from . import part6_pilot_checkpoint as pilot


def main():
    root=Path(__file__).resolve().parent
    out=root/'receipts/part10/pilot';out.mkdir(exist_ok=True)
    for name in ('production_T512.json','production_T512_expandable.json','fused_pool_parity.json'):
        src=root/'receipts/part8'/name
        if not src.exists():src=root/'receipts/part8/production_T512.json'
        shutil.copyfile(src,out/name)
    (out/'curve').mkdir(exist_ok=True)
    src=root/'receipts/part8/matched_part8_pilot_B128_bare_fused_sparse/summary.json'
    shutil.copyfile(src,out/'curve/summary.json')
    (out/'inherited_curve.json').write_text(json.dumps(dict(path=str(src.relative_to(root)),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),scope='unchanged B128 bare fused-pool matched curve from part8; no fresh pilot quality claim'),indent=2)+'\n')
    pilot.ROOT=out
    sys.argv=[sys.argv[0],'--batch','128','--no-checkpoint','--fused-pool','--steps','20','--label','fresh_gate']
    pilot.main()

if __name__=='__main__':main()
