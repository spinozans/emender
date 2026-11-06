"""Deliver the completed query phase before permitting heldout measurements."""
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from .p100_verify import OUT, LIMIT
from .c14_documents import sha
ROOT=Path('/home/erikg/emender')
REL=OUT.relative_to(ROOT)

def main():
    assert sys.executable==str(ROOT/'.venv/bin/python')
    while not (OUT/'verdict.json').exists():
        try:stat=Path('/proc/502753/stat').read_text().rsplit(') ',1)[1].split()
        except FileNotFoundError:raise RuntimeError('Verifier disappeared before verdict; do not proceed')
        assert stat[0]!='Z','Verifier died; do not proceed'
        time.sleep(10)
    v=json.loads((OUT/'verdict.json').read_text());assert v['status']=='passed' and v['region_bytes_scanned']==LIMIT
    report=ROOT/'experiments/expressivity_tasks/x43_fork_resolved/RESULT_P100_QUERY_VERIFICATION.md'
    report.write_text('# P100 independent sampled query verification\n\nPassed17 complete native branch histograms versus independent brute-force scans of all100000000000 raw pile.txt bytes. Every256-byte histogram, interval width and EOF count matches, including64-NUL negative control. Pattern offsets and hex are registered before measurement. Raw region SHA256: '+v['raw_region_sha256']+'.\n\nThe serving index and raw source stat identities and executable SHA256 match registration. No corpus materialized, native index modified, GPU acquired, union service changed or heldout evaluation run by this phase. This is a sampled query certificate, not exhaustive SA/BWT/closure/T3 certification. Full verdict/checks: receipts/p100_query_verification/verdict.json. No DiLoCo behavior; gap IDs inapplicable.\n')
    lock=(ROOT/'.git/x43_delivery.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).strip()==b'main'
    subprocess.run(['git','diff','--cached','--exit-code','--quiet'],cwd=ROOT,check=True)
    names=['registration.json','launch.json','native.json','scan.json','verdict.json','console.txt','server.stderr']
    subprocess.run(['git','add',str(report.relative_to(ROOT)),*[str(REL/n) for n in names]],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','X43 P100: query verification'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    record=dict(status='pushed',commit=commit,verdict_sha256=sha(OUT/'verdict.json'),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p=OUT/'delivery.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2)+'\n');t.replace(p)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
