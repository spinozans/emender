"""Read-only union progress snapshot; never manages or signals the service."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from .c14_paired_eval import OUT, write


def main():
    root=Path('/home/erikg/c14_consultation_features')
    service=(root/'union_service.jsonl').read_bytes()
    events=[json.loads(s) for s in service.splitlines()]
    start=[r['time'] for r in events if r['event']=='phase_start' and r['phase']=='transformer'][-1]
    data=(root/'registered_transformer/journal.jsonl').read_bytes()
    # Live snapshot may end with an in-flight write. Retain only complete lines.
    complete=data[:data.rfind(b'\n')+1]
    records=[json.loads(s) for s in complete.splitlines()]
    phase=[r for r in records if r['event']=='phase_start'][-1]
    done=[r for r in records if r['event']=='shard_complete']
    positions=sum(r['positions'] for r in done); windows=sum(r['windows'] for r in done)
    now=time.time(); baseline=5884.178857234429
    expected=phase['expected_positions']
    iso=lambda t:datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
    result=dict(observed_utc=iso(now),phase='transformer',phase_start_utc=iso(start),
        committed_windows=windows,committed_targets=positions,expected_windows=phase['expected_windows'],
        expected_targets=expected,cache_bytes=sum(r['bytes'] for r in done),
        snapshot_sha256=hashlib.sha256(complete).hexdigest(),snapshot_bytes=len(complete),
        observed_targets_per_second=positions/(now-start),baseline_targets_per_second=baseline,
        baseline_hours_from_start=expected/baseline/3600,
        baseline_boundary_utc=iso(start+expected/baseline),
        observed_rate_boundary_utc=iso(now+(expected-positions)/(positions/(now-start))) if positions else None,
        operator_45_hour_scenario_boundary_utc=iso(start+45*3600),
        projection_note='Conditional scenarios, not guarantees. Operator ~45h scenario retained; frozen continuation is 746,495,031 new targets, giving 35.24h at the historically receipted 5,884.18 targets/s. 972,375,733 total transformer exposure includes the existing 225,880,702-target Emender prefix.',
        service=subprocess.check_output(['systemctl','--user','show','emender-velocity-c14-union.service',
                 '-p','MainPID','-p','ActiveState','-p','SubState','-p','NRestarts'],text=True),
        service_modified=False,scope='journal/accounting read only; transformer shard hashes not re-audited')
    OUT.mkdir(exist_ok=True); write('transformer_progress.json',result)
    print(json.dumps(result))


if __name__=='__main__': main()
