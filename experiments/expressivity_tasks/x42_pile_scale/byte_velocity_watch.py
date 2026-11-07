"""Artifact heartbeat: inspect actual run growth, never service activity flags."""
import json
from pathlib import Path
import time
from .velocity_state import atomic_json
ROOT=Path(__file__).resolve().parent

def snapshot():
    result={}
    for arm in ('transformer','gdn2','pure'):
        root=ROOT/'runs'/('byte_velocity_'+arm+'_v1')
        state=root/'generation.json'
        saved=json.loads(state.read_text()) if state.exists() else {}
        logs=sorted(root.glob('g*/attempt*/training.jsonl'),key=lambda p:p.stat().st_mtime_ns)
        path=logs[-1] if logs else None
        step=None
        if path and path.stat().st_size:
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-8192)); lines=f.read().splitlines()
            for line in reversed(lines):
                try: step=json.loads(line); break
                except ValueError: pass
        st=path.stat() if path else None
        result[arm]=dict(generation=saved.get('cma',{}).get('generation',0)+1,
            banked_scores=len(saved.get('completed',[])),log=str(path) if path else None,
            log_bytes=st.st_size if st else 0,last_step=step,
            artifact_age_seconds=time.time()-st.st_mtime if st else None,
            stale=not st or time.time()-st.st_mtime>1800,
            failure=json.loads((root/'failure.json').read_text()) if (root/'failure.json').exists() else None,
            nomination=(root/'nomination.json').exists())
    return dict(time_ns=time.time_ns(),arms=result)

def main():
    out=ROOT/'runs/byte_velocity_heartbeat'; out.mkdir(exist_ok=True)
    with (out/'polls.jsonl').open('a',buffering=1) as f:
        while True:
            result=snapshot(); atomic_json(out/'latest.json',result)
            f.write(json.dumps(result)+'\n'); time.sleep(60)

if __name__=='__main__': main()
