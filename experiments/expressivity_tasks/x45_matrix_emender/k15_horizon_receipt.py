"""Seal one completed stage after its detached lease has been released."""
import gzip,json,sys,time
from pathlib import Path
from .k15_horizon_summary import summary
from .k15_historical_training import j,ANCHOR_SHA


def receipt(stage):
    out=Path('output/k15M')/stage; dest=Path(__file__).parent/'receipts/k15_horizon'/stage; dest.mkdir(parents=True,exist_ok=False)
    launch=json.loads((out/'launch.json').read_text()); pid=launch['pid']; proc=Path(f'/proc/{pid}/stat'); assert not proc.exists() or proc.read_text().split()[2]=='Z' or proc.read_text().split()[21]!=launch['starttime']; initial=(out/'initial').read_text(); fields=dict(l.split('=') for l in initial.splitlines() if '=' in l); assert int(fields['pid'])==pid and fields['starttime']==launch['starttime'] and initial.splitlines()[-1].split()[:3]==[str(pid),'1',str(pid)]; assert not Path('.wg/gpu_leases',fields['gpu']).exists(); hb=(out/'heartbeat').read_text(); assert max(int(l.split('=')[1]) for l in hb.splitlines() if l.startswith('heartbeat='))>int(fields['created']); assert len({l for l in hb.splitlines() if l.startswith('{"step":')})>1
    assert json.loads((out/'console').read_text().splitlines()[-1])=={'complete':True}; result=summary(out); assert j.digest(out/'rows.jsonl')==result['rows']; now=time.time(); opening=3536 if stage=='pilot' else json.loads((dest.parent/('pilot' if stage=='dev' else 'dev')/'proof.json').read_text())['upper_cumulative_seconds']; upper=int(now)-int(fields['created'])+1
    assert upper+opening<=21600 and upper+opening-3536<=5400
    (dest/'proof.json').write_text(json.dumps(dict(launch=launch,initial=initial,last_heartbeat=hb.splitlines()[-8:],closed_epoch=now,owner_gone=True,lease_absent=True,upper_stage_seconds=upper,upper_cumulative_seconds=opening+upper,anchor=ANCHOR_SHA,exe=j.digest(j.EXE)),indent=2)); (dest/'summary.json').write_text(json.dumps(result,indent=2)); (dest/'rows.jsonl.gz').write_bytes(gzip.compress((out/'rows.jsonl').read_bytes(),mtime=0)); (dest/'lifecycle.txt').write_text(hb+'\n'+(out/'console').read_text());
    if stage=='pilot': (dest/'panels.json').write_bytes((out.parent/'panels.json').read_bytes())

if __name__=='__main__': receipt(sys.argv[1])
