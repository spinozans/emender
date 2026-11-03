import json,hashlib,time,datetime,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[3]; out=root/'experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_emender_closed'; live=Path('/home/erikg/c14_consultation_features')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def snapshot():
 cg=Path('/sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service/app.slice/emender-velocity-c14-union.service')
 return dict(time=time.time(),memory={s:(cg/s).read_text() for s in ['memory.current','memory.peak','memory.stat','memory.events']},meminfo=Path('/proc/meminfo').read_text(),processes=subprocess.check_output(['ps','-eo','pid,ppid,rss,comm','--sort=-rss'],text=True).splitlines()[:20],service=subprocess.check_output(['systemctl','--user','show','emender-velocity-c14-union.service','-p','MainPID','-p','NRestarts','-p','ActiveState','-p','StartLimitBurst','-p','StartLimitIntervalUSec'],text=True))
first=snapshot(); (out/'memory_before.json').write_text(json.dumps(first,indent=2)+'\n')
j=live/'registered_emender/journal.jsonl'; records=rows(j); shards=[r for r in records if r['event']=='shard_complete']; boundary=[r for r in rows(live/'union_service.jsonl') if r['event']=='phase_boundary' and r['phase']=='emender'][-1]
assert sha(j)==boundary['journal_sha256']; windows=positions=size=0; ident=records[0]['identity']
for r in shards:
 assert r['first_window_index']==windows and r['identity']==ident
 p=j.parent/r['file']; assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
 assert r['full_compact_row_parity'] is True and len(r['slow_samples'])==3
 windows+=r['windows'];positions+=r['positions'];size+=r['bytes']
assert {r['file'] for r in shards}=={p.name for p in j.parent.glob('*.sparse')}
auditpath=root/'experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_registered_scale/stream_audit.json'; audit=json.loads(auditpath.read_text()); assert sha(auditpath)==ident['stream_audit_sha256']
assert (windows,positions)==(audit['arms'][0]['windows'],audit['arms'][0]['targets'])
assert size==boundary['cache_bytes']
for p,h in audit['receipt_hashes'].items(): assert sha(root/p)==h
assert records[-1]['event']=='phase_complete' and records[-1]['status']=='complete'
res=dict(status='passed',scope='closed Emender cache hashes, contiguity, recorded parity attestations and frozen dense receipt hashes; not native parity rerun or training acceptance',shards=len(shards),windows=windows,last_window_index=windows-1,positions=positions,cache_bytes=size,journal_sha256=sha(j),boundary=boundary,identity=ident,parity_attestations_green=True,request_cardinality_matches=(len(shards)==13699 and windows==1752576),requested_cardinality=dict(shards=13699,windows=1752576),training_status='blocked_scope_conflict',training_reason=audit['reason'],supervisor_tool='absent from enabled tools; no decision can be requested',eval_dependency='Complete frozen paired eval cache required before evaluation; no evaluation performed')
(out/'cache_validation.json').write_text(json.dumps(res,indent=2)+'\n')
evalrows=rows(live/'registered_eval/journal.jsonl'); done=[r for r in evalrows if r['event']=='shard_complete']; start=[r for r in rows(live/'union_service.jsonl') if r['event']=='phase_start' and r['phase']=='eval'][-1]['time']; now=time.time(); targets=sum(r['positions'] for r in done);rate=targets/(now-start); projected=now+(audit['eval']['targets']-targets)/rate
(out/'eval_projection.json').write_text(json.dumps(dict(observed_at=now,phase_start=start,committed_windows=sum(r['windows'] for r in done),committed_targets=targets,expected_targets=audit['eval']['targets'],observed_targets_per_second=rate,projected_boundary_utc=datetime.datetime.fromtimestamp(projected,datetime.timezone.utc).isoformat(),baseline_boundary_utc=datetime.datetime.fromtimestamp(start+audit['eval']['targets']/5884.178857234429,datetime.timezone.utc).isoformat(),policy='scenario, not guarantee; snapshot of committed journal rows; no eval cache changes',frozen_windows_sha256=audit['eval']['windows_sha256']),indent=2)+'\n')
(out/'memory_after.json').write_text(json.dumps(snapshot(),indent=2)+'\n');print(json.dumps(res))
