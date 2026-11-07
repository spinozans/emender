from pathlib import Path
import hashlib,json,gzip,os,math
root=Path('experiments/expressivity_tasks/x43_fork_resolved'); out=root/'receipts/flagship_stream';out.mkdir(exist_ok=True); assert not (out/'registration.json').exists(), 'frozen registration exists; refuse overwrite'
g=json.loads((root/'receipts/spine_decision/geometry.json').read_text())['prospective_beyond_sample']
source=Path('/mnt/nvme2n1/erikg/pile.txt'); st=source.stat(); assert {k:getattr(st,'st_ino' if k=='inode' else 'st_'+k) for k in ('size','inode','mtime_ns')}==g['source_stat']
h=hashlib.sha256(); stats={k:dict(documents=0,bytes=0,targets=0,windows=0) for k in ('train','calibration','final_test')}
manifest=out/'documents.jsonl.gz'
with source.open('rb') as f,manifest.open('wb') as dest,gzip.GzipFile(fileobj=dest,mode='wb',mtime=0) as z:
 f.seek(g['first_full_document']); pos=f.tell(); pending=b''; end=g['full_documents_end_exclusive']
 while f.tell()<end:
  data=pending+f.read(min(8<<20,end-f.tell()));parts=data.split(b'\x1e');pending=parts.pop()
  for raw in parts:
   n=len(raw);sha=hashlib.sha256(raw).hexdigest();bucket=int(sha,16)%100;split='calibration' if bucket==0 else 'final_test' if bucket==1 else 'train'
   if n>1:
    h.update(f'{pos}|{n}|{sha}|{split}\n'.encode()); s=stats[split];s['documents']+=1;s['bytes']+=n;s['targets']+=n-1;s['windows']+=math.ceil((n-1)/2048)
    z.write((json.dumps(dict(document=pos,byte_offset=pos,byte_length=n,sha256=sha,split=split),sort_keys=True)+'\n').encode())
   pos+=n+1
 assert not pending and pos==end
assert h.hexdigest()==g['document_identity_stream_sha256']
assert source.stat()==st
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((root/'receipts/p100_query_verification/registration.json').read_text()); verdict=json.loads((root/'receipts/p100_query_verification/verdict.json').read_text()); assert verdict['status']=='passed' and len(verdict['checks'])==17
r=dict(status='registered_before_extraction',source=reg['raw'],index=reg['index'],binary=reg['binary'],binary_sha256=reg['binary_sha256'],query_verdict_sha256=sha(root/'receipts/p100_query_verification/verdict.json'),raw_index_region_sha256=verdict['raw_region_sha256'],manifest=str(manifest),manifest_sha256=sha(manifest),identity_stream_sha256=h.hexdigest(),splits=stats,context=2048,consultation_cap=1024,features=260,order='source byte offset ascending within each split; all six arms consume the same training prefix, no wrap; stop before exhaustion pending amendment',split_rule=g['split_rule'],window_rule='start=0,2048,... < length-1; targets=min(2048,length-1-start); reset state each window; no padding targets; causal context raw[max(start,target-1024):target]',exposure_precedent=dict(path='experiments/expressivity_tasks/x40_emender_lm/REPORT_PART8.md',sha256=sha('experiments/expressivity_tasks/x40_emender_lm/REPORT_PART8.md'),scope='historical token throughput planning only, not byte throughput guarantee',T2048_B64_rates=dict(transformer=6233.1,gdn2=6363.6,pure=1823.2),seconds=86400,max_projected_targets=6363.6*86400,training_margin_factor=stats['train']['targets']/(6363.6*86400)),extraction=dict(positions=sum(x['targets'] for x in stats.values()),cache_bytes_at_113=sum(x['targets'] for x in stats.values())*113,floor_bytes=150000000000,baseline_rates=[5884.18,4814.02],projected_host_days=[sum(x['targets'] for x in stats.values())/v/86400 for v in (5884.18,4814.02)],not_measured_cpu_days=True,stop_line='exact registered stream only; disk floor failure; no automatic extension'),limitations=['Training still gated on nominations, qualification and approval.','Primary T2048 caches only; alternative-reset length sweeps need distinct extraction.','Historical throughput is a sizing prior; stop before stream exhaustion if scale throughput exceeds margin.','Span disjointness does not prove substring uniqueness.'])
from .c14_feature_extract import digest
r['index']['sha256']=digest(Path(r['index']['path']))
r['extraction']['storage_reservation_bytes']=200000000000
r['extraction']['logical_core_day_planning_ceiling']=r['extraction']['projected_host_days'][-1]*256
(out/'registration.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['splits']));print(r['extraction'])
