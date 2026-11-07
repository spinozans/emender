import hashlib,json,os
from pathlib import Path
root=Path('experiments/expressivity_tasks/x43_fork_resolved')
reg=json.loads(Path('experiments/expressivity_tasks/x42_pile_scale/byte_velocity_registration.json').read_text())
p=Path(reg['corpus']['offsets']) if 'corpus' in reg else Path('/home/erikg/c14_dense_bytes/pile_offsets.jsonl')
h=hashlib.sha256(); count=0;lo=None;hi=0;beyond=0
with p.open('rb') as f:
 for line in f:
  h.update(line);d=json.loads(line);a=d['byte_offset'];z=a+d['byte_length'];lo=a if lo is None else min(lo,a);hi=max(hi,z);count+=1;beyond+=a>=100_000_000_000
assert h.hexdigest()=='bb3a2b506ac4d7c1b2ae36e274331ee5ff1855e618d15df84780c5c42987bbe8'
fd=os.open('/mnt/nvme2n1/erikg/pile.txt',os.O_RDONLY);st=os.fstat(fd)
start=100_000_000_000;stop=start+1_500_000_000;pos=start;pending=b'';docstart=None;stats={k:{'documents':0,'bytes':0,'T2048_targets':0} for k in ['train','calibration','final_test']};mh=hashlib.sha256();first=None;last=None
while pos<stop:
 chunk=os.pread(fd,min(8<<20,stop-pos),pos);assert chunk
 if docstart is None:
  at=chunk.find(b'\x1e');assert at>=0;chunk=chunk[at+1:];docstart=pos+at+1;first=docstart
 data=pending+chunk;parts=data.split(b'\x1e');pending=parts.pop()
 for raw in parts:
  n=len(raw);sha=hashlib.sha256(raw).hexdigest();bucket=int(sha,16)%100;split='calibration' if bucket==0 else 'final_test' if bucket==1 else 'train'
  if n>1:
   s=stats[split];s['documents']+=1;s['bytes']+=n;s['T2048_targets']+=n-1
   mh.update(f'{docstart}|{n}|{sha}|{split}\n'.encode())
  docstart+=n+1;last=docstart
 pos+=min(8<<20,stop-pos)
assert os.fstat(fd)==st;os.close(fd)
out={'status':'passed_geometry_feasibility_not_launch_manifest','search_offsets':{'path':str(p),'sha256':h.hexdigest(),'documents':count,'min_start':lo,'max_end_exclusive':hi,'documents_fully_beyond_100GB':beyond},'prospective_beyond_sample':{'source_stat':{'size':st.st_size,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns},'read_start':start,'read_stop_exclusive':stop,'first_full_document':first,'full_documents_end_exclusive':last,'split_rule':'sha256 integer mod 100: 0 calibration, 1 final_test, otherwise train; identical contents always same split','document_identity_stream_sha256':mh.hexdigest(),'splits':stats,'trailing_partial_bytes_excluded':len(pending)},'limitations':['Existing hash-bound C14 offsets contain ZERO beyond-boundary documents. This fresh by-offset hashed sample establishes density and split feasibility, not a frozen flagship manifest.','No raw text materialized. No training or extraction. Full-stream selected-document identities and split manifests must be retained before launch.','T2048_targets counts n-1 per document; exact windows and masked tails remain launch gates.']}
(root/'receipts/spine_decision/geometry.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
