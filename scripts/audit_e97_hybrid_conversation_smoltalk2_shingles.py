"""SmolTalk2 shingle-exclusion audit for the hybrid-conversation collection.

Proves the collected hybrid-conversation records share no long token span with
the production-admitted SmolTalk2 corpus (e97-4b-smoltalk2-admitted-v1), so the
seam cohort adds new signal instead of duplicating conversation-rehearsal
training content. The span bar follows the repository's established
identity+shingle exclusion policy (commapile document-causal precedent:
5 normalized lines >= 160 chars): WIDTH consecutive tokens at ~4 chars/token.
Windows never cross record boundaries on either side; hash hits are verified
byte-exact before counting, so reported collisions are content, not hash
accidents. Fail closed on any verified collision.
"""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np

RECORD_INDEX=struct.Struct('<QQQB7x')
WIDTH=40                       # ~160-char span bar of the identity+shingle policy
MULT=np.uint64(0x9E3779B97F4A7C15)   # golden-ratio odd multiplier; uint64 wraparound
CHUNK=1<<25                    # 32M-token streaming windows over the large corpus
SCHEMA='emender-e97-hybrid-conversation-smoltalk2-shingle-audit-v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_authority(root):
 """Verify the four payload outputs against the authority manifest and return
 (manifest, starts, lengths, tokens memmap)."""
 root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
 paths={}
 for key in ('tokens','mask','index','metadata'):
  descriptor=manifest['outputs'][key];raw=Path(descriptor['path'])
  path=raw if raw.is_absolute() else root/raw
  if path.stat().st_size!=descriptor['bytes'] or sha(path)!=descriptor['sha256']:raise ValueError(f'authority output identity: {key}')
  paths[key]=path
 index=Path(paths['index']).read_bytes();metadata=Path(paths['metadata']).read_text().splitlines()
 if len(index)!=RECORD_INDEX.size*len(metadata):raise ValueError('authority shape')
 records=[RECORD_INDEX.unpack_from(index,i*RECORD_INDEX.size) for i in range(len(metadata))]
 starts=np.array([r[0] for r in records],dtype=np.uint64);lengths=np.array([r[1] for r in records],dtype=np.uint64)
 tokens=np.memmap(paths['tokens'],dtype='<u4',mode='r')
 if len(tokens)!=sum(lengths):raise ValueError('authority token count')
 return manifest,metadata,starts,lengths,tokens

def window_hashes(tokens,start,n):
 """uint64 polynomial hashes of every in-record WIDTH-token window."""
 if n<WIDTH:return np.zeros(0,dtype=np.uint64),np.zeros(0,dtype=np.uint64)
 count=n-WIDTH+1
 hashes=np.zeros(count,dtype=np.uint64)
 for k in range(WIDTH):hashes=hashes*MULT+np.asarray(tokens[start+k:start+k+count],dtype=np.uint64)
 offsets=np.arange(count,dtype=np.uint64)
 return hashes,offsets

def audit(args):
 root=Path(args.root)
 ours_manifest,ours_meta,ours_starts,ours_lengths,ours_tokens=load_authority(root/'candidate-authority')
 theirs_manifest,theirs_meta,theirs_starts,theirs_lengths,theirs_tokens=load_authority(args.smoltalk2)
 if ours_manifest.get('training_eligible') or theirs_manifest.get('training_eligible'):raise ValueError('training eligibility')
 # our windows: hash -> (record index, in-record offset), deduplicated
 our_hashes=[];our_records=[];our_offsets=[]
 for i,(start,n) in enumerate(zip(ours_starts,ours_lengths)):
  hashes,offsets=window_hashes(ours_tokens,int(start),int(n))
  our_hashes.append(hashes);our_records.append(np.full(len(hashes),i,dtype=np.uint32));our_offsets.append(offsets)
 our_hashes=np.concatenate(our_hashes) if our_hashes else np.zeros(0,dtype=np.uint64)
 our_records=np.concatenate(our_records) if our_records else np.zeros(0,dtype=np.uint32)
 our_offsets=np.concatenate(our_offsets) if our_offsets else np.zeros(0,dtype=np.uint64)
 our_windows_total=len(our_hashes)
 order=np.argsort(our_hashes);our_hashes=our_hashes[order];our_records=our_records[order];our_offsets=our_offsets[order]
 # In-collection duplicate windows are expected (whole-record sequence dedup is
 # the collection guarantee; shared templates grounded by the same runtime date
 # repeat spans across records). Deduplicate for membership; byte-exact
 # verification resolves any hash hit to real content.
 our_hashes,first=np.unique(our_hashes,return_index=True)
 our_records=our_records[first];our_offsets=our_offsets[first]
 ours_window_count=len(our_hashes)
 # their windows: stream chunks, mask boundary-crossing windows, membership test
 ends=np.cumsum(theirs_lengths,dtype=np.uint64)
 collisions=[];hash_only=0;theirs_window_count=0
 total=len(theirs_tokens)
 for c0 in range(0,total,CHUNK):
  c1=min(c0+CHUNK,total);span=c1-c0
  if span<WIDTH:break
  count=span-WIDTH+1
  hashes=np.zeros(count,dtype=np.uint64)
  for k in range(WIDTH):hashes=hashes*MULT+np.asarray(theirs_tokens[c0+k:c0+k+count],dtype=np.uint64)
  starts=np.arange(c0,c0+count,dtype=np.uint64)
  # a window is in-record iff no record end lies strictly inside (start, start+WIDTH)
  before=np.searchsorted(ends,starts,side='right')
  inside=np.searchsorted(ends,starts+WIDTH,side='left')
  valid=before==inside
  hashes=hashes[valid];starts=starts[valid]
  theirs_window_count+=len(hashes)
  if not len(hashes):continue
  where=np.searchsorted(our_hashes,hashes)
  where=np.clip(where,0,ours_window_count-1)
  hit=our_hashes[where]==hashes
  for h_index,t_start in zip(np.nonzero(hit)[0],starts[hit]):
   theirs_window=np.asarray(theirs_tokens[int(t_start):int(t_start)+WIDTH],dtype=np.uint64)
   record=int(our_records[where[h_index]]);offset=int(our_offsets[where[h_index]])
   ours_start=int(ours_starts[record])+offset
   ours_window=np.asarray(ours_tokens[ours_start:ours_start+WIDTH],dtype=np.uint64)
   if np.array_equal(theirs_window,ours_window):
    collisions.append({'our_record':json.loads(ours_meta[record])['id'],
      'their_record':int(np.searchsorted(ends,np.uint64(t_start),side='right')),
      'tokens':WIDTH})
   else:hash_only+=1
 if collisions:raise ValueError(f'SmolTalk2 shingle collisions: {len(collisions)}')
 receipt={'schema':SCHEMA,'status':'pass','shingle_policy':'width-40-token in-record windows (~160-char span bar of the commapile identity+shingle policy); hash hits verified byte-exact; windows never cross record boundaries',
  'shingle_width_tokens':WIDTH,'our_records':len(ours_meta),'our_windows':ours_window_count,'our_windows_all':our_windows_total,
  'their_records':len(theirs_meta),'their_windows':theirs_window_count,'their_tokens':total,
  'content_collisions':0,'hash_only_collisions':hash_only,
  'our_authority_sha256':sha(root/'candidate-authority/manifest.json'),
  'their_manifest_sha256':sha(Path(args.smoltalk2)/'manifest.json'),
  'their_dataset_id':theirs_manifest.get('dataset_id'),'their_dataset_revision':theirs_manifest.get('dataset_revision'),
  'checker_sha256':sha(__file__),'training_eligible':False,'packing_authorized':False,'optimizer_updates_authorized':0}
 Path(args.output).write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print('SMOLTALK2_SHINGLE_EXCLUSION_PASS',len(ours_meta),len(theirs_meta),sha(args.output))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoltalk2',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 audit(p.parse_args())
if __name__=='__main__':main()
