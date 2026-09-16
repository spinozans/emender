#!/usr/bin/env python3
"""Create a separate training admission after explicit operator authorization."""
import argparse,hashlib,json,math,shutil
from pathlib import Path
PROPOSAL_SHA='96db2e1cee91304b6c295113e517c8c1fd032c40a7fb372d9866dea8ada057be'
STATEMENT='I authorize the exact 32 update proposal.'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
def affine(authority_sha,pack_sha,key,world,context,epoch,modulus):
 meta={'authority_manifest_sha256':authority_sha,'pack_manifest_sha256':pack_sha,'sampler_key':key,'data_world_size':world,'context_size':context,'split':'train','schema':'emender-record-pack-counter-v1','sampler_mode':'epoch-permutation','epoch':epoch};digest=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode('ascii')).digest();a=int.from_bytes(digest[:8],'little')%modulus
 while math.gcd(a,modulus)!=1:a=(a+1)%modulus
 return a,int.from_bytes(digest[8:16],'little')%modulus
def admit(args):
 proposal=json.loads(args.proposal.read_text());source=json.loads((args.preparation/'manifest.json').read_text());source_packs=json.loads((args.preparation/'packs/manifest.json').read_text())
 if sha(args.proposal)!=PROPOSAL_SHA or args.authorization_statement!=STATEMENT or proposal['status']!='frozen-proposal-not-authorized' or proposal['proposed_updates']!=32 or proposal['sampler_key']!=975424:raise ValueError('authorization scope')
 if sha(args.preparation/'manifest.json')!=proposal['authority_manifest_sha256'] or sha(args.preparation/'packs/manifest.json')!=proposal['pack_manifest_sha256'] or source['training_eligible'] or source['optimizer_updates_authorized'] or source_packs['training_eligible']:raise ValueError('preparation identity')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False);(args.output/'packs').mkdir(mode=0o700)
 for descriptor in source['outputs'].values():
  src=args.preparation/Path(descriptor['path']).name;dst=args.output/src.name;shutil.copyfile(src,dst)
  if dst.stat().st_size!=descriptor['bytes'] or sha(dst)!=descriptor['sha256']:raise ValueError('authority payload copy')
 authority=dict(source);authority.update(purpose='Operator-authorized exact Pi-native 32-update SFT admission',training_eligible=True,packing_authorized=True,optimizer_updates_authorized=32,operator_internal_training_authorized=True,admission_proposal_sha256=PROPOSAL_SHA,authorization_statement=STATEMENT,automatic_retry=False,automatic_expansion=False,checkpoint_promotion=False,new_rl_updates=0)
 (args.output/'manifest.json').write_bytes(dump(authority));authority_sha=sha(args.output/'manifest.json')
 for descriptor in source_packs['outputs'].values():
  src=args.preparation/'packs'/Path(descriptor['path']).name;dst=args.output/'packs'/src.name;shutil.copyfile(src,dst)
  if dst.stat().st_size!=descriptor['bytes'] or sha(dst)!=descriptor['sha256']:raise ValueError('pack payload copy')
 packs=dict(source_packs);packs.update(authority_manifest_sha256=authority_sha,training_eligible=True,diagnostic_system_gate=None,packing_authorized=True,optimizer_updates_authorized=32,operator_internal_training_authorized=True,admission_proposal_sha256=PROPOSAL_SHA,authorization_statement=STATEMENT)
 modulus=source_packs['splits']['train']['packs'];target=affine(proposal['authority_manifest_sha256'],proposal['pack_manifest_sha256'],proposal['sampler_key'],proposal['data_world_size'],proposal['context_size'],0,modulus);found=None
 for nonce in range(args.max_nonce):
  packs['admission_nonce']=nonce;payload=dump(packs);pack_sha=hashlib.sha256(payload).hexdigest()
  if affine(authority_sha,pack_sha,proposal['sampler_key'],proposal['data_world_size'],proposal['context_size'],0,modulus)==target:found=(nonce,payload,pack_sha);break
 if found is None:raise RuntimeError('no identity-preserving admitted pack nonce found')
 nonce,payload,pack_sha=found;(args.output/'packs/manifest.json').write_bytes(payload)
 receipt={'schema':'emender-e97-pi-native-training-admission-v1','status':'authorized-exact-proposal','proposal_sha256':PROPOSAL_SHA,'authorization_statement':STATEMENT,'authority_manifest_sha256':authority_sha,'pack_manifest_sha256':pack_sha,'sampler_key':proposal['sampler_key'],'epoch_zero_affine':{'multiplier':target[0],'offset':target[1],'modulus':modulus},'admission_nonce':nonce,'exact_first_epoch_pack_sequence_preserved':True,'authorized_updates':32,'learning_rate':proposal['proposed_learning_rate'],'automatic_retry':False,'automatic_expansion':False,'checkpoint_promotion':False,'new_rl_updates':0}
 (args.output/'admission.json').write_bytes(dump(receipt));print('PI_NATIVE_TRAINING_ADMITTED',authority_sha,pack_sha,nonce,sha(args.output/'admission.json'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--proposal',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--authorization-statement',required=True);p.add_argument('--max-nonce',type=int,default=1000000);admit(p.parse_args())
if __name__=='__main__':main()
