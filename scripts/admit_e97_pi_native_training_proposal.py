#!/usr/bin/env python3
"""Create a separate training admission after explicit operator authorization.

2026-09-28 operator directive: the proof-of-work pack-nonce search is REMOVED.
The admitted schedule derives its pack order from the admitted content hashes
(content-addressed order; the planner and the trainer both derive from the same
admitted manifest hashes, so the audited runtime schedule matches the planner
output exactly by construction). No attacker model requires costly
unforgeability here: the operator is the trust root, the proposal is committed
to the pushed git remote, and every identity check is deterministic and
re-runnable (tampering is visible, not merely expensive)."""
import argparse,hashlib,json,shutil
from pathlib import Path
STATEMENT='I authorize the exact 32 update proposal.'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(value):return (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
def admit(args):
 proposal=json.loads(args.proposal.read_text());source=json.loads((args.preparation/'manifest.json').read_text());source_packs=json.loads((args.preparation/'packs/manifest.json').read_text())
 updates=int(proposal['proposed_updates'])
 if updates<=0 or updates%32:raise ValueError('updates must be a positive multiple of 32 (segment granularity)')
 statement=f'I authorize the exact {updates} update proposal.'
 if sha(args.proposal)!=args.proposal_sha256 or args.authorization_statement!=statement or proposal['status']!='frozen-proposal-not-authorized':raise ValueError('authorization scope')
 if sha(args.preparation/'manifest.json')!=proposal['authority_manifest_sha256'] or sha(args.preparation/'packs/manifest.json')!=proposal['pack_manifest_sha256'] or source['training_eligible'] or source['optimizer_updates_authorized'] or source_packs['training_eligible']:raise ValueError('preparation identity')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False);(args.output/'packs').mkdir(mode=0o700)
 for descriptor in source['outputs'].values():
  src=args.preparation/Path(descriptor['path']).name;dst=args.output/src.name;shutil.copyfile(src,dst)
  if dst.stat().st_size!=descriptor['bytes'] or sha(dst)!=descriptor['sha256']:raise ValueError('authority payload copy')
 authority=dict(source);authority.update(purpose=f'Operator-authorized exact Pi-native {updates}-update SFT admission',training_eligible=True,packing_authorized=True,optimizer_updates_authorized=updates,operator_internal_training_authorized=True,admission_proposal_sha256=args.proposal_sha256,authorization_statement=statement,automatic_retry=False,automatic_expansion=False,checkpoint_promotion=False,new_rl_updates=0)
 (args.output/'manifest.json').write_bytes(dump(authority));authority_sha=sha(args.output/'manifest.json')
 for descriptor in source_packs['outputs'].values():
  src=args.preparation/'packs'/Path(descriptor['path']).name;dst=args.output/'packs'/src.name;shutil.copyfile(src,dst)
  if dst.stat().st_size!=descriptor['bytes'] or sha(dst)!=descriptor['sha256']:raise ValueError('pack payload copy')
 packs=dict(source_packs);packs.update(authority_manifest_sha256=authority_sha,training_eligible=True,diagnostic_system_gate=None,packing_authorized=True,optimizer_updates_authorized=updates,operator_internal_training_authorized=True,admission_proposal_sha256=args.proposal_sha256,authorization_statement=statement)
 payload=dump(packs);pack_sha=hashlib.sha256(payload).hexdigest()
 (args.output/'packs/manifest.json').write_bytes(payload)
 receipt={'schema':'emender-e97-pi-native-training-admission-v1','status':'authorized-exact-proposal','proposal_sha256':args.proposal_sha256,'authorization_statement':statement,'authority_manifest_sha256':authority_sha,'pack_manifest_sha256':pack_sha,'sampler_key':proposal['sampler_key'],'pack_order':'content-addressed (admitted manifest hashes; planner and trainer derive identically)','authorized_updates':updates,'learning_rate':proposal['proposed_learning_rate'],'automatic_retry':False,'automatic_expansion':False,'checkpoint_promotion':False,'new_rl_updates':0}
 (args.output/'admission.json').write_bytes(dump(receipt));print('PI_NATIVE_TRAINING_ADMITTED',authority_sha,pack_sha,sha(args.output/'admission.json'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--proposal',type=Path,required=True);p.add_argument('--proposal-sha256',required=True);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--authorization-statement',required=True);admit(p.parse_args())
if __name__=='__main__':main()
