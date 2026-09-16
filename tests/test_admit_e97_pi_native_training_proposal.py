from scripts.admit_e97_pi_native_training_proposal import STATEMENT,affine
from scripts.plan_e97_pi_native_training_schedule import identity,permutation
from types import SimpleNamespace

def test_admission_affine_matches_runtime_planner_contract():
 args=SimpleNamespace(authority_sha256='a'*64,pack_sha256='b'*64,sampler_key=975424,world_size=8,context_size=65536)
 meta=identity(args)
 assert affine(args.authority_sha256,args.pack_sha256,args.sampler_key,8,65536,0,438)==permutation(meta,'epoch-permutation',0,438)
 assert STATEMENT=='I authorize the exact 32 update proposal.'
