from types import SimpleNamespace
from scripts.plan_e97_pi_native_training_schedule import identity,pack_id,sample_id

def test_descriptor_planner_matches_frozen_representation_bridge_sample_identity():
 args=SimpleNamespace(authority_sha256='a21dba6f80e58e87ea838a98018b40650decebed3cf4b0f98dacdb750520469d',pack_sha256='70ee78073715cf4d87b03579323de0efa0e0c71e676ca693c4034833f3529dc2',sampler_key=974223,world_size=8,context_size=65536)
 meta=identity(args)
 assert sample_id(meta,'epoch-permutation',0,0)=='03b78c5c33531ae207313782e0d5471d1e1601d9750565404a6ff61710a413bb'
 assert len({pack_id(meta,'epoch-permutation',rank,cursor,100) for cursor in range(8) for rank in range(8)})==64
