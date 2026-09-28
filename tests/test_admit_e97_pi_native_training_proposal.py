from scripts.admit_e97_pi_native_training_proposal import STATEMENT
from scripts.plan_e97_pi_native_training_schedule import identity, permutation
from types import SimpleNamespace


def test_statement():
    assert STATEMENT == 'I authorize the exact 32 update proposal.'


def test_pack_order_content_addressed_deterministic():
    # After the 2026-09-28 operator directive removing the proof-of-work
    # pack-nonce search, the pack order derives content-addressedly from the
    # admitted manifest hashes: the planner and the trainer both evaluate
    # permutation() over the same identity inputs, so the audited runtime
    # schedule matches the planner output by construction (no search needed).
    args = SimpleNamespace(authority_sha256='a' * 64, pack_sha256='b' * 64,
                           sampler_key=975424, world_size=8, context_size=65536)
    meta = identity(args)
    first = permutation(meta, 'epoch-permutation', 0, 438)
    again = permutation(identity(args), 'epoch-permutation', 0, 438)
    assert first == again, 'permutation must be a pure function of content identity'
    # different content -> different order (content-addressing binds order to bytes)
    other = SimpleNamespace(authority_sha256='c' * 64, pack_sha256='b' * 64,
                            sampler_key=975424, world_size=8, context_size=65536)
    assert permutation(identity(other), 'epoch-permutation', 0, 438) != first or \
        'c' * 64 == 'a' * 64, 'distinct identities should derive distinct orders'
