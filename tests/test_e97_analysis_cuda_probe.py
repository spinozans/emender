from dataclasses import replace

import pytest
import torch

from ndm.e97 import E97RecurrentCache, _extend_token_lineage
from scripts.probe_e97_analysis_cuda_replay import compare, verify_delta_lineage


def cache(tokens, digest):
    return E97RecurrentCache(token_ids=tuple(tokens), hidden=[torch.zeros(2)],
                             next_logits=torch.zeros(3), checkpoint='fixture',
                             token_count=len(tokens), token_lineage_sha256=digest)


def test_delta_lineage_depends_on_boundaries_but_token_state_comparison_does_not():
    prefix = _extend_token_lineage(None, [1, 2])
    chunked = cache([1, 2, 3], _extend_token_lineage(prefix, [3]))
    replay = cache([1, 2, 3], _extend_token_lineage(None, [1, 2, 3]))
    assert chunked.token_lineage_sha256 != replay.token_lineage_sha256
    verify_delta_lineage(chunked, prefix, [3])
    verify_delta_lineage(replay, None, [1, 2, 3])
    assert compare(chunked, replay) == 1
    with pytest.raises(AssertionError, match='delta lineage'):
        verify_delta_lineage(chunked, None, [1, 2, 3])


def test_probe_still_rejects_tensor_history_count_and_checkpoint_changes():
    original = cache([1, 2], _extend_token_lineage(None, [1, 2]))
    changes = [
        replace(original, hidden=[torch.ones(2)]),
        replace(original, next_logits=torch.ones(3)),
        replace(original, next_logits=torch.full((3,), float('nan'))),
        replace(original, token_ids=(1, 3)),
        replace(original, token_count=3),
        replace(original, checkpoint='different'),
    ]
    for changed in changes:
        with pytest.raises(AssertionError):
            compare(original, changed)
