"""Source-named W3 preflight: independent masks and typed checkpoint points."""
import hashlib
from importlib.metadata import version
import json
import struct

import pytest
import torch

from ndm.e97 import build_e97_model, load_e97_checkpoint
from scripts.w3_matcheddose_preflight import independent_pack


def test_independent_pack_masks_boundaries_padding_and_recounts():
    records = struct.pack('<QQQB7x', 0, 3, 3, 0) + struct.pack('<QQQB7x', 3, 2, 2, 0)
    result = independent_pack(0, struct.pack('<QQQQ', 0, 2, 5, 3),
                              struct.pack('<II', 0, 1), records, bytes([1] * 5),
                              struct.pack('<5I', 11, 12, 13, 14, 15), 8)
    assert result['inputs'] == 5 and result['targets'] == 3
    assert result['loss_mask_sha256'] == hashlib.sha256(bytes([1, 1, 0, 1, 0, 0, 0, 0])).hexdigest()
    assert result['reset_mask_sha256'] == hashlib.sha256(bytes([1, 0, 0, 1, 0, 0, 0, 0, 0])).hexdigest()
    with pytest.raises(ValueError, match='count differs'):
        independent_pack(0, struct.pack('<QQQQ', 0, 2, 5, 5), struct.pack('<II', 0, 1),
                         records, bytes([1] * 5), struct.pack('<5I', 11, 12, 13, 14, 15), 8)


def tiny_checkpoint(tmp_path, kind):
    config = dict(level='E97', dim=32, depth=1, n_heads=2, n_state=8,
                  optimizer='schedulefree', use_triton=False, mlp_ratio=1, mlp_multiple=16)
    (tmp_path / 'args.json').write_text(json.dumps(config))
    model = build_e97_model(config, vocab_size=64, use_triton=False).bfloat16()
    if kind == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.01)
        cls, ver, point = 'torch.optim.AdamW', torch.__version__, 'live-nonaveraged'
    else:
        from schedulefree import AdamWScheduleFree
        opt = AdamWScheduleFree(model.parameters(), lr=1e-5, betas=(.9, .95), warmup_steps=0)
        opt.train()
        cls, ver, point = 'schedulefree.AdamWScheduleFree', version('schedulefree'), 'saved-eval-x'
    torch.manual_seed(43)
    for _ in range(3):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
    if kind != 'adamw':
        opt.eval()
    payload = dict(model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
                   weight_mode=point, optimizer_identity=dict(schema='emender-e97-optimizer-identity-v1',
                   **{'class': cls}, version=ver, weight_point=point))
    return payload


def point_hash(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def test_adamw_train_saved_point_hash_equality_under_unchanged_sf_args(tmp_path):
    payload = tiny_checkpoint(tmp_path, 'adamw')
    path = tmp_path / 'adamw.pt'
    torch.save(payload, path)
    loaded = [load_e97_checkpoint(path, weight_mode=mode, dtype=torch.bfloat16) for mode in ('train', 'saved')]
    assert point_hash(loaded[0].model) == point_hash(loaded[1].model)
    assert all(not x.schedulefree_train_weight_swap for x in loaded)


@pytest.mark.parametrize('kind', ['adamw', 'sf'])
@pytest.mark.parametrize('mode', ['train', 'saved'])
def test_wrong_optimizer_identity_fails_closed(tmp_path, kind, mode):
    payload = tiny_checkpoint(tmp_path, kind)
    payload['optimizer_identity']['class'] = ('schedulefree.AdamWScheduleFree' if kind == 'adamw' else 'torch.optim.AdamW')
    path = tmp_path / 'wrong.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='identity/state mismatch'):
        load_e97_checkpoint(path, weight_mode=mode)


@pytest.mark.parametrize('mode', ['train', 'saved'])
def test_unmarked_adamw_cannot_use_sf_path(tmp_path, mode):
    payload = tiny_checkpoint(tmp_path, 'adamw')
    del payload['optimizer_identity']
    path = tmp_path / 'unmarked.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='explicit optimizer identity'):
        load_e97_checkpoint(path, weight_mode=mode)


@pytest.mark.parametrize('mode', ['train', 'saved'])
def test_sf_typed_and_legacy_reads_are_byte_identical(tmp_path, mode):
    payload = tiny_checkpoint(tmp_path, 'sf')
    typed, legacy = tmp_path / 'typed.pt', tmp_path / 'legacy.pt'
    torch.save(payload, typed)
    del payload['optimizer_identity']
    torch.save(payload, legacy)
    a = load_e97_checkpoint(typed, weight_mode=mode, dtype=torch.bfloat16)
    b = load_e97_checkpoint(legacy, weight_mode=mode, dtype=torch.bfloat16)
    assert point_hash(a.model) == point_hash(b.model)
    assert a.schedulefree_train_weight_swap == b.schedulefree_train_weight_swap
