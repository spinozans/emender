import numpy as np
import pytest
import torch
from torch.nn import functional as F

from .cuda_step import invented
from .natural_data import encode, tokenizer, windows
from .natural_model import Shape, TokenLM
from .test_triton_read import device as _device_fixture  # noqa: F401 -- pytest discovers named fixture
from .token_postings import fetch, index, literal, locate
from .triton_loss import token_nll


def test_byte_roundtrip_and_all_doc_local_shifts():
    enc = tokenizer()
    for raw in (
        b"\x80\xfe\xffabc",
        "technical text \u03a3\n".encode() * 40,
        b"z" * 4097,
        b"a\xf0\x9f",
    ):
        ids = encode(enc, raw)
        seq = list(windows(ids, 7))
        actual = np.concatenate([w[1 : n + 1] for w, n in seq])
        np.testing.assert_array_equal(actual, ids[1:])
        assert all(n <= 7 for _, n in seq)


def test_postings_against_occurrence_oracle_and_actual_fetch(device):
    docs = [[i, i % 17, i % 7, 0, 3] for i in range(283)] + [
        [4, 4, 4, 4, 0, 6],
        [0, 1, 0, 1, 0],
        [4, 4, 4, 4, 0, 6],
    ]
    image = index(docs, 5)
    im = image.tensors(device)
    qs = [[4, 4, 4, 4], [0, 1, 0, 1], [810, 811, 811, 810], [22, 5, 1, 0], [4, 4, 4, 4]]
    chosen, votes = locate(
        im,
        torch.tensor(qs, device=device, dtype=torch.int32),
        torch.tensor([1, 1, 1, 1, 0], device=device, dtype=torch.int32),
    )
    pages, lengths = fetch(im, chosen)
    for row, prompt in enumerate(qs):
        ids, mass = literal(docs, prompt, 5) if row < 4 else ([-1] * 4, [0] * 4)
        assert chosen[row].tolist() == ids and votes[row].tolist() == mass
        for j, h in enumerate(ids):
            if h < 0:
                assert lengths[row, j] == -1 and not pages[row, j].count_nonzero()
            else:
                assert (
                    lengths[row, j].item() == image.lens[h]
                    and pages[row, j].tolist() == image.pages[h].tolist()
                )


def test_full_vocab_loss_masks_padding_and_vjp(device):
    torch.manual_seed(31)
    logits = torch.randn(3, 4, 128, device=device, requires_grad=True)
    y = torch.tensor(
        [[3, 57, -100, 99], [0, -100, 1, 5], [39, 42, 57, -100]], device=device
    )
    up = torch.randn_like(y, dtype=torch.float32)
    got = token_nll(logits, y, 111)
    want = F.cross_entropy(
        logits[..., :111].reshape(-1, 111), y.reshape(-1), reduction="none"
    ).view_as(got)
    torch.testing.assert_close(got, want)
    a = torch.autograd.grad(got, logits, up)[0]
    b = torch.autograd.grad(want, logits, up)[0]
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    "host,arm",
    [
        ("gdn2", "g-effect"),
        ("e97", "mix-effect"),
        ("e97", "e-neutral"),
        ("gdn2", "local"),
    ],
)
def test_causality_absent_hits_and_update(device, host, arm):
    sh = Shape(vocab=48, width=12, heads=2, key=4, time=7, prefix=3, hits=2, page=3)
    torch.manual_seed(714)
    m = TokenLM(sh, host, arm).to(device)
    w, n, pages, ns, v = invented(sh, 3, device=device)
    n = n.clone()
    n[1] = 2
    ns[1] = -1
    v[1] = 0
    changed = w.clone()
    changed[:, sh.prefix :] = (changed[:, sh.prefix :] + 1) % sh.vocab
    got = m(w[:, :-1], n, pages, ns, v)
    other = m(changed[:, :-1], n, pages, ns, v)
    torch.testing.assert_close(
        got[:, : sh.prefix], other[:, : sh.prefix], atol=1e-6, rtol=1e-6
    )
    new = m(w[:, :-1], n, (pages + 3) % sh.vocab, ns, v)
    torch.testing.assert_close(got[:, : sh.prefix - 1], new[:, : sh.prefix - 1])
    torch.testing.assert_close(got[1], new[1])
    losses = m.losses(w, n, pages, ns, v)
    assert not losses[1, 2:].count_nonzero()
    losses.mean().backward()
    assert (
        m.lower.controls.weight.grad.norm() > 0 and m.upper.query.weight.grad.norm() > 0
    )
    if arm != "local":
        assert m.control.weight.grad.norm() > 0 and m.address.weight.grad.norm() > 0
    from .natural_oracle import dense_reference

    orig = {
        name: x.grad.clone() for name, x in m.named_parameters() if x.grad is not None
    }
    m.zero_grad(set_to_none=True)
    with dense_reference():
        ref = m.losses(w, n, pages, ns, v)
        ref.mean().backward()
    torch.testing.assert_close(losses, ref, atol=3e-5, rtol=3e-4)
    for name, x in m.named_parameters():
        if name in orig:
            torch.testing.assert_close(x.grad, orig[name], atol=3e-5, rtol=3e-4)
