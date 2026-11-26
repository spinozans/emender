import torch
from torch.nn import functional as F

from .affine import compile_gdn2
from .joint_fixture import (
    FIELDS,
    HEAD,
    HITS,
    TITLES,
    WIDTH,
    Delta,
    JointLM,
    change_manual,
    episodes,
)


def test_changed_evidence_changes_answer_and_no_input_alias():
    b = episodes(torch.Generator().manual_seed(14), 41)
    prompt, pages = b.prompt.clone(), b.pages.clone()
    other = change_manual(pages)
    for i, ((who, field), rows) in enumerate(
        zip(prompt.tolist(), pages.tolist(), strict=True)
    ):
        assert TITLES <= field < TITLES + FIELDS
        assert len({r[0] for r in rows}) == HITS
        assert all(r[1::2] == [TITLES, TITLES + 1] for r in rows)
        for col in (2, 4):
            assert len({r[col] for r in rows}) == HITS
        (matching,) = [r for r in rows if r[0] == who]
        assert b.answer(pages)[i].item() == matching[2 + 2 * (field - TITLES)]
    assert (b.answer(other) != b.answer(pages)).all()
    assert torch.equal(pages[..., :2], other[..., :2])
    assert torch.equal(b.pages, pages) and torch.equal(b.prompt, prompt)


def test_batched_core_matches_independent_compiled_maps_on_shared_seed():
    torch.manual_seed(21)
    head = Delta().double()
    x = torch.randn(2, 4, 5, WIDTH, dtype=torch.double)
    seed = torch.randn(2, 1, HEAD, HEAD, dtype=torch.double, requires_grad=True)
    got = head(x, seed)
    k, d, e, w, v = head.controls(x).chunk(5, dim=-1)
    for i in range(2):
        for j in range(4):
            h = compile_gdn2(
                F.normalize(k[i, j], dim=-1),
                (d[i, j] + 2).sigmoid(),
                e[i, j].sigmoid(),
                w[i, j].sigmoid() * v[i, j],
            )
            torch.testing.assert_close(
                got[i, j], h.apply(seed[i, 0]), rtol=1e-12, atol=1e-12
            )


def test_mixing_not_page_order_and_empty_hint_not_a_page():
    torch.manual_seed(31)
    model = JointLM().double()
    b = episodes(torch.Generator().manual_seed(32), 5)
    original = b.pages.clone()
    reorder = [2, 0, 3, 1]
    for mode in ("state", "effect", "none"):
        logits, p = model(b.prompt, b.pages, mode)
        got, pp = model(b.prompt, b.pages[:, reorder], mode)
        torch.testing.assert_close(logits, got)
        torch.testing.assert_close(p[:, reorder], pp)
        changed, p2 = model(b.prompt, change_manual(b.pages), mode)
        assert torch.equal(p, p2)
        if mode == "none":
            assert torch.equal(logits, changed)
        else:
            assert (logits - changed).abs().max() > 1e-8
    # Substituting another transition cannot do anything when all pages are identical.
    same = b.pages[:, :1].expand(-1, HITS, -1)
    torch.testing.assert_close(
        model(b.prompt, same, "wrong_a")[0], model(b.prompt, same, "effect")[0]
    )
    assert torch.equal(original, b.pages)
