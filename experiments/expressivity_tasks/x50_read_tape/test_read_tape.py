"""Invented documents only. Never requires a mount or opens any reserved split."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from .kernel import DEPTH, ROW, Kernel
from .program import ARMS, Tape, choose, execute, policy_loss, predict, score

DOCS = (b"ABAX", b"ABAY", b"ACAZ", b"AC!X", b"\xff\0\x1e\xff", b"AB", b"YAYAYAYAYA")


class Fixture:
    """Small full branch directory. Return true counts even for pruned children."""

    def __init__(self, docs=DOCS, floor=1):
        keys = {}
        for doc in docs:
            for j in range(len(doc) + 1):
                for d in range(min(DEPTH, j) + 1):
                    key = doc[j - d : j]
                    h = keys.setdefault(key, np.zeros(257, np.uint64))
                    h[doc[j] if j < len(doc) else 256] += 1
        self.ids = {
            k: i
            for i, k in enumerate(
                sorted(k for k, h in keys.items() if sum(h) >= floor), 1
            )
        }
        self.root = self.ids[b""]
        self.nodes = len(self.ids)
        self.table = np.zeros(self.nodes + 1, ROW)
        for k, i in self.ids.items():
            self.table[i]["h"] = keys[k]
            self.table[i]["child"] = [
                self.ids.get(k + bytes([b]), 0) for b in range(256)
            ]
        self.fetched = []

    def rows(self, ids):
        self.fetched.extend(ids.tolist())
        return self.table[ids]

    def suffix(self, raw):
        return np.array(
            [
                [self.ids.get(raw[j - d : j], 0) if d <= j else 0 for d in range(1, 9)]
                for j in range(1, len(raw) + 1)
            ],
            dtype="<u4",
        )


def reference(q, retained):
    mass = [0] * 256
    for doc in DOCS:
        for i, b in enumerate(doc):
            # Direct position-matrix row, not a histogram merge.
            w = 1
            for d in range(1, min(i, len(q), 8) + 1):
                if q[-d:] == doc[i - d : i] and q[-d:] in retained:
                    w += 32**d
            mass[b] += w
    return (np.array(mass) / sum(mass)).astype(np.float32)


@pytest.mark.parametrize("floor", [1, 2, 4])
def test_position_kernel_and_every_operand(floor):
    pages = Fixture(floor=floor)
    bank = Kernel(pages, cache=3)
    qs = (b"", b"\x00\x1e", b"\xff", b"ZA", b"BABA", b"YAYAYAYAYAY", b"AXA")
    for q in qs:
        literal = bank.read([q] * 256)
        np.testing.assert_array_equal(literal.pdf[0], reference(q, pages.ids))
        cs = np.arange(256, dtype=np.uint8)
        chosen = bank.follow(literal, cs)
        np.testing.assert_array_equal(
            chosen.pdf, np.stack([reference(q + bytes([c]), pages.ids) for c in cs])
        )
    assert bank.counts["parent_reads"] == bank.counts["child_reads"] == 256 * len(qs)


def test_not_prefetching_unchosen_pages_or_changing_observations():
    p = Fixture()
    k = Kernel(p)
    q = k.read([b"A"])
    assert p.ids[b"AB"] not in p.fetched and p.ids[b"AC"] not in p.fetched
    k.follow(q, np.array([ord("B")], np.uint8))
    assert p.ids[b"AB"] in p.fetched and p.ids[b"AC"] not in p.fetched
    np.testing.assert_array_equal(q.pdf[0], reference(b"A", p.ids))
    with pytest.raises(ValueError):
        Kernel(Fixture()).follow(q, np.array([1], np.uint8))
    with pytest.raises(PermissionError):
        Kernel(p, scope="exclude-even-one-document")


def test_reset_binary_prefix_and_pruned_child_not_cross_document():
    p = Fixture(floor=2)
    k = Kernel(p)
    x = np.array([[65, 66, 65, 88], [255, 0, 30, 255]], np.uint8)
    ans = k.scan(x)
    for i in range(x.shape[0]):
        for t in range(x.shape[1]):
            np.testing.assert_array_equal(
                ans.pdf[i, t], reference(bytes(x[i, : t + 1]), p.ids)
            )
    assert p.table[p.ids[b"AB"]]["h"][256] == 1 and b"AXA" not in p.ids
    assert (
        p.table[p.ids[b"ABA"]]["h"][88] > 0 and p.table[p.ids[b"ABA"]]["child"][88] == 0
    )


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(174)
    m = Tape()
    with torch.no_grad():
        torch.nn.init.normal_(m.act.weight, std=0.13)
        torch.nn.init.normal_(m.emit.weight, std=0.12)
    return m


@pytest.mark.parametrize("arm", ARMS)
def test_parallel_replay_matches_every_online_prefix_and_all_256_scores(model, arm):
    x = np.array([[65, 66, 255, 65, 0], [65, 65, 66, 30, 65]], np.uint8)
    full = execute(model, Kernel(Fixture()), x, arm)
    y, lp = predict(model, full)
    if lp is not None:
        torch.testing.assert_close(lp, full.selected_logp)
    for t in range(x.shape[1]):
        k = Kernel(Fixture())
        prefix = execute(model, k, x[:, : t + 1], arm)
        logits, _ = predict(model, prefix)
        torch.testing.assert_close(logits[:, -1], y[:, t])
        if arm != "ONE":
            torch.testing.assert_close(prefix.action[:, -1], full.action[:, t])
        assert k.counts["parent_reads"] == 2 * (t + 1)
        assert k.counts["child_reads"] == (0 if arm == "ONE" else 2 * (t + 1))
    # Different labels can change the loss, never the previously issued instructions.
    obj, n, _ = score(model, full, x)
    _, other, _ = score(model, full, x ^ 255)
    assert not torch.equal(n, other)
    obj.backward()
    assert model.emit.weight.grad.abs().sum() > 0
    if arm in ("BLIND", "ADAPT"):
        assert model.act.weight.grad.abs().sum() > 0


def test_same_observation_different_first_answer_changes_query_only_if_visible():
    m = Tape()
    x = np.array([[65]], np.uint8)

    def act(arm, docs):
        tr = execute(m, Kernel(Fixture(docs)), x, arm)
        return int(tr.action.item()), tr.observed.item()

    for arm in ("ADAPT", "GREEDY"):
        assert act(arm, [b"ABBB"]) == (66, 65)
        assert act(arm, [b"ACCC"]) == (67, 65)
    assert act("BLIND", [b"ABBB"]) == act("BLIND", [b"ACCC"]) == (0, 65)


def test_a_past_child_can_affect_a_later_answer_and_a_later_choice(model):
    x = np.array([[65, 66, 65]], np.uint8)
    tr = execute(model, Kernel(Fixture()), x, "ADAPT")
    repl = tr.child.clone()
    repl[:, 0] = 0
    repl[:, 0, 244] = 1
    y, p = predict(model, tr)
    z, q = predict(model, replace(tr, child=repl))
    assert torch.equal(p[:, 0], q[:, 0])  # first action precedes the changed return
    assert not torch.equal(p[:, -1], q[:, -1]) and not torch.equal(y[:, -1], z[:, -1])


def test_score_function_matches_exact_two_decision_gradient_with_delayed_loss():
    v = torch.tensor([0.2, -0.41, 0.7], dtype=torch.float64, requires_grad=True)
    exact = surrogate = 0
    for a in range(2):
        for b in range(2):
            p = torch.stack((v[0], -v[0])).softmax(0)
            q = torch.stack((v[1 + a], -v[1 + a])).softmax(0)
            path = p[a] * q[b]
            costs = torch.tensor(
                [[0.4 + 0.3 * a, 0.2 + 0.5 * b + 1.3 * (a != b)]], dtype=v.dtype
            )
            lps = torch.stack((p[a].log(), q[b].log())).view(1, 2)
            exact += path * costs.mean()
            surrogate += path.detach() * policy_loss(
                lps, costs, torch.tensor([[0.3, 0.9]])
            )
    g = torch.autograd.grad(exact, v, retain_graph=True)[0]
    h = torch.autograd.grad(surrogate, v)[0]
    torch.testing.assert_close(g, h, atol=1e-14, rtol=1e-14)
    assert g.abs().min() > 0.01


def test_one_sample_and_no_hidden_child_reads_during_fit(model):
    k = Kernel(Fixture())
    x = np.array([[65, 66, 67], [65, 66, 88]], np.uint8)
    t = execute(model, k, x, "ADAPT", torch.Generator().manual_seed(421))
    c = k.counts.copy()
    obj, _, _ = score(model, t, x)
    obj.backward()
    assert c == k.counts and c["parent_reads"] == c["child_reads"] == 6
    a, _ = choose(torch.zeros(1, 256), None)
    assert a.item() == 0
