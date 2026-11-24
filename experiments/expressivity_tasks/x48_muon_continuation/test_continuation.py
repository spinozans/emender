"""No CUDA. Tiny trainable host for optimizer semantics, real P100/warm controller for local seam test."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest
import torch
from torch import nn
from ndm.memory_program import PackedMemoryPolicy, Session
from ndm.memory_program.direct import DirectTrust
from ndm.memory_program.direct_entry import DirectEntryProgram
from experiments.expressivity_tasks.x45_matrix_emender.muon_control import split_parameters_gdn2
from experiments.expressivity_tasks.x47_xsa_attention.large_host_bench import (
    DIR,
    identity as host_identity,
    STATE_SHA,
)
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    MIN_OFFSET,
    T,
    B,
    cut,
    select,
    OUT,
    STEPS,
    Stream,
    load,
)
from experiments.expressivity_tasks.x48_muon_continuation.learning import (
    OnlyDirect,
    controller_groups,
    hybrid,
    conditional,
    direct,
    updated,
    factor,
    resources,
)
from experiments.expressivity_tasks.x48_muon_continuation.lease import authorize
from experiments.expressivity_tasks.x48_muon_continuation.train import (
    BRACKET,
    STREAM_ID,
    MAN,
    controller,
    elect,
    restore,
    save,
)


def record(i, raw, split="train"):
    return dict(
        document=i,
        byte_offset=MIN_OFFSET + 500 * i,
        byte_length=len(raw),
        sha256=sha256(raw).hexdigest(),
        split=split,
    )


def test_select_and_cut_are_disjoint_deterministic_and_hash_checked():
    raws = [bytes([i]) + bytes(range(256)) * 2 for i in range(7)]
    rs = [record(i, w) for i, w in enumerate(raws)]
    rs.append(dict(rs[0], split="final_test"))
    rs.append(dict(rs[1], split="calibration"))
    rs[2]["byte_offset"] = 0
    a = select(rs, [rs[3]["sha256"]], 3)
    b = select(reversed(rs), [rs[3]["sha256"]], 3)
    assert a == b and {r["doc"] for r in a} == {4, 5, 6}
    for r in a:
        raw = raws[r["doc"]]
        assert cut(raw, r) == raw[r["start"] : r["start"] + 129]
        with pytest.raises(ValueError):
            cut(b"!" + raw[1:], r)
        with pytest.raises(ValueError):
            cut(raw, dict(r, start=len(raw) - 127))
        with pytest.raises(ValueError):
            cut(raw, dict(r, split="calibration"))
    with pytest.raises(ValueError):
        select(rs, [rs[3]["sha256"]], 4)


def test_full_prepared_stream_and_gate_no_wrap():
    if not (OUT / "data").exists():
        pytest.skip("requires registered prepared training stream")
    stream = Stream(OUT, STREAM_ID)
    seen = set()
    for step in range(1, STEPS + 1):
        rows, words = stream.batch(step)
        assert all(len(w) == T + 1 for w in words) and len(words) == B
        seen.update(r["sha256"] for r in rows)
    assert len(seen) == B * STEPS
    for n in (0, 1025):
        with pytest.raises(ValueError):
            stream.batch(n)


class ToyHost(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embed = nn.Embedding(256, width)
        self.out = nn.Linear(width, 256)

    def forward(self, ids):
        h = self.embed(ids).cumsum(1).tanh()
        return torch.cat((h, self.out(h)), -1)


class ToyBank:
    source, version = "fixture", "v1"

    def __init__(self):
        self.queries = []

    def histograms(self, keys):
        self.queries.extend(keys)
        return [tuple((3 if b == 0 else 1) for b in range(257)) for k in keys]


def tiny(arm):
    torch.manual_seed(1357)
    net = ToyHost(8)
    head = DirectTrust(8, width=8)
    ctrl = (
        OnlyDirect(head)
        if arm == "direct"
        else DirectEntryProgram(PackedMemoryPolicy(8, dim=8, heads=2, buckets=32), head)
    )
    if arm == "program":
        with torch.no_grad():
            ctrl.entry.linear.bias.zero_()
    a = hybrid(([net.embed.weight, net.out.weight], [net.out.bias], ["embed", "out"], ["bias"]), 5e-5)[0]
    b = hybrid(controller_groups(ctrl), 3e-4)[0]
    return net, ctrl, (a, b)


def one(arm, net, ctrl, opts, step):
    words = [b"\xff\0=xy=A", b"\x1eXX=xyD"] if step == 1 else [b"!qwerty", b"zqwertb"]
    rs = [dict(doc=i + 20 * step, start=0) for i in range(len(words))]

    def f(enc):
        bank = ToyBank()
        if arm == "direct":
            return direct(ctrl, enc, bank, rs, words)
        ss = [Session(document=r["doc"], start=0) for r in rs]
        return conditional(ctrl, enc, bank, rs, words, ss)

    return updated(net, ctrl, opts, "L", step, f, width=8, microbatch=1)


def equal_tree(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("arm", ["direct", "program"])
def test_two_coupled_updates_serialize_every_optimizer_and_reproduce_next_transition(tmp_path, arm):
    torch.use_deterministic_algorithms(True, warn_only=False)
    h, c, opts = tiny(arm)
    before = c.direct.hidden.weight.detach().clone()
    old = h.embed.weight.detach().clone()
    z = one(arm, h, c, opts, 1)
    assert (
        z["targets"] == 12
        and not torch.equal(old, h.embed.weight)
        and not torch.equal(before, c.direct.hidden.weight)
        and z["update_rms"]["host"]["muon"] > 0
    )
    if arm == "program":
        assert 0 < z["work"]["program"] < 12 and opts[1].aux.state[c.entry.linear.weight]["step"] == 1
    bind = dict(arm=arm, candidate="L")
    ck = save(tmp_path, bind, 1, h, c, opts)
    h2, c2, o2 = tiny(arm)
    for changed in (dict(arm="foreign", candidate="L"), dict(arm=arm, candidate="H")):
        with pytest.raises(ValueError):
            restore(tmp_path, changed, 1, h2, c2, o2)
    assert restore(tmp_path, bind, 1, h2, c2, o2) == ck["sha256"]
    y = one(arm, h, c, opts, 2)
    y2 = one(arm, h2, c2, o2, 2)
    for k in ("preupdate_nll", "priced_nll", "train_objective", "preclip", "work"):
        assert y[k] == y2[k]
    for x, other in zip((h, c, *opts), (h2, c2, *o2), strict=True):
        assert equal_tree(x.state_dict(), other.state_dict())
    save(tmp_path, bind, 2, h, c, opts)
    assert not (tmp_path / ck["file"]).exists()
    (tmp_path / "step-0002.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="incomplete"):
        restore(tmp_path, bind, 2, h2, c2, o2)


def test_schedule_bracket_tie_and_refuse_unfunded_launch(tmp_path):
    assert [factor(i) for i in (1, 16, 896, 897, 1024)] == [1 / 16, 1, 1, 1, 1 / 128]
    y = [dict(preupdate_nll=2.0) for _ in range(BRACKET)]
    x = deepcopy(y)
    x[0]["preupdate_nll"] = 1
    assert elect(dict(H=x, L=y))[0] == "L"
    x[-1]["preupdate_nll"] = 1
    assert elect(dict(H=x, L=y))[0] == "H"
    with pytest.raises(ValueError, match="missing explicit"):
        authorize(tmp_path)


def test_registered_comparison_reports_post_election_and_refuses_a_changed_batch(tmp_path, monkeypatch):
    import experiments.expressivity_tasks.x48_muon_continuation.compare as report
    from experiments.expressivity_tasks.x48_muon_continuation.data import write_json, stream_id, sha

    root = tmp_path
    (root / "panel").mkdir()
    (root / "data").mkdir()
    (root / "gated-ref").write_text("test")
    keys = [sha256(str(i).encode()).hexdigest() for i in range(B * STEPS)]
    bs = [sha256("".join(keys[i : i + B]).encode()).hexdigest() for i in range(0, len(keys), B)]
    meta = root / "data/rows.jsonl"
    meta.write_text("".join(json.dumps({"window_sha": v}) + "\n" for v in keys))
    ident = {"rows_sha": sha(meta)}
    write_json(root / "data/identity.json", ident)
    stream = stream_id(ident)
    monkeypatch.setattr(report, "STREAM_ID", stream)
    write_json(root / "panel/charges.json", dict(direct=8000, program=9000))
    for arm, base in [("direct", 2.0), ("program", 1.95)]:
        d = root / arm
        d.mkdir()
        write_json(d / "charge.json", dict(rc=0, upper_card_s=8000 if arm == "direct" else 9000))
        write_json(d / "complete.json", dict(final_checkpoint=sha256(b"blob").hexdigest()))
        write_json(d / "election.json", dict(winner="L", train_scores=dict(L=2.0, H=3.0)))
        for k in ("L", "H", "main"):
            folder = d / k
            folder.mkdir()
            c = "L" if k == "main" else k
            steps = range(BRACKET + 1, STEPS + 1) if k == "main" else range(1, BRACKET + 1)
            bind = dict(stream=stream, ref="test", candidate=c, arm=arm)
            write_json(
                folder / "origin.json", dict(binding=bind, from_checkpoint="pilot" if k == "main" else None)
            )
            n = base if k == "main" else (2.0 if k == "L" else 3.0)
            records = [
                dict(
                    step=i,
                    arm=arm,
                    candidate=c,
                    batch_sha=bs[i - 1],
                    targets=T * B,
                    preupdate_nll=n,
                    priced_nll=n + 0.001,
                    work=dict(direct_keys=19),
                )
                for i in steps
            ]
            (folder / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            write_json(
                folder / "complete.json",
                dict(
                    optimizer_steps=len(steps),
                    main_core_s=1.0,
                    children_core_s=2.0,
                    wall_s=2.0,
                    peak_allocated_bytes=12,
                ),
            )
            (folder / "blob").write_bytes(b"blob")
            write_json(
                folder / "latest.json",
                dict(
                    step=steps[-1],
                    file="blob",
                    sha256=sha256(b"blob").hexdigest() if k == "main" else "pilot",
                ),
            )
    res = report.compare(root)
    assert res["held_out_targets"] == 0 and res["contrasts"]["last_256"]["direct_minus_program"][
        "preupdate_nll"
    ] == pytest.approx(0.05)
    p = root / "program/main/steps.jsonl"
    p.write_text(p.read_text().replace(bs[-1], "foreign"))
    with pytest.raises(ValueError, match="unpaired"):
        report.compare(root)


def test_real_origin_routing():
    if not (DIR / "final.pt").exists():
        pytest.skip("requires local registered host")
    assert host_identity()["checkpoint"] == STATE_SHA
    state = torch.load(DIR / "final.pt", weights_only=True, mmap=True, map_location="meta")

    class Skeleton:
        def named_parameters(self):
            return [(n, nn.Parameter(p)) for n, p in state.items()]

    m, a, mn, an = split_parameters_gdn2(Skeleton(), gates_on_adamw=True)
    assert (len(m), len(a), sum(p.numel() for p in m)) == (191, 406, 122255360)
    assert all("b_proj.weight" not in n for n in mn) and any("b_proj.weight" in n for n in an)
    q = controller("program")
    mu, aux, names, auxnames = controller_groups(q)
    assert (
        "program.writer.weight" in names
        and {"entry.linear.weight", "direct.vote.weight", "program.prior"} <= set(auxnames)
        and len(mu) + len(aux) == 41
    )
    q.register_parameter("unclassified", nn.Parameter(torch.zeros(3, 5)))
    with pytest.raises(ValueError, match="unclassified"):
        controller_groups(q)


@pytest.mark.parametrize("arm", ["direct", "program"])
def test_coupled_native_training_seam_synthetic_words_only(arm):
    # True P100/data sources and warm heads, synthetic text and tiny differentiable CPU host. No production weights fitted.
    if not (DIR / "final.pt").exists():
        pytest.skip("requires local registered binary/files")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(42)
    net = ToyHost(256)
    ctrl = controller(arm)
    opts = (
        hybrid(([net.embed.weight, net.out.weight], [net.out.bias], [], []), 5e-5)[0],
        hybrid(controller_groups(ctrl), 3e-4)[0],
    )
    meter = {}
    rows = [dict(doc=i + 10, start=0) for i in range(4)]
    words = [b"toy=" + bytes([j, 0, 255, 30]) * 6 for j in range(4)]

    def forward(enc):
        with resources(load(MAN), arm, ctrl, rows, words, meter) as (bank, ss):
            if arm == "direct":
                return direct(ctrl, enc, bank, rows, words)
            return conditional(ctrl, enc, bank, rows, words, ss)

    r = updated(net, ctrl, opts, "L", 1, forward)
    assert (
        r["targets"] == 108
        and r["work"]["direct_keys"] > 0
        and meter["wire_bytes"] > 0
        and r["update_rms"]["controller"]["muon"] > 0
    )
    if arm == "program":
        assert meter["private"]["observed_bytes"] == 112 and r["work"]["program"] > 0
    else:
        assert not meter["private"] and not meter["position"] and r["work"]["program"] == 0
