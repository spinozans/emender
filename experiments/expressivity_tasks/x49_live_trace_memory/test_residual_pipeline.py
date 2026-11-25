"""RB lifecycle-specific adversarial fixtures. Only invented data and public frozen coefficient receipts."""

from copy import deepcopy
import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    MIN_OFFSET,
    load,
    sha,
    write_json,
)
from experiments.expressivity_tasks.x49_live_trace_memory import (
    residual_data as data,
    residual_codec as codec,
    residual_panel as p,
    residual_run as run,
    residual_store as store,
    residual_stats as stats,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panels import SEEDS, select
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import Tiny
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
)


def fake_head(s):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(s)
        Tiny()
        return trained_base().eval()


def fake_residual_parent(tmp, mp, ck, coeff):
    mp.setattr(p.dr, "ROOT", tmp / "prior-dr")
    mp.setattr(p, "OUT", tmp)
    d = p.coarse_folder()
    d.mkdir(parents=True)
    seeds = {}
    audits = {}
    q = tmp / "qualify-residual-byte-fixture"
    q.mkdir()
    for s in SEEDS:
        k = str(s)
        path = d / f"dr-tables-{s}.npz"
        rng = np.random.default_rng(s)
        with path.open("xb") as f:
            np.savez(
                f,
                m=rng.uniform(-0.5, 0.5, (256, 32)).astype(np.float32),
                b=rng.uniform(0.1, 2, (256, 2)).astype(np.float32),
                k=rng.uniform(-0.2, 0.2, (256, 2, 32)).astype(np.float32),
            )
        seeds[k] = dict(tables_sha=sha(path), checkpoint_sha=ck(s))
        _, ax = codec.response_axes(fake_head(s))
        audits[k] = dict(**ax, checkpoint_sha=ck(s), coarse_sha=sha(path))
    f = d / "fit-frozen.json"
    write_json(f, dict(seeds=seeds, spec=data.dr.SPEC))
    mp.setattr(data, "DR_SEAL", sha(f))
    mp.setattr(
        data,
        "checked_spec",
        lambda: dict(
            decoders=seeds,
            checkpoints={str(s): ck(s) for s in SEEDS},
            u_beta={str(s): coeff(s)["U"] for s in SEEDS},
            j_beta={str(s): coeff(s)["J"] for s in SEEDS},
        ),
    )
    write_json(
        q / "passed.json",
        dict(
            ref="fixture",
            spec=data.SPEC,
            natural_windows_opened=0,
            head_audit=audits,
            states={str(s): dict(checkpoint=ck(s)) for s in SEEDS},
        ),
    )


def test_next_train_after_both_dr_roles_no_repacking(monkeypatch):
    records = []
    for i in range(23):
        off = MIN_OFFSET + 16384 * i
        k = i if i != 22 else 3
        records.append(
            dict(
                document=off,
                byte_offset=off,
                byte_length=12290,
                split="train",
                sha256=f"{k * 100 + 13:064x}",
            )
        )
    size = off + 12290
    pre = {records[14]["sha256"]}
    rank = select(records, pre, {"train": 15}, size)[0]["train"]
    monkeypatch.setattr(data.dr, "N", dict(fit=4, probe=3))
    b = dict(
        schema="x49-dr-native-train",
        spec=data.dr.SPEC,
        source=dict(size=size),
        roles=dict(fit=rank[:4], probe=rank[4:7]),
    )
    rows, ex, left = data.next_block(b, records, pre, 8)
    assert (
        rows == rank[7:15]
        and left == 14
        and ex == pre | {r["sha256"] for r in rank[:7]}
    )
    bad = deepcopy(b)
    bad["roles"]["probe"] = rank[5:8]
    with pytest.raises(ValueError, match="rank suffix"):
        data.next_block(bad, records, pre, 8)
    bad = deepcopy(b)
    bad["roles"]["probe"][0]["start"] += 4096
    with pytest.raises(ValueError, match="overlapping FIT/PROBE"):
        data.next_block(bad, records, pre, 8)


def test_qualified_axis_and_dr_table_bindings_refuse_swaps(tmp_path, monkeypatch):
    from experiments.expressivity_tasks.x49_live_trace_memory.test_alternative_pipeline import (
        fake,
    )

    fake(tmp_path, monkeypatch, "RB")
    data.prepare()
    initial = p.plan("fixture", hash_weights=False)
    cfg = dict(p.expected("fixture"), operator_note="fixture RB only")
    p.validate(cfg, "fixture")
    for k, v in [
        ("scope", "x49-dr-two-note-decoder-v1"),
        ("dr_fit_seal", "wrong"),
        ("weight_and_readouts_id", "wrong"),
        ("d_refits", 1),
        ("bits", 7),
        ("retries", 1),
    ]:
        with pytest.raises(ValueError, match="authority"):
            p.validate(dict(cfg, **{k: v}), "fixture")
    monkeypatch.setattr(run.joint, "_writer", lambda: object())
    head = fake_head(4901)
    assert run.document_options(initial, 4901, head)["axes"].nbytes == 1024
    with pytest.raises(ValueError, match="wrong RB seed/axes"):
        run.document_options(initial, 4901, fake_head(4902))
    for k in ("qualified_axes", "coarse_sha", "u_beta", "j_beta", "endpoints"):
        bad = deepcopy(initial)
        bad[k]["4901"] = bad[k]["4902"]
        with pytest.raises(ValueError, match="seed"):
            run.document_options(bad, 4901, head)
    f = tmp_path / "qualify-residual-byte-fixture/passed.json"
    orig = f.read_bytes()
    z = load(f)
    z["head_audit"]["4901"]["checkpoint_sha"] = "wrong"
    write_json(f, z)
    with pytest.raises(ValueError, match="foreign qualified RB axes"):
        p.expected("fixture")
    f.write_bytes(orig)
    f = p.coarse_folder() / "dr-tables-4901.npz"
    f.write_bytes((p.coarse_folder() / "dr-tables-4902.npz").read_bytes())
    with pytest.raises(ValueError, match="changed DR array bytes"):
        p.plan("fixture", hash_weights=False)
    cs = {
        a + ":" + b: dict(gain=0.02, ci95=[0.01, 0.03], by_seed=[0.02] * 3)
        for a, b in stats.PAIRS
    }
    assert stats.decision(cs)["consider_locked_calibration_transfer"]
    cs["R:U"]["ci95"][0] = 0
    assert (
        stats.decision(cs)["useful_bound_residual_byte"]
        and not stats.decision(cs)["consider_locked_calibration_transfer"]
    )
    cs["R:S"]["by_seed"][2] = 0
    assert not stats.decision(cs)["useful_bound_residual_byte"]


def test_codebook_replay_denominators_and_changed_mean_refusals(tmp_path):
    m = codec.Moments()
    torch.manual_seed(4914)
    h = m.add(torch.randint(21, (57,), dtype=torch.uint8), torch.randn(57, 32).tanh())
    b = m.solve()
    z = store.store(tmp_path, 4901, m, b)
    logs = [dict(fit_code_counts=h, work=dict(fit_writes=57))]
    entry = store.check_state(tmp_path, 4901, z, logs)
    assert (
        entry["seen_codes"] == sum(v > 0 for v in h)
        and store.independent(tmp_path, 4901, entry)["audited_codes"] == 256
    )
    bad = deepcopy(logs)
    bad[0]["fit_code_counts"].reverse()
    with pytest.raises(ValueError, match="occupancy"):
        store.check_state(tmp_path, 4901, z, bad)
    f = tmp_path / "rb-tables-4901.npz"
    new = b.means.numpy().copy()
    new[h.index(max(h)), 1] += 0.01
    with f.open("wb") as out:
        np.savez(
            out, means=new, global_mean=b.global_mean.numpy(), count=b.count.numpy()
        )
    with pytest.raises(ValueError, match="changed RB array bytes"):
        store.tables(tmp_path, 4901, z)
    corrupt = dict(entry, tables_sha=sha(f))
    with pytest.raises(ValueError, match="pre-probe FIT moments"):
        store.check_state(tmp_path, 4901, corrupt, logs)
    with pytest.raises(ValueError, match="independent mean"):
        store.independent(tmp_path, 4901, corrupt)
