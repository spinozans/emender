"""Matched CPU fits and fixed-weight erasure at quarter read rate."""

import hashlib
from time import monotonic, process_time

import numpy as np
import torch

from experiments.expressivity_tasks.x50_read_tape.kernel import Kernel
from experiments.expressivity_tasks.x50_read_tape.study import (
    BATCH,
    EPOCHS,
    model_at_birth,
    setup,
)
from . import data as d
from .reader import ARMS, STRIDE, T, acquire, nll, predict

__all__ = [
    "BATCH",
    "EPOCHS",
    "model_at_birth",
    "setup",
    "phase",
    "score_report",
    "order",
    "summarize",
    "expected_reads",
]
BANDS = {
    "all": (0, 1, 2, 3),
    "gaps": (1, 2, 3),
    "read": (0,),
    "age1": (1,),
    "age2": (2,),
    "age3": (3,),
}
CONTRASTS = {
    "KEEP4:DISCARD4": ("all", "gaps"),
    "KEEP4:ERASE4": ("all", "gaps"),
    "EVERY:KEEP4": ("all", "gaps"),
    "EVERY:NONE": ("all",),
    "KEEP4:NONE": ("all",),
    "DISCARD4:NONE": ("all",),
}
POLICIES = (*ARMS, "ERASE4", "UNIGRAM", "FULL_PRIOR", "SPARSE_PRIOR")


def order(seed):
    j = d.SEEDS.index(seed)
    return ARMS[j:] + ARMS[:j]


def expected_reads(nt, arm):
    return {"NONE": 0, "EVERY": nt, "DISCARD4": nt // 4, "KEEP4": nt // 4}[arm]


def sums(v):
    return v.detach().double().reshape(len(v), -1, STRIDE).sum(1).numpy()


def phase(m, pages, windows, u, seed, arm, fit, tick):
    t0, c0 = monotonic(), process_time()
    bank = Kernel(pages) if arm != "NONE" else None
    m.train(fit)
    opt = (
        torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0.01) if fit else None
    )
    n, length = windows.shape
    d.check(
        n > 0 and length == T + 1 and n % BATCH == 0 and arm in ARMS, "invalid dose"
    )
    rng = np.random.default_rng(seed)
    epochs = EPOCHS if fit else 1
    dig = hashlib.sha256()
    means = []
    steps = 0
    diag = not fit and arm == "KEEP4"
    dt = dc = at = ac = 0.0
    arch_t = arch_c = 0.0
    gap = torch.arange(T) % STRIDE != 0
    for _ in range(epochs):
        ranks = rng.permutation(n) if fit else np.arange(n)
        keys = (
            ("model", "prior", "erase", "tv", "changed") if diag else ("model", "prior")
        )
        ds = {k: np.full((n, STRIDE), np.nan) for k in keys}
        for ix in ranks.reshape(-1, BATCH):
            words = windows[ix]
            t, c = monotonic(), process_time()
            batch = acquire(bank, words[:, :-1], arm)
            arch_t += monotonic() - t
            arch_c += process_time() - c
            with torch.set_grad_enabled(fit):
                z, prior = predict(m, batch, arm, u)
                y = torch.from_numpy(words[:, 1:].astype(np.int64))
                loss = nll(z, y)
                if fit:
                    opt.zero_grad(set_to_none=True)
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(
                        m.parameters(), 1.0, error_if_nonfinite=True
                    )
                    opt.step()
            t, c = monotonic(), process_time()
            ds["model"][ix] = sums(loss)
            ds["prior"][ix] = sums(nll(prior, y))
            if arm != "NONE":
                scheduled = batch.pdf[:, ::STRIDE] if arm == "EVERY" else batch.pdf
                dig.update(scheduled.numpy().tobytes())
                dig.update(prior[:, ::STRIDE].numpy().tobytes())
            if arm != "EVERY":
                p = prior if arm == "NONE" else prior[:, gap]
                d.check(
                    torch.equal(p, u.expand_as(p)),
                    "unexpected prior on no-read prediction",
                )
            at += monotonic() - t
            ac += process_time() - c
            if diag:
                t, c = monotonic(), process_time()
                with torch.no_grad():
                    e, _ = predict(m, batch, "DISCARD4", u)
                    d.check(
                        torch.equal(z[:, 0], e[:, 0]),
                        "erasure changed first prediction",
                    )
                    ds["erase"][ix] = sums(nll(e, y))
                    tv = 0.5 * (
                        z.double().softmax(-1) - e.double().softmax(-1)
                    ).abs().sum(-1)
                    ds["tv"][ix] = sums(tv)
                    ds["changed"][ix] = sums(tv > 1e-6)
                dt += monotonic() - t
                dc += process_time() - c
            t, c = monotonic(), process_time()
            steps += 1
            tick(BATCH * T)
            at += monotonic() - t
            ac += process_time() - c
        d.check(
            all(np.isfinite(v).all() for v in ds.values()), "missing or nonfinite score"
        )
        means.append(float(ds["model"].sum() / (n * T)))
    nt = n * T * epochs
    counts = (
        bank.counts
        if bank
        else dict.fromkeys(
            (
                "parent_reads",
                "child_reads",
                "prefix_frames",
                "page_rows",
                "page_batches",
            ),
            0,
        )
    )
    d.check(
        counts["parent_reads"] == expected_reads(nt, arm)
        and counts["child_reads"] == 0
        and counts["prefix_frames"]
        == (n * epochs if arm == "EVERY" else expected_reads(nt, arm)),
        "wrong number of reads/frames",
    )
    cost = dict(
        chosen_s=monotonic() - t0 - at - dt,
        chosen_cpu_s=process_time() - c0 - ac - dc,
        acquire_s=arch_t,
        acquire_cpu_s=arch_c,
        audit_s=at,
        audit_cpu_s=ac,
        erase_s=dt,
        erase_cpu_s=dc,
    )
    return dict(
        targets=nt,
        epochs=epochs,
        batches=steps,
        documents=n,
        archive=dict(counts),
        cost=cost,
        refresh_hash=dig.hexdigest() if bank else None,
        epoch_nll=means,
    ), ds


def bands(v):
    # Last axis contains sums for four equally exposed phases of a full window.
    return {
        name: v[..., list(ages)].sum(-1) / (len(ages) * (T // STRIDE))
        for name, ages in BANDS.items()
    }


def summarize(curves, *, draws=10000, multiplicity=False):
    shape = curves["EVERY"].shape
    d.check(
        set(curves) == set(POLICIES)
        and len(shape) == 3
        and shape[0] == 3
        and shape[2] == 4
        and shape[1] > 0
        and all(v.shape == shape and np.isfinite(v).all() for v in curves.values()),
        "unpaired documents",
    )
    rates = {a: bands(v) for a, v in curves.items()}
    absolute = {
        a: {
            k: dict(mean=float(vals.mean()), by_seed=vals.mean(1).tolist())
            for k, vals in vv.items()
        }
        for a, vv in rates.items()
    }
    keys = []
    by = []
    for name, bb in CONTRASTS.items():
        a, b = name.split(":")
        for band in bb:
            keys.append((name, band))
            by.append(rates[b][band] - rates[a][band])
    by = np.stack(by)
    mean = by.mean(1)
    n = shape[1]
    rng = np.random.default_rng(5200)
    blocks = []
    for off in range(0, draws, 128):
        ix = rng.integers(n, size=(min(128, draws - off), n))
        if multiplicity:
            blocks.append(
                mean @ np.stack([np.bincount(row, minlength=n) for row in ix]).T / n
            )
        else:
            blocks.append(mean[:, ix].mean(-1))
    ci = np.percentile(
        np.concatenate(blocks, axis=1), [2.5, 97.5], axis=1, method="linear"
    )
    out = {}
    for i, (name, band) in enumerate(keys):
        lo, hi = ci[:, i].tolist()
        ys = by[i].mean(1).tolist()
        decision = (
            "win"
            if min(ys) > 0 and lo > 0
            else "loss"
            if max(ys) < 0 and hi < 0
            else "unresolved"
        )
        out.setdefault(name, {})[band] = dict(
            gain=float(by[i].mean()), ci95=[lo, hi], by_seed=ys, decision=decision
        )
    return absolute, out


def score_report():
    r = d.RUN
    z = d.frozen()
    p_sha = d.sha(r / "probe.json")
    d.check(
        d.read(r / "probe.json")["fit_seal"] == d.sha(r / "frozen.json"),
        "probe lacks current seal",
    )
    curves = {a: [] for a in POLICIES}
    jobs = {}
    support = {}
    common = None
    n = d.N["probe"]
    for seed in d.SEEDS:
        stamps = {}
        priors = {}
        for arm in ARMS:
            key = f"{seed}-{arm}"
            f = d.read(r / (key + "-fit.json"))
            p = d.read(r / (key + "-probe.json"))
            archive = r / (key + "-scores.npz")
            d.check(
                f["state"] == p["state"] == z["states"][key]
                and f["fit"] == z["fit_sha"]
                and p["probe"] == p_sha
                and p["scores"] == d.sha(archive),
                "changed score or state",
            )
            jobs[key] = {}
            stamps[arm] = []
            for label, facts, epochs in [
                ("fit", f["facts"], EPOCHS),
                ("probe", p["facts"], 1),
            ]:
                nt = d.N[label] * T * epochs
                nn = expected_reads(nt, arm)
                a = facts["archive"]
                cost = facts["cost"]
                d.check(
                    facts["targets"] == facts["batches"] * BATCH * T == nt
                    and facts["documents"] == d.N[label]
                    and facts["epochs"] == epochs
                    and len(facts["epoch_nll"]) == epochs
                    and a["parent_reads"] == nn
                    and a["child_reads"] == 0
                    and a["prefix_frames"] == (nn // T if arm == "EVERY" else nn)
                    and cost["chosen_s"] >= cost["acquire_s"] >= 0
                    and (cost["erase_s"] > 0) == (label == "probe" and arm == "KEEP4"),
                    "wrong read or timing receipt",
                )
                stamps[arm].append(facts["refresh_hash"])
                jobs[key][label] = {
                    k: facts[k]
                    for k in ("targets", "archive", "cost", "epoch_nll", "batches")
                }
            with np.load(archive, allow_pickle=False) as ds:
                fields = {"model", "prior"} | (
                    {"erase", "tv", "changed"} if arm == "KEEP4" else set()
                )
                d.check(
                    set(ds.files) == fields
                    and all(
                        v.shape == (n, 4) and np.isfinite(v).all() for v in ds.values()
                    )
                    and float(ds["model"].sum() / (n * T))
                    == p["facts"]["epoch_nll"][0],
                    "bad score schema or total",
                )
                curves[arm].append(ds["model"])
                priors[arm] = ds["prior"]
                if arm == "KEEP4":
                    curves["ERASE4"].append(ds["erase"])
                    c = ds["changed"]
                    tv = ds["tv"]
                    den = n * T // 4
                    d.check(
                        np.all((c == np.floor(c)) & (c >= 0) & (c <= T // 4))
                        and (c[:, 0] < T // 4).all()
                        and np.all((tv >= 0) & (tv <= T // 4)),
                        "invalid erasure support",
                    )
                    support[str(seed)] = dict(
                        first_predictions_equal=n,
                        targets_per_age=den,
                        changed_by_age=c.sum(0).astype(int).tolist(),
                        documents_changed_by_age=(c > 0).sum(0).tolist(),
                        mean_tv_by_age=(tv.sum(0) / den).tolist(),
                    )
        d.check(
            stamps["NONE"] == [None, None]
            and all(stamps["KEEP4"])
            and stamps["KEEP4"] == stamps["DISCARD4"] == stamps["EVERY"],
            "wrong scheduled queries or priors",
        )
        if common is None:
            common = stamps["KEEP4"][1]
        d.check(common == stamps["KEEP4"][1], "probe reads differ across seeds")
        np.testing.assert_array_equal(priors["KEEP4"], priors["DISCARD4"])
        np.testing.assert_array_equal(priors["KEEP4"][:, 0], priors["EVERY"][:, 0])
        np.testing.assert_array_equal(priors["KEEP4"][:, 1:], priors["NONE"][:, 1:])
        curves["FULL_PRIOR"].append(priors["EVERY"])
        curves["UNIGRAM"].append(priors["NONE"])
        curves["SPARSE_PRIOR"].append(priors["KEEP4"])
    curves = {k: np.stack(v) for k, v in curves.items()}
    absolute, contrasts = summarize(curves)
    _, audit = summarize(curves, multiplicity=True)
    for k, bands_ in contrasts.items():
        for b, v in bands_.items():
            np.testing.assert_allclose(
                v["ci95"], audit[k][b]["ci95"], rtol=1e-12, atol=1e-14
            )
    fraction = None
    if contrasts["EVERY:NONE"]["all"]["decision"] == "win":
        fraction = (
            contrasts["KEEP4:NONE"]["all"]["gain"]
            / contrasts["EVERY:NONE"]["all"]["gain"]
        )
    prices = {
        a: {
            role: sum(jobs[f"{s}-{a}"][role]["cost"]["chosen_s"] for s in d.SEEDS)
            * 1e6
            / (len(d.SEEDS) * T * d.N[role] * (EPOCHS if role == "fit" else 1))
            for role in d.N
        }
        for a in ARMS
    }
    return dict(
        spec=d.SPEC,
        registration=d.REG,
        qualification=(d.OUT / "qualified-ref").read_text().strip(),
        fit_seal_sha=d.sha(r / "frozen.json"),
        seed_order=list(d.SEEDS),
        documents=d.N,
        policies=absolute,
        contrasts=contrasts,
        jobs=jobs,
        prices_s_per_million=prices,
        erasure_support=support,
        useful_reuse_in_gaps=all(
            contrasts[k]["gaps"]["decision"] == "win"
            for k in ("KEEP4:DISCARD4", "KEEP4:ERASE4")
        ),
        net_gain_over_discard=contrasts["KEEP4:DISCARD4"]["all"]["decision"],
        full_gain_fraction_recovered=fraction,
        probe_speedup_every_over_keep4=prices["EVERY"]["probe"]
        / prices["KEEP4"]["probe"],
        equal_refresh_answers_and_priors=True,
        bootstrap=dict(
            draws=10000,
            seed=5200,
            paired_unit="whole document",
            multiplicity_crosscheck=True,
        ),
        scope="local native TRAIN probe; no final or scale decision",
    )
