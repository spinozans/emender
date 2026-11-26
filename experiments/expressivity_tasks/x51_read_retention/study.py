"""Fit three readers from scratch. Price the chosen path separately from the DROP diagnostic."""

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
from .reader import ARMS, T, acquire, nll, predict

__all__ = [
    "phase",
    "report",
    "compare",
    "order",
    "model_at_birth",
    "setup",
    "BATCH",
    "EPOCHS",
]
PAIRS = (("KEEP", "DISCARD"), ("DISCARD", "NONE"), ("KEEP", "NONE"), ("KEEP", "DROP"))
COUNTERS = ("parent_reads", "child_reads", "prefix_frames", "page_rows", "page_batches")


def order(seed):
    i = d.SEEDS.index(seed)
    return ARMS[i:] + ARMS[:i]


def phase(model, pages, windows, unigram, seed, arm, fit, tick):
    start, c0 = monotonic(), process_time()
    bank = None if arm == "NONE" else Kernel(pages)
    model.train(fit)
    opt = (
        torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        if fit
        else None
    )
    n, length = windows.shape
    d.check(n > 0 and length == T + 1 and n % BATCH == 0 and arm in ARMS, "wrong dose")
    rng = np.random.default_rng(seed)
    epochs = EPOCHS if fit else 1
    means = []
    steps = 0
    dig = hashlib.sha256()
    dt = dc = at = ac = 0.0
    diag = arm == "KEEP" and not fit
    for _ in range(epochs):
        ranks = rng.permutation(n) if fit else np.arange(n)
        scores = {
            k: np.full(n, np.nan)
            for k in (
                ("model", "prior", "drop", "tv_sum", "changed")
                if diag
                else ("model", "prior")
            )
        }
        for ix in ranks.reshape(-1, BATCH):
            w = windows[ix]
            x, p = acquire(bank, w[:, :-1], arm)
            with torch.set_grad_enabled(fit):
                logits, prior = predict(model, x, p, arm, unigram)
                y = torch.from_numpy(w[:, 1:].astype(np.int64))
                loss = nll(logits, y)
                if fit:
                    opt.zero_grad(set_to_none=True)
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    opt.step()
            # Receipt work is not a model read; do not charge it to a prospective decoder.
            t, c = monotonic(), process_time()
            scores["model"][ix] = loss.detach().double().sum(1).numpy()
            scores["prior"][ix] = nll(prior, y).double().sum(1).numpy()
            if p is not None:
                dig.update(p.numpy().tobytes())
                dig.update(prior.numpy().tobytes())
            at += monotonic() - t
            ac += process_time() - c
            if diag:
                t, c = monotonic(), process_time()
                with torch.no_grad():
                    alt, _ = predict(model, x, p, "DISCARD", unigram)
                    d.check(
                        torch.equal(logits[:, 0], alt[:, 0]),
                        "past content affected first prediction",
                    )
                    scores["drop"][ix] = nll(alt, y).double().sum(1).numpy()
                    tv = (
                        0.5
                        * (logits.double().softmax(-1) - alt.double().softmax(-1))
                        .abs()
                        .sum(-1)[:, 1:]
                    )
                    scores["tv_sum"][ix] = tv.sum(1).numpy()
                    scores["changed"][ix] = (tv > 1e-6).sum(1).numpy()
                dt += monotonic() - t
                dc += process_time() - c
            steps += 1
            tick(BATCH * T)
        d.check(
            all(np.isfinite(v).all() for v in scores.values()),
            "missing or nonfinite document score",
        )
        means.append(float(scores["model"].sum() / (n * T)))
    counts = bank.counts if bank is not None else dict.fromkeys(COUNTERS, 0)
    nt = n * T * epochs
    d.check(
        counts["parent_reads"] == (0 if arm == "NONE" else nt)
        and counts["child_reads"] == 0,
        "incorrect read count",
    )
    cost = dict(
        chosen_wall_s=monotonic() - start - dt - at,
        chosen_owner_cpu_s=process_time() - c0 - dc - ac,
        audit_wall_s=at,
        audit_owner_cpu_s=ac,
        drop_wall_s=dt,
        drop_owner_cpu_s=dc,
    )
    return dict(
        targets=nt,
        batches=steps,
        documents=n,
        epochs=epochs,
        archive=dict(counts),
        cost=cost,
        current_answer_and_prior_sha=dig.hexdigest() if bank is not None else None,
        mean_nll_by_epoch=means,
    ), scores


def compare(losses, *, draws=10000, multiplicity=False):
    shape = losses["KEEP"].shape
    d.check(
        set(losses) == set(ARMS) | {"DROP", "ARCHIVE_PRIOR", "UNIGRAM"}
        and len(shape) == 2
        and shape[0] == 3
        and shape[1] > 0
        and all(v.shape == shape and np.isfinite(v).all() for v in losses.values()),
        "unpaired whole documents",
    )
    policies = {
        a: dict(mean=float(v.mean() / T), by_seed=(v.mean(1) / T).tolist())
        for a, v in losses.items()
    }
    delta = np.stack([(losses[b] - losses[a]) / T for a, b in PAIRS])
    avg = delta.mean(1)
    n = shape[1]
    rng = np.random.default_rng(5100)
    blocks = []
    for offset in range(0, draws, 250):
        sample = rng.integers(n, size=(min(250, draws - offset), n))
        if multiplicity:
            counts = np.stack([np.bincount(ix, minlength=n) for ix in sample])
            blocks.append(avg @ counts.T / n)
        else:
            blocks.append(avg[:, sample].mean(-1))
    qs = np.percentile(
        np.concatenate(blocks, axis=1), [2.5, 97.5], axis=1, method="linear"
    )
    contrasts = {}
    for i, (a, b) in enumerate(PAIRS):
        lo, hi = qs[:, i].tolist()
        by = delta[i].mean(-1).tolist()
        decision = (
            "win"
            if min(by) > 0 and lo > 0
            else "loss"
            if max(by) < 0 and hi < 0
            else "unresolved"
        )
        contrasts[a + ":" + b] = dict(
            gain=float(delta[i].mean()), ci95=[lo, hi], by_seed=by, decision=decision
        )
    return policies, contrasts


def report():
    r = d.RUN
    seal = d.frozen()
    d.check(
        d.read(r / "probe.json")["fit_seal"] == d.sha(r / "frozen.json"),
        "probe precedes freeze",
    )
    curves = {a: [] for a in (*ARMS, "DROP", "ARCHIVE_PRIOR", "UNIGRAM")}
    jobs = {}
    support = {}
    probe_read_sha = None
    for seed in d.SEEDS:
        stamps = {}
        priors = {}
        for arm in ARMS:
            key = f"{seed}-{arm}"
            f = d.read(r / (key + "-fit.json"))
            p = d.read(r / (key + "-probe.json"))
            path = r / (key + "-scores.npz")
            d.check(
                f["state_sha"] == p["state_sha"] == seal["states"][key]
                and f["fit_sha"] == seal["fit_sha"]
                and p["probe_sha"] == d.sha(r / "probe.json")
                and p["scores_sha"] == d.sha(path),
                "changed fit/probe journal",
            )
            phases = [f["facts"], p["facts"]]
            stamps[arm] = [v["current_answer_and_prior_sha"] for v in phases]
            jobs[key] = {}
            for label, facts, ep in zip(
                ("fit", "probe"), phases, (EPOCHS, 1), strict=True
            ):
                nt = d.N[label] * T * ep
                d.check(
                    facts["epochs"] == ep
                    and facts["targets"] == nt == facts["batches"] * BATCH * T
                    and facts["documents"] == d.N[label]
                    and facts["archive"]["child_reads"] == 0
                    and facts["archive"]["parent_reads"]
                    == (0 if arm == "NONE" else nt),
                    "wrong exposure",
                )
                jobs[key][label] = {
                    k: facts[k]
                    for k in (
                        "targets",
                        "batches",
                        "cost",
                        "archive",
                        "mean_nll_by_epoch",
                    )
                }
            with np.load(path, allow_pickle=False) as s:
                d.check(
                    all(
                        v.shape == (d.N["probe"],) and np.isfinite(v).all()
                        for v in s.values()
                    )
                    and float(s["model"].sum() / (d.N["probe"] * T))
                    == p["facts"]["mean_nll_by_epoch"][0],
                    "wrong document scores",
                )
                curves[arm].append(s["model"])
                priors[arm] = s["prior"]
                if arm == "KEEP":
                    curves["DROP"].append(s["drop"])
                    c = s["changed"]
                    tv = s["tv_sum"]
                    den = d.N["probe"] * (T - 1)
                    d.check(
                        np.all((c == np.floor(c)) & (c >= 0) & (c < T))
                        and np.all((tv >= 0) & (tv <= T - 1)),
                        "bad DROP support",
                    )
                    support[str(seed)] = dict(
                        eligible_later_positions=den,
                        changed_positions=int(c.sum()),
                        documents_changed=int((c > 0).sum()),
                        first_predictions_equal=d.N["probe"],
                        mean_tv_later=float(tv.sum() / den),
                        mean_tv_all=float(tv.sum() / (d.N["probe"] * T)),
                    )
        d.check(
            stamps["NONE"] == [None, None]
            and stamps["KEEP"] == stamps["DISCARD"]
            and all(stamps["KEEP"]),
            "mismatched current archive answers or priors",
        )
        if probe_read_sha is None:
            probe_read_sha = stamps["KEEP"][1]
        d.check(
            stamps["KEEP"][1] == probe_read_sha, "probe queries differ across seeds"
        )
        np.testing.assert_array_equal(priors["KEEP"], priors["DISCARD"])
        curves["ARCHIVE_PRIOR"].append(priors["KEEP"])
        curves["UNIGRAM"].append(priors["NONE"])
    for arm in ("ARCHIVE_PRIOR", "UNIGRAM"):
        for c in curves[arm][1:]:
            np.testing.assert_array_equal(c, curves[arm][0])
    losses = {k: np.stack(v) for k, v in curves.items()}
    policies, contrasts = compare(losses)
    _, independent = compare(losses, multiplicity=True)
    for k in contrasts:
        np.testing.assert_allclose(
            contrasts[k]["ci95"], independent[k]["ci95"], rtol=1e-12, atol=1e-15
        )
    return dict(
        spec=d.SPEC,
        registration=d.REG,
        qualification=(d.OUT / "qualified-ref").read_text().strip(),
        freeze_sha=d.sha(r / "frozen.json"),
        documents=d.N,
        seed_order=list(d.SEEDS),
        probe_targets_per_seed=T * d.N["probe"],
        policies=policies,
        contrasts=contrasts,
        jobs=jobs,
        retention_support=support,
        retention_benefit_and_reliance=all(
            contrasts[k]["decision"] == "win" for k in ("KEEP:DISCARD", "KEEP:DROP")
        ),
        current_read_result=contrasts["DISCARD:NONE"]["decision"],
        joint_bootstrap="whole paired documents, 10000 draws, RNG 5100, linear percentiles",
        multiplicity_crosscheck=True,
        identical_current_read_hashes=True,
        scope="fresh native TRAIN probe, no final/default/scale decision",
    )
