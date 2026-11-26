"""Bounded single-thread CPU training and sealed repeated-value probes. No GPU option."""

import argparse
from math import isfinite, log
from pathlib import Path
import time

import torch

from ..semantic_tail.editions import ARMS, SEEDS, TOKENIZER, K, digest
from ..semantic_tail.measure import view
from ..semantic_tail.run import (
    check_cpu,
    make_model,
    read,
    save,
    sha,
    state_sha,
    verified as original_models,
)
from .worlds import fits, order, packet, panels
from .decompose import accumulate, finish, polarity, Stats

REGISTERED = "bd9c1c2643e62ed2c7424b464a5b09ce27831098"
OLD_REF = "1d0f5a4615d26da1aa51d9b31774e49a7d7037d1"
OLD_FREEZE = "6843161006903a7e12923b1177aeb0aff06b310fff110b5cecf570b5311bb838"
COHORTS = ("permutation", "replacement")


def fit(out, ref):
    out.mkdir(mode=0o700, exist_ok=False)
    eds = fits()
    template = make_model(SEEDS[0])
    packets = {use: [packet(template, p, use) for p in eds] for use in (False, True)}
    receipts = {}
    for seed in SEEDS:
        seq = order(seed)
        if sorted(seq) != list(range(1024)):
            raise ValueError("invalid replacement training schedule")
        for arm in ARMS:
            key = f"{seed}.{arm}"
            m = make_model(seed).train()
            initial = state_sha(m)
            opt = torch.optim.AdamW(
                m.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=0.01
            )
            law = "indexed" if arm == "indexed" else "dense"
            targets = 0
            start = time.perf_counter()
            for i in seq:
                w = packets[arm != "local"][i]
                opt.zero_grad(set_to_none=True)
                loss = w.backward_nll(m, mode=law)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), 1.0, error_if_nonfinite=True
                )
                opt.step()
                targets += len(w.targets)
            secs = time.perf_counter() - start
            if targets != 16384:
                raise ValueError("unequal dose")
            f = out / (key + ".pt")
            with f.open("xb") as fd:
                torch.save(m.state_dict(), fd)
            meta = dict(
                seed=seed,
                arm=arm,
                initial=initial,
                final=state_sha(m),
                weights_sha256=sha(f),
                order_sha256=digest(seq),
                updates=len(seq),
                targets=targets,
                fit_loop_s=secs,
            )
            side = out / (key + ".json")
            save(side, meta)
            receipts[key] = sha(side)
            print("closed replacement", key, flush=True)
    save(
        out / "frozen.json",
        dict(
            registered=REGISTERED,
            qualified=ref,
            old_freeze=OLD_FREEZE,
            fit_maps_sha256=digest(eds),
            tokenizer=TOKENIZER,
            fits=receipts,
        ),
    )


def load_all(out, old, ref, binding):
    f = out / "frozen.json"
    if sha(f) != binding:
        raise ValueError("unwitnessed replacement freeze")
    z = read(f)
    if (
        z["registered"],
        z["qualified"],
        z["old_freeze"],
        z["fit_maps_sha256"],
        z["tokenizer"],
    ) != (REGISTERED, ref, OLD_FREEZE, digest(fits()), TOKENIZER) or set(z["fits"]) != {
        f"{s}.{a}" for s in SEEDS for a in ARMS
    }:
        raise ValueError("foreign/incomplete freeze")
    # Reads only training files; never opens the original probe or evaluates its worlds.
    prior, old_meta = original_models(old, OLD_REF, OLD_FREEZE)
    ms = {("permutation", s, a): m for (s, a), m in prior.items()}
    meta = {}
    for s in SEEDS:
        seq_sha = digest(order(s))
        for a in ARMS:
            name = f"{s}.{a}"
            j = out / (name + ".json")
            f = out / (name + ".pt")
            if sha(j) != z["fits"][name]:
                raise ValueError("modified sidecar")
            t = read(j)
            m = make_model(s)
            if (
                t["initial"] != old_meta[name]["initial"]
                or t["initial"] != state_sha(m)
                or sha(f) != t["weights_sha256"]
                or (t["seed"], t["arm"], t["targets"], t["updates"], t["order_sha256"])
                != (s, a, 16384, 1024, seq_sha)
            ):
                raise ValueError("failed identity/dose receipt")
            m.load_state_dict(
                torch.load(f, map_location="cpu", weights_only=True), strict=True
            )
            if state_sha(m) != t["final"] or not all(
                v.isfinite().all() for v in m.state_dict().values()
            ):
                raise ValueError("foreign or nonfinite weights")
            ms["replacement", s, a] = m.eval().requires_grad_(False)
            meta[name] = t
    return ms, meta


def verdict(metrics):
    primary = {}
    for c in COHORTS:
        for a in ("dense", "indexed"):
            gain = []
            wins = []
            for s in SEEDS:
                r = metrics[f"{c}.{s}.{a}"]
                gain.append(metrics[f"{c}.{s}.local"]["answer_nll"] - r["answer_nll"])
                wins.append(
                    gain[-1] > 0
                    and r["single.both_correct"] > 0.5
                    and r["single.gain_vs_stale"] > 0
                    and r["swap.both_correct"] > 0.5
                )
            primary[c + "." + a] = dict(
                gains_vs_local=gain, per_seed=wins, passes=all(wins)
            )
    return primary


@torch.no_grad()
def probe(out, old, ref, binding):
    models, meta = load_all(out, old, ref, binding)
    save(
        out / "probe-opened.json",
        dict(old=OLD_FREEZE, replacement=binding, checked=len(models)),
    )
    groups = panels()
    t = {k: Stats() for k in models}
    kap = {k: polarity(m) for k, m in models.items()}
    local = {}
    weights = {}
    for keys, eds in groups:
        template = next(iter(models.values()))
        books = [packet(template, p) for p in eds]
        for key, m in models.items():
            _, _, arm = key
            if arm == "local":
                if key not in local:
                    pred = view(packet(template, eds[0], False), m, "dense")
                    local[key] = pred.logp[1::2]
                    weights[key] = torch.zeros(K, K, dtype=torch.float64)
                logp = local[key][list(keys)][None, None].expand(K, K, -1, -1)
            else:
                prob = []
                for w in books:
                    pred = view(w, m, arm)
                    prob.append(pred.logp[1::2][list(keys)])
                    g = pred.shares[1::2, 0, None] * pred.rows
                    if key not in weights:
                        weights[key] = g
                    elif not torch.equal(g, weights[key]):
                        raise ValueError("successor change altered address mass")
                logp = torch.stack(prob).reshape(K, K, 2, -1)
            accumulate(t[key], keys, logp, weights[key], kap[key], arm != "local")
    values = {f"{c}.{s}.{a}": finish(t[c, s, a], kap[c, s, a]) for c, s, a in models}
    for r in values.values():
        if (
            not all(map(isfinite, r.values()))
            or r["clamped_acc"] > 1 / K + 1e-12
            or r["clamped_nll"] < log(K) - 1e-8
        ):
            raise ValueError("invalid clamped control on balanced alternatives")
    save(
        out / "result.json",
        dict(
            scope="independent and same-bag edits on fresh repeated-value invented archives",
            registered=REGISTERED,
            qualified=ref,
            tokenizer=TOKENIZER,
            original_ref=OLD_REF,
            original_freeze=OLD_FREEZE,
            replacement_freeze=binding,
            fit_maps_sha256=digest(fits()),
            panel_maps_sha256=digest(groups),
            counts_per_fit=dict(
                panels=len(groups),
                editions=64 * len(groups),
                active_answers=3584,
                directed_single_edits=25088,
                directed_same_bag_swaps=3136,
                exact_refined_pairs=0,
            ),
            replacement_fits=meta,
            metrics=values,
            primary=verdict(values),
            additional_gpu_seconds=0,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "probe"))
    p.add_argument("out", type=Path)
    p.add_argument("--public-ref", required=True)
    p.add_argument("--original", type=Path)
    p.add_argument("--freeze-sha")
    a = p.parse_args()
    check_cpu(a.public_ref)
    if a.phase == "fit" and a.original is None and a.freeze_sha is None:
        fit(a.out, a.public_ref)
    elif a.phase == "probe" and a.original and a.freeze_sha:
        probe(a.out, a.original, a.public_ref, a.freeze_sha)
    else:
        p.error("only probe requires --original and --freeze-sha")
