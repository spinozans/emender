"""Matched causal programs, per-document receipts and exposure-weighted paired intervals."""

from collections import Counter
from hashlib import sha256
from time import monotonic, process_time

import numpy as np

from .reader import ARMS, PRE, REFRESH, INSIDE, reader, bands

BLOCK = 16
BANDS = {
    "all": (PRE, REFRESH, INSIDE),
    "refresh": (REFRESH,),
    "inside": (INSIDE,),
    "pre": (PRE,),
}
CONTRASTS = {
    "SPELL:FIRST": ("all", "inside"),
    "SPELL:LEX": ("all", "inside"),
    "LEX:FIRST": ("all", "inside"),
    "FIRST:BYTE": ("all", "refresh"),
    "SPELL:BYTE": ("all",),
}


def phase(atlas, windows, tick):
    n = len(windows)
    T = len(windows[0]) - 1
    if n % BLOCK or any(len(w) != T + 1 for w in windows):
        raise ValueError("unmatched dose")
    sizes = np.zeros((n, 3), np.int64)
    loss = np.zeros((n, len(ARMS), 3), np.float64)
    movement = np.zeros((n, 2), np.float64)
    models = {a: reader(a, atlas) for a in ARMS}
    counts = {a: Counter() for a in ARMS}
    seconds = {a: 0.0 for a in ARMS}
    cpu = {a: 0.0 for a in ARMS}
    audit_s = audit_cpu = 0.0
    stamp = sha256()
    for start in range(0, n, BLOCK):
        first = (start // BLOCK) % len(ARMS)
        order = ARMS[first:] + ARMS[:first]
        outputs = {}
        for arm in order:
            wall, ct = monotonic(), process_time()
            block = []
            for w in windows[start : start + BLOCK]:
                pred = models[arm].predict(w[:-1])
                y = np.frombuffer(w, np.uint8, offset=1)
                ls = -np.log(pred.pdf[np.arange(T), y])
                counts[arm].update(pred.counts)
                block.append((pred, ls))
            outputs[arm] = block
            seconds[arm] += monotonic() - wall
            cpu[arm] += process_time() - ct
        wall, ct = monotonic(), process_time()
        for j, w in enumerate(windows[start : start + BLOCK]):
            tag = bands(w[:-1])
            idx = start + j
            sizes[idx] = np.bincount(tag, minlength=3)
            on = tag == REFRESH
            gap = tag == INSIDE
            raw, one, lex, full = (outputs[a][j][0] for a in ARMS)
            if not (
                raw.route is None
                and np.array_equal(one.route, full.route)
                and np.array_equal(lex.route, full.route)
                and np.array_equal(on, full.route[:, 0] >= 0)
                and np.array_equal(lex.pdf[on], one.pdf[on])
                and np.array_equal(full.pdf[on], one.pdf[on])
                and np.array_equal(raw.pdf[~on], one.pdf[~on])
                and np.array_equal(raw.pdf[tag == PRE], lex.pdf[tag == PRE])
                and np.array_equal(raw.pdf[tag == PRE], full.pdf[tag == PRE])
            ):
                raise ValueError("mismatched route, first forecast or no-read control")
            for k, a in enumerate(ARMS):
                pred, ls = outputs[a][j]
                p = pred.pdf
                if not (
                    np.all(p > 0)
                    and np.isfinite(ls).all()
                    and np.allclose(p.sum(1), 1, rtol=0, atol=3e-15)
                ):
                    raise ValueError("invalid all-byte forecast")
                loss[idx, k] = np.bincount(tag, weights=ls, minlength=3)
            tv = 0.5 * np.abs(full.pdf[gap] - lex.pdf[gap]).sum(1)
            movement[idx] = np.count_nonzero(tv > 1e-12), tv.sum()
            stamp.update(full.route.tobytes())
            stamp.update(full.pdf[on].tobytes())
        tick(start + BLOCK)
        audit_s += monotonic() - wall
        audit_cpu += process_time() - ct
    facts = dict(
        windows=n,
        targets=n * T,
        same_routes_and_refresh_forecasts=True,
        checked_sha=stamp.hexdigest(),
        cost={a: dict(chosen_s=seconds[a], chosen_cpu_s=cpu[a]) for a in ARMS},
        counters={a: dict(v) for a, v in counts.items()},
        audit_s=audit_s,
        audit_cpu_s=audit_cpu,
    )
    return facts, dict(sizes=sizes, loss=loss, movement=movement)


def summarize(scores, facts, *, draws=10000, use_counts=True):
    sizes = scores["sizes"]
    loss = scores["loss"]
    m = scores["movement"]
    n, features = sizes.shape
    if (
        features != 3
        or n != facts["windows"]
        or sizes.sum() != facts["targets"]
        or np.any(sizes < 0)
        or loss.shape != (n, len(ARMS), 3)
        or not np.isfinite(loss).all()
        or (loss < 0).any()
        or m.shape != (n, 2)
        or not np.isfinite(m).all()
        or (m < 0).any()
        or not np.all(m <= sizes[:, INSIDE, None])
        or not np.all(m[:, 0] == np.floor(m[:, 0]))
    ):
        raise ValueError("invalid paired receipts")
    ns = {b: sizes[:, list(ix)].sum(-1) for b, ix in BANDS.items()}
    ys = {
        a: {b: loss[:, k, :][:, list(ix)].sum(-1) for b, ix in BANDS.items()}
        for k, a in enumerate(ARMS)
    }
    if (
        not np.array_equal(ys["SPELL"]["refresh"], ys["FIRST"]["refresh"])
        or not np.array_equal(ys["LEX"]["refresh"], ys["FIRST"]["refresh"])
        or not np.array_equal(ys["FIRST"]["inside"], ys["BYTE"]["inside"])
        or not all(np.array_equal(ys[a]["pre"], ys["BYTE"]["pre"]) for a in ARMS)
    ):
        raise ValueError("common predictions do not reconcile")
    absolute = {
        a: {
            b: (float(v.sum() / ns[b].sum()) if ns[b].sum() else None)
            for b, v in bands_.items()
        }
        for a, bands_ in ys.items()
    }
    labels = []
    diff = []
    den = []
    for key, bb in CONTRASTS.items():
        a, c = key.split(":")
        for b in bb:
            labels.append((key, b))
            diff.append(ys[c][b] - ys[a][b])
            den.append(ns[b])
    diff = np.stack(diff, 1)
    den = np.stack(den, 1)
    rng = np.random.default_rng(5300)
    res = []
    for off in range(0, draws, 96):
        ix = rng.integers(n, size=(min(96, draws - off), n))
        if use_counts:
            counts = np.stack([np.bincount(r, minlength=n) for r in ix])
            sums = counts @ diff
            total = counts @ den
        else:
            sums = diff[ix].sum(1)
            total = den[ix].sum(1)
        res.append(
            np.divide(sums, total, out=np.full_like(sums, np.nan), where=total != 0)
        )
    res = np.vstack(res)
    contrasts = {}
    for i, (k, b) in enumerate(labels):
        mask = np.isfinite(res[:, i])
        gain = float(diff[:, i].sum() / den[:, i].sum()) if den[:, i].sum() else None
        ci = (
            np.quantile(res[:, i], [0.025, 0.975], method="linear").tolist()
            if mask.all()
            else None
        )
        decision = (
            "win" if ci and ci[0] > 0 else "loss" if ci and ci[1] < 0 else "unresolved"
        )
        contrasts.setdefault(k, {})[b] = dict(
            gain=gain, ci95=ci, decision=decision, valid_draws=int(mask.sum())
        )
    return dict(
        absolute=absolute,
        contrasts=contrasts,
        targets={b: int(s.sum()) for b, s in ns.items()},
        spelling_win=all(
            contrasts["SPELL:FIRST"][b]["decision"] == "win" for b in ("all", "inside")
        ),
        contextual_tail_win=all(
            contrasts["SPELL:LEX"][b]["decision"] == "win" for b in ("all", "inside")
        ),
        changed_gap_predictions=int(m[:, 0].sum()),
        docs_with_changed_gap=int((m[:, 0] > 0).sum()),
        mean_gap_tv=float(m[:, 1].sum() / max(1, ns["inside"].sum())),
        seconds_per_million={
            a: facts["cost"][a]["chosen_s"] * 1e6 / facts["targets"] for a in ARMS
        },
    )


def check_work(facts, targets):
    read = targets["refresh"]
    gaps = targets["inside"]
    c = facts["counters"]

    def get(a, k):
        return c[a].get(k, 0)

    if (
        c["BYTE"]
        or facts["targets"] != targets["all"]
        or not facts["same_routes_and_refresh_forecasts"]
    ):
        raise ValueError("bad no-read or comparison receipt")
    for k in (
        "reads",
        "root_reads",
        "nonroot_misses",
        "nonroot_empty",
        "suffix_length_sum",
        "commits",
        "known_commits",
        "logical_entries",
    ):
        if len({get(a, k) for a in ARMS[1:]}) != 1:
            raise ValueError("reading arms did not navigate identically")
    for a in ARMS[1:]:
        root = get(a, "root_reads")
        entries = get(a, "returned_entries")
        width = 8 if a == "SPELL" else 4
        if (
            get(a, "reads") != read
            or get(a, "wire_bytes") != read + entries * width
            or (a != "SPELL" and entries != 256 * (read - root))
            or not 0 <= get(a, "known_commits") <= get(a, "commits") <= read
            or not 0 <= get(a, "nonroot_misses") <= read - root
        ):
            raise ValueError("bad request or wire count")
    if (
        any(
            get("LEX", k) != get("SPELL", k)
            for k in ("lex_supported_gap", "unsupported_gap")
        )
        or get("SPELL", "lex_supported_gap") + get("SPELL", "unsupported_gap") != gaps
        or get("SPELL", "context_supported_gap") > get("SPELL", "lex_supported_gap")
    ):
        raise ValueError("bad gap support")
