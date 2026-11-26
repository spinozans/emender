"""Paired inference from compact *private* per-document losses; never reopens text."""

from pathlib import Path

import numpy as np

from .natural_model import ARMS, HOSTS, SEEDS
from .window import get, need


BANDS = ("all", "eligible", "hit", "no_exact8")


def cases(arm):
    out = ["time.intact", "full.intact"]
    if arm != "local":
        out += ["full.drop", "full.wrong_value"]
    if arm.endswith("effect"):
        out += ["full.wrong_effect"]
    return out


def cell(host, arm, seed):
    return f"{host}-{arm}-{seed}"


def aggregate(folder, draws=10000):
    d = Path(folder)
    runs = {}
    ns = None
    for h in HOSTS:
        for arm in ARMS:
            for s in SEEDS:
                f = d / "scores" / (cell(h, arm, s) + ".npz")
                with np.load(f, allow_pickle=False) as data:
                    if ns is None:
                        ns = data["counts"]
                    need(
                        np.array_equal(data["counts"], ns)
                        and set(data.files) == {"counts", *cases(arm)},
                        "unpaired score bands",
                    )
                    for c in cases(arm):
                        x = data[c]
                        need(
                            x.shape == ns.shape
                            and np.isfinite(x).all()
                            and (x >= 0).all(),
                            "bad score receipt",
                        )
                        runs[h, arm, s, c] = x
    need(
        ns.shape == (1024, len(BANDS))
        and np.all(ns[:, 0] > 0)
        and np.all(ns[:, 3] <= ns[:, 2])
        and np.all(ns[:, 2] <= ns[:, 1])
        and np.all(ns[:, 1] <= ns[:, 0]),
        "invalid document counts",
    )
    tot = ns.sum(0)
    cols = np.flatnonzero(tot > 0)
    rng = np.random.default_rng(5410)
    # Reuse the exact paired multiplicities for every seed/arm and recompute target denominators.
    counts = rng.multinomial(len(ns), np.full(len(ns), 1 / len(ns)), size=draws).astype(
        np.float64
    )
    denom = counts @ ns[:, cols]
    need((denom > 0).all(), "empty bootstrap band")
    results = {}
    clocks = {}
    absolute = {}
    for h in HOSTS:

        def compare(a, ac, b, bc):
            ds = np.stack([runs[h, b, s, bc] - runs[h, a, s, ac] for s in SEEDS])
            avg = ds.mean(0)
            boot = (counts @ avg[:, cols]) / denom
            ci = np.quantile(boot, [0.025, 0.975], axis=0)
            signs = ds.sum(1)[:, cols] / tot[cols]
            overall = avg.sum(0)[cols] / tot[cols]
            key = f"{h}/{a}/{ac}:{b}/{bc}"
            results[key] = {
                BANDS[k]: dict(
                    gain=float(overall[j]),
                    ci=ci[:, j].tolist(),
                    seed_gains=signs[:, j].tolist(),
                    positive=bool(ci[0, j] > 0 and (signs[:, j] > 0).all()),
                )
                for j, k in enumerate(cols)
            }

        for arm in ARMS:
            for kind in ("time.intact", "full.intact"):
                y = np.stack([runs[h, arm, s, kind] for s in SEEDS])
                absolute[f"{h}/{arm}/{kind}"] = {
                    BANDS[k]: float(y.mean(0).sum(0)[k] / tot[k]) for k in cols
                }
                if arm != "local":
                    compare(arm, kind, "local", kind)
            if arm.endswith("effect") and not arm.startswith("mix"):
                for ref in ("state", "neutral"):
                    compare(arm, "full.intact", arm[0] + "-" + ref, "full.intact")
            if arm == "mix-effect":
                for ref in ("g-effect", "e-effect"):
                    compare(arm, "full.intact", ref, "full.intact")
            for case in cases(arm)[2:]:
                compare(arm, "full.intact", arm, case)
            clocks[h + "/" + arm] = [
                get(d / "fits" / cell(h, arm, s) / "fit.json")["points"] for s in SEEDS
            ]
    return dict(
        scope="fresh TRAIN probes, token-native from-scratch recurrent cores; not checkpoint confirmation",
        seeds=SEEDS,
        targets=dict(zip(BANDS, tot.tolist(), strict=True)),
        documents=len(ns),
        bootstrap_draws=draws,
        absolute=absolute,
        contrasts=results,
        training_points=clocks,
    )
