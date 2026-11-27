"""SAL-3 close: re-derive the report from sealed artifacts, plus the
index-alone diagnostic.

Nothing here evaluates the probe twice, opens a new membership, or runs on a
lease. The close re-derives the contrast and codec baselines exactly as
close2.py does (same statistical functions, same sealed losses), and the
diagnostic prices the pure-interface model: the order-0 count mixture with
no host at all, evaluated on FIT targets only, under uniform depth weights
and under the frozen XSACOUNT fit's weights.
"""

import numpy as np

from . import sal3
from .data import load, get
from .run import frames, summarize
from .settings import cpu, sha
from ..window import need


def index_alone(role="fit", weights="uniform"):
    """Bits/original byte of the pure order-0 interface mixture on FIT.

    p(v | q) = (sum_l w_l h_q_l[v]) / (sum_l w_l m_q_l), uniform 1/256 when
    no class matches. `weights` is "uniform" or the frozen fit key
    ("5751.xsacount"), whose g vector gives w = exp(g).
    """
    sal3.rebind()
    cpu()
    import torch

    from .model import SalLM

    if weights == "uniform":
        w = np.ones(9, np.float64)
    else:
        seed, arm = weights.split(".")
        m = SalLM(int(seed))
        m.load_state_dict(
            torch.load(
                sal3.JOB3 / f"{weights}.pt", map_location="cpu", weights_only=True
            ),
            strict=True,
        )
        w = np.exp(m.g.detach().numpy().astype(np.float64))
    z = np.load(sal3.DATA3 / f"bound-{role}.npz", allow_pickle=False)
    n = get(sal3.DATA3 / f"bound-{role}.json")["batches"]
    total = 0.0
    targets = 0
    for i in range(n):
        m = z[f"b{i}_m"].astype(np.float64)
        h = z[f"b{i}_h"].astype(np.float64)
        y = z[f"b{i}_targets"]
        num = np.einsum("l,qlv->qv", w, h)
        den = np.einsum("l,ql->q", w, m)
        hit = den > 0
        p = np.full(len(y), 1.0 / 256.0, np.float64)
        p[hit] = num[hit, y[hit]] / den[hit]
        total += float(-np.log(p).sum())
        targets += len(y)
    return 8.0 * total / targets, targets


def close():
    sal3.rebind()
    from . import run

    run.N = sal3.N3
    cpu()
    J, D = sal3.JOB3, sal3.DATA3

    rows = get(D / "rows.json")
    need(
        sha(D / "rows.json") == get(D / "reservation.json")["rows_sha"]
        and (D / "probe-opened").exists(),
        "unbound roles",
    )
    need(
        len(rows["roles"]) == 2
        and {k: len(v) for k, v in rows["roles"].items()} == sal3.N3
        and rows["excluded"] == 328678
        and rows["supply"] == 531,
        "role shapes",
    )
    c = get(D / "container.json")
    need(
        c["container_sha256"] == sal3.CONTAINER_SHA
        and c["bound"] is True
        and c["rows"] == int(sum(c["h0"])),
        "container receipt",
    )
    build = None
    fit = load("fit")
    probe = load("probe")
    need(
        fit.shape == (sal3.N3["fit"], 4097) and probe.shape == (sal3.N3["probe"], 4097),
        "role arrays",
    )
    frozen = get(J / "frozen.json")
    need(
        get(J / "freeze-ack.json")["frozen_sha"] == sha(J / "frozen.json")
        and frozen["rows_sha"] == get(D / "reservation.json")["rows_sha"]
        and set(frozen["fits"])
        == {f"{s}.{a}" for s in (5751, 5752) for a in ("local", "xsacount")},
        "invalid freeze",
    )
    initials = {}
    for k, v in frozen["fits"].items():
        p = J / (k + ".json")
        z = get(p)
        need(
            sha(p) == v
            and sha(J / (k + ".pt")) == z["weights_sha"]
            and z["updates"] == 2048
            and z["targets"] == sal3.N3["fit"] * 4096,
            f"fit receipt {k}",
        )
        seed, arm = k.split(".")
        need(z["seed"] == int(seed) and z["arm"] == arm, "fit identity")
        initials.setdefault(seed, set()).add(z["initial"])
    need(all(len(v) == 1 for v in initials.values()), "unpaired starts")

    z = np.load(J / "document-losses.npz", allow_pickle=False)
    losses = {k: z[k] for k in z.files}
    complete = get(J / "complete.json")
    need(
        sha(J / "document-losses.npz") == complete["scores_sha"],
        "scores seal",
    )
    q = summarize(losses, ("local", "xsacount"))
    for arm in ("local", "xsacount"):
        need(
            q["mean_bpb"][arm] == complete["quality"]["mean_bpb"][arm]
            and q["per_seed_bpb"][arm] == complete["quality"]["per_seed_bpb"][arm],
            f"re-derived {arm} mismatch",
        )
    ct = q["contrasts"]["xsacount:local"]
    k = complete["quality"]["contrasts"]["xsacount:local"]
    need(
        ct["gain"] == k["gain"]
        and ct["ci"] == k["ci"]
        and ct["seeds"] == k["seeds"]
        and ct["positive"] == k["positive"],
        "re-derived contrast mismatch",
    )

    zc = frames(probe, build, dict_bytes=sal3.dict_bytes())
    for label, v in zc.items():
        need(v == complete["zstd"][label], f"zstd {label} mismatch")

    charges = sorted(
        f.name[len("charge-") : -len(".json")] for f in J.glob("charge-*.json")
    )
    ledger = [
        __import__("json").loads(line)
        for line in (sal3.B3 / "campaign.jsonl").read_text().splitlines()
        if line.strip()
    ]
    spent = sum(e["upper_card_s"] for e in ledger)
    need(
        {e["stage"] for e in ledger} == set(charges) and spent <= sal3.CAP3,
        "campaign ledger",
    )

    uni, n_t = index_alone("fit", "uniform")
    fit_w, _ = index_alone("fit", "5751.xsacount")
    result = dict(
        campaign="sal-3",
        design_point="100 GB external XSA index (2^30-class rows) vs 2,097,152 fit targets",
        arms=("local", "xsacount"),
        seeds=(5751, 5752),
        mean_bpb=q["mean_bpb"],
        per_seed_bpb=q["per_seed_bpb"],
        contrast=ct,
        zstd=zc,
        index_alone=dict(
            uniform_weights_bits_per_shifted_target=uni,
            fitted_g_weights_bits_per_shifted_target=fit_w,
            targets=n_t,
            note="the pure order-0 interface mixture on FIT targets, no host; prices the gap that index-stored moments (the law's ladder) would close",
        ),
        frozen_sha=sha(J / "frozen.json"),
        scores_sha=sha(J / "document-losses.npz"),
        rows_sha=sha(D / "rows.json"),
        container_sha256=c["container_sha256"],
        container_rows=c["rows"],
        charges={s: get(J / f"charge-{s}.json")["upper_card_s"] for s in charges},
        campaign_card_s=spent,
        complete_sha=sha(J / "complete.json"),
    )
    return result


def main():
    r = close()
    import json
    from pathlib import Path

    out = Path(__file__).resolve().parent / "receipts/sal3-result.json"
    out.write_text(json.dumps(r, sort_keys=True, indent=1) + "\n")
    print(json.dumps(r, sort_keys=True, indent=1))


if __name__ == "__main__":
    main()
