"""SAL-2 terminal close: the report is re-derived from sealed artifacts only.

No lease, no stage, no second evaluation. Everything here reads the frozen
fits, the witnessed freeze, the sealed per-document losses, the sealed data
roles, and the campaign ledger, and re-derives the published numbers with
the same statistical functions the driver used. Run on CPU:

    .venv/bin/python -m experiments.expressivity_tasks.retrieved_span_effects.\
        suffix_attention.close2
"""

import numpy as np

from . import sal2
from .run import frames, summarize
from .settings import cpu, sha
from ..window import get, need
from .data import load


def close():
    sal2.rebind()
    from . import run

    run.N = sal2.N2  # the driver's module constant stayed SAL-1's default;
    # the true geometry is re-checked below from the sealed role receipts.
    cpu()
    J, D = sal2.JOB2, sal2.DATA2

    # --- data and freeze seals -------------------------------------------------
    rows = get(D / "rows.json")
    need(
        sha(D / "rows.json") == get(D / "reservation.json")["rows_sha"]
        and get(D / "ready.json")
        == dict(rows=sha(D / "rows.json"), build=sha(D / "build.json"), fit=sha(D / "fit.json"))
        and (D / "probe-opened").exists(),
        "unbound roles",
    )
    need(
        len(rows["roles"]) == 3
        and {k: len(v) for k, v in rows["roles"].items()} == sal2.N2
        and rows["excluded"] == 65254
        and rows["archive"]["distinct"] == sal2.N2["build"],
        "role shapes",
    )
    build = load("build")
    fit = load("fit")
    probe = load("probe")
    need(
        build.shape == (sal2.N2["build"], 4097)
        and build.shape[0] * 4096 == 2**30
        and fit.shape == (sal2.N2["fit"], 4097)
        and probe.shape == (sal2.N2["probe"], 4097),
        "archive geometry (true documents=262144, rows=2^30; complete.json's "
        "archive.documents field carried the stale un-rebound constant 2048)",
    )
    frozen = get(J / "frozen.json")
    need(
        get(J / "freeze-ack.json")["frozen_sha"] == sha(J / "frozen.json")
        and frozen["rows_sha"] == get(D / "reservation.json")["rows_sha"]
        and set(frozen["fits"])
        == {f"{s}.{a}" for s in (5751, 5752) for a in ("local", "skeleton")},
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
            and z["targets"] == sal2.N2["fit"] * 4096,
            f"fit receipt {k}",
        )
        seed, arm = k.split(".")
        need(z["seed"] == int(seed) and z["arm"] == arm, "fit identity")
        initials.setdefault(seed, set()).add(z["initial"])
    need(all(len(v) == 1 for v in initials.values()), "unpaired starts")

    # --- sealed losses -> the report --------------------------------------------
    z = np.load(J / "document-losses.npz", allow_pickle=False)
    losses = {k: z[k] for k in z.files}
    complete = get(J / "complete.json")
    need(
        sha(J / "document-losses.npz") == complete["scores_sha"],
        "scores seal",
    )
    q = summarize(losses, ("local", "skeleton"))
    for arm in ("local", "skeleton"):
        need(
            q["mean_bpb"][arm] == complete["quality"]["mean_bpb"][arm]
            and q["per_seed_bpb"][arm] == complete["quality"]["per_seed_bpb"][arm],
            f"re-derived {arm} mismatch",
        )
    c = q["contrasts"]["skeleton:local"]
    k = complete["quality"]["contrasts"]["skeleton:local"]
    need(
        c["gain"] == k["gain"]
        and c["ci"] == k["ci"]
        and c["seeds"] == k["seeds"]
        and c["positive"] == k["positive"] is False,
        "re-derived contrast mismatch",
    )

    # --- codec baselines recomputed from the sealed arrays ----------------------
    zc = frames(probe, build)
    for label, v in zc.items():
        need(
            v == complete["zstd"][label],
            f"zstd {label} mismatch",
        )

    # --- campaign ledger ---------------------------------------------------------
    charges = {
        "run": get(J / "charge-run.json"),
        "run2": get(J / "charge-run2.json"),
    }
    need(
        charges["run"]["upper_card_s"] == 197
        and charges["run"]["rc"] == 1
        and charges["run2"]["upper_card_s"] == 64
        and charges["run2"]["rc"] == 1
        and all(charges[s]["release_verified"] for s in charges)
        and all(charges[s]["renewed_progress"] for s in charges),
        "stage charges",
    )
    ledger = [
        __import__("json").loads(line)
        for line in (sal2.B2 / "campaign.jsonl").read_text().splitlines()
        if line.strip()
    ]
    need(
        len(ledger) == 2
        and sum(e["upper_card_s"] for e in ledger) == 261
        and {e["stage"] for e in ledger} == {"run", "run2"},
        "campaign ledger",
    )
    need(
        (J / "watch-ack-run2").read_text().strip() == str(complete["units"]),
        "observer witness",
    )

    result = dict(
        campaign="sal-2",
        design_point="512:1 (2^30 archive rows to 2,097,152 fit targets)",
        arms=("local", "skeleton"),
        seeds=(5751, 5752),
        mean_bpb=q["mean_bpb"],
        per_seed_bpb=q["per_seed_bpb"],
        contrast=c,
        zstd=zc,
        frozen_sha=sha(J / "frozen.json"),
        scores_sha=sha(J / "document-losses.npz"),
        rows_sha=sha(D / "rows.json"),
        charges={s: v["upper_card_s"] for s, v in charges.items()},
        campaign_card_s=261,
        stage_notes=["attempt-run-note.txt", "attempt-run2-note.txt"],
        complete_sha=sha(J / "complete.json"),
    )
    return result


def main():
    r = close()
    import json
    from pathlib import Path

    out = Path(__file__).resolve().parent / "receipts/sal2-result.json"
    out.write_text(json.dumps(r, sort_keys=True, indent=1) + "\n")
    print(json.dumps(r, sort_keys=True, indent=1))


if __name__ == "__main__":
    main()
