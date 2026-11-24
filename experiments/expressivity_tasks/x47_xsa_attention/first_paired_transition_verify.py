"""T13: compare a NEW candidate to sealed tensors-by-hash, not to a rerun or synthetic reference."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .paired_frontier_partial import partial
from .paired_frontier_verify import check_cell, sig, ROOT as SEALED
from .serving_verify import HERE, load, require

ROOT = HERE / "receipts/memory-program-t13"
PREV = "454aea719013a269cce8faef8ba2e02e0a414a02"
SEAL = "a01262f345013043c51a2b4309b3f05cb2afdd87ae3bdc9c22f4a13a4e668c65"
PROTO = "f6c17164a497d7d2aff7f5c67dd0a30686192762"
DELTA = [
    "first_paired_transition_bench.py",
    "first_paired_transition_verify.py",
    "paired_frontier_partial.py",
    "paired_frontier_verify.py",
    "test_paired_receipt.py",
]
IMMUTABLE = [
    "ndm",
    "scripts",
    *[
        f"experiments/expressivity_tasks/x47_xsa_attention/{n}.py"
        for n in (
            "paired_frontier_bench",
            "paired_frontier_fixture",
            "scoped_records",
            "private_tape",
            "branch_domain_bench",
            "memory_program_mounts",
            "direct_control_bench",
        )
    ],
    "tests/test_memory_program_paired_frontier.py",
]


def seal():
    p = SEALED / "partial.json"
    require(
        sha256(p.read_bytes()).hexdigest() == SEAL and load(p) == partial(SEALED),
        "original verified cell or receipts changed",
    )
    return load(p)


def audit(root=ROOT):
    old = seal()
    source = (root / "gated-ref").read_text().strip()
    d = load(root / "driver.json")
    end = load(root / "complete.json")
    checked = load(root / "source-check.json")
    require(
        source == (root / "source-ref").read_text().strip() == d["ref"] == end["ref"] != PREV
        and (root / "protocol-ref").read_text().strip() == PROTO
        and d["order"] == ["pair-A"]
        and d["pid"] == end["pid"]
        and d["card_s"] == end["card_s"] == 0
        and d["pid"] != load(SEALED / "driver.json")["pid"]
        and old["cell_epochs"][-1] < d["start_epoch_ns"]
        and 0 < end["wall_ns"] <= 120e9
        and checked["old"] == PREV
        and checked["new"] == source
        and checked["changed_code"] == DELTA
        and set(checked["git_objects"]) == set(IMMUTABLE)
        and all(len(v) == 2 and v[0] == v[1] and len(v[0]) == 40 for v in checked["git_objects"].values())
        and "196 passed in" in (root / "qualification.txt").read_text().splitlines()[-1]
        and "1 passed in" in (root / "receipt-qualification.txt").read_text().splitlines()[-1],
        "incorrect ownership, build or qualification",
    )
    a, _, old_actor, _ = check_cell(SEALED, "separate-A", PREV)
    b, metric, actors, epochs = check_cell(root, "pair-A", source)
    require(
        not actors & old_actor
        and sig(a) == old["signature"]
        and d["start_epoch_ns"] < epochs[0] < epochs[1] < end["end_epoch_ns"],
        "different historical oracle or overlapping actors",
    )
    diffs = []
    reports = []
    for n in range(2):
        x, y = (load(p / f"batch-{n}.json") for p in (SEALED / "separate-A", root / "pair-A"))
        # check_cell recomputes both complete source-specific authority digests; never overwrite either ref.
        require(
            x["authority"] != y["authority"] and x["ix"] == y["ix"],
            "not two independently identified admissions",
        )
        row = {}
        score = {}
        for k in (
            "ix",
            "feature_sha",
            "before",
            "before_params",
            "before_moments",
            "selector_after",
            "selectors",
            "values",
            "update",
        ):
            if x[k] != y[k]:
                diffs.append([n, k])
        for role in ("selectors", "values"):
            score[role] = {
                k: sum(i[k] != j[k] for i, j in zip(x[role]["rows"], y[role]["rows"], strict=True))
                for k in ("choices", "events", "bytes", "joint", "argmax")
            }
        for k in ("params", "grads", "moments"):
            row[k] = [name for name in x["update"][k] if x["update"][k][name] != y["update"][k][name]]
        if a[n][1:] != b[n][1:]:
            diffs.append([n, "logical dispatch/physical work per role"])
        reports.append(
            dict(
                server_independent_authorities=[x["authority"], y["authority"]],
                differing_docs=score,
                differing_tensors=row,
                reference_signature=a[n][0],
                candidate_signature=b[n][0],
                value_after=y["update"]["after"],
            )
        )
    baseline = old["metrics"]
    actual = dict(
        lf_steps=metric["archive"]["lf_steps"],
        live_starts=metric["live"]["native_starts"],
        live_appends=metric["live"]["native_appends"],
    )
    costs = (
        baseline["archive"]["lf_steps"],
        baseline["live"]["native_starts"],
        baseline["live"]["native_appends"],
    )
    once = all(v > 0 and 2 * v == z for v, z in zip(actual.values(), costs, strict=True))
    return dict(
        ref=source,
        sealed_ref=PREV,
        sealed_sha=SEAL,
        first_pair_parity=not diffs and once,
        unmatched=diffs,
        same_two_value_transitions=not diffs,
        one_set_of_native_work=once,
        actual_candidate=actual,
        distinct_train_positions=4096,
        new_selector_scores=4096,
        new_value_gradient_targets=4096,
        new_updates=2,
        new_m20_exits=2,
        new_positional_exits=2,
        new_live_exits=32,
        card_s=0,
        cumulative_x47_upper=1429,
        remaining_s=371,
        comparisons=reports,
        metrics=metric,
        four_cell_equivalence=None,
        repeatability=None,
        price_recommendation=None,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
