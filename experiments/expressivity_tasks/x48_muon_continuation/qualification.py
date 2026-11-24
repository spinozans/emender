"""Frozen CPU launch gate. Name every required node; never accept a passing test-count substring."""

import os
import re
import subprocess
import sys

from .data import OUT, ROOT, X47, check, sha, write_json

D = "experiments/expressivity_tasks/x48_muon_continuation"
OLD = X47 / "receipts/memory-program-s20/gate-nodes.txt"
NEW = (
    "test_select_and_cut_are_disjoint_deterministic_and_hash_checked",
    "test_full_prepared_stream_and_gate_no_wrap",
    "test_two_coupled_updates_serialize_every_optimizer_and_reproduce_next_transition[direct]",
    "test_two_coupled_updates_serialize_every_optimizer_and_reproduce_next_transition[program]",
    "test_schedule_bracket_tie_and_refuse_unfunded_launch",
    "test_real_origin_routing",
    "test_coupled_native_training_seam_synthetic_words_only[direct]",
    "test_coupled_native_training_seam_synthetic_words_only[program]",
    "test_registered_comparison_reports_post_election_and_refuses_a_changed_batch",
)


def nodes():
    n = (
        OLD.read_text().splitlines()
        + [f"{D}/test_continuation.py::{s}" for s in NEW]
        + [
            "tests/test_x45_muon_control.py::test_option_b_routing_mirrors_the_k10_decay_gate_exclusion",
            "tests/test_x45_optimizer_bakeoff.py::test_hybrid_muon_step_matches_pinned_convention_exactly",
        ]
    )
    check(len(n) == len(set(n)) == 243, "wrong or repeated gate nodes")
    return n


def proof(lines):
    ns = nodes()
    passed = [s.rsplit(" PASSED [", 1)[0] for s in lines if " PASSED [" in s]
    check(
        len(passed) == len(ns)
        and set(passed) == set(ns)
        and re.fullmatch(rf"=+ {len(ns)} passed in [\d.]+s =+", lines[-1]),
        "incomplete coupled continuation qualification",
    )
    return len(ns)


def main():
    os.chdir(ROOT)
    check(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("OMP_NUM_THREADS") == "4"
        and not os.environ.get("GPU_LEASE_HELD"),
        "CPU only",
    )
    ref = (OUT / "source-ref").read_text().strip()

    def start():
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    check(ref == start() and not (OUT / "gated-ref").exists(), "wrong qualification ref or repeated gate")
    check(
        not subprocess.check_output(["git", "status", "--porcelain", "--", D]), "unfrozen continuation source"
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "ndm",
            "experiments/expressivity_tasks",
            "scripts/gpu_lease.sh",
        ],
        check=True,
    )
    path = OUT / "qualification.txt"
    with path.open("x") as f:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--tb=short", "--color=no", *nodes()],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    passed = proof(path.read_text().splitlines())
    check(ref == start(), "changed ref during gates")
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "ndm",
            "experiments/expressivity_tasks",
            "scripts/gpu_lease.sh",
        ],
        check=True,
    )
    write_json(OUT / "qualification.json", dict(ref=ref, passed=passed, sha256=sha(path)))
    (OUT / "gated-ref").write_text(ref + "\n")
    print(f"{passed} frozen gates passed at {ref}", flush=True)


if __name__ == "__main__":
    main()
