"""Pure trainer-wiring / metric / verifier tests (M2-review phase-4
corrections; NO CUDA required — the re-review's "pure loss/metric tests for
both stages" + verifier-only refusal tests + wiring identity tests).

Covers, per the M2-review correction list:

  * F1  — stage-1/2 label mapping: content labels -> contiguous zero-based
          classifier indices for the loss; reverse-mapped predictions for
          the metric (both stages).
  * F5  — the length-aware microbatch plan (bounds, over-plan singles) and
          the effective-batch summed-loss accumulation gradient equivalence
          (microbatched accumulation == a single batch-64 step, on a tiny
          CPU model); the allocation model vs the rung for every arm's
          bound budgets.
  * F7  — within-word peak / at-EOS occupancy measurement and the per-k
          (keyed-by-depth) occupancy aggregation.
  * F8  — determinism coverage: the sweep union derived from all three
          bounded configs (51 points) + the other registered pools = 66
          pools; the degenerate-band mismatch rule verified by construction.
  * F9  — hash-validated receipts: tamper detection.
  * F10 — pop-off single construction: tape_appendonly and tape_unbounded
          share IDENTICAL non-pop initial parameters at the same seed.
  * F2  — verifier-only: wrong hash refuses (exit 3); a committed mirror
          with the right hash verifies and EXITS WITHOUT constructing a
          trainer (receipted ``torch_imported: false``); an untracked file
          inside the package refuses (the finding-3 scenario); structural
          AST check that the verify-only path constructs no trainer.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import trainer  # noqa: E402
import launcher  # noqa: E402
from generators.check_determinism import (  # noqa: E402
    mismatch_rule_check, sweep_points,
)
from generators.stage1_mqar import mean_per_class_recall  # noqa: E402
from generators.stage3_ladder import ACCEPT, ladder_sweep_pool  # noqa: E402

REL_PREFIX = PACKAGE_ROOT.relative_to(PACKAGE_ROOT.parents[2]).as_posix()


# ---------------------------------------------------------------------------
# F1 — stage-1/2 label mapping (pure loss/metric tests, both stages).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage,offset,n_classes,vocab", [
    ("stage1", trainer.STAGE1_LABEL_OFFSET, 64, 65),
    ("stage2", trainer.STAGE2_LABEL_OFFSET, 16, 18),
])
def test_label_mapping_loss_and_metric_both_stages(stage, offset, n_classes, vocab):
    torch.manual_seed(0)
    B, T = 3, 5
    # content labels in the stage's registered range (stage1: 1..64, stage2: 2..17)
    lo, hi = offset, offset + n_classes - 1
    y = torch.randint(lo, hi + 1, (B, T))
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, -2:] = True
    logits = torch.randn(B, T, n_classes, requires_grad=True)

    # the loss the trainer computes: cross-entropy on ZERO-BASED indices
    y_idx = torch.from_numpy(
        trainer.labels_to_classifier_indices(y.numpy(), offset))
    flat_m = mask.reshape(-1)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, n_classes)[flat_m],
        y_idx.reshape(-1)[flat_m], reduction="sum") / int(mask.sum())
    loss.backward()

    # equivalence: identical to a directly-shifted-target cross-entropy
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, n_classes)[flat_m],
        (y.reshape(-1)[flat_m] - offset), reduction="sum") / int(mask.sum())
    assert torch.allclose(loss, ref)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    # the metric the trainer reports: predictions reverse-mapped to content
    # labels, scored against the RAW content labels over the content classes
    pred_idx = logits.detach().reshape(-1, n_classes).argmax(-1).numpy()
    preds = trainer.classifier_indices_to_labels(pred_idx, offset).reshape(B, T)
    classes = np.arange(lo, hi + 1)
    metric = mean_per_class_recall(preds[mask.numpy()], y[mask].numpy(), classes)
    perfect = mean_per_class_recall(y[mask].numpy(), y[mask].numpy(), classes)
    assert 0.0 <= metric <= 1.0
    assert metric == perfect or metric < perfect
    # an all-(lo) prediction scores exactly 1/(# present classes) under the
    # registered convention (mean per-class recall over classes PRESENT in
    # the true labels - the standard balanced-accuracy convention the
    # registration names; out-of-range labels would make the metric diverge
    # from this bound). Controlled construction: y masked rows contain
    # exactly the classes [lo, lo+1, ..., lo+5], so an all-(lo) prediction
    # scores 1/6.
    y_ctrl = torch.tensor([lo + i for i in range(6)])
    ctrl_classes = np.arange(lo, lo + 6)
    assert mean_per_class_recall(
        np.full(6, lo), y_ctrl.reshape(-1).numpy(),
        ctrl_classes) == pytest.approx(1.0 / 6)
    assert mean_per_class_recall(
        y_ctrl.reshape(-1).numpy(), y_ctrl.reshape(-1).numpy(),
        ctrl_classes) == pytest.approx(1.0)

    # mapping round-trips and never leaves the classifier range
    assert (trainer.labels_to_classifier_indices(
        np.arange(lo, hi + 1), offset) == np.arange(n_classes)).all()
    back = trainer.classifier_indices_to_labels(
        trainer.labels_to_classifier_indices(
            np.array([lo, hi]), offset), offset)
    assert list(back) == [lo, hi]


def test_label_mapping_rejects_out_of_range_labels():
    # cross-entropy itself fails closed on an out-of-range target: the
    # mapping must be applied BEFORE the loss (the F1 bug's signature:
    # stage-1 label 64 in a 64-logit head, stage-2 label 17 in a 16-logit head)
    with pytest.raises(IndexError):
        torch.nn.functional.cross_entropy(
            torch.randn(2, 64), torch.full((2,), 64, dtype=torch.long))


# ---------------------------------------------------------------------------
# F5 — microbatch plan + effective-batch accumulation equivalence.
# ---------------------------------------------------------------------------
def test_microbatch_plan_bounds_and_over_plan_singles():
    lengths = np.array([10, 10, 10, 200, 10])
    mbs = trainer.microbatch_plan(lengths, token_budget=30, max_sequences=3)
    assert [len(mb) for mb in mbs] == [3, 1, 1]
    assert mbs[0] == [0, 1, 2] and mbs[1] == [3] and mbs[2] == [4]
    for mb in mbs:
        assert len(mb) <= 3
        assert len(mb) * int(lengths[mb].max()) <= 30 or len(mb) == 1
    # a single example larger than the whole budget forms its own microbatch
    assert trainer.microbatch_plan([1000], 100, 4) == [[0]]
    # padded-product semantics: max length in the microbatch is what counts
    mbs = trainer.microbatch_plan([5, 100, 5], token_budget=100, max_sequences=8)
    assert mbs == [[0], [1], [2]]   # 2*100 > 100: the long example splits the batch


def test_effective_batch_summed_loss_gradient_equivalence():
    """Microbatched summed-loss accumulation == a single effective-batch step
    (the F5 registered update): identical gradients (up to fp associativity)
    for the multiclass masked-CE normalization and the stage-3 summed BCE."""
    torch.manual_seed(7)
    model_a = torch.nn.Linear(4, 3)
    model_b = torch.nn.Linear(4, 3)
    model_b.load_state_dict(model_a.state_dict())
    x = torch.randn(8, 6, 4)                 # 8 examples = one effective batch
    mask = torch.zeros(8, 6, dtype=torch.bool)
    mask[:, 3:] = True
    y = torch.randint(0, 3, (8, 6))

    # single effective-batch step: mean CE over ALL masked positions
    loss_ref = torch.nn.functional.cross_entropy(
        model_a(x).reshape(-1, 3)[mask.reshape(-1)],
        y.reshape(-1)[mask.reshape(-1)], reduction="sum") / int(mask.sum())
    loss_ref.backward()

    # microbatched accumulation at a 3-example plan (over-plan single ok)
    mbs = trainer.microbatch_plan(np.full(8, 6), token_budget=18,
                                  max_sequences=3)
    assert [len(mb) for mb in mbs] == [3, 3, 2]
    for mb in mbs:
        loss_mb = torch.nn.functional.cross_entropy(
            model_b(x[mb]).reshape(-1, 3)[mask[mb].reshape(-1)],
            y[mb].reshape(-1)[mask[mb].reshape(-1)],
            reduction="sum") / int(mask.sum())
        loss_mb.backward()
    for (pa, pb) in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6), \
            "microbatched accumulation must equal the single batch step"

    # stage-3 style: summed BCE over the effective batch (per-POSITION
    # binary targets, matching the per-position margin shapes)
    model_c = torch.nn.Linear(4, 2)
    model_d = torch.nn.Linear(4, 2)
    model_d.load_state_dict(model_c.state_dict())
    margins_ref = model_c(x)[:, :, 0] - model_c(x)[:, :, 1]
    (torch.nn.functional.binary_cross_entropy_with_logits(
        margins_ref, (y > 0).float(), reduction="sum")).backward()
    for mb in mbs:
        m = model_d(x[mb])
        (torch.nn.functional.binary_cross_entropy_with_logits(
            m[:, :, 0] - m[:, :, 1], (y[mb] > 0).float(),
            reduction="sum")).backward()
    for (pa, pb) in zip(model_c.parameters(), model_d.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)


def test_allocation_model_fits_rung_for_every_arm():
    rung = trainer.MICROBATCH_RUNG_BYTES / 2 ** 30
    for arm in ("tape_B0", "tape_B0x4", "tape_B0x16", "tape_unbounded",
                "tape_appendonly"):
        cfg = json.loads((PACKAGE_ROOT / "configs" / f"{arm}.json").read_text())
        micro = cfg["microbatch"]
        slots = (cfg.get("budget_slots")
                 or cfg["constants"]["unbounded_cap_slots"])
        worst = 0.0
        for n in range(1, micro["max_sequences"] + 1):
            worst = max(worst, trainer.allocation_model_bytes(
                n, micro["token_budget"] // n, slots) / 2 ** 30)
        assert worst <= rung - 8.0, f"{arm}: in-plan worst {worst} GiB"
    # the manifest-covered receipt agrees
    receipt = json.loads(
        (PACKAGE_ROOT / "receipts" / "microbatch_allocation_model.json")
        .read_text())
    assert receipt["pass"] and receipt["rung_gib"] == rung


# ---------------------------------------------------------------------------
# F7 — occupancy measurement within the word + per-k aggregation.
# ---------------------------------------------------------------------------
def _fake_trace():
    from eng.routing_trace import RoutingTrace
    # B=2, T=4, NH=1, S=2: example 0 appends at t=0 and t=1 (peak 2); example
    # 1 appends at t=0, pops at t=2 (freed), PAD region appends at t=3
    case = torch.zeros(2, 4, 1, dtype=torch.int64)
    alloc = torch.full((2, 4, 1), -1, dtype=torch.int64)
    alloc[0, 0, 0] = 0
    alloc[0, 1, 0] = 1
    alloc[1, 0, 0] = 0
    alloc[1, 3, 0] = 1            # PAD-region append: MUST be excluded (F7)
    matched = torch.full((2, 4, 1), -1, dtype=torch.int64)
    pop_fired = torch.zeros(2, 4, 1, dtype=torch.bool)
    pop_fired[1, 2, 0] = True
    pop_target = torch.full((2, 4, 1), -1, dtype=torch.int64)
    pop_target[1, 2, 0] = 0
    w_on = torch.ones(2, 4, 1, dtype=torch.bool)
    freed = torch.zeros(2, 4, 1, 2, dtype=torch.int8)
    freed[1, 2, 0, 0] = 1          # example 1's slot freed by pop at t=2
    return RoutingTrace(case=case, matched=matched, pop_fired=pop_fired,
                        pop_target=pop_target, alloc_target=alloc, w_on=w_on,
                        freed=freed)


def test_peak_and_eos_live_within_word_only():
    trace = _fake_trace()
    lengths = torch.tensor([3, 3])       # EOS at t=2; t=3 is PAD region
    peaks, eos = trainer.peak_and_eos_live_from_trace(trace, lengths)
    # example 0: live series 1,2,2 -> peak 2 within the word, EOS 2
    assert peaks[0, 0] == 2 and eos[0, 0] == 2
    # example 1: alloc at t=0 (live 1), pop-freed at t=2 -> EOS 0; the PAD
    # region append at t=3 is EXCLUDED (without lengths it would read 1)
    assert peaks[1, 0] == 1 and eos[1, 0] == 0
    # full-range measurement (stage-2 fixed-length words) - correct
    # arithmetic for this trace: example 1's live series is 1,0,...,1
    # (alloc t0, pop-free t2, PAD append t3), so the full-range peak is 1;
    # the PAD-exclusion is already proven by eos[1,0] == 0 above.
    assert trainer.peak_live_from_trace(trace)[1, 0] == 1


def test_per_depth_occupancy_keyed_by_depth():
    depths = np.array([4, 4, 8, 8])
    # per-example mean-over-heads peaks: k=4 -> mean 3.3 (OUTSIDE [3.6, 4.4]);
    # k=8 -> mean 7.75 (INSIDE [7.2, 8.8])
    peaks = np.array([[3.0], [3.6], [7.5], [8.0]])
    eos = np.array([[0.0], [1.0], [0.5], [2.0]])
    labels = np.array([ACCEPT, 0, ACCEPT, 0])
    per_k = trainer.per_depth_occupancy(depths, peaks, eos, labels)
    assert set(per_k) == {"k4", "k8"}
    assert per_k["k4"]["n"] == 2 and per_k["k8"]["n"] == 2
    assert per_k["k4"]["mean_peak_live_slots_within_word"] == pytest.approx(3.3)
    assert per_k["k4"]["mean_peak_within_0.9k_1.1k"] is False  # 3.3 < 3.6
    assert per_k["k8"]["mean_peak_live_slots_within_word"] == pytest.approx(7.75)
    assert per_k["k8"]["mean_peak_within_0.9k_1.1k"] is True   # 7.2 <= 7.75 <= 8.8
    assert per_k["k8"]["mean_eos_live_slots_on_accepts"] == pytest.approx(0.5)
    assert per_k["k4"]["n_accepts"] == 1
    # a pool with no accepts at some k reports None for the pop receipt
    per_k2 = trainer.per_depth_occupancy(
        np.array([4]), np.array([[1.0]]), np.array([[9.0]]), np.array([0]))
    assert per_k2["k4"]["mean_eos_live_slots_on_accepts"] is None


# ---------------------------------------------------------------------------
# F8 — determinism coverage + the degenerate-band mismatch rule.
# ---------------------------------------------------------------------------
def test_determinism_coverage_66_pools():
    points = sweep_points()
    assert len(points) == 51, "union of the three bounded windows (17 each)"
    assert points[0] == 608 and points[16] == 864
    assert points[17] == 2816 and points[33] == 3072
    assert points[34] == 11648 and points[-1] == 11904
    # 51 sweep pools + 15 others (3 stage-1, 2 stage-2, 4 grid, stage-3
    # train/validation, 4 bins) = 66
    others = 3 + 2 + 4 + 2 + 4
    assert len(points) + others == 66
    receipt = json.loads(
        (PACKAGE_ROOT / "generators" / "receipts" / "determinism_check.json")
        .read_text())
    assert receipt["n_pools_checked"] == 66
    assert receipt["all_deterministic"] is True
    assert all(s["differs_from_registered"] for s in receipt["seed_sensitivity"])
    assert len(receipt["sweep_mismatch_rule_receipts"]) == 51
    assert all(m["pass"] for m in receipt["sweep_mismatch_rule_receipts"])


def test_mismatch_rule_by_construction():
    # k >= 2: j alternates between k-1 and k+1 (both valid, j >= 1)
    for k in (2, 736, 864, 3072):
        row = mismatch_rule_check(k)
        assert row["pass"], row
        assert row["j_never_equals_k"] and row["j_alternates"]
    # k = 1: j = 0 is excluded (j never 0), so the band is single-valued
    # (j = 2 always): no alternation is possible - and that is correct.
    row1 = mismatch_rule_check(1)
    assert row1["pass"], row1
    assert row1["j_never_equals_k"] and not row1["j_alternates"]
    assert row1["k1_uses_k_plus_1"] is True
    # the rule is receipted in the returned pool metadata
    _, _, _, meta = ladder_sweep_pool(5)
    assert "j = k+1 / max(k-1, 1) alternating" in meta["count_mismatch_rule"]
    assert "depths" in meta and len(meta["depths"]) == meta["n_examples"]


# ---------------------------------------------------------------------------
# F9 — hash-validated receipts.
# ---------------------------------------------------------------------------
def test_receipt_payload_hash_detects_tampering():
    row = {"arm": "tape_B0", "status": "complete", "updates": 0}
    h = trainer.receipt_payload_hash(row)
    row_ok = {**row, "payload_sha256": h}
    assert trainer.receipt_payload_hash(row_ok) == h     # idempotent
    tampered = {**row_ok, "status": "halt_stage_gate_failed"}
    assert trainer.receipt_payload_hash(tampered) != h
    # field order must not matter (canonical JSON)
    assert trainer.receipt_payload_hash(
        {"updates": 0, "status": "complete", "arm": "tape_B0"}) == h


# ---------------------------------------------------------------------------
# F10 — pop-off single construction: identical non-pop initial parameters.
# ---------------------------------------------------------------------------
def test_pop_disabled_shares_identical_non_pop_initial_parameters():
    for arm in ("tape_unbounded", "tape_appendonly"):
        cfg = json.loads((PACKAGE_ROOT / "configs" / f"{arm}.json").read_text())
        assert cfg["head"] == {"n_heads": 4, "key_dim": 64, "value_dim": 128}
    torch.manual_seed(0)
    unbounded = trainer.build_tape_model(json.loads(
        (PACKAGE_ROOT / "configs" / "tape_unbounded.json").read_text()))
    torch.manual_seed(0)
    appendonly = trainer.build_tape_model(json.loads(
        (PACKAGE_ROOT / "configs" / "tape_appendonly.json").read_text()))
    sa = unbounded.layer.state_dict()
    sb = appendonly.layer.state_dict()
    assert set(sa) == set(sb)
    for key in sa:
        assert torch.equal(sa[key], sb[key]), \
            f"F10: pop-off wiring must not reinitialize {key}"
    assert torch.equal(unbounded.head_binary.weight,
                       appendonly.head_binary.weight)


# ---------------------------------------------------------------------------
# F2 — verifier-only refusal / exit-without-trainer tests.
# ---------------------------------------------------------------------------
def _run_trainer(args, cwd=None):
    return subprocess.run(
        [sys.executable, str(PACKAGE_ROOT / "trainer.py"), *args],
        capture_output=True, text=True, timeout=300, cwd=cwd)


def test_verify_only_refuses_wrong_hash():
    """Wrong --frozen-freeze-hash refuses fail-closed (exit 3) — in the real
    (even uncommitted) tree the argument check precedes the committed check."""
    proc = _run_trainer(["--verify-only", "--frozen-freeze-hash", "0" * 64])
    assert proc.returncode == trainer.EXIT_FREEZE_FAILED
    assert "does not match the freeze manifest hash" in proc.stderr
    assert "freeze verification FAILED" in proc.stderr


def test_verify_only_requires_hash():
    proc = _run_trainer(["--verify-only"])
    assert proc.returncode != 0
    assert "REFUSED" in proc.stderr or "--frozen-freeze-hash" in proc.stderr


def _build_committed_mirror(tmp: Path) -> Path:
    """A committed git mirror of the package + its manifest-bound dependency
    files (same rel-path layout), for the verify-only success-path proof.
    The project repo itself is never touched."""
    mirror = tmp / "mirror"
    pkg = mirror / REL_PREFIX
    shutil.copytree(PACKAGE_ROOT, pkg,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = json.loads((PACKAGE_ROOT / "freeze_manifest.json").read_text())
    for rel in manifest["files"]:
        if rel.startswith("../"):
            dst = (pkg / rel).resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2((PACKAGE_ROOT / rel).resolve(), dst)
    subprocess.run(["git", "init", "-q", str(mirror)], check=True)
    subprocess.run(["git", "-C", str(mirror), "config", "user.email",
                    "x28e-verify@test"], check=True)
    subprocess.run(["git", "-C", str(mirror), "config", "user.name",
                    "x28e-verify"], check=True)
    subprocess.run(["git", "-C", str(mirror), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(mirror), "commit", "-qm", "mirror"],
                   check=True)
    return mirror


def test_verify_only_verifies_and_exits_without_constructing_trainer():
    """Right hash + committed mirror: verification completes and the process
    EXITS WITHOUT constructing a trainer (receipted ``torch_imported:
    false`` — structural proof no model/trainer code ran)."""
    manifest = json.loads((PACKAGE_ROOT / "freeze_manifest.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        mirror = _build_committed_mirror(Path(tmp))
        proc = subprocess.run(
            [sys.executable, str(mirror / REL_PREFIX / "trainer.py"),
             "--verify-only", "--frozen-freeze-hash",
             manifest["manifest_hash"]],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == trainer.EXIT_OK, proc.stderr
        receipt = json.loads(proc.stdout)
        assert receipt["manifest_hash"] == manifest["manifest_hash"]
        assert receipt["worktree_and_head_verified"] is True
        assert receipt["trainer_constructed"] is False
        assert receipt["torch_imported"] is False, \
            "the verify-only path must not import any model/trainer code"
        assert receipt["dependency_files_verified"] > 0
        assert receipt["package_files_verified"] > 0


def test_verify_only_refuses_untracked_file_in_package():
    """The finding-3 scenario: a stray untracked file inside the package
    dirties the worktree and verification fails closed (disk hashes of the
    manifest files still match — the DIRTY check is what refuses)."""
    manifest = json.loads((PACKAGE_ROOT / "freeze_manifest.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        mirror = _build_committed_mirror(Path(tmp))
        stray = mirror / REL_PREFIX / "results" / "receipts" / "stray.json"
        stray.parent.mkdir(parents=True)
        stray.write_text("{}\n")
        proc = subprocess.run(
            [sys.executable, str(mirror / REL_PREFIX / "trainer.py"),
             "--verify-only", "--frozen-freeze-hash",
             manifest["manifest_hash"]],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == trainer.EXIT_FREEZE_FAILED
        assert "worktree not clean" in proc.stderr


def test_verify_only_path_is_structurally_trainer_free():
    """AST check: the verify-only code path constructs no trainer and calls
    no model builder."""
    tree = ast.parse((PACKAGE_ROOT / "trainer.py").read_text())
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef)}
    verify_only = funcs["run_verify_only"]
    calls = [n for n in ast.walk(verify_only) if isinstance(n, ast.Call)]
    names = {getattr(c.func, "id", getattr(c.func, "attr", None))
             for c in calls}
    assert "StagedTrainer" not in names
    assert "build_tape_model" not in names
    assert "build_native_model" not in names
    # and the main dispatch returns from verify-only before any construction
    main_fn = funcs["main"]
    src = ast.get_source_segment((PACKAGE_ROOT / "trainer.py").read_text(),
                                 main_fn)
    assert src.index("run_verify_only") < src.index("StagedTrainer(")


# ---------------------------------------------------------------------------
# M2-review-2 additions: device wiring (finding 1), exactly-64 at 70,000
# (finding 5), arm_state identity/censor classification (findings 7/8).
# ---------------------------------------------------------------------------

def test_label_mapping_never_receives_cuda_tensor():
    """Finding 1 regression: the training loop maps labels on the CPU numpy
    array BEFORE device transfer; a CUDA tensor passed to the pure-numpy
    mapper raises (documented constraint), and the fixed call shape
    produces correct indices."""
    import torch
    y_cpu = np.arange(1, 65, dtype=np.int64).reshape(8, 8)
    mapped = trainer.labels_to_classifier_indices(y_cpu[:, :4], 1)
    assert mapped.shape == (8, 4)
    assert np.array_equal(mapped, y_cpu[:, :4] - 1)
    if torch.cuda.is_available():
        y_cuda = torch.arange(1, 65, device="cuda", dtype=torch.int64)
        try:
            trainer.labels_to_classifier_indices(y_cuda, 1)
        except (TypeError, RuntimeError, ValueError):
            pass                          # the documented constraint
        else:
            pytest.fail("np.asarray on a CUDA tensor must not silently "
                         "succeed; the wiring must map pre-transfer")


def test_batches_exactly_64_at_70k_examples():
    """Finding 5: at the registered 70,000-example pools, every yielded
    batch has exactly 64 examples; 1,093 full batches fire and the
    48-example remainder is dropped at the epoch boundary."""
    rng = np.random.default_rng(0)
    batches = list(trainer.StagedTrainer._batches(None, 70_000, rng))
    assert len(batches) == 70_000 // 64 == 1093
    assert all(len(b) == 64 for b in batches)
    flat = np.concatenate(batches)
    assert len(flat) == 1093 * 64 and len(np.unique(flat)) == len(flat)
    # the dropped remainder is exactly the 48 examples not in any batch
    dropped = np.setdiff1d(np.arange(70_000), flat)
    assert len(dropped) == 70_000 - len(flat) == 48


def _hash_valid_receipt(row: dict) -> dict:
    row = dict(row)
    row["payload_sha256"] = trainer.receipt_payload_hash(row)
    return row


def _write_arm_dir(base, arm, *, result=None, terminal=None, guard=None,
                   worker_dead=True, launch_hash=True, worker_hash=True,
                   worker_row=None):
    import launcher as L
    run = base / "results" / f"{arm}_seed0"
    rec = base / "receipts" / f"{arm}_seed0"
    run.mkdir(parents=True, exist_ok=True)
    rec.mkdir(parents=True, exist_ok=True)
    launch = {"arm": arm, "freeze_manifest_hash": "HASH"}
    if launch_hash:
        launch["payload_sha256"] = trainer.receipt_payload_hash(launch)
    (rec / "launch.json").write_text(json.dumps(launch))
    if result is not None:
        (run / "result.json").write_text(json.dumps(result))
    if guard is not None:
        (rec / "guard.json").write_text(json.dumps(guard))
    wrow = worker_row if worker_row is not None else \
        {"pid": 999999999, "starttime": "1", "launched_at": 1000.0,
         "arm": arm, "freeze_manifest_hash": "HASH"}
    if worker_hash:
        wrow = dict(wrow)
        wrow["payload_sha256"] = trainer.receipt_payload_hash(wrow)
    (rec / "worker.json").write_text(json.dumps(wrow))
    if terminal is not None:
        (rec / "terminal.json").write_text(json.dumps(terminal))
    return run, rec


def test_arm_state_identity_and_censor_classification(monkeypatch, tmp_path):
    """Findings 7/8: hash-valid but wrong-arm/missing-evidence results are
    errors; malformed JSON is an error; a bare exit-0 terminal without the
    completed result is an error; a validated identity-matching guard.json
    is the registered resource-CENSOR disposition."""
    import launcher as L
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")
    good = _hash_valid_receipt({"arm": "a1", "status": "complete",
                                "zero_updates": True})
    _write_arm_dir(tmp_path, "a1", result=good)
    assert L.arm_state("a1", frozen_hash="HASH") == "complete"
    wrong_arm = _hash_valid_receipt({"arm": "OTHER", "status": "complete",
                                     "zero_updates": True})
    _write_arm_dir(tmp_path, "a2", result=wrong_arm)
    assert L.arm_state("a2", frozen_hash="HASH") == "error"
    _write_arm_dir(tmp_path, "a3", result={"status": "complete",
                                           "no_hash": True})
    assert L.arm_state("a3", frozen_hash="HASH") == "error"
    _write_arm_dir(tmp_path, "a4")
    (tmp_path / "results" / "a4_seed0" / "result.json").write_text("{bad")
    assert L.arm_state("a4", frozen_hash="HASH") == "error"
    _write_arm_dir(tmp_path, "a5",
                   terminal=_hash_valid_receipt({"exit_code": 0}))
    assert L.arm_state("a5", frozen_hash="HASH") == "error"
    guard = _hash_valid_receipt({"arm": "a6", "reason": "disk guard TERM"})
    _write_arm_dir(tmp_path, "a6", guard=guard)
    assert L.arm_state("a6", frozen_hash="HASH") == "censored"
    guard_wrong = _hash_valid_receipt({"arm": "OTHER",
                                       "reason": "disk guard TERM"})
    _write_arm_dir(tmp_path, "a7", guard=guard_wrong)
    assert L.arm_state("a7", frozen_hash="HASH") == "error"


# ---------------------------------------------------------------------------
# M2-review-3 additions: the entry contract as a constructor invariant
# (findings 1+2) and identity-required terminal classification (finding 3).
# ---------------------------------------------------------------------------

def test_entry_context_validator_pure_refusals(tmp_path, monkeypatch):
    """Findings 1+2: _validate_entry_context is pure (no torch), refuses a
    package-contained results root, a missing launcher context, an
    unreadable context, and arm/hash mismatches; accepts a valid context."""
    import torch  # noqa: F401  (the validator itself must not need it)
    outside = tmp_path / "results_out"
    inside = PACKAGE_ROOT / "results_inside_for_test"
    monkeypatch.delenv("X28E_LAUNCHER_LAUNCH_JSON", raising=False)
    # missing context
    with pytest.raises(SystemExit):
        trainer._validate_entry_context("tape_B0", outside, "HASH")
    # in-package results root
    monkeypatch.setenv("X28E_LAUNCHER_LAUNCH_JSON", "nonexistent.json")
    with pytest.raises(SystemExit):
        trainer._validate_entry_context("tape_B0", inside, "HASH")
    # unreadable context
    ctx = tmp_path / "launch.json"
    monkeypatch.setenv("X28E_LAUNCHER_LAUNCH_JSON", str(ctx))
    ctx.write_text("{bad")
    with pytest.raises(SystemExit):
        trainer._validate_entry_context("tape_B0", outside, "HASH")
    # wrong arm (SIGNED so the refusal is the identity check itself)
    wrong_arm = {"arm": "OTHER", "freeze_manifest_hash": "HASH"}
    wrong_arm["payload_sha256"] = trainer.receipt_payload_hash(wrong_arm)
    ctx.write_text(json.dumps(wrong_arm))
    with pytest.raises(SystemExit):
        trainer._validate_entry_context("tape_B0", outside, "HASH")
    # wrong freeze hash (signed likewise)
    wrong_hash = {"arm": "tape_B0", "freeze_manifest_hash": "OTHER"}
    wrong_hash["payload_sha256"] = trainer.receipt_payload_hash(wrong_hash)
    ctx.write_text(json.dumps(wrong_hash))
    with pytest.raises(SystemExit):
        trainer._validate_entry_context("tape_B0", outside, "HASH")
    # unsigned context: fields look right but there is no checksum
    ctx.write_text(json.dumps({"arm": "tape_B0", "freeze_manifest_hash":
                               "HASH"}))
    with pytest.raises(SystemExit):
        trainer._validate_entry_context("tape_B0", outside, "HASH")
    # stale checksum: the payload changed after signing
    signed = {"arm": "tape_B0", "freeze_manifest_hash": "HASH"}
    signed["payload_sha256"] = trainer.receipt_payload_hash(signed)
    tampered = {**signed, "freeze_manifest_hash": "OTHER"}
    ctx.write_text(json.dumps(tampered))
    with pytest.raises(SystemExit):
        trainer._validate_entry_context("tape_B0", outside, "HASH")
    # valid signed context
    ctx.write_text(json.dumps(signed))
    trainer._validate_entry_context("tape_B0", outside, "HASH")


def test_constructor_calls_entry_context_before_torch():
    """Structural (finding 2): StagedTrainer.__init__ invokes
    _validate_entry_context BEFORE importing torch, so no construction path
    bypasses the entry contract."""
    src = (PACKAGE_ROOT / "trainer.py").read_text()
    init = src.split("class StagedTrainer:")[1].split("def run(")[0]
    v_pos = init.index("_validate_entry_context")
    t_pos = init.index("import torch")
    assert v_pos < t_pos, \
        "the entry contract must gate construction before any torch import"


def test_cli_refuses_without_launcher_context_in_committed_mirror():
    """Findings 1+2 end-to-end: in a committed mirror with the CORRECT
    hash and NO launcher context, the training CLI refuses (exit 3, the
    context message) WITHOUT constructing a trainer — and the path runs
    without any NameError (the round-3 PACKAGE_ROOT bug)."""
    manifest = json.loads((PACKAGE_ROOT / "freeze_manifest.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        mirror = _build_committed_mirror(Path(tmp))
        env = {k: v for k, v in os.environ.items()
               if k != "X28E_LAUNCHER_LAUNCH_JSON"}
        proc = subprocess.run(
            [sys.executable, str(mirror / REL_PREFIX / "trainer.py"),
             "--arm", "tape_B0", "--frozen-freeze-hash",
             manifest["manifest_hash"]],
            capture_output=True, text=True, timeout=300, env=env)
        assert proc.returncode == trainer.EXIT_FREEZE_FAILED, proc.stderr
        assert "launcher context" in proc.stderr, proc.stderr
        assert "NameError" not in proc.stderr, proc.stderr


def test_arm_state_requires_launch_identity_for_terminal(monkeypatch,
                                                         tmp_path):
    """Finding 3: a wrong-freeze launch receipt, a MISSING launch receipt,
    and a no-identity caller all refuse terminal classification."""
    import launcher as L
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")

    def _good(arm):
        return _hash_valid_receipt({"arm": arm, "status": "complete",
                                    "zero_updates": True})

    # wrong-freeze launch receipt
    _write_arm_dir(tmp_path, "b1", result=_good("b1"))
    (tmp_path / "receipts" / "b1_seed0" / "launch.json").write_text(json.dumps(
        {"arm": "b1", "freeze_manifest_hash": "OTHER"}))
    assert L.arm_state("b1", frozen_hash="HASH") == "error"
    # missing launch receipt
    _write_arm_dir(tmp_path, "b2", result=_good("b2"))
    (tmp_path / "receipts" / "b2_seed0" / "launch.json").unlink()
    assert L.arm_state("b2", frozen_hash="HASH") == "error"
    # no-identity caller cannot terminally classify
    _write_arm_dir(tmp_path, "b3", result=_good("b3"))
    assert L.arm_state("b3", frozen_hash="") == "error"
    # identity-bound happy path still completes
    _write_arm_dir(tmp_path, "b4", result=_good("b4"))
    assert L.arm_state("b4", frozen_hash="HASH") == "complete"


# ---------------------------------------------------------------------------
# M2-review-4 additions: trained-completion evidence (F1) and the validated
# censor receipt (F2).
# ---------------------------------------------------------------------------

def test_arm_state_accepts_trained_completion_evidence(monkeypatch,
                                                        tmp_path):
    """F1: the trainer's real completion schema carries total_updates (no
    zero_updates field); a genuine identity-matched trained result must
    classify as complete, and missing ALL evidence fields is an error."""
    import launcher as L
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")
    trained = _hash_valid_receipt({"arm": "c1", "status": "complete",
                                   "stage3_validation_balanced_accuracy":
                                       0.9125,
                                   "total_updates": 43_200})
    _write_arm_dir(tmp_path, "c1", result=trained)
    assert L.arm_state("c1", frozen_hash="HASH") == "complete"
    bare = _hash_valid_receipt({"arm": "c2", "status": "complete"})
    _write_arm_dir(tmp_path, "c2", result=bare)
    assert L.arm_state("c2", frozen_hash="HASH") == "error"


def test_arm_state_censor_receipt_is_validated(monkeypatch, tmp_path):
    """F2: censor.json is terminal evidence and must be hash-valid,
    arm-matching, and identity-bound; existence alone proves nothing."""
    import launcher as L
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")

    def _censor_row(arm):
        return _hash_valid_receipt({"arm": arm, "reason":
                                    "disk guard TERM"})

    # identity-bound happy path
    _write_arm_dir(tmp_path, "d1", guard=None)
    (tmp_path / "receipts" / "d1_seed0" / "censor.json").write_text(json.dumps(
        _censor_row("d1")))
    assert L.arm_state("d1", frozen_hash="HASH") == "censored"
    # wrong arm inside a hash-valid censor receipt
    _write_arm_dir(tmp_path, "d2", guard=None)
    (tmp_path / "receipts" / "d2_seed0" / "censor.json").write_text(json.dumps(
        _censor_row("OTHER")))
    assert L.arm_state("d2", frozen_hash="HASH") == "error"
    # malformed JSON
    _write_arm_dir(tmp_path, "d3", guard=None)
    (tmp_path / "receipts" / "d3_seed0" / "censor.json").write_text("{bad")
    assert L.arm_state("d3", frozen_hash="HASH") == "error"
    # tampered payload (stored hash does not match recomputed)
    _write_arm_dir(tmp_path, "d4", guard=None)
    row = _censor_row("d4")
    row["reason"] = "tampered after hashing"
    (tmp_path / "receipts" / "d4_seed0" / "censor.json").write_text(json.dumps(
        row))
    assert L.arm_state("d4", frozen_hash="HASH") == "error"
    # missing launch receipt: identity cannot bind
    _write_arm_dir(tmp_path, "d5", guard=None)
    (tmp_path / "receipts" / "d5_seed0" / "censor.json").write_text(json.dumps(
        _censor_row("d5")))
    (tmp_path / "receipts" / "d5_seed0" / "launch.json").unlink()
    assert L.arm_state("d5", frozen_hash="HASH") == "error"
    # no-identity caller: never censored without validation
    _write_arm_dir(tmp_path, "d6", guard=None)
    (tmp_path / "receipts" / "d6_seed0" / "censor.json").write_text(json.dumps(
        _censor_row("d6")))
    assert L.arm_state("d6", frozen_hash="") == "error"


# ---------------------------------------------------------------------------
# M2-review-5 additions: hash-validated launch and worker receipts; invalid
# guard evidence is an error, never a fall-through.
# ---------------------------------------------------------------------------

def test_arm_state_launch_receipt_must_be_hash_valid(monkeypatch, tmp_path):
    """F1: a tampered launch receipt (freeze field changed without
    re-hashing) and an unsigned one never bind identity."""
    import launcher as L
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")
    good = _hash_valid_receipt({"arm": "e1", "status": "complete",
                                "total_updates": 1000})
    # tampered launch: correct-looking fields, stale payload hash
    _write_arm_dir(tmp_path, "e1", result=good, launch_hash=False)
    tampered = json.loads(
        (tmp_path / "receipts" / "e1_seed0" / "launch.json").read_text())
    tampered["payload_sha256"] = trainer.receipt_payload_hash(
        {k: v for k, v in tampered.items() if k != "payload_sha256"}
        | {"freeze_manifest_hash": "OTHER"})
    (tmp_path / "receipts" / "e1_seed0" / "launch.json").write_text(
        json.dumps(tampered))
    assert L.arm_state("e1", frozen_hash="HASH") == "error"
    # unsigned launch receipt
    _write_arm_dir(tmp_path, "e2", result=_hash_valid_receipt(
        {"arm": "e2", "status": "complete", "total_updates": 1000}),
        launch_hash=False)
    assert L.arm_state("e2", frozen_hash="HASH") == "error"


def test_arm_state_worker_receipt_is_hash_valid_and_schema_safe(
        monkeypatch, tmp_path):
    """F2: a tampered worker receipt, a hash-missing one, and a
    schema-broken-but-parsable one are errors — none crash the watcher."""
    import launcher as L
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")
    # tampered: stored hash does not match the content on disk
    _write_arm_dir(tmp_path, "f1",
                   worker_row={"pid": 999999999, "starttime": "1",
                               "launched_at": 1000.0, "arm": "f1",
                               "freeze_manifest_hash": "HASH"})
    wpath = tmp_path / "receipts" / "f1_seed0" / "worker.json"
    wrow = json.loads(wpath.read_text())
    wrow["pid"] = 1          # mutate AFTER hashing: checksum now stale
    wpath.write_text(json.dumps(wrow))
    assert L.arm_state("f1", frozen_hash="HASH") == "error"
    # unsigned worker receipt
    _write_arm_dir(tmp_path, "f2", worker_hash=False)
    assert L.arm_state("f2", frozen_hash="HASH") == "error"
    # schema-broken but parsable {}: HASH-SIGNED so the failure is the
    # schema branch itself (M2-review-6 finding 4), not the hash branch
    _write_arm_dir(tmp_path, "f3", worker_hash=True, worker_row={})
    assert L.arm_state("f3", frozen_hash="HASH") == "error"
    # missing starttime only, also hash-signed
    _write_arm_dir(tmp_path, "f4", worker_hash=True,
                   worker_row={"pid": 12345})
    assert L.arm_state("f4", frozen_hash="HASH") == "error"
    # direct alive() assertions for the missing-field schema
    assert L.alive({}) is False
    assert L.alive({"pid": 12345}) is False
    assert L.alive({"starttime": "1"}) is False


def test_arm_state_invalid_guard_is_error_not_fallthrough(monkeypatch,
                                                          tmp_path):
    """F3: a guard receipt that fails validation (hash/arm/identity) is an
    ERROR even when the worker is alive, and when no worker receipt
    exists — never a silent running/queued classification."""
    import launcher as L
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")
    def _me_alive(arm):
        return {"pid": os.getpid(), "starttime":
                Path(f"/proc/{os.getpid()}/stat").read_text()
                .rsplit(") ", 1)[1].split()[19], "launched_at": time.time(),
                "arm": arm, "freeze_manifest_hash": "HASH"}
    # invalid guard + LIVE worker: error, not running
    _write_arm_dir(tmp_path, "g1", guard={"arm": "g1", "reason": "TERM",
                                          "not_hashed": True},
                   worker_row=_me_alive("g1"))
    assert L.arm_state("g1", frozen_hash="HASH") == "error"
    # identity-mismatched (wrong arm) guard, hash-valid: error
    _write_arm_dir(tmp_path, "g2",
                   guard=_hash_valid_receipt({"arm": "OTHER",
                                             "reason": "TERM"}))
    assert L.arm_state("g2", frozen_hash="HASH") == "error"
    # invalid guard + NO worker receipt: error, not queued
    rec = tmp_path / "receipts" / "g3_seed0"
    _write_arm_dir(tmp_path, "g3", guard={"arm": "g3", "reason": "TERM",
                                          "not_hashed": True})
    (rec / "worker.json").unlink()
    assert L.arm_state("g3", frozen_hash="HASH") == "error"
    # control: no guard at all + live worker => running
    _write_arm_dir(tmp_path, "g4", worker_row=_me_alive("g4"))
    assert L.arm_state("g4", frozen_hash="HASH") == "running"


# ---------------------------------------------------------------------------
# M2-review-6 additions: validated signal paths (F1), group-continuity
# before unrecorded kills (F2), watchdog timestamp schema (F3).
# ---------------------------------------------------------------------------

def _signal_sandbox(monkeypatch, tmp_path, arm):
    """Patch the launcher's signal surface into a recording sandbox."""
    import launcher as L
    kills, killpgs = [], []
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "freeze_manifest_hash", lambda: "HASH")
    rec = tmp_path / "receipts" / f"{arm}_seed0"
    rec.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(L.os, "kill",
                        lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(L.os, "killpg",
                        lambda pgid, sig: killpgs.append((pgid, sig)))
    return L, rec, kills, killpgs


def test_escalate_never_signals_on_corrupt_evidence(monkeypatch, tmp_path):
    """F1: _escalate validates the guard AND worker receipts at its own
    consumption points; hash-invalid or malformed evidence produces no
    signals and no escalation receipt."""
    L, rec, kills, killpgs = _signal_sandbox(monkeypatch, tmp_path, "h1")
    guard = _hash_valid_receipt({"arm": "h1", "reason": "disk",
                                 "group_members": [{"pid": 111,
                                                    "starttime": "50"}]})
    worker = _hash_valid_receipt({"pid": 111, "starttime": "50",
                                  "launched_at": 1.0, "arm": "h2",
                                  "freeze_manifest_hash": "HASH"})
    # stale-hash worker receipt
    rec.joinpath("guard.json").write_text(json.dumps(guard))
    rec.joinpath("worker.json").write_text(json.dumps(
        {**worker, "pid": 222}))          # mutated after hashing
    L._escalate("h1")
    assert kills == [] and killpgs == []
    # unsigned worker receipt
    rec.joinpath("worker.json").write_text(json.dumps(
        {"pid": 111, "starttime": "50", "launched_at": 1.0}))
    L._escalate("h1")
    assert kills == [] and killpgs == []
    # hash-invalid guard receipt
    rec.joinpath("worker.json").write_text(json.dumps(worker))
    rec.joinpath("guard.json").write_text(json.dumps(
        {**guard, "extra": "mutated after hashing"}))
    L._escalate("h1")
    assert kills == [] and killpgs == []
    # malformed guard JSON
    rec.joinpath("guard.json").write_text("{bad")
    L._escalate("h1")
    assert kills == [] and killpgs == []


def test_escalate_group_continuity_gates_unrecorded_kills(monkeypatch,
                                                         tmp_path):
    """F2: with the leader dead, unrecorded members of a possibly
    REPLACEMENT group are never signalled; identity-matching recorded
    members still are. With the leader alive, unrecorded owned-group
    members are reached (continuity proven)."""
    L, rec, kills, killpgs = _signal_sandbox(monkeypatch, tmp_path, "h2")
    guard = _hash_valid_receipt({"arm": "h2", "reason": "disk",
                                 "group_members": [
                                     {"pid": 111, "starttime": "50"},
                                     {"pid": 112, "starttime": "51"}]})
    worker = _hash_valid_receipt({"pid": 111, "starttime": "50",
                                  "launched_at": 1.0, "arm": "h2",
                                  "freeze_manifest_hash": "HASH"})
    rec.joinpath("guard.json").write_text(json.dumps(guard))
    rec.joinpath("worker.json").write_text(json.dumps(worker))
    # replacement group: leader pid 111 now has a NEW starttime, and an
    # unrecorded child 999 exists; the original leader is dead.
    replacement = [{"pid": 111, "starttime": "9900"},
                   {"pid": 999, "starttime": "9901"}]
    monkeypatch.setattr(L, "group_members", lambda pgid: replacement)
    monkeypatch.setattr(L, "alive", lambda record: False)
    L._escalate("h2")
    # recycled leader skipped; unrecorded child untouched (no continuity)
    assert kills == [] and killpgs == []
    # control: identity-matching recorded member IS signalled (a fresh
    # guard each phase: _escalate receipts escalated_at when it runs)
    def _fresh_guard():
        rec.joinpath("guard.json").write_text(json.dumps(
            _hash_valid_receipt({"arm": "h2", "reason": "disk",
                                 "group_members": [
                                     {"pid": 111, "starttime": "50"},
                                     {"pid": 112, "starttime": "51"}]})))
    _fresh_guard()
    mixed = [{"pid": 112, "starttime": "51"},   # recorded, matching
             {"pid": 999, "starttime": "9901"}]  # unrecorded
    monkeypatch.setattr(L, "group_members", lambda pgid: mixed)
    L._escalate("h2")
    assert kills == [(112, L.signal.SIGKILL)] and killpgs == []
    # control: leader alive -> group continuity proven -> unrecorded
    # members are reached AND killpg fires
    kills.clear(); killpgs.clear()
    _fresh_guard()
    monkeypatch.setattr(L, "alive", lambda record: True)
    L._escalate("h2")
    signalled = {pid for pid, _ in kills}
    assert signalled == {112, 999} and killpgs == [(111, L.signal.SIGKILL)]


def test_guard_kill_validates_evidence_and_restricts_leaderless_term(
        monkeypatch, tmp_path):
    """F1+F2 for guard_kill: corrupt worker evidence means no TERM and no
    re-signed guard receipt; with the leader dead, TERM reaches only
    identity-matching recorded members."""
    L, rec, kills, killpgs = _signal_sandbox(monkeypatch, tmp_path, "h3")
    monkeypatch.setattr(L, "group_members", lambda pgid: [
        {"pid": 111, "starttime": "50"},
        {"pid": 999, "starttime": "9901"}])
    # neutralize the escalation timer BEFORE any guard_kill call
    class _DummyTimer:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass
    monkeypatch.setattr(L.threading, "Timer", _DummyTimer)
    # corrupt worker receipt: nothing signalled, and the guard carries ONLY
    # the launcher-signed censor DECISION — term_sent_at never set, no
    # re-signing of the (absent) invalid evidence
    rec.joinpath("worker.json").write_text("{bad")
    L.guard_kill("h3", {"arm": "h3", "free_bytes": 1})
    assert kills == [] and killpgs == []
    grow = json.loads(rec.joinpath("guard.json").read_text())
    assert grow["payload_sha256"] == trainer.receipt_payload_hash(
        {k: v for k, v in grow.items() if k != "payload_sha256"})
    assert "term_sent_at" not in grow
    # valid worker, leader dead, NO recorded members in the guard row
    # (the real first-call flow: members are receipted only AFTER TERM):
    # ownership of the group is unproven -> NOBODY is TERMed (fail-closed)
    worker = _hash_valid_receipt({"pid": 111, "starttime": "50",
                                  "launched_at": 1.0, "arm": "h3",
                                  "freeze_manifest_hash": "HASH"})
    rec.joinpath("worker.json").write_text(json.dumps(worker))
    monkeypatch.setattr(L, "alive", lambda record: False)
    L.guard_kill("h3", {"arm": "h3", "free_bytes": 1})
    assert kills == [] and killpgs == []
    grow = json.loads(rec.joinpath("guard.json").read_text())
    assert grow["term_mode"].startswith("per-member TERM restricted")
    assert "0 signalled" in grow["term_mode"]
    # with recorded members present in the guard row (e.g. a re-run after
    # a prior receipted attempt), identity-matching members ARE TERMed and
    # the unrecorded replacement-group child is not
    rec.joinpath("guard.json").write_text(json.dumps(_hash_valid_receipt({
        "arm": "h3", "reason": "disk",
        "group_members": [{"pid": 111, "starttime": "50"},
                          {"pid": 112, "starttime": "51"}]})))
    L.guard_kill("h3", {"arm": "h3", "free_bytes": 1})
    assert kills == [(111, L.signal.SIGTERM)] and killpgs == []
    grow = json.loads(rec.joinpath("guard.json").read_text())
    assert grow["term_mode"].startswith("per-member TERM restricted")
    assert "1 signalled" in grow["term_mode"]


def test_watchdog_worker_timestamp_schema(monkeypatch, tmp_path):
    """F3: hash-valid worker receipts missing launched_at, or carrying a
    nonnumeric/nonfinite one, are arm errors at classification time — the
    watchdog's time.time() - record['launched_at'] can never KeyError or
    TypeError."""
    import launcher as L
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")
    _write_arm_dir(tmp_path, "i1",
                   worker_row={"pid": 999999999, "starttime": "1"})
    assert L.arm_state("i1", frozen_hash="HASH") == "error"
    _write_arm_dir(tmp_path, "i2",
                   worker_row={"pid": 999999999, "starttime": "1",
                               "launched_at": "not-a-number"})
    assert L.arm_state("i2", frozen_hash="HASH") == "error"
    _write_arm_dir(tmp_path, "i3",
                   worker_row={"pid": 999999999, "starttime": "1",
                               "launched_at": float("inf")})
    assert L.arm_state("i3", frozen_hash="HASH") == "error"


# ---------------------------------------------------------------------------
# M2-review-7 additions: the sequential TERM->KILL lifecycle (F1),
# arm-bound guard validation in the signal paths (F2).
# ---------------------------------------------------------------------------

def test_leaderless_lifecycle_never_promotes_replacement_group(
        monkeypatch, tmp_path):
    """F1 end-to-end: guard_kill with a dead leader TERMs nobody and
    receipts an EMPTY trusted set; the subsequent _escalate cannot promote
    the replacement group (recycled leader + unrecorded child) into KILL
    targets. Zero signals across the whole lifecycle."""
    L, rec, kills, killpgs = _signal_sandbox(monkeypatch, tmp_path, "j1")
    worker = _hash_valid_receipt({"pid": 111, "starttime": "50",
                                  "launched_at": 1.0, "arm": "j1",
                                  "freeze_manifest_hash": "HASH"})
    rec.joinpath("worker.json").write_text(json.dumps(worker))
    monkeypatch.setattr(L, "alive", lambda record: False)
    replacement = [{"pid": 111, "starttime": "9900"},   # recycled leader
                   {"pid": 999, "starttime": "9901"}]   # replacement child
    monkeypatch.setattr(L, "group_members", lambda pgid: replacement)
    L.guard_kill("j1", {"arm": "j1", "free_bytes": 1})
    assert kills == [] and killpgs == []
    grow = json.loads(rec.joinpath("guard.json").read_text())
    assert grow["group_members"] == []      # nothing promoted to trusted
    # the scheduled escalation fires with the same replacement group
    L._escalate("j1")
    assert kills == [] and killpgs == []    # zero signals lifecycle-wide
    grow = json.loads(rec.joinpath("guard.json").read_text())
    assert grow.get("escalation_kills") == []


def test_signal_paths_refuse_wrong_arm_and_malformed_members(monkeypatch,
                                                             tmp_path):
    """F2: a checksum-valid but WRONG-ARM guard, and a signed guard with
    malformed member records, produce no signalling and no re-signing in
    both guard_kill and _escalate."""
    for arm in ("k1", "k2"):
        L, rec, kills, killpgs = _signal_sandbox(monkeypatch, tmp_path, arm)
        worker = _hash_valid_receipt({"pid": 111, "starttime": "50",
                                      "launched_at": 1.0, "arm": arm,
                                      "freeze_manifest_hash": "HASH"})
        rec.joinpath("worker.json").write_text(json.dumps(worker))
        monkeypatch.setattr(L, "alive", lambda record: True)
        monkeypatch.setattr(L, "group_members", lambda pgid: [
            {"pid": 111, "starttime": "50"}])
        if arm == "k1":
            guard = _hash_valid_receipt({"arm": "OTHER",   # wrong arm
                                         "group_members":
                                             [{"pid": 111,
                                               "starttime": "50"}]})
        else:
            guard = _hash_valid_receipt({"arm": "k2",        # malformed
                                         "group_members": [{}]})  # members
        rec.joinpath("guard.json").write_text(json.dumps(guard))
        before = rec.joinpath("guard.json").read_text()
        L._escalate(arm)
        assert kills == [] and killpgs == []
        assert rec.joinpath("guard.json").read_text() == before
        L.guard_kill(arm, {"arm": arm, "free_bytes": 1})
        assert kills == [] and killpgs == []
        # guard_kill must not re-sign invalid evidence: the wrong-arm
        # guard stands untouched (term_sent_at never written)
        assert rec.joinpath("guard.json").read_text() == before


# ---------------------------------------------------------------------------
# M2-review-8 additions: typed identity values (F1), arm/freeze-bound
# worker receipts (F2), signed identity fixtures (F3).
# ---------------------------------------------------------------------------

def test_malformed_identity_values_never_signal_or_raise(monkeypatch,
                                                          tmp_path):
    """F1: signed receipts with a list pid in guard members, or a
    null-byte pid in the worker, produce no exception, zero signals, and
    an unchanged guard file."""
    L, rec, kills, killpgs = _signal_sandbox(monkeypatch, tmp_path, "l1")
    worker = _hash_valid_receipt({"pid": 111, "starttime": "50",
                                  "launched_at": 1.0, "arm": "l1",
                                  "freeze_manifest_hash": "HASH"})
    rec.joinpath("worker.json").write_text(json.dumps(worker))
    guard = _hash_valid_receipt({"arm": "l1", "reason": "disk",
                                 "group_members": [{"pid": [],
                                                    "starttime": "50"}]})
    rec.joinpath("guard.json").write_text(json.dumps(guard))
    before = rec.joinpath("guard.json").read_text()
    monkeypatch.setattr(L, "alive", lambda record: True)
    monkeypatch.setattr(L, "group_members", lambda pgid: [
        {"pid": 111, "starttime": "50"}])
    L._escalate("l1")            # malformed member value: no TypeError
    L.guard_kill("l1", {"arm": "l1", "free_bytes": 1})
    assert kills == [] and killpgs == []
    assert rec.joinpath("guard.json").read_text() == before
    # null-byte pid in a signed worker: classification is a contained
    # error, never a ValueError from /proc path handling
    _write_arm_dir(tmp_path, "l2",
                   worker_row={"pid": "\u0000", "starttime": "1",
                               "launched_at": 1.0, "arm": "l2",
                               "freeze_manifest_hash": "HASH"})
    assert L.arm_state("l2", frozen_hash="HASH") == "error"


def test_worker_receipts_are_arm_and_freeze_bound(monkeypatch, tmp_path):
    """F2: a checksum-valid worker receipt naming a DIFFERENT arm or a
    different freeze is corrupt evidence everywhere it is consumed:
    classification error, no watchdog consumption, zero signals."""
    import launcher as L
    monkeypatch.setattr(L, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(L, "RECEIPTS", tmp_path / "receipts")
    # cross-arm: worker says arm m1, consumed for arm m2
    _write_arm_dir(tmp_path, "m2",
                   worker_row={"pid": 999999999, "starttime": "1",
                               "launched_at": 1.0, "arm": "m1",
                               "freeze_manifest_hash": "HASH"})
    assert L.arm_state("m2", frozen_hash="HASH") == "error"
    # wrong freeze: worker freeze field disagrees with the caller's
    _write_arm_dir(tmp_path, "m3",
                   worker_row={"pid": 999999999, "starttime": "1",
                               "launched_at": 1.0, "arm": "m3",
                               "freeze_manifest_hash": "OTHER"})
    assert L.arm_state("m3", frozen_hash="HASH") == "error"
    # cross-arm worker in a SIGNAL path: zero signals, guard untouched
    L2, rec, kills, killpgs = _signal_sandbox(monkeypatch, tmp_path, "m4")
    worker = _hash_valid_receipt({"pid": 111, "starttime": "50",
                                  "launched_at": 1.0, "arm": "OTHER",
                                  "freeze_manifest_hash": "HASH"})
    rec.joinpath("worker.json").write_text(json.dumps(worker))
    guard = _hash_valid_receipt({"arm": "m4", "reason": "disk",
                                 "group_members": [{"pid": 111,
                                                    "starttime": "50"}]})
    rec.joinpath("guard.json").write_text(json.dumps(guard))
    before = rec.joinpath("guard.json").read_text()
    monkeypatch.setattr(L2, "alive", lambda record: True)
    monkeypatch.setattr(L2, "group_members", lambda pgid: [
        {"pid": 111, "starttime": "50"}])
    L2._escalate("m4")
    L2.guard_kill("m4", {"arm": "m4", "free_bytes": 1})
    assert kills == [] and killpgs == []
    assert rec.joinpath("guard.json").read_text() == before
