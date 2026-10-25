"""Staged X28E trainer — post-M2-review correction package.

Binding: REGISTRATION.md "Training curriculum (staging receipts gate
progression)", "Frozen recipe and data contract", "Milestones", the
M2-review correction list (findings 1-11 + adjudications), and the standard
chain ("trainer verifies worktree AND HEAD"). This module is the staged
trainer for the seven registered arms:

  * stages 1-3 for the five tape arms (stage transitions gated on the
    registered receipt gates); stage-3-only for the two native controls
    (stages 1-2 are tape-ignition scaffolding with no native analogue — the
    declared T6 curriculum confound);
  * the frozen recipe: AdamW betas (0.9, 0.999) eps 1e-8 wd 1e-4, lr 1e-4
    CONSTANT, EFFECTIVE batch 64, grad clip 1.0, stage budgets 15k/15k/30k
    (total cap 60,000), eval cadence 500, seed 0;
  * M2-review correction F5 (registered effective-batch accumulation): each
    optimizer step consumes EXACTLY 64 examples; the 64 examples are
    processed in length-aware microbatches bound per arm in the config
    (``microbatch.token_budget`` / ``microbatch.max_sequences``, chosen so
    the predicted peak allocation fits a 40-GiB rung with margin, receipted);
    the summed loss over the 64 examples is accumulated across microbatches
    and ONE optimizer step fires per 64 examples — mathematically the
    registered single batch-64 update;
  * M2-review correction F1 (label mapping): stage-1/2 content labels
    (1..64 / 2..17) are mapped to contiguous zero-based classifier indices
    for the loss and predictions are reverse-mapped for reporting;
  * M2-review correction F6 (censor paths): an unbounded-arm cap hit
    (R-CAP at the T5 machine cap), a nonfinite loss/gradient/evaluation
    logit, or a projected single-example allocation above the registered
    40-GiB rung is a CENSORED receipt (fail-closed, never a scientific
    verdict); bounded-arm R-CAP merge is normal architecture behavior;
  * M2-review correction F7 (occupancy receipts): peak live slots are
    measured WITHIN each word's length only (to its EOS; PAD-region behavior
    after a word's EOS never enters a receipt), end-of-word occupancy is
    measured at the EOS token, and OOD occupancy is reported PER-K keyed by
    depth (the registered [0.9k, 1.1k] check consumes per-k data);
  * validation-only checkpoint selection with LATEST-BEST tie-break (``>=``):
    the selected checkpoint is the only one ever evaluated on the immutable
    OOD bins / sweep pools;
  * stage-3 supervision: end-of-word BCE on a TWO-LOGIT MARGIN reading the
    FINAL READ output (G8); stage-1/2 supervision: masked cross-entropy at
    the registered supervised positions;
  * the tape layer (the learned store machinery) persists across stages; the
    per-stage scalar-token embedding and per-stage multi-class head are
    stage-local and rebuilt at each stage transition (the optimizer is
    rebuilt with them, same frozen hyperparameters);
  * a stage that exhausts its budget without passing its receipt gate HALTS
    the arm with the stage recorded (no automatic retry; a replacement
    requires a NEW registration).

M2-BEFORE-TRAINING, MECHANICALLY ENFORCED (M2-review correction F2): the
worktree+HEAD verification is UNCONDITIONAL on every training entrypoint —
the ``--require-committed`` bypass no longer exists in any form. No
optimizer update is reachable unless ``--frozen-freeze-hash`` matches the
freeze manifest hash (SHA-256 over the canonical manifest file mapping; see
freeze_manifest.json's ``hash_rule``). Verification re-hashes EVERY
manifest-listed file from disk — package files AND the load-bearing
dependency identities bound by the manifest (M2-review correction F4: the
native ``../ndm/...`` implementation files and ``../scripts/gpu_lease.sh``,
rel paths from the package) — checks the manifest self-hash, checks the
worktree is clean and every bound byte is committed at HEAD. Verification
always targets the ACTUAL EXECUTING package: the executing module's own
directory is the verified package root (no ``--package-root`` override
exists). ``--verify-only`` verifies the complete freeze and exits WITHOUT
constructing a trainer or importing any model code (receipted structurally:
``torch_imported: false`` in the verify-only receipt).

M2-review correction F3: runtime artifacts live OUTSIDE the frozen package:
``--results-root`` defaults to the sibling house-standard results directory
``experiments/expressivity_tasks/results_x28e_memory_head/`` (created at
launch, never inside the package); launcher receipts are written there too,
so the package subtree stays verifiable-clean.

Usage (post-M2, via launcher.py only):
    python trainer.py --arm tape_B0 --frozen-freeze-hash <manifest hash>
    python trainer.py --verify-only --frozen-freeze-hash <manifest hash>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# The ACTUAL EXECUTING package is the only package that can ever be verified
# (M2-review correction F2): imports resolve against this root, and freeze
# verification hashes exactly this tree.
EXECUTING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXECUTING_ROOT.parents[2]
sys.path.insert(0, str(EXECUTING_ROOT))       # eng/, generators/
sys.path.insert(0, str(REPO_ROOT))            # ndm native controls

# M2-review correction F3: runtime artifacts NEVER inside the frozen package.
RESULTS_ROOT_DEFAULT = EXECUTING_ROOT.parent / "results_x28e_memory_head"

STAGE_GATE = 0.99              # registered receipt-gate threshold (stages 1-2)
EVAL_CADENCE = 500             # registered validation cadence (updates)
BATCH = 64                     # registered EFFECTIVE batch (F5 accumulation)
TOTAL_CAP = 60_000
STAGE_BUDGETS = {"stage1": 15_000, "stage2": 15_000, "stage3": 30_000}

# Stage-1/2 content-label offsets (M2-review correction F1): stage-1 content
# ids are 1..64 (stage1_mqar.PAD=0), stage-2 content ids are 2..17
# (stage2_append_necessity.PAD=0, GO=1). Content labels map to contiguous
# zero-based classifier indices for the loss; predictions reverse-map for
# reporting.
STAGE1_LABEL_OFFSET = 1
STAGE2_LABEL_OFFSET = 2

# M2-review correction F5: the registered allocation rung. Per-arm
# microbatch budgets are bound in configs/*.json so the PREDICTED peak
# allocation of every in-plan microbatch fits this rung with margin
# (receipts/microbatch_allocation_model.json receipts the prediction).
MICROBATCH_RUNG_BYTES = 40 * 2 ** 30

EXIT_OK = 0
EXIT_NO_HASH = 2
EXIT_FREEZE_FAILED = 3
EXIT_STAGE_GATE_FAILED = 4
EXIT_CENSORED_NONFINITE = 5


# ---------------------------------------------------------------------------
# Freeze verification (M2-before-training, mechanical; F2/F4 corrections).
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hash(files: Dict[str, str]) -> str:
    """SHA-256 over the canonical manifest mapping: sorted, newline-
    terminated lines ``<sha256>  <relpath>`` (freeze_manifest.json's
    ``hash_rule``; one spelling of the concept). Includes the F4 dependency
    identities (rel paths like ``../ndm/...``) — changes there invalidate
    the freeze."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{files[rel]}  {rel}\n".encode())
    return h.hexdigest()


def receipt_payload_hash(row: Dict) -> str:
    """SHA-256 over the canonical JSON of a receipt payload (sorted keys,
    no whitespace) — the house-standard hash-checked receipt (M2-review
    correction F9). The ``payload_sha256`` field itself is excluded."""
    payload = {k: v for k, v in row.items() if k != "payload_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=int).encode()).hexdigest()


def verify_freeze(frozen_hash: str) -> Dict:
    """Fail-closed verification of the ACTUAL EXECUTING package. UNCONDITIONAL
    worktree+HEAD verification (M2-review correction F2 — no bypass exists).
    Raises SystemExit (exit 3) on ANY mismatch: a manifest-listed file
    missing or changed on disk (package files AND F4 dependency identities),
    a manifest self-hash mismatch, an argument-hash mismatch, a dirty
    worktree for any bound path, or a bound file not committed at HEAD."""
    def fail(why: str):
        print(f"freeze verification FAILED ({why}); zero updates.", file=sys.stderr)
        raise SystemExit(EXIT_FREEZE_FAILED)

    package_root = EXECUTING_ROOT
    manifest_path = package_root / "freeze_manifest.json"
    if not manifest_path.is_file():
        fail("freeze_manifest.json missing")
    manifest = json.loads(manifest_path.read_text())
    files: Dict[str, str] = manifest["files"]
    for rel, expected in sorted(files.items()):
        path = (package_root / rel).resolve()
        if not path.is_file():
            fail(f"missing {rel}")
        if sha256_file(path) != expected:
            fail(f"bytes changed: {rel}")
    if manifest_hash(files) != manifest["manifest_hash"]:
        fail("manifest self-hash mismatch")
    if frozen_hash != manifest["manifest_hash"]:
        fail("--frozen-freeze-hash does not match the freeze manifest hash")
    # UNCONDITIONAL standard chain: clean worktree AND HEAD bytes for every
    # bound path — the package subtree AND the F4 dependency identities.
    rel_prefix = package_root.relative_to(REPO_ROOT).as_posix()
    import os
    bound_paths = [rel_prefix] + [
        os.path.normpath(f"{rel_prefix}/{rel}")
        for rel in sorted(files) if rel.startswith("../")]
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--",
         *bound_paths],
        capture_output=True, text=True).stdout.strip()
    if dirty:
        fail(f"worktree not clean for the package and its bound dependencies:\n"
             f"{dirty}")
    disk_manifest_hash = sha256_file(manifest_path)
    for rel in sorted(list(files) + ["freeze_manifest.json"]):
        git_rel = os.path.normpath(f"{rel_prefix}/{rel}")
        try:
            blob = subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{git_rel}"])
        except subprocess.CalledProcessError:
            fail(f"not committed at HEAD: {rel}")
        expected = (disk_manifest_hash if rel == "freeze_manifest.json"
                   else files[rel])
        if hashlib.sha256(blob).hexdigest() != expected:
            fail(f"HEAD bytes differ from the freeze: {rel}")
    # M2-review-2 finding 3: VALIDATE the installed environment identities
    # bound in the manifest (importlib.metadata reads — no package import,
    # verify-only-safe). The manifest binds the versions measured at freeze
    # build; any drift fails closed.
    import importlib.metadata as _md
    env = manifest.get("dependency_binding", {}).get("environment", {})
    dist_names = env.get("distribution_names", {"fla": "flash-linear-attention"})
    for key, dist in (("fla_version", dist_names.get("fla",
                                                        "flash-linear-attention")),
                      ("torch_version", "torch")):
        bound = env.get(key)
        if bound is None:
            continue
        try:
            installed = _md.version(dist)
        except Exception as exc:
            fail(f"environment identity unreadable: {dist} ({exc})")
        if installed != bound:
            fail(f"environment identity drift: {dist} installed {installed} "
                 f"!= frozen {bound}")
    return manifest


# ---------------------------------------------------------------------------
# Label mapping (M2-review correction F1 — pure, unit-tested).
# ---------------------------------------------------------------------------
def labels_to_classifier_indices(labels, offset: int):
    """Content labels (e.g. stage-1 1..64, stage-2 2..17) -> contiguous
    zero-based classifier indices (0..n_classes-1) for the cross-entropy."""
    return np.asarray(labels) - offset


def classifier_indices_to_labels(indices, offset: int):
    """Zero-based classifier indices -> content labels, for reporting."""
    return np.asarray(indices) + offset


# ---------------------------------------------------------------------------
# Microbatch plan + allocation model (M2-review correction F5 — pure).
# ---------------------------------------------------------------------------
def microbatch_plan(lengths: Sequence[int], token_budget: int,
                    max_sequences: int) -> List[List[int]]:
    """Length-aware microbatch packing (registered effective-batch
    accumulation). Splits example indices (in pool order) into microbatches
    subject to the per-arm bounds: ``n_mb * max(length in mb) <=
    token_budget`` (the padded allocation product) and ``n_mb <=
    max_sequences``. A single example that alone exceeds the token budget
    forms its own microbatch (over-plan single, receipted by the allocation
    model). Pure — unit-tested."""
    mbs: List[List[int]] = []
    cur: List[int] = []
    cur_max = 0
    for i in range(len(lengths)):
        length_i = int(lengths[i])
        new_max = max(cur_max, length_i)
        if cur and ((len(cur) + 1) * new_max > token_budget
                    or len(cur) + 1 > max_sequences):
            mbs.append(cur)
            cur, cur_max = [], 0
            new_max = length_i
        cur.append(i)
        cur_max = new_max
    if cur:
        mbs.append(cur)
    return mbs


def allocation_model_bytes(n_seq: int, t_pad: int, slots: int,
                           n_heads: int = 4, key_dim: int = 64,
                           value_dim: int = 128, ckpt_interval: int = 16,
                           backward: bool = True, spilled: bool = True) -> int:
    """Predicted peak allocation (bytes) of one fused tape microbatch at the
    frozen constants (G7/G14: NH=4, K=64, V=128, CKPT_INTERVAL=16, bf16
    sparse checkpoints, fp32 per-step replay cache). The registered
    per-arm budgets are chosen so every in-plan microbatch fits
    MICROBATCH_RUNG_BYTES with margin (receipts/microbatch_allocation_model.
    json). Pure — unit-tested; the per-arm smokes receipt the measured
    peaks against this prediction."""
    per_ckpt_seq = n_heads * slots * ((key_dim + value_dim) * 2 + 9)
    ckpts = (t_pad // ckpt_interval + 1) * n_seq * per_ckpt_seq
    t_freed = n_seq * t_pad * n_heads * slots                 # int8 trace
    out = n_seq * t_pad * n_heads * value_dim * 4
    scratch = (n_seq * n_heads * slots
               * (key_dim * 4 + value_dim * 4 + 4 + 1 + 4 + 4 + 4)
               ) if spilled else 0
    peak_fwd = ckpts + t_freed + out + scratch
    if not backward:
        return peak_fwd
    bwd_cache = n_seq * n_heads * (
        (ckpt_interval + 1) * slots * (key_dim + value_dim) * 4
        + slots * 4 + slots + slots * 4)
    g_adj = n_seq * n_heads * slots * (key_dim * 4 + value_dim * 4 + 4)
    dp = n_seq * n_heads * slots * 4
    return peak_fwd + bwd_cache + g_adj + dp


# ---------------------------------------------------------------------------
# Models.
# ---------------------------------------------------------------------------
def build_tape_model(arm_cfg: Dict):
    """The tape-arm model: per-stage scalar-token embedding -> fused tape
    layer (G12: W_o over the concatenated per-head read outputs + residual)
    -> per-stage heads. The layer and the binary head persist across the
    curriculum; embed/multi-class heads are stage-local.

    ``tape_appendonly`` arm wiring (registered: pop disabled, decay + merge
    retained; M2-review correction F10): the head is constructed EXACTLY ONCE
    via the ``HEAD_CLS`` seam and pop is disabled in the forward wiring
    (pop strengths identically zero — strictly below ``pop_threshold`` so
    the read-path POP never fires architecturally). No second random
    initialization exists: at the same seed ``tape_unbounded`` and
    ``tape_appendonly`` share identical non-pop initial parameters
    (unit-tested in eng/tests/test_trainer_wiring.py)."""
    import torch
    import torch.nn as nn
    from eng.tape_head import TapeHeadConfig, init_state
    from eng.fused import FusedTapeHead, FusedTapeLayer
    from eng.fused.tape_kernels import (
        DEFAULT_CKPT_INTERVAL, DEFAULT_MAX_RESIDENT_SLOTS, DEFAULT_TILE_S,
        fused_tape_scan,
    )

    class PopDisabledFusedTapeHead(FusedTapeHead):
        """``tape_appendonly`` wiring: pop strengths identically zero in the
        forward pass. Construction is inherited unchanged (F10)."""

        def forward(self, x, state=None, collect_trace=True):
            B, T, _ = x.shape
            cfg = self.cfg
            head = self.head
            NH, K, V = cfg.n_heads, cfg.key_dim, cfg.value_dim
            k = head.W_k(x).view(B, T, NH, K)
            q = head.W_q(x).view(B, T, NH, K)
            v = head.W_v(x).view(B, T, NH, V)
            write_strength = torch.sigmoid(head.W_write(x))
            pop_strength = torch.zeros(B, T, NH, device=x.device,
                                      dtype=torch.float32)   # pop OFF
            append_logit = head.W_append(x)
            if state is None:
                state = init_state(B, cfg, x.device, torch.float32)
            return fused_tape_scan(
                k, q, v, write_strength, pop_strength, append_logit, state, cfg,
                append_err_coef=head.append_err_coef,
                ckpt_interval=self.ckpt_interval,
                checkpoint_dtype=self.checkpoint_dtype,
                collect_trace=collect_trace,
                tile_s=self.tile_s, max_resident=self.max_resident)

    class PopDisabledFusedTapeLayer(FusedTapeLayer):
        # F10: the head is constructed exactly once, through the HEAD_CLS
        # seam — identical construction order and RNG consumption as
        # FusedTapeLayer, so the same seed gives identical parameters.
        HEAD_CLS = PopDisabledFusedTapeHead

    class TapeWordModel(nn.Module):
        """Stage-local embed/head attach points; shared tape layer + binary head."""

        def __init__(self, d_model, cfg, pop_enabled):
            super().__init__()
            self.cfg = cfg
            self.d_model = d_model
            layer_cls = FusedTapeLayer if pop_enabled else PopDisabledFusedTapeLayer
            self.layer = layer_cls(
                d_model, cfg, ckpt_interval=DEFAULT_CKPT_INTERVAL,
                checkpoint_dtype=torch.bfloat16, tile_s=DEFAULT_TILE_S,
                max_resident=DEFAULT_MAX_RESIDENT_SLOTS)
            self.head_binary = nn.Linear(d_model, 2)
            self.embed: Optional[nn.Embedding] = None
            self.head_multi: Optional[nn.Linear] = None

        def attach_multiclass_stage(self, vocab_size: int, n_classes: int):
            """Fresh stage-local embedding + multi-class head (stage 1/2)."""
            self.embed = nn.Embedding(vocab_size, self.d_model)
            self.head_multi = nn.Linear(self.d_model, n_classes)

        def attach_binary_stage(self, vocab_size: int):
            """Fresh stage-local embedding for the ladder (stage 3)."""
            self.embed = nn.Embedding(vocab_size, self.d_model)

        def forward(self, x, collect_trace=True):
            return self.layer(self.embed(x), collect_trace=collect_trace)

        def binary_logits(self, x, lengths, collect_trace=False):
            """End-of-word two-logit margin reading the FINAL READ output:
            y_T = W_o(final read) + residual at the final (EOS) token (G8;
            the final token's own write is gradient-dead)."""
            y, _, _, trace = self.forward(x, collect_trace=collect_trace)
            idx = torch.arange(x.shape[0], device=x.device)
            return self.head_binary(y[idx, lengths - 1]), trace

        def multi_logits(self, x, collect_trace=False):
            y, _, _, trace = self.forward(x, collect_trace=collect_trace)
            return self.head_multi(y), trace

    constants = arm_cfg["constants"]
    budget = arm_cfg.get("budget_slots")
    cfg = TapeHeadConfig(
        key_dim=arm_cfg["head"]["key_dim"], value_dim=arm_cfg["head"]["value_dim"],
        n_heads=arm_cfg["head"]["n_heads"],
        budget=budget,                       # None => unbounded arm (T5 cap)
        unbounded_cap=constants["unbounded_cap_slots"],
        match_threshold=constants["match_threshold"],
        eps_residual=constants["eps_residual"],
        pop_threshold=constants["pop_threshold"],
        write_enable_threshold=constants["write_enable_threshold"],
        free_threshold=constants["free_threshold"],
        decay_rate=constants["decay_rate"],
        usage_decay=constants["usage_decay"])
    return TapeWordModel(arm_cfg["d_model"], cfg, arm_cfg["pop_enabled"])


def build_native_model(arm_cfg: Dict, vocab_size: int):
    """The native controls. M2-review correction F11: the CONSTRUCTOR
    totals at the frozen vocab-5 encoding are 562,834 (gdn2_native) and
    569,888 (e97_native) — the shared embedding/output weight
    (vocab_size * dim, tied) changes the whole-model count vs the inherited
    X27 vocab-8 labels (563,218 / 570,464); the inherited BACKBONE
    dimensions are unchanged. Receipted by construction in
    receipts/native_parameter_totals.json. The binary decision reads the
    final position's LM logits[:, :2] margin (X23E house convention)."""
    from ndm.models.hybrid_ladder import HybridLadderLM
    size = arm_cfg["constructor_size"]
    e97 = arm_cfg["registration_name"] == "e97_native"
    model = HybridLadderLM(
        vocab_size=vocab_size, **size,
        layer_pattern=["E97" if e97 else "fla-gdn"],
        layer_kwargs=[({"state_activation": "tanh", "raw_write": False} if e97
                       else {"allow_neg_eigval": True})],
        use_triton_e88=e97, mlp_ratio=arm_cfg["mlp_ratio"])
    if e97:
        for layer in model.layers:
            layer.recurrent_state_precision = "legacy"
            layer.fused_inference = True
    return model


# ---------------------------------------------------------------------------
# Occupancy receipts (M2-review correction F7: within-word, at-EOS, per-k).
# ---------------------------------------------------------------------------
def trace_live_series(trace):
    """Per-token live-slot series [B, T, NH] from the routing trace:
    live_t = cumulative(append allocations - traced frees), starting from
    the fresh (empty) store."""
    import torch
    alloc = (trace.alloc_target >= 0).to(torch.int64)          # [B,T,NH]
    freed = (trace.freed != 0).to(torch.int64).sum(-1)         # [B,T,NH]
    return (alloc - freed).cumsum(dim=1)


def peak_live_from_trace(trace):
    """Peak live-slot count per (batch, head) over the full traced range
    (stage-2 instrumentation pools: the whole sequence is the word)."""
    return trace_live_series(trace).max(dim=1).values           # [B,NH]


def peak_and_eos_live_from_trace(trace, lengths):
    """M2-review correction F7: peak live slots measured WITHIN each word's
    length only (tokens at/after its EOS are excluded — PAD-region behavior
    never enters an occupancy receipt), and the end-of-word live-slot count
    AT the EOS token. Returns (peaks [B,NH], eos_live [B,NH])."""
    import torch
    live_t = trace_live_series(trace)                          # [B,T,NH]
    B, T, _ = live_t.shape
    lengths = torch.as_tensor(lengths, device=live_t.device)
    in_word = (torch.arange(T, device=live_t.device)[None, :]
               < lengths[:, None])                             # [B,T]
    peaks = live_t.masked_fill(~in_word[:, :, None], 0).max(dim=1).values
    eos_live = live_t[torch.arange(B, device=live_t.device), lengths - 1]
    return peaks, eos_live


def per_depth_occupancy(depths, peaks, eos_live, labels,
                        accept_label: int = 1) -> Dict:
    """Per-k occupancy receipt keyed by depth (M2-review correction F7):
    for every depth k present in the pool, the mean peak live slots within
    the word, the registered [0.9k, 1.1k] tolerance check on that mean, and
    the mean end-of-word (EOS) live slots on accepts (the pop receipt).
    ``peaks``/``eos_live`` are per-example scalars (mean over heads),
    aligned with ``depths``/``labels``. Pure numpy — unit-tested."""
    depths = np.asarray(depths)
    peaks = np.asarray(peaks, dtype=np.float64).reshape(len(depths), -1)
    eos_live = np.asarray(eos_live, dtype=np.float64).reshape(len(depths), -1)
    labels = np.asarray(labels)
    per_k: Dict = {}
    for k in np.unique(depths):
        sel = depths == k
        mean_peak = float(peaks[sel].mean())
        acc = sel & (labels == accept_label)
        entry: Dict = {
            "k": int(k), "n": int(sel.sum()),
            "mean_peak_live_slots_within_word": mean_peak,
            "mean_peak_within_0.9k_1.1k": bool(
                0.9 * int(k) <= mean_peak <= 1.1 * int(k)),
            "n_accepts": int(acc.sum()),
            "mean_eos_live_slots_on_accepts": (
                float(eos_live[acc].mean()) if acc.any() else None),
        }
        per_k[f"k{int(k)}"] = entry
    return per_k


# ---------------------------------------------------------------------------
# Entry-context validation (M2-review-3 finding 2): a CONSTRUCTOR INVARIANT,
# not a main() check — no path to StagedTrainer construction without a
# validated launcher/lease context and an out-of-package results root.
# ---------------------------------------------------------------------------
def _validate_entry_context(arm: str, results_root: Path,
                            frozen_hash: str) -> None:
    """Raises SystemExit(EXIT_FREEZE_FAILED) on any entry-contract
    violation. Pure: no torch import, no model construction."""
    root = EXECUTING_ROOT.resolve()
    resolved = Path(results_root).resolve()
    if root == resolved or root in resolved.parents:
        print("REFUSED: --results-root must be OUTSIDE the frozen package "
              "(F3).", file=sys.stderr)
        raise SystemExit(EXIT_FREEZE_FAILED)
    launch_json = os.environ.get("X28E_LAUNCHER_LAUNCH_JSON")
    if not launch_json:
        print("REFUSED: training requires a launcher context "
              "(X28E_LAUNCHER_LAUNCH_JSON); the trainer is not a valid "
              "entrypoint outside the launcher/lease/watchdog chain.",
              file=sys.stderr)
        raise SystemExit(EXIT_FREEZE_FAILED)
    try:
        launch = json.loads(Path(launch_json).read_text())
    except Exception as exc:
        print(f"REFUSED: unreadable launcher context ({exc}).",
              file=sys.stderr)
        raise SystemExit(EXIT_FREEZE_FAILED)
    # M2-review-7 finding 3: the context receipt is CHECKSUM-VALIDATED like
    # every other receipt — unsigned or stale-checksum launch evidence is
    # refused even when its fields look right (the launcher signs launch
    # receipts via write_json, so genuine contexts always pass).
    if not isinstance(launch, dict) or "payload_sha256" not in launch:
        print("REFUSED: launcher context is not a signed receipt.",
              file=sys.stderr)
        raise SystemExit(EXIT_FREEZE_FAILED)
    stored = launch["payload_sha256"]
    computed = receipt_payload_hash(
        {k: v for k, v in launch.items() if k != "payload_sha256"})
    if stored != computed:
        print("REFUSED: launcher context checksum is stale or invalid.",
              file=sys.stderr)
        raise SystemExit(EXIT_FREEZE_FAILED)
    if launch.get("arm") != arm or \
            launch.get("freeze_manifest_hash") != frozen_hash:
        print("REFUSED: launcher context does not match this arm and/or "
              "freeze hash.", file=sys.stderr)
        raise SystemExit(EXIT_FREEZE_FAILED)


# ---------------------------------------------------------------------------
# The staged trainer.
# ---------------------------------------------------------------------------
class StagedTrainer:
    def __init__(self, arm: str, results_root: Path, frozen_hash: str,
                 device: str = "cuda:0"):
        self.arm = arm
        self.device = device
        self.freeze_verified = False   # the optimizer-step kill switch
        # M2-review-3 finding 2: the ENTRY CONTRACT is a CONSTRUCTOR
        # INVARIANT — no path to StagedTrainer construction without a
        # validated launcher/lease context and an out-of-package
        # results root (centralized here; main() delegates; --verify-only
        # never constructs this class). The freeze is verified BEFORE the
        # launcher-context read so the context file's freeze hash is
        # checked against the VERIFIED manifest hash.
        self.manifest = verify_freeze(frozen_hash)
        self.freeze_verified = True
        _validate_entry_context(arm, results_root, frozen_hash)
        import torch
        torch.manual_seed(0)
        np.random.seed(0)
        self.arm_cfg = json.loads(
            (EXECUTING_ROOT / "configs" / f"{arm}.json").read_text())
        self.recipe = self.arm_cfg["recipe"]
        self.is_tape = self.arm_cfg["architecture"] == "slot_store_tape_head"
        self.is_unbounded_tape = bool(
            self.is_tape and self.arm_cfg.get("budget_slots") is None)
        self.micro = self.arm_cfg["microbatch"]          # F5, freeze-bound
        if self.is_tape:
            self.model = build_tape_model(self.arm_cfg).to(device)
            self.slots = self.model.cfg.slots
        else:
            from generators.stage3_ladder import LADDER_VOCAB
            self.model = build_native_model(self.arm_cfg, LADDER_VOCAB).to(device)
            self.slots = 0
        self.run_root = results_root / f"{arm}_seed0"
        self.receipts_root = results_root / "receipts" / f"{arm}_seed0"
        self.markers_root = self.run_root / "markers"
        for d in (self.run_root, self.receipts_root, self.markers_root):
            d.mkdir(parents=True, exist_ok=True)
        self.optimizer = None
        self.total_updates = 0
        self.best: Dict = {"metric": -math.inf, "update": None, "stage": None}
        self.training_log = self.run_root / "training.jsonl"
        self._write_json(self.run_root / "trainer_identity.json", {
            "arm": arm, "pid": __import__("os").getpid(), "host":
            __import__("socket").gethostname(), "started_at": time.time(),
            "frozen_freeze_hash": frozen_hash,
            "freeze_manifest_files": len(self.manifest["files"])})

    # -- receipts (hash-validated per the house standard, F9) ---------------
    @staticmethod
    def _write_json(path: Path, row: Dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(row)
        row["payload_sha256"] = receipt_payload_hash(row)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(row, indent=2, default=int) + "\n")
        tmp.replace(path)

    def _log(self, row: Dict):
        with self.training_log.open("a") as f:
            f.write(json.dumps(row, default=int) + "\n")

    def _marker(self, name: str, row: Dict):
        self._write_json(self.markers_root / f"{name}.json", row)

    # -- microbatch plan (F5) ------------------------------------------------
    def _plan_microbatches(self, lengths) -> List[List[int]]:
        """Length-aware plan at the arm's freeze-bound budgets, with the
        fail-closed rung check: an above-rung SINGLE example is CENSORED
        infrastructure (a multi-example above-rung microbatch cannot be
        produced by the bound budgets and fails closed as a config bug)."""
        lengths = np.asarray(lengths)
        mbs = microbatch_plan(lengths, self.micro["token_budget"],
                              self.micro["max_sequences"])
        if self.is_tape:
            for mb in mbs:
                t_pad = int(lengths[mb].max())
                predicted = allocation_model_bytes(len(mb), t_pad, self.slots)
                if predicted > MICROBATCH_RUNG_BYTES:
                    if len(mb) == 1:
                        self._censor(
                            f"projected single-example allocation "
                            f"{predicted / 2**30:.1f} GiB at {t_pad} tokens "
                            f"exceeds the registered 40-GiB rung "
                            f"(right-censored infrastructure)")
                    raise SystemExit(
                        "microbatch plan produced an above-rung multi-example "
                        "microbatch (bound-budget config bug); zero updates")
        return mbs

    # -- censor paths (F6, fail-closed) --------------------------------------
    def _censor(self, why: str):
        """CENSORED infrastructure, never a training verdict (X27
        softplus-small receipt; M2-review correction F6: unbounded cap hit,
        nonfinite loss/gradient/evaluation logits, projected above-rung
        single example). Fail closed."""
        self._write_json(self.receipts_root / "censor.json", {
            "arm": self.arm, "censor": why, "at": time.time(),
            "total_updates": self.total_updates,
            "note": "CENSORED infrastructure; no failure-to-learn adjudicated; "
                    "any replacement arm requires a NEW registration"})
        raise SystemExit(EXIT_CENSORED_NONFINITE)

    def _check_trace_censor(self, trace):
        """Unbounded-arm cap hit (R-CAP at the T5 machine cap) is a CENSOR
        receipt; bounded-arm R-CAP merge is normal architecture behavior."""
        if not (self.is_unbounded_tape and trace is not None):
            return
        import torch
        from eng.routing_trace import R_CAP
        if bool((trace.case == R_CAP).any()):
            self._censor(
                "unbounded arm cap hit: R-CAP merge at the T5 machine cap "
                "(131,072 slots) — right-censored infrastructure, never a "
                "scientific verdict")

    def _check_eval_logits(self, logits):
        """Nonfinite evaluation logits are CENSORED, never an accuracy
        number (M2-review correction F6)."""
        import torch
        if not bool(torch.isfinite(logits).all()):
            self._censor("nonfinite evaluation logits (CENSORED; no accuracy "
                         "number is reported for this evaluation)")

    # -- optimizer (the M2 kill switch lives here) ---------------------------
    def _ensure_optimizer(self):
        """Constructed ONLY after freeze verification; the registered frozen
        recipe, verbatim. Rebuilt at stage transitions to include the
        stage-local embedding/head parameters."""
        if self.optimizer is None:
            if not self.freeze_verified:
                raise SystemExit("no optimizer without a verified freeze "
                                 "(M2-before-training, mechanical)")
            import torch
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.recipe["lr"],
                betas=tuple(self.recipe["betas"]), eps=self.recipe["eps"],
                weight_decay=self.recipe["weight_decay"])

    def _begin_effective_batch(self):
        """One optimizer step consumes exactly the 64 examples of the
        effective batch (F5): gradients accumulate across the batch's
        microbatches from a clean zero."""
        self._ensure_optimizer()
        self.optimizer.zero_grad(set_to_none=True)

    def _backward_microbatch(self, loss):
        """Accumulate one microbatch's summed-loss contribution into the
        effective batch's gradients (F5). Fail-closed nonfinite checks."""
        import torch
        if not torch.isfinite(loss.detach()):
            self._censor("nonfinite loss")
        loss.backward()
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                self._censor("nonfinite gradient")

    def _optimizer_step(self):
        """The single optimizer step of the effective batch (F5): clip, the
        registered update-count bookkeeping, one AdamW step over the
        accumulated (summed-loss) gradients — mathematically the registered
        single batch-64 update."""
        import torch
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.recipe["grad_clip"])
        self.total_updates += 1
        if self.total_updates > TOTAL_CAP:
            raise SystemExit("registered 60,000-update cap exceeded "
                             "(right-censoring boundary)")
        self.optimizer.step()

    # -- checkpoint selection (validation-only, latest-best tie-break) --------
    def _select_checkpoint(self, stage: str, update: int, metric: float):
        import torch
        if metric >= self.best["metric"]:      # tie-break: the LATEST best wins
            self.best = {"metric": float(metric), "update": update, "stage": stage}
            torch.save(self.model.state_dict(), self.run_root / "best.pt")
            self._write_json(self.run_root / "checkpoint.json", {
                "arm": self.arm, "stage": stage, "update": update,
                "validation_metric": float(metric),
                "selected_by": "validation-only; tie-break latest-best"})

    def _batches(self, n: int, rng: np.random.Generator):
        """Exactly-64 example batches (M2-review-2 finding 5): the
        registered rule is 'each optimizer step consumes exactly 64
        examples'; the trailing short slice is DROPPED at the epoch
        boundary (at 70,000 examples: 1,093 full steps + a 48-example
        remainder dropped). No example is lost ACROSS epochs — each epoch
        re-permutes the full pool with the registered seed sequence, so a
        different remainder is dropped each epoch (the epoch-boundary
        policy, registered in amendment F5)."""
        perm = rng.permutation(n)
        n_full = (n // BATCH) * BATCH
        for i in range(0, n_full, BATCH):
            yield perm[i:i + BATCH]

    @staticmethod
    def _pool_lengths(pool) -> np.ndarray:
        """Per-example token lengths. Stage-1/2 pools are fixed-length words
        (the whole sequence is the word); stage-3 pools carry lengths."""
        if len(pool) >= 3 and isinstance(pool[2], np.ndarray) \
                and pool[2].ndim == 1:
            return np.asarray(pool[2])
        return np.full(pool[0].shape[0], pool[0].shape[1], dtype=np.int64)

    # -- stages 1 and 2 (tape arms only): masked multi-class recall ----------
    def _run_multiclass_stage(self, stage: str, pools, vocab_size: int,
                             n_classes: int, classes: np.ndarray,
                             label_offset: int):
        import torch
        import torch.nn.functional as F
        from generators.stage1_mqar import mean_per_class_recall
        if not self.is_tape:
            raise ValueError(f"{stage} has no native analogue (T6 confound)")
        self.best = {"metric": -math.inf, "update": None, "stage": None}
        self.model.attach_multiclass_stage(vocab_size, n_classes)
        self.model.to(self.device)
        self.optimizer = None                     # include the stage-local params
        train, val = pools["train"], pools["validation"]
        lengths_tr = self._pool_lengths(train)
        rng = np.random.default_rng(0)
        updates = 0
        budget = STAGE_BUDGETS[stage]
        while updates < budget:
            for idx in self._batches(train[0].shape[0], rng):
                # F5: one optimizer step per 64 examples; the summed loss is
                # normalized by the batch's TOTAL supervised positions, so
                # the accumulated gradient is exactly a single batch-64
                # mean-cross-entropy step.
                self._begin_effective_batch()
                norm = int(train[2][idx].sum())
                if norm == 0:
                    raise SystemExit("effective batch has no supervised "
                                     "positions (data contract violation)")
                for mb in self._plan_microbatches(lengths_tr[idx]):
                    loc = idx[mb]
                    t_pad = int(lengths_tr[loc].max())
                    x = torch.from_numpy(
                        np.ascontiguousarray(train[0][loc][:, :t_pad])).to(self.device)
                    y = torch.from_numpy(
                        np.ascontiguousarray(train[1][loc][:, :t_pad])).to(self.device)
                    m = torch.from_numpy(
                        np.ascontiguousarray(train[2][loc][:, :t_pad])).to(self.device)
                    # Map labels to classifier indices on the CPU numpy array
                    # BEFORE device transfer (M2-review-2 finding 1: mapping a
                    # CUDA tensor through np.asarray crashes on the first
                    # training batch; the pure-numpy helpers stay the single
                    # mapping implementation).
                    y_idx = torch.from_numpy(np.ascontiguousarray(
                        labels_to_classifier_indices(
                            train[1][loc][:, :t_pad], label_offset)
                    )).to(self.device)
                    self.model.train()
                    logits, trace = self.model.multi_logits(
                        x, collect_trace=self.is_unbounded_tape)
                    self._check_trace_censor(trace)
                    flat_m = m.reshape(-1)
                    loss = F.cross_entropy(
                        logits.reshape(-1, n_classes)[flat_m],
                        y_idx.reshape(-1)[flat_m],
                        reduction="sum") / norm
                    self._backward_microbatch(loss)
                self._optimizer_step()
                updates += 1
                if updates % EVAL_CADENCE == 0 or updates >= budget:
                    metric = self._validate_multiclass(
                        val, n_classes, classes, mean_per_class_recall,
                        label_offset)
                    self._select_checkpoint(stage, updates, metric)
                    self._log({"stage": stage, "update": updates,
                               "validation_mean_per_class_recall": metric})
                    if metric >= STAGE_GATE:      # receipt gate: complete early
                        return metric, True
                if updates >= budget:
                    break
        return self.best["metric"], False

    def _validate_multiclass(self, val, n_classes, classes, metric_fn,
                             label_offset: int):
        import torch
        self.model.eval()
        lengths = self._pool_lengths(val)
        preds, trues = [], []
        with torch.no_grad():
            for mb in self._plan_microbatches(lengths):
                t_pad = int(lengths[mb].max())
                x = torch.from_numpy(
                    np.ascontiguousarray(val[0][mb][:, :t_pad])).to(self.device)
                logits, trace = self.model.multi_logits(
                    x, collect_trace=self.is_unbounded_tape)
                self._check_eval_logits(logits)          # F6 censor path
                self._check_trace_censor(trace)
                m = val[2][mb][:, :t_pad].reshape(-1)
                # F1: reverse-map classifier indices to content labels.
                p_idx = logits.reshape(-1, n_classes).argmax(-1).cpu().numpy()
                preds.append(classifier_indices_to_labels(p_idx, label_offset)[m])
                trues.append(val[1][mb][:, :t_pad].reshape(-1)[m])
        return metric_fn(np.concatenate(preds), np.concatenate(trues), classes)

    # -- stage 3: the ladder (all arms) --------------------------------------
    def _forward_binary(self, x, lengths, collect_trace: bool):
        """One forward: the end-of-word two-logit margin AND (for tape arms,
        when traced) the final state and routing trace. Native arms read the
        final position's LM logits[:, :2] margin (X23E convention)."""
        import torch
        idx = torch.arange(x.shape[0], device=x.device)
        if self.is_tape:
            y, _, state, trace = self.model.forward(
                x, collect_trace=collect_trace)
            return self.model.head_binary(y[idx, lengths - 1]), state, trace
        return self.model(x)[idx, lengths - 1, :2], None, None

    def _binary_logits(self, x, lengths, collect_trace: bool = False):
        logits, _, _ = self._forward_binary(x, lengths, collect_trace)
        return logits

    def _validate_ladder(self, val):
        import torch
        from generators.stage3_ladder import binary_report
        self.model.eval()
        lengths = self._pool_lengths(val)
        preds, trues = [], []
        with torch.no_grad():
            for mb in self._plan_microbatches(lengths):
                t_pad = int(lengths[mb].max())
                x = torch.from_numpy(
                    np.ascontiguousarray(val[0][mb][:, :t_pad])).to(self.device)
                ln = torch.from_numpy(lengths[mb]).to(self.device)
                logits, _, trace = self._forward_binary(
                    x, ln, collect_trace=self.is_tape)
                self._check_eval_logits(logits)          # F6 censor path
                self._check_trace_censor(trace)
                preds.append((logits[:, 1] > logits[:, 0]).long().cpu().numpy())
                trues.append(val[1][mb])
        return binary_report(np.concatenate(trues),
                             np.concatenate(preds))["balanced_accuracy"]

    def _run_ladder_stage(self):
        import torch
        import torch.nn.functional as F
        from generators.stage3_ladder import (
            LADDER_VOCAB, ladder_train_pool, ladder_validation_pool,
        )
        self.best = {"metric": -math.inf, "update": None, "stage": None}
        if self.is_tape:
            self.model.attach_binary_stage(LADDER_VOCAB)
            self.model.to(self.device)
            self.optimizer = None                 # include the stage-local embed
        train = ladder_train_pool()
        val = ladder_validation_pool()
        lengths = self._pool_lengths(train)
        rng = np.random.default_rng(0)
        updates = 0
        budget = STAGE_BUDGETS["stage3"]
        while updates < budget:
            for idx in self._batches(train[0].shape[0], rng):
                self._begin_effective_batch()
                for mb in self._plan_microbatches(lengths[idx]):
                    loc = idx[mb]
                    t_pad = int(lengths[loc].max())
                    x = torch.from_numpy(
                        np.ascontiguousarray(train[0][loc][:, :t_pad])).to(self.device)
                    labels = torch.from_numpy(train[1][loc]).to(self.device)
                    ln = torch.from_numpy(lengths[loc]).to(self.device)
                    self.model.train()
                    logits, _, trace = self._forward_binary(
                        x, ln, collect_trace=self.is_unbounded_tape)
                    self._check_trace_censor(trace)
                    margin = logits[:, 1] - logits[:, 0]
                    # F5: summed BCE over the 64 examples of the effective
                    # batch, accumulated across microbatches == the single
                    # batch-64 summed-loss step.
                    loss = F.binary_cross_entropy_with_logits(
                        margin, labels.float(), reduction="sum")
                    self._backward_microbatch(loss)
                self._optimizer_step()
                updates += 1
                if updates % EVAL_CADENCE == 0 or updates >= budget:
                    metric = self._validate_ladder(val)
                    self._select_checkpoint("stage3", updates, metric)
                    self._log({"stage": "stage3", "update": updates,
                               "validation_balanced_accuracy": metric})
                if updates >= budget:
                    break
        return self.best["metric"]

    # -- occupancy receipts (F7) ----------------------------------------------
    def _occupancy_growth_receipt(self, grid_pools) -> Dict:
        """Stage-2 slot-occupancy growth receipt: mean peak live slots per
        registered item count K (grid 2/4/8/16), keyed by K. Gate: STRICTLY
        INCREASING with item count — the append gate actually firing."""
        import torch
        receipt: Dict = {"gate": "stage2 occupancy growth receipt (registered)",
                         "grid": {}}
        means = {}
        self.model.eval()
        with torch.no_grad():
            for k, pool in grid_pools.items():
                inputs = pool[0]
                lengths = self._pool_lengths(pool)
                peaks = []
                for mb in self._plan_microbatches(lengths):
                    t_pad = int(lengths[mb].max())
                    x = torch.from_numpy(
                        np.ascontiguousarray(inputs[mb][:, :t_pad])).to(self.device)
                    _, _, _, trace = self.model.forward(x, collect_trace=True)
                    self._check_trace_censor(trace)
                    peaks.append(peak_live_from_trace(trace)
                                 .float().mean(dim=-1).cpu().numpy())
                means[k] = float(np.concatenate(peaks).mean())
                receipt["grid"][f"K{k}"] = {"n_examples": int(inputs.shape[0]),
                                            "mean_peak_live_slots": means[k]}
        ks = sorted(means)
        growth = all(means[ks[i]] < means[ks[i + 1]] for i in range(len(ks) - 1))
        receipt["strictly_increasing"] = bool(growth)
        receipt["pass"] = bool(growth)
        return receipt

    # -- the staged curriculum -----------------------------------------------
    def run(self):
        from generators.stage1_mqar import (
            STAGE1_SPLITS, VOCAB_SIZE as S1_VOCAB, generate_pool as gen1,
        )
        from generators.stage2_append_necessity import (
            N_CONTENT, OCCUPANCY_GRID, STAGE2_SPLITS,
            VOCAB_SIZE as S2_VOCAB, generate_pool as gen2,
        )
        if self.is_tape:
            # stage 1 — pure associative recall (MQAR family)
            pools1 = {k: gen1(s.length, s.n_examples, s.seed)
                      for k, s in STAGE1_SPLITS.items()}
            metric, gate = self._run_multiclass_stage(
                "stage1", pools1, vocab_size=S1_VOCAB,
                n_classes=S1_VOCAB - 1, classes=pools1["validation"][3],
                label_offset=STAGE1_LABEL_OFFSET)
            self._marker("stage1_gate", {
                "gate": ">=99% validation mean per-class recall",
                "metric": metric, "pass": bool(gate)})
            if not gate:
                self._halt("stage1", metric)
            # stage 2 — append-necessity (fresh-key selective copy)
            pools2 = {k: gen2(s.length, s.n_to_copy, s.n_examples, s.seed)
                      for k, s in STAGE2_SPLITS.items()}
            metric, gate = self._run_multiclass_stage(
                "stage2", pools2, vocab_size=S2_VOCAB, n_classes=N_CONTENT,
                classes=pools2["validation"][4],
                label_offset=STAGE2_LABEL_OFFSET)
            occ = None
            if gate:      # the occupancy receipt fires once the accuracy gate holds
                grid = {g.n_to_copy: gen2(g.length, g.n_to_copy, g.n_examples,
                                          g.seed) for g in OCCUPANCY_GRID}
                occ = self._occupancy_growth_receipt(grid)
                self._write_json(self.receipts_root / "occupancy_growth.json", occ)
                gate = occ["pass"]
            self._marker("stage2_gate", {
                "gate": ">=99% validation mean per-class recall AND receipted "
                        "slot-occupancy growth (strictly increasing mean peak "
                        "live slots across K in {2,4,8,16})",
                "metric": metric, "occupancy_receipt": occ, "pass": bool(gate)})
            if not gate:
                self._halt("stage2", metric)
        # stage 3 — the ladder (tape AND native arms; natives are stage-3-only)
        self.best = {"metric": -math.inf, "update": None, "stage": None}
        final_metric = self._run_ladder_stage()
        self._final_evaluation()
        self._write_json(self.run_root / "result.json", {
            "arm": self.arm, "status": "complete",
            "stage3_validation_balanced_accuracy": final_metric,
            "selected_checkpoint": self.best, "total_updates": self.total_updates})
        return EXIT_OK

    def _halt(self, stage: str, metric):
        """A stage failure halts the arm with the stage recorded (no
        automatic retry; a replacement requires a NEW registration)."""
        self._write_json(self.run_root / "result.json", {
            "arm": self.arm, "status": "halt_stage_gate_failed",
            "stage": stage, "metric": metric})
        raise SystemExit(EXIT_STAGE_GATE_FAILED)

    def _final_evaluation(self):
        """The SELECTED checkpoint is the only one evaluated on the immutable
        bins and sweep pools; per-pool binary reports (balanced accuracy,
        per-class recalls, confusion, class counts, majority baseline) plus
        the occupancy receipts (F7): peak live slots WITHIN each word's
        length, end-of-word live slots at the EOS token on accepts, and
        PER-K occupancy keyed by depth (the registered [0.9k, 1.1k] check
        consumes per-k data, not one pool mean)."""
        import torch
        from generators.stage3_ladder import (
            ACCEPT, BIN_ORDER, binary_report, ladder_bin_pool, ladder_sweep_pool,
        )
        state = torch.load(self.run_root / "best.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        report: Dict = {"arm": self.arm, "selected_update": self.best["update"]}
        pools = {name: ladder_bin_pool(name) for name in BIN_ORDER}
        if self.is_tape and self.arm_cfg.get("fine_sweep"):
            for k in self.arm_cfg["fine_sweep"]["points"]:
                pools[f"sweep_k{k}"] = ladder_sweep_pool(k)
        for name, (inputs, labels, lengths, meta) in pools.items():
            preds, peaks_all, eos_all = [], [], []
            with torch.no_grad():
                for mb in self._plan_microbatches(lengths):
                    t_pad = int(lengths[mb].max())
                    x = torch.from_numpy(
                        np.ascontiguousarray(inputs[mb][:, :t_pad])).to(self.device)
                    ln = torch.from_numpy(lengths[mb]).to(self.device)
                    logits, st, trace = self._forward_binary(
                        x, ln, collect_trace=self.is_tape)
                    self._check_eval_logits(logits)      # F6 censor path
                    self._check_trace_censor(trace)
                    preds.append((logits[:, 1] > logits[:, 0]).long().cpu().numpy())
                    if self.is_tape:
                        # F7: within-word peak + at-EOS occupancy, per example
                        # (mean over heads), aligned with labels/depths.
                        mb_peaks, mb_eos = peak_and_eos_live_from_trace(
                            trace, lengths[mb])
                        peaks_all.append(
                            mb_peaks.float().mean(dim=-1).cpu().numpy())
                        eos_all.append(
                            mb_eos.float().mean(dim=-1).cpu().numpy())
            entry = {"meta": meta,
                     "report": binary_report(labels, np.concatenate(preds))}
            if self.is_tape:
                peaks = np.concatenate(peaks_all)      # [n]
                eos = np.concatenate(eos_all)          # [n]
                acc = labels == ACCEPT
                entry["occupancy_receipt"] = {
                    "measurement": "peak live slots within each word's length "
                                    "(to its EOS; PAD-region behavior excluded) "
                                    "and end-of-word live slots at the EOS "
                                    "token (F7)",
                    "mean_peak_live_slots_within_word": float(peaks.mean()),
                    "mean_end_of_word_live_slots_on_accepts": (
                        float(eos[acc].mean()) if bool(acc.any()) else None),
                    # per-k keyed by depth: the registered [0.9k, 1.1k] check
                    "per_k": per_depth_occupancy(
                        meta["depths"], peaks, eos, labels, accept_label=ACCEPT),
                }
            report[name] = entry
        self._write_json(self.run_root / "final_eval.json", report)
        return report


# ---------------------------------------------------------------------------
# Verify-only mode (F2): verifies the freeze and EXITS — structurally before
# any model/trainer import (no torch import can occur on this path; the
# receipt proves it with ``torch_imported: false``).
# ---------------------------------------------------------------------------
def run_verify_only(frozen_hash: str) -> int:
    manifest = verify_freeze(frozen_hash)
    receipt = {
        "verify_only": True,
        "frozen_freeze_hash": frozen_hash,
        "manifest_hash": manifest["manifest_hash"],
        "package_files_verified": len(
            [r for r in manifest["files"] if not r.startswith("../")]),
        "dependency_files_verified": len(
            [r for r in manifest["files"] if r.startswith("../")]),
        "worktree_and_head_verified": True,
        "committed_verification": "unconditional (F2)",
        "torch_imported": ("torch" in sys.modules),
        "trainer_constructed": False,
        "exit_code": EXIT_OK,
        "note": "verification complete; process exits without constructing "
                "a trainer or importing any model code",
    }
    print(json.dumps(receipt, indent=2))
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",
                        choices=["tape_B0", "tape_B0x4", "tape_B0x16",
                                 "tape_unbounded", "tape_appendonly",
                                 "gdn2_native", "e97_native"],
                        help="the training arm (required unless --verify-only)")
    parser.add_argument("--frozen-freeze-hash",
                        help="the freeze manifest hash; REQUIRED for any "
                             "optimizer update (M2-before-training)")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify the complete freeze (hashes, manifest, "
                             "unconditional committed state, dependency "
                             "identities) and exit WITHOUT constructing a "
                             "trainer or importing any model code")
    parser.add_argument("--results-root", type=Path,
                        default=RESULTS_ROOT_DEFAULT,
                        help="runtime artifacts root (F3): OUTSIDE the "
                             "frozen package; default "
                             "experiments/expressivity_tasks/"
                             "results_x28e_memory_head/")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.frozen_freeze_hash:
        print("REFUSED: --frozen-freeze-hash is required — no optimizer update "
              "is reachable without a verified M2 freeze (zero updates).",
              file=sys.stderr)
        return EXIT_NO_HASH
    if args.verify_only:
        return run_verify_only(args.frozen_freeze_hash)
    if not args.arm:
        parser.error("--arm is required unless --verify-only")
    # M2-review-3 findings 1+2: main() DELEGATES the entry contract to the
    # constructor invariant (the former inline checks carried an undefined
    # PACKAGE_ROOT symbol — EXECUTING_ROOT is the defined root); the
    # constructor enforces the same contract on every path.
    trainer = StagedTrainer(args.arm, args.results_root,
                           args.frozen_freeze_hash, args.device)
    return trainer.run()


if __name__ == "__main__":
    raise SystemExit(main())
