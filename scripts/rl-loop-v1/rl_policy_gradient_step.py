#!/usr/bin/env python3
"""Single-GPU policy-gradient train step (e97-rl-loop-v1 v2 — the second
train step alongside the receipts-SFT step).

This is the first train step in the programme that takes a true reinforcement
gradient: the audited GRPO-style candidate arithmetic
(ndm/e97_outcome_rl_candidate.py — implemented in the E1 era, NEVER taken an
update) is wired end-to-end to real trajectories, real sealed-validator
rewards, and a real optimizer step.

Inputs, all digest-verified fail-closed:
  * a PG trajectory batch (rl_pg_batch.py): the POLICY'S OWN emitted assistant
    turns from on-policy attempt episodes (passes AND failures — the reward
    contrast), with binary sealed-validator outcomes grouped by lake task id;
  * the CURRENT lane checkpoint (train parent, sha-verified);
  * a k3-KL ANCHOR, selected by --anchor-mode:
      - "parent" (the default, the textbook GRPO reference): the anchor IS
        the trajectory-generating policy == the checkpoint being trained, so
        the k3 term has exactly zero gradient on the first update and tethers
        steps 2+ back toward the data-generating weights.  The bank's lanes
        generate each cycle's episodes from the same lineage they then train,
        so parent == behavior policy — the honest reference when behavior
        logprobs cannot be recomputed.
      - "fixed": a pinned anchor checkpoint (the E3-u256 substrate) for a
        multi-generation tether.  MEASURED (v2 qualification, recorded as a
        negative result): the bank lineages sit ~3555 nats/token from the
        substrate in k3 terms, so the k3 term EXPLODES (kl_part 4886 vs
        surrogate_part 0.002) and dominates the update — the fixed-substrate
        anchor is unusable as a gentle tether and is kept only as a
        screenable, explicitly-chosen variant.

Computation (mirrors the qualified logprob machinery,
scripts/qualify_e97_native_rl_logprobs.py, as closely as a grad-enabled pass
allows):
  * rows are (episode, assistant turn) pairs in the assay's turn_layout
    (alignment-128 padded rows; prefix = full episode tokens before the turn;
    generated = the turn's tokens);
  * per-token policy logprobs are captured with the assay's chunked-head
    forward hook (FP32 log_softmax over each loss chunk's masked rows), under
    the same autocast-BF16 + FP32-logits chunked-CE geometry the SFT step and
    the assay use.  ONE measured deviation from the SFT/assay geometry: the
    chunked-CE torch.utils.checkpoint REPLAY is disabled for the capture
    forward — the non-reentrant checkpoint's saved-tensor consistency check
    rejects the hook's escaped per-token logprob tensors on recomputation
    (measured: CheckpointError "10 saved vs 6 recomputed").  Numerics are
    identical (same chunked head, same FP32 log_softmax); only the memory
    discipline changes — each row's CE graph is retained until that row's
    immediate backward instead of being replayed (per-row layouts are small;
    the SFT step's layer-group gradient checkpointing stays ON);
  * per-row internal-CE consistency gate: the model's own chunked-CE sum over
    the row's mask must equal minus the sum of the captured logprobs to within
    the assay's ce_mean_delta_max tolerance (1e-4 on the per-token mean) —
    the in-step descendant of the assay's ce_head_consistency check;
  * advantages = the candidate's group_advantages over the batch's episode
    rewards (complete same-task groups; zero-variance groups contribute zero
    advantage by the audited arithmetic);
  * the loss = the candidate's policy_loss VERBATIM, summed over rows exactly
    as its docstring prescribes ("chunk contributions must be summed"): each
    row's contribution is divided by that row's whole-episode token count and
    the batch's distinct-episode count, and gradients accumulate row by row
    so only one row's graph is ever alive;
  * importance ratio, honestly: the bank's episodes are greedy-decoded and
    carry no generation-time logprob trace, and superseded generating
    checkpoints are reclaimed, so behavior-policy logprobs cannot be
    recomputed.  old_logp is therefore the CURRENT weights' detached logprobs
    at each step start (ratio ≡ 1; single-use trajectories; the clipped
    surrogate is exercised but inactive at ratio 1 — the first-update
    gradient is exactly REINFORCE-with-group-baseline + k3-KL tether).
    Recorded in every receipt as importance_ratio_mode="recomputed-on-policy".
    GRID mode adds an explicit no-grad pre-pass under CURRENT train weights
    before advantages; its detached outputs supply old_logp in the update
    pass. Generating checkpoint sha values are provenance, never denominators.
    The realized-KL guard measures the update displacement on replay tokens,
    NOT behavior-policy drift from a collector seed or superseded lineage.

Fail-closed guards (NO checkpoint is saved when any trips; the calling lane
falls back to the receipts-SFT step for that cycle):
  nonfinite advantages/loss/grad-norm, CE-consistency mismatch, coverage
  mismatch, and realized post-step k3-KL against the pre-step weights above
  --kl-guard.

Receipts: every step appends a digest-linked record to the PG train-event
stream (--pg-stream, canonical JSON, hash chain via prev_receipt_sha256)
carrying the full parameter set, loss, KL and advantage statistics — the
v2 evidence chain, no SQLite anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/erikg/emender")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BATCH_SCHEMA = "emender-rl-loop-pg-batch-v1"
PG_STEP_SCHEMA = "emender-rl-loop-pg-step-v1"
PG_RECEIPT_SCHEMA = "emender-rl-loop-pg-step-receipt-v1"
CE_CONSISTENCY_TOL = 1e-4  # the assay's ce_mean_delta_max


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(path: Path, event: str, **values) -> None:
    from scripts.train_e97_4b_pi_sft import emit as _emit

    _emit(Path(path), event, 0, **values)


def append_pg_receipt(stream_path: Path, receipt: dict) -> str:
    """Append one digest-linked PG train-event receipt (hash chain, fsync)."""
    from ndm.e97_onpolicy_records import canonical_json

    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    digest = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    chained = {**body, "receipt_sha256": digest}
    line = canonical_json(chained) + "\n"
    with Path(stream_path).open("a") as handle:
        handle.write(line)
        handle.flush()
        import os

        os.fsync(handle.fileno())
    return digest


def pg_stream_tail(stream_path: Path) -> str | None:
    path = Path(stream_path)
    if not path.is_file():
        return None
    tail = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                tail = line
    return json.loads(tail)["receipt_sha256"] if tail else None


def verify_pg_stream(stream_path: Path) -> list[dict]:
    """Walk the PG train-event chain and verify every receipt's digest.

    SERIALIZATION FORMS (recorded per receipt in the returned entries):
      - "string-key-lexicographic" (current): every receipt field is a
        JSON-native type, so canonical_json round-trips and the digest
        recomputes directly.
      - "int-key-numeric" (legacy, receipts written 2026-09-27 before the
        string-key fix): the body's advantages.per_episode was emitted with
        INT keys; canonical_json sorted those numerically at write time
        (0,1,2,...,10,11) while the round-tripped STRING keys sort
        lexicographically (0,1,10,11,2,...).  The recorded digest is VALID —
        it binds the write-time bytes — and this verifier reproduces those
        bytes by re-intifying the per_episode keys before hashing.  Receipts
        whose digest matches NEITHER form fail closed.
    """
    from ndm.e97_onpolicy_records import canonical_json

    def _legacy_body(body):
        if not isinstance(body, dict):
            return body
        advantages = body.get("advantages")
        if not isinstance(advantages, dict):
            return body
        per_episode = advantages.get("per_episode")
        if not isinstance(per_episode, dict) or not per_episode:
            return body
        if not all(isinstance(key, str) and key.isdigit() for key in per_episode):
            return body
        legacy = dict(body)
        legacy_advantages = dict(advantages)
        legacy_advantages["per_episode"] = {int(key): value
                                            for key, value in per_episode.items()}
        legacy["advantages"] = legacy_advantages
        return legacy

    receipts = []
    expected_prev = None
    with Path(stream_path).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank pg-stream line at {number}")
            receipt = json.loads(line)
            if receipt.get("schema") != PG_RECEIPT_SCHEMA:
                raise ValueError("unknown pg-stream receipt schema")
            if set(receipt) != {"schema", "receipt_sha256", "prev_receipt_sha256",
                                "body"}:
                raise ValueError("pg-stream receipt field set mismatch")
            base = {"schema": receipt["schema"],
                    "prev_receipt_sha256": receipt["prev_receipt_sha256"],
                    "body": receipt["body"]}
            forms = {
                "string-key-lexicographic": base,
                "int-key-numeric": {**base,
                                    "body": _legacy_body(receipt["body"])},
            }
            form = None
            for name, candidate in forms.items():
                digest = hashlib.sha256(
                    canonical_json(candidate).encode("utf-8")).hexdigest()
                if digest == receipt["receipt_sha256"]:
                    form = name
                    break
            if form is None:
                raise ValueError(f"pg-stream receipt digest mismatch at {number}")
            if expected_prev is not None and \
                    receipt["prev_receipt_sha256"] != expected_prev:
                raise ValueError(f"pg-stream chain link mismatch at {number}")
            receipts.append({"receipt": receipt, "serialization_form": form})
            expected_prev = receipt["receipt_sha256"]
    return receipts


def _k3_stats(new_logp, ref_logp):
    """Per-token k3 KL estimator stats (the candidate's exp(d)-d-1 form)."""
    import torch

    delta = (ref_logp.detach().float() - new_logp.detach().float())
    kl = torch.exp(delta) - delta - 1.0
    return {
        "mean": float(kl.mean()),
        "max": float(kl.max()),
        "p99": float(torch.quantile(kl, 0.99)),
    }


def _capture_row(model, prefix_ids, generated_ids, device, *, alignment,
                 grad: bool):
    """One row's per-generated-token logprobs — the qualified capture path.

    Identical layout and hook numerics to the logprob assay's teacher-forced
    path (turn_layout + chunked lm_head forward hook + FP32 log_softmax), but
    the captured values keep the autograd graph when ``grad`` is requested so
    the policy-gradient loss backpropagates through them.  The internal
    chunked-CE sum is returned for the consistency gate and never
    backpropagated.
    """
    import torch

    from scripts.qualify_e97_native_rl_logprobs import turn_layout

    tokens, valid, reset, mask = turn_layout(
        list(prefix_ids), list(generated_ids), device, alignment)
    labels = tokens[:, 1:]
    values: list = []
    offset = 0

    def head_hook(_module, _inputs, logits):
        nonlocal offset
        width = logits.shape[1]
        selected = mask[:, offset:offset + width]
        if bool(selected.any()):
            target_ids = labels[:, offset:offset + width][selected]
            lp = torch.log_softmax(logits[selected].float(), -1).gather(
                1, target_ids[:, None]).squeeze(1)
            values.append(lp)
        offset += width

    handle = model.lm_head.register_forward_hook(head_hook)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(tokens, return_loss=True, loss_mask=mask,
                        valid_mask=valid, reset_before=reset,
                        loss_reduction="sum")
    finally:
        handle.remove()
    if offset != tokens.shape[1] - 1:
        raise RuntimeError("head hook chunk coverage mismatch")
    if sum(item.numel() for item in values) != len(generated_ids):
        raise RuntimeError("captured logprob count mismatch")
    gathered = torch.cat(values)
    return gathered, float(loss.detach().float())


def _replay_current_logprobs(model, rows, device, args):
    """Grid-only pre-pass: score every turn under the current train weights.

    Generating checkpoint identities and any cached logprobs are provenance,
    never importance-ratio denominators. Keep the qualified capture geometry
    and CE/coverage guards identical to the update pass.
    """
    import torch

    logprobs = []
    with torch.no_grad():
        for row_number, (_episode, _entry, turn) in enumerate(rows):
            gathered, internal_ce = _capture_row(
                model, turn["prefix"], turn["generated"], device,
                alignment=args.alignment, grad=False)
            ce_mean = internal_ce / len(turn["generated"])
            captured_mean = float(gathered.detach().float().mean())
            if abs(ce_mean + captured_mean) > CE_CONSISTENCY_TOL:
                raise RuntimeError(
                    f"row {row_number}: chunked-CE/logprob consistency failed: "
                    f"{ce_mean} vs {-captured_mean}")
            logprobs.append(gathered.detach().clone().cpu())
    return logprobs


def _configure_train_geometry(model, args):
    """The SFT step's qualified geometry: chunked MLP, projection-chunk 2048."""
    from ndm.models.e97 import E97SplitEditLayer

    model.gradient_checkpointing = True
    model.gradient_checkpoint_group_size = args.gradient_checkpoint_group_size
    chunk_modules = 0
    for module in model.modules():
        if hasattr(module, "checkpoint_chunk_size"):
            module.checkpoint_chunk_size = args.mlp_checkpoint_chunk_size
            chunk_modules += 1
    if chunk_modules != 18:
        raise RuntimeError(
            f"expected 18 chunkable SwiGLU modules, found {chunk_modules}")
    scan_modules = 0
    for module in model.modules():
        if isinstance(module, E97SplitEditLayer):
            if args.projection_chunk_size:
                if int(module.projection_chunk_size) <= 0:
                    raise RuntimeError("projection-chunk-size override requires "
                                       "an inherited chunked scan geometry")
                module.projection_chunk_size = args.projection_chunk_size
            scan_modules += 1
    if scan_modules != 18:
        raise RuntimeError(f"expected 18 E97 split-edit mixers, found {scan_modules}")


def main() -> None:
    import torch

    from ndm.e97 import load_e97_checkpoint
    from ndm.e97_outcome_rl_candidate import group_advantages, policy_loss
    from scripts.train_e97_4b_pi_sft import (EXPECTED_PARAMETERS, SCHEMA,
                                             atomic_save, build_optimizer,
                                             configure_precision)

    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--anchor-mode", choices=("parent", "fixed"),
                        default="parent",
                        help="k3-KL reference: parent (the trajectory-"
                             "generating policy — textbook GRPO) or a fixed "
                             "pinned checkpoint (measured: substrate anchor "
                             "explodes the k3 term; screen-only variant)")
    parser.add_argument("--anchor-checkpoint", type=Path, default=None)
    parser.add_argument("--anchor-sha256", default=None)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--batch-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--pg-stream", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-tag", default="pg-step")
    parser.add_argument("--skip-checkpoint-save", action="store_true",
                        help="run the full guarded step but do not write the "
                             "24GB checkpoint (screen runs; recorded in the "
                             "receipt as checkpoint_saved=false)")
    parser.add_argument("--steps", type=int, default=1)
    # ---- policy-gradient parameters (screened by rl_pg_screen.py)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--kl-guard", type=float, default=0.25,
                        help="fail-closed realized per-token k3-KL bound "
                             "per update (vs the pre-step weights)")
    parser.add_argument("--max-prefix-tokens", type=int, default=16384)
    parser.add_argument("--max-turn-tokens", type=int, default=512)
    # ---- optimizer/precision (identical defaults to rl_train_step.py)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-checkpoint-group-size", type=int, default=3)
    parser.add_argument("--mlp-checkpoint-chunk-size", type=int, default=16384)
    parser.add_argument("--projection-chunk-size", type=int, default=2048)
    parser.add_argument("--loss-logits-fp32", action="store_true", default=True)
    parser.add_argument("--checkpoint-loss-chunks", action="store_true", default=True)
    parser.add_argument("--loss-chunk-size", type=int, default=128,
                        help="the logprob assay's qualified chunk geometry")
    parser.add_argument("--capture-no-ce-checkpoint", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-capture-no-ce-checkpoint", dest="capture_no_ce_checkpoint",
                        action="store_false")
    parser.add_argument("--disable-bf16-reduced-precision-reduction",
                        action="store_true", default=True)
    parser.add_argument("--offload-schedulefree-state", action="store_true",
                        default=True)
    parser.add_argument("--schedulefree-offload-bucket-numel", type=int,
                        default=262144)
    parser.add_argument("--schedulefree-offload-pin-memory", type=int,
                        choices=(0, 1), default=1)
    parser.add_argument("--schedulefree-offload-release-gradients", type=int,
                        choices=(0, 1), default=1)
    parser.add_argument("--sr-seed", type=int, default=927413)
    parser.add_argument("--alignment", type=int, default=128,
                        help="the logprob assay's qualified row alignment")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("steps must be positive")
    if not 0 < args.clip < 1 or args.kl_beta < 0 or args.kl_guard <= 0:
        raise SystemExit("loss bounds")
    if sha256_file(args.parent_checkpoint) != args.parent_sha256:
        raise SystemExit("parent checkpoint identity mismatch")
    if args.anchor_mode == "fixed":
        if args.anchor_checkpoint is None or args.anchor_sha256 is None:
            raise SystemExit("fixed anchor mode requires --anchor-checkpoint "
                             "and --anchor-sha256")
        if sha256_file(args.anchor_checkpoint) != args.anchor_sha256:
            raise SystemExit("anchor checkpoint identity mismatch")
    batch = json.loads(args.batch.read_text())
    if batch.get("schema") != BATCH_SCHEMA:
        raise SystemExit("unsupported batch schema")
    from ndm.e97_onpolicy_records import canonical_json

    body = {key: value for key, value in batch.items() if key != "batch_sha256"}
    if hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() \
            != args.batch_sha256 or batch["batch_sha256"] != args.batch_sha256:
        raise SystemExit("batch manifest digest mismatch")
    if not batch["contrast"]:
        raise SystemExit("batch has no contrastive group (no PG signal)")
    # ---- advantages from the sealed-validator outcomes (the audited
    #      candidate's group normalization; zero-variance groups -> zero).
    #      Singleton groups cannot form a GRPO group (the candidate fails
    #      closed on them) and are dropped honestly before the loss.
    group_counts: dict[str, int] = {}
    for entry in batch["entries"]:
        group_counts[entry["group_id"]] = \
            group_counts.get(entry["group_id"], 0) + 1
    entries = [entry for entry in batch["entries"]
               if group_counts[entry["group_id"]] >= 2]
    dropped_singletons = len(batch["entries"]) - len(entries)
    if not entries:
        raise SystemExit("no complete task group in batch")
    episode_rewards = torch.tensor(
        [entry["reward"] for entry in entries], dtype=torch.float32)
    group_names = sorted({entry["group_id"] for entry in entries})
    group_index = {name: index for index, name in enumerate(group_names)}
    episode_groups = torch.tensor(
        [group_index[entry["group_id"]] for entry in entries],
        dtype=torch.long)
    grid_mode = "streams_root" in batch
    if not grid_mode:
        episode_advantages = group_advantages(episode_rewards, episode_groups)
        advantage_by_episode = {
            index: float(episode_advantages[index]) for index in range(len(entries))}
    episode_count = len(entries)
    rows = [(episode, entry, turn)
            for episode, entry in enumerate(entries)
            for turn in entry["turns"]]
    if not rows:
        raise SystemExit("empty batch")
    for entry in batch["entries"]:
        for turn in entry["turns"]:
            if len(turn["prefix"]) > args.max_prefix_tokens:
                raise SystemExit("row prefix exceeds the qualified layout bound")
            if len(turn["generated"]) > args.max_turn_tokens:
                raise SystemExit("row turn exceeds the qualified layout bound")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


    parameters = {
        "lr": args.lr, "kl_beta": args.kl_beta, "clip": args.clip,
        "kl_guard": args.kl_guard, "grad_clip": args.grad_clip,
        "steps": args.steps, "alignment": args.alignment,
        "loss_chunk_size": args.loss_chunk_size,
        "gradient_checkpoint_group_size": args.gradient_checkpoint_group_size,
        "mlp_checkpoint_chunk_size": args.mlp_checkpoint_chunk_size,
        "projection_chunk_size": args.projection_chunk_size,
        "weight_decay": args.weight_decay, "warmup_steps": args.warmup_steps,
        "importance_ratio_mode": "recomputed-on-policy",
        "anchor_mode": args.anchor_mode,
        "anchor_semantics": (
            "parent == the trajectory-generating policy (textbook GRPO "
            "reference); k3 term has zero gradient on the first update"),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)

    def emit_start():
        emit(args.log_jsonl, "start",
             trainer="rl_policy_gradient_step (single-GPU v2 policy gradient)",
             run_tag=args.run_tag, parent_checkpoint=str(args.parent_checkpoint),
             parent_sha256=args.parent_sha256,
             anchor_mode=args.anchor_mode,
             anchor_checkpoint=(str(args.anchor_checkpoint)
                               if args.anchor_checkpoint is not None else None),
             anchor_sha256=(args.anchor_sha256 if args.anchor_mode == "fixed"
                            else args.parent_sha256),
             batch=str(args.batch), batch_sha256=args.batch_sha256,
             source_commit=args.source_commit,
             episodes=episode_count, rows=len(rows),
             groups={"names": group_names,
                     **{name: batch["groups"][name] for name in group_names}},
             contrastive_groups=batch["contrastive_groups"],
             dropped_singleton_group_episodes=dropped_singletons,
             parameters=parameters,
             advantages={"per_episode": advantage_by_episode,
                         "group_index": group_index})
    if not grid_mode:
        emit_start()

    # ---- phase 1: reference logprobs.
    #      parent mode: the reference IS the parent == the weights being
    #      trained == the trajectory-generating policy, so the reference
    #      logprobs are the step-1 behavior logprobs (old_logp) captured in
    #      the training pass itself — no second model is ever loaded.
    #      fixed mode: a pinned anchor model's no-grad pass, then FREED.
    reference_logprobs: list = []
    if args.anchor_mode == "fixed":
        anchor_loaded = load_e97_checkpoint(
            args.anchor_checkpoint, args_json=args.args_json, device=device,
            dtype=torch.bfloat16, weight_mode="saved", use_triton=True, mmap=True)
        anchor_model = anchor_loaded.model
        anchor_model.eval()
        anchor_model.loss_chunk_size = args.loss_chunk_size
        anchor_model.loss_logits_fp32 = True
        anchor_model.checkpoint_loss_chunks = False
        with torch.no_grad():
            for _episode, entry, turn in rows:
                gathered, _ = _capture_row(anchor_model, turn["prefix"],
                                          turn["generated"], device,
                                          alignment=args.alignment, grad=False)
                reference_logprobs.append(gathered.cpu())
        del anchor_loaded, anchor_model
        torch.cuda.empty_cache()
        emit(args.log_jsonl, "anchor_pass", rows=len(rows),
             max_hbm_allocated=torch.cuda.max_memory_allocated(),
             max_hbm_reserved=torch.cuda.max_memory_reserved())
    else:
        emit(args.log_jsonl, "anchor_pass", rows=len(rows),
             anchor_mode="parent",
             note=("reference == current parent; grid records are replayed "
                   "before advantages" if grid_mode else
                   "reference == parent == trajectory-generating policy; "
                   "reference logprobs are captured as the step-1 old_logp"))

    # ---- phase 2: the training forward/backward under the CURRENT weights
    loaded = load_e97_checkpoint(
        args.parent_checkpoint, args_json=args.args_json, device=device,
        dtype=torch.bfloat16, weight_mode="train", use_triton=True, mmap=True)
    core_model = loaded.model.train()
    precision_policy = configure_precision(core_model, args)
    # ---- HONEST DEVIATION (recorded in the precision policy and every
    #      receipt): the chunked-CE torch.utils.checkpoint replay cannot
    #      coexist with the head-hook capture in grad mode — the non-reentrant
    #      checkpoint's saved-tensor consistency check rejects the hook's
    #      escaped per-token logprob tensors on recomputation (measured:
    #      CheckpointError "10 saved vs 6 recomputed").  The PG capture
    #      therefore runs with the CE-chunk replay DISABLED: identical chunked
    #      head/FP32-log_softmax numerics, but each row's CE graph is retained
    #      until that row's immediate backward (bounded — one ~6K-token row at
    #      a time, freed before the next row) instead of replayed.  The
    #      layer-group gradient checkpointing (the SFT step's memory
    #      discipline) stays ON and is unaffected.
    if args.capture_no_ce_checkpoint:
        core_model.checkpoint_loss_chunks = False
        precision_policy["checkpoint_loss_chunks"] = False
        precision_policy["pg_capture_deviation"] = (
            "chunk-CE checkpoint replay disabled for the grad-mode head-hook "
            "capture (non-reentrant checkpoint saved-tensor consistency "
            "rejects escaped hook tensors); identical chunked FP32-logit "
            "numerics, per-row CE graphs freed at each row's backward")
    _configure_train_geometry(core_model, args)
    parameter_count = sum(p.numel() for p in core_model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"E97 4B parameter mismatch: {parameter_count}")
    optimizer = build_optimizer(core_model.parameters(), args,
                               named_parameters=core_model.named_parameters())
    initialization = optimizer.initialize_state_()
    optimizer.assert_state_offloaded()
    optimizer.train()

    # Grid collection policies may be the seed or a superseded lineage.
    # Explicitly replay ALL rows before computing advantages or any gradient.
    # This is the existing recomputed-on-policy arithmetic, not behavior-
    # policy importance sampling; do not use collector/generation logprobs.
    replay_logprobs = None
    if grid_mode:
        replay_logprobs = _replay_current_logprobs(core_model, rows, device, args)
        episode_advantages = group_advantages(episode_rewards, episode_groups)
        advantage_by_episode = {
            index: float(episode_advantages[index]) for index in range(len(entries))}
        parameters["grid_logprob_replay"] = "current-train-weights-before-advantages"
        parameters["anchor_semantics"] = (
            "parent == current training checkpoint, not collection policy; "
            "k3 tether starts at zero and realized KL measures update displacement")
        emit_start()

    step_records = []
    final_update = 0
    for update in range(1, args.steps + 1):
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        if grid_mode and update > 1:
            replay_logprobs = _replay_current_logprobs(core_model, rows, device, args)
        old_logprobs: list = []
        row_losses = []
        row_kl_anchor = []
        surrogate_parts = []
        kl_parts = []
        for row_number, (_episode, entry, turn) in enumerate(rows):
            gathered, internal_ce = _capture_row(
                core_model, turn["prefix"], turn["generated"], device,
                alignment=args.alignment, grad=True)
            generated = len(turn["generated"])
            # internal-CE consistency gate (the assay's ce_head_consistency
            # descendant): the model's own chunked-CE sum must equal minus the
            # captured logprob sum to the assay's per-token tolerance.
            ce_mean = internal_ce / generated
            captured_mean = float(gathered.detach().float().mean())
            if abs(ce_mean + captured_mean) > CE_CONSISTENCY_TOL:
                raise RuntimeError(
                    f"row {row_number}: chunked-CE/logprob consistency failed: "
                    f"{ce_mean} vs {-captured_mean}")
            old = (replay_logprobs[row_number].to(device) if grid_mode
                   else gathered.detach().clone())
            old_logprobs.append(old.cpu())
            if args.anchor_mode == "parent" and update == 1:
                # parent mode, first update: the reference IS the parent ==
                # the weights being trained, so old == reference exactly.
                reference = old
            else:
                reference = reference_logprobs[row_number].to(device)
            advantage = advantage_by_episode[_episode]
            episode_tokens = float(entry["episode_tokens"])
            mask = torch.ones((1, generated), dtype=torch.bool, device=device)
            row_loss = policy_loss(
                gathered[None, :], old[None, :], reference[None, :],
                torch.tensor([advantage], device=device), mask,
                torch.tensor([episode_tokens], device=device), episode_count,
                clip=args.clip, kl_beta=args.kl_beta)
            row_losses.append(float(row_loss.detach().float()))
            kl = _k3_stats(gathered, reference)
            row_kl_anchor.append(kl)
            with torch.no_grad():
                ratio = torch.exp(
                    (gathered.detach() - old).float())
                denom = episode_tokens * episode_count
                objective = torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1 - args.clip, 1 + args.clip) * advantage)
                surrogate_parts.append(float((-objective / denom).sum()))
                delta = (reference - gathered.detach()).float()
                kl_parts.append(float(
                    (args.kl_beta * (torch.exp(delta) - delta - 1) / denom).sum()))
            row_loss.backward()
            del row_loss, gathered, old, reference, mask
        grad_norm = torch.nn.utils.clip_grad_norm_(
            core_model.parameters(),
            args.grad_clip if args.grad_clip > 0 else float("inf"))
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"nonfinite PG gradient norm at update {update}")
        total_loss = sum(row_losses)
        if not all(_finite(value) for value in row_losses) or \
                not _finite(total_loss):
            raise RuntimeError(f"nonfinite PG loss at update {update}")
        if args.anchor_mode == "parent" and update == 1:
            # freeze the step-1 behavior logprobs as the parent reference for
            # any subsequent steps (the k3 tether engages from update 2)
            reference_logprobs = list(old_logprobs)
        optimizer.step()
        final_update = update

        # ---- post-step no-grad pass: realized KL vs the pre-step weights
        #      (the fail-closed blowout guard) and vs the fixed anchor.
        realized_kl_old = []
        realized_kl_anchor = []
        with torch.no_grad():
            for row_number, (_episode, _entry, turn) in enumerate(rows):
                gathered, _ = _capture_row(
                    core_model, turn["prefix"], turn["generated"], device,
                    alignment=args.alignment, grad=False)
                realized_kl_old.append(
                    _k3_stats(gathered, old_logprobs[row_number].to(device)))
                realized_kl_anchor.append(
                    _k3_stats(gathered,
                              reference_logprobs[row_number].to(device)))
        mean_kl_old = sum(item["mean"] for item in realized_kl_old) / len(rows)
        max_kl_old = max(item["max"] for item in realized_kl_old)
        mean_kl_anchor = sum(item["mean"] for item in realized_kl_anchor) / len(rows)
        step_records.append({
            "update": update,
            "loss": total_loss,
            "surrogate_part": sum(surrogate_parts),
            "kl_part": sum(kl_parts),
            "grad_norm": float(grad_norm),
            "pre_step_kl_vs_anchor": row_kl_anchor,
            "realized_kl_vs_old": {"mean": mean_kl_old, "max": max_kl_old},
            "realized_kl_vs_anchor_mean": mean_kl_anchor,
            "row_losses": row_losses,
            "step_seconds": time.monotonic() - begin,
            "max_hbm_allocated": torch.cuda.max_memory_allocated(),
            "max_hbm_reserved": torch.cuda.max_memory_reserved(),
        })
        emit(args.log_jsonl, "step", update=update, loss=total_loss,
             surrogate_part=sum(surrogate_parts), kl_part=sum(kl_parts),
             grad_norm=float(grad_norm),
             realized_kl_vs_old_mean=mean_kl_old,
             realized_kl_vs_old_max=max_kl_old,
             realized_kl_vs_anchor_mean=mean_kl_anchor,
             rows=len(rows), episodes=episode_count,
             step_seconds=time.monotonic() - begin)
        if not _finite(mean_kl_old) or mean_kl_old > args.kl_guard:
            emit(args.log_jsonl, "guard", update=update,
                 guard="realized-kl-blowout", realized_kl_vs_old_mean=mean_kl_old,
                 kl_guard=args.kl_guard)
            raise SystemExit(
                f"KL blowout guard tripped at update {update}: "
                f"realized k3-KL {mean_kl_old} > {args.kl_guard} (no checkpoint "
                f"saved; the lane falls back to the SFT step)")

    optimizer.eval()
    loss_value = sum(record["loss"] for record in step_records) / len(step_records)
    checkpoint = None
    digest = None
    if not args.skip_checkpoint_save:
        checkpoint = args.output_root / (
            f"checkpoint_agent_pg_u{final_update:06d}_loss_{loss_value:.4f}.pt")
        payload = {
        "schema": SCHEMA,
        "model_state_dict": core_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "sft_updates": 0,
        "pg_updates": final_update,
        "sft_total_tokens": 0,
        "assistant_target_tokens": sum(len(turn["generated"])
                                        for _, _, turn in rows),
        "loss": loss_value,
        "weight_mode": "saved-eval-x",
        "parent_checkpoint": str(args.parent_checkpoint),
        "parent_checkpoint_sha256": args.parent_sha256,
        "train_step_channel": "policy-gradient",
        "trainer": "rl_policy_gradient_step single-GPU v2",
        "pg_parameters": parameters,
        "pg_batch_sha256": args.batch_sha256,
        "pg_anchor_checkpoint": (str(args.anchor_checkpoint)
                                 if args.anchor_checkpoint is not None
                                 else str(args.parent_checkpoint)),
        "pg_anchor_mode": args.anchor_mode,
        "pg_anchor_sha256": (args.anchor_sha256 if args.anchor_mode == "fixed"
                             else args.parent_sha256),
        "pg_step_records": step_records,
        "pg_advantages": {"per_episode": {str(index): value for index, value
                                          in advantage_by_episode.items()},
                          "group_index": group_index,
                          "rewards": [float(r) for r in episode_rewards]},
        "data_world_size": 1,
        "context_size": max(len(turn["prefix"]) + len(turn["generated"])
                            for _, _, turn in rows),
        "island_size": 1,
        "diloco_k": 1,
        "grad_clip": args.grad_clip,
        "optimizer_state_storage": "cpu-offloaded",
        "boundary_aware_packs": False,
        "sampler_mode": "pg-trajectory-batch",
        "new_stage_weight_mode": "train",
        "sft_precision": precision_policy,
        "diloco_merge_enabled": False,
        "source_commit": args.source_commit,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "gradient_checkpoint_group_size": args.gradient_checkpoint_group_size,
        "mlp_checkpoint_chunk_size": args.mlp_checkpoint_chunk_size,
        "projection_chunk_size": args.projection_chunk_size,
        "cuda_graph_chunk_scan": False,
        "merge_bucket_numel": 0,
    }
        atomic_save(checkpoint, payload)
        del payload
        digest = sha256_file(checkpoint)
    emit(args.log_jsonl, "checkpoint", update=final_update,
         checkpoint=str(checkpoint) if checkpoint is not None else None,
         checkpoint_bytes=(checkpoint.stat().st_size
                           if checkpoint is not None else None),
         checkpoint_sha256=digest, loss=loss_value,
         checkpoint_saved=checkpoint is not None)
    emit(args.log_jsonl, "complete", updates=final_update,
         requested_steps=args.steps, rows=len(rows), episodes=episode_count,
         final_loss=loss_value,
         realized_kl_vs_old_mean=step_records[-1]["realized_kl_vs_old"]["mean"])

    # ---- digest-linked v2 train-event receipt (hash chain, fsynced)
    receipt_body = {
        "run_tag": args.run_tag,
        "parent_checkpoint_sha256": args.parent_sha256,
        "anchor_mode": args.anchor_mode,
        "anchor_checkpoint_sha256": (args.anchor_sha256
                                      if args.anchor_mode == "fixed"
                                      else args.parent_sha256),
        "batch_sha256": args.batch_sha256,
        "checkpoint_sha256": digest,
        "checkpoint_path": (str(checkpoint)
                            if checkpoint is not None else None),
        "checkpoint_saved": checkpoint is not None,
        "source_commit": args.source_commit,
        "parameters": parameters,
        "advantages": {"per_episode": {str(index): value for index, value
                                       in advantage_by_episode.items()},
                       "group_index": group_index,
                       "rewards": [float(r) for r in episode_rewards],
                       "groups": batch["groups"]},
        "episodes": episode_count,
        "rows": len(rows),
        "steps": [
            {"update": record["update"], "loss": record["loss"],
             "surrogate_part": record["surrogate_part"],
             "kl_part": record["kl_part"],
             "grad_norm": record["grad_norm"],
             "realized_kl_vs_old": record["realized_kl_vs_old"],
             "realized_kl_vs_anchor_mean": record["realized_kl_vs_anchor_mean"],
             "pre_step_kl_vs_anchor_mean": sum(
                 item["mean"] for item in record["pre_step_kl_vs_anchor"])
             / len(record["pre_step_kl_vs_anchor"]),
             "step_seconds": record["step_seconds"]}
            for record in step_records],
        "created_unix": time.time(),
    }
    receipt_digest = append_pg_receipt(args.pg_stream, {
        "schema": PG_RECEIPT_SCHEMA,
        "prev_receipt_sha256": pg_stream_tail(args.pg_stream),
        "body": receipt_body})
    print(json.dumps({
        "schema": PG_STEP_SCHEMA, "run_tag": args.run_tag,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_sha256": digest,
        "updates": final_update, "loss": loss_value,
        "pg_receipt_sha256": receipt_digest,
        "realized_kl_vs_old_mean":
            step_records[-1]["realized_kl_vs_old"]["mean"],
        "grad_norm": step_records[-1]["grad_norm"]}, sort_keys=True))


def _finite(value: float) -> bool:
    import math

    return math.isfinite(float(value))


if __name__ == "__main__":
    main()
