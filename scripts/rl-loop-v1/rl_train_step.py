#!/usr/bin/env python3
"""Single-GPU masked-SFT step over the RL loop pack (e97-rl-loop-v1 stage f).

HONEST DEVIATION (documented in the workspace README gap list): the canonical
trainer scripts/train_e97_4b_pi_sft.py hard-requires qualified worlds of 8 or
64 ranks in eight-rank islands and therefore cannot run on ONE GPU.  This
prototype step reuses the canonical machinery verbatim -- the strict loader
(ndm.e97.load_e97_checkpoint), the precision policy
(scripts.train_e97_4b_pi_sft.configure_precision with the qualified chunk-2048
geometry), the optimizer builder (build_optimizer -> CPU-offloaded
AdamWScheduleFree), the pack dataset (MaskedSFTPackedDataset with
boundary-aware epoch-permutation packs), and the canonical checkpoint schema
(emender-e97-4b-pi-masked-sft-v1) -- but runs a world of one rank with no DDP
and no DiLoCo merge.  The target-token-normalized objective is preserved:
loss = sum-CE over assistant targets / total targets in the sampled pack.

One GPU only (CUDA_VISIBLE_DEVICES set by the lease broker upstream).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

REPO_ROOT = Path("/home/erikg/emender")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity, sha256
from ndm.e97 import load_e97_checkpoint
from ndm.schedulefree_offload import CPUOffloadAdamWScheduleFree
from scripts.train_e97_4b_pi_sft import (EXPECTED_PARAMETERS, SCHEMA,
                                         atomic_save, build_optimizer,
                                         configure_precision, emit)

TRAINER_SOURCE_SHA = sha256(REPO_ROOT / "scripts" / "train_e97_4b_pi_sft.py")


def sampled_pack_records(data, absolute_index: int) -> dict:
    """Bind the sampler's actual pack to its complete authority records."""
    pack_id = data.pack_id_at(absolute_index)
    pack = data.packs[pack_id]
    start = int(pack["record_offset"])
    stop = start + int(pack["record_count"])
    return {"pack_id": f"pack-{pack_id:08d}",
            "record_ids": [int(i) for i in data.pack_record_ids[start:stop]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sampler-key", type=int, default=970001)
    parser.add_argument("--gradient-checkpoint-group-size", type=int, default=3)
    parser.add_argument("--mlp-checkpoint-chunk-size", type=int, default=16384)
    parser.add_argument("--projection-chunk-size", type=int, default=2048)
    parser.add_argument("--loss-logits-fp32", action="store_true", default=True)
    parser.add_argument("--checkpoint-loss-chunks", action="store_true", default=True)
    parser.add_argument("--loss-chunk-size", type=int, default=2048)
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
    args = parser.parse_args()

    if args.steps <= 0 or args.context_size <= 0:
        raise SystemExit("steps and context-size must be positive")
    if args.context_size % args.projection_chunk_size:
        raise SystemExit("context-size must divide evenly by projection-chunk-size")
    if sha256(args.parent_checkpoint) != args.parent_sha256:
        raise SystemExit("parent checkpoint identity mismatch")
    if sha256(args.authority_root / "manifest.json") != args.authority_sha256:
        raise SystemExit("masked-SFT authority manifest mismatch")
    if sha256(args.pack_root / "manifest.json") != args.pack_sha256:
        raise SystemExit("masked-SFT pack manifest mismatch")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    # New-stage semantics exactly like the canonical trainer: reconstruct the
    # train/y weights from the parent's saved x weights for a fresh optimizer.
    loaded = load_e97_checkpoint(
        args.parent_checkpoint, args_json=args.args_json, device=device,
        dtype=torch.bfloat16, weight_mode="train", use_triton=True, mmap=True)
    core_model = loaded.model.train()
    precision_policy = configure_precision(core_model, args)
    core_model.gradient_checkpointing = True
    core_model.gradient_checkpoint_group_size = args.gradient_checkpoint_group_size
    chunk_modules = 0
    for module in core_model.modules():
        if hasattr(module, "checkpoint_chunk_size"):
            module.checkpoint_chunk_size = args.mlp_checkpoint_chunk_size
            chunk_modules += 1
    if args.mlp_checkpoint_chunk_size > 0 and chunk_modules != 18:
        raise RuntimeError(f"expected 18 chunkable SwiGLU modules, found {chunk_modules}")
    from ndm.models.e97 import E97SplitEditLayer

    scan_modules = 0
    for module in core_model.modules():
        if isinstance(module, E97SplitEditLayer):
            if args.projection_chunk_size:
                if int(module.projection_chunk_size) <= 0:
                    raise RuntimeError("projection-chunk-size override requires an "
                                       "inherited chunked projection/scan geometry")
                module.projection_chunk_size = args.projection_chunk_size
            scan_modules += 1
    if scan_modules != 18:
        raise RuntimeError(f"expected 18 E97 split-edit mixers, found {scan_modules}")
    parameter_count = sum(parameter.numel() for parameter in core_model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"E97 4B parameter mismatch: {parameter_count}")

    optimizer = build_optimizer(core_model.parameters(), args,
                               named_parameters=core_model.named_parameters())
    initialization = optimizer.initialize_state_()
    optimizer.assert_state_offloaded()
    optimizer.train()

    identity = SFTSamplerIdentity(
        authority_manifest_sha256=args.authority_sha256,
        pack_manifest_sha256=args.pack_sha256,
        sampler_key=args.sampler_key,
        data_world_size=1,
        context_size=args.context_size,
    )
    data = MaskedSFTPackedDataset(
        args.authority_root, args.pack_root, identity=identity, rank=0,
        verify_payload_hashes=True, sampler_mode="epoch-permutation")
    if not data.boundary_aware:
        raise RuntimeError("RL loop packs must be boundary-aware v2 packs")

    args.output_root.mkdir(parents=True, exist_ok=True)
    emit(args.log_jsonl, "start", 0,
         trainer="rl_train_step (single-GPU prototype; canonical trainer "
                 "requires 8-rank islands)",
         canonical_trainer_sha256=TRAINER_SOURCE_SHA,
         parent_checkpoint=str(args.parent_checkpoint),
         parent_sha256=args.parent_sha256,
         source_commit=args.source_commit, world_size=1,
         context_size=args.context_size,
         boundary_aware_packs=True, sampler_mode="epoch-permutation",
         lr=args.lr, sft_precision=precision_policy,
         gradient_checkpoint_group_size=args.gradient_checkpoint_group_size,
         mlp_checkpoint_chunk_size=args.mlp_checkpoint_chunk_size,
         projection_chunk_size=args.projection_chunk_size,
         total_parameters=parameter_count,
         optimizer_state_initialized=initialization)

    total_tokens = 0
    total_targets = 0
    recent_losses: list[float] = []
    final_update = 0
    sampled_packs = []
    for update in range(1, args.steps + 1):
        final_update = update
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        sampled_pack = sampled_pack_records(data, data.next_absolute_rank_sample_index)
        tokens, masks, valid_masks, reset_masks, lengths, target_counts = \
            data.get_boundary_aware_batch(1, device=device)
        island_targets = target_counts.sum().to(torch.int64)
        if int(island_targets) <= 0:
            raise RuntimeError("pack sampled no assistant targets")
        observed = int(masks.sum())
        if observed != int(target_counts[0]):
            raise RuntimeError("boundary-aware batch target accounting mismatch")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            part = core_model(
                tokens, return_loss=True, loss_mask=masks, valid_mask=valid_masks,
                reset_before=reset_masks, loss_reduction="sum")
            # World of one: the target-token-normalized gradient of the pack.
            scaled = part * (1.0 / island_targets.to(torch.float32))
        scaled.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            core_model.parameters(),
            args.grad_clip if args.grad_clip > 0 else float("inf"))
        if not torch.isfinite(grad_norm):
            raise RuntimeError("nonfinite SFT gradient norm")
        optimizer.step()
        sampled_packs.append(sampled_pack)
        step_loss = float(part.detach().float()) / int(island_targets)
        recent_losses.append(step_loss)
        total_tokens += int(lengths.sum())
        total_targets += int(target_counts.sum())
        emit(args.log_jsonl, "step", 0, update=update, loss=step_loss,
             global_tokens=int(lengths.sum()), global_targets=int(target_counts.sum()),
             rank_sample_ids=list(data.last_batch_sample_ids),
             sampled_pack=sampled_pack,
             total_tokens=total_tokens, total_targets=total_targets,
             grad_norm=float(grad_norm),
             step_seconds=time.monotonic() - begin,
             max_hbm_allocated=torch.cuda.max_memory_allocated(),
             max_hbm_reserved=torch.cuda.max_memory_reserved())

    optimizer.eval()
    loss_value = sum(recent_losses) / len(recent_losses)
    checkpoint = args.output_root / f"checkpoint_agent_sft_u{final_update:06d}_loss_{loss_value:.4f}.pt"
    payload = {
        "schema": SCHEMA,
        "model_state_dict": core_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "sft_updates": final_update,
        "sft_total_tokens": total_tokens,
        "assistant_target_tokens": total_targets,
        "loss": loss_value,
        "weight_mode": "saved-eval-x",
        "parent_checkpoint": str(args.parent_checkpoint),
        "parent_checkpoint_sha256": args.parent_sha256,
        "authority_manifest_sha256": args.authority_sha256,
        "pack_manifest_sha256": args.pack_sha256,
        "data_world_size": 1,
        "context_size": args.context_size,
        "island_size": 1,
        "diloco_k": 1,
        "grad_clip": args.grad_clip,
        "optimizer_state_storage": CPUOffloadAdamWScheduleFree.state_storage
        if args.offload_schedulefree_state else "accelerator",
        "boundary_aware_packs": True,
        "sampler_mode": "epoch-permutation",
        "new_stage_weight_mode": "train",
        "sft_precision": precision_policy,
        "diloco_merge_enabled": False,
        "source_commit": args.source_commit,
        "sampler_key": args.sampler_key,
        "sampler_cursor": final_update,
        "sampled_packs": sampled_packs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "gradient_checkpoint_group_size": args.gradient_checkpoint_group_size,
        "mlp_checkpoint_chunk_size": args.mlp_checkpoint_chunk_size,
        "projection_chunk_size": args.projection_chunk_size,
        "cuda_graph_chunk_scan": False,
        "merge_bucket_numel": 0,
        "trainer": "rl_train_step single-GPU prototype",
        "canonical_trainer_sha256": TRAINER_SOURCE_SHA,
    }
    atomic_save(checkpoint, payload)
    del payload
    digest = sha256(checkpoint)
    emit(args.log_jsonl, "checkpoint", 0, update=final_update,
         checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
         checkpoint_sha256=digest, loss=loss_value,
         parent_checkpoint_sha256=args.parent_sha256,
         authority_manifest_sha256=args.authority_sha256,
         pack_manifest_sha256=args.pack_sha256, sampled_packs=sampled_packs)
    emit(args.log_jsonl, "complete", 0, updates=final_update,
         requested_steps=args.steps, total_tokens=total_tokens,
         assistant_target_tokens=total_targets, final_loss=loss_value)
    data.close()
    print(json.dumps({"schema": "emender-rl-loop-train-step-v1",
                      "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                      "updates": final_update, "loss": loss_value}, sort_keys=True))


if __name__ == "__main__":
    main()
