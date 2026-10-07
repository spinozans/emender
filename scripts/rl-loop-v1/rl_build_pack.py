#!/usr/bin/env python3
"""Convert verified RL receipts into a masked-SFT authority + packs.

This is the receipts->packs stage, following the
scripts/prepare_e97_pi_native_repair_training.py precedent (teacher-pilot
records in the canonical Pi-native codec layout -> emender-e97-tulu3-masked-sft-v1
authority) and the teacher pilot's own record encoder
(scripts/build_e97_pi_native_curriculum.encode_candidate).

Every receipt is re-verified (digest chain + full re-encode) before it may
become a training record; nothing unverified is ever packed.

CPU only.  The pack build itself is delegated to the canonical
scripts/build_e97_sft_packs.py (boundary-aware v2 packs, epoch-permutation
sampler) so the pack format is byte-for-byte the cohort format the canonical
trainer consumes.

NOTE (honest scope): this prototype authority is workspace-local.  The full
proposal/admission machinery (freeze -> audit -> operator authorization) that
gated the E1/E2/E3 segments is out of v1 scope and is NOT claimed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX

from rl_common import REPO_ROOT, ensure_layout, sha256_file, workspace_paths
from rl_receipts import walk_stream


def build_authority(receipts: list[dict], output: Path, cycle: int) -> dict:
    from scripts.build_e97_pi_native_curriculum import encode_candidate

    enc = tiktoken.get_encoding("p50k_base")
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "tokens": output / "tokens.uint32.bin",
        "mask": output / "assistant_mask.uint8.bin",
        "index": output / "records.idx",
        "metadata": output / "records.jsonl",
    }
    offset = 0
    targets_total = 0
    rows = []
    for receipt in receipts:
        ids, mask, supervised = encode_candidate(
            receipt["episode"]["episode_text"], receipt["episode"]["generations"],
            receipt["episode"]["supervise_from"], enc)
        if len(ids) != receipt["episode"]["tokens"] or sum(mask) != receipt["episode"]["targets"]:
            raise SystemExit(f"receipt {receipt['receipt_sha256'][:16]} does not re-encode")
        rows.append((receipt, ids, mask, supervised))
    max_tokens = max((len(ids) for _, ids, _, _ in rows), default=0)
    with paths["tokens"].open("wb") as token_out, \
         paths["mask"].open("wb") as mask_out, \
         paths["index"].open("wb") as ix, \
         paths["metadata"].open("w") as metadata_out:
        for receipt, ids, mask, supervised in rows:
            token_out.write(struct.pack(f"<{len(ids)}I", *ids))
            mask_out.write(bytes(mask))
            ix.write(RECORD_INDEX.pack(offset, len(ids), int(sum(mask)), 0))
            metadata_out.write(json.dumps({
                "id": f"rl-loop-{receipt['task_id']}-{receipt['receipt_sha256'][:16]}",
                "receipt_sha256": receipt["receipt_sha256"],
                "kind": receipt["kind"],
                "source": f"rl-loop-cycle-{cycle:04d}",
                "task_id": receipt["task_id"],
                "task_sha256": receipt["task_sha256"],
                "policy_checkpoint_sha256": receipt["policy_checkpoint"]["checkpoint_sha256"],
                "supervise_from": receipt["episode"]["supervise_from"],
                "supervised_units": supervised,
                "tokens": len(ids),
                "assistant_target_tokens": int(sum(mask)),
            }, sort_keys=True) + "\n")
            offset += len(ids)
            targets_total += int(sum(mask))
    outputs = {key: {"path": value.name, "bytes": value.stat().st_size,
                     "sha256": sha256_file(value)}
               for key, value in paths.items()}
    manifest = {
        "schema": AUTHORITY_SCHEMA,
        "status": "complete",
        "training_eligible": True,
        "tokenizer": "p50k_base",
        "purpose": ("e97-rl-loop-v1 prototype cycle receipts (verified-only "
                    "on-policy/teacher-corrected episodes); workspace-local, "
                    "no proposal/admission machinery"),
        "counts": {"records": len(rows), "tokens": offset,
                   "assistant_target_tokens": targets_total,
                   "train_records": len(rows), "validation_records": 0},
        "receipt_chain_head": (receipts[-1]["receipt_sha256"] if receipts else None),
        "cycle": cycle,
        "outputs": outputs,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n")
    manifest_sha = sha256_file(output / "manifest.json")
    return {"manifest_sha256": manifest_sha, "max_record_tokens": max_tokens,
            "records": len(rows), "tokens": offset, "targets": targets_total}


def build_packs(authority_root: Path, packs_root: Path, *, authority_sha: str,
                context_size: int, python: str) -> dict:
    packs_root.mkdir(parents=True, exist_ok=False)
    command = [
        python, str(REPO_ROOT / "scripts" / "build_e97_sft_packs.py"),
        "--authority-root", str(authority_root),
        "--output-root", str(packs_root),
        "--context-size", str(context_size),
        "--authority-manifest-sha256", authority_sha,
        "--boundary-aware",
        "--sampler-mode", "epoch-permutation",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"pack build failed:\n{completed.stdout}\n{completed.stderr}")
    pack_sha = sha256_file(packs_root / "manifest.json")
    pack_manifest = json.loads((packs_root / "manifest.json").read_text())
    return {"pack_manifest_sha256": pack_sha,
            "packs": pack_manifest["splits"]["train"]["packs"],
            "stdout_tail": completed.stdout.strip()[-400:]}


def select_window_receipts(receipts: list[dict], consumed: set,
                             window: int) -> list[dict]:
    """Era-10 dose fix (receipted 2026-10-06): the oldest N unconsumed
    receipts from the WHOLE verified stream, FIFO. The cycle-scoped mode
    packed ~1 receipt / ~215 target tokens per step — homeopathic against
    the wash's 348,798-target course; the window drains the verified
    backlog (~7.4k receipts collected while the channel was starved) at
    ~32 receipts packed per window (exposure is recorded by the trainer).

    Era-11 (receipted 2026-10-06, same day): KIND PRIORITY — teacher-
    corrected receipts first, on-policy-success only as filler. Evidence:
    the FIFO backlog is dominated by on-policy successes the training
    lineage already fits at loss 0.086-0.18 (its own trajectories from its
    own seed), so mixed windows tripped the NO_SIGNAL_STEP_LOSS_FLOOR and
    were correctly skipped (24/26 windows burned without adopting). Real
    signal lives in teacher-corrected receipts (adopted steps: loss 0.87,
    1.26) — novel GLM trajectories the model has never produced. FIFO
    order is preserved WITHIN each kind."""
    pool = [r for r in receipts if r["receipt_sha256"] not in consumed]
    teacher = [r for r in pool if r.get("kind") == "teacher-corrected"]
    onpolicy = [r for r in pool if r.get("kind") != "teacher-corrected"]
    return (teacher + onpolicy)[:window]


def load_consumed_ledger(path: Path | None) -> set:
    consumed: set = set()
    if path is not None and Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            try:
                consumed.add(json.loads(line)["receipt_sha256"])
            except Exception:
                continue
    return consumed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--receipts-limit", type=int, default=0,
                        help="0 = every verified receipt in the stream")
    parser.add_argument("--min-targets", type=int, default=512,
                        help="fail closed below this many assistant target tokens")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--window", type=int, default=0,
                        help="era-10: pack the oldest N unconsumed receipts from "
                             "the whole stream (FIFO) instead of this cycle only")
    parser.add_argument("--consumed-ledger", type=Path, default=None,
                        help="JSONL of consumed receipt shas; required with "
                             "--window so nothing trains twice unintentionally")
    args = parser.parse_args()
    paths = workspace_paths(args.workspace)
    ensure_layout(paths)
    enc = tiktoken.get_encoding("p50k_base")
    receipts = walk_stream(paths, enc=enc)
    if args.receipts_limit:
        receipts = receipts[:args.receipts_limit]
    if args.window:
        if not args.consumed_ledger:
            raise SystemExit("--window requires --consumed-ledger")
        consumed = load_consumed_ledger(args.consumed_ledger)
        cycle_receipts = select_window_receipts(receipts, consumed, args.window)
        if not cycle_receipts:
            # fallback: this cycle's fresh receipts, if any are unconsumed
            cycle_receipts = [r for r in receipts
                              if r["cycle"] == args.cycle
                              and r["receipt_sha256"] not in consumed]
    else:
        cycle_receipts = [item for item in receipts if item["cycle"] == args.cycle]
    if not cycle_receipts:
        raise SystemExit(f"no verified receipts for cycle {args.cycle}")
    total_targets = sum(item["episode"]["targets"] for item in cycle_receipts)
    if total_targets < args.min_targets:
        raise SystemExit(f"not enough training targets: {total_targets} < {args.min_targets}")
    print(f"PACK receipts={len(cycle_receipts)} targets={total_targets}", flush=True)

    cycle_dir = paths["packs"] / f"cycle-{args.cycle:04d}"
    cycle_dir.mkdir(parents=True, exist_ok=False)
    authority = build_authority(cycle_receipts, cycle_dir / "authority", args.cycle)
    print(f"PACK authority manifest sha256={authority['manifest_sha256']} "
          f"records={authority['records']} tokens={authority['tokens']} "
          f"targets={authority['targets']}", flush=True)
    # context size: smallest 2048-multiple covering every record (the train
    # step uses the qualified --projection-chunk-size 2048, so the context must
    # divide evenly by 2048); at least one full record must fit per pack.
    context = max(2048, -(-authority["max_record_tokens"] // 2048) * 2048)
    packs = build_packs(cycle_dir / "authority", cycle_dir / "packs",
                        authority_sha=authority["manifest_sha256"],
                        context_size=context, python=args.python)
    print(f"PACK packs={packs['packs']} pack manifest sha256={packs['pack_manifest_sha256']} "
          f"context_size={context}", flush=True)
    # Packing is inventory, not optimizer exposure. Never advance consumption here.
    if args.window:
        with (paths["root"] / "packed-receipts.jsonl").open("a") as ledger:
            for r in cycle_receipts:
                ledger.write(json.dumps({
                    "receipt_sha256": r["receipt_sha256"],
                    "packed_cycle": args.cycle,
                    "window": args.window,
                }) + "\n")
    (cycle_dir / "build-summary.json").write_text(json.dumps({
        "schema": "emender-rl-loop-pack-build-v1",
        "cycle": args.cycle,
        "receipts": len(cycle_receipts),
        "receipt_cycles_span": [min(r["cycle"] for r in cycle_receipts),
                                 max(r["cycle"] for r in cycle_receipts)],
        "window": args.window,
        "authority_manifest_sha256": authority["manifest_sha256"],
        "pack_manifest_sha256": packs["pack_manifest_sha256"],
        "context_size": context,
        "max_record_tokens": authority["max_record_tokens"],
    }, sort_keys=True, indent=1) + "\n")


if __name__ == "__main__":
    main()
