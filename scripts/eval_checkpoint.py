#!/usr/bin/env python3
"""Offline evaluator for train.py checkpoints.

Scores saved checkpoints on a fixed held-out tensor after training, without
running any training. The fixed-tensor scoring path mirrors train.py's
--final_heldout_eval block: load tensor chunks, run forward-only
model(batch, return_loss=True), aggregate CE in nats/token, and convert to BPB
with the tensor's bytes_per_token.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as train_mod  # noqa: E402


CSV_FIELDS = ["step", "tokens", "ce", "bpb", "split", "checkpoint"]
CHECKPOINT_RE = re.compile(r"checkpoint_step_(?P<step>\d+)(?:_loss_[0-9.]+)?\.pt$")


def log(message: str) -> None:
    print(f"[eval_checkpoint] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score train.py checkpoints on fixed held-out tensors."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path, help="One checkpoint .pt path.")
    group.add_argument("--run-dir", type=Path, help="Run directory containing checkpoints.")
    parser.add_argument(
        "--glob",
        default="checkpoint_step_*.pt",
        help="Checkpoint glob relative to --run-dir (default: checkpoint_step_*.pt).",
    )
    parser.add_argument(
        "--scoring-tensor",
        "--heldout_tensor",
        dest="scoring_tensor",
        required=True,
        type=Path,
        help="Primary fixed held-out tensor .pt with chunks and bytes_per_token.",
    )
    parser.add_argument(
        "--ood_tensor",
        type=Path,
        default=None,
        help="Optional separate/OOD fixed tensor .pt. Not fabricated by this script.",
    )
    parser.add_argument("--out", "--out-csv", dest="out_csv", required=True, type=Path)
    parser.add_argument(
        "--y-mode",
        choices=["train", "eval", "saved"],
        default="train",
        help=(
            "Schedule-free checkpoint mode. 'train' loads the optimizer state and "
            "calls optimizer.train() to swap saved x/eval weights to y/train "
            "weights. 'eval'/'saved' evaluates the model_state_dict as stored."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("HELDOUT_EVAL_BS", "8")),
        help="Forward-only evaluation batch size (default: HELDOUT_EVAL_BS or 8).",
    )
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=None,
        help=(
            "Override tokens column. Default is step * batch_size * chunk_size * "
            "grad_accum * world_size from checkpoint args/config."
        ),
    )
    parser.add_argument(
        "--args-json",
        type=Path,
        default=None,
        help="Override args/config JSON path. Default: checkpoint metadata, then sibling args.json.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device after leasing (default: cuda). Use cpu only for metadata/debug validation.",
    )
    parser.add_argument(
        "--no-lease",
        action="store_true",
        help="Do not acquire a GPU lease. Intended only for debugging/help paths.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue scoring later checkpoints if one checkpoint fails.",
    )
    return parser.parse_args()


def maybe_reexec_with_gpu_lease(argv: list[str], no_lease: bool) -> None:
    if no_lease or os.environ.get("EVAL_CHECKPOINT_GPU_LEASED") == "1":
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["EVAL_CHECKPOINT_GPU_LEASED"] = "1"
        return
    lease_script = REPO_ROOT / "scripts" / "gpu_lease.sh"
    if not lease_script.exists():
        raise FileNotFoundError(f"GPU lease broker not found: {lease_script}")
    quoted = " ".join(shlex.quote(part) for part in [sys.executable, *argv])
    cmd = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"eval \"$({shlex.quote(str(lease_script))} acquire 1)\" && "
        f"export EVAL_CHECKPOINT_GPU_LEASED=1 && "
        f"exec {quoted}"
    )
    os.execvp("bash", ["bash", "-lc", cmd])


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def checkpoint_step(path: Path, checkpoint: dict[str, Any]) -> int:
    if "step" in checkpoint:
        return int(checkpoint["step"])
    match = CHECKPOINT_RE.search(path.name)
    if match:
        return int(match.group("step"))
    raise ValueError(f"cannot infer checkpoint step from {path}")


def checkpoint_args(path: Path, checkpoint: dict[str, Any], override: Path | None) -> dict[str, Any]:
    if override is not None:
        return load_json(override)
    for key in ("args", "config", "cfg"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return dict(value)
    sibling = path.parent / "args.json"
    if sibling.exists():
        return load_json(sibling)
    raise FileNotFoundError(
        f"no args/config found in checkpoint and no sibling args.json beside {path}; "
        "pass --args-json"
    )


def default_train_args() -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["train.py", "--data", "__offline_eval_dummy__"]
        return train_mod.parse_args()
    finally:
        sys.argv = old_argv


def namespace_from_config(config: dict[str, Any]) -> argparse.Namespace:
    ns = default_train_args()
    for key, value in config.items():
        if hasattr(ns, key):
            setattr(ns, key, value)
    ns.level = train_mod.parse_level(str(ns.level))
    if getattr(ns, "use_triton", None) is None:
        e97_family = str(ns.level) in ("E97", "97") or bool(getattr(ns, "e88_raw_write", 0))
        ns.use_triton = 1 if (e97_family and bool(getattr(ns, "bf16", False))) else 0
    if getattr(ns, "r_h_mode", "auto") == "auto" and ns.level != "mamba2":
        full_wh_levels = {1, 33, 42, 51, 52, 53, 56, 57, 58, 60}
        matrix_state_levels = {70, 71, 72, 73}
        level_int = int(ns.level) if str(ns.level).isdigit() else 0
        ns.r_h_mode = "spectral_norm" if level_int in full_wh_levels else "none"
        if level_int in matrix_state_levels:
            ns.r_h_mode = "none"
    return ns


def vocab_size_from_args(args: argparse.Namespace) -> int:
    if getattr(args, "tokenizer", None):
        import tiktoken

        return int(tiktoken.get_encoding(args.tokenizer).n_vocab)
    return 256


def parse_layer_kwargs(args: argparse.Namespace) -> dict[str, Any] | None:
    layer_kwargs: dict[str, Any] = {}
    if getattr(args, "head_type_logits", None) is not None:
        layer_kwargs["head_type_logits"] = [
            float(x) for x in str(args.head_type_logits).split(",")
        ]
        layer_kwargs["gdn_allow_neg_eigval"] = bool(getattr(args, "gdn_allow_neg_eigval", 1))
    if getattr(args, "corner_mixture", None) is not None:
        layer_kwargs["corner_mixture"] = [float(x) for x in str(args.corner_mixture).split(",")]
    for key in ("lam_max", "beta_max", "igain_max"):
        value = getattr(args, key, None)
        if value is not None:
            layer_kwargs[key] = value
    if getattr(args, "layer_kwargs", None) is not None:
        layer_kwargs.update(json.loads(args.layer_kwargs))
    return layer_kwargs or None


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    vocab_size = vocab_size_from_args(args)
    level = args.level
    if level == "mamba2":
        from ndm.models.mamba2_baseline import Mamba2LM, create_mamba2_model

        if args.dim is not None and args.depth is not None:
            model = Mamba2LM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                d_state=args.mamba_d_state,
                expand=args.mamba_expand,
                headdim=64,
                loss_chunk_size=args.loss_chunk_size,
            )
        else:
            model = create_mamba2_model(
                target_params=args.params, vocab_size=vocab_size, expand=args.mamba_expand
            )
    elif level == "mamba3":
        from ndm.models.mamba3_baseline import Mamba3LM

        mamba3_chunk_size = min(args.chunk_size, 64)
        if args.mamba3_mimo:
            mamba3_chunk_size = min(mamba3_chunk_size, max(16, 64 // max(1, args.mamba3_mimo_rank)))
        model = Mamba3LM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            d_state=args.mamba_d_state,
            expand=args.mamba_expand,
            headdim=args.mamba3_headdim,
            is_mimo=bool(args.mamba3_mimo),
            mimo_rank=args.mamba3_mimo_rank,
            mamba_chunk_size=mamba3_chunk_size,
            loss_chunk_size=args.loss_chunk_size,
        )
    elif level == "hybrid":
        from ndm.models.hybrid_ladder import HybridLadderLM

        layer_pattern = [part.strip() for part in args.hybrid_pattern.split(",") if part.strip()]
        layer_kwargs = []
        for layer_level in layer_pattern:
            kw: dict[str, Any] = {}
            if args.hybrid_m2rnn_heads is not None and layer_level in ("m2rnn", "m2rnn-paper"):
                kw["n_heads"] = args.hybrid_m2rnn_heads
            if layer_level in ("m2rnn", "m2rnn-paper"):
                kw.update(
                    k_head_dim=args.m2rnn_k_head_dim,
                    v_head_dim=args.m2rnn_v_head_dim,
                    num_q_heads=args.m2rnn_q_heads,
                    num_k_heads=args.m2rnn_k_heads,
                    num_v_heads=args.m2rnn_v_heads,
                    num_f_heads=args.m2rnn_f_heads,
                    num_g_heads=args.m2rnn_g_heads,
                    num_weight_heads=args.m2rnn_weight_heads,
                    use_conv=bool(args.use_conv) or layer_level == "m2rnn-paper",
                    d_conv=args.d_conv,
                    output_norm=bool(args.m2rnn_output_norm) or layer_level == "m2rnn-paper",
                    normalize_qk=bool(args.m2rnn_normalize_qk),
                    use_residual=bool(args.m2rnn_use_residual),
                    state_weight_trainable=not bool(args.m2rnn_freeze_state_weight),
                    gradient_clipping=(
                        args.m2rnn_state_grad_clip
                        if args.m2rnn_state_grad_clip is not None
                        else (1.0 if layer_level == "m2rnn-paper" else None)
                    ),
                )
            layer_kwargs.append(kw)
        model = HybridLadderLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            layer_pattern=layer_pattern,
            layer_kwargs=layer_kwargs,
            n_state=args.n_state,
            n_heads=args.n_heads,
            expansion=args.expansion,
            use_gate=bool(args.use_gate),
            gate_activation=args.gate_activation,
            dropout=args.dropout,
        )
    elif level == "m2rnn":
        from ndm.models.m2rnn_baseline import M2RNNLM, create_m2rnn_model

        if args.dim is not None and args.depth is not None:
            model = M2RNNLM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                n_heads=args.n_heads,
                n_state=args.n_state,
                expansion=args.expansion,
                paper_shape=args.m2rnn_paper_shape,
                k_head_dim=args.m2rnn_k_head_dim,
                v_head_dim=args.m2rnn_v_head_dim,
                num_q_heads=args.m2rnn_q_heads,
                num_k_heads=args.m2rnn_k_heads,
                num_v_heads=args.m2rnn_v_heads,
                num_f_heads=args.m2rnn_f_heads,
                num_g_heads=args.m2rnn_g_heads,
                num_weight_heads=args.m2rnn_weight_heads,
                use_gate=bool(args.use_gate),
                use_residual=bool(args.m2rnn_use_residual),
                state_weight_trainable=not bool(args.m2rnn_freeze_state_weight),
                use_conv=bool(args.use_conv),
                d_conv=args.d_conv,
                output_norm=bool(args.m2rnn_output_norm),
                normalize_qk=bool(args.m2rnn_normalize_qk),
                dropout=args.dropout,
                gradient_clipping=args.m2rnn_state_grad_clip,
                gradient_checkpointing=False,
                loss_chunk_size=args.loss_chunk_size,
            )
        else:
            model = create_m2rnn_model(target_params=args.params, vocab_size=vocab_size)
    elif level == "gru":
        from ndm.models.gru_baseline import GRULM, create_gru_model

        model = (
            GRULM(vocab_size=vocab_size, dim=args.dim, depth=args.depth, expansion_factor=args.expansion)
            if args.dim is not None and args.depth is not None
            else create_gru_model(target_params=args.params, vocab_size=vocab_size)
        )
    elif level == "lstm":
        from ndm.models.lstm_baseline import LSTMLM, create_lstm_model

        model = (
            LSTMLM(vocab_size=vocab_size, dim=args.dim, depth=args.depth, expansion_factor=args.expansion)
            if args.dim is not None and args.depth is not None
            else create_lstm_model(target_params=args.params, vocab_size=vocab_size)
        )
    elif level == "mingru":
        from ndm.models.min_rnn_baseline import MinGRULM, create_mingru_model

        model = (
            MinGRULM(vocab_size=vocab_size, dim=args.dim, depth=args.depth, expansion_factor=args.expansion)
            if args.dim is not None and args.depth is not None
            else create_mingru_model(target_params=args.params, vocab_size=vocab_size)
        )
    elif level == "minlstm":
        from ndm.models.min_rnn_baseline import MinLSTMLM, create_minlstm_model

        model = (
            MinLSTMLM(vocab_size=vocab_size, dim=args.dim, depth=args.depth, expansion_factor=args.expansion)
            if args.dim is not None and args.depth is not None
            else create_minlstm_model(target_params=args.params, vocab_size=vocab_size)
        )
    elif level == "cudagru":
        from ndm.models.cuda_gru import CudaGRULM

        model = CudaGRULM(
            vocab_size=vocab_size, dim=args.dim, depth=args.depth, expansion_factor=args.expansion
        )
    elif level == "cudalstm":
        from ndm.models.cuda_lstm import CudaLSTMLM

        model = CudaLSTMLM(
            vocab_size=vocab_size, dim=args.dim, depth=args.depth, expansion_factor=args.expansion
        )
    elif level in ("E94", "E94r"):
        from ndm.models.e94 import E94Model

        model = E94Model(
            vocab_size=vocab_size,
            dim=args.dim,
            n_heads=args.n_heads,
            head_dim=args.n_state,
            depth=args.depth,
            dropout=args.dropout,
            tie_embedding=True,
            use_gate=bool(args.use_gate),
            use_permutation=bool(getattr(args, "use_permutation", 1)),
            gradient_checkpointing=False,
        )
    elif level == "E94nr":
        from ndm.models.e94 import E94NoResidualModel

        model = E94NoResidualModel(
            vocab_size=vocab_size,
            n_heads=args.n_heads,
            head_dim=args.n_state,
            depth=args.depth,
            dropout=args.dropout,
            share_layer_weights=False,
        )
    elif level == "E94oh":
        from ndm.models.e94 import E94OneHotModel

        model = E94OneHotModel(
            vocab_size=vocab_size,
            K=args.n_heads,
            head_dim=args.n_state,
            depth=args.depth,
            dropout=args.dropout,
            gradient_checkpointing=False,
        )
    elif isinstance(level, str) and level.lower() == "e88_fused":
        from ndm.models.e88_fused import E88FusedLM

        model = E88FusedLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            n_heads=args.n_heads,
            n_state=args.n_state,
            expansion=args.expansion,
            use_gate=bool(args.use_gate),
            checkpoint_interval=args.checkpoint_interval,
        )
    elif args.dim is not None and args.depth is not None:
        model = train_mod.LadderLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            level=level,
            layer_kwargs=parse_layer_kwargs(args),
            expansion=args.expansion,
            n_groups=args.n_groups,
            n_state=args.n_state,
            n_slots=args.n_slots,
            n_heads=args.n_heads,
            top_k=args.top_k,
            k_fast=args.k_fast,
            k_slow=args.k_slow,
            use_gate=bool(args.use_gate),
            gate_activation=args.gate_activation,
            linear_state=bool(args.linear_state),
            use_write_gate=bool(args.use_write_gate),
            e88_decay_mode=args.e88_decay_mode,
            e88_value_residual=bool(args.e88_value_residual),
            e88_raw_write=bool(args.e88_raw_write),
            state_expansion=args.state_expansion,
            r_h_mode=args.r_h_mode,
            use_conv=bool(args.use_conv),
            d_conv=args.d_conv,
            gdn2_mlp_ratio=args.gdn2_mlp_ratio,
            dropout=args.dropout,
            checkpoint_interval=args.checkpoint_interval,
            gradient_checkpointing=False,
            projection_chunk_size=args.projection_chunk_size,
            loss_chunk_size=args.loss_chunk_size,
            use_triton=bool(args.use_triton),
            mlp_ratio=args.mlp_ratio,
            mlp_multiple=args.mlp_multiple,
        )
    else:
        model = train_mod.create_ladder_model(
            target_params=args.params,
            level=level,
            vocab_size=vocab_size,
            expansion=args.expansion,
            n_groups=args.n_groups,
            state_expansion=args.state_expansion,
            r_h_mode=args.r_h_mode,
        )

    model = model.to(device)
    if bool(getattr(args, "bf16", False)):
        model = model.bfloat16()
    return model


def make_schedulefree_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    import schedulefree

    knob_suffixes = ("lam_raw", "beta_raw", "igain_raw", "gamma_raw")
    knob_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.endswith(suffix) for suffix in knob_suffixes):
            knob_params.append(param)
        else:
            base_params.append(param)
    if float(getattr(args, "knob_lr_mult", 1.0)) != 1.0 and knob_params:
        param_groups: Any = [
            {"params": base_params, "lr": args.lr},
            {"params": knob_params, "lr": args.lr * args.knob_lr_mult},
        ]
    else:
        param_groups = model.parameters()
    return schedulefree.AdamWScheduleFree(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        warmup_steps=getattr(args, "warmup_steps", 0),
    )


def load_checkpoint_weights(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    model_args: argparse.Namespace,
    y_mode: str,
) -> bool:
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"state_dict mismatch: missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
    swapped = False
    if (
        y_mode == "train"
        and getattr(model_args, "optimizer", "adamw") == "schedulefree"
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer = make_schedulefree_optimizer(model, model_args)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer.train()
        swapped = True
    return swapped


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint is not None:
        return [args.checkpoint.resolve()]
    assert args.run_dir is not None
    paths = sorted(args.run_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"no checkpoints matched {args.run_dir / args.glob}")
    return [path.resolve() for path in paths]


def load_scoring_tensor(path: Path) -> SimpleNamespace:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "chunks" not in payload:
        raise ValueError(f"{path} must be a .pt dict containing key 'chunks'")
    chunks = payload["chunks"]
    if not torch.is_tensor(chunks) or chunks.ndim != 2:
        raise ValueError(f"{path} chunks must be a rank-2 tensor [N, chunk+1]")
    if "bytes_per_token" not in payload:
        raise ValueError(f"{path} must contain bytes_per_token")
    return SimpleNamespace(
        path=path,
        chunks=chunks.long(),
        bytes_per_token=float(payload["bytes_per_token"]),
        scored_tokens=payload.get("scored_tokens"),
    )


@torch.no_grad()
def score_tensor(
    model: torch.nn.Module,
    scoring: SimpleNamespace,
    device: torch.device,
    batch_size: int,
    use_bf16: bool,
) -> tuple[float, float, int]:
    model.eval()
    chunks = scoring.chunks
    batch_size = max(1, min(int(batch_size), int(chunks.shape[0])))
    total_nll = 0.0
    total_tokens = 0
    autocast_enabled = device.type == "cuda" and use_bf16
    for start in range(0, chunks.shape[0], batch_size):
        batch = chunks[start : start + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = model(batch, return_loss=True)
        if isinstance(loss, tuple):
            loss = loss[0]
        scored = int(batch.shape[0] * (batch.shape[1] - 1))
        total_nll += float(loss.item()) * scored
        total_tokens += scored
    ce = total_nll / max(total_tokens, 1)
    bpb = (ce / math.log(2.0)) / scoring.bytes_per_token
    if not math.isfinite(ce) or not math.isfinite(bpb):
        raise RuntimeError(f"non-finite score ce={ce} bpb={bpb}")
    return ce, bpb, total_tokens


def tokens_at_step(step: int, model_args: argparse.Namespace, override: int | None) -> int:
    if override is not None:
        return int(override)
    batch_size = int(getattr(model_args, "batch_size", 1) or 1)
    chunk_size = int(getattr(model_args, "chunk_size", 1) or 1)
    grad_accum = int(getattr(model_args, "grad_accum", 1) or 1)
    world_size = int(
        getattr(model_args, "_world_size", 0)
        or getattr(model_args, "world_size", 0)
        or os.environ.get("WORLD_SIZE", "1")
    )
    return int(step) * batch_size * chunk_size * grad_accum * max(world_size, 1)


def read_done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        done = set()
        for row in reader:
            checkpoint = row.get("checkpoint")
            split = row.get("split", "primary")
            if checkpoint:
                done.add((str(Path(checkpoint).resolve()), split))
            elif row.get("step"):
                done.add((row["step"], split))
        return done


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    maybe_reexec_with_gpu_lease(sys.argv[1:], args.no_lease)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    paths = checkpoint_paths(args)
    scoring_sets = [("primary", load_scoring_tensor(args.scoring_tensor))]
    if args.ood_tensor is not None:
        scoring_sets.append(("ood", load_scoring_tensor(args.ood_tensor)))

    done = read_done_keys(args.out_csv)
    for ckpt_path in paths:
        missing_splits = [
            (name, scoring)
            for name, scoring in scoring_sets
            if (str(ckpt_path.resolve()), name) not in done
        ]
        if not missing_splits:
            log(f"skip already-scored checkpoint: {ckpt_path}")
            continue
        try:
            log(f"loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" not in checkpoint:
                raise KeyError(f"{ckpt_path} has no model_state_dict")
            cfg = checkpoint_args(ckpt_path, checkpoint, args.args_json)
            model_args = namespace_from_config(cfg)
            step = checkpoint_step(ckpt_path, checkpoint)
            model = build_model(model_args, device)
            swapped = load_checkpoint_weights(model, checkpoint, model_args, args.y_mode)
            log(
                f"strict load OK step={step}; schedulefree_y_swap={swapped}; "
                f"level={model_args.level} params={model.get_num_params():,}"
            )
            for split, scoring in missing_splits:
                ce, bpb, _scored = score_tensor(
                    model=model,
                    scoring=scoring,
                    device=device,
                    batch_size=args.batch_size,
                    use_bf16=bool(getattr(model_args, "bf16", False)),
                )
                row = {
                    "step": step,
                    "tokens": tokens_at_step(step, model_args, args.tokens_per_step),
                    "ce": f"{ce:.8f}",
                    "bpb": f"{bpb:.8f}",
                    "split": split,
                    "checkpoint": str(ckpt_path),
                }
                append_row(args.out_csv, row)
                done.add((str(ckpt_path.resolve()), split))
                log(
                    f"scored split={split} step={step} ce={ce:.6f} "
                    f"bpb={bpb:.6f} -> {args.out_csv}"
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            if not args.keep_going:
                raise
            log(f"ERROR scoring {ckpt_path}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
