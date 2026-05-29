#!/usr/bin/env python3
"""Generation smoke for local converted HF artifact directories."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.smoke_local_checkpoint_generation import install_cpu_fallbacks  # noqa: E402


def module_name(obj: Any) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def unique_layer_classes(model: torch.nn.Module) -> list[str]:
    core = getattr(model, "model", model)
    if hasattr(core, "layers"):
        return sorted({module_name(layer) for layer in core.layers})
    if hasattr(core, "blocks"):
        return sorted({module_name(layer) for layer in core.blocks})
    return []


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["e88", "gdn", "m2rnn"], required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--prompt", default="The theorem states")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--cpu-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--cuda-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    if not (artifact_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"missing model.safetensors in {artifact_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    start = time.time()
    torch_dtype = parse_dtype(args.cpu_dtype if device.type == "cpu" else args.cuda_dtype)
    cfg = AutoConfig.from_pretrained(str(artifact_dir), trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir), local_files_only=True)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        str(artifact_dir),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        output_loading_info=True,
        local_files_only=True,
    )

    missing = loading_info.get("missing_keys") or []
    unexpected = loading_info.get("unexpected_keys") or []
    mismatched = loading_info.get("mismatched_keys") or []

    cpu_fallbacks: list[str] = []
    if device.type == "cpu":
        core = getattr(model, "model", model)
        cpu_fallbacks = install_cpu_fallbacks(core)
        model = model.to(dtype=torch_dtype)
    else:
        model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    finite_steps: list[bool] = []
    output_ids = input_ids
    logits = None
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            logits = model(output_ids).logits
            finite_steps.append(bool(torch.isfinite(logits).all().item()))
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_id], dim=-1)
        logits = model(output_ids).logits
        finite_steps.append(bool(torch.isfinite(logits).all().item()))

    new_ids = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=False)
    all_logits_finite = all(finite_steps)
    key_clean = not missing and not unexpected and not mismatched
    param_count = int(sum(param.numel() for param in model.parameters()))
    dtype_sample = str(next(model.parameters()).dtype)
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else None

    result = {
        "ok": bool(new_ids and generated_text and all_logits_finite and key_clean),
        "model": args.model,
        "artifact_dir": str(artifact_dir),
        "device": args.device,
        "torch_dtype_requested": str(torch_dtype),
        "dtype_sample": dtype_sample,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": cuda_name,
        "config_class": cfg.__class__.__name__,
        "model_class": model.__class__.__name__,
        "model_class_module": model.__class__.__module__,
        "core_model_class": module_name(getattr(model, "model", model)),
        "layer_classes": unique_layer_classes(model),
        "model_identity": getattr(cfg, "model_identity", None),
        "repo_id": getattr(cfg, "repo_id", None),
        "release_revision_name": getattr(cfg, "release_revision_name", None),
        "release_candidate": getattr(cfg, "release_candidate", None),
        "source_checkpoint_path": getattr(cfg, "source_checkpoint_path", None),
        "source_checkpoint_sha256": getattr(cfg, "source_checkpoint_sha256", None),
        "level": getattr(cfg, "level", None),
        "checkpoint_step": getattr(cfg, "checkpoint_step", None),
        "checkpoint_loss": getattr(cfg, "checkpoint_loss", None),
        "param_count": param_count,
        "tokenizer_class": tokenizer.__class__.__name__,
        "prompt": args.prompt,
        "prompt_token_ids": input_ids[0].detach().cpu().tolist(),
        "generated_new_token_ids": new_ids,
        "generated_new_text": generated_text,
        "generated_new_text_repr": repr(generated_text),
        "all_logits_finite": all_logits_finite,
        "generated_nonempty": bool(generated_text),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "cpu_fallbacks": cpu_fallbacks,
        "elapsed_seconds": round(time.time() - start, 3),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "ok",
                    "model",
                    "artifact_dir",
                    "device",
                    "generated_new_token_ids",
                    "generated_new_text_repr",
                    "all_logits_finite",
                    "missing_keys",
                    "unexpected_keys",
                    "mismatched_keys",
                )
            },
            sort_keys=True,
        )
    )

    del model, logits, output_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
