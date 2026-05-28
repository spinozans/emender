#!/usr/bin/env python3
"""Minimal HF generation smoke for pinned v0.1 revisions.

The script loads exactly one Hugging Face model revision, runs a tiny greedy
generation, and writes compact JSON evidence. Public repositories work without
credentials; if HF_TOKEN is present it is used. The script intentionally avoids
writing caches or model artifacts anywhere except the caller-provided Hugging
Face cache path.
"""

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
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.smoke_local_checkpoint_generation import install_cpu_fallbacks  # noqa: E402


PINNED_MODELS: dict[str, dict[str, Any]] = {
    "e88": {
        "repo_id": "poietic-pbc/emender-e88-1.3b",
        "revision": "v0.1",
        "expected_sha": "a2e56cb82eec5e01ae6eb501569359c5ff64af6b",
        "identity": "Emender/E88",
    },
    "gdn": {
        "repo_id": "poietic-pbc/gdn-1.3b",
        "revision": "v0.1",
        "expected_sha": "556df7f00969c6a8dbeb381e3c8b51cf0c0385f9",
        "identity": "GDN",
    },
    "m2rnn": {
        "repo_id": "poietic-pbc/m2rnn-cma-1.3b",
        "revision": "v0.1",
        "expected_sha": "8181b77803e130ffd78e37c33aa4d58c27e719c2",
        "identity": "M2RNN-CMA",
    },
}


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
    parser.add_argument("--model", choices=sorted(PINNED_MODELS), required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--prompt", default="The theorem states")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--cpu-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--cuda-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or False

    spec = PINNED_MODELS[args.model]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    start = time.time()
    api = HfApi()
    info = api.repo_info(
        spec["repo_id"],
        repo_type="model",
        revision=spec["revision"],
        token=token,
    )
    refs = api.list_repo_refs(spec["repo_id"], repo_type="model", token=token)
    if getattr(info, "sha", None) != spec["expected_sha"]:
        raise SystemExit(
            f"{spec['repo_id']} revision {spec['revision']} resolved sha "
            f"{getattr(info, 'sha', None)} != expected {spec['expected_sha']}"
        )

    torch_dtype = parse_dtype(args.cpu_dtype if device.type == "cpu" else args.cuda_dtype)
    cfg = AutoConfig.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        trust_remote_code=True,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        token=token,
    )
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        token=token,
        output_loading_info=True,
    )

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
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    del attention_mask
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
    param_count = int(sum(p.numel() for p in model.parameters()))
    dtype_sample = str(next(model.parameters()).dtype)
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else None

    result = {
        "ok": bool(new_ids and generated_text and all_logits_finite),
        "model": args.model,
        "repo_id": spec["repo_id"],
        "expected_identity": spec["identity"],
        "revision": spec["revision"],
        "expected_sha": spec["expected_sha"],
        "resolved_sha": getattr(info, "sha", None),
        "private": getattr(info, "private", None),
        "branches": [branch.name for branch in refs.branches],
        "tags": [tag.name for tag in refs.tags],
        "device": args.device,
        "torch_dtype_requested": str(torch_dtype),
        "dtype_sample": dtype_sample,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": cuda_name,
        "config_class": cfg.__class__.__name__,
        "model_class": model.__class__.__name__,
        "model_class_module": model.__class__.__module__,
        "core_model_class": module_name(getattr(model, "model", model)),
        "layer_classes": unique_layer_classes(model),
        "model_identity": getattr(cfg, "model_identity", None),
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
        "missing_keys": loading_info.get("missing_keys"),
        "unexpected_keys": loading_info.get("unexpected_keys"),
        "mismatched_keys": loading_info.get("mismatched_keys"),
        "cpu_fallbacks": cpu_fallbacks,
        "elapsed_seconds": round(time.time() - start, 3),
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("ok", "model", "repo_id", "revision", "device", "generated_new_token_ids", "generated_new_text_repr", "all_logits_finite")}, sort_keys=True))

    del model, logits, output_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
