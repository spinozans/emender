#!/usr/bin/env python3
"""Prepare and upload private Hugging Face staging artifacts for v0.1 models.

The script intentionally writes model artifacts outside the git checkout by
default. It creates/preserves private model repos, uploads to a mutable staging
branch, and never creates public repos or immutable release tags.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKDIR = Path(f"/tmp/release-v01-private-hf-staging-{os.environ.get('USER', 'agent')}")
ORG = "poietic-pbc"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    identity: str
    short_name: str
    checkpoint: Path
    smoke_label: str
    smoke_new_token_ids: list[int]
    smoke_new_text_repr: str
    smoke_param_count: int


SPECS = [
    ModelSpec(
        key="e88",
        repo_id=f"{ORG}/emender-e88-1.3b",
        identity="Emender/E88",
        short_name="Emender E88 1.3B",
        checkpoint=Path(
            "/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/"
            "levelE88_1270M_20260511_233832/checkpoint_step_1281000_loss_2.6850.pt"
        ),
        smoke_label="e88",
        smoke_new_token_ids=[218, 218],
        smoke_new_text_repr="'\\x1e\\x1e'",
        smoke_param_count=1_273_191_856,
    ),
    ModelSpec(
        key="gdn",
        repo_id=f"{ORG}/gdn-1.3b",
        identity="GDN",
        short_name="GDN 1.3B",
        checkpoint=Path(
            "/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/"
            "levelfla-gdn_1270M_20260511_233832/checkpoint_step_1686000_loss_2.6105.pt"
        ),
        smoke_label="gdn",
        smoke_new_token_ids=[318, 318],
        smoke_new_text_repr="' is is'",
        smoke_param_count=1_352_352_498,
    ),
    ModelSpec(
        key="m2rnn",
        repo_id=f"{ORG}/m2rnn-cma-1.3b",
        identity="M²RNN-CMA",
        short_name="M2RNN-CMA 1.3B",
        checkpoint=Path(
            "/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/"
            "levelm2rnn_1270M_20260511_175023/checkpoint_step_1212000_loss_2.6870.pt"
        ),
        smoke_label="m2rnn",
        smoke_new_token_ids=[2109, 34059],
        smoke_new_text_repr="'........Officers'",
        smoke_param_count=1_307_101_140,
    ),
]


CONFIGURATION_NDM = '''\
"""Configuration for private NDM/Emender staging checkpoints."""

from __future__ import annotations

from transformers import PretrainedConfig


class NdmConfig(PretrainedConfig):
    model_type = "ndm"

    def __init__(
        self,
        vocab_size: int = 50281,
        dim: int = 1664,
        depth: int = 12,
        level: str = "E88",
        n_heads: int | None = None,
        n_state: int = 32,
        expansion: float = 1.0,
        n_groups: int = 32,
        n_slots: int = 64,
        use_gate: bool = True,
        gate_activation: str = "sigmoid",
        linear_state: bool = False,
        use_write_gate: bool = False,
        e88_decay_mode: str = "mamba",
        e88_value_residual: bool = False,
        r_h_mode: str = "auto",
        state_expansion: int = 2,
        use_conv: bool = False,
        d_conv: int = 4,
        top_k: int | None = None,
        k_fast: int | None = None,
        k_slow: int | None = None,
        checkpoint_interval: int = 16,
        projection_chunk_size: int = 0,
        loss_chunk_size: int = 0,
        use_triton: bool = False,
        m2rnn_paper_shape: bool = False,
        m2rnn_k_head_dim: int | None = None,
        m2rnn_v_head_dim: int | None = None,
        m2rnn_q_heads: int | None = None,
        m2rnn_k_heads: int | None = None,
        m2rnn_v_heads: int | None = None,
        m2rnn_f_heads: int | None = None,
        m2rnn_g_heads: int | None = None,
        m2rnn_weight_heads: int | None = None,
        m2rnn_use_residual: bool = True,
        m2rnn_freeze_state_weight: bool = False,
        m2rnn_output_norm: bool = False,
        m2rnn_normalize_qk: bool = False,
        m2rnn_state_grad_clip: float | None = None,
        tokenizer_name: str = "p50k_base",
        private_staging: bool = True,
        release_revision_name: str = "staging",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            bos_token_id=kwargs.pop("bos_token_id", 50256),
            eos_token_id=kwargs.pop("eos_token_id", 50256),
            pad_token_id=kwargs.pop("pad_token_id", 50256),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", True),
            is_decoder=kwargs.pop("is_decoder", True),
            is_encoder_decoder=kwargs.pop("is_encoder_decoder", False),
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.level = level
        self.n_heads = n_heads
        self.n_state = n_state
        self.expansion = expansion
        self.n_groups = n_groups
        self.n_slots = n_slots
        self.use_gate = use_gate
        self.gate_activation = gate_activation
        self.linear_state = linear_state
        self.use_write_gate = use_write_gate
        self.e88_decay_mode = e88_decay_mode
        self.e88_value_residual = e88_value_residual
        self.r_h_mode = r_h_mode
        self.state_expansion = state_expansion
        self.use_conv = use_conv
        self.d_conv = d_conv
        self.top_k = top_k
        self.k_fast = k_fast
        self.k_slow = k_slow
        self.checkpoint_interval = checkpoint_interval
        self.projection_chunk_size = projection_chunk_size
        self.loss_chunk_size = loss_chunk_size
        self.use_triton = use_triton
        self.m2rnn_paper_shape = m2rnn_paper_shape
        self.m2rnn_k_head_dim = m2rnn_k_head_dim
        self.m2rnn_v_head_dim = m2rnn_v_head_dim
        self.m2rnn_q_heads = m2rnn_q_heads
        self.m2rnn_k_heads = m2rnn_k_heads
        self.m2rnn_v_heads = m2rnn_v_heads
        self.m2rnn_f_heads = m2rnn_f_heads
        self.m2rnn_g_heads = m2rnn_g_heads
        self.m2rnn_weight_heads = m2rnn_weight_heads
        self.m2rnn_use_residual = m2rnn_use_residual
        self.m2rnn_freeze_state_weight = m2rnn_freeze_state_weight
        self.m2rnn_output_norm = m2rnn_output_norm
        self.m2rnn_normalize_qk = m2rnn_normalize_qk
        self.m2rnn_state_grad_clip = m2rnn_state_grad_clip
        self.tokenizer_name = tokenizer_name
        self.private_staging = private_staging
        self.release_revision_name = release_revision_name
'''


MODELING_NDM = '''\
"""HF wrapper for private NDM/Emender staging checkpoints.

This custom-code loader expects the `ndm` source package to be installed in the
runtime environment. The private staging Docker smoke installs the repository
before loading these Hugging Face artifacts.
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_ndm import NdmConfig


def _bool(value) -> bool:
    return bool(value)


def _build_ndm_model(config: NdmConfig) -> nn.Module:
    if str(config.level).lower() == "m2rnn":
        module = importlib.import_module("ndm.models.m2rnn_baseline")
        return module.M2RNNLM(
            vocab_size=config.vocab_size,
            dim=config.dim,
            depth=config.depth,
            n_heads=config.n_heads,
            n_state=config.n_state,
            expansion=config.expansion,
            paper_shape=_bool(config.m2rnn_paper_shape),
            k_head_dim=config.m2rnn_k_head_dim,
            v_head_dim=config.m2rnn_v_head_dim,
            num_q_heads=config.m2rnn_q_heads,
            num_k_heads=config.m2rnn_k_heads,
            num_v_heads=config.m2rnn_v_heads,
            num_f_heads=config.m2rnn_f_heads,
            num_g_heads=config.m2rnn_g_heads,
            num_weight_heads=config.m2rnn_weight_heads,
            use_gate=_bool(config.use_gate),
            use_residual=_bool(config.m2rnn_use_residual),
            state_weight_trainable=not _bool(config.m2rnn_freeze_state_weight),
            use_conv=_bool(config.use_conv),
            d_conv=config.d_conv,
            output_norm=_bool(config.m2rnn_output_norm),
            normalize_qk=_bool(config.m2rnn_normalize_qk),
            dropout=0.0,
            gradient_clipping=config.m2rnn_state_grad_clip,
            gradient_checkpointing=False,
            loss_chunk_size=0,
        )

    module = importlib.import_module("ndm.models.ladder_lm")
    return module.LadderLM(
        vocab_size=config.vocab_size,
        dim=config.dim,
        depth=config.depth,
        level=config.level,
        expansion=config.expansion,
        n_groups=config.n_groups,
        n_state=config.n_state,
        n_slots=config.n_slots,
        n_heads=config.n_heads,
        top_k=config.top_k,
        k_fast=config.k_fast,
        k_slow=config.k_slow,
        use_gate=_bool(config.use_gate),
        gate_activation=config.gate_activation,
        linear_state=_bool(config.linear_state),
        use_write_gate=_bool(config.use_write_gate),
        e88_decay_mode=config.e88_decay_mode,
        e88_value_residual=_bool(config.e88_value_residual),
        state_expansion=config.state_expansion,
        r_h_mode=config.r_h_mode,
        use_conv=_bool(config.use_conv),
        d_conv=config.d_conv,
        dropout=0.0,
        checkpoint_interval=config.checkpoint_interval,
        gradient_checkpointing=False,
        projection_chunk_size=0,
        loss_chunk_size=0,
        use_triton=_bool(config.use_triton),
    )


class NdmForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NdmConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: NdmConfig):
        super().__init__(config)
        self.model = _build_ndm_model(config)

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def tie_weights(self):
        if hasattr(self.model, "lm_head") and hasattr(self.model, "embedding"):
            self.model.lm_head.weight = self.model.embedding.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        del attention_mask, kwargs
        if input_ids is None:
            raise ValueError("NdmForCausalLM requires input_ids")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        logits = self.model(input_ids, return_loss=False)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        del past_key_values
        return {"input_ids": input_ids}
'''


def load_args(checkpoint: Path) -> dict[str, Any]:
    args_path = checkpoint.parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"missing args.json beside checkpoint: {args_path}")
    return json.loads(args_path.read_text())


def bool_arg(value: Any) -> bool:
    return bool(value)


def clean_model_args(args: dict[str, Any]) -> dict[str, Any]:
    """Keep architecture fields and drop local path/training-only fields."""

    keys = {
        "level",
        "dim",
        "depth",
        "n_heads",
        "n_state",
        "expansion",
        "n_groups",
        "n_slots",
        "use_gate",
        "gate_activation",
        "linear_state",
        "use_write_gate",
        "state_expansion",
        "r_h_mode",
        "use_conv",
        "d_conv",
        "top_k",
        "k_fast",
        "k_slow",
        "checkpoint_interval",
        "projection_chunk_size",
        "loss_chunk_size",
        "use_triton",
        "m2rnn_paper_shape",
        "m2rnn_k_head_dim",
        "m2rnn_v_head_dim",
        "m2rnn_q_heads",
        "m2rnn_k_heads",
        "m2rnn_v_heads",
        "m2rnn_f_heads",
        "m2rnn_g_heads",
        "m2rnn_weight_heads",
        "m2rnn_use_residual",
        "m2rnn_freeze_state_weight",
        "m2rnn_output_norm",
        "m2rnn_normalize_qk",
        "m2rnn_state_grad_clip",
        "tokenizer",
        "chunk_size",
        "params",
        "bf16",
        "seed",
    }
    cleaned = {k: args.get(k) for k in sorted(keys) if k in args}
    cleaned["use_gate"] = bool_arg(cleaned.get("use_gate", True))
    cleaned["linear_state"] = bool_arg(cleaned.get("linear_state", False))
    cleaned["use_write_gate"] = bool_arg(cleaned.get("use_write_gate", False))
    cleaned["use_conv"] = bool_arg(cleaned.get("use_conv", False))
    cleaned["use_triton"] = bool_arg(cleaned.get("use_triton", False))
    cleaned["m2rnn_paper_shape"] = bool_arg(cleaned.get("m2rnn_paper_shape", False))
    cleaned["m2rnn_output_norm"] = bool_arg(cleaned.get("m2rnn_output_norm", False))
    cleaned["m2rnn_normalize_qk"] = bool_arg(cleaned.get("m2rnn_normalize_qk", False))
    cleaned["bf16"] = bool_arg(cleaned.get("bf16", True))
    return cleaned


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_tokenizer_files(out_dir: Path, tokenizer_name: str) -> dict[str, Any]:
    if tokenizer_name != "p50k_base":
        raise ValueError(f"unsupported tokenizer for staging export: {tokenizer_name}")
    import tiktoken
    from transformers.convert_slow_tokenizer import TikTokenConverter

    enc = tiktoken.get_encoding(tokenizer_name)
    tiktoken_dir = out_dir / "tiktoken"
    tiktoken_dir.mkdir(parents=True, exist_ok=True)
    bpe_path = tiktoken_dir / "tokenizer.model"
    with bpe_path.open("wb") as handle:
        for token, rank in sorted(enc._mergeable_ranks.items(), key=lambda item: item[1]):
            handle.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")

    tokenizer = TikTokenConverter(
        vocab_file=str(bpe_path),
        pattern=enc._pat_str,
        additional_special_tokens=enc._special_tokens,
    ).converted()
    tokenizer.save(str(out_dir / "tokenizer.json"))

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_input_names": ["input_ids", "attention_mask"],
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "name_or_path": tokenizer_name,
        "private_staging_tokenizer": tokenizer_name,
    }
    special_tokens = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    write_json(out_dir / "tokenizer_config.json", tokenizer_config)
    write_json(out_dir / "special_tokens_map.json", special_tokens)
    return {"tokenizer_name": tokenizer_name, "vocab_size": enc.n_vocab, "eot_token": enc.eot_token}


def build_config(spec: ModelSpec, model_args: dict[str, Any], ckpt_meta: dict[str, Any], tokenizer_meta: dict[str, Any]) -> dict[str, Any]:
    clean_args = clean_model_args(model_args)
    level = clean_args["level"]
    config = {
        "model_type": "ndm",
        "architectures": ["NdmForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_ndm.NdmConfig",
            "AutoModelForCausalLM": "modeling_ndm.NdmForCausalLM",
        },
        "torch_dtype": "bfloat16",
        "vocab_size": tokenizer_meta["vocab_size"],
        "bos_token_id": tokenizer_meta["eot_token"],
        "eos_token_id": tokenizer_meta["eot_token"],
        "pad_token_id": tokenizer_meta["eot_token"],
        "tie_word_embeddings": True,
        "private_staging": True,
        "release_revision_name": "staging",
        "model_identity": spec.identity,
        "repo_id": spec.repo_id,
        "checkpoint_step": ckpt_meta["step"],
        "checkpoint_loss": ckpt_meta["loss"],
        "smoke_param_count": spec.smoke_param_count,
        "tokenizer_name": tokenizer_meta["tokenizer_name"],
        "level": level,
        "dim": clean_args["dim"],
        "depth": clean_args["depth"],
        "n_heads": clean_args.get("n_heads"),
        "n_state": clean_args.get("n_state", 64),
        "expansion": clean_args.get("expansion", 1.0),
        "n_groups": clean_args.get("n_groups", 32),
        "n_slots": clean_args.get("n_slots", 64),
        "use_gate": clean_args.get("use_gate", True),
        "gate_activation": clean_args.get("gate_activation", "sigmoid"),
        "linear_state": clean_args.get("linear_state", False),
        "use_write_gate": clean_args.get("use_write_gate", False),
        "e88_decay_mode": clean_args.get("e88_decay_mode", "mamba"),
        "e88_value_residual": clean_args.get("e88_value_residual", False),
        "r_h_mode": clean_args.get("r_h_mode", "auto"),
        "state_expansion": clean_args.get("state_expansion", 2),
        "use_conv": clean_args.get("use_conv", False),
        "d_conv": clean_args.get("d_conv", 4),
        "top_k": clean_args.get("top_k"),
        "k_fast": clean_args.get("k_fast"),
        "k_slow": clean_args.get("k_slow"),
        "checkpoint_interval": clean_args.get("checkpoint_interval", 16),
        "projection_chunk_size": 0,
        "loss_chunk_size": 0,
        "use_triton": clean_args.get("use_triton", False),
        "m2rnn_paper_shape": clean_args.get("m2rnn_paper_shape", False),
        "m2rnn_k_head_dim": clean_args.get("m2rnn_k_head_dim"),
        "m2rnn_v_head_dim": clean_args.get("m2rnn_v_head_dim"),
        "m2rnn_q_heads": clean_args.get("m2rnn_q_heads"),
        "m2rnn_k_heads": clean_args.get("m2rnn_k_heads"),
        "m2rnn_v_heads": clean_args.get("m2rnn_v_heads"),
        "m2rnn_f_heads": clean_args.get("m2rnn_f_heads"),
        "m2rnn_g_heads": clean_args.get("m2rnn_g_heads"),
        "m2rnn_weight_heads": clean_args.get("m2rnn_weight_heads"),
        "m2rnn_use_residual": clean_args.get("m2rnn_use_residual", True),
        "m2rnn_freeze_state_weight": clean_args.get("m2rnn_freeze_state_weight", False),
        "m2rnn_output_norm": clean_args.get("m2rnn_output_norm", False),
        "m2rnn_normalize_qk": clean_args.get("m2rnn_normalize_qk", False),
        "m2rnn_state_grad_clip": clean_args.get("m2rnn_state_grad_clip"),
        "ndm_training_args_sanitized": clean_args,
    }
    return config


def write_model_card(out_dir: Path, spec: ModelSpec, config: dict[str, Any], safetensors_size: int) -> None:
    text = f"""\
---
library_name: transformers
pipeline_tag: text-generation
private_staging: true
tags:
- private-staging
- ndm
- emender
- recurrent-language-model
---

# {spec.short_name}

**PRIVATE STAGING STATUS:** this repository is private staging only under
`{ORG}`. It is not public release material, must not be redistributed, and must
not be treated as approval to publish the model or create immutable `v0.1`
release tags.

## Model Identity

- Identity: {spec.identity}
- Repository: `{spec.repo_id}`
- Intended revision for downstream tests: `staging`
- Checkpoint step: `{config["checkpoint_step"]}`
- Checkpoint loss recorded in source checkpoint: `{config["checkpoint_loss"]}`
- Parameter count from local smoke model construction: `{spec.smoke_param_count:,}`
- Tokenizer: `p50k_base`

## Staging Artifact Contents

This staging revision contains the minimal normal-load artifact set:

- `config.json`
- `configuration_ndm.py`
- `modeling_ndm.py`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `generation_config.json`
- `requirements.txt`
- `model.safetensors`

The safetensors file was converted from the selected local checkpoint's
`model_state_dict`; raw training checkpoint files and optimizer states were not
uploaded. Local converted safetensors size before upload:
`{safetensors_size}` bytes.

## Smoke Evidence

The selected checkpoint passed local and Docker-local CPU/GPU smoke tests before
this private staging upload. The minimal generation prompt was `The theorem
states`, with greedy `max_new_tokens=2`.

- Smoke label: `{spec.smoke_label}`
- New token IDs: `{spec.smoke_new_token_ids}`
- Decoded new text: `{spec.smoke_new_text_repr}`
- Logits finite: `true`

## Loading Notes

This private staging loader uses Hugging Face custom code
(`trust_remote_code=True`) and expects the matching `ndm` source package to be
installed in the runtime environment. The downstream private-HF Docker smoke is
responsible for validating that container load path before any public release
decision.

Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{spec.repo_id}"
revision = "staging"
tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision, token=True)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    revision=revision,
    trust_remote_code=True,
    token=True,
)
```

No repository visibility should be changed to public based on this staging
upload.
"""
    out_dir.joinpath("README.md").write_text(text)


def convert_weights(spec: ModelSpec, out_dir: Path, force: bool) -> dict[str, Any]:
    safetensors_path = out_dir / "model.safetensors"
    if safetensors_path.exists() and safetensors_path.stat().st_size > 0 and not force:
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()
        return {
            "path": str(safetensors_path),
            "size": safetensors_path.stat().st_size,
            "keys": len(keys),
            "metadata": metadata,
            "skipped_existing": True,
        }

    print(f"[{spec.key}] loading checkpoint for safetensors conversion: {spec.checkpoint}", flush=True)
    ckpt = torch.load(str(spec.checkpoint), map_location="cpu", mmap=True, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    tensors: dict[str, torch.Tensor] = {}
    seen_storage: dict[tuple[int, int, int, tuple[int, ...], tuple[int, ...]], str] = {}
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        prefixed = f"model.{name}"
        value = tensor.detach().cpu().contiguous()
        storage_key = (
            value.untyped_storage().data_ptr(),
            value.storage_offset(),
            value.numel(),
            tuple(value.size()),
            tuple(value.stride()),
        )
        if storage_key in seen_storage:
            value = value.clone()
        else:
            seen_storage[storage_key] = prefixed
        tensors[prefixed] = value
    metadata = {
        "format": "pt",
        "private_staging": "true",
        "repo_id": spec.repo_id,
        "model_identity": spec.identity,
        "checkpoint_step": str(ckpt.get("step")),
        "checkpoint_loss": str(ckpt.get("loss")),
        "source_state_dict": "model_state_dict",
    }
    save_file(tensors, str(safetensors_path), metadata=metadata)
    del tensors, state_dict, ckpt

    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        saved_metadata = handle.metadata()
    return {
        "path": str(safetensors_path),
        "size": safetensors_path.stat().st_size,
        "keys": len(keys),
        "metadata": saved_metadata,
        "skipped_existing": False,
    }


def prepare_artifacts(spec: ModelSpec, workdir: Path, force: bool) -> dict[str, Any]:
    if not spec.checkpoint.exists():
        raise FileNotFoundError(spec.checkpoint)
    model_args = load_args(spec.checkpoint)
    ckpt_head = torch.load(str(spec.checkpoint), map_location="cpu", mmap=True, weights_only=False)
    ckpt_meta = {"step": ckpt_head.get("step"), "loss": ckpt_head.get("loss")}
    del ckpt_head

    out_dir = workdir / spec.key
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_meta = write_tokenizer_files(out_dir, model_args.get("tokenizer", "p50k_base"))
    config = build_config(spec, model_args, ckpt_meta, tokenizer_meta)
    write_json(out_dir / "config.json", config)
    write_json(
        out_dir / "generation_config.json",
        {
            "bos_token_id": tokenizer_meta["eot_token"],
            "eos_token_id": tokenizer_meta["eot_token"],
            "pad_token_id": tokenizer_meta["eot_token"],
            "do_sample": False,
            "max_new_tokens": 32,
            "transformers_version": "4.57.3",
        },
    )
    out_dir.joinpath("configuration_ndm.py").write_text(CONFIGURATION_NDM)
    out_dir.joinpath("modeling_ndm.py").write_text(MODELING_NDM)
    out_dir.joinpath("requirements.txt").write_text(
        "\n".join(
            [
                "torch>=2.9",
                "transformers>=4.57",
                "safetensors>=0.7",
                "tiktoken>=0.12",
                "# Runtime must also install the matching ndm source checkout.",
                "",
            ]
        )
    )
    weights = convert_weights(spec, out_dir, force=force)
    write_model_card(out_dir, spec, config, weights["size"])
    return {
        "repo_id": spec.repo_id,
        "identity": spec.identity,
        "artifact_dir": str(out_dir),
        "config": config,
        "weights": weights,
        "files": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()),
    }


def validate_local_artifact(spec: ModelSpec, artifact_dir: Path) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(str(artifact_dir), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
    encoded = tokenizer.encode("The theorem states")
    expected = [464, 44728, 2585]
    if encoded != expected:
        raise RuntimeError(f"{spec.key} tokenizer mismatch: {encoded} != {expected}")
    with safe_open(artifact_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys:
            raise RuntimeError(f"{spec.key} safetensors has no keys")
        if "model.embedding.weight" not in keys:
            raise RuntimeError(f"{spec.key} safetensors missing model.embedding.weight")
    return {
        "config_class": config.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "prompt_token_ids": encoded,
        "safetensors_keys": len(keys),
    }


def verify_auth(api: HfApi) -> dict[str, Any]:
    who = api.whoami(token=True)
    orgs = who.get("orgs") or []
    org_names = {org.get("name") for org in orgs if isinstance(org, dict)}
    if ORG not in org_names:
        raise RuntimeError(f"authenticated account does not list {ORG} membership")
    return {"name": who.get("name"), "orgs": sorted(name for name in org_names if name)}


def private_repo_info(api: HfApi, repo_id: str, revision: str | None = None) -> dict[str, Any]:
    info = api.repo_info(repo_id, repo_type="model", revision=revision, token=True)
    private = getattr(info, "private", None)
    if private is not True:
        raise RuntimeError(f"{repo_id} exists but private={private}; refusing to update")
    return {
        "repo_id": repo_id,
        "private": private,
        "sha": getattr(info, "sha", None),
        "siblings": len(getattr(info, "siblings", []) or []),
        "revision": revision,
    }


def ensure_private_repo(api: HfApi, repo_id: str) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    try:
        info = private_repo_info(api, repo_id)
        events.append({"action": "repo_info", "result": info, "command": f"HfApi().repo_info({repo_id!r}, repo_type='model', token=True)"})
        return {"created": False, "events": events}
    except RepositoryNotFoundError:
        events.append({"action": "repo_info_missing", "repo_id": repo_id})

    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=False, token=True)
    events.append(
        {
            "action": "create_repo",
            "repo_id": repo_id,
            "private_requested": True,
            "command": f"HfApi().create_repo(repo_id={repo_id!r}, repo_type='model', private=True, exist_ok=False, token=True)",
        }
    )
    info = private_repo_info(api, repo_id)
    events.append({"action": "repo_info_after_create", "result": info})
    return {"created": True, "events": events}


def bootstrap_main_and_branch(api: HfApi, repo_id: str, revision: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    bootstrap_text = (
        "# Private staging bootstrap\n\n"
        "This private repo is initialized only so the mutable staging branch can be created. "
        "Full v0.1 release-candidate artifacts are uploaded to the `staging` branch.\n"
    )
    from huggingface_hub import CommitOperationAdd

    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        token=True,
        commit_message="Initialize private staging repository",
        operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=bootstrap_text.encode())],
    )
    events.append(
        {
            "action": "bootstrap_main",
            "repo_id": repo_id,
            "commit": commit.oid,
            "command": "HfApi().create_commit(... revision='main', path_in_repo='README.md')",
        }
    )
    api.create_branch(repo_id=repo_id, repo_type="model", branch=revision, revision="main", exist_ok=True, token=True)
    events.append(
        {
            "action": "create_branch",
            "repo_id": repo_id,
            "branch": revision,
            "source_revision": "main",
            "command": f"HfApi().create_branch(repo_id={repo_id!r}, repo_type='model', branch={revision!r}, revision='main', exist_ok=True, token=True)",
        }
    )
    return events


def refs_summary(api: HfApi, repo_id: str) -> dict[str, Any]:
    refs = api.list_repo_refs(repo_id, repo_type="model", token=True)
    branches = [branch.name for branch in refs.branches]
    tags = [tag.name for tag in refs.tags]
    if "v0.1" in tags:
        raise RuntimeError(f"{repo_id} has forbidden immutable v0.1 tag after this task")
    return {"branches": branches, "tags": tags, "v0.1_tag_present": "v0.1" in tags}


def upload_artifact(api: HfApi, prepared: dict[str, Any], revision: str, dry_run: bool) -> dict[str, Any]:
    repo_id = prepared["repo_id"]
    artifact_dir = Path(prepared["artifact_dir"])
    events = ensure_private_repo(api, repo_id)
    upload_events = list(events["events"])
    if dry_run:
        return {"repo_id": repo_id, "dry_run": True, "events": upload_events}

    upload_events.extend(bootstrap_main_and_branch(api, repo_id, revision))
    command = (
        "HfApi().upload_folder("
        f"repo_id={repo_id!r}, folder_path={str(artifact_dir)!r}, "
        "repo_type='model', revision='staging', token=True, "
        "commit_message='Upload private v0.1 staging artifacts')"
    )
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(artifact_dir),
        repo_type="model",
        revision=revision,
        token=True,
        commit_message="Upload private v0.1 staging artifacts",
        ignore_patterns=[".cache/**", "*.pt", "*.pth"],
    )
    upload_events.append(
        {
            "action": "upload_folder",
            "repo_id": repo_id,
            "revision": revision,
            "commit": commit.oid,
            "commit_url": commit.commit_url,
            "command": command,
        }
    )
    info = private_repo_info(api, repo_id, revision=revision)
    upload_events.append({"action": "repo_info_after_upload", "result": info})
    upload_events.append({"action": "refs_after_upload", "result": refs_summary(api, repo_id)})
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": revision,
        "commit": commit.oid,
        "commit_url": commit.commit_url,
        "private": info["private"],
        "events": upload_events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--revision", default="staging")
    parser.add_argument("--models", nargs="*", choices=[spec.key for spec in SPECS], default=[spec.key for spec in SPECS])
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and validate local artifacts without uploading.")
    parser.add_argument("--dry-run-upload", action="store_true", help="Exercise private repo checks but skip upload commits.")
    parser.add_argument("--force", action="store_true", help="Rebuild artifact directories and safetensors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.revision == "v0.1":
        raise SystemExit("refusing to create or upload to immutable v0.1")
    selected = [spec for spec in SPECS if spec.key in args.models]
    args.workdir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "started_at_unix": time.time(),
        "workdir": str(args.workdir),
        "revision": args.revision,
        "prepare_only": args.prepare_only,
        "dry_run_upload": args.dry_run_upload,
        "models": [],
        "auth": None,
        "uploads": [],
    }

    api = HfApi()
    auth = verify_auth(api)
    manifest["auth"] = auth
    print(f"HF auth OK for user={auth['name']} orgs={auth['orgs']} (token not printed)", flush=True)

    for spec in selected:
        print(f"[{spec.key}] preparing artifacts for {spec.repo_id}", flush=True)
        prepared = prepare_artifacts(spec, args.workdir, force=args.force)
        validation = validate_local_artifact(spec, Path(prepared["artifact_dir"]))
        prepared["local_validation"] = validation
        manifest["models"].append(prepared)
        print(
            f"[{spec.key}] prepared {len(prepared['files'])} files; "
            f"safetensors={prepared['weights']['size']} bytes; validation={validation}",
            flush=True,
        )

    if not args.prepare_only:
        for prepared in manifest["models"]:
            print(f"[{prepared['repo_id']}] uploading to revision {args.revision}", flush=True)
            upload = upload_artifact(api, prepared, args.revision, dry_run=args.dry_run_upload)
            manifest["uploads"].append(upload)
            print(f"[{prepared['repo_id']}] upload result: {upload}", flush=True)

    manifest["finished_at_unix"] = time.time()
    manifest_path = args.workdir / "upload_manifest.json"
    write_json(manifest_path, manifest)
    print(f"manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
