#!/usr/bin/env python3
"""Convert the E97 4B Pi-instruction checkpoint (bf16, optimizer-state-laden
.pt) into a GGUF v3 file for the llama.cpp CPU port (Stage P1).

CPU-only by construction: torch.load(map_location='cpu', mmap=True) never
touches CUDA and the optimizer state pages are never materialized.

Tensor mapping (checkpoint name -> GGUF name), verified against the promoted
checkpoint's actual state dict. The main mapping trap: the pre-mixer norms
live in a top-level ``layer_norms.{i}`` array (i in 0..17) with the final norm
at ``layer_norms.18`` — there is no ``layers.{i}.norm``:

  embedding.weight            -> token_embd.weight          [50281, 3840] (tied;
                                 output.weight is NOT written; the fork ties
                                 via emender_e97.tie_word_embeddings=true)
  layer_norms.{i}.weight      -> blk.{i}.attn_norm.weight    (pre-mixer RMSNorm)
  layers.{i}.norm_2.weight    -> blk.{i}.ffn_norm.weight     (pre-MLP RMSNorm)
  layers.{i}.mixer.qkv_proj.weight -> blk.{i}.mixer_qkv.weight  [11520, 3840] (fused)
  layers.{i}.mixer.a_proj.weight   -> blk.{i}.mixer_a.weight    [60, 3840]
  layers.{i}.mixer.A_log           -> blk.{i}.mixer_a_log       [60] upcast f32
  layers.{i}.mixer.dt_bias         -> blk.{i}.mixer_dt_bias     [60] upcast f32
  layers.{i}.mixer.erase_gate_proj.weight -> blk.{i}.mixer_erase.weight [3840, 3840]
  layers.{i}.mixer.value_write_gate_proj.weight -> blk.{i}.mixer_vw.weight
  layers.{i}.mixer.g_proj.weight   -> blk.{i}.mixer_g.weight    [3840, 3840]
  layers.{i}.mixer.o_proj.weight   -> blk.{i}.mixer_o.weight    [3840, 3840]
  layers.{i}.mlp.w1.weight        -> blk.{i}.ffn_gate.weight     [9600, 3840]
  layers.{i}.mlp.w2.weight        -> blk.{i}.ffn_up.weight       [9600, 3840]
  layers.{i}.mlp.w3.weight        -> blk.{i}.ffn_down.weight     [3840, 9600]
  norm.weight (final)             -> output_norm.weight

Usage:
  .venv/bin/python scripts/convert_e97_4b_to_gguf.py \
      --checkpoint <checkpoint.pt> --output <out.gguf>
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e97_gguf_io import GGUFWriter, GGUFReader  # noqa: E402

ARCH = "emender_e97"
N_LAYER = 18
D_MODEL = 3840
N_HEAD = 60
HEAD_DIM = 64
MLP_HIDDEN = 9600
VOCAB = 50281
RMS_EPS = 1e-5
CONTEXT = 65536
EOT_ID = 50256
N_BASE_MERGES = 50000


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def build_vocab_and_merges(enc):
    from e97_p50k_codec import tokens_and_merges
    return tokens_and_merges(enc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-sha256", default="d81464982c3ebc0d72769a87e079068bf535d6ca2010185dc1bb03ce264b8f5b")
    ap.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    t0 = time.time()
    digest = sha256_of(args.checkpoint)
    print(f"checkpoint sha256 {digest} ({args.checkpoint.stat().st_size / 2**30:.1f} GiB)", flush=True)
    if digest != args.expect_sha256:
        raise SystemExit("checkpoint SHA-256 mismatch")

    ck = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["model_state_dict"]
    assert len(sd) == 237, f"expected 237 tensors, got {len(sd)}"
    assert ck["weight_mode"] == "saved-eval-x"
    print("state dict mapped (mmap; optimizer state pages never materialized)", flush=True)

    # ---- shape/dtype contract ----
    expected = {
        "embedding.weight": ((VOCAB, D_MODEL), torch.bfloat16),
        "norm.weight": ((D_MODEL,), torch.bfloat16),
        "layers.0.mixer.qkv_proj.weight": ((3 * D_MODEL, D_MODEL), torch.bfloat16),
        "layers.0.mixer.a_proj.weight": ((N_HEAD, D_MODEL), torch.bfloat16),
        "layers.0.mixer.A_log": ((N_HEAD,), torch.bfloat16),
        "layers.0.mixer.dt_bias": ((N_HEAD,), torch.bfloat16),
        "layers.0.mixer.o_proj.weight": ((D_MODEL, D_MODEL), torch.bfloat16),
        "layers.0.mlp.w1.weight": ((MLP_HIDDEN, D_MODEL), torch.bfloat16),
        "layers.0.mlp.w3.weight": ((D_MODEL, MLP_HIDDEN), torch.bfloat16),
        "layer_norms.0.weight": ((D_MODEL,), torch.bfloat16),
        "norm.weight": ((D_MODEL,), torch.bfloat16),
    }
    for key, (shape, dtype) in expected.items():
        got = tuple(sd[key].shape)
        assert got == shape, f"{key}: {got} != {shape}"
        assert sd[key].dtype == dtype, f"{key}: {sd[key].dtype} != {dtype}"

    # ---- tokenizer ----
    import tiktoken

    enc = tiktoken.get_encoding("p50k_base")
    tokens, merges, _vb = build_vocab_and_merges(enc)
    print(f"tokenizer: {len(tokens)} tokens, {len(merges)} merges", flush=True)

    # ---- write GGUF ----
    w = GGUFWriter(args.output)
    w.add_kv("general.architecture", ARCH)
    w.add_kv("general.name", "emender-e97-4b-pi-instruction")
    w.add_kv("general.file_type", 0)  # all F32
    w.add_kv("general.source_checkpoint_sha256", digest)
    w.add_kv(f"{ARCH}.context_length", CONTEXT)
    w.add_kv(f"{ARCH}.embedding_length", D_MODEL)
    w.add_kv(f"{ARCH}.block_count", N_LAYER)
    w.add_kv(f"{ARCH}.feed_forward_length", MLP_HIDDEN)
    w.add_kv(f"{ARCH}.attention.head_count", N_HEAD)
    w.add_kv(f"{ARCH}.attention.key_length", HEAD_DIM)
    w.add_kv(f"{ARCH}.attention.value_length", HEAD_DIM)
    w.add_kv(f"{ARCH}.attention.layer_norm_rms_epsilon", RMS_EPS)
    w.add_kv(f"{ARCH}.mixer.state_dim", HEAD_DIM)
    w.add_kv(f"{ARCH}.tie_word_embeddings", True)
    w.add_kv("tokenizer.ggml.model", "gpt2")
    w.add_kv("tokenizer.ggml.tokens", tokens)
    w.add_kv("tokenizer.ggml.merges", merges)
    w.add_kv("tokenizer.ggml.eos_token_id", EOT_ID)
    w.add_kv("tokenizer.ggml.add_bos_token", False)

    plan: list[tuple[str, str]] = [("token_embd.weight", "embedding.weight")]
    for i in range(N_LAYER):
        plan += [
            (f"blk.{i}.attn_norm.weight", f"layer_norms.{i}.weight"),
            (f"blk.{i}.mixer_qkv.weight", f"layers.{i}.mixer.qkv_proj.weight"),
            (f"blk.{i}.mixer_a.weight", f"layers.{i}.mixer.a_proj.weight"),
            (f"blk.{i}.mixer_a_log", f"layers.{i}.mixer.A_log"),
            (f"blk.{i}.mixer_dt_bias", f"layers.{i}.mixer.dt_bias"),
            (f"blk.{i}.mixer_erase.weight", f"layers.{i}.mixer.erase_gate_proj.weight"),
            (f"blk.{i}.mixer_vw.weight", f"layers.{i}.mixer.value_write_gate_proj.weight"),
            (f"blk.{i}.mixer_g.weight", f"layers.{i}.mixer.g_proj.weight"),
            (f"blk.{i}.mixer_o.weight", f"layers.{i}.mixer.o_proj.weight"),
            (f"blk.{i}.ffn_norm.weight", f"layers.{i}.norm_2.weight"),
            (f"blk.{i}.ffn_gate.weight", f"layers.{i}.mlp.w1.weight"),
            (f"blk.{i}.ffn_up.weight", f"layers.{i}.mlp.w2.weight"),
            (f"blk.{i}.ffn_down.weight", f"layers.{i}.mlp.w3.weight"),
        ]
    plan.append(("output_norm.weight", "norm.weight"))
    assert len(plan) == 236  # 237 minus lm_head (tied)

    for idx, (gname, ckpt_key) in enumerate(plan):
        t = sd[ckpt_key]
        w.add_f32_tensor(gname, t)
        if idx % 32 == 0:
            print(f"  [{idx + 1}/{len(plan)}] {gname} <- {ckpt_key} {tuple(t.shape)}", flush=True)
    w.close()
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB) in {time.time() - t0:.0f}s", flush=True)

    # ---- verify: bitwise equality of every tensor vs the checkpoint ----
    if args.verify:
        r = GGUFReader(args.output)
        n_bad = 0
        for gname, ckpt_key in plan:
            ref = sd[ckpt_key].float().numpy()
            got = r.get_f32_tensor(gname)
            if got.shape != ref.shape or not np.array_equal(got, ref):
                print(f"MISMATCH {gname}")
                n_bad += 1
        assert r.tensor_infos["token_embd.weight"]["dims"] == [D_MODEL, VOCAB]
        print(f"verification: {len(plan) - n_bad}/{len(plan)} tensors bitwise-identical to checkpoint fp32 upcast")
        if n_bad:
            raise SystemExit("tensor verification failed")


if __name__ == "__main__":
    main()
