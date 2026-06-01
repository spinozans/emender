#!/usr/bin/env python3
"""Fix #1 verification: initialize the TRAINABLE harness model FROM the public
HF v0.3 safetensors (strict load), prove weight-equivalence, and show load
sanity on the canonical held-out slice.

For each model:
  1. Load the (already fresh-downloaded) public @v0.3 model.safetensors; read its
     checkpoint_step provenance from the safetensors metadata.
  2. Build the live elman-harness model from the pinned args.json, strip the HF
     `model.` prefix, drop the tied `model.lm_head.weight`, and STRICT-load the
     v0.3 weights into the harness model. Report missing/unexpected = none.
  3. Independently build a 2nd harness model and load the pinned .pt via the
     y-mode swap (exactly what the PRIOR run used). Compare the two state dicts
     tensor-by-tensor (max abs diff) -> settles whether v0.3-init == prior init.
  4. Run the canonical held-out slice forward through the v0.3-initialised
     harness model -> nats/token (load sanity, expect ~2.55-2.56).

GPU 0. REAL data only.
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")
CLEAN = "/tmp/v03-init-clean-cache"
os.environ["HF_HOME"] = CLEAN
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CLEAN, "hub")

ELMAN_DIR = "/home/erikg/elman"
sys.path.insert(0, ELMAN_DIR)
sys.path.insert(0, os.path.join(ELMAN_DIR, "elman", "cuda"))
sys.path.insert(0, "/home/erikg/ndm/.wg-worktrees/agent-757/scripts")  # measure_pile_bpb_elman

import json as _json
from pathlib import Path
from types import SimpleNamespace
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import measure_pile_bpb_elman as H

REPOS = {
    "e88":   ("poietic-pbc/emender-e88-1.3b",  "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/e88"),
    "gdn":   ("poietic-pbc/gdn-1.3b",          "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/gdn"),
    "m2rnn": ("poietic-pbc/m2rnn-cma-1.3b",    "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/m2rnn"),
}
CKPT_PT = {
    "e88":  "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/e88/checkpoint_step_1542000_loss_2.5970.pt",
    "gdn":  "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/gdn/checkpoint_step_2031000_loss_2.7303.pt",
    "m2rnn":"/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/m2rnn/checkpoint_step_1491000_loss_2.7347.pt",
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_level(s):
    if isinstance(s, str) and s.startswith("log_"): return s
    try: return int(s)
    except (ValueError, TypeError): return s


def resolve_r_h_mode(level):
    full = {1, 33, 42, 51, 52, 53, 56, 57, 58, 60}
    li = int(level) if str(level).isdigit() else 0
    return "spectral_norm" if li in full else "none"


def build_model(a_json, vocab_size):
    a = SimpleNamespace(**a_json)
    level = parse_level(a.level)
    if str(a.level).lower() == "m2rnn":
        from elman.models.m2rnn_baseline import M2RNNLM
        return M2RNNLM(
            vocab_size=vocab_size, dim=a.dim, depth=a.depth, n_heads=a.n_heads,
            n_state=a.n_state, expansion=a.expansion,
            paper_shape=bool(getattr(a, "m2rnn_paper_shape", False)),
            k_head_dim=getattr(a, "m2rnn_k_head_dim", None),
            v_head_dim=getattr(a, "m2rnn_v_head_dim", None),
            num_q_heads=getattr(a, "m2rnn_q_heads", None),
            num_k_heads=getattr(a, "m2rnn_k_heads", None),
            num_v_heads=getattr(a, "m2rnn_v_heads", None),
            num_f_heads=getattr(a, "m2rnn_f_heads", None),
            num_g_heads=getattr(a, "m2rnn_g_heads", None),
            num_weight_heads=getattr(a, "m2rnn_weight_heads", None),
            use_gate=bool(getattr(a, "use_gate", 1)),
            use_residual=bool(getattr(a, "m2rnn_use_residual", 1)),
            state_weight_trainable=not bool(getattr(a, "m2rnn_freeze_state_weight", 0)),
            use_conv=bool(getattr(a, "use_conv", 0)), d_conv=getattr(a, "d_conv", 4),
            output_norm=bool(getattr(a, "m2rnn_output_norm", 0)),
            normalize_qk=bool(getattr(a, "m2rnn_normalize_qk", 0)),
            dropout=getattr(a, "dropout", 0.0),
            gradient_clipping=getattr(a, "m2rnn_state_grad_clip", None),
            gradient_checkpointing=False,
            loss_chunk_size=getattr(a, "loss_chunk_size", 0))
    from elman.models import LadderLM
    return LadderLM(
        vocab_size=vocab_size, dim=a.dim, depth=a.depth, level=level,
        expansion=getattr(a, "expansion", 1.0), n_groups=getattr(a, "n_groups", 32),
        n_state=getattr(a, "n_state", 64), n_slots=getattr(a, "n_slots", 64),
        n_heads=getattr(a, "n_heads", None), top_k=getattr(a, "top_k", None),
        k_fast=getattr(a, "k_fast", None), k_slow=getattr(a, "k_slow", None),
        use_gate=bool(getattr(a, "use_gate", 1)),
        gate_activation=getattr(a, "gate_activation", "sigmoid"),
        linear_state=bool(getattr(a, "linear_state", 0)),
        use_write_gate=bool(getattr(a, "use_write_gate", 0)),
        e88_decay_mode=getattr(a, "e88_decay_mode", "mamba"),
        e88_value_residual=bool(getattr(a, "e88_value_residual", 0)),
        e88_raw_write=bool(getattr(a, "e88_raw_write", 0)),
        state_expansion=getattr(a, "state_expansion", 2),
        r_h_mode=resolve_r_h_mode(level), use_conv=bool(getattr(a, "use_conv", 0)),
        d_conv=getattr(a, "d_conv", 4), dropout=getattr(a, "dropout", 0.0),
        checkpoint_interval=getattr(a, "checkpoint_interval", 16),
        gradient_checkpointing=False,
        projection_chunk_size=getattr(a, "projection_chunk_size", 0),
        loss_chunk_size=getattr(a, "loss_chunk_size", 0),
        use_triton=bool(getattr(a, "use_triton", 0)))


def strip_prefix(sd):
    """HF stores 'model.<k>' + tied 'model.lm_head.weight'. Strip the prefix."""
    out = {}
    for k, v in sd.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        out[nk] = v
    return out


def load_v03_into_harness(model, st_sd):
    hk = set(model.state_dict().keys())
    sd = strip_prefix(st_sd)
    # Drop the tied output head if the harness ties it internally (no such param).
    dropped = []
    if "lm_head.weight" in sd and "lm_head.weight" not in hk:
        del sd["lm_head.weight"]; dropped.append("lm_head.weight (tied -> dropped)")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    return dropped, list(missing), list(unexpected)


def load_pt_ymode(model, pt_path, a_json):
    ckpt = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if a_json.get("optimizer") == "schedulefree" and "optimizer_state_dict" in ckpt:
        import schedulefree
        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=a_json.get("lr", 3e-4),
            weight_decay=a_json.get("weight_decay", 0.01), betas=(0.9, 0.95))
        opt.load_state_dict(ckpt["optimizer_state_dict"]); opt.train(); del opt
    return ckpt.get("step"), ckpt.get("loss")


def main():
    import tiktoken
    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)}")
    ids, total_bytes = None, None
    # canonical slice via the republish lib's loader (sha-checked)
    sys.path.insert(0, "/home/erikg/ndm/.wg-worktrees/agent-777/scripts")
    import hf_v03_republish_lib as L
    ids, total_bytes = L.load_ids()
    log(f"canonical slice: {total_bytes} bytes, {ids.size(1)} tokens")

    out = {}
    for key, (repo, ckptdir) in REPOS.items():
        log(f"================= {key} ({repo}) =================")
        a_json = json.loads(Path(ckptdir, "args.json").read_text())
        vocab = tiktoken.get_encoding(a_json["tokenizer"]).n_vocab
        st_path = hf_hub_download(repo, "model.safetensors", revision="v0.3")
        # read provenance metadata
        from safetensors import safe_open
        with safe_open(st_path, framework="pt") as f:
            meta = f.metadata() or {}
        st_sd = load_file(st_path)
        log(f"  v0.3 safetensors checkpoint_step={meta.get('checkpoint_step')} ymode={meta.get('ymode_export')}")

        # (A) init the trainable harness FROM v0.3 safetensors (strict)
        m_v03 = build_model(a_json, vocab)
        dropped, missing, unexpected = load_v03_into_harness(m_v03, st_sd)
        log(f"  STRICT load v0.3 -> harness: missing={missing} unexpected={unexpected} dropped={dropped}")
        m_v03 = m_v03.to(device).bfloat16().eval()

        # (B) prior-run init: pinned .pt + y-mode swap
        m_pt = build_model(a_json, vocab)
        step, loss = load_pt_ymode(m_pt, CKPT_PT[key], a_json)
        log(f"  pinned .pt y-mode init: step={step} loss={loss}")
        m_pt = m_pt.bfloat16()

        # (C) tensor-equivalence v0.3-init vs pinned-init (bf16, the training dtype)
        sd_v = {k: v.detach().cpu() for k, v in m_v03.state_dict().items()}
        sd_p = {k: v.detach().cpu() for k, v in m_pt.state_dict().items()}
        assert set(sd_v) == set(sd_p), "key set mismatch"
        max_abs = 0.0; n_exact = 0; worst = None
        for k in sd_v:
            d = (sd_v[k].float() - sd_p[k].float()).abs().max().item()
            if d == 0.0: n_exact += 1
            if d > max_abs: max_abs, worst = d, k
        log(f"  EQUIVALENCE v0.3-init vs pinned-init: max_abs_diff={max_abs:.3e} "
            f"(worst={worst}); exact_match_tensors={n_exact}/{len(sd_v)}")
        del m_pt

        # (D) load sanity: canonical slice forward through v0.3-init harness
        block = float(H.block_loss_sanity(m_v03, ids, 2048, device))
        few = float(L.few_window_nats(m_v03, ids, device, nwin=8))
        log(f"  LOAD-SANITY v0.3-init: block_nats={block:.4f}  few8_window_nats={few:.4f}")

        out[key] = {"repo": repo, "v03_sha_meta_step": meta.get("checkpoint_step"),
                    "ymode_export": meta.get("ymode_export"),
                    "strict_load": {"missing": missing, "unexpected": unexpected, "dropped": dropped},
                    "pinned_pt_step": step, "pinned_pt_loss": loss,
                    "equivalence_max_abs_diff": max_abs, "equivalence_worst_tensor": worst,
                    "equivalence_exact_tensors": n_exact, "n_tensors": len(sd_v),
                    "loadsanity_block_nats": block, "loadsanity_few8_nats": few}
        del m_v03; torch.cuda.empty_cache()
        Path("scripts/v03_init_verify_result.json").write_text(json.dumps(out, indent=2))
    log("wrote scripts/v03_init_verify_result.json")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
