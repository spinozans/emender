#!/usr/bin/env python3
"""Rigorous kernel-invocation confirmation for the gdn2-mlp reference.

task: re-run-offline.

gdn2-mlp does not use the E88 eager path; its recurrence is the external
GatedDeltaNet-2 fused Triton kernel `chunk_gdn2` (or `fused_recurrent_gdn2` for
<=64-token decode), dispatched in lit_gpt.gdn2.GatedDeltaNet2.forward. The
offline scorer feeds a full T=2048 sequence with not-training, so the dispatch
selects the CHUNK kernel. This proves "fused" by counting ACTUAL kernel-function
invocations (NOT the use_triton/mode config flag): we wrap chunk_gdn2 /
fused_recurrent_gdn2 in the live lit_gpt.gdn2 module namespace and run one real
forward over a few held-out chunks, asserting the fused kernel fired and no
eager E88 loop ran.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_checkpoint as ec  # noqa: E402

HELDOUT = Path(
    "/mnt/nvme1n1/erikg/ref_emender_mlp.contaminated_2057/"
    "heldout_pile_tail_p50k_2048_1m.pt"
)
GDN2_DIR = Path(
    "/mnt/nvme1n1/erikg/ref_gdn2_mlp/runs/levelgdn2-mlp_100m_20260615_212627"
)
GDN2_STEP = 125000

COUNT = {"chunk_gdn2": 0, "fused_recurrent_gdn2": 0, "eager_e88": 0}


def main():
    device = torch.device("cuda")
    ckpt_path = sorted(GDN2_DIR.glob(f"checkpoint_step_{GDN2_STEP:06d}_*.pt"))[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ec.checkpoint_args(ckpt_path, checkpoint, None)
    model_args = ec.namespace_from_config(cfg)
    model = ec.build_model(model_args, device)
    ec.load_checkpoint_weights(model, checkpoint, model_args, "train")

    # Patch the REAL kernel functions in the live GatedDeltaNet module namespace
    # (the layer calls the bare names chunk_gdn2 / fused_recurrent_gdn2). The
    # external checkout is loaded under a custom module name, so discover every
    # loaded module that actually holds these symbols and patch each one.
    def make(orig, attr):
        def w(*a, **k):
            COUNT[attr] += 1
            return orig(*a, **k)
        return w

    patched = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        for attr in ("chunk_gdn2", "fused_recurrent_gdn2"):
            fn = getattr(mod, attr, None)
            if callable(fn) and not getattr(fn, "_invocation_wrapped", False):
                wrapped = make(fn, attr)
                wrapped._invocation_wrapped = True
                setattr(mod, attr, wrapped)
                patched.append(f"{mod_name}.{attr}")
    # Eager E88 sentinel (must remain 0 for gdn2).
    try:
        from ndm.models.e88_fla_hybrid import E88FLAHybrid
        orig_eager = E88FLAHybrid._apply_state_activation

        def eager_w(self, pre):
            COUNT["eager_e88"] += 1
            return orig_eager(self, pre)

        E88FLAHybrid._apply_state_activation = eager_w
    except Exception:
        pass

    print(f"[verify_gdn2] patched kernels in lit_gpt.gdn2: {patched}", flush=True)

    scoring = ec.load_scoring_tensor(HELDOUT)
    model.eval()
    # One real forward over the first 8 held-out chunks (T=2048 each) -> chunk dispatch.
    batch = scoring.chunks[:8].to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(batch, return_loss=True)

    fused = COUNT["chunk_gdn2"] + COUNT["fused_recurrent_gdn2"]
    print(
        f"[verify_gdn2] step={GDN2_STEP} kernel_calls={COUNT} "
        f"(chunk dispatch expected: T=2048>64, not-training)",
        flush=True,
    )
    if not (COUNT["chunk_gdn2"] > 0 and COUNT["eager_e88"] == 0):
        raise SystemExit(
            f"[fused-guard] FAIL gdn2: chunk_gdn2={COUNT['chunk_gdn2']} "
            f"eager_e88={COUNT['eager_e88']}; expected chunk_gdn2>0 AND eager==0"
        )
    print(
        f"[verify_gdn2] PASS gdn2 FUSED via kernel-invocation: "
        f"chunk_gdn2={COUNT['chunk_gdn2']} fused_recurrent_gdn2="
        f"{COUNT['fused_recurrent_gdn2']} eager_e88={COUNT['eager_e88']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
