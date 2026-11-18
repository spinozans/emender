"""GDN2+growth feasibility smoke (PREREGISTRATION_GDN2_GROWTH.md, section 6).

The registered S-gates, on one registered-shape GDN2 block
(``GDN2ExternalMLPLayer(dim=576, expansion=2., head_dim=128, use_conv=True)``
-- the registered 150M arm's mixer shape, 2 heads):

S-1  inertness: the grown module (H=4, born per the registration, o_proj's
     new input columns zeroed) vs the H=2 module on identical inputs, fp32
     and bf16 autocast, training-mode forward -- plus the fully-alive birth
     (no o_proj zeroing) receipted as the honest contrast.
S-2  segmentation parity: 64-token segment-carry vs the whole-window chunk
     pass (the kernel's own chunk_size is 64, so the split aligns); and a
     mid-window growth continuation (segments at H=2, then the grown module
     with the zero-padded state) vs the un-grown continuation.
S-3  trains through the event: 3 steps at H=2, the registrar event, steps
     after -- finite loss/grads, the ONE-STEP WAKE-UP receipt (nonzero grad
     on o_proj's new columns at the first post-event step; the new head's
     internal parameters still silent at that step, training from the
     second), optimizer-state carry receipted on shared slices, and the
     carry-gradient legality (grads through a chunk call's initial_state vs
     the whole-window path).
S-4  kernel/shape tolerance: forward+backward at H in {2, 4, 8} at the
     registered window shape (T=2048), walls and finite grads receipted.
S-5  statistic anchors: the participation ratio PR and the delta
     self-interference co-diagnostic iota from the real kernel's own
     per-chunk state trajectory (return_intermediate_states=True) at init.

GDN2's code is READ-ONLY throughout: the registrar builds fresh modules and
copies slices; the kernel is called through its own public entry points.
Any gate that cannot be met is an impossibility receipt, not a patch.

Usage (cards 2/3, one leased card):

    eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
    OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_smoke \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/gdn2_growth_smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

from ndm.models import external_gdn2
from ndm.models.external_gdn2 import GDN2ExternalMLPLayer

external_gdn2._load_gdn2_class()
chunk_gdn2 = external_gdn2._GDN2_OPS_MODULE.chunk_gdn2

ROOT = Path(__file__).resolve().parent
DIM, HEAD_DIM, EXPANSION, D_CONV = 576, 128, 2.0, 4
HEAD_V_DIM = int(HEAD_DIM * EXPANSION)


def build_layer(heads: int, seed: int) -> GDN2ExternalMLPLayer:
    torch.manual_seed(seed)
    return GDN2ExternalMLPLayer(
        dim=DIM, expansion=EXPANSION, head_dim=HEAD_DIM, num_heads=heads,
        use_conv=True, d_conv=D_CONV, gdn2_mlp_ratio=2.0,
    ).cuda()


def mixer(layer: GDN2ExternalMLPLayer):
    """The external GatedDeltaNet2 module (layer.gdn2 is our ExternalLayer)."""
    return layer.gdn2.gdn2


# --------------------------------------------------------------------------
# The registrar (the registration's section 3 birth protocol)
# --------------------------------------------------------------------------

# head-count-dependent parameters, by their suffix in named_parameters()
KEY_ROW = ("gdn2.q_proj.weight", "gdn2.k_proj.weight", "gdn2.b_proj.weight",
           "gdn2.f_proj.1.weight")
VAL_ROW = ("gdn2.v_proj.weight", "gdn2.w_proj.weight",
           "gdn2.g_proj.1.weight", "gdn2.g_proj.1.bias")
CONV_KEY = ("gdn2.q_conv1d.weight", "gdn2.k_conv1d.weight")
CONV_VAL = ("gdn2.v_conv1d.weight",)
O_PROJ = "gdn2.o_proj.weight"  # new INPUT columns (dim 1) zeroed at the event


def _prefix_slices(old_shape, new_shape):
    """Per-dimension slice covering the shared prefix."""
    return tuple(slice(0, o) if o != n else slice(None)
                 for o, n in zip(old_shape, new_shape))


def grow_layer(old: GDN2ExternalMLPLayer, heads_new: int, seed: int,
               wake_up: bool = True) -> GDN2ExternalMLPLayer:
    """Born at GDN2's own init; shared slices copied; o_proj's new input
    columns zeroed iff ``wake_up`` (the registration's event-time output
    invariance + one-step wake-up protocol)."""
    old_heads = mixer(old).num_heads
    assert heads_new > old_heads
    new = build_layer(heads_new, seed)
    old_key, new_key = old_heads * HEAD_DIM, heads_new * HEAD_DIM
    old_val, new_val = old_heads * HEAD_V_DIM, heads_new * HEAD_V_DIM
    rules = {}
    for names, dim in [(KEY_ROW + CONV_KEY + ("gdn2.A_log", "gdn2.dt_bias"), 0),
                      (VAL_ROW + CONV_VAL, 0)]:
        for n in names:
            rules[n] = dim
    for (name, p_old), (_, p_new) in zip(old.named_parameters(),
                                         new.named_parameters()):
        suffix = name.split(".", 1)[1] if "." in name else name
        with torch.no_grad():
            if suffix == O_PROJ:
                p_new[:, :old_val] = p_old
                if wake_up:
                    p_new[:, old_val:new_val] = 0.0
            elif suffix in KEY_ROW:
                p_new[:old_key] = p_old
            elif suffix in VAL_ROW:
                p_new[:old_val] = p_old
            elif suffix in CONV_KEY:
                p_new[:old_key] = p_old
            elif suffix in CONV_VAL:
                p_new[:old_val] = p_old
            elif suffix in ("gdn2.A_log", "gdn2.dt_bias"):
                sl = _prefix_slices(p_old.shape, p_new.shape)
                p_new[sl] = p_old
            else:
                # head-count-independent (f_proj.0, g_proj.0, the MLP, norm_2):
                # copied WHOLE -- leaving any of these at fresh init would
                # break event-time invariance.
                assert p_old.shape == p_new.shape, name
                p_new.copy_(p_old)
    return new


def carry_optimizer_state(opt_old, opt_new, old: GDN2ExternalMLPLayer,
                          new: GDN2ExternalMLPLayer):
    """AdamW surgery: slice-carry exp_avg / exp_avg_sq / step on shared
    parameters by position (both optimizers iterate model.parameters());
    fresh (zero) state on the new regions."""
    old_params = dict(old.named_parameters())
    new_params = dict(new.named_parameters())
    carried = []
    for group_new, group_old in zip(opt_new.param_groups, opt_old.param_groups):
        assert len(group_new["params"]) == len(group_old["params"])
        for p_new, p_old in zip(group_new["params"], group_old["params"]):
            name_old = [k for k, v in old_params.items() if v is p_old][0]
            name_new = [k for k, v in new_params.items() if v is p_new][0]
            assert name_old == name_new, (name_old, name_new)
            st_old = opt_old.state[p_old]
            if not st_old:
                continue
            st_new = opt_new.state[p_new]
            st_new["step"] = st_old.get("step", torch.tensor(0))
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st_old:
                    buf_old = st_old[key]
                    buf_new = st_new.setdefault(key, torch.zeros_like(p_new))
                    buf_new[_prefix_slices(buf_old.shape, buf_new.shape)] = buf_old
            carried.append(name_new)
    return carried


# --------------------------------------------------------------------------
# The kernel path, replicated non-invasively for the statistics gates
# --------------------------------------------------------------------------

def _kernel_inputs(core, x):
    """Replicate GatedDeltaNet2.forward's pre-kernel projections (read-only:
    calls the module's own submodules; GDN2's code is never modified)."""
    q, _ = core.q_conv1d(x=core.q_proj(x), cache=None, output_final_state=False)
    k, _ = core.k_conv1d(x=core.k_proj(x), cache=None, output_final_state=False)
    v, _ = core.v_conv1d(x=core.v_proj(x), cache=None, output_final_state=False)
    g = (-core.A_log.float().exp().repeat_interleave(core.head_k_dim)
         * F.softplus(core.f_proj(x).float() + core.dt_bias))
    b = core.b_proj(x).sigmoid()
    w = core.w_proj(x).sigmoid()
    q, k, g = (t.view(*t.shape[:-1], core.num_heads, core.head_k_dim)
               for t in (q, k, g))
    v = v.view(*v.shape[:-1], core.num_heads, core.head_v_dim)
    b = b.view(*b.shape[:-1], core.num_heads, core.head_k_dim)
    w = w.view(*w.shape[:-1], core.num_heads, core.head_v_dim)
    return q, k, v, g, b, w


def _chunk_call(core, q, k, v, g, b, w, initial_state=None,
                output_final_state=False, return_intermediate_states=False):
    """Uniform (o, final_state, h) return: the public chunk_gdn2 wrapper
    returns (o, final_state), or (o, final_state, h) when intermediate
    states are requested (inference mode only)."""
    out = chunk_gdn2(
        q=q, k=k, v=v, g=g, b=b, w=w,
        A_log=core.A_log, dt_bias=core.dt_bias,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
        return_intermediate_states=return_intermediate_states,
    )
    if return_intermediate_states:
        o, s, h = out
        return o, s, h
    return out[0], out[1], None


# --------------------------------------------------------------------------
# The gates
# --------------------------------------------------------------------------

def gate_s1_inertness(receipt):
    x = torch.randn(2, 512, DIM, device="cuda")
    old = build_layer(2, 3401).eval()
    grown = grow_layer(old, 4, 3411, wake_up=True).eval()
    alive = grow_layer(old, 4, 3411, wake_up=False).eval()
    with torch.no_grad():
        o_old = old(x)[0]
        o_grown = grown(x)[0]
        o_alive = alive(x)[0]
    old_n = sum(p.numel() for p in old.parameters())
    new_n = sum(p.numel() for p in grown.parameters())
    receipt["s1_inertness"] = {
        "fp32_max_abs_diff_wake_up": float((o_old - o_grown).abs().max()),
        "fp32_bit_identical": bool(torch.equal(o_old, o_grown)),
        "fp32_max_abs_diff_fully_alive_birth": float((o_old - o_alive).abs().max()),
        "o_old_abs_max": float(o_old.abs().max()),
        "parameters": {"old": old_n, "grown": new_n, "delta": new_n - old_n,
                       "note": "growth adds trainable parameters (the K8 "
                               "zero-parameter property does not transfer: "
                               "GDN2's keys are learned)"},
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ob_old = old(x)[0].float()
        ob_grown = grown(x)[0].float()
    receipt["s1_inertness"]["bf16_max_abs_diff_wake_up"] = float(
        (ob_old - ob_grown).abs().max())
    d = receipt["s1_inertness"]
    ok = d["fp32_bit_identical"] or d["fp32_max_abs_diff_wake_up"] <= 1e-5
    d["verdict"] = "met" if ok else "NOT MET"
    return ok


def _readout(core, o_kernel, x):
    """Replicate GatedDeltaNet2.forward's post-kernel readout (the o_norm /
    g_proj / o_proj tail) on a kernel-level output [B, T, H, head_v_dim]."""
    gate = core.g_proj(x).view(*x.shape[:-1], core.num_heads, core.head_v_dim)
    o = core.o_norm(o_kernel, gate)
    o = o.reshape(*o.shape[:-2], -1)
    return core.o_proj(o)


def gate_s2_segmentation(receipt):
    T = 2048
    x = torch.randn(2, T, DIM, device="cuda")
    old = build_layer(2, 3402).eval()
    grown = grow_layer(old, 4, 3412, wake_up=True).eval()
    core, gcore = mixer(old), mixer(grown)
    old_heads = core.num_heads
    seg = 64
    with torch.inference_mode():
        q, k, v, g, b, w = _kernel_inputs(core, x)
        o_whole, s_whole, h = _chunk_call(
            core, q, k, v, g, b, w, output_final_state=True,
            return_intermediate_states=True)
        assert h is not None and h.ndim == 5 and h.shape[1] == T // seg, \
            tuple(h.shape)  # h layout: [B, num_chunks, H, K, V]
        diffs, state_diffs = [], []
        state = None
        for c, t0 in enumerate(range(0, T, seg)):
            sl = slice(t0, t0 + seg)
            o_seg, state, _ = _chunk_call(
                core, q[:, sl].contiguous(), k[:, sl].contiguous(),
                v[:, sl].contiguous(), g[:, sl].contiguous(),
                b[:, sl].contiguous(), w[:, sl].contiguous(),
                initial_state=state, output_final_state=True)
            diffs.append(float((o_seg - o_whole[:, sl]).abs().max()))
            # h[:, c+1] is the state AFTER chunk c (h[:, 0] is the empty
            # initial state); the last segment's exit state is s_whole itself
            ref = h[:, c + 1] if c + 1 < h.shape[1] else s_whole
            state_diffs.append(float((state - ref).abs().max()))
        # mid-window growth: first half at H=2, second half on the grown module
        fire = T // 2
        state2 = None
        o_pre = []
        for t0 in range(0, fire, seg):
            sl = slice(t0, t0 + seg)
            o_seg, state2, _ = _chunk_call(
                core, q[:, sl].contiguous(), k[:, sl].contiguous(),
                v[:, sl].contiguous(), g[:, sl].contiguous(),
                b[:, sl].contiguous(), w[:, sl].contiguous(),
                initial_state=state2, output_final_state=True)
            o_pre.append(o_seg)
        o_pre_kernel = torch.cat(o_pre, dim=1)
        gq, gk, gv, gg, gb, gw = _kernel_inputs(gcore, x)
        proj_diff = max(
            float((gq[:, :, :old_heads] - q).abs().max()),
            float((gk[:, :, :old_heads] - k).abs().max()),
            float((gv[:, :, :old_heads] - v).abs().max()),
            float((gb[:, :, :old_heads] - b).abs().max()),
            float((gw[:, :, :old_heads] - w).abs().max()),
            float((gg[:, :, :old_heads] - g).abs().max()))
        padded = torch.zeros(2, gcore.num_heads, HEAD_DIM, HEAD_V_DIM,
                             dtype=state2.dtype, device="cuda")
        padded[:, :old_heads] = state2
        o_post = []
        for t0 in range(fire, T, seg):
            sl = slice(t0, t0 + seg)
            o_seg, grown_state, _ = _chunk_call(
                gcore, gq[:, sl].contiguous(), gk[:, sl].contiguous(),
                gv[:, sl].contiguous(), gg[:, sl].contiguous(),
                gb[:, sl].contiguous(), gw[:, sl].contiguous(),
                initial_state=padded, output_final_state=True)
            o_post.append(o_seg)
            padded = grown_state
        o_post_kernel = torch.cat(o_post, dim=1)
        # the mid-window event is compared at the LAYER-READOUT level (the
        # kernel's o carries the head axis, whose size IS the growth event):
        # each half is read out through its own module's tail, and the grown
        # module's o_proj zero columns must erase the new heads' contribution
        layer_walk = torch.cat([
            _readout(core, o_pre_kernel, x[:, :fire]),
            _readout(gcore, o_post_kernel, x[:, fire:]),
        ], dim=1)
        mid_diff = float((layer_walk - _readout(core, o_whole, x)).abs().max())
        old_state_diff = float(
            (grown_state[:, :old_heads] - s_whole).abs().max())
        new_state_norm = float(grown_state[:, old_heads:].norm())
    receipt["s2_segmentation"] = {
        "segment_len": seg, "T": T,
        "max_abs_output_diff_whole_vs_segments": max(diffs),
        "segments_bit_identical": max(diffs) == 0.0,
        "max_abs_state_diff_vs_kernel_trajectory": max(state_diffs),
        "mid_window_growth": {
            "fire_boundary": fire,
            "max_abs_output_diff_full_walk": mid_diff,
            "old_head_projection_diff_at_event": proj_diff,
            "old_head_final_state_diff": old_state_diff,
            "new_head_final_state_frobenius_norm": new_state_norm,
        },
        "h_shape": list(h.shape),
    }
    d = receipt["s2_segmentation"]
    ok = (max(diffs) <= 1e-5 and mid_diff <= 1e-5 and new_state_norm > 0.0)
    d["verdict"] = "met" if ok else "NOT MET"
    return ok


def _carry_grad_check(receipt):
    """The mid-window carry must be differentiable: gradients through a
    segmented (initial_state-carried) path vs the whole-window path."""
    T, seg = 128, 64
    x = torch.randn(2, T, DIM, device="cuda")
    core = mixer(build_layer(2, 3403))
    q, k, v, g, b, w = _kernel_inputs(core, x)
    g = g.detach()  # a fixed input in this check; sharing its projection
    # subgraph across the two backward passes would free it on the first
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, b, w)]
    qs, ks, vs, bs, ws = leaves
    o_whole, _, _ = _chunk_call(core, qs, ks, vs, g, bs, ws)
    o_whole.pow(2).sum().backward()
    g_whole = [t.grad.clone() for t in leaves]

    leaves2 = [t.detach().clone().requires_grad_(True) for t in (q, k, v, b, w)]
    qs2, ks2, vs2, bs2, ws2 = leaves2
    o1, s1, _ = _chunk_call(core, qs2[:, :seg].contiguous(),
                         ks2[:, :seg].contiguous(), vs2[:, :seg].contiguous(),
                         g[:, :seg].contiguous(), bs2[:, :seg].contiguous(),
                         ws2[:, :seg].contiguous(), output_final_state=True)
    o2, _, _ = _chunk_call(core, qs2[:, seg:].contiguous(),
                        ks2[:, seg:].contiguous(), vs2[:, seg:].contiguous(),
                        g[:, seg:].contiguous(), bs2[:, seg:].contiguous(),
                        ws2[:, seg:].contiguous(), initial_state=s1,
                        output_final_state=True)
    torch.cat([o1, o2], dim=1).pow(2).sum().backward()
    rel = []
    for t2, g1 in zip(leaves2, g_whole):
        denom = g1.abs().max().clamp(min=1e-12)
        rel.append(float(((t2.grad - g1).abs().max() / denom)))
    receipt["s3_train"]["carry_grad_check"] = {
        "relative_max_grad_diffs_q_k_v_b_w": rel,
        "max_relative": max(rel),
        "note": "whole-window vs 64-token-segment carried path, fp32; the "
                "kernel's backward returns dh0 so the carry is differentiable",
    }
    return max(rel) <= 5e-2


def gate_s3_train_through_event(receipt):
    receipt["s3_train"] = {}
    carry_ok = _carry_grad_check(receipt)

    B, T = 4, 512
    old = build_layer(2, 3404).train()
    x = torch.randn(B, T, DIM, device="cuda")
    target = torch.randn(B, T, DIM, device="cuda")
    opt_old = torch.optim.AdamW(old.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        loss = F.mse_loss(old(x)[0], target)
        opt_old.zero_grad()
        loss.backward()
        opt_old.step()
        losses.append(float(loss.detach()))

    grown = grow_layer(old, 4, 3414, wake_up=True).train()
    opt_new = torch.optim.AdamW(grown.parameters(), lr=1e-3)
    carry_optimizer_state(opt_old, opt_new, old, grown)
    params_old = dict(old.named_parameters())
    params_new = dict(grown.named_parameters())
    old_heads = mixer(old).num_heads
    key, val = old_heads * HEAD_DIM, old_heads * HEAD_V_DIM
    # the optimizer-state carry receipt: shared slices equal, new regions zero
    st_old = opt_old.state[params_old["gdn2.gdn2.q_proj.weight"]]
    st_new = opt_new.state[params_new["gdn2.gdn2.q_proj.weight"]]
    carried_diff = float((st_new["exp_avg"][:key] - st_old["exp_avg"]).abs().max())
    fresh_norm = float(st_new["exp_avg"][key:].abs().max())

    wake = {}
    for step in range(2):
        loss = F.mse_loss(grown(x)[0], target)
        opt_new.zero_grad()
        loss.backward()
        oproj = params_new["gdn2.gdn2.o_proj.weight"]
        kproj = params_new["gdn2.gdn2.k_proj.weight"]
        if step == 0:
            wake["first_post_event_step"] = {
                "loss": float(loss.detach()),
                "o_proj_new_column_grad_abs_max":
                    float(oproj.grad[:, val:].abs().max()),
                "o_proj_old_column_grad_abs_max":
                    float(oproj.grad[:, :val].abs().max()),
                "new_head_k_proj_grad_abs_max":
                    float(kproj.grad[key:].abs().max()),
            }
        opt_new.step()
        losses.append(float(loss.detach()))
    loss = F.mse_loss(grown(x)[0], target)
    opt_new.zero_grad()
    loss.backward()
    wake["second_post_event_step"] = {
        "loss": float(loss.detach()),
        "new_head_k_proj_grad_abs_max":
            float(params_new["gdn2.gdn2.k_proj.weight"].grad[key:].abs().max()),
        "o_proj_new_column_weight_abs_max":
            float(params_new["gdn2.gdn2.o_proj.weight"][:, val:].detach().abs().max()),
    }
    grads_finite = all(torch.isfinite(p.grad).all().item()
                       for p in grown.parameters() if p.grad is not None)
    receipt["s3_train"].update({
        "losses": losses,
        "grads_finite": grads_finite,
        "optimizer_carry": {
            "shared_slice_exp_avg_max_diff": carried_diff,
            "new_region_exp_avg_abs_max": fresh_norm,
        },
        "wake_up": wake,
    })
    first = wake["first_post_event_step"]
    ok = (grads_finite and all(math.isfinite(v) for v in losses)
          and first["o_proj_new_column_grad_abs_max"] > 0.0
          and first["new_head_k_proj_grad_abs_max"] == 0.0
          and wake["second_post_event_step"]["new_head_k_proj_grad_abs_max"] > 0.0
          and carried_diff == 0.0 and carry_ok)
    receipt["s3_train"]["verdict"] = "met" if ok else "NOT MET"
    return ok


def gate_s4_kernel_tolerance(receipt):
    receipt["s4_kernel_tolerance"] = {}
    ok = True
    for heads in (2, 4, 8):
        layer = build_layer(heads, 3405).train()
        x = torch.randn(2, 2048, DIM, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        loss = layer(x)[0].pow(2).sum()
        loss.backward()
        torch.cuda.synchronize()
        wall = time.time() - t0
        finite = all(torch.isfinite(p.grad).all().item()
                     for p in layer.parameters() if p.grad is not None)
        peak = torch.cuda.max_memory_allocated() / (1 << 30)
        receipt["s4_kernel_tolerance"][f"heads_{heads}"] = {
            "forward_backward_wall_s": round(wall, 3),
            "grads_finite": finite,
            "peak_allocated_gb": round(peak, 2),
        }
        ok = ok and finite
        del layer, x, loss
        torch.cuda.empty_cache()
    receipt["s4_kernel_tolerance"]["verdict"] = "met" if ok else "NOT MET"
    return ok


def gate_s5_statistics(receipt):
    T = 2048
    x = torch.randn(2, T, DIM, device="cuda")
    core = mixer(build_layer(2, 3406).eval())
    pr_all, iota_all = [], []
    with torch.inference_mode():
        q, k, v, g, b, w = _kernel_inputs(core, x)
        o, s_final, h = _chunk_call(
            core, q, k, v, g, b, w, output_final_state=True,
            return_intermediate_states=True)
        assert h.ndim == 5 and h.shape[1] == T // 64, tuple(h.shape)
        # the state AFTER 64-token boundary c is h[:, c+1]; h[:, 0] is the
        # empty initial state (PR undefined there -- excluded, receipted)
        for c in range(h.shape[1] - 1):
            sv = torch.linalg.svdvals(h[:, c + 1].float())
            pr = sv.sum(dim=-1) ** 2 / (sv ** 2).sum(dim=-1)  # [B, H]
            pr_all.extend(pr.reshape(-1).tolist())
        # the co-diagnostic: write collision against the segment-entry state
        khat = F.normalize(k.reshape(-1, core.num_heads, HEAD_DIM).float(),
                           dim=-1).reshape(k.shape)
        for c in range(h.shape[1]):
            s_entry = (h[:, c].float() if c > 0 else torch.zeros(
                h.shape[0], core.num_heads, HEAD_DIM, HEAD_V_DIM,
                device="cuda"))
            t0, t1 = c * 64, (c + 1) * 64
            write_dir = b[:, t0:t1].float() * khat[:, t0:t1].float()
            num = torch.linalg.norm(
                torch.einsum("bthk,bhkv->bthv", write_dir, s_entry), dim=-1)
            den = (torch.linalg.norm(s_entry, dim=(-2, -1))[:, None]
                   * torch.linalg.norm(write_dir, dim=-1)).clamp(min=1e-12)
            iota_all.extend((num / den).reshape(-1).tolist())
    receipt["s5_statistics"] = {
        "shape": {"batch": 2, "T": T, "heads": core.num_heads,
                  "d_k": HEAD_DIM, "d_v": HEAD_V_DIM,
                  "boundaries": T // 64},
        "participation_ratio": {
            "definition": "PR(S) = (sum sigma)^2 / sum sigma^2 per "
                          "(head, 64-token boundary state = h[:, c+1], "
                          "the state AFTER chunk c; the empty initial "
                          "state h[:, 0] is excluded), d_k=128; "
                          "tau_sat = 0.75*d_k = 96",
            "first_boundary_mean": sum(pr_all[:4]) / 4,
            "last_boundary_mean": sum(pr_all[-4:]) / 4,
            "mean": sum(pr_all) / len(pr_all),
            "min": min(pr_all), "max": max(pr_all),
        },
        "delta_self_interference": {
            "definition": "iota = mean_t ||S_entry^T (b.k)|| / "
                          "(||S_entry||_F ||b.k||) per (head, segment); "
                          "the entry state of segment c is h[:, c] "
                          "(the state after chunk c-1; the first segment's "
                          "entry is the empty state, clamped denominator); "
                          "NON-GATING co-diagnostic",
            "mean": sum(iota_all) / len(iota_all),
            "min": min(iota_all), "max": max(iota_all),
        },
        "verdict": "receipted (anchors, no pass/fail)",
    }
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "receipts/gdn2_growth_smoke.json")
    parser.add_argument("--seed", type=int, default=3401)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "the GDN2 chunk kernels are CUDA-only"
    torch.backends.cuda.matmul.allow_tf32 = False
    receipt = {
        "probe": "gdn2_growth_smoke",
        "registration": "PREREGISTRATION_GDN2_GROWTH.md (draft, section 6)",
        "config": {
            "layer": "GDN2ExternalMLPLayer(dim=576, expansion=2., "
                     "head_dim=128, use_conv=True) -- the registered 150M "
                     "arm's mixer shape",
            "axis": "heads (supervisor-decided 2026-11-22); snap {2,4,8}",
            "seed": args.seed,
        },
        "environment": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
        },
    }
    gates = {}
    gates["S-1 inertness"] = gate_s1_inertness(receipt)
    gates["S-2 segmentation"] = gate_s2_segmentation(receipt)
    gates["S-3 train-through-event"] = gate_s3_train_through_event(receipt)
    gates["S-4 kernel tolerance"] = gate_s4_kernel_tolerance(receipt)
    gates["S-5 statistic anchors"] = gate_s5_statistics(receipt)
    receipt["gates"] = gates
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: ("met" if v else "NOT MET")
                      for k, v in gates.items()}, indent=1))
    failed = [k for k, v in gates.items() if not v]
    if failed:
        raise SystemExit(f"SMOKE GATES NOT MET: {failed}")
    print(f"all smoke gates met; receipt: {args.out}")


if __name__ == "__main__":
    main()
