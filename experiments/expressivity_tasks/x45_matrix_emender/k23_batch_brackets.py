"""K23 -- THE STEP-LEVEL SPEED PROGRAM: the SCRATCH BATCHING/GRAPH levers.

Zero training.  This file is the K23 lane's named scratch space: every lever
below re-schedules or re-hosts the COMMITTED composed scan's kernels and is
lived in ONLY here -- the committed engine files (``k15_composition.py`` and
everything it imports) are imported verbatim and NEVER edited; the driver
(``k23_step_speed.py``) hash-binds them to prove it.  Nothing in this file
is imported by any committed engine or training machine.

THE QUESTION (the operator's): the composed scan is LATENCY-bound -- one
launch exposes batch*heads = 16 independent rows on a 142-SM card (K21's
receipt), so most of the card idles while each row's serial token walk runs.
INTERLEAVING/BATCHING the window-groups of a layer into one launch is pure
RE-SCHEDULING of INDEPENDENT rows: per-output-element arithmetic order is
preserved by construction (each program's tile arithmetic is untouched; the
grid only decides WHICH SM runs which row).  THE RECEIPTS BAR is therefore
empirical, as always: a lever counts only if every output channel of the
committed opaque op pair is ``torch.equal`` per group, on captured REAL
activations; the Triton lowering's layout machinery is warp-dependent (the
K21 4/16-warp precedent), so the batched grid is CHECKED, never assumed.

THE INSTRUMENTS (all on layer 6's EXACT production scan arguments, captured
with a recording shim on the committed mixer's own ``composed_scan`` binding,
one FULL probe step's 8 window-groups):

1. ``forward batching``   the committed ``composed_scan`` / ``_k15_scan_fwd_op``
   called on the CONCATENATION of the 8 window-groups' scan arguments
   (batch 8 -> 64, grid 16 -> 128 programs), sliced back per group and
   compared channel by channel against each group's own committed outputs.
   Brackets {1, 2, 4, 8} groups expose {16, 32, 64, 128} rows.

2. ``the K21 stack under the batched grid``   the K21 scratch kernel
   (``k21_micro_brackets.scratch_forward_launch``) with the three
   dataflow-identical levers (lazy_norms + lazy_old_row + reduce_nr) at the
   receipted (8 warps, 1 stage) and the committed-stage bracket, on the
   batch-64 concatenation -- the patch and the batching may interact, and
   the composed receipt must hold for the combination the lane recommends.

3. ``backward batching``  the committed ``_k15_scan_bwd_op`` (the whole
   adjoint: segment replay + prep + both walks) on the concatenated groups,
   its 7 gradient tensors sliced per group and compared against each group's
   own committed backward.  The backward's kernel set launches with
   rows = batch*heads too, so the same re-scheduling question is asked and
   answered empirically.

4. ``cuda-graph capture``  the committed kernels replayed from a CUDA graph
   (same kernels, same arguments, same order -- bit-identity by construction,
   still checked): the launch-overhead share of the scan's per-call walls,
   measured as graph-replay wall vs eager wall for both the per-group
   committed launch train and the batched one.

Usage: driven by ``k23_step_speed.py`` (the lane driver); never runs alone.
"""
from __future__ import annotations

import statistics
import time

import torch

from .k15_composition import _k15_scan_bwd_op, _k15_scan_fwd_op
from .k21_micro_brackets import OUTPUT_KEYS, bit_identity, time_launch

# The 13 output channels' BATCH axis: ckpt_* tensors are laid out
# (nseg, batch, heads, ...) -- every other channel leads with batch.
BATCH_DIM = {name: 0 for name in OUTPUT_KEYS}
for _name in ("ckpt_state", "ckpt_live", "ckpt_keys"):
    BATCH_DIM[_name] = 1

GRAD_KEYS = ("dpayload", "dwrite_keys", "dquery_keys", "ddecay", "dgates",
             "dstate", "dkeys")


# ------------------------------------------------------------- the batching


def cat_scan_args(groups):
    """Concatenate the window-groups' five scan-argument tensors along the
    batch axis (pure re-scheduling: each group's rows are untouched)."""
    return tuple(torch.cat([list(g)[i] for g in groups], dim=0).contiguous()
                 for i in range(5))


def zeros_store(payload_cat, rank, key_dim):
    """The committed wrapper's own zero store for the concatenated batch."""
    batch, _, heads, value_dim = payload_cat.shape
    device = payload_cat.device
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device=device)
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device=device)
    return state, live, keys


def batched_fwd_reference(scan_args_cat, state, live, keys, policy_f32,
                          rank, seg, gated):
    """The committed opaque op pair on the concatenated batch."""
    out = _k15_scan_fwd_op(*scan_args_cat, state, live, keys, int(rank),
                           int(seg), *policy_f32, bool(gated))
    return dict(zip(OUTPUT_KEYS, out))


def slice_outputs(batched: dict, group: int, group_batch: int) -> dict:
    """Slice the batched scan's outputs back to one window-group's own
    layout (the committed per-group call's own shapes)."""
    out = {}
    for name, tensor in batched.items():
        dim = BATCH_DIM[name]
        lo, hi = group * group_batch, (group + 1) * group_batch
        out[name] = tensor.narrow(dim, lo, group_batch)
    return out


def per_group_identity(per_group_reference: list, batched: dict,
                       n_groups: int, group_batch: int) -> dict:
    """The receipts bar under batching: every channel, every group."""
    reports = []
    all_equal = True
    for g in range(n_groups):
        report = bit_identity(per_group_reference[g],
                              slice_outputs(batched, g, group_batch))
        reports.append(dict(
            group=g, bit_identical=report["bit_identical"],
            failed_channels=report["channels"]))
        all_equal &= report["bit_identical"]
    return dict(bit_identical=all_equal, per_group=reports)


# ------------------------------------------------------------- the backward


def scan_bwd_call(scan_args, fwd_out, dreads, rank, seg, scale, gated):
    """The committed whole-window adjoint op on one call's own forward
    outputs (decisions as recorded constants -- the standing doctrine)."""
    return _k15_scan_bwd_op(
        scan_args[0], scan_args[1], scan_args[2], scan_args[3], scan_args[4],
        dreads, fwd_out["slot"], fwd_out["op"], fwd_out["erase"],
        fwd_out["live_after"], fwd_out["payload"], fwd_out["live_pre"],
        fwd_out["ckpt_state"], fwd_out["ckpt_live"], fwd_out["ckpt_keys"],
        int(rank), int(seg), float(scale), bool(gated))


def slice_grads(grads, group: int, group_batch: int):
    """The 7 gradient tensors' per-group slices (all lead with batch)."""
    return tuple(tensor.narrow(0, group * group_batch, group_batch)
                 for tensor in grads)


def grads_bit_identical(ref_grads, got_grads) -> dict:
    """``torch.equal`` per gradient tensor -- the backward's receipts bar."""
    fields = {}
    all_equal = True
    for name, ref, got in zip(GRAD_KEYS, ref_grads, got_grads):
        equal = bool(torch.equal(ref, got))
        diff = (float((ref.float() - got.float()).abs().max())
                if ref.numel() else 0.0)
        fields[name] = dict(bit_identical=equal, max_abs_diff=diff)
        all_equal &= equal
    return dict(bit_identical=all_equal, channels=fields)


# -------------------------------------------------------- the graph brackets


def capture_graph(build, warmup: int = 3):
    """Capture ``build()``'s kernel train into a CUDA graph on the current
    device (the standard warmup-then-capture pattern; every allocation the
    train makes lands in the graph's private pool).  ``build`` must launch
    only capture-safe work (the committed Triton kernels + eager allocations
    are; the data pipeline and the optimizer are not)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            build()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = build()
    torch.cuda.synchronize()
    return graph, outputs


def replay_median(graph, iters: int = 10) -> list[float]:
    """CUDA-synced per-replay wall of a captured graph (ms)."""
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        started = time.monotonic()
        graph.replay()
        torch.cuda.synchronize()
        samples.append((time.monotonic() - started) * 1000.0)
    return samples


def host_issue_ms(fn, warmup: int = 3) -> float:
    """The HOST-SIDE issue cost of one call (no sync inside the timed
    window): if this equals the synced wall the call is host/launch-bound;
    if it is far below, the device owns the wall."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    started = time.monotonic()
    fn()
    issue = (time.monotonic() - started) * 1000.0
    torch.cuda.synchronize()
    return issue


def wall_median(fn, warmup: int = 3, iters: int = 10) -> float:
    """The synced per-call wall (ms), median not mean."""
    return statistics.median(time_launch(fn, warmup, iters))
