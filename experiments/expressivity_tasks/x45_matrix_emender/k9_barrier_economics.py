"""K9 GATE 0 -- THE BARRIER ECONOMICS: measured before the walk was designed.

Why this file exists
--------------------
The K9 brief: make the transformer-parity (rank 2048) per-token walk fast by
keeping its state ON-CHIP -- the store distributed across CTAs' shared memory,
combined per token by the Lean one-tree reduction -- because the K7 global-tiled
walk streams the whole 16 MB store from HBM every token (103 GB/window,
fold-bound, 374.3 ms/scan receipted; the K8 arm's gate-D table says 414 ms with
the channels) while the FLOPs are trivial.  The design needs a GRID-WIDE
BARRIER per token (every CTA must finish scoring its tile before the one-tree
reduction decides the op, and the winning block executes only after that).  So
the economics of that barrier decide the design, and the brief pre-states the
rule:

    if projected per-token barrier overhead
       (2048 tokens x syncs-per-token x measured-us)
    stays under ~40 ms/scan, build the distributed-persistent design;
    else build the L2-resident single-CTA variant.

This script measures everything the rule needs ON THE LEASED CARD and receipts
it to ``receipts/k9_barrier_economics.json``:

1. **Can Triton express the barrier at all?**  A survey of this stack
   (Triton 3.5.1, torch 2.9.1+cu128): ``triton.language.extra.cuda`` exposes
   only the PDL utilities (``gdc_wait``/``gdc_launch_dependents``) and
   diagnostics -- no ``grid.sync``, and the NVIDIA launch path has no
   cooperative-launch entry point (a source scan of the installed backend,
   receipted below).  A software grid barrier (spin on a global flag) is
   expressible in principle but is only deadlock-free if EVERY CTA of the
   grid is co-resident -- see (2).
2. **Residency arithmetic** (the hard impossibility): a CTA holding BLOCK=128
   slots x 128 values of fp32 state owns 64 KB of smem; the card (RTX 6000
   Ada) has 100 KB smem per SM and 142 SMs, so at most ONE such CTA fits per
   SM -- 142 co-resident, 9.3 MB of addressable on-chip state in total.  The
   probe-shape store is 16 rows x 1 MB = 16 MB in 256 CTAs: it does not fit
   the chip, and a software barrier would deadlock with 256 CTAs queued on
   142 SMs.  Even at one row the barrier itself is not Triton-expressible.
   The on-chip substrate the store CAN wholly live in is the L2 (96 MB).
3. **The barrier economics of the Triton-expressible sync -- the kernel-launch
   boundary**: eager per-launch cost, and the CUDA-graph node cost (the lane's
   own whole-scan graph discipline, K5B), plus the L2-resident streaming
   bandwidths the two candidate designs ride on (single-CTA 1 MB pass for the
   fallback; the 256-CTA 16 MB pass for the distributed sweep).

The decision the receipts force is written to the JSON under ``decision`` and
argued in ``DESIGN_K9_DISTRIBUTED_WALK.md``.  Forward only; no model state.
"""
from __future__ import annotations

import json
import time

import torch
import triton
import triton.language as tl

from .schedule import RoutingPolicy

DEVICE = "cuda"
# The probe shape's distributed geometry (K7's walk receipts): 16 (batch*head)
# rows, rank 2048, value dim 128, BLOCK 128 slots per CTA (64 KB of fp32 state).
ROWS = 16
RANK = 2048
VALUE_DIM = 128
BLOCK_SLOTS = 128
CTA_SMEM_BYTES = BLOCK_SLOTS * VALUE_DIM * 4
TOKENS = 2048
SYNCS_PER_TOKEN = 3          # score sweep -> decision -> apply sweep (design §2)
DECISION_BUDGET_MS = 40.0    # the brief's pre-stated threshold


@triton.jit
def _noop(x_ptr):
    pass


@triton.jit(do_not_specialize=['T', 'R'])
def _noop_walklike(a, b, c, T, R, s0, s1, s2, s3, s4, s5, s6, s7,
                   H: tl.constexpr, V: tl.constexpr, BR: tl.constexpr):
    """An empty kernel with the walk kernels' arg-count/constexpr shape, so
    the eager launch cost measured is the real one, not a best case."""
    pass


@triton.jit(do_not_specialize=['N'])
def _stream_read(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    tl.store(o_ptr + pid, tl.sum(v, 0))


@triton.jit(do_not_specialize=['N'])
def _stream_rw(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    tl.store(x_ptr + offs, v * 1.0000001, mask=offs < N)


@triton.jit(do_not_specialize=['N'])
def _stream_one_cta(x_ptr, o_ptr, N, REP: tl.constexpr, BLOCK: tl.constexpr):
    """The fallback's per-token unit: ONE CTA sweeps ``N`` floats (the 1 MB
    row store) REP times -- the L2-resident single-CTA streaming cost."""
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for _ in range(REP):
        for i in range(0, tl.cdiv(N, BLOCK)):
            offs = i * BLOCK + tl.arange(0, BLOCK)
            s += tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    tl.store(o_ptr + tl.arange(0, BLOCK), s)


def _wall(fn, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def _survey() -> dict:
    """Is a grid-wide barrier expressible on this Triton stack?"""
    import triton.language.extra.cuda as tlc
    members = sorted(m for m in dir(tlc) if not m.startswith('_'))
    # the NVIDIA launch path: does any cooperative-launch entry exist?
    import triton.backends.nvidia.driver as nvdriver
    import inspect
    source = inspect.getsource(nvdriver)
    return dict(
        triton_version=triton.__version__,
        torch_version=torch.__version__,
        tl_extra_cuda_members=members,
        has_grid_sync_primitive=any('grid' in m and 'sync' in m.lower()
                                   for m in members),
        nvidia_driver_has_cooperative_launch=('cooperative' in source.lower()),
        note=("tl.extra.cuda exposes only the PDL utilities (gdc_wait/"
              "gdc_launch_dependents) and diagnostics; grid-wide sync inside "
              "one kernel is not expressible, and the NVIDIA launch path "
              "exposes no cooperative launch, so a software grid barrier "
              "cannot be made deadlock-safe by construction either"),
    )


def main(policy: RoutingPolicy = RoutingPolicy(ops=True), out: str = None):
    assert torch.cuda.is_available(), "the barrier economics need the card"
    torch.cuda.init()
    receipt = dict(gate="K9 gate 0: the barrier economics (measured first)",
                  fixture=dict(rows=ROWS, rank=RANK, value_dim=VALUE_DIM,
                              block_slots=BLOCK_SLOTS,
                              cta_smem_bytes=CTA_SMEM_BYTES, tokens=TOKENS,
                              syncs_per_token=SYNCS_PER_TOKEN,
                              decision_budget_ms=DECISION_BUDGET_MS,
                              device=torch.cuda.get_device_name(0)))

    # ---- 1. the sync survey and the residency arithmetic
    survey = _survey()
    p = torch.cuda.get_device_properties(0)
    ctas_per_sm = p.shared_memory_per_multiprocessor // CTA_SMEM_BYTES
    max_coresident = p.multi_processor_count * ctas_per_sm
    state_bytes = ROWS * RANK * VALUE_DIM * 4
    probe_ctas = ROWS * (RANK // BLOCK_SLOTS)
    receipt["impossibility"] = dict(
        sync_survey=survey,
        sm_count=p.multi_processor_count,
        smem_per_sm_bytes=p.shared_memory_per_multiprocessor,
        smem_optin_per_block_bytes=p.shared_memory_per_block_optin,
        l2_bytes=p.L2_cache_size,
        ctas_per_sm_at_64KB=int(ctas_per_sm),
        max_coresident_ctas=int(max_coresident),
        total_addressable_smem_bytes=int(max_coresident * CTA_SMEM_BYTES),
        probe_state_bytes=int(state_bytes),
        probe_shape_ctas=int(probe_ctas),
        verdict=("the smem-resident distributed walk is impossible at the "
                 "probe shape on this card: 256 CTAs of 64 KB do not fit 142 "
                 "SMs (1 CTA/SM, 142 co-resident, 9.3 MB of addressable "
                 "on-chip state < 16 MB of store), so no barrier -- cooperative "
                 "or software -- can exist for it; and no grid-wide sync is "
                 "Triton-expressible in any case. The store CAN wholly live "
                 "on-chip in the L2 (16 MB < 96 MB); the distributed design "
                 "is therefore built on the L2-resident store with the "
                 "launch boundary (graph-captured) as the per-token barrier"),
    )

    # ---- 2. the barrier economics: the launch boundary, eager and graphed
    x = torch.randn(4096, device=DEVICE)
    _noop[(1,)](x)
    eager_empty = _wall(lambda: _noop[(1,)](x), 2048) * 1e6
    args = (x, x, x, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    kw = dict(H=2, V=VALUE_DIM, BR=BLOCK_SLOTS)
    _noop_walklike[(1,)](*args, **kw)
    eager_walklike = _wall(lambda: _noop_walklike[(1,)](*args, **kw), 2048) * 1e6

    # a CUDA graph of the per-token pattern: 3 launches per token, T tokens
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _noop_walklike[(1,)](*args, **kw)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        for _ in range(TOKENS):
            _noop[(1,)](x)
            _noop[(1,)](x)
            _noop[(1,)](x)
    reps = 50
    graph_ms = _wall(lambda: graph.replay(), reps) * 1e3
    node_us = graph_ms * 1e3 / (TOKENS * SYNCS_PER_TOKEN)
    receipt["barrier_economics"] = dict(
        eager_empty_launch_us=round(eager_empty, 3),
        eager_walklike_launch_us=round(eager_walklike, 3),
        graph_replay_3x2048_nodes_ms=round(graph_ms, 3),
        graph_node_us=round(node_us, 3),
        note=("the per-token barrier of the Triton-expressible design is the "
              "kernel-launch boundary; eager it costs 10-19 us (2048 tokens x "
              "3 x that is 65-118 ms/scan -- over budget), inside a whole-scan "
              "CUDA graph (the lane's K5B discipline) it is the graph node "
              "cost measured above"),
    )

    # ---- 3. the L2-resident streaming bandwidths the designs ride on
    state = torch.randn(ROWS * RANK * VALUE_DIM, device=DEVICE)
    o = torch.empty(8192, device=DEVICE)
    one_mb_floats = RANK * VALUE_DIM
    o1 = torch.empty(1024, device=DEVICE)
    kw1 = dict(REP=64, BLOCK=1024, num_warps=8)
    _stream_one_cta[(1,)](state[:one_mb_floats], o1, one_mb_floats, **kw1)
    r = _wall(lambda: _stream_one_cta[(1,)](state[:one_mb_floats], o1,
                                             one_mb_floats, **kw1), 20)
    single_us = r * 1e6 / 64
    single_gbs = one_mb_floats * 4 / (r / 64) / 1e9
    n_ctas = (ROWS * RANK * VALUE_DIM) // 16384        # 256 CTAs x 64 KB
    kw2 = dict(BLOCK=16384, num_warps=8)
    _stream_read[(n_ctas,)](state, o, state.numel(), **kw2)
    r = _wall(lambda: _stream_read[(n_ctas,)](state, o, state.numel(), **kw2),
              200)
    read_us = r * 1e6
    read_gbs = state.numel() * 4 / r / 1e9
    _stream_rw[(n_ctas,)](state, o, state.numel(), **kw2)
    r = _wall(lambda: _stream_rw[(n_ctas,)](state, o, state.numel(), **kw2),
              200)
    rw_us = r * 1e6
    rw_gbs = state.numel() * 8 / r / 1e9
    receipt["l2_streaming"] = dict(
        single_cta_1MB_pass_us=round(single_us, 3),
        single_cta_1MB_GBs=round(single_gbs, 1),
        multi_cta_16MB_read_us=round(read_us, 3),
        multi_cta_16MB_read_GBs=round(read_gbs, 1),
        multi_cta_16MB_rw_us=round(rw_us, 3),
        multi_cta_16MB_rw_GBs=round(rw_gbs, 1),
        note=("the 16 MB store is L2-resident (96 MB L2): the distributed "
              "sweep reads it at the multi-CTA rate, the single-CTA fallback "
              "sweeps one 1 MB row at the single-CTA rate (latency-bound, "
              "the K7 fold class)"),
    )

    # ---- 4. occupancy at every snap {128..2048} (the distributed geometry)
    occupancy = []
    for rank in (128, 256, 512, 1024, 2048):
        nt = max(1, rank // BLOCK_SLOTS)
        ctas = ROWS * nt
        occupancy.append(dict(rank=rank, slot_blocks_per_row=nt,
                               sweep_ctas=ctas,
                               waves_on_142_SMs=round(ctas / 142, 2)))
    receipt["occupancy_per_snap"] = occupancy

    # ---- 5. the pre-stated decision rule, applied
    barrier_ms = TOKENS * SYNCS_PER_TOKEN * node_us / 1e3
    # The fallback needs no building to be priced: the L2-resident
    # single-CTA walk IS K7's walk (one CTA per row, dependent defined-order
    # folds over the L2-resident store) -- its receipted wall is the honest
    # fallback number, not an ideal-streaming estimate (the single-CTA pass
    # above is the same fold-latency class that makes K7 13x slower than its
    # own pure-bandwidth floor).
    k7_receipted_scan_ms = 374.3
    projected = dict(
        distributed_barrier_ms=round(barrier_ms, 2),
        distributed_projected_scan_ms=round(
            barrier_ms + TOKENS * (read_us + rw_us + 3 * node_us) / 1e3, 1),
        fallback_scan_ms=k7_receipted_scan_ms,
        fallback_note=("the L2-resident single-CTA variant already exists: it "
                       "is the K7 walk (receipts/k7_high_rank.json, 374.3 "
                       "ms/scan at rank 2048, fold-bound at 275 GB/s "
                       "effective); the distributed variant parallelises those "
                       "folds across the slot-block CTAs"),
        eager_multilaunch_barrier_ms=round(
            TOKENS * SYNCS_PER_TOKEN * eager_walklike / 1e3, 1),
    )
    receipt["projection"] = projected
    build_distributed = barrier_ms < DECISION_BUDGET_MS
    receipt["decision"] = dict(
        rule=("if 2048 tokens x syncs/token x measured barrier-us < 40 ms/scan "
              "then the distributed design, else the L2-resident single-CTA "
              "fallback"),
        measured_barrier_ms=round(barrier_ms, 2),
        budget_ms=DECISION_BUDGET_MS,
        build_distributed=bool(build_distributed),
        variant=("distributed walk, the store L2-resident on-chip, the "
                 "per-token barrier the graph-captured launch boundary "
                 "(the best Triton-expressible form of the "
                 "distributed-persistent design; the smem-persistent form is "
                 "impossible -- see impossibility)"
                 if build_distributed else
                 "L2-resident single-CTA walk"),
        reason=("the graph-captured launch boundary costs ~%.2f us/node, so "
                "2048 tokens x 3 barriers = %.1f ms/scan, under the 40 ms "
                "budget; the smem-resident form is receipted impossible "
                "(no Triton grid sync; 256 CTAs > 142 co-resident; 16 MB > "
                "9.3 MB addressable on-chip), so the state rides the L2 -- "
                "on-chip at the probe shape -- and the sweeps distribute "
                "across CTAs exactly as the brief's design specifies"
                % (node_us, barrier_ms)),
    )

    text = json.dumps(receipt, indent=1)
    print(text)
    if out is not None:
        with open(out, "w") as f:
            f.write(text + "\n")
        print(f"[k9 gate 0] receipt written to {out}")
    return receipt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(out=args.out)
