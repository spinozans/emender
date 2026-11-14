"""The matrix-state Emender arm: a capacity-matched LM built on ``scan.py``.

One block is the registered GDN-2 block shape with the mixer swapped:

    h = x + mixer(norm1(x))
    h = h + mlp(norm2(h))

The mixer is the matrix store: ``S in R^{rank x d_v}`` per head, three projections
(query into the value space, value, output), one scalar DECAY gate per token and
head, and the five schedule-driven rank-1 ops.  **The rank is not a parameter**:
the query lives in the value space and addresses *rows*, so ``rank`` only sizes the
state.  ``matrix_append`` and ``matrix_ops`` are therefore parameter-identical, and
growing the rank changes the memory footprint and nothing else.

Parameters are matched to the 150 M board exactly as every other arm in this
project does it (``model_for`` in ``x42_pile_scale/byte_velocity.py``): total at
``CAPACITY`` within 0.1 %, width/depth taken from the registered GDN-2 shape so the
backbone is the same size, and the difference in mixer parameters absorbed by the
MLP.  That absorption is a deconfound and is reported as one: this mixer is much
cheaper per layer than GDN-2's, so its SwiGLU is wider at matched total capacity.

There is no activation checkpointing anywhere in this arm (``checkpoint`` is what
produces ``ReplayBackward``/``reverse_kernel``/``Replay``, 78 % of the tape arm's
step): the backward is ordinary autograd through the chunked scan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from .scan import routed_chunk_scan
from .schedule import RoutingPolicy, route_chunk

VOCAB = 256
CAPACITY = 150_003_984
CAPACITY_TOLERANCE = 1e-3
# The registered GDN-2 150M backbone, for an apples-to-apples width and depth.
REFERENCE_WIDTH = 576
REFERENCE_DEPTH = 24
# The DECAY gate starts at a 64-token horizon (lam0 = exp(-1/64)); a 128-slot store
# written by a 2048-byte window is rewritten ~16 times, so the horizon is short
# enough that stale rows leave the read on their own.
DECAY_HORIZON_TOKENS = 64
DECAY_FLOOR = 1e-4
# See MatrixMixer: the chunk loop is launch-bound, so it is compiled and graphed.
COMPILE_MODE = "default"
# One compiled scan shared by every layer: the arm has 24 identical mixers, so a
# per-module graph would be 24 compiles of the same shapes.  The mode is "default"
# (inductor fusion, no CUDA graph) because per-module CUDA graphs share a
# cudagraph-trees pool and invalidate each other's outputs between a forward and its
# backward -- that fault is recorded in RESULT_MATRIX_EMENDER.md.
_SCAN_CACHE: dict = {}


def compiled_scan(mode: str = COMPILE_MODE):
    """The chunk scan, compiled once and shared by every mixer in the model."""
    if mode not in _SCAN_CACHE:
        _SCAN_CACHE[mode] = torch.compile(routed_chunk_scan, mode=mode, dynamic=False)
    return _SCAN_CACHE[mode]


@dataclass(frozen=True)
class MatrixShape:
    width: int = REFERENCE_WIDTH
    depth: int = REFERENCE_DEPTH
    hidden: int = 3352
    heads: int = 2
    value_dim: int = 128
    rank: int = 128
    rank_max: int = 128
    vocab: int = VOCAB

    def __post_init__(self):
        if self.rank > self.rank_max:
            raise ValueError("the active rank must be a prefix of the allocated rank")
        if self.rank_max < 1:
            raise ValueError("rank_max must be positive")


def parameter_count(shape: MatrixShape) -> dict:
    """Exact parameter count of the arm (rank contributes nothing by construction)."""
    d, hidden, heads, value_dim, vocab = (shape.width, shape.hidden, shape.heads,
                                          shape.value_dim, shape.vocab)
    mixer = d * heads * value_dim * 3 + d * heads + heads
    norms = 2 * d
    mlp = 3 * d * hidden
    per_layer = norms + mixer + mlp
    total = vocab * d * 2 + d + shape.depth * per_layer
    return dict(total=total, trainable=total, per_layer=per_layer, mixer=mixer,
                mlp=mlp, norms=norms, embedding=vocab * d, head=vocab * d,
                final_norm=d, capacity=CAPACITY,
                capacity_relative=total / CAPACITY - 1.0)


def matrix_config(capacity: int = CAPACITY, heads: int = 2, value_dim: int = 128,
                  rank: int = 128, rank_max: int | None = None,
                  width: int = REFERENCE_WIDTH, depth: int = REFERENCE_DEPTH,
                  vocab: int = VOCAB) -> MatrixShape:
    """Solve the MLP width that lands the arm on ``capacity`` at a fixed backbone."""
    rank_max = rank if rank_max is None else rank_max
    fixed = vocab * width * 2 + width + depth * (2 * width + width * heads * value_dim * 3
                                                 + width * heads + heads)
    remaining = capacity - fixed
    if remaining <= 0:
        raise ValueError("capacity is below the compulsory parameters")
    hidden = int(remaining // (depth * 3 * width) // 8 * 8)
    shape = MatrixShape(width=width, depth=depth, hidden=hidden, heads=heads,
                        value_dim=value_dim, rank=rank, rank_max=rank_max, vocab=vocab)
    if abs(parameter_count(shape)["capacity_relative"]) >= CAPACITY_TOLERANCE:
        raise ValueError(
            f"no MLP width lands {width}x{depth} on {capacity}: got "
            f"{parameter_count(shape)['total']}")
    return shape


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


class MatrixMixer(nn.Module):
    """The store: query/value/output projections plus the DECAY gate and the scan."""

    def __init__(self, shape: MatrixShape, policy: RoutingPolicy, chunk_tokens: int,
                 compile_scan: bool = False, fused_scan: bool = False):
        super().__init__()
        if fused_scan and compile_scan:
            raise ValueError("the fused K1 scan is one kernel per chunk already; "
                             "do not torch.compile it (use compile_scan=False)")
        value_dim = shape.heads * shape.value_dim
        self.heads = shape.heads
        self.value_dim = shape.value_dim
        self.rank = shape.rank
        self.rank_max = shape.rank_max
        self.policy = policy
        self.chunk_tokens = chunk_tokens
        self.query = nn.Linear(shape.width, value_dim, bias=False)
        self.value = nn.Linear(shape.width, value_dim, bias=False)
        self.out = nn.Linear(value_dim, shape.width, bias=False)
        self.decay = nn.Linear(shape.width, shape.heads, bias=True)
        with torch.no_grad():
            lam0 = math.exp(-1.0 / DECAY_HORIZON_TOKENS)
            self.decay.bias.fill_(math.log(lam0 / (1.0 - lam0)))
        # Telemetry is opt-in: keeping the scan's tensors alive would pin the
        # CUDA-graph static pool and trip "output of CUDAGraphs overwritten".
        self.collect_telemetry = False
        self.telemetry = None
        # The chunk loop is launch-bound (measured 43 s/step eager against 5-9 s/step
        # compiled at chunk 128), so the train path compiles it and captures it in a
        # CUDA graph ("reduce-overhead"): the binding cost is the per-kernel launch
        # count, not bytes or FLOPs.  Compiling the whole 24-layer forward instead is
        # impractical — a single graph that large did not finish compiling in 50 min.
        self.compile_scan = compile_scan
        # Stage K1 (``k1_schedule_scan``): the schedule-decoded state scan as one
        # fused kernel.  Default-off; the eager reference stays the default path.
        self.fused_scan = fused_scan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out, result = self.mix(x)
        if self.collect_telemetry:
            self.telemetry = result
        return out

    def mix(self, x: torch.Tensor):
        batch, tokens, _ = x.shape
        query = self.query(x).view(batch, tokens, self.heads, self.value_dim)
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
        decay = torch.sigmoid(self.decay(x)).clamp(min=DECAY_FLOOR, max=1.0 - 1e-6)
        scan = compiled_scan(COMPILE_MODE) if self.compile_scan else routed_chunk_scan
        result = scan(
            query, payload, decay, router=route_chunk, policy=self.policy,
            rank=self.rank, rank_max=self.rank_max, chunk_tokens=self.chunk_tokens,
            fused=self.fused_scan)
        return (self.out(result.reads.reshape(batch, tokens, self.heads * self.value_dim)),
                result)


class MatrixBlock(nn.Module):
    def __init__(self, shape: MatrixShape, policy: RoutingPolicy, chunk_tokens: int,
                 compile_scan: bool = False, fused_scan: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(shape.width, eps=1e-5)
        self.mixer = MatrixMixer(shape, policy, chunk_tokens, compile_scan, fused_scan)
        self.norm2 = nn.RMSNorm(shape.width, eps=1e-5)
        self.mlp = SwiGLU(shape.width, shape.hidden)

    def forward(self, x):
        h = x + self.mixer(self.norm1(x))
        return h + self.mlp(self.norm2(h))


class MatrixModel(nn.Module):
    def __init__(self, shape: MatrixShape, policy: RoutingPolicy, chunk_tokens: int = 64,
                 init_scale: float = 1.0, compile_scan: bool = True,
                 fused_scan: bool = False):
        super().__init__()
        self.shape = shape
        self.policy = policy
        self.embed = nn.Embedding(shape.vocab, shape.width)
        self.norm = nn.RMSNorm(shape.width, eps=1e-5)
        self.head = nn.Linear(shape.width, shape.vocab, bias=False)
        self.blocks = nn.ModuleList(
            MatrixBlock(shape, policy, chunk_tokens, compile_scan, fused_scan)
            for _ in range(shape.depth))
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.mul_(init_scale)
        from fla.modules import FusedLinearCrossEntropyLoss
        self.criterion = FusedLinearCrossEntropyLoss()

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def loss(self, ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Same loss path as the registered `gdn2` arm (bf16 compute, fused linear CE).
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.features(ids)
            return self.criterion(hidden.to(torch.bfloat16).contiguous(),
                                  targets.contiguous(),
                                  self.head.weight.to(torch.bfloat16))

    def counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return dict(total=total, trainable=sum(p.numel() for p in self.parameters()
                                               if p.requires_grad))

    def collect_telemetry(self, enabled: bool = True) -> "MatrixModel":
        for block in self.blocks:
            block.mixer.collect_telemetry = enabled
        return self

    def rank_trajectory(self) -> list:
        blocks = [block.mixer.telemetry for block in self.blocks
                  if block.mixer.telemetry is not None]
        if not blocks:
            return []
        return blocks[0].stats.summary()["rank_trajectory"]


def model_for(shape: MatrixShape, policy: RoutingPolicy, chunk_tokens: int = 64,
              init_scale: float = 1.0, compile_scan: bool = True,
              fused_scan: bool = False) -> MatrixModel:
    model = MatrixModel(shape, policy, chunk_tokens, init_scale, compile_scan, fused_scan)
    observed = model.counts()
    expected = parameter_count(shape)
    if observed["total"] != expected["total"]:
        raise ValueError(f"parameter audit disagrees: {observed} vs {expected}")
    if abs(expected["capacity_relative"]) >= CAPACITY_TOLERANCE:
        raise ValueError(f"arm is off the 150 M board: {expected['capacity_relative']:+.4%}")
    return model
