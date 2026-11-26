"""Integer-only training packets. One live row per REAL shifted target, no padding.

`inputs` and route construction never consume the final target of a window.
Shared local K/V row p has a successor in the live inputs and is readable only
by q>p in that same window. All starts/lengths are validated at construction.
"""

from dataclasses import dataclass
from itertools import accumulate, chain
from operator import index as integer

import torch

from ..corpus_kernel.anchors import proposals
from ..corpus_kernel.index import tokens
from ..corpus_kernel.plan import Layout, Plan, tiles
from ..corpus_kernel.attention import TILE
from ..corpus_kernel.work import bind


def longs(seq, device):
    return torch.tensor(seq, dtype=torch.int64, device=device)


@dataclass(frozen=True)
class Inputs:
    ids: torch.Tensor
    lengths: tuple
    spans: torch.Tensor

    @classmethod
    def pack(cls, windows, device):
        lengths = tuple(len(p) for p in windows if p)
        offset = tuple(accumulate(lengths, initial=0))
        return cls(
            longs(list(chain.from_iterable(windows)), device),
            lengths,
            longs(list(zip(offset[:-1], lengths, strict=True)), device).reshape(-1, 2),
        )


class WindowArchive:
    def __init__(self, m, index):
        if index.tokenizer != m.tokenizer:
            raise ValueError("foreign tokenizer")
        self.device = m.embed.weight.device
        self.index = index
        self.vocab = m.vocab
        self.rank = len(m.omega)
        self.layout = Layout(index.version, len(index), m.leaf)
        if any(p and max(p) >= m.vocab for p in index.documents):
            raise ValueError("archive token outside vocabulary")
        self.inputs = Inputs.pack([p[:-1] for p in index.documents], self.device)
        self.order = longs(index.order, self.device)
        self.successors = longs(index.successor, self.device)
        self._resident = None

    def check(self, m):
        if (m.embed.weight.device, m.tokenizer, m.vocab, m.leaf, len(m.omega)) != (
            self.device,
            self.index.tokenizer,
            self.vocab,
            self.layout.leaf,
            self.rank,
        ):
            raise ValueError("incompatible archive and model")

    def _seek(self, windows, depth):
        from ..corpus_kernel.triton_frontier import ResidentIndex

        if self._resident is None:
            self._resident = ResidentIndex(self.index, self.device)
        return (
            self._resident.intervals(
                longs(windows, self.device),
                torch.full(
                    (len(windows),), depth, dtype=torch.int64, device=self.device
                ),
                min_match=depth,
            )
            .cpu()
            .tolist()
        )

    def windows(
        self,
        documents,
        *,
        chunk_targets=128,
        min_match=1,
        lookback=0,
        local=True,
        corpus=True,
        allow_skip=True,
        locator="cpu",
    ):
        return Windows(
            self,
            documents,
            chunk_targets,
            min_match,
            lookback,
            local,
            corpus,
            allow_skip,
            locator,
        )


@dataclass(frozen=True)
class Block:
    start: int
    stop: int
    plan: Plan
    work: object
    local_tiles: torch.Tensor


class Windows:
    def __init__(
        self,
        archive,
        documents,
        chunk_targets,
        min_match,
        lookback,
        local,
        corpus,
        allow_skip,
        locator,
    ):
        self.archive = archive
        self.local = local
        self.corpus = corpus
        self.allow_skip = allow_skip
        dev = archive.device
        size = integer(chunk_targets)
        depth = integer(min_match)
        back = None if lookback is None else integer(lookback)
        if (
            size < 1
            or depth < 1
            or (back is not None and back < 0)
            or locator not in ("cpu", "triton")
        ):
            raise ValueError("invalid window work settings")
        layout = (
            archive.layout
            if corpus
            else Layout(archive.index.version, 0, archive.layout.leaf)
        )
        if not layout.rows and not allow_skip:
            raise ValueError("first frontier needs corpus or a base LM row")
        ins = []
        labels = []
        for source, seq in documents:
            archive.index.check_live(source)
            p = tokens(seq)
            if p and max(p) >= archive.vocab:
                raise ValueError("live token outside vocabulary")
            if len(p) > 1:
                ins.append(p[:-1])
                labels.extend(p[1:])
        if not labels:
            raise ValueError("no real shifted targets")
        self.inputs = Inputs.pack(ins, dev)
        self.targets = longs(labels, dev)
        self.blocks = []
        local_rows = []
        starts = []
        counts = []
        frontiers = []

        def emit():
            base = self.blocks[-1].stop if self.blocks else 0
            hits = (
                proposals(
                    archive.index,
                    frontiers,
                    min_match=depth,
                    lookback=back,
                    seek=archive._seek if locator == "triton" else None,
                )
                if corpus
                else [() for _ in counts]
            )
            plan = Plan.build(layout, hits)
            work = bind(plan, archive.rank, (), dev, tile=TILE)
            ranges = (
                (
                    (g, s, s + n)
                    for g, (s, n) in enumerate(zip(starts, counts, strict=True))
                )
                if local
                else ()
            )
            lt = longs(tiles(ranges, TILE), dev).reshape(-1, 3)
            self.blocks.append(Block(base, base + len(counts), plan, work, lt))
            starts.clear()
            counts.clear()
            frontiers.clear()

        pos = lbase = 0
        for p in ins:
            if local:
                local_rows.extend(range(pos, pos + len(p) - 1))
            for t in range(len(p)):
                starts.append(lbase)
                counts.append(t)
                if corpus:
                    # Only this frontier and its bounded lookback can nominate rows. Never pass its label.
                    start = 0 if back is None else max(0, t + 1 - depth - back)
                    frontiers.append(p[start : t + 1])
                if len(counts) == size:
                    emit()
            pos += len(p)
            lbase += len(p) - 1
        if counts:
            emit()
        self.blocks = tuple(self.blocks)
        self.local_rows = longs(local_rows, dev)
        self.layout = layout

    def loss(self, m, *, mode="indexed", backend="torch", checkpoint_blocks=False):
        from .training import loss

        return loss(self, m, mode, backend, checkpoint_blocks)

    def predict(self, m, *, mode="indexed", backend="torch"):
        from .training import predict

        return predict(self, m, mode, backend)
