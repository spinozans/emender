"""Colinear chaining over exact-match anchors, and the correspondence it implies.

This is stage 2 of the operator chain in DESIGN_SUFFIX_ATTENTION.md (R2.2):

    seeds -> CHAIN -> correspondence -> fork

Stage 1 (`xsa seeds` / `xsa query`) yields *anchors*: (query offset, corpus
offset, exact match length) triples, in no particular order and with no
guarantee that they lie on one diagonal. A bag of anchors says "this document
shares a string with me". A *chain* is a set of anchors that is monotone in both
coordinates and consistent in its diagonal drift, so the diagonal it defines is
a positional pointer: given a query coordinate it names the corpus coordinate
that answers for it. That pointer is the only thing that makes the fork site
(`xsa branches` at the aligned corpus position) well defined.

Nothing here reads the corpus or the index; it is pure geometry over anchors.
The scoring rule is a sparse DP in the minimap2/BWA-MEM family, with two
deliberate simplifications, both registered and testable:

* a chain's score is *covered query bytes* minus gap penalties, so overlapping
  anchors cannot inflate it;
* a pair of anchors may overlap by up to `overlap_tol` bytes, and only the
  non-overlapping part of the later anchor is credited.

Anchors come from exact matches, so there is no "alignment" to score once the
diagonal is fixed: the DP is over ordering, not over edit distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# Registered defaults. `diag_penalty` makes one byte of unexplained diagonal
# drift cost exactly what one matched byte earns, so a chain of exact matches
# only survives if it is genuinely colinear. `gap_extend` is cheaper per byte
# because an unmatched stretch between two exact matches is normal (a sentence
# in between, a changed argument) while a *diagonal shift* is not.
DEFAULT_DIAG_PENALTY = 1.0
DEFAULT_GAP_EXTEND = 0.25
DEFAULT_OVERLAP_TOL = 8
DEFAULT_MIN_SCORE = 8.0
DEFAULT_MAX_EXTRAPOLATION = 0

MATCHED = "matched"
ABUTTING = "abutting"
GAP = "gap"


@dataclass(frozen=True, order=True)
class Anchor:
    """One exact match: query[query_pos:query_pos+length] == corpus[corpus_pos:...]."""

    query_pos: int
    corpus_pos: int
    length: int
    seed_id: int = 0
    multiplicity: int = 0


@dataclass(frozen=True)
class Chain:
    """A colinear, gap-consistent set of anchors, in query order."""

    anchors: tuple[Anchor, ...]
    score: float

    @property
    def n_anchors(self) -> int:
        return len(self.anchors)

    @property
    def query_start(self) -> int:
        return self.anchors[0].query_pos

    @property
    def query_end(self) -> int:
        return max(a.query_pos + a.length for a in self.anchors)

    @property
    def corpus_start(self) -> int:
        return min(a.corpus_pos for a in self.anchors)

    @property
    def corpus_end(self) -> int:
        return max(a.corpus_pos + a.length for a in self.anchors)

    @property
    def covered_query_bytes(self) -> int:
        covered, cursor = 0, None
        for lo, hi in sorted((a.query_pos, a.query_pos + a.length) for a in self.anchors):
            if cursor is None or lo > cursor:
                covered += hi - lo
                cursor = hi
            elif hi > cursor:
                covered += hi - cursor
                cursor = hi
        return covered

    @property
    def diagonal(self) -> int:
        """corpus_pos - query_pos of the first anchor (the chain's frame)."""
        return self.anchors[0].corpus_pos - self.anchors[0].query_pos


@dataclass(frozen=True)
class Correspondence:
    """The corpus position that answers for one query position."""

    query_pos: int
    corpus_pos: int
    kind: str
    extrapolation_bytes: int
    chain_index: int
    chain_score: float
    anchor_index: int

    @property
    def usable(self) -> bool:
        """Usable means the answering byte is a matched or immediately abutting one."""
        return self.kind in (MATCHED, ABUTTING)


def _penalty(left: Anchor, right: Anchor, diag_penalty: float, gap_extend: float) -> float | None:
    """Gap cost for right following left on one chain, or None if not colinear."""
    dq = right.query_pos - (left.query_pos + left.length)
    dc = right.corpus_pos - (left.corpus_pos + left.length)
    return diag_penalty * abs(dq - dc) + gap_extend * (max(0, dq) + max(0, dc))


def _gain(right: Anchor, left: Anchor | None, overlap_tol: int) -> int | None:
    """Credited query bytes for `right` after `left`, or None if the pair is illegal."""
    if left is None:
        return right.length
    dq = right.query_pos - (left.query_pos + left.length)
    dc = right.corpus_pos - (left.corpus_pos + left.length)
    if dq < -overlap_tol or dc < -overlap_tol:
        return None
    # Overlapping query bytes were already credited by `left`; corpus overlap is
    # not credited either, so the score stays a count of newly matched bytes.
    overlap = max(0, -dq, -dc)
    gain = right.length - overlap
    return gain if gain > 0 else None


def _dedupe(anchors: Iterable[Anchor]) -> list[Anchor]:
    """Fix one representative per identical (query, corpus, length) triple."""
    best: dict[tuple[int, int, int], Anchor] = {}
    for a in anchors:
        if a.length <= 0:
            continue
        key = (a.query_pos, a.corpus_pos, a.length)
        keep = best.get(key)
        if keep is None or (a.seed_id, -a.multiplicity) < (keep.seed_id, -keep.multiplicity):
            best[key] = a
    return list(best.values())


def chain_anchors(
    anchors: Iterable[Anchor],
    *,
    diag_penalty: float = DEFAULT_DIAG_PENALTY,
    gap_extend: float = DEFAULT_GAP_EXTEND,
    overlap_tol: int = DEFAULT_OVERLAP_TOL,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[Chain]:
    """Sparse DP chaining. Returns chains by decreasing score, deterministically.

    The DP is over anchors ordered by (corpus_pos, query_pos): f[i] is the best
    score of a colinear chain ending at anchor i, f[i] = gain(i, j) + f[j] -
    penalty(i, j) maximised over legal predecessors j (or the bare anchor). Any
    anchor may sit in at most one returned chain; chains are extracted best
    first, so a crowded region yields one chain plus its fragments.
    """
    keep = _dedupe(anchors)
    keep.sort(key=lambda a: (a.corpus_pos, a.query_pos, -a.length, a.seed_id, a.multiplicity))
    n = len(keep)
    score = [0.0] * n
    pred = [-1] * n
    for i in range(n):
        gain = _gain(keep[i], None, overlap_tol)
        best, best_j = float(gain), -1
        for j in range(i):
            gain_ij = _gain(keep[i], keep[j], overlap_tol)
            if gain_ij is None:
                continue
            penalty = _penalty(keep[j], keep[i], diag_penalty, gap_extend)
            assert penalty is not None
            cand = score[j] + gain_ij - penalty
            if cand > best:
                best, best_j = cand, j
        score[i], pred[i] = best, best_j

    order = sorted(range(n), key=lambda i: (-score[i], -keep[i].length, keep[i].corpus_pos, i))
    used = [False] * n
    chains: list[Chain] = []
    for i in order:
        if used[i] or score[i] <= min_score:
            continue
        path = []
        k = i
        while k >= 0 and not used[k]:
            path.append(k)
            used[k] = True
            k = pred[k]
        members = tuple(sorted((keep[x] for x in path), key=lambda a: (a.query_pos, a.corpus_pos)))
        chains.append(Chain(anchors=members,
                            score=_chain_score(members, diag_penalty, gap_extend, overlap_tol)))
    return chains


def _chain_score(anchors: Sequence[Anchor], diag_penalty: float, gap_extend: float,
                 overlap_tol: int) -> float:
    """Recompute a chain's score from its members (covered bytes minus gaps)."""
    if not anchors:
        return 0.0
    total = float(anchors[0].length)
    for left, right in zip(anchors, anchors[1:]):
        gain = _gain(right, left, overlap_tol)
        if gain is None:
            # Extraction never joins illegal pairs; a caller-built tuple may.
            raise ValueError("anchors are not colinear enough to be one chain")
        penalty = _penalty(left, right, diag_penalty, gap_extend)
        assert penalty is not None
        total += gain - penalty
    return total


def correspondences(chain: Chain, chain_index: int = 0) -> list[Correspondence]:
    """Every query coordinate this chain can answer for, with its kind.

    A coordinate inside an anchor is MATCHED (the corpus byte is the same byte,
    verified). One immediately after an anchor is ABUTTING (the anchor's right
    edge is the alignment point; no gap is being assumed). Anything further
    right on the same diagonal is a GAP answer and is only returned by
    `correspondence_at` when the caller allows extrapolation.
    """
    out: list[Correspondence] = []
    for idx, a in enumerate(chain.anchors):
        for q in range(a.query_pos, a.query_pos + a.length):
            out.append(Correspondence(q, a.corpus_pos + (q - a.query_pos), MATCHED, 0,
                                      chain_index, chain.score, idx))
        out.append(Correspondence(a.query_pos + a.length,
                                  a.corpus_pos + a.length, ABUTTING, 0, chain_index,
                                  chain.score, idx))
    return out


def correspondence_at(
    chain: Chain,
    query_pos: int,
    *,
    max_extrapolation: int = DEFAULT_MAX_EXTRAPOLATION,
    chain_index: int = 0,
) -> Correspondence | None:
    """The corpus position answering for `query_pos`, or None.

    Preference order is MATCHED, ABUTTING, GAP: verified evidence first. A GAP
    answer is an extrapolation along the chain's diagonal and is only offered up
    to `max_extrapolation` bytes past the last anchor that ends at or before the
    query position.
    """
    chosen: Correspondence | None = None
    for idx, a in enumerate(chain.anchors):
        if a.query_pos <= query_pos:
            offset = query_pos - a.query_pos
            if offset < a.length:
                return Correspondence(query_pos, a.corpus_pos + offset, MATCHED, 0,
                                      chain_index, chain.score, idx)
            if offset == a.length:
                chosen = Correspondence(query_pos, a.corpus_pos + a.length, ABUTTING, 0,
                                        chain_index, chain.score, idx)
            elif offset <= a.length + max_extrapolation:
                if chosen is None or chosen.kind != ABUTTING:
                    chosen = Correspondence(query_pos, a.corpus_pos + offset, GAP,
                                            offset - a.length, chain_index, chain.score, idx)
    return chosen


def best_correspondence(
    chains: Sequence[Chain],
    query_pos: int,
    *,
    max_extrapolation: int = DEFAULT_MAX_EXTRAPOLATION,
    require_usable: bool = True,
) -> Correspondence | None:
    """The strongest chain that answers for `query_pos`.

    Ties between chains are broken by MATCHED > ABUTTING > GAP, then by gap
    distance, then by chain score, then by corpus position, so the choice is
    deterministic.
    """
    rank = {MATCHED: 0, ABUTTING: 1, GAP: 2}
    best: Correspondence | None = None
    best_key: tuple | None = None
    for index, chain in enumerate(chains):
        corr = correspondence_at(chain, query_pos, max_extrapolation=max_extrapolation,
                                 chain_index=index)
        if corr is None or (require_usable and not corr.usable):
            continue
        key = (rank[corr.kind], corr.extrapolation_bytes, -corr.chain_score, corr.corpus_pos)
        if best_key is None or key < best_key:
            best, best_key = corr, key
    return best


def cluster_regions(chains: Sequence[Chain], *, gap: int = 2000) -> list[list[Chain]]:
    """Group chains whose corpus spans are within `gap` bytes (region clustering).

    This is the chaining analogue of `xsa fanout`'s region clustering: the
    cluster count is how many distinct corpus regions compete, which is the
    fan-out that matters for a bounded candidate frontier.
    """
    ordered = sorted((c for c in chains), key=lambda c: (c.corpus_start, c.corpus_end))
    clusters: list[list[Chain]] = []
    end = None
    for chain in ordered:
        if clusters and chain.corpus_start - end <= gap:
            clusters[-1].append(chain)
            end = max(end, chain.corpus_end)
        else:
            clusters.append([chain])
            end = chain.corpus_end
    return clusters


def fork_outcome(histogram: Mapping[int, int], actual: int) -> dict:
    """Where the model's actual next byte sits in a branch histogram.

    `histogram` maps next-byte -> occurrence count at the aligned corpus site
    (the `xsa branches` answer). Rank is 1-based over distinct bytes ordered by
    decreasing count (ties by byte value, deterministic); mass is the byte's
    share of all occurrences. An absent byte gets rank None and mass 0.0 -- that
    is a real outcome, not a missing cell.
    """
    total = sum(histogram.values())
    if total == 0:
        return {"present": False, "rank": None, "mass": 0.0, "expected_count": 0,
                "distinct": 0, "total": 0, "top1_mass": 0.0}
    order = sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = None
    for position, (byte, _count) in enumerate(order, start=1):
        if byte == actual:
            rank = position
            break
    count = histogram.get(actual, 0)
    return {
        "present": rank is not None,
        "rank": rank,
        "mass": count / total,
        "expected_count": count,
        "distinct": len(histogram),
        "total": total,
        "top1_mass": order[0][1] / total,
    }
