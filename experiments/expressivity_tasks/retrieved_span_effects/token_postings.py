"""Immutable token postings + GPU lexical candidate provider, distinct from learned reranking."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import triton
import triton.language as tl

BASE = 65536


@dataclass
class Image:
    pages: np.ndarray
    lens: np.ndarray
    codes: np.ndarray
    offsets: np.ndarray
    postings: np.ndarray
    vote: np.ndarray

    def tensors(self, device):
        return tuple(
            torch.from_numpy(x).to(device)
            for x in (
                self.pages,
                self.lens,
                self.codes,
                self.offsets,
                self.postings,
                self.vote,
            )
        )


def index(documents, span):
    pages = []
    seen = set()
    for doc in documents:
        doc = np.asarray(doc, dtype=np.int32)
        if doc.ndim != 1 or not len(doc) or doc.min() < 0 or doc.max() >= BASE - 1:
            raise ValueError("invalid document tokens")
        for i in range(0, len(doc), span):
            page = doc[i : i + span]
            key = page.tobytes()
            if key not in seen:
                seen.add(key)
                pages.append(page)
    if not pages:
        raise ValueError("empty archive")
    lens = np.array(list(map(len, pages)), dtype=np.int32)
    mat = np.zeros((len(pages), span), np.int32)
    for i, page in enumerate(pages):
        mat[i, : len(page)] = page
    token = mat.astype(np.int64) + 1
    codes = []
    rows = []
    for gram in (1, 2, 3):
        n = span + 1 - gram
        if n <= 0:
            continue
        live = np.arange(n)[None, :] + gram <= lens[:, None]
        code = token[:, :n].copy()
        for j in range(1, gram):
            code = code * BASE + token[:, j : j + n]
        codes.append(code[live])
        rows.append(np.nonzero(live)[0].astype(np.int32))
    code = np.concatenate(codes)
    row = np.concatenate(rows)
    order = np.lexsort((row, code))
    code, row = code[order], row[order]
    keep = np.r_[True, (code[1:] != code[:-1]) | (row[1:] != row[:-1])]
    code, row = code[keep], row[keep]
    keys, start = np.unique(code, return_index=True)
    offsets = np.r_[start, len(row)].astype(np.int64)
    gram = 1 + (keys >= BASE).astype(np.int32) + (keys >= BASE**2).astype(np.int32)
    vote = gram * (
        1
        + np.floor(256 * np.log((1 + len(pages)) / (1 + np.diff(offsets)))).astype(
            np.int32
        )
    )
    return Image(mat, lens, keys, offsets, row, vote.astype(np.int32))


@triton.jit
def _seek(
    X,
    On,
    Code,
    Off,
    Vote,
    R,
    W: tl.constexpr,
    N: tl.constexpr,
    F: tl.constexpr,
    BL: tl.constexpr,
    IT: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    f = tl.arange(0, BL)
    end = f // 3
    width = f % 3 + 1
    x = tl.load(X + b * W + end, end < W, 0).to(tl.int64) + 1
    prev = tl.load(X + b * W + end - 1, (end >= 1) & (end < W), 0).to(tl.int64) + 1
    before = tl.load(X + b * W + end - 2, (end >= 2) & (end < W), 0).to(tl.int64) + 1
    key = (
        x
        + tl.where(width >= 2, prev * 65536, 0)
        + tl.where(width == 3, before * 4294967296, 0)
    )
    lo = tl.full((BL,), 0, tl.int32)
    hi = tl.full((BL,), N, tl.int32)
    for _ in range(IT):
        mid = (lo + hi) // 2
        val = tl.load(Code + mid, mid < N, 9223372036854775807)
        right = (lo < hi) & (val < key)
        hi = tl.where((lo < hi) & ~right, mid, hi)
        lo = tl.where(right, mid + 1, lo)
    hit = (
        (f < F)
        & (width <= end + 1)
        & (lo < N)
        & (tl.load(On + b) != 0)
        & (tl.load(Code + lo, lo < N, -1) == key)
    )
    start = tl.load(Off + lo, hit, 0)
    stop = tl.load(Off + lo + 1, hit, 0)
    vote = tl.load(Vote + lo, hit, 0)
    pos = (b * F + f) * 3
    tl.store(R + pos, start, f < F)
    tl.store(R + pos + 1, stop, f < F)
    tl.store(R + pos + 2, vote, f < F)


@triton.jit
def _vote(
    R,
    Page,
    Scores,
    N: tl.constexpr,
    F: tl.constexpr,
    BS: tl.constexpr,
    SH: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    f = tl.program_id(1)
    shard = tl.program_id(2)
    ix = (b * F + f) * 3
    start = tl.load(R + ix) + shard * BS
    stop = tl.load(R + ix + 1)
    weight = tl.load(R + ix + 2).to(tl.int32)
    lane = tl.arange(0, BS)
    while start < stop:
        pos = start + lane
        page = tl.load(Page + pos, pos < stop, 0)
        tl.atomic_add(
            Scores + b * N + page, weight + 0 * lane, pos < stop, sem="relaxed"
        )
        start += BS * SH


@triton.jit
def _top(
    Scores,
    Tmp,
    Ids,
    Mass,
    N: tl.constexpr,
    NT: tl.constexpr,
    P: tl.constexpr,
    BS: tl.constexpr,
    FINAL: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    i = tile * BS + tl.arange(0, BS)
    if FINAL:
        key = tl.load(Tmp + b * NT * P + i, i < NT * P, 0)
    else:
        s = tl.load(Scores + b * N + i, i < N, 0).to(tl.int64)
        key = tl.where(s > 0, s * (N + 1) + N - i, 0)
    # Deterministic: larger score, then smaller page ID. Zero never fills a miss.
    for j in tl.static_range(P):
        best = tl.max(key, 0)
        if FINAL:
            tl.store(Ids + b * P + j, tl.where(best > 0, N - best % (N + 1), -1))
            tl.store(Mass + b * P + j, best // (N + 1))
        else:
            tl.store(Tmp + (b * NT + tile) * P + j, best)
        key = tl.where(key == best, 0, key)


@torch.no_grad()
def locate(image, prefix, enabled, hits=4):
    page, lens, codes, off, posting, vote = image
    n = len(lens)
    b, w = prefix.shape
    f = 3 * w
    if not (
        n > 0
        and len(codes) > 0
        and 0 < w <= 64
        and 0 < hits <= 16
        and enabled.shape == (b,)
        and prefix.is_contiguous()
        and enabled.is_contiguous()
    ):
        raise ValueError("invalid locator layout")
    device = prefix.device
    desc = torch.empty((b, f, 3), device=device, dtype=torch.int64)
    scores = torch.zeros((b, n), device=device, dtype=torch.int32)
    _seek[(b,)](
        prefix,
        enabled,
        codes,
        off,
        vote,
        desc,
        w,
        len(codes),
        f,
        triton.next_power_of_2(f),
        len(codes).bit_length(),
    )
    _vote[(b, f, 4)](desc, posting, scores, n, f, 128, 4)
    nt = triton.cdiv(n, 256)
    tmp = torch.empty((b, nt, hits), device=device, dtype=torch.int64)
    ids = torch.empty((b, hits), device=device, dtype=torch.int32)
    mass = torch.empty_like(ids)
    _top[(b, nt)](scores, tmp, ids, mass, n, nt, hits, 256, False)
    _top[(b, 1)](
        scores, tmp, ids, mass, n, nt, hits, triton.next_power_of_2(nt * hits), True
    )
    return ids, mass


@triton.jit
def _get(Pages, Lens, Ids, Y, L, N: tl.constexpr, T: tl.constexpr, BT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    page = tl.load(Ids + row).to(tl.int64)
    t = tl.arange(0, BT)
    n = tl.load(Lens + page, (page >= 0) & (page < N), -1)
    val = tl.load(Pages + page * T + t, t < n, 0)
    tl.store(Y + row * T + t, val, t < T)
    tl.store(L + row, n)


@torch.no_grad()
def fetch(image, ids):
    pages, lens, *_ = image
    n, t = pages.shape
    ids = ids.contiguous()
    y = ids.new_empty((*ids.shape, t))
    lengths = torch.empty_like(ids)
    _get[(ids.numel(),)](pages, lens, ids, y, lengths, n, t, triton.next_power_of_2(t))
    return y, lengths


def literal(documents, prefix, span, hits=4):
    # Independent occurrence enumeration for small invented inputs.
    pages = list(
        dict.fromkeys(
            tuple(doc[j : j + span])
            for doc in documents
            for j in range(0, len(doc), span)
        )
    )
    totals = [0] * len(pages)
    for j in range(len(prefix)):
        for g in (1, 2, 3):
            if j + 1 < g:
                continue
            key = tuple(prefix[j + 1 - g : j + 1])
            ids = [
                i
                for i, p in enumerate(pages)
                if any(p[a : a + g] == key for a in range(len(p) - g + 1))
            ]
            w = g * (1 + math.floor(256 * math.log((1 + len(pages)) / (1 + len(ids)))))
            for i in ids:
                totals[i] += w
    chosen = sorted(
        (i for i in range(len(pages)) if totals[i]), key=lambda i: (-totals[i], i)
    )[:hits]
    return chosen + [-1] * (hits - len(chosen)), [totals[i] for i in chosen] + [0] * (
        hits - len(chosen)
    )
