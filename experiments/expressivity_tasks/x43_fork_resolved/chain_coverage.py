"""Coverage measurement for the suffix-kernel operator chain on real text.

Runs the four stages of DESIGN_SUFFIX_ATTENTION.md R2.2 against the pile-head100
index through the existing `xsa` CLI:

    seeds        xsa seeds      adaptive seed length solved to a target multiplicity
    chain        chaining.py    colinear DP + correspondence (stage 2, built here)
    fork         xsa branches   next-byte histogram at the aligned corpus position

Two sources are measured with the same code path:

    transcript   a real agent conversation (message text + tool output), the
                 case the design actually claims
    pile         byte-uniform positions in the indexed corpus itself -- the
                 contrast panel, which also carries the source's own occurrence
                 in the index and is therefore NOT comparable on its own

Transcript bytes are never written to a receipt: only hashes, sizes, counts and
derived statistics.

Usage:
    python chain_coverage.py measure --source transcript --out receipts/...json
    python chain_coverage.py measure --source pile       --out receipts/...json
    python chain_coverage.py report --transcript ... --pile ...
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import time

from chaining import (
    Anchor,
    best_correspondence,
    chain_anchors,
    cluster_regions,
    fork_outcome,
)

XSA = "/home/erikg/sxgc/xsa/target/release/xsa"
CONTAINER = "/mnt/nvme2n1/erikg/P100c/pile-head100.sxi2"
CORPUS = "/mnt/nvme2n1/erikg/pile.txt"
INDEXED_BYTES = 100_000_000_000  # pile-head100: the container's text length
DEPTHS = (2, 4, 8, 16, 32)
MODES = ("scan", "scan+site")
BAND = max(DEPTHS)  # one query band per sampled position; depths are suffixes of it
TARGET_MULT = 50  # xsa seeds default: solve seed length for this multiplicity
ANCHOR_LIMIT = 8  # xsa seeds' own default position sample per seed
PILE_POSITION_FLOOR = 1024  # exclude the first 1 KiB, as CONTEXT_DEPTH.md does
EXTRAPOLATION = 16  # loose correspondence: a chain edge up to this far before the site
MIN_SITE_LEN = 12  # xsa seeds' default --min-len: the shortest seed the design counts
# Support-depth ladder: the fraction of positions whose context suffix of length L
# occurs at least twice in the corpus. It costs one width query per (position, L)
# and no locate, so it is the cheap answer to "how much of the loss is the L knob".
LADDER_LENGTHS = (2, 4, 6, 8, 10, 12, 16, 20, 24, 32)
TRANSCRIPT_DIR = "/home/erikg/.pi/agent/sessions/--home-erikg-emender--"
TRANSCRIPT_BYTES = 1_000_000
ROLE_ORDER = ("user", "assistant", "toolResult")
TEXT_BLOCKS = ("text", "thinking")  # toolCall arguments and toolResult output are added below


# --------------------------------------------------------------------- sources
def load_transcript(directory: str, target_bytes: int) -> dict:
    """Byte stream of a real conversation plus out-of-band message boundaries.

    The stream is the conversation's textual bytes in file order: text blocks,
    thinking blocks, tool-call arguments (JSON) and tool-result output. Message
    boundaries are kept beside the stream, never in it, so a sampled position
    can be required to sit inside one message.
    """
    paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".jsonl")
    )
    # Deterministic choice: the first session (by name) long enough to carry the
    # whole budget, so the source is a single real conversation.
    for path in paths:
        stream = bytearray()
        spans = []
        roles: dict[str, int] = {}
        blocks: dict[str, int] = {}
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read())
            handle.seek(0)
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = record.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                if not isinstance(content, list):
                    continue
                role = message.get("role") or record.get("type") or "unknown"
                start = len(stream)
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    kind = block.get("type")
                    if kind in TEXT_BLOCKS:
                        text = block.get(kind)
                    elif kind == "toolCall":
                        text = json.dumps(block.get("arguments"), sort_keys=True)
                    elif kind == "toolResult":
                        text = block.get("output") or block.get("text")
                    else:
                        text = None
                    if not isinstance(text, str):
                        continue
                    payload = text.encode("utf-8", "surrogatepass")
                    stream += payload
                    blocks[kind] = blocks.get(kind, 0) + 1
                if len(stream) > start:
                    spans.append((start, len(stream), role))
                    roles[role] = roles.get(role, 0) + 1
                if len(stream) >= target_bytes:
                    break
        if len(stream) >= target_bytes:
            raw = bytes(stream[:target_bytes])
            spans = [(s, min(e, target_bytes), r) for s, e, r in spans
                     if s < target_bytes and min(e, target_bytes) - s > BAND]
            return {
                "kind": "transcript",
                "path": path,
                "file_sha256": digest.hexdigest(),
                "stream_sha256": hashlib.sha256(raw).hexdigest(),
                "stream_bytes": len(raw),
                "messages": len(spans),
                "roles": roles,
                "blocks": blocks,
                "data": raw,
                "spans": spans,
            }
    raise SystemExit(f"no session in {directory} reaches {target_bytes} bytes")


class PileText:
    """Read-only byte access to the indexed corpus by offset."""

    def __init__(self, path: str = CORPUS):
        self.handle = open(path, "rb")

    def read(self, offset: int, length: int) -> bytes:
        self.handle.seek(offset)
        return self.handle.read(length)


# ------------------------------------------------------------------ xsa stages
def run_xsa(args: list[str], text: bytes | None = None, timeout: float | None = None):
    started = time.monotonic()
    done = subprocess.run(
        [XSA] + args,
        input=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    if done.returncode != 0:
        raise SystemExit(f"xsa {args[:2]} failed: {done.stderr.decode('utf-8', 'replace')[-2000:]}")
    return done.stdout, done.stderr.decode("utf-8", "replace"), elapsed


def seeds_for_band(band: bytes) -> tuple[list[dict], float]:
    """One `xsa seeds --limit 0` over one query band (no locate: scan only)."""
    out, _err, elapsed = run_xsa(
        ["seeds", CONTAINER, "-", "--target-mult", str(TARGET_MULT),
         "--limit", "0", "--jsonl"],
        text=band,
    )
    rows = []
    for line in out.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        rows.append({
            "text_offset": obj["text_offset"],
            "length": obj["length"],
            "occurrences": obj["occurrences"],
        })
    return rows, elapsed


def locate_patterns(patterns: list[bytes], limit: int) -> tuple[dict[int, list[int]], float, str]:
    """One batched `xsa query --patterns ... --limit N` for every distinct seed."""
    usable = [(i, p) for i, p in enumerate(patterns) if b"\x1e" not in p]
    dropped = len(patterns) - len(usable)
    payload = b"\x1e".join(p for _i, p in usable) + b"\x1e"
    with open("/tmp/chain_coverage_patterns.bin", "wb") as handle:
        handle.write(payload)
    out, err, elapsed = run_xsa(
        ["query", CONTAINER, "--patterns", "/tmp/chain_coverage_patterns.bin",
         "--limit", str(limit), "--output", "jsonl"]
    )
    positions: dict[int, list[int]] = {}
    for line in out.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        positions[usable[obj["pattern_id"]][0]] = list(obj["offsets"])
    note = f"dropped_patterns_with_0x1e={dropped}"
    return positions, elapsed, note


def counts_of_patterns(patterns: list[bytes]) -> tuple[list[int], list[int], float]:
    """Interval widths only (no locate): the parity check against seeds' mult.

    `--count` emits `pattern_id<TAB>pattern<TAB>occurrences` TSV only (it refuses
    --output jsonl), so a pattern containing a tab or a newline cannot be parsed
    back unambiguously. Those patterns are skipped and counted, not guessed at.
    """
    keep = [i for i, p in enumerate(patterns)
            if b"\x1e" not in p and not any(b < 0x20 for b in p)]
    payload = b"\x1e".join(patterns[i] for i in keep) + b"\x1e"
    with open("/tmp/chain_coverage_counts.bin", "wb") as handle:
        handle.write(payload)
    out, _err, elapsed = run_xsa(
        ["query", CONTAINER, "--patterns", "/tmp/chain_coverage_counts.bin", "--count"]
    )
    widths = []
    for line in out.decode("utf-8", "replace").splitlines():
        if not line or not line[0].isdigit():
            continue
        head, rest = line.split("\t", 1)
        _pattern, count = rest.rsplit("\t", 1)
        widths.append(int(count))
        assert int(head) == len(widths) - 1, line[:80]
    return keep, widths, elapsed


def _read_exact(stream, length: int) -> bytes:
    data = b""
    while len(data) < length:
        chunk = stream.read(length - len(data))
        if not chunk:
            raise SystemExit("branches serve: closed before the frame completed")
        data += chunk
    return data


def branch_histograms(contexts: list[bytes], jobs: int = 8, k: int = 256) -> tuple[list[dict], float]:
    """One `xsa branches --serve` session: next-byte histograms for all contexts.

    `--cross-doc` is required here: the pile-head100 container carries no m15
    document-start member, so the boundary-aware answer is unavailable and the
    conflated histogram is the same convention CONTEXT_DEPTH.md used.
    """
    if not contexts:
        return [], 0.0
    started = time.monotonic()
    proc = subprocess.Popen(
        [XSA, "branches", CONTAINER, "--serve", "-j", str(jobs), "-k", str(k), "--cross-doc"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None and proc.stdout is not None
    for ctx in contexts:
        proc.stdin.write(struct.pack("<I", len(ctx)) + ctx)
    proc.stdin.flush()
    out = []
    for expected in range(len(contexts)):
        # Each serve response is a u32 frame length, then the fixed frame:
        # id, l, r, width, eof, other (6 x u64), a u16 branch count, then
        # (byte u8, count u64) per branch.
        (length,) = struct.unpack("<I", _read_exact(proc.stdout, 4))
        frame = _read_exact(proc.stdout, length)
        ctx_id, _left, _right, width, eof, other = struct.unpack("<6Q", frame[:48])
        (nbytes,) = struct.unpack("<H", frame[48:50])
        hist = {}
        cursor = 50
        for _ in range(nbytes):
            byte = frame[cursor]
            (count,) = struct.unpack("<Q", frame[cursor + 1:cursor + 9])
            hist[byte] = count
            cursor += 9
        if ctx_id != expected:
            raise SystemExit(f"branches serve: id {ctx_id} != {expected}")
        out.append({"width": width, "eof": eof, "other": other, "hist": hist})
    proc.stdin.close()
    proc.wait(timeout=120)
    return out, time.monotonic() - started


# ------------------------------------------------------------------- sampling
def transcript_positions(source: dict, count: int, seed: int) -> tuple[list[int], dict]:
    """Seeded uniform positions inside single messages, with the exclusion count."""
    eligible = [(s, e) for s, e, _role in source["spans"] if e - s > BAND]
    total = sum(e - s - BAND for s, e in eligible)
    excluded = source["stream_bytes"] - total
    rng = random.Random(seed)
    picks = rng.sample(range(total), min(count, total))
    out = []
    for pick in sorted(picks):
        for s, e in eligible:
            span = e - s - BAND
            if pick < span:
                out.append(s + BAND + pick)
                break
            pick -= span
    return out, {"eligible_bytes": total, "excluded_bytes": excluded}


def pile_positions(count: int, seed: int) -> tuple[list[int], dict]:
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(PILE_POSITION_FLOOR, INDEXED_BYTES - BAND), count))
    return picks, {"eligible_bytes": INDEXED_BYTES - BAND - PILE_POSITION_FLOOR,
                   "excluded_bytes": PILE_POSITION_FLOOR}


# ------------------------------------------------------------------ the stages
def _positions_for(source: str, pile: "PileText", positions: int, seed: int):
    """Sampled prediction positions, the byte accessors, and the exclusion counts."""
    if source == "transcript":
        loaded = load_transcript(TRANSCRIPT_DIR, TRANSCRIPT_BYTES)
        stream = loaded.pop("data")
        spans = loaded.pop("spans")
        loaded["spans"] = len(spans)
        points, sampling = transcript_positions(
            {"stream_bytes": loaded["stream_bytes"], "spans": spans}, positions, seed)

        def next_byte(p):
            return stream[p]

        def context(p, d):
            return stream[max(0, p - d):p]
    elif source == "pile":
        loaded = {"kind": "pile", "container": CONTAINER, "corpus": CORPUS,
                  "indexed_bytes": INDEXED_BYTES}
        points, sampling = pile_positions(positions, seed)

        def next_byte(p):
            return pile.read(p, 1)[0]

        def context(p, d):
            return pile.read(p - d, d)
    else:
        raise SystemExit(f"unknown source {source}")
    return loaded, points, sampling, next_byte, context


def measure(source: str, positions: int, seed: int,
            anchor_limit: int = ANCHOR_LIMIT) -> dict:
    pile = PileText()
    loaded, points, sampling, next_byte, context = _positions_for(source, pile, positions, seed)
    timings = {"seeds_seconds": 0.0, "seeds_calls": 0, "count_seconds": 0.0,
               "locate_seconds": 0.0, "ladder_locate_seconds": 0.0,
               "branches_seconds": 0.0}

    # ---- stage 1: xsa seeds over each 32-byte band (scan only, --limit 0).
    # One invocation per band: `seeds` suppresses a seed contained in an earlier
    # one within a run, so concatenating bands into one file would let one
    # sample's seeds delete another's. With --limit 0 the call performs no
    # locate walk, so its cost is start-up plus the interval solve.
    bands, scan_rows = {}, []
    for index, p in enumerate(points):
        band = context(p, BAND)
        bands[index] = band
        rows, elapsed = seeds_for_band(band)
        timings["seeds_calls"] += 1
        timings["seeds_seconds"] += elapsed
        scan_rows.append(rows)

    # ---- stage 2: one pattern table for both seeding modes.
    #   scan  : the seeds `xsa seeds` accepted on the band (offset, length).
    #   site  : the suffix of the context that ends exactly at the prediction
    #           position, at the longest length whose multiplicity is >= 2. This
    #           is the seeding that makes a correspondence *possible at all*
    #           (R2.3 loss condition (a): no exact stretch of length >= 12).
    patterns: list[bytes] = []
    pattern_index: dict[bytes, int] = {}

    def pattern_id(pattern: bytes) -> int:
        pid = pattern_index.get(pattern)
        if pid is None:
            pid = pattern_index[pattern] = len(patterns)
            patterns.append(pattern)
        return pid

    scan_lookup = []  # (sample, query_pos_in_band, length, seeds_mult, pid)
    for index, rows in enumerate(scan_rows):
        for row in rows:
            off, length = row["text_offset"], row["length"]
            pattern = bands[index][off:off + length]
            if len(pattern) != length:
                continue
            scan_lookup.append((index, off, length, row["occurrences"],
                                pattern_id(pattern)))

    ladder_lookup = []  # (sample, depth, length, pid)
    for index in range(len(points)):
        band = bands[index]
        for depth in DEPTHS:
            for length in range(depth, MIN_SITE_LEN - 1, -1):
                ladder_lookup.append((index, depth, length,
                                      pattern_id(band[BAND - length:])))

    ladder_ids = {(index, length): pattern_id(bands[index][BAND - length:])
                  for index in range(len(points)) for length in LADDER_LENGTHS}

    every = list(dict.fromkeys([p for p in patterns]))
    checked, widths, count_seconds = counts_of_patterns(every)
    timings["count_seconds"] = count_seconds
    width_of = {pid: widths[i] for i, pid in enumerate(checked)}
    parity = sum(1 for (_s, _o, _l, mult, pid) in scan_lookup if width_of.get(pid) == mult)
    parity_checked = sum(1 for (_s, _o, _l, _m, pid) in scan_lookup if pid in width_of)

    # The longest site-ending suffix with multiplicity >= 2 (an occurrence other
    # than its own source): outermost first, so the first hit is the longest.
    ladder = {length: {"n": len(points),
                       "ge2": sum(1 for index in range(len(points))
                                  if width_of.get(ladder_ids[(index, length)], 0) >= 2),
                       "ge50": sum(1 for index in range(len(points))
                                   if width_of.get(ladder_ids[(index, length)], 0) >= TARGET_MULT)}
              for length in LADDER_LENGTHS}

    site_choice: dict[tuple[int, int], tuple[int, int]] = {}
    for sample_index, depth, length, pid in ladder_lookup:
        key = (sample_index, depth)
        if key in site_choice:
            continue
        if width_of.get(pid, 0) >= 2:
            site_choice[key] = (length, pid)

    wanted = list(dict.fromkeys([pid for (_s, _o, _l, _m, pid) in scan_lookup] +
                                [pid for _l, pid in site_choice.values()]))
    located, locate_seconds, locate_note = locate_patterns(
        [patterns[pid] for pid in wanted], anchor_limit)
    timings["locate_seconds"] = locate_seconds
    positions_of = {pid: located.get(index, []) for index, pid in enumerate(wanted)}

    scan_anchors: dict[int, list[Anchor]] = {i: [] for i in range(len(points))}
    for seed_id, (sample_index, off, length, mult, pid) in enumerate(scan_lookup):
        for corpus_pos in positions_of.get(pid, []):
            scan_anchors[sample_index].append(
                Anchor(query_pos=off, corpus_pos=corpus_pos, length=length,
                       seed_id=seed_id, multiplicity=mult))
    site_anchors: dict[tuple[int, int], list[Anchor]] = {}
    for (sample_index, depth), (length, pid) in site_choice.items():
        for corpus_pos in positions_of.get(pid, []):
            site_anchors.setdefault((sample_index, depth), []).append(
                Anchor(query_pos=BAND - length, corpus_pos=corpus_pos, length=length,
                       seed_id=-1, multiplicity=width_of.get(pid, 0)))

    # ---- stage 3: chain, take the correspondence, then read the fork at the site.
    cells = []
    fork_contexts: list[bytes] = []
    fork_labels: list[tuple[int, int]] = []
    for sample_index, p in enumerate(points):
        for depth in DEPTHS:
            floor = BAND - depth
            scan_in_band = [a for a in scan_anchors[sample_index] if a.query_pos >= floor]
            for mode in ("scan", "scan+site"):
                if mode == "scan":
                    anchors = scan_in_band
                    site_anchor = None
                    chosen_length = max((a.length for a in anchors), default=0)
                else:
                    site_anchor = site_anchors.get((sample_index, depth))
                    chosen_length = site_choice.get((sample_index, depth), (0, 0))[0]
                    anchors = scan_in_band + (site_anchor or [])
                chains = chain_anchors(anchors)
                strict = best_correspondence(chains, BAND, require_usable=True)
                loose = best_correspondence(chains, BAND, max_extrapolation=EXTRAPOLATION,
                                            require_usable=False)
                corr = strict or loose
                cell = {
                    "sample": sample_index,
                    "depth": depth,
                    "mode": mode,
                    "anchors": len(anchors),
                    "seeds": len({a.seed_id for a in anchors if a.seed_id >= 0}),
                    "site_seed_length": chosen_length,
                    "chains": len(chains),
                    "regions": len(cluster_regions(chains)) if chains else 0,
                    "best_score": chains[0].score if chains else 0.0,
                    "best_anchors": chains[0].n_anchors if chains else 0,
                    "best_chain_has_site_anchor": bool(site_anchor) and any(
                        a.seed_id < 0 for a in chains[0].anchors) if chains else False,
                    "site_level": None,
                    "kind": None,
                    "extrapolation": None,
                    "fork_width": None,
                }
                if corr is not None:
                    site = corr.corpus_pos
                    cell["site"] = site
                    cell["site_level"] = "abutting" if strict is not None else "extrapolated"
                    cell["kind"] = corr.kind
                    cell["extrapolation"] = corr.extrapolation_bytes
                    # On the pile panel the query text IS corpus text, so a site
                    # inside the query's own band is the source occurrence, not
                    # an occurrence found elsewhere. The transcript panel has no
                    # such occurrence to find.
                    cell["site_is_source"] = (source == "pile") and abs(site - p) <= BAND + 1
                    if site - depth >= 0:
                        # On the pile panel the source occurrence of the very same
                        # context is inside the index and contributes its own next
                        # byte to any histogram taken at a site whose context
                        # matches. That one occurrence is identified exactly and
                        # removed, so the fork is an elsewhere-only answer.
                        cell["fork_drops_source"] = (
                            source == "pile" and pile.read(site - depth, depth) == context(p, depth))
                        fork_contexts.append(pile.read(site - depth, depth))
                        fork_labels.append((len(cells), sample_index))
                cells.append(cell)

    histograms, branches_seconds = branch_histograms(fork_contexts)
    timings["branches_seconds"] = branches_seconds
    actuals = [next_byte(p) for p in points]
    majority = collections.Counter(actuals).most_common(1)[0][0]
    for (cell_index, sample_index), hist in zip(fork_labels, histograms):
        cell = cells[cell_index]
        histogram = dict(hist["hist"])
        width = hist["width"]
        actual = actuals[sample_index]
        cell["hit_marginal"] = actual == majority
        if cell.get("fork_drops_source") and histogram.get(actual):
            histogram[actual] -= 1
            width -= 1
            if histogram[actual] == 0:
                del histogram[actual]
        cell["fork_width"] = width
        cell["fork"] = fork_outcome(histogram, actual)

    result = {
        "source": loaded,
        "sampling": sampling,
        "positions": points,
        "cells": cells,
        "seeds": {
            "calls": timings["seeds_calls"],
            "seconds_total": timings["seeds_seconds"],
            "seconds_per_call": timings["seeds_seconds"] / max(1, timings["seeds_calls"]),
            "rows": sum(len(rows) for rows in scan_rows),
            "samples_without_seed": sum(1 for rows in scan_rows if not rows),
            "target_mult": TARGET_MULT,
            "band_bytes": BAND,
        },
        "support_ladder": ladder,
        "site_seeding": {
            "min_site_len": MIN_SITE_LEN,
            "positions_with_site_seed": len(site_choice),
            "positions_total": len(points) * len(DEPTHS),
            "chosen_lengths": percentiles([length for length, _pid in site_choice.values()]),
        },
        "locate": {
            "distinct_patterns_scanned": len(patterns),
            "distinct_patterns_located": len(wanted),
            "note": locate_note,
            "seconds_total": locate_seconds,
            "taken_positions": sum(len(v) for v in positions_of.values()),
            "limit_per_pattern": anchor_limit,
            "mult_parity_cells": parity,
            "mult_parity_cells_checked": parity_checked,
            "count_seconds": count_seconds,
        },
        "branches": {"contexts": len(fork_contexts), "seconds_total": branches_seconds,
                     "jobs": 8, "k": 256, "mode": "cross-doc"},
        "seed": seed,
    }
    return result


# ------------------------------------------------------------------- reporting
def percentiles(values):
    """Nearest-rank p50/p90/p99 and max (None when empty)."""
    clean = sorted(v for v in values)
    if not clean:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "max": None}
    def pick(q):
        index = max(0, min(len(clean) - 1, int(round(q * (len(clean) - 1)))))
        return clean[index]
    return {"n": len(clean), "p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99),
            "max": clean[-1]}


def _panel(rows):
    """One (source, mode) panel: per-depth distributions over sampled positions."""
    with_anchors = [c for c in rows if c["anchors"] > 0]
    with_site = [c for c in rows if c["site_level"]]
    strict = [c for c in with_site if c["site_level"] == "abutting"]
    # The pile panel's own-copy answer is not evidence from elsewhere: the
    # cross-site subset keeps only sites outside the query's own band, which is
    # the only subset comparable with the transcript panel.
    cross = [c for c in with_site if not c.get("site_is_source")]
    abutting_cross = [c for c in cross if c["site_level"] == "abutting"]
    forks = [c["fork"] for c in cross if c.get("fork")]
    abutting_forks = [c["fork"] for c in abutting_cross if c.get("fork")]
    return {
        "positions": len(rows),
        "anchors": percentiles([c["anchors"] for c in rows]),
        "chains": percentiles([c["chains"] for c in rows]),
        "regions": percentiles([c["regions"] for c in rows]),
        "best_score": percentiles([c["best_score"] for c in rows]),
        "site_seed_length": percentiles([c["site_seed_length"] for c in rows
                                         if c["site_seed_length"]]),
        "positions_with_anchors": len(with_anchors),
        "positions_with_site": len(with_site),
        "site_fraction": len(with_site) / max(1, len(rows)),
        "positions_abutting": len(strict),
        "abutting_fraction": len(strict) / max(1, len(rows)),
        "joint_chains": sum(1 for c in rows if c["best_anchors"] > 1),
        "extrapolation": percentiles([c["extrapolation"] for c in with_site
                                      if c["extrapolation"] is not None]),
        "positions_with_source_site": len(with_site) - len(cross),
        "cross_positions": len(cross),
        "cross_fraction": len(cross) / max(1, len(rows)),
        "fork_width": percentiles([c["fork_width"] for c in cross
                                   if c["fork_width"] is not None]),
        "fork_distinct_bytes": percentiles([f["distinct"] for f in forks]),
        "fork_present": sum(1 for f in forks if f["present"]),
        "fork_rank": percentiles([f["rank"] for f in forks if f["present"]]),
        "fork_mass": percentiles([f["mass"] for f in forks if f["present"]]),
        "fork_top1_mass": percentiles([f["top1_mass"] for f in forks]),
        "fork_mass_all": percentiles([f["mass"] for f in forks]),
        "fork_top1_is_actual": sum(1 for f in forks if f["present"] and f["rank"] == 1),
        "fork_dropped_source": sum(1 for c in cross if c.get("fork_drops_source")),
        "hit_marginal": sum(1 for c in rows if c.get("hit_marginal")),
        "hit_marginal_cross": sum(1 for c in cross if c.get("hit_marginal")),
        "abutting_cross": len(abutting_cross),
        "abutting_fork_present": sum(1 for f in abutting_forks if f["present"]),
        "abutting_fork_top1": sum(1 for f in abutting_forks if f["present"] and f["rank"] == 1),
        "abutting_fork_width": percentiles([c["fork_width"] for c in abutting_cross
                                            if c["fork_width"] is not None]),
        "abutting_fork_distinct": percentiles([f["distinct"] for f in abutting_forks]),
    }


def summary_of(result: dict) -> dict:
    cells = result["cells"]
    out = {"source": result["source"]["kind"], "cells": len(cells), "panels": {},
           "support_ladder": result["support_ladder"],
           "site_seeding": result["site_seeding"], "positions": len(result["positions"])}
    for mode in MODES:
        mode_cells = [c for c in cells if c["mode"] == mode]
        out["panels"][mode] = {str(depth): _panel([c for c in mode_cells if c["depth"] == depth])
                               for depth in DEPTHS}
    return out


def _fmt(block, keys):
    if block["n"] == 0:
        return "-"
    return "/".join(str(block[key]) for key in keys if block[key] is not None) or "-"


def render_ladder(reports: dict) -> str:
    """Support-depth ladder side by side: transcript vs pile, per suffix length."""
    lines = ["| suffix length L | transcript mult>=2 | transcript mult>=50 | "
             "pile mult>=2 | pile mult>=50 |", "|---:|---:|---:|---:|---:|"]
    for length in LADDER_LENGTHS:
        cells = []
        for name in ("transcript", "pile"):
            ladder = reports.get(name, {}).get("support_ladder", {}).get(str(length))
            if ladder is None:
                cells.extend(["-", "-"])
            else:
                cells.extend([f"{ladder['ge2']}/{ladder['n']} = {100*ladder['ge2']/max(1,ladder['n']):.1f}%",
                              f"{ladder['ge50']}/{ladder['n']}"])
        lines.append(f"| {length} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_markdown(reports: dict, results: dict) -> str:
    lines = []
    for name in ("transcript", "pile"):
        if name not in reports:
            continue
        summary = reports[name]
        lines.append(f"#### {name} source (n={summary['positions']} positions)\n")
        for mode in MODES:
            lines.append(f"**seeding mode `{mode}`**\n")
            lines.append("| depth | anchors p50/p90/p99/max | chains p50/p90/p99 | regions "
                         "p50/p90/p99 | best chain score p50/p90/p99 | joint chains "
                         "(>1 anchor) | abutting site | any site (<=16B gap) | cross-site sites |")
            lines.append("|---:|---|---|---|---|---:|---:|---:|---:|")
            for depth in DEPTHS:
                row = summary["panels"][mode][str(depth)]
                lines.append(
                    f"| {depth} | {_fmt(row['anchors'], ('p50','p90','p99','max'))} | "
                    f"{_fmt(row['chains'], ('p50','p90','p99'))} | "
                    f"{_fmt(row['regions'], ('p50','p90','p99'))} | "
                    f"{_fmt(row['best_score'], ('p50','p90','p99'))} | "
                    f"{row['joint_chains']}/{row['positions']} | "
                    f"{row['positions_abutting']}/{row['positions']} | "
                    f"{row['positions_with_site']}/{row['positions']} "
                    f"= {100*row['site_fraction']:.1f}% | {row['cross_positions']} |")
            lines.append("")
        lines.append("**fork at the aligned site (cross-site correspondences only; the source "
                     "occurrence removed where it is in the same context)**\n")
        lines.append("| mode | depth | sites | source occ. removed | fork width "
                     "p50/p90/p99/max | distinct next bytes p50/p90/p99 | actual byte present | "
                     "top-1 branch is the actual byte | rank (when present) p50/p90 | mass "
                     "(when present) p50/p90/p99 | top-1 branch mass p50/p90 | abutting only: "
                     "present | abutting only: top-1 | abutting only: fork width p50/p90/p99 | "
                     "chance: actual = stream majority |")
        lines.append("|---|---:|---:|---:|---|---|---:|---:|---|---|---|---:|---:|---|")
        for mode in MODES:
            for depth in DEPTHS:
                row = summary["panels"][mode][str(depth)]
                lines.append(
                    f"| {mode} | {depth} | {row['cross_positions']} | "
                    f"{row['fork_dropped_source']} | "
                    f"{_fmt(row['fork_width'], ('p50','p90','p99','max'))} | "
                    f"{_fmt(row['fork_distinct_bytes'], ('p50','p90','p99'))} | "
                    f"{row['fork_present']}/{row['cross_positions']} | "
                    f"{row['fork_top1_is_actual']}/{row['cross_positions']} | "
                    f"{_fmt(row['fork_rank'], ('p50','p90'))} | "
                    f"{_fmt(row['fork_mass'], ('p50','p90','p99'))} | "
                    f"{_fmt(row['fork_top1_mass'], ('p50','p90'))} | "
                    f"{row['abutting_fork_present']}/{row['abutting_cross']} | "
                    f"{row['abutting_fork_top1']}/{row['abutting_cross']} | "
                    f"{_fmt(row['abutting_fork_width'], ('p50','p90','p99'))} | "
                    f"{row['hit_marginal_cross']}/{row['cross_positions']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    meas = sub.add_parser("measure")
    meas.add_argument("--source", choices=("transcript", "pile"), required=True)
    meas.add_argument("--positions", type=int, default=150)
    meas.add_argument("--seed", type=int, default=20261113)
    meas.add_argument("--limit", type=int, default=ANCHOR_LIMIT,
                      help="positions located per seed (xsa seeds' default is 8)")
    meas.add_argument("--out", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--transcript", required=True)
    rep.add_argument("--pile", required=True)
    args = parser.parse_args(argv)

    if args.command == "measure":
        result = measure(args.source, args.positions, args.seed, args.limit)
        result["summary"] = summary_of(result)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print(json.dumps({k: v for k, v in result.items()
                          if k in ("seeds", "locate", "branches", "sampling")}, indent=2))
    else:
        loaded, reports = {}, {}
        for name in ("transcript", "pile"):
            with open(getattr(args, name)) as handle:
                loaded[name] = json.load(handle)
            # Recomputed from the stored per-cell rows, so a report can be
            # re-derived without re-running any measurement.
            reports[name] = summary_of(loaded[name])
        print(render_ladder(reports))
        print(render_markdown(reports, loaded))
    return 0


if __name__ == "__main__":
    sys.exit(main())
