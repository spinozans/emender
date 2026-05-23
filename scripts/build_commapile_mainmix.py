#!/usr/bin/env python3
"""Build a Comma/Common-Pile main-mix training stream.

The output format intentionally matches the existing commapile.txt format:
UTF-8 document text with ASCII record separator (0x1e) between documents.

The builder is quota-based. It uses the Comma v0.1 main-stage effective token
mix as source probabilities, converts those probabilities to byte quotas for
the final mmap-friendly training file, then writes documents until each source
quota is reached. This makes source exposure correct for the current random
byte-window loader.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple


# Effective token counts from the Comma v0.1 README main-stage table, in B toks.
COMMA_MAIN_EFFECTIVE_TOKENS_B: Dict[str, float] = {
    "arxiv_abstracts": 3.4,
    "arxiv_papers": 35.8,
    "biodiversity_heritage_library": 2.5,
    "caselaw_access_project": 19.7,
    "cccc": 91.4,
    "data_provenance_initiative": 5.5,
    "doab": 18.2,
    "foodista": 0.15,
    "github_archive": 66.1,
    "library_of_congress": 2.4,
    "libretexts": 0.56,
    "news": 0.38,
    "oercommons": 0.07,
    "peS2o": 260.0,
    "pre_1929_books": 12.4,
    "pressbooks": 0.86,
    "project_gutenberg": 5.7,
    "public_domain_review": 0.010,
    "pubmed": 36.6,
    "python_enhancement_proposals": 0.016,
    "regulations": 8.2,
    "stackexchange": 143.2,
    "stackv2_edu": 135.5,
    "stackv2_html": 2.5,
    "ubuntu_irc": 11.1,
    "uk_hansard": 14.0,
    "usgpo": 2.2,
    "uspto": 39.4,
    "wikimedia": 94.7,
    "wikiteam": 17.2,
    "youtube": 4.7,
}


SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp]?i?b?|[kmgtp])?\s*$", re.I)


def parse_size(text: str) -> int:
    m = SIZE_RE.match(text)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid size: {text!r}")
    value = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    powers_1000 = {"": 1, "b": 1, "k": 10**3, "kb": 10**3, "m": 10**6, "mb": 10**6,
                   "g": 10**9, "gb": 10**9, "t": 10**12, "tb": 10**12,
                   "p": 10**15, "pb": 10**15}
    powers_1024 = {"kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40,
                   "pib": 2**50}
    if suffix in powers_1024:
        return int(value * powers_1024[suffix])
    if suffix in powers_1000:
        return int(value * powers_1000[suffix])
    raise argparse.ArgumentTypeError(f"invalid size suffix in {text!r}")


def parse_separator(text: str) -> bytes:
    return bytes(text, "utf-8").decode("unicode_escape").encode("utf-8")


def stable_seed(base_seed: int, name: str, extra: int = 0) -> int:
    h = hashlib.blake2b(f"{base_seed}:{name}:{extra}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little")


def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1000 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1000.0
    return f"{n}B"


@dataclass
class SourceStats:
    target_bytes: int
    written_bytes: int = 0
    records: int = 0
    sampled_payload_bytes: int = 0
    sampled_records: int = 0
    epochs_started: int = 0
    embedded_separators_replaced: int = 0
    json_errors: int = 0
    read_errors: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.target_bytes - self.written_bytes)

    @property
    def avg_payload_bytes(self) -> float:
        if self.sampled_records:
            return self.sampled_payload_bytes / self.sampled_records
        if self.records:
            return self.written_bytes / self.records
        return 4096.0


@dataclass
class TextRecord:
    data: bytes
    replaced_separators: int = 0


@dataclass
class SourceReader:
    root: Path
    source: str
    seed: int
    delimiter: bytes
    buffer_records_limit: int
    buffer_bytes_limit: int
    stats: SourceStats
    files: List[Path] = field(init=False)
    rng: random.Random = field(init=False)
    epoch: int = 0
    file_order: List[Path] = field(default_factory=list)
    file_index: int = 0
    handle: Optional[gzip.GzipFile] = None
    buffer: List[TextRecord] = field(default_factory=list)
    buffer_bytes: int = 0

    def __post_init__(self) -> None:
        self.files = sorted((self.root / self.source).glob("*.jsonl.gz"))
        if not self.files:
            raise FileNotFoundError(f"no jsonl.gz shards for source {self.source}")
        self.rng = random.Random(stable_seed(self.seed, self.source))
        self._start_epoch()

    def _start_epoch(self) -> None:
        self.epoch += 1
        self.stats.epochs_started = self.epoch
        order_rng = random.Random(stable_seed(self.seed, self.source, self.epoch))
        self.file_order = list(self.files)
        order_rng.shuffle(self.file_order)
        self.file_index = 0
        self._close_handle()

    def _close_handle(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _open_next_file(self) -> bool:
        self._close_handle()
        if self.file_index >= len(self.file_order):
            self._start_epoch()
        path = self.file_order[self.file_index]
        self.file_index += 1
        try:
            self.handle = gzip.open(path, "rb")
            return True
        except OSError:
            self.stats.read_errors += 1
            self.handle = None
            return False

    def _read_next_record(self) -> Optional[TextRecord]:
        while True:
            if self.handle is None and not self._open_next_file():
                continue
            assert self.handle is not None
            line = self.handle.readline()
            if not line:
                self._close_handle()
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                self.stats.json_errors += 1
                continue
            text = obj.get("text")
            if not isinstance(text, str) or not text:
                continue
            data = text.encode("utf-8", errors="replace")
            replaced = data.count(self.delimiter)
            if replaced:
                data = data.replace(self.delimiter, b" ")
            self.stats.sampled_payload_bytes += len(data)
            self.stats.sampled_records += 1
            return TextRecord(data=data, replaced_separators=replaced)

    def fill_buffer(self) -> None:
        while (
            len(self.buffer) < self.buffer_records_limit
            and self.buffer_bytes < self.buffer_bytes_limit
        ):
            rec = self._read_next_record()
            if rec is None:
                return
            self.buffer.append(rec)
            self.buffer_bytes += len(rec.data)

    def pop_random(self) -> TextRecord:
        if not self.buffer:
            self.fill_buffer()
        if not self.buffer:
            raise RuntimeError(f"source {self.source} produced no records")
        idx = self.rng.randrange(len(self.buffer))
        rec = self.buffer[idx]
        last = self.buffer.pop()
        if idx < len(self.buffer):
            self.buffer[idx] = last
        self.buffer_bytes -= len(rec.data)
        if self.buffer_bytes < self.buffer_bytes_limit // 2:
            self.fill_buffer()
        return rec

    def close(self) -> None:
        self._close_handle()


def source_targets(target_bytes: int) -> Dict[str, int]:
    total = sum(COMMA_MAIN_EFFECTIVE_TOKENS_B.values())
    targets = {
        source: int(round(target_bytes * weight / total))
        for source, weight in COMMA_MAIN_EFFECTIVE_TOKENS_B.items()
    }
    delta = target_bytes - sum(targets.values())
    if delta:
        largest = max(targets, key=targets.get)
        targets[largest] += delta
    return targets


def choose_source(
    rng: random.Random,
    stats: Dict[str, SourceStats],
    total_written: int,
    target_total: int,
) -> Optional[str]:
    active: List[Tuple[str, int]] = [
        (source, st.remaining) for source, st in stats.items() if st.remaining > 0
    ]
    if not active:
        return None

    # Deficit scheduling keeps every prefix of the output close to the target
    # byte mixture. Dividing by average payload size avoids over-selecting
    # sources with very large documents.
    active_stats = [stats[source] for source, _ in active]
    global_avg = sum(st.avg_payload_bytes for st in active_stats) / len(active_stats)
    lookahead_total = total_written + max(1, int(global_avg))
    weighted: List[Tuple[str, float]] = []
    for source, _ in active:
        st = stats[source]
        target_share = st.target_bytes / target_total
        deficit = target_share * lookahead_total - st.written_bytes
        if deficit > 0:
            weighted.append((source, deficit / max(1.0, st.avg_payload_bytes)))

    if not weighted:
        weighted = [
            (source, remaining / max(1.0, stats[source].avg_payload_bytes))
            for source, remaining in active
        ]

    total = sum(weight for _, weight in weighted)
    draw = rng.random() * total
    acc = 0
    for source, weight in weighted:
        acc += weight
        if draw < acc:
            return source
    return weighted[-1][0]


def write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def build(args: argparse.Namespace) -> None:
    root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} exists; pass --force to overwrite")
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists() and not args.force:
        raise FileExistsError(f"{partial} exists; remove it or pass --force")

    delimiter = parse_separator(args.separator)
    if delimiter != b"\x1e":
        print(f"warning: non-standard delimiter {delimiter!r}", file=sys.stderr)

    targets = source_targets(args.target_bytes)
    stats = {
        source: SourceStats(target_bytes=target)
        for source, target in targets.items()
        if target > 0
    }
    missing = [source for source in stats if not (root / source).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing source directories: {', '.join(missing)}")

    rng = random.Random(args.seed)
    readers = {
        source: SourceReader(
            root=root,
            source=source,
            seed=args.seed,
            delimiter=delimiter,
            buffer_records_limit=args.buffer_records,
            buffer_bytes_limit=args.buffer_bytes,
            stats=stats[source],
        )
        for source in stats
    }
    for reader in readers.values():
        reader.fill_buffer()

    progress_path = output.with_suffix(output.suffix + ".progress.jsonl")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    start_time = time.time()
    total_written = 0
    total_records = 0
    next_log = args.log_every_bytes
    last_log_time = start_time
    sha = hashlib.sha256()

    print(f"input_root={root}")
    print(f"output={output}")
    print(f"partial={partial}")
    print(f"target_bytes={args.target_bytes} ({fmt_bytes(args.target_bytes)})")
    print(f"delimiter={delimiter!r}")
    print(f"sources={len(stats)}")

    with partial.open("wb", buffering=1024 * 1024) as out, progress_path.open("w") as progress:
        while True:
            source = choose_source(rng, stats, total_written, sum(st.target_bytes for st in stats.values()))
            if source is None:
                break
            rec = readers[source].pop_random()
            prefix = b"" if total_records == 0 else delimiter
            payload = prefix + rec.data
            out.write(payload)
            sha.update(payload)

            n = len(payload)
            total_written += n
            total_records += 1
            st = stats[source]
            st.written_bytes += n
            st.records += 1
            st.embedded_separators_replaced += rec.replaced_separators

            if total_written >= next_log:
                now = time.time()
                elapsed = max(1e-9, now - start_time)
                interval = max(1e-9, now - last_log_time)
                done = total_written / max(1, args.target_bytes)
                row = {
                    "time": now,
                    "elapsed_s": elapsed,
                    "total_written": total_written,
                    "target_bytes": args.target_bytes,
                    "done": done,
                    "records": total_records,
                    "mb_s_overall": total_written / elapsed / 1e6,
                    "mb_s_interval": args.log_every_bytes / interval / 1e6,
                    "source_written_bytes": {k: v.written_bytes for k, v in stats.items()},
                    "source_records": {k: v.records for k, v in stats.items()},
                    "source_sampled_records": {k: v.sampled_records for k, v in stats.items()},
                }
                progress.write(json.dumps(row, sort_keys=True) + "\n")
                progress.flush()
                print(
                    f"{fmt_bytes(total_written)} / {fmt_bytes(args.target_bytes)} "
                    f"({done:.2%}), records={total_records:,}, "
                    f"{row['mb_s_overall']:.1f} MB/s overall",
                    flush=True,
                )
                last_log_time = now
                next_log += args.log_every_bytes

    for reader in readers.values():
        reader.close()

    partial.replace(output)
    elapsed = time.time() - start_time
    total_target = sum(st.target_bytes for st in stats.values())
    total_actual = sum(st.written_bytes for st in stats.values())
    manifest = {
        "created_unix": time.time(),
        "elapsed_s": elapsed,
        "input_root": str(root),
        "output": str(output),
        "target_bytes_requested": args.target_bytes,
        "target_bytes_sum": total_target,
        "actual_bytes_sum": total_actual,
        "actual_file_size": output.stat().st_size,
        "sha256": sha.hexdigest(),
        "delimiter_hex": delimiter.hex(),
        "seed": args.seed,
        "buffer_records": args.buffer_records,
        "buffer_bytes": args.buffer_bytes,
        "mixture": "comma_v0.1_main_stage_effective_tokens",
        "mixture_effective_tokens_b": COMMA_MAIN_EFFECTIVE_TOKENS_B,
        "sources": {
            source: {
                "target_bytes": st.target_bytes,
                "actual_bytes": st.written_bytes,
                "target_share": st.target_bytes / total_target,
                "actual_share": st.written_bytes / total_actual,
                "records": st.records,
                "sampled_records": st.sampled_records,
                "avg_payload_bytes": st.avg_payload_bytes,
                "epochs_started": st.epochs_started,
                "embedded_separators_replaced": st.embedded_separators_replaced,
                "json_errors": st.json_errors,
                "read_errors": st.read_errors,
            }
            for source, st in sorted(stats.items())
        },
    }
    write_json(manifest_path, manifest)
    print(f"completed {output}")
    print(f"actual_size={fmt_bytes(output.stat().st_size)} elapsed_h={elapsed/3600:.2f}")
    print(f"sha256={sha.hexdigest()}")
    print(f"manifest={manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", help="Comma v0.1 training dataset root")
    parser.add_argument("--output", required=True, help="Output .txt path")
    parser.add_argument("--target-bytes", type=parse_size, default=parse_size("1TB"))
    parser.add_argument("--separator", default="\\x1e", help="Document delimiter; default ASCII RS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-records", type=int, default=4096)
    parser.add_argument("--buffer-bytes", type=parse_size, default=parse_size("16MiB"))
    parser.add_argument("--log-every-bytes", type=parse_size, default=parse_size("10GB"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
