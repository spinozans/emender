#!/usr/bin/env python3
"""Conservative retention guard for the live E97/Emender DiLoCo run.

The default mode is a dry-run inventory/plan.  Deletion requires --delete and
only removes checkpoint_step_*.pt files that are not selected by the keep policy,
are older than the newest checkpoint, are older than --min-age-seconds, and have
a stable complete-checkpoint size.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


CHECKPOINT_RE = re.compile(
    r"checkpoint_step_(?P<step>\d+)_loss_(?P<loss>[-+0-9.eE]+)\.pt$"
)


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    path: Path
    run_dir: Path
    step: int
    loss: float | None
    size: int
    mtime: float

    @property
    def name(self) -> str:
        return self.path.name


@dataclasses.dataclass(frozen=True)
class Plan:
    checkpoints: list[Checkpoint]
    keep: dict[Path, set[str]]
    delete: list[Checkpoint]
    skip: dict[Path, str]
    complete_size: int | None
    newest: Checkpoint | None
    free_before: int
    total_before: int
    used_before: int


def parse_checkpoint(path: Path) -> tuple[int, float | None] | None:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        return None
    try:
        loss = float(match.group("loss"))
    except ValueError:
        loss = None
    return int(match.group("step")), loss


def discover_checkpoints(root: Path) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    for path in sorted(root.glob("runs/levelE97_100m_*/checkpoint_step_*.pt")):
        parsed = parse_checkpoint(path)
        if parsed is None:
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        step, loss = parsed
        checkpoints.append(
            Checkpoint(
                path=path,
                run_dir=path.parent,
                step=step,
                loss=loss,
                size=stat.st_size,
                mtime=stat.st_mtime,
            )
        )
    return sorted(checkpoints, key=lambda ckpt: (ckpt.run_dir.name, ckpt.step, ckpt.name))


def checkpoint_sort_key(ckpt: Checkpoint) -> tuple[int, float, str]:
    return ckpt.step, ckpt.mtime, ckpt.name


def resolve_resume_paths(root: Path) -> set[Path]:
    paths: set[Path] = set()
    for json_path in [root / "launch_manifest.json", *sorted((root / "runs").glob("*/args.json"))]:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        resume = data.get("resume")
        if isinstance(resume, str) and resume:
            paths.add(Path(resume))
        cmd = data.get("cmd")
        if isinstance(cmd, list):
            for idx, token in enumerate(cmd[:-1]):
                if token == "--resume" and isinstance(cmd[idx + 1], str):
                    paths.add(Path(cmd[idx + 1]))
    return paths


def complete_checkpoint_size(checkpoints: list[Checkpoint]) -> int | None:
    if not checkpoints:
        return None
    counts = Counter(ckpt.size for ckpt in checkpoints)
    size, count = counts.most_common(1)[0]
    # A singleton odd size is not enough evidence for completeness.
    if count < 2 and len(checkpoints) > 1:
        return None
    return size


def build_plan(
    root: Path,
    *,
    latest_active: int,
    milestone_every: int,
    critical_steps: set[int],
    min_age_seconds: int,
    extra_keep_paths: set[Path],
    now: float | None = None,
) -> Plan:
    now = time.time() if now is None else now
    usage = shutil.disk_usage(root)
    checkpoints = discover_checkpoints(root)
    keep: dict[Path, set[str]] = defaultdict(set)
    skip: dict[Path, str] = {}
    complete_size = complete_checkpoint_size(checkpoints)
    newest = max(checkpoints, key=lambda ckpt: (ckpt.mtime, ckpt.step, ckpt.name), default=None)

    by_run: dict[Path, list[Checkpoint]] = defaultdict(list)
    for ckpt in checkpoints:
        by_run[ckpt.run_dir].append(ckpt)

    for run_dir, run_ckpts in by_run.items():
        ordered = sorted(run_ckpts, key=checkpoint_sort_key)
        if ordered:
            keep[ordered[0].path].add("segment-first")
            keep[ordered[-1].path].add("segment-latest")

    if newest is not None:
        keep[newest.path].add("global-latest")
        active = sorted(by_run[newest.run_dir], key=checkpoint_sort_key)
        for ckpt in active[-latest_active:]:
            keep[ckpt.path].add(f"latest-{latest_active}-active")

    for ckpt in checkpoints:
        if ckpt.step in critical_steps:
            keep[ckpt.path].add("resume-proof-critical")
        if milestone_every > 0 and ckpt.step % milestone_every == 0:
            keep[ckpt.path].add(f"{milestone_every}-step-milestone")
        if ckpt.path in extra_keep_paths:
            keep[ckpt.path].add("resume-config")
        try:
            if (ckpt.run_dir / "latest.pt").resolve(strict=True) == ckpt.path:
                keep[ckpt.path].add("latest-symlink-target")
        except FileNotFoundError:
            pass

    delete: list[Checkpoint] = []
    newest_path = newest.path if newest is not None else None
    newest_mtime = newest.mtime if newest is not None else None
    for ckpt in checkpoints:
        if keep.get(ckpt.path):
            continue
        if newest_path is not None and ckpt.path == newest_path:
            skip[ckpt.path] = "newest-checkpoint"
            continue
        if newest_mtime is not None and ckpt.mtime >= newest_mtime:
            skip[ckpt.path] = "not-older-than-newest"
            continue
        age = now - ckpt.mtime
        if age < min_age_seconds:
            skip[ckpt.path] = f"younger-than-min-age:{int(age)}s"
            continue
        if complete_size is None or ckpt.size != complete_size:
            skip[ckpt.path] = "non-modal-or-unknown-complete-size"
            continue
        delete.append(ckpt)

    return Plan(
        checkpoints=checkpoints,
        keep=dict(keep),
        delete=sorted(delete, key=lambda ckpt: (ckpt.run_dir.name, ckpt.step, ckpt.name)),
        skip=skip,
        complete_size=complete_size,
        newest=newest,
        free_before=usage.free,
        total_before=usage.total,
        used_before=usage.used,
    )


def fmt_bytes(value: int) -> str:
    return f"{value:,} bytes ({value / 1_000_000_000:.2f} GB / {value / 1024**3:.2f} GiB)"


def fmt_time(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat()


def estimate_interval_seconds(checkpoints: list[Checkpoint], run_dir: Path | None) -> float | None:
    if run_dir is None:
        return None
    active = sorted(
        [ckpt for ckpt in checkpoints if ckpt.run_dir == run_dir],
        key=checkpoint_sort_key,
    )
    if len(active) < 2:
        return None
    intervals = [
        right.mtime - left.mtime
        for left, right in zip(active, active[1:])
        if right.mtime > left.mtime
    ]
    if not intervals:
        return None
    return sum(intervals) / len(intervals)


def write_report(plan: Plan, root: Path, deleted: list[Checkpoint], report_path: Path) -> None:
    bytes_planned = sum(ckpt.size for ckpt in plan.delete)
    bytes_deleted = sum(ckpt.size for ckpt in deleted)
    usage_after = shutil.disk_usage(root)
    active_dir = plan.newest.run_dir if plan.newest else None
    interval = estimate_interval_seconds(plan.checkpoints, active_dir)
    rate = None
    if interval and plan.complete_size:
        rate = plan.complete_size / interval
    remaining_dense_seconds = usage_after.free / rate if rate else None

    lines: list[str] = []
    lines.append("# E97 Checkpoint Retention Guard Report")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}")
    lines.append(f"Root: `{root}`")
    lines.append("")
    lines.append("## Filesystem")
    lines.append("")
    lines.append(f"- Before free: {fmt_bytes(plan.free_before)}")
    lines.append(f"- After free: {fmt_bytes(usage_after.free)}")
    lines.append(f"- Before used: {fmt_bytes(plan.used_before)}")
    lines.append(f"- After used: {fmt_bytes(usage_after.used)}")
    lines.append(f"- Planned delete bytes: {fmt_bytes(bytes_planned)}")
    lines.append(f"- Deleted bytes: {fmt_bytes(bytes_deleted)}")
    lines.append("")
    lines.append("## Inventory")
    lines.append("")
    lines.append(f"- Checkpoint count before: {len(plan.checkpoints)}")
    lines.append(f"- Keep count: {len(plan.keep)}")
    lines.append(f"- Planned delete count: {len(plan.delete)}")
    lines.append(f"- Deleted count: {len(deleted)}")
    lines.append(f"- Complete checkpoint size: {fmt_bytes(plan.complete_size or 0)}")
    if plan.newest:
        lines.append(
            f"- Newest checkpoint: `{plan.newest.path}` "
            f"mtime={fmt_time(plan.newest.mtime)}"
        )
    if interval and plan.complete_size:
        lines.append(f"- Observed active checkpoint interval: {interval:.1f} seconds")
        lines.append(f"- Dense growth rate: {rate * 3600 / 1_000_000_000:.2f} GB/hour")
        if remaining_dense_seconds:
            lines.append(
                f"- Time to full at dense cadence after pruning: "
                f"{remaining_dense_seconds / 3600:.1f} hours"
            )
    lines.append("")
    lines.append("## Keep List")
    lines.append("")
    for path in sorted(plan.keep, key=lambda p: str(p)):
        reasons = ", ".join(sorted(plan.keep[path]))
        lines.append(f"- `{path}` - {reasons}")
    lines.append("")
    lines.append("## Delete List")
    lines.append("")
    for ckpt in plan.delete:
        marker = "deleted" if ckpt in deleted else "planned"
        lines.append(
            f"- {marker}: `{ckpt.path}` size={ckpt.size} "
            f"mtime={fmt_time(ckpt.mtime)}"
        )
    lines.append("")
    lines.append("## Safety")
    lines.append("")
    lines.append("- The training process was not stopped, killed, restarted, or reconfigured.")
    lines.append("- Only `checkpoint_step_*.pt` files from the computed redundant set were eligible.")
    lines.append("- The newest checkpoint and `latest.pt` target were kept.")
    lines.append("- Files with non-modal size, too-new mtime, or not older than newest were skipped.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_plan(plan: Plan, *, root: Path, delete_mode: bool) -> None:
    action = "DELETE" if delete_mode else "DRY-RUN"
    print(f"mode={action}")
    print(f"root={root}")
    print(f"checkpoint_count={len(plan.checkpoints)}")
    print(f"free_before={plan.free_before}")
    print(f"complete_size={plan.complete_size}")
    if plan.newest:
        print(f"newest={plan.newest.path}")
    print(f"keep_count={len(plan.keep)}")
    for path in sorted(plan.keep, key=lambda p: str(p)):
        print(f"KEEP {path} reasons={','.join(sorted(plan.keep[path]))}")
    print(f"delete_count={len(plan.delete)}")
    for ckpt in plan.delete:
        print(f"DELETE {ckpt.path} size={ckpt.size} mtime={fmt_time(ckpt.mtime)}")
    for path, reason in sorted(plan.skip.items(), key=lambda item: str(item[0])):
        print(f"SKIP {path} reason={reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/nvme1n1/erikg/diloco_8gpu/emender"),
    )
    parser.add_argument("--latest-active", type=int, default=5)
    parser.add_argument("--milestone-every", type=int, default=10_000)
    parser.add_argument("--critical-step", type=int, action="append", default=[500, 72_000, 72_500])
    parser.add_argument("--keep-path", type=Path, action="append", default=[])
    parser.add_argument("--min-age-seconds", type=int, default=1_800)
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")

    extra_keep_paths = {path.resolve() for path in args.keep_path}
    extra_keep_paths.update(path.resolve() for path in resolve_resume_paths(root))

    plan = build_plan(
        root,
        latest_active=args.latest_active,
        milestone_every=args.milestone_every,
        critical_steps=set(args.critical_step),
        min_age_seconds=args.min_age_seconds,
        extra_keep_paths=extra_keep_paths,
    )
    print_plan(plan, root=root, delete_mode=args.delete)

    deleted: list[Checkpoint] = []
    if args.delete:
        for ckpt in plan.delete:
            try:
                current = ckpt.path.stat()
            except FileNotFoundError:
                continue
            if current.st_size != ckpt.size or current.st_mtime != ckpt.mtime:
                print(f"SKIP_CHANGED {ckpt.path}", file=sys.stderr)
                continue
            ckpt.path.unlink()
            deleted.append(ckpt)
            print(f"DELETED {ckpt.path}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_report(plan, root, deleted, args.report)
        print(f"report={args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
