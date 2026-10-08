#!/usr/bin/env python3
"""Digest-verified policy-gradient trajectory batch builder (e97-rl-loop-v1 v2).

The policy-gradient train step (rl_policy_gradient_step.py) trains on the
POLICY'S OWN emitted assistant turns with the sealed-validator outcome as the
reward — which means the batch must include FAILED on-policy attempts, not
just the verified receipts (receipts exist only for passed episodes; the
failed attempts are the reward contrast).  Every row is derived from the
immutable per-cycle attempt evidence the loop already publishes:

  <episodes-root>/episodes/cycle-NNNN/<task>/attempt/episode-private.json
  <episodes-root>/episodes/cycle-NNNN/<task>/attempt/grade.json     (stage=policy)

and is verified by RE-ENCODING the episode with the same encode_candidate
machinery the receipts and the pack builder use (token counts, target counts,
assistant-span token boundaries).  Split isolation is honored: only
train-split (receipt-eligible) tasks are ever batched; development-split
attempts are skipped.  Groups are keyed by the LAKE TASK ID (the trailing
hyphen field of the round-scoped pool id), so same-task attempts across pool
rounds/cycles form the GRPO group the candidate's group_advantages requires.

The manifest is a single canonical-JSON artifact whose sha256 binds every row
(episode/grade file digests, token ids, spans, rewards, group stats).  The
train step verifies the manifest digest before loading anything heavy.

NO SQLite anywhere (no-database amendment); filesystem evidence only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from ndm.e97_onpolicy_records import canonical_json

BATCH_SCHEMA = "emender-rl-loop-pg-batch-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lake_task_id(task_id: str) -> str:
    """The lake task id under round-scoped pool ids (bank-r0350-<lake-id>)."""
    return task_id.rsplit("-", 1)[-1]


def _episode_turn_rows(record: Mapping[str, Any], enc) -> list[dict[str, Any]]:
    """Per-turn rows for one attempt episode, verified by full re-encode.

    Mirrors encode_candidate's span-finding logic to locate each assistant
    turn's token span, then asserts the spans reproduce the encode_candidate
    assistant mask exactly (fail closed on any drift).
    """
    from scripts.build_e97_pi_native_curriculum import encode_candidate

    generations = [dict(item) for item in record["generations"]]
    valid_turns = [item for item in generations if item.get("reason") == "valid"]
    if not valid_turns:
        return []
    text = record["native_record"]
    ids, mask, supervised = encode_candidate(text, generations, 0, enc)
    raw = text.encode()
    bound = {0: 0}
    position = 0
    for index, token in enumerate(ids, 1):
        position += len(enc.decode_single_token_bytes(token))
        bound[position] = index
    rows = []
    cursor = 0
    for item in generations:
        turn_text = enc.decode(item["token_ids"])
        needle = ("\n\nAssistant:\n" + turn_text).encode()
        start = raw.find(needle, cursor)
        if start < 0:
            raise ValueError("generated turn absent from record")
        left = start + len("\n\nAssistant:\n".encode())
        right = start + len(needle)
        cursor = right
        if left not in bound or right not in bound:
            raise ValueError("assistant span token boundary")
        if item.get("reason") == "valid":
            rows.append({
                "turn": int(item["turn"]),
                "prefix": ids[:bound[left]],
                "generated": ids[bound[left]:bound[right]],
                "span": [bound[left], bound[right]],
            })
    if not all(all(mask[row["span"][0]:row["span"][1]]) for row in rows):
        raise ValueError("turn spans do not reproduce the assistant mask")
    return rows


def collect_batch(episodes_root: Path, cycles: list[int], *,
                   min_group: int = 2, group_by: str = "task",
                   max_episodes: int = 0) -> dict[str, Any]:
    """Walk the given cycles' attempt evidence; build the batch manifest body.

    group_by="task" groups same-task attempts by lake task id (the original
    GRPO semantics — for round-refreshed tasks that repeat across rounds).
    group_by="family" groups by the sealed template family — the honest
    grouping when the pool serves ONE-SHOT sealed instances (the lakeexp
    tranches: every task id appears exactly once, so per-task groups are
    singletons and the candidate fails closed on them); instances of one
    sealed family share the task structure and the sealed validator
    program, and the pass/fail contrast across instances is the reward
    signal.  max_episodes keeps only the MOST RECENT episodes (the freshest
    lineages) and records the cut honestly.
    """
    if group_by not in ("task", "family"):
        raise SystemExit("group_by must be task or family")
    import tiktoken

    enc = tiktoken.get_encoding("p50k_base")
    episodes_dir = episodes_root / "episodes"
    if not episodes_dir.is_dir():
        raise SystemExit(f"no episodes tree under {episodes_root}")
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cycle in sorted(cycles):
        cycle_dir = episodes_dir / f"cycle-{cycle:04d}"
        if not cycle_dir.is_dir():
            skipped.append({"cycle": cycle, "reason": "missing cycle dir"})
            continue
        summary_path = cycle_dir / "collect-summary.json"
        if not summary_path.is_file():
            skipped.append({"cycle": cycle, "reason": "missing collect summary"})
            continue
        summary = json.loads(summary_path.read_text())
        policy_sha = summary.get("policy_checkpoint_sha256")
        split_by_task = {}
        for outcome in summary.get("outcomes", []):
            split_by_task[outcome["task_id"]] = outcome
        for task_dir in sorted(p for p in cycle_dir.iterdir() if p.is_dir()):
            task_id = task_dir.name
            outcome = split_by_task.get(task_id)
            if outcome is None:
                skipped.append({"cycle": cycle, "task": task_id,
                                "reason": "task absent from collect summary"})
                continue
            if not outcome.get("receipt_eligible", False):
                continue  # sealed train/development split isolation
            attempt = task_dir / "attempt"
            episode_path = attempt / "episode-private.json"
            grade_path = attempt / "grade.json"
            if not episode_path.is_file() or not grade_path.is_file():
                skipped.append({"cycle": cycle, "task": task_id,
                                "reason": "missing attempt evidence"})
                continue
            grade = json.loads(grade_path.read_text())
            if grade.get("stage") != "policy":
                raise SystemExit(f"attempt grade is not stage=policy: {grade_path}")
            record = json.loads(episode_path.read_text())
            try:
                rows = _episode_turn_rows(record, enc)
            except ValueError as exc:
                skipped.append({"cycle": cycle, "task": task_id,
                                "reason": f"re-encode verification failed: {exc}"})
                continue
            if not rows:
                skipped.append({"cycle": cycle, "task": task_id,
                                "reason": "no valid assistant turns"})
                continue
            entries.append({
                "cycle": cycle,
                "task_id": task_id,
                "group_id": (outcome.get("template") or lake_task_id(task_id))
                if group_by == "family" else lake_task_id(task_id),
                "group_by": group_by,
                "template": outcome.get("template"),
                "split": outcome.get("split", "train"),
                "reward": 1.0 if bool(grade["passed"]) else 0.0,
                "grade_passed": bool(grade["passed"]),
                "episode_status": record.get("status"),
                "episode_sha256": sha256_file(episode_path),
                "grade_sha256": sha256_file(grade_path),
                "policy_checkpoint_sha256": policy_sha,
                "episode_tokens": _episode_token_count(record, rows, enc),
                "turns": rows,
            })
    if max_episodes and len(entries) > max_episodes:
        dropped = len(entries) - max_episodes
        entries = entries[-max_episodes:]
        skipped.append({"reason": f"recency cap: {dropped} older episodes "
                                   f"dropped (max-episodes={max_episodes})"})
    groups: dict[str, dict[str, Any]] = {}
    for entry in entries:
        group = groups.setdefault(entry["group_id"], {"episodes": 0, "passes": 0})
        group["episodes"] += 1
        group["passes"] += int(entry["reward"])
    for group in groups.values():
        fails = group["episodes"] - group["passes"]
        group["fails"] = fails
        group["variance"] = bool(group["passes"] and fails)
    contrastive = sorted(gid for gid, group in groups.items()
                         if group["variance"] and group["episodes"] >= min_group)
    return {
        "schema": BATCH_SCHEMA,
        "created_unix": time.time(),
        "episodes_root": str(episodes_root),
        "encoder": "p50k_base",
        "cycles": sorted(cycles),
        "min_group": int(min_group),
        "group_by": group_by,
        "max_episodes": int(max_episodes),
        "entries": entries,
        "skipped": skipped,
        "groups": groups,
        "contrastive_groups": contrastive,
        "contrast": bool(contrastive),
    }


def available_cycles(root: Path) -> list[int]:
    return sorted(int(p.name.split("-")[1])
                  for p in (root / "episodes").glob("cycle-*")
                  if p.is_dir() and len(p.name.split("-")) == 2
                  and p.name.split("-")[1].isdigit())


def collect_grid_batch(episodes_root: Path, streams_root: Path, *,
                       cycles: list[int] | None = None,
                       last_cycles: int | None = None, min_group: int = 2,
                       group_by: str = "task", max_episodes: int = 0) -> dict[str, Any]:
    """Union immutable attempts, with per-lane windows and one global cap.

    Independent lane cycle counters are not wall-time comparable. Recency
    uses the write-once collect summary's mtime_ns, with numeric lane/cycle/
    task ties. Source and clock annotations exist only in grid manifests.
    """
    if not streams_root.is_dir():
        raise SystemExit(f"no streams root: {streams_root}")
    if cycles is None and last_cycles is None:
        raise SystemExit("provide --cycles or --last-cycles")
    roots = sorted((p for p in streams_root.iterdir()
                    if p.is_dir() and p.name.startswith("lane-")
                    and p.name[5:].isdigit()), key=lambda p: (int(p.name[5:]), p.name))
    entries, skipped, lane_cycles = [], [], {}
    for root in roots:
        if not (root / "episodes").is_dir():
            continue
        selected = cycles if cycles is not None else available_cycles(root)[-last_cycles:]
        lane_cycles[root.name] = sorted(selected)
        body = collect_batch(root, selected, min_group=min_group, group_by=group_by)
        for entry in body["entries"]:
            summary = root / "episodes" / f"cycle-{entry['cycle']:04d}" / "collect-summary.json"
            entries.append({**entry, "source_lane": root.name,
                            "collected_mtime_ns": summary.stat().st_mtime_ns})
        skipped.extend({**item, "source_lane": root.name} for item in body["skipped"])
    entries.sort(key=lambda e: (e["collected_mtime_ns"], int(e["source_lane"][5:]),
                                e["cycle"], e["task_id"]))
    if max_episodes and len(entries) > max_episodes:
        dropped = len(entries) - max_episodes
        entries = entries[-max_episodes:]
        skipped.append({"reason": f"recency cap: {dropped} older episodes "
                                   f"dropped (max-episodes={max_episodes})"})
    groups = {}
    for entry in entries:
        group = groups.setdefault(entry["group_id"], {"episodes": 0, "passes": 0})
        group["episodes"] += 1
        group["passes"] += int(entry["reward"])
    for group in groups.values():
        group["fails"] = group["episodes"] - group["passes"]
        group["variance"] = bool(group["passes"] and group["fails"])
    contrastive = sorted(gid for gid, g in groups.items()
                         if g["variance"] and g["episodes"] >= min_group)
    return {"schema": BATCH_SCHEMA, "created_unix": time.time(),
            "episodes_root": str(episodes_root), "streams_root": str(streams_root),
            "encoder": "p50k_base", "lane_cycles": lane_cycles,
            "cycles": sorted({c for selected in lane_cycles.values() for c in selected}),
            "min_group": int(min_group), "group_by": group_by,
            "max_episodes": int(max_episodes), "entries": entries, "skipped": skipped,
            "groups": groups, "contrastive_groups": contrastive,
            "contrast": bool(contrastive)}


def _episode_token_count(record: Mapping[str, Any], rows: list[dict[str, Any]],
                         enc) -> int:
    """Whole-episode token count (the candidate loss's per-row denominator).

    The last row's prefix+generated IS the full episode token sequence
    (encode_candidate spans are contiguous through the final turn), but count
    from the record's own text so an interrupted episode (trailing turns that
    failed validation and are excluded from rows) still counts every sampled
    token honestly.
    """
    from scripts.build_e97_pi_native_curriculum import encode_candidate

    ids, _, _ = encode_candidate(record["native_record"],
                                 record["generations"], 0, enc)
    return len(ids)


def bind_batch_sha(body: Mapping[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(canonical_json(dict(body)).encode("utf-8")).hexdigest()
    return {**body, "batch_sha256": digest}


def verify_batch(manifest: Mapping[str, Any], *, reencode: bool = True) -> dict[str, Any]:
    """Fail-closed manifest verification (digests, counts, optional re-encode)."""
    if manifest.get("schema") != BATCH_SCHEMA:
        raise ValueError("unsupported batch schema")
    body = {key: value for key, value in manifest.items() if key != "batch_sha256"}
    digest = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    if digest != manifest["batch_sha256"]:
        raise ValueError("batch manifest digest does not bind the body")
    entries = manifest["entries"]
    for entry in entries:
        if entry["split"] != "train":
            raise ValueError("sealed split isolation violation in batch")
        if entry["reward"] not in (0.0, 1.0):
            raise ValueError("reward must be binary sealed-validator outcome")
        for row in entry["turns"]:
            if not row["prefix"] or not row["generated"]:
                raise ValueError("empty row prefix/generated")
            if row["span"][1] - row["span"][0] != len(row["generated"]):
                raise ValueError("row span does not match generated length")
        root = Path(manifest["episodes_root"])
        if "streams_root" in manifest:
            source = entry.get("source_lane", "")
            if not source.startswith("lane-") or not source[5:].isdigit():
                raise ValueError("invalid batch source lane")
            root = Path(manifest["streams_root"]) / source
        elif "source_lane" in entry:
            raise ValueError("source lane requires a streams root")
        episode_path = (root / "episodes" /
                        f"cycle-{entry['cycle']:04d}" / entry["task_id"] /
                        "attempt" / "episode-private.json")
        grade_path = episode_path.with_name("grade.json")
        if sha256_file(episode_path) != entry["episode_sha256"]:
            raise ValueError(f"episode identity drift: {episode_path}")
        if sha256_file(grade_path) != entry["grade_sha256"]:
            raise ValueError(f"grade identity drift: {grade_path}")
        if reencode:
            import tiktoken

            enc = tiktoken.get_encoding("p50k_base")
            record = json.loads(episode_path.read_text())
            rows = _episode_turn_rows(record, enc)
            if rows != entry["turns"]:
                raise ValueError(f"re-encode does not reproduce rows: {episode_path}")
            if _episode_token_count(record, rows, enc) != entry["episode_tokens"]:
                raise ValueError(f"episode token count does not reproduce: {episode_path}")
    return dict(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build")
    p.add_argument("--episodes-root", type=Path, required=True,
                   help="a lane root or the v1 workspace root (contains episodes/)")
    p.add_argument("--streams-root", type=Path, default=None,
                   help="union attempts from all lane-N directories; replay under current policy")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cycles", default=None,
                   help="comma-separated cycle numbers (default: none)")
    p.add_argument("--last-cycles", type=int, default=None,
                   help="the N most recent cycles present on disk")
    p.add_argument("--min-group", type=int, default=2)
    p.add_argument("--group-by", choices=("task", "family"), default="task",
                   help="group key: lake task id (round-refreshed tasks) or "
                        "sealed template family (one-shot lakeexp instances)")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="keep only the N most recent episodes (0 = all)")
    p.add_argument("--require-contrast", action="store_true",
                   help="exit nonzero when no contrastive group exists")
    p.add_argument("--reencode-verify", action="store_true", default=True)

    q = sub.add_parser("verify")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--no-reencode", action="store_true")

    args = parser.parse_args()
    if args.command == "build":
        episodes_dir = args.episodes_root / "episodes"
        available = sorted(int(p.name.split("-")[1]) for p in episodes_dir.glob("cycle-*")
                           if p.is_dir() and "-" in p.name
                           and p.name.split("-")[1].isdigit()
                           and "crashed" not in p.name)
        if args.cycles is not None:
            cycles = [int(item) for item in args.cycles.split(",") if item.strip()]
        elif args.last_cycles is not None:
            cycles = available[-args.last_cycles:]
        else:
            raise SystemExit("provide --cycles or --last-cycles")
        if args.streams_root is None:
            body = collect_batch(args.episodes_root, cycles,
                                 min_group=args.min_group,
                                 group_by=args.group_by,
                                 max_episodes=args.max_episodes)
        else:
            body = collect_grid_batch(args.episodes_root, args.streams_root,
                                      cycles=cycles if args.cycles is not None else None,
                                      last_cycles=args.last_cycles,
                                      min_group=args.min_group,
                                      group_by=args.group_by,
                                      max_episodes=args.max_episodes)
        manifest = bind_batch_sha(body)
        verify_batch(manifest, reencode=bool(args.reencode_verify))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output).write_text(json.dumps(manifest, sort_keys=True) + "\n")
        summary = {
            "schema": "emender-rl-loop-pg-batch-summary-v1",
            "manifest": str(args.output),
            "batch_sha256": manifest["batch_sha256"],
            "episodes": len(manifest["entries"]),
            "turns": sum(len(entry["turns"]) for entry in manifest["entries"]),
            "groups": manifest["groups"],
            "contrastive_groups": manifest["contrastive_groups"],
            "contrast": manifest["contrast"],
            "skipped": len(manifest["skipped"]),
        }
        print(json.dumps(summary, sort_keys=True))
        if args.require_contrast and not manifest["contrast"]:
            raise SystemExit(3)
    else:
        manifest = json.loads(args.manifest.read_text())
        verify_batch(manifest, reencode=not args.no_reencode)
        print(json.dumps({"schema": "emender-rl-loop-pg-batch-verified-v1",
                          "batch_sha256": manifest["batch_sha256"],
                          "episodes": len(manifest["entries"]),
                          "contrast": manifest["contrast"]}, sort_keys=True))


if __name__ == "__main__":
    main()
