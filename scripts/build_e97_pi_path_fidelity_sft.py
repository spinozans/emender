#!/usr/bin/env python3
"""Build broad randomized Pi-v2 literal-path and recovery trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_agent_protocol import E97_PI_AGENT_SYSTEM_V2
from scripts.build_e97_pi_finalization_repair_sft import serialize_live_aligned
from scripts.build_e97_pi_instruction_sft import ENCODING, action, read_result, split
from scripts.build_e97_pi_recover_read_sft import live_missing_read_result

KINDS = ("literal-read", "literal-pair", "literal-write", "literal-edit", "failed-read-recovery")
EXTENSIONS = ("json", "txt", "toml", "yaml", "md", "py")
USER_TEMPLATES = (
    "Use exactly the path {path}. {request}",
    "Work on `{path}` (do not substitute another location). {request}",
    "The requested file is {path}. {request}",
    "At the current working directory, {request_lower} Path: {path}",
)


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def random_path(rng: random.Random, index: int) -> str:
    depth = rng.randint(1, 4)
    components = [f"zone_{rng.getrandbits(28):07x}" for _ in range(depth)]
    name = f"asset_{index:09d}_{rng.getrandbits(32):08x}.{rng.choice(EXTENSIONS)}"
    return "/".join([*components, name])


def read_turn(path: str, content: str) -> list[tuple[str, str]]:
    lines = content.rstrip("\n").splitlines()
    return [action("read", {"path": path, "offset": 1, "limit": 40}),
            ("tool", read_result(path, lines, 1))]


def trajectory(kind: str, index: int, rng: random.Random):
    path = random_path(rng, index)
    marker = f"marker-{rng.getrandbits(48):012x}"
    value = rng.randint(10_000, 9_999_999)
    content = f"marker={marker}\nvalue={value}\n"
    target_positions = None

    if kind == "literal-read":
        request = "Read it and report its marker and exact value."
        user = rng.choice(USER_TEMPLATES).format(
            path=path, request=request, request_lower=request[0].lower() + request[1:])
        turns = [*read_turn(path, content),
                 ("assistant", f"Final: `{path}` contains marker `{marker}` and value `{value}`.")]
    elif kind == "literal-pair":
        other = random_path(rng, index + 50_000_000)
        other_value = value + rng.randint(1, 50_000)
        user = (f"Read exactly `{path}` and `{other}`. Report which has the larger value "
                "and give that exact value; do not infer alternate paths.")
        turns = [*read_turn(path, content),
                 *read_turn(other, f"marker={marker}\nvalue={other_value}\n"),
                 ("assistant", f"Final: `{other}` has the larger value, `{other_value}`.")]
    elif kind == "literal-write":
        payload = json.dumps({"marker": marker, "value": value}, indent=2) + "\n"
        user = (f"Write the requested JSON payload to exactly `{path}` with marker `{marker}` "
                f"and integer value `{value}`. Do not choose a conventional filename.")
        turns = [action("write", {"path": path, "content": payload}),
                 ("tool", f"Successfully wrote {len(payload.encode())} bytes to {path}"),
                 ("assistant", f"Final: Wrote marker `{marker}` and value `{value}` to `{path}`.")]
    elif kind == "literal-edit":
        replacement = value + rng.randint(1, 50_000)
        old = f"value={value}"
        new = f"value={replacement}"
        user = (f"In exactly `{path}`, change `{old}` to `{new}` after reading the file. "
                "Do not substitute another path.")
        turns = [*read_turn(path, content),
                 action("edit", {"path": path, "oldText": old, "newText": new}),
                 ("tool", f"Successfully replaced 1 block(s) in {path}."),
                 ("assistant", f"Final: Updated `{path}` from `{old}` to `{new}`.")]
    elif kind == "failed-read-recovery":
        wrong = path.rsplit("/", 1)[0] + f"/default.{path.rsplit('.', 1)[-1]}"
        user = (f"Read exactly `{path}` and report marker `{marker}`. If another path fails, "
                "recover by using the literal path in this request.")
        turns = [action("read", {"path": wrong, "offset": 1, "limit": 40}),
                 ("tool", live_missing_read_result(wrong)),
                 *read_turn(path, content),
                 ("assistant", f"Final: Recovered with literal path `{path}`; marker is `{marker}`.")]
        # The failed guess is authentic context, not a behavior target.
        target_positions = {4, 6}
    else:
        raise ValueError(kind)
    task = {"kind": kind, "path": path, "marker": marker, "value": value}
    return user, turns, task, target_positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=9_740_971)
    args = parser.parse_args()
    if args.records <= 0 or args.records % len(KINDS):
        raise SystemExit(f"records must be a positive multiple of {len(KINDS)}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    encoding = tiktoken.get_encoding(ENCODING)
    counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0,
              "train_records": 0, "validation_records": 0}
    kind_counts = {kind: 0 for kind in KINDS}
    offset = 0
    with paths["tokens"].open("wb") as token_out, paths["mask"].open("wb") as mask_out, \
         paths["index"].open("wb") as index_out, paths["metadata"].open("w") as metadata_out:
        for record_index in range(args.records):
            kind = KINDS[record_index % len(KINDS)]
            identity = f"pi-v2-path-fidelity-{kind}-{record_index:09d}"
            user, turns, task, selected = trajectory(
                kind, record_index, random.Random(args.seed + record_index))
            messages = [("system", E97_PI_AGENT_SYSTEM_V2), ("user", user), *turns]
            if selected is not None:
                # trajectory positions are relative to turns; system+user add two.
                selected = {position for position in selected}
            tokens, masks, complete = serialize_live_aligned(
                messages, encoding, target_mode="all-assistant",
                target_assistant_positions=selected)
            validation = int(split(identity))
            token_out.write(struct.pack(f"<{len(tokens)}I", *tokens))
            mask_out.write(bytes(masks))
            index_out.write(RECORD_INDEX.pack(offset, len(tokens), sum(masks), validation))
            metadata_out.write(json.dumps({
                "id": identity, "source": "emender-pi-v2-path-fidelity-v1",
                "split": validation, "kind": kind, "tokens": len(tokens),
                "targets": sum(masks), "user": user,
                "task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
                "serialization_sha256": hashlib.sha256(complete.encode()).hexdigest(),
            }, sort_keys=True) + "\n")
            offset += len(tokens)
            counts["records"] += 1
            counts["tokens"] += len(tokens)
            counts["assistant_target_tokens"] += sum(masks)
            counts["validation_records" if validation else "train_records"] += 1
            kind_counts[kind] += 1
    manifest = {
        "schema": AUTHORITY_SCHEMA, "status": "complete",
        "purpose": "broad randomized Pi-v2 literal-path fidelity and failed-read recovery",
        "system_prompt": E97_PI_AGENT_SYSTEM_V2, "tokenizer": ENCODING,
        "seed": args.seed, "kinds": list(KINDS), "kind_counts": kind_counts,
        "failure_context_policy": "failed guesses are untargeted; corrected actions and finals are targeted",
        "evaluation_exclusion": {
            "diagnostic_v3": "behavior influenced curriculum design; no further blind claim",
            "untouched_v4_manifest_sha256": "8d0d5d350a39c9f5c007d98e4a0010f4a06f39e5a7fb89ecd9ccc4e37e66b8d8",
            "policy": "random nonce paths avoid known evaluation directory and filename templates",
        },
        "counts": counts, "outputs": {name: entry(path) for name, path in paths.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
