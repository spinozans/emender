#!/usr/bin/env python3
"""wash_build_packs.py — era-7 degeneracy wash: build an SFT authority from
every verified, screen-clean teacher-corrected receipt in the bank's lane
streams (operator directive 2026-10-03/04: "why not? go?").

Selection (fail-closed):
  - receipts verified by walk_stream (digest-chain intact) in every lane
  - kind == teacher-corrected only (the clean teacher episodes)
  - text-level degeneracy screen mirroring rl_loop_driver.degeneracy_screen:
    (1) >=2 assistant Analysis samples all identical -> reject
    (2) last ToolResult isError -> reject
Build uses the sealed rl_build_pack builders (receipt re-encode enforced).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts")
import rl_build_pack as bp
from rl_common import workspace_paths
from rl_receipts import walk_stream

BANK = Path("/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/bank-singlelearner-B")
OUT = Path("/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/wash-v2")


def text_screen(episode_text: str) -> bool:
    analyses = re.findall(r"\nAssistant:\nAnalysis: (.*(?:\n(?!!).*)*)\nCommentary:",
                           episode_text)
    analyses = [a.strip() for a in analyses if a.strip() and a.strip() != "null"]
    results = re.findall(r'"isError":\s*(true|false)', episode_text)
    if len(analyses) >= 2 and len(set(analyses)) == 1:
        return False
    if results and results[-1] == "true":
        return False
    return True


def main() -> None:
    import tiktoken
    enc = tiktoken.get_encoding("p50k_base")
    selected = []
    total = rejected = 0
    for lane in sorted(BANK.glob("lanes/lane-*")):
        paths = workspace_paths(lane)
        for receipt in walk_stream(paths, enc=enc):
            if receipt.get("kind") != "teacher-corrected":
                continue
            total += 1
            if text_screen(receipt["episode"]["episode_text"]):
                selected.append(receipt)
            else:
                rejected += 1
    print(f"WASH selection: teacher receipts={total} clean={len(selected)} "
          f"screen-rejected={rejected}")
    if len(selected) < 500:
        raise SystemExit("not enough clean teacher receipts for a wash")
    targets = sum(r["episode"]["targets"] for r in selected)
    tokens = sum(r["episode"]["tokens"] for r in selected)
    print(f"WASH dose: records={len(selected)} episode-tokens={tokens} "
          f"assistant-targets={targets}")
    OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    authority = bp.build_authority(selected, OUT / "authority", cycle=0)
    print(f"WASH authority manifest sha256={authority['manifest_sha256']} "
          f"records={authority['records']} tokens={authority['tokens']} "
          f"targets={authority['targets']}")
    context = max(2048, -(-authority["max_record_tokens"] // 2048) * 2048)
    packs = bp.build_packs(OUT / "authority", OUT / "packs",
                            authority_sha=authority["manifest_sha256"],
                            context_size=context,
                            python=sys.executable)
    print(f"WASH packs={packs['packs']} pack manifest sha256="
          f"{packs['pack_manifest_sha256']} context_size={context}")
    (OUT / "wash-build-summary.json").write_text(__import__("json").dumps({
        "schema": "emender-era7-wash-build-v1",
        "selected_receipts": len(selected),
        "teacher_receipts_total": total,
        "screen_rejected": rejected,
        "authority_manifest_sha256": authority["manifest_sha256"],
        "pack_manifest_sha256": packs["pack_manifest_sha256"],
        "context_size": context,
        "episode_tokens": tokens,
        "assistant_targets": targets,
    }, sort_keys=True, indent=1) + "\n")


if __name__ == "__main__":
    main()
