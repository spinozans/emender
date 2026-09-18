#!/usr/bin/env python3
"""Generate GPU reference logits for the E97 llama.cpp port qualification panel.

Run on ONE GPU in a juncture window (the 8-GPU training arc must not be
touched while it runs — the launcher pattern from scripts/serve_e97_interactive.sh
applies: acquire a single-GPU lease, verify the checkpoint SHA-256 pin, run,
release). CPU-only import surface until --device cuda is honored at run time.

For each of the 32 frozen prompts (prompts/qual-panel-32.json) this dumps:
  - the token ids (p50k_base, no specials) — the shared identity between the
    GPU reference and the llama.cpp port;
  - the next-token top-64 logits at the final prompt position;
  - a 64-token greedy continuation (temperature 0).

Output: <workdir>/reference/panel-reference.json (ids + top-64 logits +
greedy text) and <workdir>/reference/greedy-ids.json (continuation ids for
exact token-agreement scoring). The acceptance gates in
docs/validation/e97-llama-cpp-cpu-port-scope-v1.md §5 consume these files.

Usage (at the juncture window, under a 1-GPU lease):
  .venv/bin/python scripts/generate_e97_reference_logits.py \
      --checkpoint <promoted.pt> \
      --args-json /mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json \
      --panel prompts/qual-panel-32.json \
      --output-dir /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-llama-cpp-port-v1/reference \
      --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROMOTED_SHA256 = "d81464982c3ebc0d72769a87e079068bf535d6ca2010185dc1bb03ce264b8f5b"
GREEDY_TOKENS = 64
TOP_K = 64


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--args-json", type=Path, required=True)
    ap.add_argument("--panel", type=Path, default=Path("prompts/qual-panel-32.json"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu allowed for smoke tests only; reference MUST be cuda")
    args = ap.parse_args()

    digest = sha256_of(args.checkpoint)
    if digest != PROMOTED_SHA256:
        raise SystemExit(f"checkpoint SHA-256 mismatch: {digest}")
    print(f"checkpoint verified: {digest}", flush=True)

    import torch  # deferred: no CUDA import until we know we may run
    import tiktoken

    from ndm.e97 import load_e97_checkpoint

    enc = tiktoken.get_encoding("p50k_base")
    panel = json.loads(args.panel.read_text())
    assert panel["schema"] == "emender-e97-llama-cpp-qual-panel-v1"
    prompts = panel["prompts"]
    assert len(prompts) == 32

    print(f"loading checkpoint on {args.device} (weight_mode=saved)...", flush=True)
    loaded = load_e97_checkpoint(
        args.checkpoint,
        args_json=args.args_json,
        device=torch.device(args.device),
        weight_mode="saved",
    )
    model = loaded.model
    model.eval()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = {"schema": "emender-e97-llama-cpp-reference-v1", "checkpoint_sha256": digest,
                 "panel": str(args.panel), "device": args.device, "weight_mode": "saved",
                 "tokenization": "tiktoken p50k_base, encode(text), no specials", "records": []}
    greedy_ids = {"schema": "emender-e97-llama-cpp-reference-greedy-v1", "records": []}

    with torch.no_grad():
        for i, item in enumerate(prompts):
            ids = enc.encode(item["prompt"], allowed_special=set())
            input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
            logits = model(input_ids)  # [1, T, vocab]
            final = logits[0, -1].float().cpu()
            top = torch.topk(final, TOP_K)
            # greedy continuation from the final prompt position
            cont = []
            cur = input_ids
            for _ in range(GREEDY_TOKENS):
                step = model(cur)[0, -1]
                nxt = int(torch.argmax(step).item())
                cont.append(nxt)
                if nxt == enc.eot_token:
                    break
                cur = torch.cat([cur, torch.tensor([[nxt]], device=args.device)], dim=1)
            cont_text = enc.decode(cont)
            reference["records"].append({
                "id": item["id"], "category": item["category"], "n_prompt_tokens": len(ids),
                "prompt_token_ids": ids,
                "next_token_top64_ids": [int(v) for v in top.indices.tolist()],
                "next_token_top64_logits": [round(float(v), 6) for v in top.values.tolist()],
            })
            greedy_ids["records"].append({
                "id": item["id"], "greedy_ids": cont, "greedy_text": cont_text,
            })
            print(f"[{i + 1}/32] {item['id']}: {len(ids)} prompt tokens, greedy '{cont_text[:40]!r}...'", flush=True)

    (out_dir / "panel-reference.json").write_text(json.dumps(reference, indent=1))
    (out_dir / "greedy-ids.json").write_text(json.dumps(greedy_ids, indent=1))
    print(f"wrote {out_dir}/panel-reference.json and greedy-ids.json", flush=True)


if __name__ == "__main__":
    main()
