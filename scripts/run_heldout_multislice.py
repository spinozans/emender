#!/usr/bin/env python3
"""Driver: measure E88 / GDN / M2RNN held-out BPB on every slice in the
multi-slice manifest, via the KNOWN-GOOD elman harness forward
(scripts/measure_pile_bpb_elman.py, y-mode swap, ctx2048 stride1024,
block-loss sanity gate). GPU 0 ONLY.

GPU 0 is shared with training/other eval jobs, so before each run we wait
until GPU 0 has enough free memory, and we retry on transient OOM. Batch size
only affects memory/speed, NOT the BPB value (windows are scored identically).

REAL MEASUREMENT — no fabricated numbers.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/erikg/ndm/.wg-worktrees/agent-758")
MANIFEST = ROOT / "paper/review/heldout_multislice_slices.json"
MEASURE = ROOT / "scripts/measure_pile_bpb_elman.py"
RESULTS = Path("/tmp/heldout_slices/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# Use the canonical PINNED (rotation-immune) checkpoints — the exact pinned
# steps from the task spec. Live training rotated step_1542000 out of /tmp
# mid-run, but the paper-pinned copies survive here.
PIN = "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints"
MODELS = {
    "e88": f"{PIN}/e88/checkpoint_step_1542000_loss_2.5970.pt",
    "fla-gdn": f"{PIN}/gdn/checkpoint_step_2031000_loss_2.7303.pt",
    "m2rnn": f"{PIN}/m2rnn/checkpoint_step_1491000_loss_2.7347.pt",
}

GPU = os.environ.get("HELDOUT_GPU", "1")  # dedicated free GPU (training/racers killed)
FREE_THRESHOLD_MIB = 10000  # need weights + optimizer-state swap + activations
BATCH_LADDER = [16, 8, 4, 2]  # try big first; drop on OOM (BPB is batch-invariant)
MAX_RETRIES = 60
POLL_SECS = 30


def gpu0_free_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", GPU]
    ).decode().strip().splitlines()[0]
    return int(out)


def wait_for_headroom():
    while True:
        free = gpu0_free_mib()
        if free >= FREE_THRESHOLD_MIB:
            print(f"[driver] GPU{GPU} free={free} MiB >= {FREE_THRESHOLD_MIB}; launching", flush=True)
            return
        print(f"[driver] GPU{GPU} free={free} MiB < {FREE_THRESHOLD_MIB}; waiting {POLL_SECS}s", flush=True)
        time.sleep(POLL_SECS)


def run_one(slice_obj, model_name, ckpt) -> dict:
    out_path = RESULTS / f"{slice_obj['name']}__{model_name}.json"
    if out_path.exists():
        try:
            r = json.loads(out_path.read_text())
            if r.get("bpb") is not None:
                print(f"[driver] {out_path.name} already done (bpb={r['bpb']:.4f}); skip", flush=True)
                return r
        except Exception:
            pass
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["MEASURE_ALLOWED_GPU"] = GPU
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    batch_idx = 0
    for attempt in range(1, MAX_RETRIES + 1):
        wait_for_headroom()
        batch = BATCH_LADDER[min(batch_idx, len(BATCH_LADDER) - 1)]
        cmd = [
            sys.executable, str(MEASURE),
            "--checkpoint", ckpt, "--name", model_name,
            "--slice", slice_obj["path"],
            "--expect-sha", slice_obj["sha256"],
            "--context", "2048", "--stride", "1024",
            "--batch-size", str(batch),
            "--out", str(out_path),
        ]
        print(f"[driver] === {slice_obj['name']} / {model_name} attempt {attempt} batch={batch} ===", flush=True)
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        tail = "\n".join(proc.stdout.splitlines()[-6:])
        print(tail, flush=True)
        if proc.returncode == 0 and out_path.exists():
            r = json.loads(out_path.read_text())
            if r.get("bpb") is not None:
                print(f"[driver] OK {slice_obj['name']}/{model_name} bpb={r['bpb']:.4f} "
                      f"blk={r['block_loss_nats']:.3f} batch={batch}", flush=True)
                return r
            print(f"[driver] sanity gate failed: {r.get('error')}", flush=True)
            return r
        err = proc.stderr or proc.stdout
        if "out of memory" in err.lower() or "OutOfMemory" in err:
            batch_idx += 1  # drop to a smaller batch
            print(f"[driver] OOM on attempt {attempt}; dropping to batch="
                  f"{BATCH_LADDER[min(batch_idx, len(BATCH_LADDER)-1)]}, backing off", flush=True)
            time.sleep(POLL_SECS)
            continue
        print(f"[driver] NON-OOM failure (rc={proc.returncode}):\n{err[-1500:]}", flush=True)
        return {"name": model_name, "slice": slice_obj["name"], "error": err[-1500:], "bpb": None}
    return {"name": model_name, "slice": slice_obj["name"], "error": "max retries exhausted", "bpb": None}


def main():
    manifest = json.loads(MANIFEST.read_text())
    slices = manifest["slices"]
    grid = {}
    for s in slices:
        grid[s["name"]] = {}
        for mname, ckpt in MODELS.items():
            r = run_one(s, mname, ckpt)
            grid[s["name"]][mname] = r
            # persist running aggregate after every run
            (RESULTS / "_aggregate.json").write_text(json.dumps({
                "slices": slices, "grid": grid,
            }, indent=2))
    print("[driver] ALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
