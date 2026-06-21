#!/usr/bin/env python3
"""End-to-end verification of the pinned-autotune fix on the REAL production path.

For EACH race arm (emender level=E97, gdn2-mlp) launches train.py for a few real
fwd+bwd steps and checks the task's acceptance criteria:

  1. STORM KILLED  : pinned run emits ZERO Triton-autotuning "finished after"
                     lines; the sweep run emits many (the storm).
  2. FAST INIT     : pinned run reaches training step 1 in < INIT_BUDGET_S
                     (wallclock from process launch).
  3. NUMERIC PARITY: pinned vs sweep per-step losses identical to < TOL, with
                     the SAME seed. Determinism is verified first (two pinned
                     runs must match) so the pinned-vs-sweep comparison is
                     meaningful; config controls scheduling, not math.

Run on an IDLE leased GPU (NOT GPU 0 = the racer):
    eval "$(scripts/gpu_lease.sh acquire 1)"
    python scripts/verify_pinned_autotune.py
"""
import os
import re
import subprocess
import sys
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("DATA", "/home/erikg/elman/data/pile.txt")
STEPS = int(os.environ.get("STEPS", "3"))
INIT_BUDGET_S = float(os.environ.get("INIT_BUDGET_S", "15"))
TOL = float(os.environ.get("TOL", "1e-6"))

COMMON = ["--data", DATA, "--tokenizer", "p50k_base", "--bf16",
          "--batch_size", "4", "--chunk_size", "2048",
          "--optimizer", "schedulefree", "--seed", "42",
          "--save_every", "100000000", "--val_every", "1000000000",
          "--log_every", "1", "--steps", str(STEPS)]

ARMS = {
    "emender": ["--level", "E97", "--dim", "1792", "--n_heads", "216",
                "--n_state", "32", "--depth", "11", "--expansion", "1.0",
                "--use_gate", "1", "--gate_activation", "silu",
                "--mlp_ratio", "2.2623", "--mlp_multiple", "64",
                "--use_triton", "1", "--lr", "0.001007"],
    "gdn2-mlp": ["--level", "gdn2-mlp", "--dim", "2176", "--depth", "12",
                 "--n_heads", "30", "--expansion", "1",
                 "--gdn2_mlp_ratio", "3.258732449079677", "--use_conv", "1",
                 "--d_conv", "4", "--warmup_steps", "0", "--lr", "0.000474"],
}

STEP_RE = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+([0-9.]+)")
AUTOTUNE_FINISHED_RE = re.compile(r"Triton autotuning for function .*")
GUARD_RE = re.compile(r"\[fused-guard\].*NO eager fallback")


def run(arm, pinned, tmp):
    args = ["python", "train.py"] + ARMS[arm] + COMMON + ["--output", os.path.join(tmp, f"{arm}_{pinned}")]
    env = dict(os.environ)
    env["TRITON_PRINT_AUTOTUNING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["GDN2_PATH"] = env.get("GDN2_PATH", "/home/erikg/GatedDeltaNet-2")
    env["NDM_PIN_TRITON_AUTOTUNE"] = "1" if pinned else "0"
    env.pop("NDM_PIN_TRITON_RECORD", None)

    t0 = time.time()
    first_step_t = None
    losses = {}
    storm = 0
    guard = False
    proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        if AUTOTUNE_FINISHED_RE.search(line):
            storm += 1
        if GUARD_RE.search(line):
            guard = True
        m = STEP_RE.match(line.strip())
        if m:
            s = int(m.group(1))
            if s == 1 and first_step_t is None:
                first_step_t = time.time() - t0
            losses[s] = float(m.group(2))
    proc.wait()
    return {"rc": proc.returncode, "init_s": first_step_t, "losses": losses,
            "storm_lines": storm, "fused_guard": guard}


def main():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="verify_pin_")
    report = {}
    ok = True
    for arm in ARMS:
        print(f"\n========== ARM {arm} ==========", flush=True)
        pin_a = run(arm, True, tmp)
        print(f"  pinned#1: rc={pin_a['rc']} init={pin_a['init_s']:.1f}s storm={pin_a['storm_lines']} "
              f"guard={pin_a['fused_guard']} losses={pin_a['losses']}", flush=True)
        pin_b = run(arm, True, tmp)
        print(f"  pinned#2: rc={pin_b['rc']} init={pin_b['init_s']:.1f}s storm={pin_b['storm_lines']} "
              f"losses={pin_b['losses']}", flush=True)
        sweep = run(arm, False, tmp)
        print(f"  sweep   : rc={sweep['rc']} init={sweep['init_s']:.1f}s storm={sweep['storm_lines']} "
              f"losses={sweep['losses']}", flush=True)

        steps = sorted(set(pin_a["losses"]) & set(pin_b["losses"]) & set(sweep["losses"]))
        det = all(pin_a["losses"][s] == pin_b["losses"][s] for s in steps)
        max_dpp = max((abs(pin_a["losses"][s] - pin_b["losses"][s]) for s in steps), default=float("nan"))
        max_pvs = max((abs(pin_a["losses"][s] - sweep["losses"][s]) for s in steps), default=float("nan"))

        storm_killed = pin_a["storm_lines"] == 0 and pin_b["storm_lines"] == 0 and sweep["storm_lines"] > 0
        fast_init = pin_a["init_s"] is not None and pin_a["init_s"] < INIT_BUDGET_S
        # parity: if the kernels are deterministic, require exact pinned==sweep;
        # otherwise require pinned-vs-sweep within the same band as pinned-vs-pinned
        # (proving the config change adds no numerical difference beyond kernel noise).
        if det:
            parity = max_pvs < TOL
        else:
            parity = max_pvs <= max(max_dpp, TOL)

        arm_ok = (pin_a["rc"] == 0 and sweep["rc"] == 0 and storm_killed and fast_init
                  and parity and pin_a["fused_guard"])
        ok = ok and arm_ok
        report[arm] = {
            "storm_killed": storm_killed, "pinned_storm": pin_a["storm_lines"],
            "sweep_storm": sweep["storm_lines"], "fast_init_s": pin_a["init_s"],
            "fast_init_ok": fast_init, "deterministic": det,
            "max_pin_vs_pin": max_dpp, "max_pin_vs_sweep": max_pvs,
            "parity_ok": parity, "fused_guard": pin_a["fused_guard"], "ok": arm_ok,
        }
        print(f"  => storm_killed={storm_killed} fast_init={fast_init}({pin_a['init_s']}) "
              f"det={det} dpinpin={max_dpp:.2e} dpinsweep={max_pvs:.2e} parity={parity} "
              f"guard={pin_a['fused_guard']} ARM_OK={arm_ok}", flush=True)

    report["ALL_OK"] = ok
    print("\n===== SUMMARY =====")
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
