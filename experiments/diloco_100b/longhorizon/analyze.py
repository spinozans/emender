#!/usr/bin/env python3
"""diloco-loss-parity-longhorizon: turn results.txt into (a) per-arm held-out BPB
trajectories (each arm's OWN tokens/step) and (b) a matched-STEP gap table vs the
DDP reference. REAL measured BPB on the agent-1433 preflight heldout (bpt 3.938).

Lines consumed:
  BPB_RESULT <tag> <step> <bpb> <ce> <tokens>
  PHASE2_TOKSTEP <tag> <tokens_per_step>     (optional; default 86016 = 7-GPU bs6)
  PHASE2_TOKS <tag> <global_tok_per_s>       (optional; measured throughput)

Usage: python analyze.py results.txt
"""
import sys, collections

DEFAULT_TOKSTEP = 6 * 2048 * 7  # 86016, 7-GPU bs6 ctx2048

path = sys.argv[1] if len(sys.argv) > 1 else 'results.txt'
data = collections.defaultdict(dict)   # data[step][tag] = bpb (first-wins)
tokstep = {}                            # tag -> tokens/step
tps = {}                               # tag -> measured global tok/s
tags = []
with open(path) as f:
    for line in f:
        p = line.split()
        if not p:
            continue
        if p[0] == 'BPB_RESULT' and len(p) >= 4:
            tag, step, bpb = p[1], int(p[2]), float(p[3])
            if tag not in data[step]:      # keep clean periodic ckpt (see train.py fix)
                data[step][tag] = bpb
            if tag not in tags:
                tags.append(tag)
        elif p[0] == 'PHASE2_TOKSTEP' and len(p) >= 3:
            tokstep[p[1]] = int(p[2])
        elif p[0] == 'PHASE2_TOKS' and len(p) >= 3:
            tps[p[1]] = float(p[2])


def tstep(tag):
    return tokstep.get(tag, DEFAULT_TOKSTEP)


# --- per-arm trajectories (each arm's own tokens/step) ---
print("# Per-arm held-out BPB trajectory (heldout bpt=3.938)\n")
for t in tags:
    steps_t = sorted(s for s in data if t in data[s])
    thr = f"  ~{tps[t]/1000:.1f}k tok/s" if t in tps else ""
    print(f"## {t}  (tok/step={tstep(t)}){thr}")
    for s in steps_t:
        print(f"   step {s:>5}  {s*tstep(t)/1e6:>7.1f} Mtok   BPB {data[s][t]:.4f}")
    print()

# --- matched-STEP gap vs DDP reference ---
ddp_tag = next((t for t in tags if t.startswith('ddp')), None)
if ddp_tag:
    print(f"# matched-STEP gap vs {ddp_tag} (positive = arm worse). NOTE: arms with a\n"
          f"# different tok/step than the DDP ref are NOT matched-token at equal step;\n"
          f"# use the Mtok column above for those.\n")
    dil = [t for t in tags if t != ddp_tag]
    hdr = f"{'step':>6} {'Mtok(7gpu)':>11} " + " ".join(f"{t[:16]:>16}" for t in dil)
    print(hdr); print("-" * len(hdr))
    for s in sorted(data):
        if ddp_tag not in data[s]:
            continue
        ref = data[s][ddp_tag]
        row = f"{s:>6} {s*DEFAULT_TOKSTEP/1e6:>11.1f} "
        for t in dil:
            v = data[s].get(t)
            row += f"{v-ref:>+16.4f} " if v is not None else f"{'-':>16} "
        print(row)
