# preflight-100b — RESULTS (measured)

Pre-flight for the 100B-token seed-model run. **All numbers below are REAL,
measured on the 8×RTX 6000 Ada box (49 GB each, PCIe, no NVLink), commapile_mainmix
1 TB, p50k_base tokenizer, ctx 2048, bf16 + FUSED Triton, schedule-free AdamW.**
Date 2026-06-14. Task `preflight-100b`.

Configs (byte-identical to `cmaes_search_v2.build_train_command`, lb-compare geometries):
- **emender-mlp** = E97 split-edit **DELTA** (`e88_raw_write=0`) + SwiGLU MLP,
  dim1792 nh216 ns32 dep11 mlp2.2623 — **measured 1,286,589,072 params** (= plan 1286.6M).
- **gdn2-mlp** = GDN-2 + SwiGLU MLP, dim2176 nh30 dep12 mlp3.2587, use_conv —
  **measured 1,286,713,448 params** (= plan 1286.7M).

train.py was made **torchrun/DDP-capable** (opt-in; single-GPU path byte-identical,
verified). Launch all runs via `experiments/preflight_100b/run_ddp.sh`.

---

## 1. THROUGHPUT — emender-mlp 1.3B, 7-GPU DDP (steady state)

| metric | value |
|---|---|
| max per-GPU batch (single-GPU probe) | **8** (40.7 GB) |
| max per-GPU batch **under DDP** | **6** (bs8 OOMs: DDP buckets + NCCL add ~3 GB) |
| global batch (bs6 × 7) | 42 seq × 2048 = **86,016 tok/update** |
| **steady-state per-GPU tok/s** | **4,470** |
| **steady-state GLOBAL tok/s** | **31,291** (8 windows, steps 75–250) |
| peak GPU mem / rank (bs6) | **38,497 MB** (reserved 40,114) — ~80 % of 49 GB |
| GPU util | **100 % on all 7**, GPU 7 left free (lease 7) |
| bf16 + fused | **asserted no-eager on ALL 7 ranks** (`use_triton=1` fused-guard) |
| loss | 10.6 → 4.82 monotone, no NaN |

grad_accum=4 (+ DDP `no_sync` comm amortization) → **31,272 global tok/s, unchanged**
⇒ the run is **not** gradient-comm-frequency-bound.

## 2. E97 SPEEDUP AT 1.3B — emender-mlp / gdn2-mlp (MEASURED, identical conditions)

| arm | batch | per-GPU tok/s | global tok/s | peak mem/GPU |
|---|---|---|---|---|
| emender-mlp | **bs4 (matched)** | 3,211 | **22,474** | 28,942 MB |
| gdn2-mlp | **bs4 (matched)** | 3,290 | **23,034** | 35,715 MB |
| emender-mlp | bs6 (its DDP-max) | 4,470 | 31,291 | 38,497 MB |

- **Matched-conditions (bs4) ratio emender/gdn2 = 0.976× → a TIE.**
  The grok-scale **1.26–1.56× speedup is NOT reproduced at 1.3B** — **REFUTED**,
  consistent with the post-mortem warning that prior E97 throughput claims were wrong.
- Emender's real edge is **memory**: 28.9 GB vs gdn2's 35.7 GB at bs4 → emender fits
  bs6 where gdn2 OOMs. At each arm's own DDP-max batch the aggregate ratio is
  **1.358×**, but that is a batch-size/memory-efficiency effect, **not** a per-token
  kernel speedup. Honest statement: **per-token kernel throughput is a tie (~0.98×);
  emender wins ~1.36× aggregate only because it runs a larger batch.**

## 3. END-TO-END pipeline verification

- **DDP gradient sync**: loss decreases on both arms (emender 7.97→4.82, gdn2 8.14→5.51),
  finite throughout, no NaN. ✓
- **Checkpoint SAVE+RELOAD roundtrip**: DDP-saved 7.72 GB checkpoint (146 tensors)
  reloads into a fresh single-process model `strict=True` with **0 missing / 0
  unexpected** keys; reload held-out **BPB 1.7745 == in-run FINAL_HELDOUT_BPB 1.7745**
  (y-mode averaged weights bit-faithful). ✓ (`ckpt_roundtrip.py` → `ROUNDTRIP_OK`)
- **Held-out BPB eval**: runs, finite. CE 4.8438, **BPB 1.7745 (avg) / 1.7645 (non-avg)**,
  65,536 scored tokens, bytes/token 3.9380 (commapile tail, p50k_base). ✓
  (Barely-trained 250-step value — this verifies the EVAL PATH, not convergence.)
- **Fused, no eager, all 7 ranks**: `[fused-guard] rank R/7 ... use_triton=1 -> fused
  split-edit Triton kernel, NO eager fallback` printed by every rank; once `use_triton=1`
  the forward hard-imports the Triton kernel (an import failure RAISES, never silently
  drops to eager), and the run completed → fused path executed on all ranks. ✓

## 4. SCALING ANALYSIS (the headline risk)

| configuration | aggregate tok/s | per-GPU | efficiency vs 1-GPU |
|---|---|---|---|
| 1× GPU (emender bs6) | 8,600 | 8,600 | 100 % (baseline) |
| **7× GPU DDP** (bs6) | **31,291** | 4,470 | **52 %** |
| **7× INDEPENDENT procs** (bs6, no DDP) | **~62,000** | ~8,900 | **~103 % (near-linear)** |

7 independent processes (genuinely concurrent: same start, ±5 s end) scale
**near-linearly** → the box CPU / NVMe / PCIe-H2D / power are **NOT** the bottleneck.
The DDP halving is **entirely the per-step all-reduce of the 1.29B bf16 gradient over
PCIe (no NVLink)**. grad_accum does not fix it (the 2.6 GB all-reduce costs the same
per sync regardless of accumulation). **This is precisely the regime the frontier
`SCHEDULEFREE_DILOCO` design targets: periodic (K-step) weight sync instead of
per-step all-reduce recovers the ~62k independent ceiling.**

## 5. PROJECTED WALL-CLOCK (from measured tok/s)

| path | tok/s | to 16B gate | to 100B seed |
|---|---|---|---|
| **7-GPU DDP (measured, train.py today)** | 31,291 | **5.9 days** | **37.0 days** |
| gdn2-mlp control (bs4 DDP) | 23,034 | 8.0 days | 50.3 days |
| **DiLoCo-style / independent ceiling** (measured) | ~62,000 | **~3.0 days** | **~18.7 days** |
| 1× GPU single (no parallelism) | 8,600 | 21.5 days | 134 days |

## 6. Accept / reject

**GO for the 16B <1bpb gate** (≈ 6 days on 7-GPU DDP today; ≈ 3 days with DiLoCo).
This is well within budget and is the decision-relevant milestone.

**100B at standard 7-GPU DDP = ~37 days (> 3 weeks)** — exceeds the "~3 weeks" frame.
**To bring 100B under 3 weeks, do NOT use vanilla per-step DDP**: switch to the
DiLoCo-style periodic-sync parallelism the frontier design already specifies
(measured independent ceiling ~62k tok/s → **~19 days to 100B**). The 1-GPU-per-arm
recommendation in `docs/SCALE_PLAN.md §2.5` is corroborated: per-step DDP on this
no-NVLink box wastes ~48 % of the GPUs.

**Arm choice:** emender-mlp and gdn2-mlp are a **throughput tie per token** at 1.3B
(0.98×); emender-mlp keeps its loss/capability lead (SCALE_PLAN §0) and is more
memory-efficient (larger batch). The 1.26–1.56× throughput premise must **not** be
carried into the run (refuted here, per pre-flight §4.0 mandate).

### Reproduce
```
# emender-mlp 7-GPU DDP (bs6 max):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HELDOUT_REPORT_NONAVG=1 HELDOUT_EVAL_BS=4 \
  bash experiments/preflight_100b/run_ddp.sh emender 250 6 /tmp/out \
  --final_heldout_eval --heldout_tensor experiments/preflight_100b/heldout_comma_p50k_2048.pt
# gdn2-mlp matched (bs4):  bash experiments/preflight_100b/run_ddp.sh gdn2 150 4 /tmp/out_gd
# checkpoint roundtrip:    python experiments/preflight_100b/ckpt_roundtrip.py --ckpt_dir /tmp/out --heldout .../heldout_comma_p50k_2048.pt
```
Raw logs: `experiments/preflight_100b/{emender_ddp,emender_ddp_bs4,gdn2_ddp_bs4,emender_ddp_ga4}.log`.
