# DiLoCo seed-race I=4 — Frontier dress-rehearsal (task: diloco-seed-race-2)

**Question.** Take a *mature* single-GPU emender checkpoint (step 150500 ≈ 1.233 B
tokens) and continue it as a **4-island DiLoCo** run with **plain averaging**,
racing ahead ~3 B tokens. Does I=4 seeded DiLoCo **track or beat** the single-GPU
continuation at **matched total tokens** (i.e. does the SWA-style averaging
benefit, already seen at I≤4 from a 528 M seed in `diloco-scaling-law`, hold at a
mature seed and a long race)? This is the literal "train locally, then scale out
on Frontier" simulation.

## Frozen recipe (byte-identical geometry to the reference run's `args.json`)

Seed: `/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750/checkpoint_step_150500_loss_3.0442.pt`
(newest emender checkpoint; ~1.233 B tokens). **All 4 replicas** resume from it
with **full state incl. the schedule-free optimizer `z`** (`--resume` →
`load_checkpoint`).

E97 emender, dim 1792, depth 11, n_heads 216, n_state 32, mlp_ratio 2.2623,
mlp_multiple 64, embed_dim 1024, **gate_activation silu** (NOT train.py's default
`sigmoid` — silent-correctness trap on resume), use_triton 1, bf16, schedule-free
lr 0.001007, warmup 0, batch_size 4 (native per replica), chunk_size 2048,
data pile.txt, tokenizer p50k_base.

DiLoCo: `--diloco --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0`
(**PLAIN averaging**; outer_beta=0.9 DIVERGES — `diloco-scaling-law` measured
degradation +1.8 → +35 BPB — so it is forbidden). PCIe-NCCL hygiene:
`NCCL_P2P_DISABLE=1`, `TORCH_NCCL_ENABLE_MONITORING=0`, sequential subgroup
warmup (in-code). FUSED only; the fused-guard asserts on every rank (no eager).

Launched DETACHED via `scripts/launch_detached_run.sh` (survives agent exit; the
4-GPU lease is owned by the detached process). References on GPUs 0-1 are left
untouched; exactly 4 GPUs leased so ≥1 stays free for the capability track.

`--steps 240000` (resume from 150500 → 89500 DiLoCo steps → 89500×32768 = 2.93 B
additional tokens ≈ 3 B). `--save_every 3000` (= 12×diloco_k, so every periodic
save lands on a merge step → the saved rank-0 checkpoint is the **post-merge
cross-island CONSENSUS**). `--keep_checkpoints 30` (retain the whole curve).

## Why save_every=3000 and not the loose "~0.5 B" cadence

The single-GPU reference run **stops at step 244141 = 2.0 B total tokens**, while
the seed is at 1.233 B. The matched-token head-to-head therefore lives entirely
in the **overlap window [1.233 B, 2.0 B]**. A 0.5 B cadence lands ~1 consensus
checkpoint there — too thin for the "effect sizes, not single noisy points"
discipline the task demands. `save_every=3000` lands **7 consensus checkpoints**
in the overlap (steps 153000…171000), so the comparison is an effect size over
many points.

## Token bookkeeping (matched TOTAL tokens)

`bs*chunk = 4*2048 = 8192` tokens / optimizer step / replica.
- single-GPU reference (world=1, no resume): `total = step * 8192`
- I=4 DiLoCo (world=4, seed @ S0=150500): the seed phase was single-GPU, only the
  DiLoCo phase is ×W islands → `total = S0*8192 + (step - S0)*8192*4`.

`eval_checkpoint.py`'s own `tokens` column multiplies the *full* step by
world_size, so it 4×-overcounts the pre-seed tokens for the DiLoCo run.
`analyze_seed_race_i4.py` therefore ALWAYS recomputes total tokens from the
`step` column with the seed-aware formula above.

## Scoring (offline, matched tokens, fused, y-mode)

Both runs are scored on the **same shared pile-tail held-out tensor**
(`heldout_pile_tail_p50k_2048_1m.pt`, md5 `8e1198ab…`, 512×2049, the tail 10 % of
pile.txt held out from training, bytes/token 3.945) — the identical tensor the
references were scored on in `offline-eval-references`. `--y-mode train` rebuilds
the schedule-free train weights `y` from the saved `x`/`z`; forward-only; FUSED.

DiLoCo consensus checkpoints are scored against the single-GPU emender reference
(which keeps training on GPU 0). Degradation(T) = `diloco_bpb(T) − ref_bpb(T)`
(reference interpolated at matched total tokens); **negative ⇒ DiLoCo better ⇒
SWA benefit holds**. The reference run-to-run noise band (max−min over its curve)
is the null scale for the effect size.

## Files

- `analyze_seed_race_i4.py` — matched-token analysis + overlay plot + verdict.
- `../../scripts/run_diloco_seed_race_i4.sh` — detached launch wrapper invocation.
- `../../scripts/score_seed_race_i4.sh` — offline scoring of both run-dirs.
- `reference_heldout_bpb.csv`, `diloco_i4_heldout_bpb.csv` — scored curves.
- `seed_race_i4_degradation.csv`, `seed_race_i4_verdict.json`,
  `seed_race_i4_overlay.png` — analysis outputs.
- `REPORT.md` — the read (track / beat / degrade) with effect sizes + confound audit.

## Confound stack (any "degrades" reading must clear ALL of these)

1. seed loaded with **full state incl. SF `z`** (`--resume`)
2. **plain averaging** beta=0 (not the divergent beta=0.9)
3. `gate_activation=silu` passed explicitly (else wrong activation silently)
4. **FUSED**, no eager (fused-guard on every rank)
5. checkpoints are **post-merge consensus** (save_every = k·diloco_k)
6. **matched TOTAL tokens** via the seed-aware piecewise formula
