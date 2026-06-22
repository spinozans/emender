# SF-DiLoCo P4 Outer Regime Report

Task: `sf-diloco-p4`  
Run date: 2026-06-22  
Run root: `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes`

## Setup

All four regimes were run from scratch at matched tokens on the local 8-GPU
box, using clean sequential 2-GPU leases from `scripts/gpu_lease.sh acquire
--no-wait`. No arms were co-run.

Shared configuration:

- Model: E97, dim 1792, 216 heads, state 32, depth 11, MLP ratio 2.2623.
- Data: real Pile text at `/home/erikg/elman/data/pile.txt`.
- Tokenizer: `p50k_base`.
- Precision/kernel: bf16, `--use_triton 1`; every run printed fused-guard
  `NO eager fallback` on both ranks.
- DiLoCo: `K=250`, `steps=1100`, `batch_size=4`, `chunk_size=2048`,
  `world_size=2`.
- Held-out tensor: `experiments/lb_compare_20260613/heldout_p50k_2048.pt`,
  regenerated in this worktree from the fixed lb-compare builder. It contains
  64 chunks x 2049 tokens, 131072 scored tokens, and
  `bytes_per_token=3.878128`.
- Held-out scoring mode: ScheduleFree eval/averaged basis (`mode=x`), matching
  the P2 comparator logs.

Regimes:

- A: ScheduleFree inner + plain average outer:
  `--diloco_outer_optimizer avg --diloco_outer_lr 1.0 --diloco_outer_beta 0.0`
- B: ScheduleFree inner + fixed-momentum outer, P2 best matched-gain momentum
  config:
  `--diloco_outer_optimizer momentum --diloco_outer_lr 0.5 --diloco_outer_beta 0.5`
- C: ScheduleFree inner + SF-SGD outer, export x:
  `--diloco_outer_optimizer sfsgd --diloco_export_basis x --diloco_outer_lr 1.0 --diloco_outer_beta 0.1`
- D: ScheduleFree inner + SF-SGD outer, export y:
  `--diloco_outer_optimizer sfsgd --diloco_export_basis y --diloco_outer_lr 1.0 --diloco_outer_beta 0.1`

## Held-Out BPB

Matched-token final held-out BPB on the fixed disjoint lb-compare tensor:

| Regime | Final held-out BPB | Final held-out CE | Curve BPB @ step 1000 | Fused guard |
| --- | ---: | ---: | ---: | --- |
| A avg | **2.0361** | **5.4733** | **2.028718** | both ranks |
| B momentum beta=0.5 lr=0.5 | 2.0909 | 5.6205 | 2.097485 | both ranks |
| C sfsgd export x | 3.4744 | 9.3395 | 2.771051 | both ranks |
| D sfsgd export y | 2.0426 | 5.4908 | 2.036803 | both ranks |

Held-out curve checkpoints:

| Regime | 4.098M tok | 8.196M tok | 12.294M tok | 16.392M tok |
| --- | ---: | ---: | ---: | ---: |
| A avg | 2.426505 | 2.130690 | 2.045954 | **2.028718** |
| B momentum beta=0.5 lr=0.5 | 2.898700 | 2.230354 | 2.144774 | 2.097485 |
| C sfsgd export x | 2.463686 | 2.251456 | 2.364541 | 2.771051 |
| D sfsgd export y | 2.433242 | 2.137712 | 2.054900 | 2.036803 |

## Post-Sync Shock

Shock is computed from logged training windows as:
`loss_at_merge_step - loss_at_previous_logged_step`. Recovery is the first later
logged step whose loss is at or below the previous logged pre-merge loss.

| Regime | Mean jump | Max jump | Mean positive jump | Max recovery steps |
| --- | ---: | ---: | ---: | ---: |
| A avg | -0.07798 | 0.08430 | 0.08430 | 50 |
| B momentum beta=0.5 lr=0.5 | -0.02522 | 0.15590 | 0.10555 | 75 |
| C sfsgd export x | -0.09976 | -0.01850 | 0.00000 | 50 |
| D sfsgd export y | 0.02754 | 0.20070 | 0.11423 | 50 |

Per-merge jumps:

| Regime | Merge 1 | Merge 2 | Merge 3 | Merge 4 | Final merge |
| --- | ---: | ---: | ---: | ---: | ---: |
| A avg | -0.1414 | -0.0227 | -0.0679 | -0.2422 | 0.0843 |
| B momentum beta=0.5 lr=0.5 | -0.0456 | -0.1769 | 0.1559 | 0.0552 | -0.1147 |
| C sfsgd export x | -0.0708 | -0.0185 | -0.1345 | -0.0983 | -0.1767 |
| D sfsgd export y | -0.0745 | -0.1305 | 0.0732 | 0.2007 | 0.0688 |

The C run is the cautionary case: post-sync train loss looked smooth, but held-
out BPB degraded badly. This is why P4 should be judged by matched-token
held-out BPB, not short-run training loss.

## Verdict

Plain average outer remains the best outer optimizer for SF-DiLoCo-SF in this
P4 comparison.

Reason:

- A has the lowest final held-out BPB: 2.0361.
- D export-y SF-SGD is close at 2.0426, but does not beat A and has the largest
  observed positive merge jump, 0.2007.
- B fixed momentum trails A on every held-out checkpoint and final BPB.
- C export-x SF-SGD is unacceptable by held-out BPB despite smooth training loss.

Recommended Frontier DiLoCo recipe remains:

```text
--optimizer schedulefree
--diloco
--diloco_k 250
--diloco_outer_optimizer avg
--diloco_outer_lr 1.0
--diloco_outer_beta 0.0
```

Deployment is still gated on Frontier ROCm readiness; this local P4 result only
selects the local SF-DiLoCo-SF outer optimizer.

## Artifacts

- Launcher: `scripts/launch_sf_diloco_p4.sh`
- Analyzer: `scripts/analyze_sf_diloco_p4.py`
- Summary JSON: `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes/summary.json`
- Logs:
  - `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes/logs/A_avg.log`
  - `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes/logs/B_momentum_beta05_lr05.log`
  - `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes/logs/C_sfsgd_export_x.log`
  - `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes/logs/D_sfsgd_export_y.log`
- Held-out curves:
  - `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes/A_avg_heldout_curve.csv`
  - `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes/B_momentum_beta05_lr05_heldout_curve.csv`
  - `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes/C_sfsgd_export_x_heldout_curve.csv`
  - `/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes/D_sfsgd_export_y_heldout_curve.csv`

## Validation

- `bash -n scripts/launch_sf_diloco_p4.sh`: pass.
- `python -m py_compile scripts/analyze_sf_diloco_p4.py`: pass.
- `python -m pytest tests/test_diloco_merge.py -q`: 11 passed.
- All four regimes completed from scratch at matched tokens.
- All four regimes printed fused-guard `NO eager fallback` on ranks 0 and 1.
- Held-out BPB and post-sync shock/recovery metrics are reported above.
