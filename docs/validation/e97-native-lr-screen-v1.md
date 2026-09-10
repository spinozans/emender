# Native LR screen v1: execution plan

`configs/pi/e97-native-lr-screen-v1.json` freezes five new rates, in order:
`2e-4, 0.00047431158698290157, 1e-3, 1e-5, 2e-6`. Reuse the completed `5e-5`
control. Every new case starts fresh SR optimizer state from the same parent
live/y point, uses the same eight updates and exact sample schedule, and
consumes 1,490,478 total / 415,097 native-agent targets.

`prepare_e97_native_lr_screen.py` verifies retained baseline inventories and
renders the proven smoke launcher with exact-once anchors. Numerical training
code and flags other than LR are unchanged. Each case has separate outputs,
Triton caches and a fresh eight-rank process set. The wrapper additionally
captures lease acquisition into a checked assignment before `eval`, so a failed
acquisition cannot become a successful empty `eval`. No retry, promotion or
automatic sustained expansion is permitted. Any process or identity failure
stops the chain and retains its artifacts.

Evaluate two completed rates together, each in x and y, on the exact previously
frozen examples. The final pair repeats the existing 5e-5 control; this repeat
is retained separately, not substituted for the original baseline measurement.
The retained control worker and evaluator use the already qualified immutable
source exports. Per-case deadlines are 2,400 seconds with 30-second kill grace.
The whole bounded screen has a six-hour ceiling, not an unbounded controller.

This is a **small LR screen**, not the substantive training budget. Results are
for shortlisting acquisition/retention trade-offs, not final LR selection or
agent capability claims. Subsequent larger-exposure work targets tens of millions
of agent targets, with intermediate learning/retention/execution checks.

Architecture scope remains ADR-003 safety intent R07/R12, R14/NDP13, R16 and
NDP15 checkpoint atomicity. No elastic/native-data-plane/async/Frontier/ROCm
or communicator-shrink claim. No source commands are executed by this screen.

## Completed screen

`proc_6e35` exited successfully after 6,837 seconds (2026-09-10 UTC).
Controller source: `530fe54f2b055a022bfde935feef243fa1eff178`.
Evidence: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-lr-screen-v1`.
All five new trials completed eight updates, exact planned sample IDs/counts,
and complete finite BF16 optimizer checkpoints. This mechanical success does
not make the badly degraded high-rate checkpoints acceptable models.

Train/y results on the fixed panel (original, not repeated, control):

| LR | Development NLL | Conversation NLL | Tool-retention token accuracy | Valid development first frames |
|---|---:|---:|---:|---:|
| parent | 1.5552 | 1.6811 | 100% | 0/2 |
| 2e-6 | 1.2034 | 1.6714 | 100% | 0/2 |
| 1e-5 | **1.1177** | 1.6773 | 100% | 0/2 |
| 5e-5 | 1.1510 | 1.7075 | 100% | 2/2 |
| 2e-4 | 1.6111 | 2.0141 | 97.34% | 0/2 |
| 4.7431158698290157e-4 | 3.0929 | 2.8511 | 77.87% | 0/2 |
| 1e-3 | 15.9900 | 17.6210 | 8.06% | 0/2 |

Saved/x preserves the broad development/retention ranking. At 1e-5, fitting
first-frame validity is 2/2 for y and 1/2 for x, while development is 0/2 in
both. No rate matched a development argument payload, and no tools were
executed. The two-example generation panels are diagnostic, not reliable
capability rankings. The control repeat's x aggregate equals the original;
y differs only in development assistant NLL by +1.0900921352252624e-6, with
identical accuracy and generation counts. This is inference repeatability,
not the unmeasured optimizer-continuation comparison.

**Decision:** use 1e-5 provisionally for larger exposure because it gives the
best development likelihood while preserving retention. It is not declared
optimal, and the generation trade-off with 5e-5 remains open. Preserve all
trials, including the degraded 1e-3 case; no checkpoint is promoted. Move to
50M native-agent targets plus conversation/retention rather than another set
of eight-update pilots.
