# X15 — two probes, preregistered before launch

Operator directive: `/home/erikg/emender-scratch/fix-fanout/taskX15-probes.md`.
Standalone lambda01 expressivity only. Physical card6 is GDN2 zero-shot;
physical card7 is scratch Muon@3x. X13 cards4/5 and X14 cards0–3 are untouched.
Exclusive local GPU leases fail closed on busy/held cards, no silent rebinding.
No WorksGood service/dispatcher or distributed training machinery is used.

## 1. GDN2 runtime zero-shot (card6)

Hash-pinned X11 aggregate-best champion: step482500, SHA256
`49a0f96fe7890ca9bb8a860698641ad8407dca412a3b38a01d3ede996c011cd0`.
T16 historical score: P1–P4=512 each, P5=318/512. Strict stock X11 model load;
stock FLA GDN2 bf16, no fp32 recurrence adapter. Trusted stage10/X11 builder,
frozen-pool generator, forward_batch and padded_forward only; NO stage12 import.
First reproduce **all16 historical T16 position counts exactly**, then evaluate
zero-shot T64. Only positional evaluator expected loss reshape/prefix fallback
changes from16 to64. No input/target/mask transformation; 66 tokens right-padded
to80 by the unchanged padder. No optimizer instantiated, no training updates.
Champion hash before/after, strict state keys and unchanged model state checked.
Frozen seed0 validation,512 sequences, per-position counts/CE all64; P1–P16
comparison columns mandatory, P17–P64 reported too. No checkpoint selection.

Registered predictions:
(a) If GDN2 holds its trained positions at approximately512 on T64, both
architectures share runtime-generalizes-beyond-training; runtime section symmetric.
(b) If trained positions collapse on longer contexts, horizon runtime behaviors
DIFFER: receipted asymmetry. Exact counts reported; intermediate/mixed outcomes
not forced into either branch. Compare trained P1–P4 separately, not aggregate
T64 accuracy diluted by unseen positions.
(c) P5+ columns always: does near-learned P5 consolidate zero-shot on longer input?
Report P5 relative to318; no fine-tuning or post-score extensions.

## 2. E97 hybrid-Muon@3x deep-first T64 (card7)

Scratch torch.manual_seed0; numpy.default_rng0 fresh procedural training batches.
E97 dim256/four layers,9,900,800 parameters,p61,vocab3721, batch32, uniform CE
P1–P64 (P1 INCLUDED, unlike X14). No ladder, carry supervision or checkpoint load.
Stage10-pattern loop and actual X10 train_update only; legacy bf16 state,
fp32 parameters/bf16 autocast, sequential Triton evaluation, clip1.
Unchanged X10 hybrid_muon.py: semantic full-matrix routing; FP32 momentum/NS5,
Nesterov0.95, quintic coefficients(3.4445,-4.7750,2.0315), Frobenius eps1e-7,
approximate RMS matching0.2*sqrt(max(rows,cols)); auxiliary AdamW(.9,.999),eps1e-8.
All-parameter weight decay0.01, including ignored no-weight-decay tags.
Both groups share LR: linear warmup5K to3e-3, cosine to3e-5 at1,000,000.

**Schedule ruling recorded before launch:** supervisor approved stretching the
cosine across the full1M ceiling, NOT reaching floor at500K then holding. This
matches historical T64 AdamW's full-ceiling schedule shape; X10b's multiplier,
warmup, peak/floor, optimizer and all-parameter decay are retained. X12 is a
historical comparison only, not a code dependency; X13's audit concern is not
silently imported as an established causal conclusion.

512-example frozen seed0 T64 validation, exact array SHA256 verified; evaluation
at0/every500/1M. Full per-position P1–P64 count/CE, aggregate, first>=50/first512,
maximum anywhere vs fixed-final count, contiguous perfect prefix and optimizer
metrics. Full training loss/LR/gradient norm/optimizer time per update.
Entry + aggregate-best + latest/final retention <=3 files plus atomic staging;
pruning receipts retain hashes. No metric/stagnation/all-perfect stop.1M ceiling
ONLY. No restart, retry, extra seed, extension, tuning or causal capacity claim.

Registered predictions:
(a) If any position reaches>=50/512, Muon@3x escapes the marginal at T64 where
historical AdamW+cosine could not: supports optimizer-dependent learning failure
at these operating points; direction-normalized updates can climb the1/64 foothold.
Transient first crossing is descriptive; fixed-final retention reported separately.
(b) If Muon also sits at exact-marginal through1M, failure is optimizer-robust
across these AdamW/Muon operating points, strengthening the horizon/foothold-share
account (both optimizers escaped atT16). A one-seed conditional result, not a
universal optimizer ranking or proof of impossibility.

**Exact-marginal signature:** report full500-step validation CE trajectory versus
ln3721=8.221747728346622; per-position CE, all64 counts, maximum count and material
positions in `marginal_signature.json`. Uniform joint-state chance1/3721 means
expected512/3721=0.1376 correct per position. Show training loss trajectory alongside.
Near-marginal CE is NOT proof of zero learning; the materiality bars and all64
columns carry verdicts. No post-score numerical equivalence tolerance or chance
hypothesis test invented; intermediate outcomes explicitly remain inconclusive.

## Registration, resource safety and wake

Before confirmatory launch: CPU/adjacent regressions; leased card7 disposable
seed99 peak-LR25-update T64 probe and bit-exact next-update model/optimizer reload;
all64 evaluation finite; executable project/dependency/pool/champion bindings;
commit + push registration and source. Registration immutable. Full model seed99
probe uses no confirmatory frozen pool and no score selection.

Wall/memory bounds source-bound in registration from disposable probe:
310887 seconds /8GiB allocated. External worker wall guard + lease heartbeat;
finite loss/gradient/eval checks. Infrastructure/time/memory failure is censored,
not optimizer failure; absent crossings right-censored at observed step.

Native watcher for supervisor arming:
`.venv/bin/python -m scripts.watch_invertible_stage15_probes --interval 900`
from `/home/erikg/emender`. Exit0 only for both bound DONE/result receipts;
exit1 failure/dead/reused process; `--once` exit2 pending. Writes growth report
and exact-marginal series on terminal wake; `--once --report` gives censored
interim snapshot. Providing a command is NOT supervisor-side arming. Supervisor
explicitly agreed to arm the native wake after launch report. No automatic retry.

On wake: audit source/pool/terminal/final-checkpoint hashes, full consecutive
finite loss/LR streams and full1M ceiling, generate growth report, independently
review, commit + push. Host reboot requires human rearming. Frontier queue and
resilient DiLoCo requirement IDs are not applicable to these standalone probes.
