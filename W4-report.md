# W4 chill modulation wash — exploratory

Status: **complete — exploratory observations only.** The fixed512-update trajectory and single14-case read completed on GPU2; **zero boundary collisions**, registered classification **(a)**. No gate verdict or promotion claim. Task authority: `/home/erikg/emender-scratch/fix-fanout/taskW4-chillwash.md`. Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/w4-chillwash/`.

## Registered design

One detached world1 trajectory on one leased free GPU from **2–7 only**. Cards **0/1 are excluded**; X12 T64 continues there. Reuse W3's verified wash-1 train-y fresh-state conversion, exact parent SHA256 `cb50c6586c91cb8f01e3c614b504ecc53dbad087a38af225b9e62ef762a24737`. Ordinary AdamW (.9,.95), eps1e-8, wd.01, fresh moments, warmup0, external cosine **2e-6 → 2e-8**, exactly512 updates. Reuse W3's exact pack IDs/order, tokens, target/valid/reset masks, target-normalized loss, precision and geometry: **154,386 targets / 3,571,182 inputs**. Authority SHA256 `55aca12a4952338e576909d2e52541b5b5807ad14a970410309f740eff510184`; pack SHA256 `f9b3a5e103c5d94f692fcd736dda95cfea75c78bc9614d247c37019f1aacc17e`.

Pin entry and fixed-final checkpoint; independently copy/hash the complete checkpoint **at read time, before any evaluation**. No intermediate capability reads, trajectory restarts, retries, or best-checkpoint selection.

## W4-only amended read protocol

The timestamped preregistration in `receipts/w4-chillwash/preregistration.json` applies **only to this exploratory read**. Frozen W3 evaluator/runner and strict fail-stop remain byte-identical for later verdict reads. `scripts/w4_stage_b_chill.py` is a separately bound derivative: same14 cases/order, greedy generation, budgets, tool/CLI/Pi/tokenizer identities, grading and model-immutability checks. The exact lexical private-text-in-events predicate now logs each matching private field's case ID, UTF8 SHA256, byte length, field index and occurrence count, marks that case **boundary-collision**, and continues to the **next case**, never retries. Underlying grades remain separate from collision flags. Trip text/private transcripts remain outside Git.

Exactly **one** Stage-B pass. The optional execution replica is **omitted** to keep the read to Stage-B plus a single saved-mode docNLL pass. docNLL uses the original frozen scorer functions and8-doc/198,295-target panel, trajectory-only; no numeric bounds. No multi-repetition scalar rule, min/max advantage threshold, gate verdict or promotion claim applies.

## Predictions registered before launch

- **(a)** If the chill run (2e-6) reads through all14 Stage-B cases with zero boundary collisions, the collision is attributable to the1e-5 modulation amplitude — dose-related, not recipe-fundamental.
- **(b)** If collisions still appear (any amplitude), the anneal perturbs the scratchpad/public boundary in an amplitude-independent way — a genuine behavioral finding about the recipe class.
- **(c)** Valid/correct counts are exploratory observations only. Print historical parent **12/6/1, 13/7/0, 12/6/1** and arm A **12/5/0** alongside, with no verdict applied.

These are the operator-requested exploratory attribution predictions, not causal identification from one trajectory/read. W3's case11 guard collision was cross-turn public/private lexical text reuse, not demonstrated transport leakage; that distinction remains relevant to either classification.

## Validation and evidence

Prelaunch CPU suite: **37 passed in9.98s**, including a14-case mocked instrument run that trips on case11 and still publishes case14, exact per-field collision digests, fixed cosine endpoints/one-fifth W3 schedule, coverage-only classification, create-once hash-checked pins, committed registration requirement, unchanged grading/generation AST and frozen W3 evaluator hash, and cards0/1 exclusion. Command/output: `receipts/w4-chillwash/tests-prelaunch.txt`.

Safety intent: **R07/NDP15** atomic publication, **R12** exact parent/state binding, **R14/NDP13** bounded execution, **R16** immutable evidence discipline from the resilient DiLoCo design/gap matrix. This is local world1 SFT, not resilient-system, Frontier or scale acceptance. No WG service/dispatcher/workers are used; the local GPU lease utility only provides resource exclusion.

## Launch (2026-10-21 00:58:25Z)

Committed/pushed prelaunch contract and harness: **e2e9b883**. Registration SHA256 **be5edaa3f6617ca83d72b96cb989fdb5768f747db1a3affea0c0da732dc08ebc**. Detached launcher **PID3228240 / PPID1 / starttime261025104**, lambda01; physical **GPU2**, UUID `GPU-d1b85657-fee1-7219-48a0-52b1b323f9c1`. Independent observer **PID3447419 / PPID1 / starttime261030626**, polling60s. Consoles: `/home/erikg/emender-scratch/w4-chillwash-console.txt` and `w4-chillwash-watcher.txt`.

Root-capacity evidence at actual launch: **504,919,502,848 free bytes**, above the inherited200GiB safety floor. First update exactly reproduced W3's first-pack loss **1.2679857436057442**,482 targets/6,677 inputs, actual=scheduled LR**2e-6**. This is telemetry, not a capability observation. `receipts/w4-chillwash/launch-receipt.json`, launcher/watcher/capacity/start/entry-pin and first-update copies preserve launch evidence. All14,070 Pi inventory payloads were reverified before launch, along with independent binary recount and replay of all512 W3 batches/masks.

## Completion and pin-at-read

The detached pipeline completed once at **2026-10-21 02:14:37Z**, exit0, **76m12s** after launch. Exactly512 updates, **3,571,182 inputs /154,386 targets**. Scheduled=actual LR at every step; endpoints2e-6 and2e-8; cumulative target-normalized objective exposure512, cumulative LR-weighted exposure**0.0005171200000000003**. Peak training HBM allocated36,214,916,096 bytes, reserved39,504,052,224 bytes. No restart, repeated read, trajectory selection or promotion.

Fixed-final checkpoint SHA256 **`cd9bd5efa3313774d1a2e869a5b4f4d2cc2659daa649126efc658708e3e8fe43`**. A separate read-time copy was full-hash verified and pinned at **02:03:28Z**, before Stage-B plan creation or model sampling. Primary Stage-B uses train mode; the docNLL saved-mode read is the **same ordinary-AdamW live, nonaveraged point**, not an SF x/y interpolation. Stage-B's before/after model fingerprints agree, all parameters BF16, no gradients; read peak HBM13,609,114,624 bytes.

The observer retained **77 occupancy snapshots**, with **zero foreign compute processes on selected GPU2**. Launcher and observer exited; lease released; GPU2 returned to4MiB/0%. X12 remained on0/1. Occupancy observations are sampled, not a claim of continuous external-process exclusion.

## Exploratory table — one Stage-B pass

`invalid` counts terminal reasons beginning `invalid_`, not14−valid. All14 cases were attempted and published in the registered order. Collision flags are separate from unchanged underlying grades; no scalar threshold is applied.

| Read | Valid /14 | Correct first action /14 | Invalid | Boundary-collision cases | End-to-end observations /14 |
|---|---:|---:|---:|---:|---:|
| **W4 chill, one greedy pass** | **10** | **5** | **1** | **0** | **1** |
| Historical parent, repetition1 | 12 | 6 | 1 | 0 | 1 |
| Historical parent, repetition2 | 13 | 7 | 0 | 0 | 1 |
| Historical parent, repetition3 | 12 | 6 | 1 | 0 | 1 |
| Historical arm A | 12 | 5 | 0 | 0 | 0 |

Historical rows are **context only**, not new reads or scalar rules. Optional exploratory execution replica: **omitted as registered**; no execution measurement is inferred.

| Case, registered order | Valid | Correct | Terminal reason | Boundary collision |
|---|---:|---:|---|---|
| copy-01024 | 1 | 0 | finished | no |
| bind-01024 | 1 | 1 | finished | no |
| path-01024 | 1 | 1 | turn_budget | no |
| copy-32768 | 1 | 1 | finished | no |
| bind-32768 | 0 | 0 | generation_budget | no |
| path-32768 | 1 | 1 | turn_budget | no |
| choice-read-explicit | 1 | 1 | finished | no |
| choice-grep-symbol | 1 | 0 | turn_budget | no |
| choice-bash-command | 1 | 0 | turn_budget | no |
| choice-edit-file | 1 | 0 | turn_budget | no |
| **choice-write-file (case11)** | **0** | **0** | **invalid_frame** | **no** |
| choice-process-start | 0 | 0 | generation_budget | no |
| choice-source-check | 0 | 0 | generation_budget | no |
| choice-fetch-url | 1 | 0 | turn_budget | no |

## Trip log and registered (a)/(b) classification

**Trip log: empty —0 matching private fields /0 boundary-collision cases across14 case records.** Durable log: `reads/stage-b/boundary-trips.jsonl`; exact empty-file SHA256 **`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`**. The publication audit independently recomputed the lexical predicate and UTF8 content digests from every saved case's actions/events, without serving, generating, regrading or replaying anything. Had a trip occurred, the registered log would contain case ID and content digest, not private text.

**Registered classification: (a)** — the2e-6 trajectory completed the14-case read with zero boundary collisions. In the operator's preregistered exploratory terms, this points to the1e-5 modulation amplitude as dose-related rather than recipe-fundamental. **(b) is not selected.**

Crucial qualification: case11 now terminated **invalid_frame**, without a valid first action; zero lexical collisions does **not** mean case11 produced a valid frame. This one-seed/one-read contrast is consistent with prediction(a), not a causal proof of amplitude attribution or an amplitude-independent mechanism. W3's original collision was cross-turn lexical text reuse. Counts and termination reasons above remain exploratory observations, with no quality verdict applied.

## Saved-mode docNLL — trajectory only

Exactly one pass of the frozen8-doc/**198,295-target** panel `5725d8206ac90e5ad4bbb4acd492bf4832f704ccb1d39615bae1c87fd55edc9f`, using the bound historical GPU scorer's unchanged loading/padding/scoring functions. Overall token-weighted mean NLL: **2.547767718597032**. No contemporary baseline was rescored; no relative-retention conclusion, threshold or stop rule is inferred.

| Document | Targets | Mean NLL |
|---|---:|---:|
| project_gutenberg | 32,768 | 3.303655902 |
| pre_1929_books | 26,829 | 3.168878779 |
| arxiv_papers | 32,768 | 1.345134335 |
| caselaw_access_project | 29,078 | 2.345354341 |
| ubuntu_irc | 32,768 | 2.296802389 |
| libretexts | 8,120 | 3.565925598 |
| news | 3,196 | 3.638834645 |
| stackexchange | 32,768 | 2.557840788 |

## Final validation and receipts

- **65 tests passed in8.53s**: W4 tests plus W3 exact masks/typed points, regime/serialization, capacity, occupancy guard, and SFT precision integration. `receipts/w4-chillwash/tests-final.txt`; Bash syntax and `git diff --check` also passed.
- Read-only `audit-read.py` verified every512 actual update's pack/record/sample IDs, input/target counts, scheduled/actual LR, cumulative exposure, finite telemetry; all three entry/final/read checkpoint pins; every14 ordered case row and matching private field; model immutability, saved docNLL coverage and all occupancy samples. `publication-audit.json`, status verified. No extra model read, scoring or generation occurred during publication.
- **263 raw evidence files** individually SHA256-bound in `raw-evidence-manifest.json`, SHA256 `ea3580c64c34956bf128d711b863d54c7d402d532fba36b7d8c6d759ac9a0c40`. Checkpoint payloads are separately fully verified in the audit. Private transcripts stay in the artifact directory; Git contains only measurements, fingerprints, digests and receipts.
- Preregistration/source bindings remained unchanged throughout; W3's original strict evaluator and runner remain byte-identical. Complete training log, checkpoint identities/pins,14-case summary, empty trip log, docNLL records, launch/terminal and audit evidence are in `receipts/w4-chillwash/`.

Scope remains exactly this local exploratory wash/read. No further run, repeat, promotion or scaling is authorized by these observations.
