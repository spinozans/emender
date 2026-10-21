# W4 chill modulation wash — exploratory

Status: prelaunch registration; no measurements yet. Task authority: `/home/erikg/emender-scratch/fix-fanout/taskW4-chillwash.md`. Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/w4-chillwash/`.

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

## Exploratory observations

Pending the fixed trajectory and pinned read. No observations or (a)/(b) classification are claimed before completion.
