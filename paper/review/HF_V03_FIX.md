# Fixing the HuggingFace v0.3 forward — root cause, patch, verification

**Task:** `fix-hf-v03`. Make the published `poietic-pbc/{emender-e88-1.3b,
gdn-1.3b, m2rnn-cma-1.3b}@v0.3` checkpoints produce a CORRECT forward (sane loss)
through the bundled `trust_remote_code` code, verified locally against the
known-good live-harness forward. **Nothing is pushed** (public re-upload is a
separate, approval-gated step). GPU 0 only. REAL verification only.

---

## TL;DR / Verdict

The broken v0.3 forward is **NOT** a compute-graph or config bug. The bundled
modeling code (`NdmForCausalLM` + the elman `LadderLM`/`M2RNNLM` forward) and
`config.json` are **correct**. The single defect is the **published
`model.safetensors`: they are schedule-free *x-mode* (eval-extrapolated) weights**,
which are catastrophic at inference. The usable *y-mode* (training) weights are
recovered only from the source training checkpoint's optimizer state — which the HF
artifact does **not** contain.

Proof (this task), all through the **genuine bundled `NdmForCausalLM.forward`** on
the full canonical slice (ctx 2048, stride 1024, 9,999,511-byte denominator):

| Model | published **x-mode** weights | re-exported **y-mode** weights (nats / BPB) | harness reference (nats / BPB) | Δ nats |
|---|---:|---:|---:|---:|
| E88 | 18.26 nats (CATASTROPHIC) | **2.559794 / 0.966140** | 2.559794 / 0.966140 | +0.000000 |
| GDN | 101.70 nats (CATASTROPHIC) | **2.559746 / 0.966122** | 2.559748 / 0.966123 | −0.000002 |
| M2RNN-CMA | 18.42 nats (CATASTROPHIC) | **2.547022 / 0.961320** | 2.547022 / 0.961320 | +0.000000 |

Through the *same* bundled code path, re-exported **y-mode** weights reproduce the
live-harness reference **to ≤2×10⁻⁶ nats** (a strict `0 missing / 0 unexpected`
load), while the published **x-mode** weights give 18–102 nats. This isolates the
defect to the **weights**, not the forward.

> **Consequence for the task premise.** The task brief states "it is a
> COMPUTE-GRAPH / CONFIG bug … NOT bad weights" and "Do NOT touch the weights (they
> are correct)." The evidence below contradicts that premise: the weights are the
> problem, and **no edit to `modeling_ndm.py` or `config.json` can fix an x-mode
> weight tensor.** The real fix is a **weight re-export**, which necessarily changes
> the file SHA. This is reported faithfully rather than forcing a code/config edit
> that cannot work.

---

## 1. Root cause (precise)

**The v0.3 export froze the schedule-free `model_state_dict` in x-mode and shipped
no optimizer state.**

These three runs were trained with the **schedule-free** optimizer
(`AdamWScheduleFree`; each run's `args.json` has `"optimizer": "schedulefree"`).
Schedule-free keeps two parameter views:
- **x-mode** — the eval-extrapolated weighted average the optimizer exposes after
  `optimizer.eval()`. `train.py` calls `optimizer.eval()` before checkpointing, so
  the saved `model_state_dict` is **x-mode**.
- **y-mode** — the training weights the optimizer actually evaluates gradients at,
  recovered by loading `optimizer_state_dict` and calling `optimizer.train()` (the
  swap documented in `elman/generate.py:130-147` and used by
  `scripts/measure_pile_bpb_elman.py:161-183`).

For these architectures the x-mode view is **catastrophic at inference** (~18–102
nats/token), while y-mode matches the training loss (~2.6 nats). The v0.3 HF export
saved the bare x-mode `model_state_dict` (87 model tensors, **zero** optimizer /
schedule-free-state keys), so **y-mode cannot be recovered from the HF download at
all.**

### Why it is the weights and not the config / compute graph

Three independent discriminators (prior task `e88-heldout-hf`, reconfirmed here):

1. **Config flag is not the cause.** Loading the HF v0.3 safetensors into the
   *known-good* live-harness `LadderLM` gives **17.43 nats**, and the value is
   **identical for `r_h_mode="none"` and `r_h_mode="auto"`**
   (`scripts/e88_rootcause_result.json`). The config flag the HF wrapper passes raw
   (`r_h_mode="auto"`) changes nothing.
2. **Same good forward, two weight sources, both broken.** Through the identical
   harness forward: source `.pt` `model_state_dict` *without* the swap → **18.05
   nats**; HF safetensors → **17.81 nats** (`scripts/e88_export_vs_source_result.json`).
   Both are x-mode/broken; the defect travels with the **weights**.
3. **Correct weights through the bundled code are sane.** This task: loading
   **y-mode** weights into the **genuine bundled `NdmForCausalLM`** gives a strict
   `0 missing / 0 unexpected` load and **block-loss 1.8092**, exactly the harness
   reference — see §3. The bundled forward + config are fine.

The HF safetensors also differ from the nearest source `.pt` on 75/87 tensors
(different retained step), but that is immaterial — both are x-mode.

---

## 2. The known-good reference (ground truth)

`scripts/measure_pile_bpb_elman.py` (live elman harness, y-mode swap applied),
canonical slice, ctx 2048 / stride 1024, full UTF-8 byte denominator 9,999,511
(`paper/review/E88_HELDOUT_HARNESS.md`, `paper/review/heldout_harness_json/`):

| Model | step scored | block-loss (nats) | mean nats/token | **Held-out BPB** |
|---|---:|---:|---:|---:|
| E88 | 1,542,000 | 1.8092 | 2.5598 | 0.9661 |
| GDN | 2,031,000 | 1.6022 | 2.5597 | 0.9661 |
| M2RNN-CMA | 1,491,000 | 1.7106 | 2.5470 | 0.9613 |

These three checkpoint steps are still present in the source run directories and
**still carry `optimizer_state_dict`**, so the y-mode swap is reproducible. (The
*nominal* published steps — E88 1,524,000 / GDN 1,998,000 / M2RNN 1,467,000 — have
since been rolled off by the live runs' checkpoint pruning; the held-out BPB is
stable to a few ten-thousandths across these adjacent converged steps.)

---

## 3. Verification — y-mode re-export reproduces the harness through the bundled code

Script: `scripts/hf_v03_fix_verify.py` (GPU 0 only; `CUDA_VISIBLE_DEVICES=0` pinned
before torch). For each repo it (A) loads the **published** v0.3 safetensors into the
genuine bundled `NdmForCausalLM` and confirms the catastrophic forward; (B) applies
the schedule-free y-mode swap from the source `.pt` onto the bundled model's inner
module and measures the **full canonical-slice BPB through
`NdmForCausalLM.forward`**; (C) stages a corrected repo dir and reloads it via
`AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)`.

Raw results: `scripts/hf_v03_fix_verify_result.json`; log
`scripts/hf_v03_fix_verify.log`.

### 3a. Published x-mode weights through the bundled forward (bug reproduced)

| Model | strict load | block-loss (nats) | 5-window (nats) | verdict |
|---|---|---:|---:|---|
| E88 | 0 missing / 0 unexpected | 17.56 | 18.26 | CATASTROPHIC |
| GDN | 0 missing / 0 unexpected | 103.42 | 101.70 | CATASTROPHIC |
| M2RNN-CMA | 0 missing / 0 unexpected | 18.37 | 18.42 | CATASTROPHIC |

### 3b. Re-exported y-mode weights through the bundled forward (fixed — FULL slice)

| Model | strict load | block-loss | mean nats/token | BPB | Δnats vs harness | match <0.01 |
|---|---|---:|---:|---:|---:|:--:|
| E88 | 0 missing / 0 unexpected | 1.8092 | 2.559794 | 0.966140 | +0.000000 | ✓ |
| GDN | 0 missing / 0 unexpected | 1.6037 | 2.559746 | 0.966122 | −0.000002 | ✓ |
| M2RNN-CMA | 0 missing / 0 unexpected | 1.7106 | 2.547022 | 0.961320 | +0.000000 | ✓ |

(2,616,009 tokens scored per model; same canonical slice / byte denominator as the
harness.) The y-mode swap (`schedulefree.AdamWScheduleFree` →
`optimizer.load_state_dict(optimizer_state_dict)` → `optimizer.train()`) makes the
**bundled** `NdmForCausalLM.forward` reproduce the harness to ≤2×10⁻⁶ nats — the
bundled forward + config are confirmed correct; only the published weights were wrong.

### 3c. Staged corrected dirs + `from_pretrained` reload + generation

Staging dirs (local, NOT pushed): `hf_v03_fix_staging/{emender-e88-1.3b,gdn-1.3b,
m2rnn-cma-1.3b}/` — each holds `config.json` (+ `num_hidden_layers` added for
`generate`), the **packaging-patched** `configuration_ndm.py` / `modeling_ndm.py`,
tokenizer files, and the **re-exported y-mode `model.safetensors`** (saved with
`format: pt` metadata). Each loads via
`AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)`:

| Model | from_pretrained | block-loss (nats) | 5-window (nats) | sane | generation |
|---|---|---:|---:|:--:|:--:|
| E88 | OK | 1.8073 | 2.0427 | ✓ | ✓ (coherent text) |
| GDN | OK | 1.6073 | 1.9045 | ✓ | ✓ (coherent text) |
| M2RNN-CMA | OK | 1.7106 | 1.9947 | ✓ | ✓ (coherent text) |

Generation greedy-decodes fluent English (e.g. E88: *"The history of science is that
it is the most important science in the world…"*). The packaging patch (vendored
`ndm`→`elman` import fallback, `tie_weights(self, *args, **kwargs)`,
`all_tied_weights_keys`, `num_hidden_layers`) is what lets a clean
`from_pretrained` run on this box's `transformers` 5.9.0; it changes no numerics.

---

## 4. The patch, per repo

There is **no `config.json` edit and no `modeling_ndm.py` edit that fixes the
forward** — an x-mode weight tensor cannot be corrected by code. The required patch
for every repo is identical:

1. **Re-export y-mode `model.safetensors` (the fix).** Load the source training
   checkpoint that still carries `optimizer_state_dict`, apply the schedule-free
   swap (`optimizer.train()`), and export those weights. Sanity-gate at ~2.6 nats
   (block-loss ~1.6–1.8) before publishing. Implemented + verified by
   `scripts/hf_v03_fix_verify.py`; staged under `hf_v03_fix_staging/<repo>/`.

Two **packaging** patches are additionally required for a *clean-machine*
`from_pretrained` (they do not affect numerics; they are why a bare
`AutoModelForCausalLM.from_pretrained(repo, …)` does not even run on a fresh box —
see `E88_HELDOUT_HF.md` Layers 1–2):

2. **Vendor the model source.** `modeling_ndm.py` does
   `importlib.import_module("ndm.models.*")` against a private package. Vendor
   `ladder_lm.py` / `m2rnn_baseline.py` (+ deps) beside `modeling_ndm.py`, or make
   the repo pip-installable, so the import resolves without the private `ndm`/elman
   tree. (For this task's local verification the `ndm.models.*` names are shimmed
   onto the installed elman package.)
3. **Make `NdmForCausalLM` transformers-version robust.** On `transformers` 5.9.0
   the loader calls `model.all_tied_weights_keys` and
   `tie_weights(missing_keys=…, recompute_mapping=False)`; the shipped class lacks
   the attribute and uses `tie_weights(self)`. Add `all_tied_weights_keys` and a
   `tie_weights(self, *args, **kwargs)` signature (or pin a compatible
   `transformers` in the model card).

---

## 5. Recommended publish approach + paper-citation implication

Two options were on the table:

- **(A) Patch the v0.3 code files in place** (weights/SHA unchanged, the paper
  citation `@v0.3` still resolves). **Not viable** — the bug is in the *weights*,
  not the code. Editing `modeling_ndm.py`/`config.json` cannot change an x-mode
  tensor into a y-mode tensor; the forward would stay broken.
- **(B) Cut a new revision (e.g. `v0.3.1`)** with re-exported y-mode weights (+ the
  two packaging fixes). The weights change ⇒ `model.safetensors` SHA changes ⇒ the
  paper citation must be updated to the new revision.

**Recommendation: (B) — publish a new revision `v0.3.1`.** Because the fix
necessarily changes the weight tensors, the file SHA changes no matter what; there
is no SHA-preserving in-place patch. Keep `v0.3` as-is for provenance, publish
`v0.3.1` with the verified y-mode weights + vendored code + transformers-robust
class, and update the paper's checkpoint citation from `v0.3` to `v0.3.1`. The
held-out BPB the new revision reproduces (E88/GDN ~0.966, M2RNN ~0.961) is what the
paper's reproducibility claim should point at.

*(Publishing is approval-gated and is handled separately — nothing here is pushed.)*

---

## 6. Compliance with the task's "Do NOT" list

- **No public push.** All work is local: `scripts/hf_v03_fix_verify.py`,
  `scripts/hf_v03_fix_verify_result.json`, `hf_v03_fix_staging/`. The
  `poietic-pbc/*@v0.3` repos were only **read**.
- **v0.1 / v0.2 untouched.** Not accessed.
- **Published weights untouched.** The v0.3 safetensors were read-only; the y-mode
  re-export is a *new* local file in a staging dir.
- **`paper/main.typ` NOT modified.**

---

## 7. Validation checklist

- [x] Root cause stated precisely: published v0.3 safetensors are schedule-free
      **x-mode** weights (no optimizer state shipped) → catastrophic forward;
      bundled code/config are correct. Evidence: identical loss for
      `r_h_mode` none/auto; same good forward broken for both x-mode weight sources;
      y-mode through the bundled code reproduces the harness.
- [x] Corrected local staging dir per repo; loads via `trust_remote_code`
      (`from_pretrained` OK for all three, §3c); y-mode weights load strict
      (0 missing / 0 unexpected).
- [x] Verified sane loss on the canonical slice matching the harness within <0.01
      nats for ALL THREE (actually ≤2×10⁻⁶): E88 2.559794 / 0.966140, GDN 2.559746
      / 0.966122, M2RNN 2.547022 / 0.961320 (§3b). Generation confirmed working (§3c).
- [x] `paper/review/HF_V03_FIX.md` written with root cause + patch + verification +
      recommended publish approach & citation implication.
- [x] No public HF push; v0.1/v0.2/weights untouched; `main.typ` NOT modified.

_Generated from REAL GPU-0 runs: `scripts/hf_v03_fix_verify.py` (+ prior-lineage
discriminators `scripts/e88_rootcause_test.py`, `scripts/e88_export_vs_source.py`).
Raw JSON alongside._
