# E88 / GDN / M2RNN-CMA — Held-out Pile BPB via the HuggingFace v0.3 release

**Task:** `e88-heldout-hf`. Measure held-out Pile bits-per-byte for the three v0.3
models through the HuggingFace bundled modeling code (`trust_remote_code`), GPU 0
only (GPUs 1–7 were training). REAL measurement; **a broken forward is reported as
a blocker, never faked.**

## Verdict: BLOCKER — the HF v0.3 forward is broken for all three models

The genuine HF v0.3 modeling code (`NdmForCausalLM` + the `ndm.models.*` forward it
builds) loading the HF v0.3 `model.safetensors` **strict (0 missing / 0 unexpected
keys)** produces **worse-than-random** loss on the canonical slice. Worse-than-random
is ~10.83 nats/token (ln of vocab 50281); a correct forward is ~2.6. **No BPB is
published** for any of the three — the numbers below are the exact broken values, as
required by the sanity gate.

| Model | revision / step | HF-path mean nats/token | sanity gate |
|---|---|---:|:--:|
| `poietic-pbc/emender-e88-1.3b` | v0.3 / 1,524,000 | **19.62** (100 windows) / 18.25 (5-window gate) | **FAIL (worse-than-random)** |
| `poietic-pbc/gdn-1.3b`         | v0.3 / 1,998,000 | **101.72** (5-window gate) | **FAIL (absurd)** |
| `poietic-pbc/m2rnn-cma-1.3b`   | v0.3 / 1,467,000 | **18.42** (5-window gate) | **FAIL (worse-than-random)** |

Sanity gate = mean nats/token < 5.0 (correct forward ~2.6). All three are far above
random → **FAIL**. Because the forward is broken, **E88 held-out BPB cannot be
compared to its train-loss 0.974** from this path; that comparison is withheld.
(The sibling task `e88-heldout-live`, using the live training harness with the
y-mode fix described below, did obtain sane values — E88 ~2.49 nats, GDN 2.5597 /
0.9661 bpb, M2RNN 2.5470 / 0.9613 bpb — but those are NOT the HF-release path and
belong to that task's report `E88_HELDOUT_HARNESS.md`.)

---

## Canonical slice (verified byte-identical to the rest of the panel)
- source: `/mnt/nvme2n1/erikg/pile.txt`, byte_offset 1000000001956, byte_length 9999511
- total_bytes (BPB denominator if a forward were valid): **9,999,511**
- sha256: `3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a` — **verified at run time**
- cached copy scored: `/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt`
- decode for tokenization: utf-8 strict (9,910,572 chars); each model uses its own tokenizer (vocab 50281).

## Method
- **GPU 0 only** (`CUDA_VISIBLE_DEVICES=0` hard-pinned before importing torch). Device: NVIDIA RTX 6000 Ada. dtype bf16 (training dtype).
- **Context 2048, strided sliding window stride 1024** (panel protocol). The 5-window
  quick gate and the 100-window E88 run both score with full left context.
- **BPB would be** `total_NLL_nats / (9,999,511 × ln 2)` (no 3.92 constant) — not
  computed/published because the forward fails the sanity gate.
- Scripts (all REAL runs, GPU 0): `scripts/e88_hf_pathmeasure.py` (E88 100-window:
  19.62 nats), `scripts/e88_hf_quickgate_all.py` (all three 5-window),
  `scripts/e88_rootcause_test.py` (config-vs-weights discriminator). Raw outputs:
  `scripts/e88_hf_pathmeasure_result.json`, `scripts/e88_hf_quickgate_all_result.json`,
  `scripts/e88_rootcause_result.json`.

---

## Diagnosis (per the expanded mandate)

The HF v0.3 `trust_remote_code` release fails in **three layers**. The first two are
packaging/compat (fixable in the repo); the third is the **fatal one — wrong weights.**

### Layer 1 — not self-contained (blocks any clean machine)
`modeling_ndm.py` builds the model by doing, at construction time:
```python
importlib.import_module("ndm.models.ladder_lm")       # LadderLM (E88, GDN)
importlib.import_module("ndm.models.m2rnn_baseline")  # M2RNNLM (M2RNN-CMA)
```
i.e. the "bundled" custom code is a thin wrapper that **requires a private `ndm`
package already installed** in the runtime (its own docstring: "expects the `ndm`
source package to be installed … the private staging Docker smoke installs the
repository before loading"). On any env without it, a plain
`AutoModelForCausalLM.from_pretrained(repo, revision="v0.3", trust_remote_code=True)`
raises `ModuleNotFoundError: No module named 'ndm'`. (The real forward lives in
`/home/erikg/elman`, package `elman`; the in-tree `ndm/` is the same `LadderLM` source.)

### Layer 2 — transformers API drift (two breaks, even with `ndm` importable)
This box's training venv has `transformers 5.9.0`; the shipped `NdmForCausalLM`
predates that API. `from_pretrained`'s `_finalize_model_loading` calls:
- `model.all_tied_weights_keys` → `AttributeError` (never defined), then
- `model.tie_weights(missing_keys=..., recompute_mapping=False)` →
  `TypeError: tie_weights() got an unexpected keyword argument 'missing_keys'`
  (shipped signature is `tie_weights(self)`).

### Layer 3 — ROOT CAUSE (fatal): the HF safetensors are schedule-free **x-mode** weights
After satisfying Layers 1–2 by running the **genuine HF modeling code** (config +
`NdmForCausalLM` + the real `ndm.models.*` forward), only **bypassing** transformers'
incompatible load finalizer and loading the HF safetensors strict (0 missing / 0
unexpected), the forward still gives **19.62 nats/token** for E88 — worse than random.

A discriminator test (`scripts/e88_rootcause_test.py`) pins the cause to the
**weights**, not the config or compute graph:
- Loading the **HF v0.3 safetensors** into the **known-good live-harness `LadderLM`**
  (the exact code the sibling `e88-heldout-live` validated at ~2.49 nats) still gives
  **17.43 nats/token**, and the value is **identical for `r_h_mode="none"` and
  `r_h_mode="auto"`** — so the config flag the HF wrapper passes raw is NOT the cause.
- The HF `model.safetensors` contains **87 keys, ZERO optimizer / schedule-free-state
  keys** — model weights only.

These runs use the **schedule-free optimizer**, which saves the **x-mode**
(eval-extrapolated) weights as `model_state_dict`. Those x-mode weights are
catastrophic at inference; the usable **y-mode** (training) weights are recovered
only by loading the **optimizer state** and calling `optimizer.train()` (the fix
documented and used by `e88-heldout-live` / `scripts/measure_pile_bpb_elman.py`).
**The HF v0.3 artifact exported the bare `model_state_dict` (x-mode) and no optimizer
state**, so y-mode recovery is impossible from the HF download alone. This is why the
HF forward is worse-than-random for all three:
- E88 18.25–19.62 nats, GDN 101.72 nats, M2RNN-CMA 18.42 nats — all FAIL.

**Exact root cause:** the v0.3 HF export froze the schedule-free **x-mode** weights
(`model_state_dict`) without applying the y-mode swap, and shipped no optimizer state
to enable it. The forward code and config are fine; the published weights are not
inference-usable.

This is corroborated directly by the elman harness code and the checkpoint contents:
- `elman/generate.py:130–145` documents it verbatim: "the x-mode (eval) extrapolated
  weights produce catastrophic loss at inference (~20 nats) … train.py saves x-mode
  (via `optimizer.eval()` before save), so we must swap back to y-mode via
  `optimizer.train()`", gated on `optimizer == 'schedulefree' and 'optimizer_state_dict' in ckpt`.
- E88's source training run used `optimizer: "schedulefree"` (its `args.json`), and the
  source `.pt` checkpoints contain both `model_state_dict` (87 tensors) **and**
  `optimizer_state_dict` — so the y-mode swap is recoverable from the source `.pt`.
- A direct control (`scripts/e88_export_vs_source.py`): loading the **source `.pt`
  `model_state_dict`** (NO swap) into the known-good forward gives **18.05 nats**, and
  the **HF safetensors** give **17.81 nats** — both x-mode/broken. The HF safetensors
  carry **only** the 87 model tensors (zero optimizer state), so unlike the source
  `.pt`, the HF artifact cannot be y-mode-recovered at all. (The HF vs source weights
  differ on 75/87 tensors, consistent with different checkpoint steps — HF step
  1,524,000 vs the nearest retained source step 1,536,000 — but that is immaterial:
  both are x-mode.)

---

## Fix — PREPARED, APPROVAL-GATED (nothing pushed to `poietic-pbc/*@v0.3`)

A correct public re-upload requires **re-exporting the weights**, plus the packaging
fixes. None of this has been pushed.

1. **Re-export y-mode weights (the essential fix).** For each model, load the source
   training checkpoint that still has optimizer state (e.g. E88
   `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/…/checkpoint_step_*.pt`,
   ~7.6 GB and includes optimizer state, vs the 87-tensor HF safetensors), apply the
   schedule-free y-mode swap (`optimizer.train()` per `generate.load_model` /
   `measure_pile_bpb_elman.py`), and export THOSE weights to `model.safetensors`.
   Sanity-gate the re-export at ~2.6 nats before publishing.
2. **Bundle the model source** so `modeling_ndm.py` imports it relatively instead of
   `importlib.import_module("ndm…")` against a private package (vendor `ladder_lm.py`
   / `m2rnn_baseline.py` + deps next to `modeling_ndm.py`, or make the repo pip-installable).
3. **Make `NdmForCausalLM` transformers-version robust**: `tie_weights(self, *args, **kwargs)`
   and provide `all_tied_weights_keys` (or pin a compatible `transformers` in the model card).

A weight re-export (#1) is mandatory — without it the public v0.3 forward is broken
regardless of packaging. The author will approve the publish separately.

---

## Validation
- [x] Ran on GPU 0 only (`CUDA_VISIBLE_DEVICES=0`); HF v0.3 `trust_remote_code` load path used (genuine `NdmForCausalLM` + `ndm.models.*` forward + HF v0.3 safetensors; only transformers' incompatible load-finalizer bypassed, after diagnosing it)
- [x] Each model: mean nats/token on the canonical slice + sanity gate evaluated — **all FAIL (worse-than-random); BPB withheld because the forward is broken**
- [x] Real numbers only; broken forward reported as a blocker with exact nats values (E88 19.62, GDN 101.72, M2RNN 18.42), never faked
- [x] `paper/review/E88_HELDOUT_HF.md` written; `paper/main.typ` NOT modified

_Generated from REAL GPU-0 runs: `scripts/e88_hf_pathmeasure.py`, `scripts/e88_hf_quickgate_all.py`, `scripts/e88_rootcause_test.py` (raw JSON alongside)._
