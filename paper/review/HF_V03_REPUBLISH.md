# Republishing HuggingFace v0.3 — weights-only y-mode overwrite (republish-hf-v03)

**Task:** `republish-hf-v03`. Repair the broken HuggingFace v0.3 release by
**overwriting** the schedule-free *x-mode* `model.safetensors` with the verified
*y-mode* (training) weights, **in place on revision `v0.3`**, for all three
public repos. Author-approved public write (Erik Garrison). Scope: **weights
only** on `v0.3`; modeling code, `config.json`, and tokenizer left unchanged;
`v0.1`/`v0.2` not touched. GPU 0 only. REAL data / REAL forward only.

## TL;DR / Verdict

**The public v0.3 release now WORKS.** For all three repos, `model.safetensors`
on `@v0.3` was overwritten with the y-mode weights recovered via the
schedule-free `optimizer.train()` swap (root cause + recovery established in
`HF_V03_FIX.md`). The replacement was gated locally through the genuine bundled
`NdmForCausalLM.forward` **before** upload, and re-verified by a **clean-cache
post-upload readback** straight from the public hub via
`AutoModelForCausalLM.from_pretrained(repo, revision='v0.3', trust_remote_code=True)`:

| Model | published **x-mode** (before) | **y-mode** readback `@v0.3` (after) — nats / BPB | harness ref (nats / BPB) | gate |
|---|---:|---:|---:|:--:|
| Emender/E88 | 18.25 nats (CATASTROPHIC) | **2.559775 / 0.966133** | 2.559794 / 0.966140 | PASS |
| GDN | 101.70 nats (CATASTROPHIC) | **2.559748 / 0.966123** | 2.559748 / 0.966123 | PASS |
| M2RNN-CMA | 18.42 nats (CATASTROPHIC) | **2.547022 / 0.961320** | 2.547022 / 0.961320 | PASS |

The post-upload readback reproduces the live-harness held-out BPB to ≤2×10⁻⁵
nats, and was confirmed **twice** independently (§4). Only the **weights** changed: modeling code / `config.json` / tokenizer are
byte-identical to the prior `v0.3` (verified by SHA), and `v0.1`/`v0.2` resolve
to exactly the same commits as before.

---

## 1. Source of the corrected weights

The verified y-mode safetensors from `fix-hf-v03` (staged at
`/home/erikg/ndm/.wg-worktrees/agent-757/hf_v03_fix_staging/<repo>/`) were
reused. Those exports were full-canonical-slice verified to reproduce the
live-harness held-out BPB to ≤2×10⁻⁶ nats through the genuine bundled
`NdmForCausalLM.forward` (`HF_V03_FIX.md` §3b; raw
`agent-757/scripts/hf_v03_fix_verify_result.json`). The y-mode swap is the
documented schedule-free recovery: load the source training checkpoint that
still carries `optimizer_state_dict`, build `AdamWScheduleFree`,
`optimizer.load_state_dict(...)`, `optimizer.train()`.

Source checkpoints (still carrying optimizer state; the y-mode swap is
reproducible from them):

| Model | source `.pt` step | live-harness ref nats / BPB |
|---|---:|---:|
| E88 | 1,542,000 | 2.5597944649733653 / 0.9661400952828046 |
| GDN | 2,031,000 | 2.5597479088882387 / 0.9661225236765155 |
| M2RNN-CMA | 1,491,000 | 2.5470223021966170 / 0.9613195135013596 |

### Structural fidelity of the uploaded file

The published x-mode `model.safetensors` stored the tied output head explicitly
(`model.lm_head.weight`), giving 87 / 297 / 150 tensors for E88 / GDN / M2RNN.
The verified y-mode export had de-duplicated that tied tensor (shared storage),
so the uploaded file re-materialises `model.lm_head.weight =
model.embedding.weight.clone()` to reproduce the **exact published key set** —
only tensor **values** change (x-mode → y-mode). The bundled code ties
`model.lm_head.weight ← model.embedding.weight` (`_tied_weights_keys`), so this
is the correct tied head. Metadata records the y-mode provenance
(`source_state_dict = "model_state_dict (y-mode; schedule-free
AdamWScheduleFree optimizer.train() swap)"`, `ymode_export=true`,
`checkpoint_step = <scored step>`).

Builder + local gate: `scripts/hf_v03_republish_build.py` (raw
`scripts/hf_v03_republish_build_result.json`).

---

## 2. Pre-upload local gate (through the genuine bundled forward)

Each freshly built `model.safetensors` was staged beside the **published**
`@v0.3` code/config/tokenizer (copied byte-for-byte from the clean HF snapshot)
and loaded via `AutoModelForCausalLM.from_pretrained(stage_dir,
trust_remote_code=True)` — i.e. exactly what `@v0.3` resolves to after the
weights-only overwrite — then scored on the canonical slice through
`NdmForCausalLM.forward`:

| Model | from_pretrained | block-loss (nats) | gate (block ∈ [1.5,1.9]) | verdict |
|---|---|---:|:--:|:--:|
| E88 | OK | 1.809225 | PASS | SANE |
| GDN | OK | 1.605884 | PASS | SANE |
| M2RNN-CMA | OK | 1.710585 | PASS | SANE |

The block-losses match the verified `fix-hf-v03` reference exactly (E88 1.809225,
M2RNN 1.710585). All three gates PASS before any byte was uploaded.

---

## 3. The public write (weights only, in place on v0.3)

Script: `scripts/hf_v03_republish_upload.py` (approval-gated; adapted from
`scripts/publish_v03_public_hf.py`'s preserve-tag capture + unauthenticated
readback). HF commits are immutable, so "in place on `v0.3`" means `v0.3` now
resolves to weights-corrected content: `model.safetensors` was uploaded
**alone** to `main` (other files inherited unchanged from the parent commit),
the upload was read back at the new commit (LFS sha + size match; every
non-weight file SHA unchanged), then the `v0.3` tag was moved to the new commit.
`v0.1`/`v0.2` were captured before and asserted byte-identical after.

| Model | `v0.3` resolved SHA — old (x-mode) → new (y-mode) | new `model.safetensors` sha256 | code/config/tokenizer |
|---|---|---|:--:|
| E88 | `8a9fedafa01c37f88c2eb767df95dc9246640cbd` → `3cadd30532ed53c2250a4ece168a54eb74954262` | `201c425a58df2e70…` | unchanged |
| GDN | `a3c4c11cfaa2021e837d091216b269c192cad2b5` → `682d72a936a37d092d91513862038dc8178cf0fd` | `da2562af45fa61f1…` | unchanged |
| M2RNN-CMA | `aa2d6defb169c4cb9b5f740c17d733fd2c7f9a9e` → `67fd44127c5c26577e9dea6a5cb3ff2a0fb5a3eb` | `73873ee1b671278e…` | unchanged |

Remote LFS sha256 of the uploaded `model.safetensors` matches the local file for
all three; the six non-weight files (`config.json`, `configuration_ndm.py`,
`modeling_ndm.py`, `special_tokens_map.json`, `tokenizer_config.json`,
`tokenizer.json`) hash identically before and after the overwrite.

### v0.1 / v0.2 untouched (resolved SHA AND ref target unchanged)

| Model | v0.1 resolved (before = after) | v0.2 resolved (before = after) |
|---|---|---|
| E88 | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | `be77d1ee5b744ffa653e9b03abd5254d2ef8a41c` |
| GDN | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | `7395b6b6588726a3bca963aa7e6150e0971e71d6` |
| M2RNN-CMA | `8181b77803e130ffd78e37c33aa4d58c27e719c2` | `2e5f8f3be8a7c8ac42802485afb40d023874ea06` |

(The annotated-tag ref targets for v0.1/v0.2 were likewise asserted unchanged.)
Raw: `/tmp/republish-v03-upload-agent-764/summary.json`.

---

## 4. POST-UPLOAD readback from the public @v0.3 (mandatory)

Script: `scripts/hf_v03_republish_readback.py` (GPU 0). A **fresh, isolated HF
cache** (`HF_HOME=/tmp/republish-v03-readback-clean-cache`) forces a real
re-download of the `@v0.3` content as it resolves AFTER the overwrite; then, for
each repo, `AutoModelForCausalLM.from_pretrained(repo, revision='v0.3',
trust_remote_code=True)` runs the FULL canonical slice (ctx 2048 / stride 1024,
9,999,511-byte denominator, 2,616,009 tokens scored) through the genuine bundled
`NdmForCausalLM.forward`. The only environment shim is `ndm.models.* → elman`
(the documented local-verification shim; not a repo change). transformers
4.57.3 needs no `tie_weights` patch.

| Model | `from_pretrained(@v0.3)` | block-loss (nats) | mean nats/token | **Held-out BPB** | Δnats vs harness | gate |
|---|---|---:|---:|---:|---:|:--:|
| E88 | OK | 1.808465 | **2.559775** | **0.966133** | −1.91×10⁻⁵ | PASS |
| GDN | OK | 1.602250 | **2.559748** | **0.966123** | +0.0 | PASS |
| M2RNN-CMA | OK | 1.710585 | **2.547022** | **0.961320** | +0.0 | PASS |

All three load strict via `from_pretrained` from the freshly-downloaded public
`@v0.3` and reproduce the live-harness held-out BPB to ≤2×10⁻⁵ nats over the
full 2,616,009-token slice — **the public v0.3 release verifiably produces a
correct forward.** (Before the overwrite the same call gave 18.25 / 101.70 /
18.42 nats — the catastrophic x-mode forward; the x-mode `@v0.3` baseline was
re-confirmed on this box via the identical `from_pretrained` path before any
write.)

**Confirmed twice, independently.** The readback was run once when GPU 0 first
had headroom and again on a separately-wiped clean cache on a now-idle,
contention-free GPU 0 — both runs full-slice SANE with matching BPB (the only
movement is E88's 6th decimal, bf16 nondeterminism: 0.966142 vs 0.966133, both
≪ the 0.01-nats gate). Earlier OOM aborts during the run were pure GPU-0/GPU-1
memory contention from concurrent jobs (failures at `.to(cuda)` / activation
alloc), never a wrong forward; the readback is resumable and was retried until a
contention-free pass completed. Raw of the canonical (fresh, contention-free)
run: `scripts/hf_v03_republish_readback_result.json`; the first sane run is kept
at `scripts/hf_v03_republish_readback_result.gpu0.json`.

Raw: `scripts/hf_v03_republish_readback_result.json`.

---

## 5. Validation checklist

- [x] **y-mode weights reproduce verified-sane numbers locally before upload.**
      Pre-upload gate through the genuine bundled forward (published code + new
      weights via `from_pretrained`): E88 block 1.809225, GDN 1.605884, M2RNN
      1.710585 — all PASS, matching the `fix-hf-v03` reference (§2).
- [x] **v0.3 `model.safetensors` overwritten for all three; modeling/config/
      tokenizer unchanged.** v0.3 moved E88 `8a9fedafa…`→`3cadd30532…`, GDN
      `a3c4c11cf…`→`682d72a936…`, M2RNN `aa2d6defb…`→`67fd44127c…`; remote LFS
      sha256 matches local; the six non-weight files hash identically before/
      after (§3).
- [x] **v0.1/v0.2 SHAs unchanged.** Resolved SHA and annotated-tag ref target
      asserted byte-identical before and after for all three repos (§3).
- [x] **POST-UPLOAD readback from public `@v0.3` is sane (~0.966 BPB) for all
      three.** Clean-cache full-slice `from_pretrained(@v0.3)`: 0.966133 /
      0.966123 / 0.961320 BPB, gate PASS — confirmed twice independently (§4).
- [x] **`HF_V03_REPUBLISH.md` written; `paper/main.typ` NOT modified; no
      fabrication** (every number from a real GPU-0 forward; raw JSON alongside
      each script).

---

## 6. Note on v0.1 / v0.2

`v0.1` and `v0.2` were left **untouched** per the approval scope (and verified
unchanged, §3). They were published from the same x-mode export pipeline and are
therefore **likely also x-mode-broken** at inference — a separate follow-up
should re-verify and, if confirmed, apply the same y-mode overwrite. This task's
scope was `v0.3` weights only.

`paper/main.typ` was **NOT** modified. No fabrication: every number above is from
a real GPU-0 forward through the genuine bundled code, raw JSON alongside each
script.
