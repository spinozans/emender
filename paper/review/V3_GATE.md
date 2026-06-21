# V3 Synthesis Gate — accept/reject

# **GO**

Single end-to-end consistency check before the V3 PDF is built and pushed.
The author's verbatim concern — "Are we going to get appropriate numbers out
of the results?" — is answered **yes**: every numeric source agrees, the method
is the pinned conversion, the E88-at-front ranking is consistent everywhere,
Figure 2 is the regenerated real asset, caveats are intact, no hype was
introduced, and `bash paper/build.sh` compiles a real PDF with `typst`.

> Read-only verification. **No edits were made to `paper/main.typ` or any figure
> asset.** The only file written by this task is this report
> (`paper/review/V3_GATE.md`).

---

## Checklist

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | bpb internal consistency | ✅ PASS | main.typ 0.974/0.977/0.980 everywhere == V3_NUMBERS.md == authoritative table; no stale values |
| 2 | Method (×0.368163588, not 0.391275) | ✅ PASS | pinned JSON constant; recompute reproduces table |
| 3 | Ranking (E88 nominally front) | ✅ PASS | caption L874–876, body L906–907; no "E88 trails" |
| 4 | Figure 2 regenerated, values match | ✅ PASS | `figure_2.png` real 109 KB PNG (1360×765); non-draft ref; steps/as-of match text |
| 5 | Caveats intact | ✅ PASS | single-realization, ongoing/as-of snapshot, unequal-step/token all present |
| 6 | Tone gate (no hype) | ✅ PASS | no superlatives on the bpb result; flip stated plainly |
| 7 | HF status (observation) | ⚠️ NOTE (non-blocker) | v3-push-current FAILED (approval-gated); but cited v0.2 HF links still resolve, so no 404 |
| 8 | `bash paper/build.sh` compiles | ✅ PASS | real `typst compile` → `Garrison_2026_Emender-3c52a9b7.pdf`, rc=0 |

---

## Detail + evidence

### 1. bpb internal consistency — PASS
Every bpb in `paper/main.typ` is the 4-dp rounding of the authoritative table and
agrees with `paper/review/V3_NUMBERS.md`.

- Abstract `main.typ:73`: E88 "0.974 bits per byte".
- §1 `main.typ:160`, §1 Viability `main.typ:256`: E88 "0.974 bits per byte".
- Figure-2 caption `main.typ:895–896`: "E88 0.974 BPB, GDN 0.977 BPB, and
  M²RNN-CMA 0.980 BPB".
- §5 body `main.typ:906–907`: "E88 reaches 0.974 bpb on The Pile, GDN 0.977,
  M²RNN-CMA 0.980".
- §10 Conclusion `main.typ:1662`: "0.974 bpb".

Authoritative table (task ticket / V3_NUMBERS.md): E88 0.973863, FLA-GDN
0.977215, M2RNN 0.980065 → round to **0.974 / 0.977 / 0.980**, exactly the
paper's values. V3_NUMBERS.md's own snapshot recompute (0.973765 / 0.976965 /
0.979845) rounds identically (Δbpb < 0.0003, logs advanced ~50 steps).

**No stale values anywhere.** `grep -nE '0\.977[^2]|0\.981|0\.984|0\.391275'`
on `main.typ` matches only the *correct* new GDN value "0.977" (L895, L907);
the old ordering `GDN 0.970 < E88 0.977 < M²RNN 0.983` and the old constant
`0.391275` appear nowhere in the paper.

### 2. Method — PASS
`bpb = nats/token × log₂(e) / bytes_per_token`, constant pinned in
`scripts/estimate_tokenizer_bytes_per_token.json`:
`"bits_per_byte_per_nat_per_token": 0.3681635882200934`,
`"mean_bytes_per_token": 3.918625`. main.typ states the same:
"bytes/token = 3.92" (caption L884) and "≈ nats/token × 0.368" (L916).

Spot-recompute (all three):
```
E88     2.645190 × 0.368163588 = 0.973863  → 0.974  ✓
FLA-GDN 2.654296 × 0.368163588 = 0.977215  → 0.977  ✓
M2RNN   2.662036 × 0.368163588 = 0.980065  → 0.980  ✓
```
Matches the authoritative table. The old `0.391275` multiplier is absent from
the paper (referenced only in V3_NUMBERS.md as "not the old 0.391275").

### 3. Ranking — PASS
E88 is consistently at the front; nothing says E88 trails.
- Caption `main.typ:874–876`: "E88 is the lowest-BPB endpoint in the current
  snapshot, GDN is second, and M²RNN-CMA trails them across the sampled window."
- §5 body `main.typ:906–907`, `:919` ("M²RNN-CMA trails E88") — correct direction.
- Abstract is neutral ("lands in the same loss-vs-wallclock band as Gated
  DeltaNet"), no claim that E88 trails.
`grep` for an "E88 behind/trails" sentence → none.

### 4. Figure 2 — PASS
- Reference `main.typ:870`: `image("results/figure_2/figure_2.png", …)` — the
  **non-draft** asset. `grep -c draft main.typ` → 0; no `figure_2_draft.png`
  reference remains.
- The asset is a **real figure**: `file paper/results/figure_2/figure_2.png` →
  "PNG image data, 1360 × 765, 8-bit/color RGBA, non-interlaced", 109,276 bytes,
  valid `\211PNG\r\n` header. (Renamed off `_draft`; the internal preview lives
  separately as `figure_2_smooth_preview.png`.)
- As-of and step counts match the text: caption `main.typ:902` cites the
  "2026-05-31T13:49:33Z active-log snapshot", matching V3_NUMBERS.md's as-of;
  caption `main.typ:909–910` cites steps 1,523,250 (E88) / 1,999,300 (GDN) /
  1,466,400 (M²RNN), matching the V3_NUMBERS.md endpoint table. CSVs
  (`E88_NDM.csv`, `FLA_GDN.csv`, `M2RNN_CMA.csv`) and `AS_OF.md`/`SOURCES.md`
  are present and were regenerated alongside the figure.

### 5. Caveats — PASS
Caveats were not dropped to flatter the new lead:
- Single realization / single-seed: caption `main.typ:890–891` "The plotted
  trajectory is one realization per architecture"; §6 probes use three seeds.
- Ongoing / as-of: caption `main.typ:902` "Recorded from a 2026-05-31T13:49:33Z
  active-log snapshot; training continues."
- Unequal step/token: the cohort is explicitly "cohort band, not equal exact
  size", curves are read at differing steps (1,523,250 / 1,999,300 / 1,466,400)
  with differing tokens seen (≈15.6B / 16.4B / 15.0B per V3_NUMBERS.md), and the
  framing is matched **wallclock**, not matched tokens.

### 6. Tone gate — PASS
No superlative/celebratory language was introduced on the bpb result. The racer
prose uses measured wording: "occupy the same sub-1-bpb band", "E88 is the
lowest-BPB endpoint in the current snapshot", "all three are sub-1-bpb". The
only "outperform"/"dominate"/"best" hits in the document are pre-existing and
not about E88's bpb result: related work (Mamba-3 `:348`, GDN-2 `:1449`), the
field description (`:365`), future-work speculation (`:1752`, `:1758`), and
"best-tuning"/"best-effort" protocol phrasing (`:833`, `:1569`). The E88↔GDN
flip is stated plainly; no new framing paragraphs, no added abstract emphasis.

### 7. HF status — NOTE (non-blocker)
`v3-push-current` is **FAILED**, not done: it blocked on missing human approval
for a public HF write (approval-gated `scripts/publish_v02_public_hf.py`), and
created prereqs `v3-validate-current` and `v3-approval-gated`. So the *current*
(newer) checkpoints were not pushed.

However, the paper does not assert a live link to those new checkpoints. It
cites the **v0.2** release URLs (`main.typ:314–318`):
`huggingface.co/poietic-pbc/{emender-e88-1.3b,gdn-1.3b,m2rnn-cma-1.3b}/tree/v0.2`.
Per the dependency's own hub-API readback (v3-push-current log, 13:50:58Z), all
three repos are public with tags v0.1+v0.2 resolving, so the cited links do not
404 and the reproducibility claim remains backed at v0.2. Per the gate rule
("do not fail on HF alone unless the paper asserts a live HF link that 404s"),
this is **not a blocker**. (Independent network re-fetch was not performed by
this gate; the basis is agent-663's recorded hub-API check.)

### 8. Compiles — PASS
`typst` is installed (`/home/erikg/.cargo/bin/typst`); `paper/build.sh` runs a
real `typst compile main.typ "$OUTPUT"` (no placeholder fallback). Running
`bash paper/build.sh` exited **rc=0** and produced
`paper/Garrison_2026_Emender-3c52a9b7.pdf`.

---

## Verdict: **GO**

All eight checks pass; the single ⚠️ (HF) is an explicit non-blocker because the
paper cites the v0.2 release that still resolves, not the unpushed current
checkpoints. The numbers are consistent end-to-end, computed by the pinned
method, ranked with E88 at the front, visualized by a real regenerated figure,
properly caveated, free of hype, and the document compiles to a real PDF.

**Follow-ups (not gate blockers):**
- `v3-push-current` (FAILED, approval-gated) → its successors `v3-validate-current`
  and `v3-approval-gated` own pushing the *current* checkpoints if/when a human
  approves a public HF write. Until then the paper correctly points at v0.2.

---

*Generated by the V3 synthesis gate (task `v3-synthesis-gate`, read-only).*
