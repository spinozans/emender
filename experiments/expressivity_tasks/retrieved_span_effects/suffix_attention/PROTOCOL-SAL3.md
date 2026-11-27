# SAL-3 protocol: the count law over the real 100 GB XSA index

Registered before any new natural membership access. This is the
working-system test: the same count law and host as SAL-1/SAL-2, but the
archive is the actual 100 GB index, served by the production transport.
The standing mandate (autonomous operation until intervention) and
Amendment A1 (ORDER2 deferred) carry over.

## Why this point

SAL-1 (4:1) and SAL-2 (512:1) measured the in-RAM count skeleton at
−0.00426 [−0.01633,+0.00412] and −0.00238 [−0.00586,+0.00124] bits/original
byte: a 128× archive growth moved the effect less than its own CI. The
fitted weights show the mechanism was never suppressed — the corpus prior
stayed at half the budget and the depth weights escalated — but the
archive's counts are *redundant where dense* (common n-grams the local
host already predicts) and *unreliable where informative* (rare contexts
with single-digit backing). The value-bearing regime needs roughly
50–500× more backing than 1 GiB: the 100 GB class, which is where X43's
consultation won by ~0.45 bits/byte (1.657 vs 2.109, out-of-index Pile).
SAL-2's registered decision rule 2 fires: no further in-RAM ratio points;
the successor is the transport at 100 GB scale.

## The archive (existing, sealed artifacts; no new BUILD role)

- Container: `/mnt/nvme2n1/erikg/PILE100-REHEARSAL/pile-head100.sxi2`
  (65,533,578,480 bytes), **SXI2 v10**, n = 100,000,000,001,
  runs = 31,381,597,694, chi_complete, m15 document starts, m20
  boundary-aware context trie (depth 8, min-support 1000), built by the
  rehearsal pipeline (`checkpoint.journal` BUILD_DONE
  `15419c427b5bdc9fb63548901ca6cf83d54f486da890f66faa55021f0a0dc053`).
  The text is the first 100,000,000,000 bytes of
  `/mnt/nvme2n1/erikg/pile.txt` (the X43 corpus). A fresh full-file
  sha256 is receipted at prep and must equal the BUILD_DONE value.
- Serving binary: the pinned private copy
  `output/retrieved-span-effects/sal-3/tools/xsa` (sha256
  `8ff46bbf…8d25fc`, built from `/home/erikg/sxgc` — a live tree whose
  build drifted once mid-campaign, which is why the artifact is frozen
  and sha-verified at every use; the cross-build smoke produced
  byte-identical rows before pinning).
- Transport: `xsa consult` (CONSULT.md contract): JSONL windows with
  explicit corpus bases, `--left d` spine budget per pass, per-position
  branch histograms under the no-cross law (a match or branch count
  never spans a document boundary; the framing byte is the END branch),
  journaled, resumable, RSS-bounded.

## The law placement

Unchanged SKELETON source and SalLM host (633,291 parameters): per target,
the corpus mass is `exp(log_pi − log N + log Σ_ℓ w_ℓ m_ℓ)` with the
telescoped exact-depth classes m_ℓ, h_ℓ. Two registered convention notes:

- **True-document contexts.** The live spine is cut at the document's
  actual start (consult's spine law), not at the 64-byte window or the
  4097-byte row. Contexts are the real d bytes before each target inside
  its document; the local arm still sees only its 65-byte windows. This
  is strictly more honest than the SAL-1/2 window-bounded contexts and
  is exact at every target position.
- **Depth 0** is the archive's successor marginal: h_0[v] = Σ_c h_1[c][v]
  over all 256 single-byte contexts c, N = Σ_v h_0[v], both measured
  exactly by 256 single-byte consult contexts and verified against the
  brute-force gate.

## Arms and dose

- **LOCAL**: identical host, no archive.
- **XSACOUNT**: the count law with m/h bound off-card from the transport.
- Seeds 5751/5752; 2,048 updates; batch 16; warmup 32; cosine 2e-3→2e-4;
  FIT 512 fresh documents (2,097,152 targets); PROBE 128 fresh documents
  (524,416 scored bytes), out of the index by construction (all roles are
  drawn beyond the 100 GB region), frozen, witnessed, opened and scored
  exactly once.

## Binding discipline (off-card, receipted)

For each role and each depth d = 1..8: one `consult` pass
(`--left d`, `-j 96`, RSS-bounded) over the role's windows; rows parsed
and assembled into the same batch tensors the SAL-1/2 driver consumes
(m (Q,9), h (Q,9,256), plus the depth-0 constants), cached under sha
receipts binding: rows.json (the role seal), the consult journals'
`rows_sha256` per pass, the container sha, and the binary sha. Depths
1..7 telescope exact-depth classes c_d − c_{d+1}; the deepest class is
the >=8 class, exactly as SAL-1/2's bind_stats. The stage loads bound
batches only under matching receipts; any mismatch fails closed. No
consult query touches any probe byte: probe windows are bound only after
the freeze is witnessed, exactly like SAL-2.

## Qualification (invented inputs; zero natural memberships)

1. **A1 parity, re-run now** on this container + binary: brute-force
   (`gates/consult_oracle`) scan parity for a fresh context set spanning
   depths 1..8, document-edge positions, and the END branch; every
   (width, eof, pairs) must match exactly.
2. **Depth sweep**: consult passes at `--left 1..8` over a fixed sample
   window set; per-position width/eof/pairs must equal the gate's
   no-cross brute-force counts for every d, and every used row must be
   boundary-exact.
3. **Determinism**: two passes over the sample set must produce identical
   row sha256.
4. **Rate**: an A3-style measurement at `-j 96` receipted for stage
   budgeting.

## Decision rules (registered before the probe opens)

Primary contrast: mean `XSACOUNT − LOCAL` bits/original byte over the
probe bytes, per seed and pooled, the SAL bootstrap CI family.

1. **Positive** (pooled CI excludes 0 and both seeds positive): the count
   law pays at system scale — the working system's core is measured. Next:
   the depth/copy channel (long matches), the learned-core ORDER2
   successor, and scale-up measurements.
2. **Not positive**: the count law inside one softmax source does not
   express the index's value even at 100 GB; the placement, not the
   ratio, is the remaining lever (X43's input-feature consultation is the
   placement that won). Next registration changes placement, not scale.
3. Either way: zstd baselines are reported but are not the criterion.

zstd baselines: level-3 plain; level-3 with the archive's raw bytes as a
shared dictionary — attempted at the full 100 GB, falling back to the
largest prefix that completes inside the stage budget, with the fallback
labeled in the receipt (the fallback is a weaker baseline for the codec,
never for the neural arms).

## Campaign

Ledger `output/retrieved-span-effects/sal-3/campaign.jsonl`; cap
**6,000 card-seconds** (the probe's nine transport passes run inside the
leased stage after the witnessed freeze — about 4.7M consult positions,
~1,000 card-s of measured transport — so the cap carries one continuation);
end `1795845600` (the operator's one-day mandate); no automatic retry;
failed stages are results, continued under new names with retained notes;
stage-suffixed artifacts; the quarantined probe subprocess, the optional
post-extraction bind subprocess, the parameterized ready-seal set and the
run-lineage closure are the SAL-2 published fixes carried forward.
