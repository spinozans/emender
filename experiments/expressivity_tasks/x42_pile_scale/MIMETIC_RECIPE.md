# X42: initialization, geometry, and the standing mimetic recipe

**Decision: FOLLOW MIMICRY. Keep the frozen basis and replay spine.** The operator adopted the verdict at `4e16f7be`. The pure search now starts a separate objective lineage with transformer-donor initialization. Historical random-init generations are superseded, never resumed or mixed into the new CMA-ES state.

The question was why the pure memory model lagged the transformer: missing tokenwise computation, insufficient memory budgets, narrow reads, a frozen semantic basis, or difficult early optimization? The investigation tested these suspects in that order. Lower scores are better.

| Suspect | Receipt and result | Decision |
|---|---|---|
| Missing MLP | [Lineage audit](receipts/lineage_suspects/LINEAGE.md): all compared models contain memory/attention **and MLP** blocks. | Refuted as the architectural explanation. |
| Budgets | [Ablations](receipts/lineage_suspects/RESULTS.md): 4x budgets score 4.769, NLL 4.766; baseline score 4.802, NLL 4.743. | No material improvement under the registered 0.1 thresholds. |
| Read width | [Ablations](receipts/lineage_suspects/RESULTS.md): maximum width 256 score 4.890, NLL 4.831. | No material improvement. |
| Frozen basis | [Geometry verdict](receipts/mimetic_geometry/VERDICT.md): unfrozen score 4.893 versus frozen reference 4.802, **0.091 worse**. | Refuted as a demonstrated short-horizon cost; keep it frozen. |
| Initialization | [Geometry verdict](receipts/mimetic_geometry/VERDICT.md): mimetic score **4.531**, NLL 4.471; improvement **0.271** over the 4.802 reference. | Adopt donor initialization as the standing pure recipe. |

These are conditional experimental conclusions, not universal impossibility results. The frozen spine is vindicated for this decision: unfreezing added trainable semantic/key parameters without improving the registered outcome. It does not justify abandoning the certified replay machinery.

## The geometry boundary

“Frozen basis” does not mean frozen effective representations. Per-head semantic embeddings and key projections remain frozen; management and the certified replay spine remain intact. The shared content projection learns coordinates within that basis. Input/content projection, queries, values, read_out, and MLP remain trainable. [Architecture audit](receipts/pure_v2_architecture_audit/audit.json) and [mapping](receipts/mimetic_geometry/mapping.json) distinguish what can learn from what receives donor weights.

The transplant copies the input K/V singular-vector basis, rank-limited donor K content factors, projected donor V rows and biases, matching donor output columns, and MLP neurons (replicated with outgoing-weight multiplicity correction where needed). Queries, semantic embeddings, key projections, management, vocabulary embeddings/head, and norms keep baseline initialization. The mapping is lossy and is not function-equivalent attention. Frozen tensors are checked unchanged. The standing implementation writes a separate initialization receipt for each candidate.

## Donor provenance and reproducibility

The donor is the **existing receipted fresh 2700-second transformer**, trained on the same attested Pile corpus using the best available transformer-search recipe, not an undocumented external checkpoint. [Provenance](receipts/mimetic_geometry/donor_provenance.json) records that original search runners saved no attributable weights, so a fresh donor was trained; recipe score 3.685661 and actual donor NLL 3.763635 are different measurements. [Mapping](receipts/mimetic_geometry/mapping.json) and the [new registration](velocity_registration.json) bind the exact checkpoint SHA-256: `d5e0fa40ba32f8b7616cc70440d15834c906e142b30ec2f6fa9dced276c91289`. Corpus hashes are in the provenance; training/evaluation protocol is in [protocol.json](receipts/mimetic_geometry/protocol.json). Donor weights remain local at `receipts/mimetic_geometry/donor_weights.pt`; no `.pt` file is committed. A missing or changed donor fails closed.

## Pure search transition

[Archive receipt](receipts/mimicry_adoption/baseline_archive.json) and its copied generation/parent/failure snapshots preserve the NON-MIMETIC arm. The work order's 4 generations / 39 candidates / best 4.802 was an earlier snapshot: inspection found **52 completed eligible results, best 4.796119**, including generation-5 candidates. The service was already failed before intervention and was explicitly stopped; its failure receipt is preserved. This does not alter the fixed 4.802 reference of the earlier controlled experiments.

The fresh search uses `runs/velocity_mimetic_v1`, a new smoke run, seed 4201, population 12, and the existing score, 2700-second training clock, batch 64, context 2048, evaluation and broker protocol. Donor initialization is outside the candidate training clock; donor pretraining is additional compute. No completed scores, CMA moments, population or generation state are imported. Registration, regime source hashes, donor identity and [source amendment](receipts/mimicry_adoption/rebind.json) are committed together before restart. The launcher continues to acquire card 1 through the broker; no protected workload or card 7 is targeted. Transformer/GDN2 searches, probes, ownership proofs and anchors are left running without changes.

## Pre-registered 1.3B recipe

The [1.3B component amendment](receipts/mimicry_adoption/scale_1p3b_amendment.json) records the decision now, before scale terminal runs: use a trained transformer donor on the same corpus, sized appropriately; calibrate the geometry-aware content/value/read_out/MLP mapping; preserve the frozen semantic/key basis, management, and certified spine. Record the scale donor architecture, corpus/checkpoint hashes and exact mapping before launch. This component receipt does not assert a selected scale checkpoint or completed 1.3B validation.

## Evidence availability and limits

[Receipt audit](receipts/mimicry_adoption/receipt_audit.json) records `git cat-file` checks of HEAD and fetched origin/main for every historical receipt linked above. All eight audited historical receipts are committed and pushed; no missing or unpushed historical receipt was found. Any exception is listed explicitly in that audit. The adoption artifacts are committed and pushed with this document. Binary donor weights are intentionally local-only; the archive indexes local candidate result hashes rather than committing every training artifact.

One seed and one 2700-second horizon cannot establish scale transfer or significance. Different throughput means different token counts; the donor consumed 3.343x the mimetic arm's tokens, and donor pretraining adds compute. No shuffled/scale-matched random donor control or joint mimicry+unfreezing experiment was run. Unfreezing changed trainable capacity and inherited baseline optimization settings. All pure comparison arms still trigger pressure_runaway, with near-unit write density: mimicry improves NLL without solving that degeneracy. See the [full summary](receipts/mimetic_geometry/summary.json) for telemetry. No resilient DiLoCo behavior changes; its requirement IDs are not applicable.
