# X43 C13 — coverage scaling of frozen C12 disjoint models

Coverage scaling is mixed: Emender improves total/fork from X to self, while transformer worsens both. Raw self inputs lose total to C3 fusion for both heads. Stacking is complementary: total/fork become **1.650482/2.379249** and **1.694013/2.470112**, beating dense C3 fusion for both heads. OOD-self remains **2.450782/2.410592**: slightly worse than dense for Emender, better for transformer. Retain calibration plus forced walk in the tested recipe; learned consultation does not replace it.

Predictions were committed in `67c26772` before optimizer recovery or swapped inference; SHA256 `d8da80b016bb619a04d5c811597d19c8b587c4ea6ac538436bac88a3d5d0778f`. No new training arm or adaptation was run. C12 retained no disjoint checkpoint, so the exact original 6,293/29,640 steps were replayed. Both X-input and all-zero whole-held-out probability vectors match exactly (maximum probability error 0 across all four 111,539-target vectors), passing the preregistered parity gate before any swapped output is adopted. The recovered models then remain frozen.

The self array contains training-prefix counts using the unchanged C11 direct-SA device file. “Full coverage” does not mean every held-out position has support: 9,049 final targets have empty self support. The context selector is corpus-wide and transductive. This requested C11 geometry also changes metadata relative to the X geometry, so this is a coverage-plus-interface swap, not a clean isolated coverage intervention.

All losses are nats/byte on the same 55,770 final targets. Y-fork has 32,415 targets; OOD-self has 9,049; OOD-X has 41,668. OOD columns are different masks, retained separately. Dense/C11 historical controls are reused.

| Head | Training / eval integration | Total | Y fork | OOD-self | OOD-X |
|---|---|---:|---:|---:|---:|
| emender | Dense / neural | 2.222975 | 2.529165 | 2.444723 | 2.216768 |
| emender | Dense / C3 calibration + forced walk | 1.673012 | 2.424886 | 2.444723 | — |
| emender | C11 self inputs / neural | 3.872928 | 5.330632 | 6.551557 | — |
| emender | C11 self inputs / C3 fusion | 3.870401 | 5.327873 | 6.551557 | — |
| emender | C12 disjoint / X inputs | 2.116847 | 2.434766 | 2.423721 | 2.214777 |
| emender | C12 disjoint / all zero | 3.078657 | 3.040701 | 4.267185 | 3.202396 |
| emender | C12 disjoint / self inputs | 1.834650 | 2.397116 | 2.450782 | 1.849784 |
| emender | C12 disjoint / self inputs + C3 fusion | 1.650482 | 2.379249 | 2.450782 | 1.649322 |
| transformer | Dense / neural | 1.924201 | 2.634117 | 2.466627 | 1.901476 |
| transformer | Dense / C3 calibration + forced walk | 1.746996 | 2.558332 | 2.466627 | — |
| transformer | C11 self inputs / neural | 4.953106 | 7.317413 | 6.060058 | — |
| transformer | C11 self inputs / C3 fusion | 4.941959 | 7.298356 | 6.060058 | — |
| transformer | C12 disjoint / X inputs | 1.816094 | 2.470846 | 2.383454 | 1.810907 |
| transformer | C12 disjoint / all zero | 2.474033 | 2.871956 | 2.918144 | 2.464290 |
| transformer | C12 disjoint / self inputs | 1.835328 | 2.523442 | 2.410592 | 1.818276 |
| transformer | C12 disjoint / self inputs + C3 fusion | 1.694013 | 2.470112 | 2.410592 | 1.683711 |

## Preregistered outcomes

emender: P1: total improves, fork improves; P2: total does not improve, fork improves, ood_self does not improve; P3: total improves, fork improves, ood_self does not improve.

transformer: P1: total does not improve, fork does not improve; P2: total does not improve, fork improves, ood_self improves; P3: total improves, fork improves, ood_self does not improve.

P1 compares self to X total/fork. P2 compares raw self inputs to dense C3 fusion. P3 compares self plus fusion to raw self. Equal empty-support stacking loss is expected: C3 abstains there, so fusion cannot repair that stratum. Calibration uses only the first held-out half and is frozen before final scoring; its four features and forced-walk/empty-support rules match C3. No final-test tuning or model selection occurred.

This table closes the requested tier-1 integration comparison empirically, including negative outcomes. It does not establish general operator skill or a universal deployment recipe. Device removal shows live feature dependence; it does not show that nothing was encoded into weights. Foreign support distributions can supply predictive shared English substrings, so disjointness does not make all partial copying impossible.

## Validation and limitations

Broker-leased physical GPU 2, allowlist 2/4; no GPU 7 execution, no WorksGood, Frontier, Slurm or subagents. Python `/home/erikg/emender/.venv/bin/python` only. No checkpoint files produced or committed. Recovery metadata records exact steps, complete parity-vector denominators and errors. Independent CPU validation replays 60 loss/denominator pairs, frozen logistic fusion, protocol/source/input hashes, GPU/terminal evidence and verdict signs. Program bindings are independently checked after the receipt commit. Resilient DiLoCo requirement IDs are inapplicable.

Residual risks: one seed/pair, no uncertainty estimate, transductive context selection, metadata geometry shift between X and self arrays, historical controls with different steps/precision/active parameters, no prefix-only selector, and no isolated semantic-operator ablation. Self arrays still have empty support. This experiment cannot infer the effect of coverage alone.

Manual notes: `contact_supervisor` is absent from the complete active tool catalog. No substitute orchestration service was used. Unrelated pre-existing workspace changes were preserved.
