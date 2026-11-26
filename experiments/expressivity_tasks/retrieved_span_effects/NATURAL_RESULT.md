# Natural token panel: the four retrieved pages made predictions worse

**Keep LOCAL in this pilot.** On 1,101,793 next-token targets from 1,024 fresh TRAIN-probe documents, LOCAL beat every trained passage reader on its own host, both at equal dose and at the registered training-clock checkpoint. The GDN-2 seeded reader did use a small passage-specific component, but this did **not** establish that applying passages to the live state is better than reading their blank-start states. This panel used four lexically proposed pages, **not** the [whole-corpus attention head](CORPUS_ATTENTION.md).

[Protocol](NATURAL_PROTOCOL.md) · [reporting rules](NATURAL_REPORTING.md) · [audited numeric receipt](receipts/natural_panel_result.json). The receipt includes absolute losses on every band, all 74 full/time/control contrasts with paired seed gains and intervals, 12 additional time contrasts, and each fit's measured price. No rerun, new selection, or final-test access.

## The registered primary did not pass

Gain means comparator NLL minus candidate NLL, in nats per real token. Positive favors the candidate. These are equal-dose, all-target GDN-2 host/GDN-2 reader results. Intervals are 10,000 paired **whole-document** resamples, recomputing target denominators, RNG 5410. The rule required a positive 95% lower bound **and** all three seed gains positive on **each** row.

| G-EFFECT vs | Gain | 95% interval | Seed gains (5411, 5412, 5413) |
|---|---:|---:|---|
| separately trained G-STATE, `F(0)` | +0.000576 | [−0.002606, +0.003649] | −0.003483, −0.001917, +0.007127 |
| separately trained G-NEUTRAL, `F(0)+S` on support | −0.015365 | [−0.018847, −0.011937] | −0.032282, −0.004584, −0.009229 |
| same fitted weights, rotate only `F_i(S)−F_i(0)` | **+0.00005891** | **[+0.00002303, +0.00009515]** | +0.00003568, +0.00004826, +0.00009280 |

Only the residual-rotation comparison passed. Swapping those residuals while retaining the blank writes, keys, weights, gate and live seed was measurably harmful. It does not follow that the **whole** retrieved hint was helpful, that the separately trained EFFECT beat STATE, or that the experiment identified a per-hit interaction between an affine multiplier and blank write.

## Content, controls and other readers

Mean all-real-target NLL across the three paired seeds; **lower is better**. Time and full refer to different frozen weights, not two evaluation budgets.

| Reader | GDN-2 host, full | GDN-2, time | E97 host, full | E97, time |
|---|---:|---:|---:|---:|
| LOCAL | **5.324373** | **5.247044** | **5.364015** | **5.286850** |
| G-STATE | 5.411689 | 5.268867 | 5.454718 | 5.308334 |
| G-NEUTRAL | 5.395749 | 5.273660 | 5.445612 | 5.316211 |
| G-EFFECT | 5.411113 | 5.275305 | 5.453666 | 5.307655 |
| E-STATE | 5.415012 | 5.271510 | 5.457725 | 5.311227 |
| E-NEUTRAL | 5.406164 | 5.278534 | 5.452836 | 5.311255 |
| E-EFFECT | 5.415133 | 5.280690 | 5.455771 | 5.317901 |
| MIX-EFFECT (2+2 heads) | 5.411697 | 5.295291 | 5.453919 | 5.330374 |

- **Content vs LOCAL:** G-STATE gains were **−0.087317** [−0.091834, −0.082861] on GDN-2 and **−0.090702** [−0.095427, −0.086043] on E97. E-STATE gains were −0.090639 [−0.095285, −0.086038] and −0.093709 [−0.098621, −0.088848], respectively. Across **all 14** reader-vs-own-LOCAL comparisons the deficits were 0.071376–0.093709 full and 0.020805–0.048247 at matched time: every interval excluded zero in favor of LOCAL and every paired seed had that direction. LOCAL retained the eligible seed/support/gate, not the passage values. Models and readers trained together from the first update.
- **Same weights, different evidence:** on all 14 full-dose passage arms *dropping the hint improved* NLL by 0.037883–0.066826; *rotating whole values improved* it by 0.011145–0.014805. All corresponding intervals and all seeds favored the interventions. On GDN-2/G-EFFECT specifically the intact gains were −0.052110 [−0.056388, −0.047746] vs drop and −0.012355 [−0.013005, −0.011710] vs wrong value. This differs from the much smaller, **positive** intact vs wrong-residual comparison above. The model was sensitive to the passages; that is not the same as lowering held-out loss with the intact read.
- **Coverage:** 837,564 eligible targets, all supported; 802,922 eligible for the cyclic four-present/equal-length donor (95.86% of supported; 72.87% of all). Ineligible targets stay in the all-target denominator. Eligibility is not proof of a changed pooled tensor. On 832,647 `no_exact8` diagnostic targets, all 14 full-dose readers still lost to their own LOCAL by 0.091321–0.120388 nats/token, with intervals and seed directions agreeing. This is not a near-duplicate audit.
- **Exploratory families and mix:** neither host established a full-dose MIX advantage over both single EFFECT readers. GDN-2 MIX:E-EFFECT had mean +0.003436 [+0.000632, +0.006210] but seed 5412 reversed; MIX:G-EFFECT was −0.000584 with interval crossing zero. E97's corresponding gains were +0.001853 and −0.000253, both unresolved. At the earlier time point MIX lost to both singles on both hosts. The earlier checkpoint had lower mean probe loss than the full checkpoint for all 16 configurations. Repeating FIT about 67 times is a plausible concern, not an identified mechanism or permission to select a checkpoint on this PROBE.

## Dose, cost and closure

- Pinned `p50k_base`, 2,048 BUILD / 8,192 FIT / 1,024 disjoint locked TRAIN PROBE documents after 50,790 exclusions. BUILD: 2,189,598 tokens, 69,305 distinct pages of up to 32 tokens, 2,057,869 codes, 5,788,594 postings; 11.789 s. FIT: 8,795,579 unique shifted **positions** across its windows (not unique token IDs), 141,448 windows; 5.373 s preparation plus 0.957 s to prepare routes. PROBE: 17,723 windows, 17,480 supported windows; 0.762 s preparation plus 0.032 s for routes. No live suffix in search; one lookup per eligible window; no state across windows/documents. Whole-document exclusion does not prove passage novelty.
- All 48 fits reached 148,096 optimizer updates at batch 64, identical ordered exposures across the 16 arms within each seed: 589,373,547 / 589,374,099 / 589,374,345 real targets **per fit** at the three respective seeds. Total 28,289,951,856 repeated training exposures. Matched checkpoint threshold 393.150 s; actual synchronized train times 393.159–393.462 s, 78,016–86,208 steps, 310,476,262–343,079,585 targets. Not exactly equal wall time, and not equal dose at that checkpoint.
- Synchronized full training seconds per million real targets: LOCAL **1.145–1.147**, single readers **1.244–1.257**, MIX **1.261–1.266**. Per fit 674.95–746.09 s synchronized, 705.75–778.69 s wall including checkpoint work. Intact probe scoring including diagnostics: LOCAL 1.091–1.238, single 1.346–1.523, MIX 1.456–1.534 seconds per million. These are batched scoring clocks, not generation latency.
- Independent BUILD seal and all 48 time/full fit pairs witnessed before the sole PROBE opening. The [receipt audit](receipt_audit.py) verified hashes, target counts, paired exposures, coverage, release, and reproduced the registered aggregates from saved loss/count receipts, without opening text or loading weights. Frozen manifest `819b49e84adb81b3229ddc2a36fcea750c66e35bcfcd8d5a785b1b5f2a650823`; terminal `dbea90b837c8b3fcd741ae716283bb668ec3b3adfd497caa5e57cc2c9579e9e7`. Panel rc=0, independently witnessed, released, ≤36,959 card-seconds. Including its two prerequisite meters: ≤37,027 card-seconds. Later whole-corpus engineering meters are accounted separately.

**Decision:** do not nominate these four-page readers or retry on this spent cohort. The result is about this lexical page proposer, dose and co-trained small recurrent cores. It does not test semantic attention over every corpus row, a production checkpoint, direct token pointers, or a different data scale.
