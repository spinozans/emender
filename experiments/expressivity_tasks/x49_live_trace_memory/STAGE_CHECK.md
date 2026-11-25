# X49 calibration: the fixed private table helps slightly at long horizons; learning the keys hurts

All 12 registered trajectories reached 1,024 updates. This stage check uses the 110 **calibration** documents only: 450,560 shifted external bytes per cell per seed, including misses. Both calibration leases closed with 660 witnessed document/endpoint windows each, rc=0 (≤320 and ≤456 card-s). No final scores were read for this report. [Frozen panel](PANEL.md) · [aggregate, per-seed effects and evidence hashes](receipts/calibration_stage_check.json).

| Memory horizon (same full host context) | A: bytes/current computation | B: earlier neural value | F: B + fixed private keys | C: B + learned private keys |
|---|---:|---:|---:|---:|
| 128 | 1.324154 | 1.323505 | 1.324047 | 1.561985 |
| 1,024 | 1.173016 | 1.171993 | 1.171343 | 1.367846 |
| 4,096 | 1.147137 | 1.146041 | **1.145268** | 1.330614 |

Entries are mean nats per byte across the three paired seeds; lower is better. At 4,096, B improved over A by 0.001096 and F improved over B by a further 0.000773, each in every seed. At 128 F instead lost 0.000542 to B. C was worse than F by 0.185346 at 4,096 and lost at every tested horizon and seed. F's extra path includes the observed follower, value and additional head capacity, not just an address policy. Rotating **only its private neural payload** at 4,096 cost F just 0.000022; that intervention does not explain the whole F−B predictor difference. B's literal-value rotation cost 0.000834. No calibration-only scale claim. B is the adopted working baseline for local follow-ons; both F and C already use it. That practical choice is distinct from the designer's preregistered Lux nomination rule.

## Post-hoc diagnostic: C gets a large sampled gradient even when keys cannot affect a prediction

At the first natural update of **each seed**, F and C have matching host/head birth digests, observed bytes, draws, work and pre-update NLL. The trace-output correction is exactly zero, so the probability of a target byte cannot yet depend on the sampled key. Both arms nevertheless have the same ordinary NLL gradient, including the trainable zero-output projection. Only C adds the likelihood-ratio term. Its advantages are detached, and its affine key maps have zero weights at birth, so on this first step the extra gradient reaches **only** the private policy tensors. Its expected contribution is zero at birth, but a realized sampled gradient need not be. The two parameter blocks are disjoint: up to norm rounding the sampled policy norm is `sqrt(preclip_C² − preclip_F²)`.

| Seed | F joint norm | C joint norm | Inferred key-policy share of C squared norm |
|---|---:|---:|---:|
| 4901 | 1.099 | 180.239 | 99.9963% |
| 4902 | 2.523 | 186.819 | 99.9818% |
| 4903 | 1.139 | 155.530 | 99.9946% |

Across all three trajectories C clipped on 3,072/3,072 updates (median preclip 213.38), F on 547/3,072 (median 0.640). **The birth isolation does not extend to later steps**, when weights, policy, and host gradients differ. These observations expose a noisy credit signal; they do not establish that clipping alone caused the calibration gap. The normalized Muon and adaptive AdamW updates are not proportional to raw norm.

**One targeted control, not another arm of this panel.** C already computes its own literal-only log-probabilities before consulting private keys. A detached per-byte advantage can therefore use `NLL(actual trace) − NLL(same-weight literal answer)` rather than `NLL(actual trace) − 2`. Targets enter only after predictions; the host/observed text path never consumes private actions. Conditional on the document and weights, this baseline is independent of the chosen file/query actions. Sum the differences over the registered write-effect interval, not just the positions where the sampled key hit. No extra private lookup; no separate B checkpoint, per-key teacher, or change to the ordinary all-byte loss. At zero correction this removes the spurious first-step private policy gradient exactly, while leaving the value readout free to learn.

[`posthoc_credit_control.py`](posthoc_credit_control.py) runs a CPU synthetic check, not a natural optimizer. On an actual hard-read fixture, the key gradient at birth went from norm 1.544 to exactly zero; with a nonzero trace payoff it remained nonzero. The ordinary pathwise gradients were unchanged in the tested zero-key-map configurations. In an independent one-record oracle we enumerated both 256-way actions (65,536 pairs) for matching and mismatching literal priors. Both estimators' exact mean logit gradients agreed with differentiation of the expected loss to <6×10⁻¹⁵; sample-gradient variance was lower with the literal control **on these fixtures**. No natural gain or universal variance reduction is inferred. The completed C learner and frozen endpoint map are unchanged.
