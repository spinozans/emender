# X43 part 2: exact-count interpolation at forks

Frozen exact-count interpolation improves final-test NLL for both models. With training-only branch counts, the feature gate reduces Emender NLL by 0.060610 nats/byte and transformer NLL by 0.044048. The much larger in-corpus gains are transductive diagnostics, because the index includes the scored target and future held-out bytes.

Calibration: 55,769 targets; final test: 55,770 targets, corpus offsets [1059624, 1115393]. All following table values use only final test (natural logs). Models are frozen and their complete per-position NLL and accuracy replay exactly to floating-point tolerance on leased physical GPU 2.

| Counts | Architecture | Model alone | Fixed interpolation | Learned interpolation | Learned + forced walk | Walk + exact oracle |
|---|---|---:|---:|---:|---:|---:|
| train | emender | 2.222975 | 2.190951 | 2.162365 | 1.673012 | infinity |
| train | transformer | 1.924201 | 1.903182 | 1.880153 | 1.746996 | infinity |
| corpus | emender | 2.222975 | 1.323822 | 1.323716 | 0.570759 | 0.570865 |
| corpus | transformer | 1.924201 | 0.964047 | 0.963992 | 0.570810 | 0.570865 |

The interpolation-only columns retain model probabilities at forced positions. The walk variants replace covered forced probabilities with one, leaving uncovered positions to the model. Corpus walk + oracle is 0.570865 overall, zero on forced positions and 0.982172 on forks for either architecture. It is a diagnostic baseline, not a strict ceiling: the learned corpus mixture slightly improves on it. Train walk + oracle has infinite NLL; zero-probability targets are included, not discarded.

| Train-count gate | Emender | Transformer |
|---|---:|---:|
| Grid-selected index weight | 0.14 | 0.07 |
| Registered scalar index weight | 0.143360 | 0.065271 |
| Registered scalar final-test NLL | 2.190950 | 1.903171 |
| Learned fork NLL | 2.424886 | 2.558332 |
| Learned forced NLL | 1.798005 | 0.938890 |
| Learned + walk forced NLL | 0.629466 | 0.620921 |

Train counts are absent for 2,709 of 32,415 final-test forks (8.36%); the gate abstains exactly and their model NLL is 2.739430 / 2.886269. A further 17,066 forks have nonempty counts but zero target count. These are distinct failure modes: a nonempty index can confidently exclude the observed byte. Both corpus gates choose fixed weight 1; corpus support is never empty because it includes the target.

| Train support size | Fork positions | Emender learned index weight | Transformer learned index weight |
|---|---:|---:|---:|
| [0,1) | 2709 | 0.0000 | 0.0000 |
| [1,2) | 11636 | 0.0705 | 0.0239 |
| [2,4) | 11708 | 0.2194 | 0.0961 |
| [4,8) | 4602 | 0.4816 | 0.2499 |
| [8,16) | 1420 | 0.7569 | 0.4932 |
| [16,257) | 340 | 0.9302 | 0.7899 |

Train-count index weight falls with context length: on lengths [4,8), mean weights are 0.2950 / 0.1555; on [8,16), 0.1189 / 0.0436; on [16,32), 0.0144 / 0.0035. Full support/context telemetry and fitted coefficients are in receipts/part2/fusion.json. One-branch training support hurts both models even after learned gating (Emender fork NLL 2.694526 -> 2.747261; transformer 3.063104 -> 3.081827). These conditional strata are descriptive and correlated; they do not establish a causal feature effect.

Fusion math, normalization, gate fitting, split boundaries and abstention are fixed in DERIVATION_PART2.md. Exact probabilities, counts, per-position gates, model target probabilities and replay hashes are retained in receipts/part2. Calibration chooses gates without final-test NLL selection. The preregistered scalar and the requested grid/feature extensions are all reported.

Validation passes: 511 exhaustive tiny/random SA cases, exact count/mixture endpoint tests, 200 independent substring-scan count checks, artifact hashes, retained prediction replay, exact abstention and walk checks, and final-test NLL recomputation. Initial execution failed because SciPy was absent; pip was also absent. Installing scipy==1.18.1 with uv for the required interpreter resolved it. An unrestricted repository diff check found pre-existing whitespace errors in unrelated experiment reports; the scoped part-2 diff check passes.

Follow-up B (fork-driven write gate) was not run within this session. No write-gate training, write density, occupancy or velocity gain is claimed. It still needs its training-only mask derivation, write/prediction time alignment, matched step/time interpretation and Bernoulli controls bound before execution; changing only a write bias would not establish the registered comparison. The requested commit subject is retained, but this report explicitly records that its write-gate component remains unrun.

Residual risks: corpus-derived labels and maximal-context selection use held-out information even in the train-count arm; contexts may exceed the models’ 128-byte reset windows. The results price this corpus/index experiment, not causal generalization. Single corpus, seed and short training; unequal original optimizer steps. No end-to-end index latency or velocity score was measured. Mixture argmax accuracy was not reconstructed from target-probability receipts, so that original registration metric remains unmeasured. Chi_position is not established as minimum suffixient cardinality. The contact_supervisor tool was not available in the active tool catalog.

Reproduce from the repository root:

```bash
uv pip install --python /home/erikg/emender/.venv/bin/python scipy==1.18.1
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.fusion --test
bash experiments/expressivity_tasks/x43_fork_resolved/run_fusion_leased.sh
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.fusion
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.validate_fusion
```

The GPU wrapper restricts broker acquisition to physical cards 2 and 4 and fails
closed without a lease. Analysis and validation use CPU array arithmetic only.
Existing checkpoints remain local; no .pt file is included in this change.
