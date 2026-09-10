# E97 SFT precision integration: CPU candidate wiring

Date: 2026-09-10. **Opt-in implementation; full-model qualification pending.**
No sustained training, selected learning rate or checkpoint promotion is claimed.
The rebuilt Open-SWE authority remains `training_eligible:false`.

## Integrated behavior

`scripts/train_e97_4b_pi_sft.py` now exposes explicit numerical-policy options:

- `--optimizer-precision bf16-sr-candidate` uses the named-parameter SR optimizer
  with BF16 parameters and CPU-offloaded BF16 persistent optimizer state.
- `--sr-seed` binds the counter stream; parameter names/layout and the optimizer's
  own schema remain independently validated on restoration.
- `--loss-logits-fp32 --loss-chunk-size N --checkpoint-loss-chunks` selects
  temporary FP32 CE logits with an explicit chunk limit of at most 4,096 tokens.
  Rematerialization is required: merely looping over chunks would still retain
  all their vocabulary-wide softmax tensors until backward.
- `--disable-bf16-reduced-precision-reduction` explicitly disables reduced-
  precision BF16 GEMM reductions. This is a candidate control, not a demonstrated
  fix for the prior MLP-chunk sensitivity failure.

The nested `sft_precision` checkpoint identity binds the optimizer/schema/seed,
CE and GEMM policy, effective CE chunk size, MLP/checkpoint grouping, learning
rate, weight decay and warmup. A changed policy fails resume. Missing policy is
accepted only through the explicitly selected unmodified legacy compatibility
path; it cannot authorize an SR or FP32-CE resume.

The SR trainer path currently **requires eight-rank full-world DDP with the
redundant outer merge disabled**. This trainer saves rank-0 optimizer state;
independent islands' different moments cannot be restored faithfully from that
one payload. Multi-island SR training is therefore rejected, rather than inferred
from the separate synthetic merge qualification.

`ndm/e97.py` recognizes SR checkpoint state and restores its recorded BF16 live-y
backup for `weight_mode='train'`. It does not reconstruct a newly rounded y from
averaged x/z or run an optimizer step during loading. `weight_mode='saved'`
continues to select the recorded x weights. Unknown/conflicting precision metadata
and partial backups/state fail closed. The loader also supports an already-held
file descriptor with explicit path identity; authentication of those bytes is
still the caller's responsibility.

The SR candidate now initializes every parameter's state before its first step,
including a parameter with no initial gradient. Nonzero-step checkpoints must
contain every initialized z/moment slot. Synchronous checkpoint publication drops
the temporary payload reference so exported y backups are not retained across
subsequent training windows.

## Validation and retained failures

- Initial clean export `/tmp/e97-sft-precision-publish-toZOq8`, `proc_bba7`:
  **106 passed, one skipped, two failed** in 217 seconds. The numeric fixture
  happened to survive inverse interpolation, and the staged export retained an
  old source-string assertion for the previous weight-mode interface.
- Replacement export `/tmp/e97-sft-precision-publish-v2-QragVE`, `proc_5ff9`:
  **108 passed, one GPU-only skipped** in 249 seconds. The inverse-restoration test
  now includes a deliberate finite BF16 cancellation edge, y=2^-8 and z=1 at
  beta1=.9, and still requires exact equality to recorded y plus inequality to
  reconstructed y. Only the matching weight-mode test hunk was staged; unrelated
  worktree tests were not included.
- Subsequent CE-rematerialization additions: **24 focused tests passed**, plus
  **two additional loader tests passed**. Saved-tensor hooks show that checkpointed
  CE does not retain the vocabulary-wide rows that the uncheckpointed loop does;
  the tiny E97's losses and all parameter gradients agree exactly. Tiny E97
  checkpoint reloads recover both x and y exactly, and descriptor loading uses
  held bytes after the pathname has been replaced.
- Final export `/tmp/e97-sft-precision-publish-v3-wdCuGr`, `proc_9354`:
  **113 passed, one GPU-only skipped** in 217 seconds. Tested Git tree:
  `b0ec6e5cd58dee1af434a56faa3f872824bb9531`. The publication changes only this
  evidence paragraph relative to that tested tree.

Logs, the tested patch and tree identity are retained read-only at
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/sft-precision-integration-v1`;
inventory SHA-256:
`a34dde287e9b63f83d183f043ce4d671dd96ec4ce7dd23989eee1831658e63ef`.
None of this CPU evidence substitutes for actual E97 4B CUDA updates,
long-context backward, full-world restart or measured accelerator memory.

The complete regression command covers:

```text
tests/test_e97_sft_precision_integration.py
tests/test_e97_facade.py
tests/test_e97_4b_pi_instruction_sft.py
tests/test_schedulefree_sr_candidate.py
tests/test_schedulefree_sr_distributed.py
tests/test_schedulefree_cpu_offload.py
tests/test_e97_ce_precision_candidate.py
tests/test_diloco_merge.py
```

It runs with CUDA hidden, OMP/MKL threads set to one, no bytecode/cache writes,
and a 300-second process deadline. The distributed fixture is local Gloo.

## Architecture scope and next gate

[ADR-003](../RESILIENT_DILOCO_COMPUTE_POOL.md) governs this fixed-world path.
Applicable safety intent: **R07/R12** committed complete checkpoint/recovery,
**R16** exact-source evidence discipline, and **NDP15** checkpoint atomicity.
This change introduces no restart supervisor, database, automatic retry or scale
permission. **R14/NDP13** runtime failure containment still needs the bounded
actual execution gate. Elastic R02–R06/R08–R11, NDP02's no-all-rank property,
NDP17's native ladder, V21S01–V21S17 and ISP01–ISP07 are unclaimed.

Subsequent [native-trajectory numerical qualification](e97-native-sft-numerics-v2.md)
passed real E97 4B forward/backward at 14,841 and 65,423 tokens for the specific
checkpointed-CE/MLP4096 policy. Effective updates, all-target packed/DDP memory,
x/y publication and fresh-process continuation still need qualification. The
previous failed MLP-chunk configuration remains failed. Native source admission,
whole-trajectory mixing/sampling, model-serving integration and the matched
higher-LR-inclusive learning experiment remain separate prerequisites.
