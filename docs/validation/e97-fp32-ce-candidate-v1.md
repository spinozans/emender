# Opt-in FP32-logit CE candidate

Date: 2026-09-10 UTC. Status: **bounded CE-precision subgate passed**;
overall chunk qualification remains false.
No default, frozen training recipe, or checkpoint has been changed.

## Finding

The completed full-model chunk diagnostic exposed a discrepancy between reported
CE19.50 and FP32 NLL19.437574 computed from the captured BF16 head logits. A
checkpoint-free synthetic CUDA probe (`proc_5a8d`, exit zero, two seconds) reproduced
the precision issue in the installed **PyTorch 2.9.1+cu128 / RTX 6000 Ada** stack:

| Input/evaluation path | Result |
|---|---:|
| BF16 logits, autocast disabled | 19.5, BF16 result |
| BF16 logits, autocast enabled | 19.5, FP32 result |
| Explicit FP32 logits | 19.4375 |
| Manual FP32 log-softmax | 19.4375 |

The constructed winning logit is 20, target logit 0.5625, remaining logits -20.
The result reproduced at vocabulary sizes 64, 50,281, and 50,304, including a
128-row case with only one valid target. An FP32 output dtype alone is therefore
not sufficient precision evidence in this installed stack. This is not a claim
about every PyTorch release or proof that this scalar discrepancy caused failed
agent acquisition.

Probe root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/ce-precision-probe-v1`.
Inventory: `3a0004995bd8112a6d77b348fd0d47c912c677c477890128ad8b813901322c83`.

## Candidate implementation and CPU checks

`ndm/models/ladder_lm.py` adds **`loss_logits_fp32=False`**. Only explicit opt-in
casts temporary head logits to FP32 before dense or chunked cross-entropy.
Parameters and optimizer state remain unchanged. The chunked cast is bounded by
CE chunk length times vocabulary width; the dense path still materializes a
full logits tensor and is not a long-context memory qualification.

No SFT CLI or production recipe selects this flag. A future integration must
record it explicitly in numerical/checkpoint/recipe identity; it must not be
silently enabled on a resumed historical run.

`tests/test_e97_ce_precision_candidate.py` exercises dense and chunked CPU paths
with controlled BF16 logits, verifies the corrected scalar and gradients, and
checks that all parameters remain BF16. The candidate, boundary, gradient-metric,
and decision-metric suites passed **25 tests** together, with CUDA hidden.

## Frozen GPU assay

**`proc_97d6`**, `e97-fp32-ce-candidate-v1`, uses one exclusively leased GPU and a
3,600-second bound. Root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/fp32-ce-candidate-v1`.
Inventory: `f6c168bddaccf1083da9a9377093be16524f792ca656c0417240243817226b70`.
Qualifier: `f2bdfe0ee73c8b0d8f103aa402e79ba87d7af83fe6407a54d4228a70b5afef16`.

`proc_97d6` exited zero in **177 seconds**, including post-run source checks.
Summary SHA-256:
`c11d384f4b765b1f78458c1cd1e5a75d2fc7044096a01e8c90d5a0d93bf647aa`.

The opt-in scalar matched independent FP32 NLL across all six configurations;
maximum absolute error was **1.9073486328125e-6**, below the frozen 1e-4 limit.
Dense loss was 19.437576; pilot, CE-only, and repeated-pilot losses were 19.437574;
MLP-only and combined losses were 19.250092. Model parameter digests were unchanged,
as were the checkpoint/args/probe authorities.

The separate chunk-sensitivity gate remains failed. With the corrected CE, worst
gradient-relative L2 was 0.025016 for pilot/CE-only/repeated pilot, versus 0.070749
for MLP-only/combined changes. These are new candidate measurements, not a revision
of the historical failed gate or a claim that MLP chunking is fully qualified.

The same consumed prefix and unchanged parent train/y were used. Dense, pilot,
CE-only, MLP-only, combined, and repeated-pilot configurations remain separate.
Every opt-in CE scalar must match independently recomputed FP32 target NLL within
**1e-4**. Source/model parameter identity checks and tanh recurrence-reference
checks remain in place. There is no optimizer step or checkpoint export.

This does not relax the failed MLP chunking gate. The diagnostic reports CE
precision and overall chunk qualification separately; process completion must
not be read as `qualification_pass:true`.

ADR-003 remains the distributed production authority; **R16** supplies the
applicable exact-source evidence intent. No distributed recovery behavior,
Frontier/ROCm, 64K packed backward, optimizer integration, or SFT convergence is
qualified here. See `e97-response-gradient-qualification-v1.md` for the retained
failed gate and chunk-attribution evidence.
