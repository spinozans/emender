# v0.1 Racer Checkpoint Pinning

Date: 2026-05-27
Task: `release-v01-racer-checkpoint-pin`

This note pins the v0.1 release candidates for the three 1.27B-class paper
models and records the Figure 2 refresh decision. No model weights were
uploaded, no HuggingFace or GitHub model artifacts were published, and no repo
visibility was changed.

## Inputs

Required starting context:

- `wg show v22-racer-refresh`
- `wg show release-v01-preflight`
- Live training logs under `/tmp/pile_convergence_3arch/ctx2k/` and
  `/tmp/pile_convergence_m2rnn/ctx2k/`

The paper BPB conversion uses the pinned tokenizer estimate in
`scripts/estimate_tokenizer_bytes_per_token.json`:

- mean bytes/token: `3.918625`
- BPB per nat/token: `0.368164044389` (`log2(e) / 3.918625`)

## Release Candidate Checkpoints

The release candidates are the latest stable checkpoint files present when this
task pinned them. The `latest.pt` symlinks resolved to these paths at pin time.
The checkpoint loss is the raw save-step training loss from the filename and is
not the 10K-smoothed Figure 2 loss.

| Model | Pinned checkpoint | Step | Raw checkpoint loss | Raw checkpoint BPB | mtime UTC | Size | SHA256 |
|---|---|---:|---:|---:|---|---:|---|
| E88 / NDM | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1281000_loss_2.6850.pt` | 1,281,000 | 2.6850 | 0.988519 | 2026-05-27 20:18:43.401195158 +0000 | 7,639,217,707 bytes | `2ccb8851c798c5aa72ff0d6d45318496b6fbbc952c8379c7d56b23281ecedcfb` |
| GDN | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_1686000_loss_2.6105.pt` | 1,686,000 | 2.6105 | 0.961091 | 2026-05-27 19:59:32.707195158 +0000 | 8,114,430,987 bytes | `9e2b8baad914d9b7ab28f411fc65d37875fa269db3fc2f6e37503fd4c1730148` |
| M2RNN-CMA | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1212000_loss_2.6870.pt` | 1,212,000 | 2.6870 | 0.989256 | 2026-05-27 19:51:20.514195158 +0000 | 7,842,766,221 bytes | `2ce9ada25d374c0bab7f20017d8ff5324a8583b8dc46bcd6180aefe923866197` |

Resolved symlinks at pin time:

- E88 / NDM: `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/latest.pt`
- GDN: `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/latest.pt`
- M2RNN-CMA: `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/latest.pt`

## Figure 2 Loss Snapshot

Figure 2 was regenerated from the same live logs using:

- `python paper/results/figure_2/smooth.py`
- `python paper/results/figure_2/plot_normalized.py`

The regenerated CSV tails used for the paper are:

| Model | Log source | Tail step | Raw tail loss | 10K-smoothed loss | 10K-smoothed BPB | Wallclock h | Tail timestamp UTC |
|---|---|---:|---:|---:|---:|---:|---|
| E88 / NDM | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log` | 1,281,300 | 2.6489 | 2.659895 | 0.979277 | 467.201 | 2026-05-27T20:25:13+00:00 |
| GDN | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log` | 1,687,500 | 2.6267 | 2.647847 | 0.974841 | 472.146 | 2026-05-27T20:25:03+00:00 |
| M2RNN-CMA | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log` | 1,213,600 | 2.6954 | 2.673693 | 0.984356 | 433.362 | 2026-05-27T20:24:59+00:00 |

Rounded Figure 2/prose values after regeneration:

- E88 / NDM: `0.979` BPB, unchanged from the current paper value.
- GDN: `0.975` BPB, unchanged from the current paper value.
- M2RNN-CMA: `0.984` BPB, drifted from the current paper value `0.993`.

Because M2RNN-CMA drifted at the reported precision, Figure 2, the Figure 2
snapshot table, and matching prose numbers were updated from this same
regenerated data source.

## Config Metadata For Conversion And Loading

Shared run metadata:

- Dataset: `/home/erikg/elman/data/pile.txt`
- Tokenizer: `p50k_base`
- Context/chunk size: `2048`
- Target training steps: `10000000`
- Save interval: `3000`
- Log interval: `50`
- Checkpoint retention: `96`
- Optimizer: `schedulefree`
- BF16: `true`
- Seed: `42`

| Model | Level | Params label | embed_dim | dim | depth | expansion | n_heads | n_state | batch_size | lr | use_triton | resume source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E88 / NDM | `E88` | `1270M` | 1024 | 1664 | 12 | 1.0 | 370 | 32 | 5 | 0.000867767847776187 | 1 | `/tmp/pile_convergence_3arch/ctx2k/e88_repair_from231k_ckpt/levelE88_1270M_20260511_172925/latest.pt` |
| GDN | `fla-gdn` | `1270M` | 1024 | 2688 | 21 | 2.0 | 44 | 64 | 4 | 0.002871 | 0 | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_ckpt/levelfla-gdn_1270M_20260507_180327/latest.pt` |
| M2RNN-CMA | `m2rnn` | `1270M` | 1024 | 1920 | 21 | 1.0 | 370 | 16 | 5 | 0.0006020919750502334 | 0 | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_ckpt/levelm2rnn_1270M_20260509_144653/latest.pt` |

Other loader-relevant args are common across all three candidates unless noted:
`state_expansion=2`, `n_groups=32`, `n_slots=64`, `use_gate=1`,
`use_permutation=1`, `linear_state=0`, `use_write_gate=0`, `use_conv=0`,
`dropout=0.0`, `weight_decay=0.01`, `grad_accum=1`, and `grad_clip=1.0`.
Gate activation is `silu` for E88 / NDM and `sigmoid` for GDN and M2RNN-CMA.

## Guardrail Result

Only source/report files and the tracked Figure 2 source artifacts were changed.
No checkpoints, safetensors files, HF caches, Docker layers, generated PDFs, or
token-bearing files were staged or committed. No model weights were uploaded.
