# E97 Open-SWE private-analysis cap audit

Date: 2026-09-08

## Status

Complete, measurement-only, and non-training evidence. This audit does not authorize
truncation or summarization and emits no source reasoning text.

## Immutable inputs

- Action-only Open-SWE authority manifest:
  `382159f77d9075a99facf987a7b1f7a674fdf969762e21e0a862ec59ba303e60`
- Prior token audit:
  `2a93c05fe0c879c8953e0e28388a5ec6d2ff697965778a4954a86a750f9ac3fd`
- Tokenizer: `p50k_base`
- Auditor: `scripts/audit_e97_open_swe_analysis_caps.py`

## Result

Artifact:

`/mnt/nvme2n1/erikg/e97_systematic_posttraining/open_swe_reasoning_cap_trajectory_audit.json`

SHA-256:

`5b536a4ec4a2225ace2c75448acecc38417e5e90c45421c4fae8f577ae8c278e`

Source-binding attestation:

`/mnt/nvme2n1/erikg/e97_systematic_posttraining/open_swe_reasoning_cap_trajectory_audit.attestation.json`

Attestation SHA-256:

`d394ff2f728ed1b258fade5d31f788eb45e226b2303f6190233b84e0cd7e167f`

Auditor source SHA-256:

`125df0dfc11a22405bf21e33205d06742c7de4471da9d6f953b8a515a0291315`

The 14,363 resolved trajectories contain 760,526 assistant reasoning messages.
Exactly 64 messages exceed 2,048 tokens. They occur in 60 complete trajectories,
span 50 repositories, and range from 2,056 to 8,891 tokens. Language counts for
the affected trajectories are:

| Language | trajectories |
|---|---:|
| Go | 18 |
| Java | 5 |
| JavaScript | 6 |
| PHP | 2 |
| Python | 14 |
| Rust | 3 |
| TypeScript | 12 |

After the existing source filters and action normalization, 14,314 trajectories
remain candidates. Complete-trajectory exclusion effects are:

| Per-message cap | excluded trajectories | retained trajectories | excluded assistant messages | excluded reasoning tokens |
|---:|---:|---:|---:|---:|
| 1,024 | 1,249 | 13,065 | 82,122 | 13,767,523 |
| 2,048 | 60 | 14,254 | 4,525 | 946,206 |
| 4,096 | 3 | 14,311 | 177 | 41,258 |

For the 2,048-token policy, complete-trajectory exclusion removes:

- 60 / 14,314 trajectories (0.4192%);
- 4,525 / 758,160 assistant messages (0.5968%); and
- 946,206 / 80,684,738 reasoning tokens (1.1727%).

The larger message count is intentional: when one reasoning message exceeds the
cap, every turn from that trajectory is excluded so the converter never emits a
partial semantic trajectory merely to satisfy the analysis bound.

The existing action normalizer rejects 24 additional source-eligible trajectories:
2 empty replacements, 11 `insert`, 1 `insert_line`, 7 `undo_edit`, and 3 unsupported
`grep` command variants. These are conversion-coverage exclusions, not cap
exclusions, and must remain explicit in matched-branch receipts.

## Decision

Use a provisional **2,048 `p50k_base` token cap per emitted private-analysis
message**, plus the protocol's 65,536-byte hard limit. If folding a source-local
`think` turn into the next executable action would exceed either bound, exclude
the complete trajectory. Do not truncate. Do not summarize without a separately
versioned, replay-audited transformation.

For representation experiments, action-only and bounded-private-analysis arms
must be intersected on the exact retained trajectory identities. Report source
trajectories, emitted segments, assistant target tokens, reasoning target tokens,
and all complete-trajectory exclusion receipts separately.
