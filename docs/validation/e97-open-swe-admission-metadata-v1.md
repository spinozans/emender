# Open-SWE proposed repair pool: admission metadata audit

Date: 2026-09-10 UTC. Status: completed; **not source admission**.

`proc_5fc5` completed in **seven seconds**, including four CPU tests and before/
after source checks. The audit reads only trajectory ID, instance ID, repository,
declared license, language, and resolved status. It reads no trajectory messages,
patches, commands, or protected task prompts, and emits no training derivative.

Root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/open-swe-admission-metadata-v1`.
Inventory: `c0633eefc9067d9541aae6e554b53e725f30393277a6c63f1a9aab94b1340c7d`.
Summary: `2b5704f9585fdab6d757b4ec966cac1184b05a7bc1b14592f6a96bf3dc3e676e`.
Cohort receipt: `ba497d77dbbb2d39399fcdc19adb162e6efb4d2d569b7ec19ee066001755f3b3`.
Protected manifest: `939bcd66768884a1e5ec44bcf11fdf5d09602ebdabe86d4caefffd4ace8dbb60`.

## Existing validation split has substantial task overlap

The proposed full-context envelope contains **7,232 trajectories**, 87,107,899
assistant targets, and **1,230 normalized repository identities**.

- Existing train: **7,167 trajectories / 3,745 distinct problems / 86,333,951 targets**.
- Existing validation: **65 trajectories / 65 distinct problems / 773,948 targets**.
- **53/65 validation problems also occur in training** (81.5%).
- **63/65 validation trajectories use repositories represented in training** (96.9%).
- There are 61 shared repositories and 53 shared repository-plus-instance IDs.

Thus trajectory-ID separation is not problem separation. The existing validation
split must not be used as evidence of unseen-task generalization. This is a
concrete overlap finding, stronger than the earlier warning that independence
had not been established. Prior explicitly consumed-example likelihood probes
remain fitting diagnostics; their interpretation does not depend on this split.

A problem-grouped development policy is necessary before an LR fitness panel is
frozen. Repository-disjoint development can provide an additional transfer axis.
Removing overlap now does **not** undo earlier-lineage exposure or create a fresh
final holdout. All of these trajectories descend from the previously used
Open-SWE candidate family; full historical sample exposure remains unaudited.

Multiple demonstrations per problem also matter for sampling and normalization:
7,167 trajectories are not 7,167 distinct tasks. Freeze whether problems or
trajectories receive equal weight, and report repetition explicitly.

## Declared licenses: inventory, not legal verification

| Declared license | Train trajectories | Train targets | Validation trajectories |
|---|---:|---:|---:|
| MIT | 3,922 | 47,024,045 | 36 |
| Apache-2.0 | 2,279 | 27,107,028 | 19 |
| BSD-3-Clause | 909 | 11,582,901 | 9 |
| BSD-2-Clause | 57 | 619,977 | 1 |

No missing/unclassified metadata was found, and no repository had multiple declared
license strings within this cohort. These source-row declarations are not an audit
of licenses at the relevant commits, dataset terms, attribution obligations, or
other admission requirements.

The corpus spans Python, Go, JavaScript, TypeScript, PHP, Rust, Java, and C; the
full language and token-weighted inventory is retained in `results/summary.json`.

## Narrow protected-repository check

**Zero matches** against the four whole-repository exclusions in the pinned real-
repository manifest: python-humanize/humanize, pallets/markupsafe,
more-itertools/more-itertools, and prettytable/prettytable.

This is exact normalized owner/repository matching only. It is not clearance of
all protected panels, aliases, fork ancestry, semantic overlap, or prior exposure.
No protected task content was read.

## Disposition

Keep the existing raw sources and immutable audit receipts. Rebuild source-faithful,
complete-context supervision rather than train the known-incompatible conversion.
Do not admit the 86.3M-target training proposal yet: tool-runtime equivalence,
commentary visibility, a grouped split/sampling policy, broader overlap/lineage
review, and licensing/admission receipts remain gates. The preferred channel
policy is to preserve public commentary as public, not silently move it into
private analysis; that requires a separately reviewed protocol revision and new
sizing receipts. The earlier 87.1M envelope remains only the measured proposal,
not the size of that not-yet-built revised representation.
