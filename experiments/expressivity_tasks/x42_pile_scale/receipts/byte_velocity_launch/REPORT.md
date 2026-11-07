# X42 byte-native searches — launch receipt, 2026-11-07 UTC

Three 150M-class BYTE CMA-ES searches are live on physical cards 2/3/4:
transformer/GDN2/bounded pure respectively. Card 1 token pure search retained its
original service MainPID 3442279 and configuration. Cards 0/5/6 remain reserved.
No WorksGood machinery was started. No corpus text materialized, no /mnt/nvme2n1
write, no ~/sxgc access, no .pt committed. Python exclusively the project .venv.

Registration chain pushed before each corresponding attempt trained:
217b7263 scale-board amendment; 011ad929 byte protocol and source/stream identity;
e4350cba detached broker amendment; 2a3cbabd fixed-shape masked padding amendment; b9aeb960 common-document
occupancy-length probe amendment. All preceded their corresponding training.
Current registration SHA256:
`98aa0b6844f04bce78cc612a6f64f547882e054b468c71db24c50d636161fd9e`.
It binds 228 dependencies plus source stat, offsets/document index digests and
C14 binding receipt. Every consumed document slice is rehashed at read-by-offset.
Two token-truncated documents stay excluded. No token nominations transfer.

The initial user services reparented broker wrappers to the systemd user manager,
violating literal PPID=1. Those attempts were interrupted, retained and unscored.
User scope units now preserve service cgroup management while allowing init-owned
Popen launchers/wrappers: `emender-byte-search-transformer-common-prefix.scope`,
`emender-byte-search-gdn2-common-prefix.scope`, `emender-byte-search-pure-common-prefix.scope`.
All use DEVNULL stdin, new sessions and retained logs. The wrappers and parent
search processes have PPID=1. Broker ownership/starttime and at least one
heartbeat renewal were checked against the initial lease snapshots.

A second engineering interruption addressed variable-tail kernel compilation.
The candidate populations/RNG were retained without any prior score. Every
candidate restarted from initialization under the committed masked-padding
amendment. Interrupted spans are retained in the two failure receipts; no partial
checkpoint or recovery of those spans is claimed. Training keeps all real tails,
masking padded targets (-100). Proxy validation uses 24 full document windows to
keep padded positions out of occupancy telemetry. Loss/gradient equivalence is
tested, including raw byte IDs 0/255 and zero gradient at padded positions.

`live_process_checks_all_advancing.json` attests advancing steps/real bytes in
all three current common-prefix attempts, PPID=1, owner/starttime and heartbeat renewal. Actual
parameter totals: transformer 150,003,216; GDN2 149,855,670; pure 150,005,952
(148,982,784 trainable). All meet the registered total-capacity tolerance 0.1%.
The first pure full update took 207 active seconds including first-use kernels;
this is measured overhead, not a claimed pure throughput advantage.

Generation 1's 12 sampled candidate vectors and complete CMA/RNG state per family
are persisted in runs/byte_velocity_{transformer,gdn2,pure}_v1/generation.json.
All three first candidates completed and their scores were banked in the atomic
generation states. `first_banked_scores.json` retains complete results, policies,
result/state hashes, measured byte counts and registration identities. Every
result is eligible, has at least 2700 active seconds and evaluates 49,152 real byte
targets. These are generation-1 candidate-0 proxy results, not nominations or
final-test comparisons.

| Family | Active seconds | Real training byte targets | Proxy NLL (nats/byte) | Search score |
|---|---:|---:|---:|---:|
| transformer | 2701.100 | 100,300,090 | 1.633290842 | 1.633290842 |
| gdn2 | 2701.021 | 84,833,095 | 1.478350341 | 1.478350341 |
| pure | 2702.710 | 5,944,489 | 3.275108536 | 3.347923984 |

The derived evaluation wall times were 0.390s / 2.675s / 77.879s for
transformer / GDN2 / pure. Pure's score includes the registered penalties;
compare raw NLL separately. Both baseline evaluations emitted checkpoint warnings
under no_grad; their completed finite scores and banked generation states are
retained. These warnings are not training failures.


## Artifact heartbeat for the parent

Already armed: `emender-byte-velocity-heartbeat.service`, polling every 60 seconds.
Read `runs/byte_velocity_heartbeat/latest.json` and `polls.jsonl` for per-family
training-log byte growth, latest step/real-byte counts, artifact age, banked scores,
failures and nomination presence. Staleness threshold is 1800 seconds; a service
ActiveState is never progress evidence. This records artifacts but sends no
external alert. Parent should arm its own alert on missing poll growth, `stale`,
`failure`, or nonadvancing step/byte counters, allowing explicitly observed setup
and evaluation phases. Use the latest attempt's artifacts, not a prior attempt.

Current parent commands are `byte_velocity parent --arm FAMILY` under the scopes
above. A failed family stops without a score. Before restart, verify its broker
owner is dead and its lease state, retain/archive failure evidence, and invoke the
same command in a new detached user scope; no preemption or candidate checkpoint
resume. Do not start a second live parent. Sources and JSON registration are
immutable while services run; changing a bound source fails the next preflight.
Transient scopes must be recreated after reboot with the same arguments/state.

## Horizon and flagship launch gate

12 generations × 12 candidates × 2700 seconds = 108 active card-hours per family,
4.5 days before setup/evaluation/overshoot and host contention. Planning: around
November 11–12 UTC (4.5–5+ days from this launch), subject to failures and overhead.
The suggested 6h/generation/3-day pace is not compatible with this serial
12×45-minute population budget. Each family uses one card; families run concurrently.

All three committed byte nominations, qualified 1.3B scale configurations, cap
nonbinding proof, Wikipedia spine bindings, precision/operation checks and signed
compute/extraction/storage budgets gate the six-arm BYTE flagship. No 1.3B training
is launched here. Token velocity nominations complete separately on their own board.

Validation: 23 focused tests passed (byte boundary/tamper checks, vocabulary/capacity,
search endpoint ranges, masked loss/gradient normalization, same-document length probes, CMA state and score
penalties); Python compilation and shell syntax checks passed. No resilient DiLoCo
behavior changed: compute-pool gap-matrix requirement IDs are not applicable.

## Common-document scoring correction

A third engineering interruption corrected occupancy length probes that selected
different documents at different lengths. The current registered probes all use
source document 289, start 0, at lengths 512/1024/2048/4096. No completed score
existed before this correction. Entire attempts were discarded, retaining their
last completed counters and failure reason. The active initial vectors/RNG are
unchanged; there was no selection using held-out scores. The current live checks
above were refreshed under the new registration after fresh steps and lease
renewals in all three families. Earlier checks remain archived.

The discarded engineering attempts consumed at least 1.749 active
card-hours in completed updates across three cards. Additional in-flight partial
updates and setup/execution overhead were lost and are not quantified as saved
progress. No checkpoint recovery or zero-cost retry is claimed.

## Final fresh progress checks

`post_bank_progress_before.json` and `post_bank_progress_final.json` prove fresh
step/real-byte/log growth in each family's next candidate under the same live
wrapper PID/starttime. All wrappers still have PPID=1, matching live broker
ownership and renewed heartbeats. Each family has one banked score and is
continuing generation 1. `artifact_heartbeat_final.json` independently records
current growth and banked scores, with no stale/failure flags. Source stat identity
still matches the registration; token service and reserved-card evidence are
refreshed separately.

Latest next-candidate counters: transformer step 295 / 32,266,763 real byte targets, gdn2 step 231 / 25,234,177 real byte targets, pure step 1 / 96,023 real byte targets.

Pure's second candidate completed its first update in 481.990 active seconds, including cold native compilation. This observed overhead is retained inside its 2700-second budget; setup/compilation is not represented as completed optimizer steps. Future overhead/failures can extend the horizon.
