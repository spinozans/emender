# F2 — E4/E6/E7 implementation report (bank remains paused)

## Scope and publication
- E4: `206bc899` — metadata binding plus non-default-checkpoint regression; pushed to origin/main.
- E6/E7: `0e8b9938` — global refresh dedupe, bounded replay policy, development exclusion, emergency backoff; pushed to origin/main.
- E6 preserved-evidence backstop: `cde904a5` — claim-time filtering and executable-pending accounting; pushed to origin/main.
- All NVMe production edits copied verbatim to `scripts/rl-loop-v1/{rl_bank.py,rl_bank_coordinator.py,standing_supply.py}`; all three `cmp` checks passed.
- No bank/ files, historical evaluation artifacts, lane/packer/trainer/driver/unit-test sources, or emender-paper files were edited. No daemon, training, GPU evaluation, scheduler submission, or bank restart was launched.

## E4: checkpoint identity
Both `summary.json` and `model-before-private.json` now emit `plan["checkpoint_sha256"]`. The default SHA remains only the freeze-time default; loading already used the plan and is unchanged.
The regression executes the real `run()` control flow with mocked GPU/tool dependencies, checks the non-default load path, and verifies summary SHA == model-before SHA == frozen plan SHA.

### Historical correction notice
Read-only inventory of `summary.json` under `/mnt/nvme2n1/erikg/e97_systematic_posttraining` found **65 known-mislabeled Stage-B summaries**. Each was matched to its frozen plan by the summary’s `plan_sha256`; no unmatched Stage-B summaries were found in this inventory. All report the obsolete SHA `e9c2d47b24dc419b3ec6c354a77ea585dd00cfc6697a086e1d975ddfc08e3296`. The table gives the authoritative plan SHA instead; historical bytes were NOT rewritten. The same evaluator defect affected model-before metadata.
This corrects metadata, not historical intended-checkpoint runner misbindings. For example, the original m28 plan binds m27; its corrected r2 plan is separately listed. No weight-loading or metric correction is claimed here.

Paths below are relative to `/mnt/nvme2n1/erikg/e97_systematic_posttraining/`.

| Known-mislabeled summary | Digest-matched frozen plan | Authoritative checkpoint SHA256 |
|---|---|---|
| `e97-e1-chat-agent-prep-v1/e1-juncture-evals/seg1-u128/stage-b/summary.json` | `e97-e1-chat-agent-prep-v1/e1-juncture-evals/seg1-u128/stage-b-plan/plan-private.json` | `5ee1e7d303ed8d85b15de97bbfd097911f8d20390cd7c1c6e530d521b3914142` |
| `e97-e1-chat-agent-prep-v1/e1-juncture-evals/seg3-u128/stage-b/summary.json` | `e97-e1-chat-agent-prep-v1/e1-juncture-evals/seg3-u128/stage-b-plan/plan-private.json` | `3bc3f1a53bdf0a14b76e0694d788a30a1a4f24a8815dcfd66110d6cfa6cbaf5c` |
| `e97-e1-chat-agent-u256-dual-gate-v1/stage-b/summary.json` | `e97-e1-chat-agent-u256-dual-gate-v1/stage-b-plan/plan-private.json` | `853a3b95d0b60a61f8cbfacf9564d4f7140a1bcaded4af091562e711ea91cd8b` |
| `e97-e2-chat-agent-prep-v1/e2-juncture-evals/seg1-u128/stage-b/summary.json` | `e97-e2-chat-agent-prep-v1/e2-juncture-evals/seg1-u128/stage-b-plan/plan-private.json` | `ca2f1374729b01da48257bee32ab8becf1fd016e30b088ddf5aa032b089d87e3` |
| `e97-e2-chat-agent-prep-v1/e2-juncture-evals/seg3-u128/stage-b/summary.json` | `e97-e2-chat-agent-prep-v1/e2-juncture-evals/seg3-u128/stage-b-plan/plan-private.json` | `0a812191432df50fc1784a76d94c6f0bd368de292bb41031f954fa16901bccce` |
| `e97-e2-chat-agent-u256-dual-gate-v1/stage-b/summary.json` | `e97-e2-chat-agent-u256-dual-gate-v1/stage-b-plan/plan-private.json` | `03dad2acf7978845fe2d8e5bf0343235c50a0612a19343408c482993580456a4` |
| `e97-e2-chat-agent-u512-dual-gate-v1/stage-b/summary.json` | `e97-e2-chat-agent-u512-dual-gate-v1/stage-b-plan/plan-private.json` | `5ea4e078329f0d960b7311b09bccf26def3913ef818dfb0d109d91cecd08a7e1` |
| `e97-e3-chat-agent-prep-v1/e3-juncture-evals/seg1-u128/stage-b/summary.json` | `e97-e3-chat-agent-prep-v1/e3-juncture-evals/seg1-u128/stage-b-plan/plan-private.json` | `8e0d425cc5cf2db9aa8f019e1740d728d681a9697185ba85e527162c6ab9eed0` |
| `e97-e3-chat-agent-prep-v1/e3-juncture-evals/seg2-u256/stage-b/summary.json` | `e97-e3-chat-agent-prep-v1/e3-juncture-evals/seg2-u256/stage-b-plan/plan-private.json` | `3770621c66563e8d86eb10dc1320e068c12e4bb89815aef823c7f8f5bd9c36d0` |
| `e97-e3-chat-agent-prep-v1/e3-juncture-evals/seg3-u384/stage-b/summary.json` | `e97-e3-chat-agent-prep-v1/e3-juncture-evals/seg3-u384/stage-b-plan/plan-private.json` | `f309bebf7aaf513182a742ae9cceced1c329a4f02f2f6fe18182843d027316dc` |
| `e97-e3-chat-agent-prep-v1/e3-juncture-evals/seg4-u512/stage-b/summary.json` | `e97-e3-chat-agent-prep-v1/e3-juncture-evals/seg4-u512/stage-b-plan/plan-private.json` | `34660c0ce911b8b0f2b356c7c0a3c903fc379accbb79ce2446293cf020f62aa0` |
| `e97-e4-corpus-arc-v1/e4-juncture-evals/seg1-u64/stage-b/summary.json` | `e97-e4-corpus-arc-v1/e4-juncture-evals/seg1-u64/stage-b-plan/plan-private.json` | `1cca3561938f2e6f7a3199b5ce5ab629d412a4470be0b069141b93b8859958c3` |
| `e97-e4-corpus-arc-v1/e4-juncture-evals/seg2-u128/stage-b/summary.json` | `e97-e4-corpus-arc-v1/e4-juncture-evals/seg2-u128/stage-b-plan/plan-private.json` | `d65c871faf055de7915973542f3c7a8d63328dbf4e4bea721ba86d2e11e6e736` |
| `e97-e4-corpus-arc-v1/e4-juncture-evals/seg3-u192/stage-b/summary.json` | `e97-e4-corpus-arc-v1/e4-juncture-evals/seg3-u192/stage-b-plan/plan-private.json` | `03a40f6777dc55b0693550e21b6489c5681757e4b3e433bec2f9b99a160d2bdc` |
| `e97-e4-corpus-arc-v1/e4-juncture-evals/seg4-u256/stage-b/summary.json` | `e97-e4-corpus-arc-v1/e4-juncture-evals/seg4-u256/stage-b-plan/plan-private.json` | `df0bb5ee0adcf4dfae7afe154d741b499444ec5699dde0028bdb6385ca34eb5c` |
| `e97-e4-corpus-arc-v1/e4-juncture-evals/seg5-u320/stage-b/summary.json` | `e97-e4-corpus-arc-v1/e4-juncture-evals/seg5-u320/stage-b-plan/plan-private.json` | `1d44958ac22db1b5df1feb7792988c3ee4977fd8705b47e7a0e355b0526f82f7` |
| `e97-e4-corpus-arc-v1/e4-juncture-evals/seg6-u384/stage-b/summary.json` | `e97-e4-corpus-arc-v1/e4-juncture-evals/seg6-u384/stage-b-plan/plan-private.json` | `8eb1e45138a41620aeec2b04513fa90ee1273eb22b5d3866215d6d26c32dfa1f` |
| `e97-rl-loop-v1/bank-gate-v1/stage-b/summary.json` | `e97-rl-loop-v1/bank-gate-v1/stage-b-plan/plan-private.json` | `7479164b45ace9ef0f964ced40bae4bd8e6706cd1e56bf5550b025d20c2570c7` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-0028-stageb-r2/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-0028-stageb-r2/stage-b-plan/plan-private.json` | `9fcc5e3bf2f41993dbf1d71eeef130c230bcf73d1227760ca088bcd41b036087` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-0029-stageb-r2/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-0029-stageb-r2/stage-b-plan/plan-private.json` | `70465b5256c0c78ecc996ab1a2feae1ce0d1c51650bbc68f5790beea757b3175` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-0030-stageb-r2/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-0030-stageb-r2/stage-b-plan/plan-private.json` | `0b702154a9050f24d879eb0225399386182fcd01638dcc9e811e6967dc8b7f79` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-011-fast-read-v1/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-011-fast-read-v1/stage-b-plan/plan-private.json` | `51cafff22b9505746957d88cc5854cf6e5a6ae18d15871f36b7ad0ab93b65cda` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-013-fast-read-v1/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-013-fast-read-v1/stage-b-plan/plan-private.json` | `db81ad190fe866d2d9028e55ec8f403435000bfdb73223461379fbc91c131e97` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-027-fast-read-v1/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-027-fast-read-v1/stage-b-plan/plan-private.json` | `9738ca8428c7ab2d022c161f195c48dc2187c759419834f0f02b41698c54c82f` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-028-fast-read-v1/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-028-fast-read-v1/stage-b-plan/plan-private.json` | `9738ca8428c7ab2d022c161f195c48dc2187c759419834f0f02b41698c54c82f` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-031-fast-read-v1/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-031-fast-read-v1/stage-b-plan/plan-private.json` | `d142822e654d3a06e1d31630d3324784b81163e1d7690b6742f54ee528ebb9e0` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-032-fast-read-v1/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-032-fast-read-v1/stage-b-plan/plan-private.json` | `c8ecf976db430028aec23bd3b0debe4bbf29f7b0c1773a41747265dc7173c31f` |
| `e97-rl-loop-v1/bank-heldout-reads/merge-033-fast-read-v1/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/merge-033-fast-read-v1/stage-b-plan/plan-private.json` | `1973647489dbfc4277025087bc093895f9932ab831da1577e6fe37794a357eaf` |
| `e97-rl-loop-v1/bank-heldout-reads/wash-verdict-read-v1/stage-b/summary.json` | `e97-rl-loop-v1/bank-heldout-reads/wash-verdict-read-v1/stage-b-plan/plan-private.json` | `2f2b209d3b13f5ed5a7205aefa15188091cfd7fd1bc366d0ea414f2e012457e0` |
| `pi-native-repair-eval-v1/stage-b/summary.json` | `pi-native-repair-eval-v1/stage-b-plan/plan-private.json` | `0085efcdd82299fc9ce16be8e8d8464bcee2b92bdb4cad991e77918de48ed908` |
| `pi-native-repair2-eval-v1/stage-b/summary.json` | `pi-native-repair2-eval-v1/stage-b-plan/plan-private.json` | `2f62a5c9acb50d442da161d3f0d698399b1007d4186077d2c58f91dd2258eaa7` |
| `pi-native-repair3-eval-v1/stage-b/summary.json` | `pi-native-repair3-eval-v1/stage-b-plan/plan-private.json` | `cc7eabcd69e292bf94cf743a64207240e5402ce19ef02a448e2fd03e243d04d7` |
| `pi-native-repair4-eval-v1/stage-b/summary.json` | `pi-native-repair4-eval-v1/stage-b-plan/plan-private.json` | `5d4b82d78df573863b197e0c37a615dd5b9574cf84a0c8a818da308af63b0e26` |
| `pi-native-repair5-arc-juncture-evals/seg1-u32-stageb/stage-b/summary.json` | `pi-native-repair5-arc-juncture-evals/seg1-u32-stageb/stage-b-plan/plan-private.json` | `ebc3db609060179f632fbf5fe41166f9a75e38642685d842e80beef37a2a8973` |
| `pi-native-repair5-arc-juncture-evals/seg1-u32/stage-b/summary.json` | `pi-native-repair5-arc-juncture-evals/seg1-u32/stage-b-plan/plan-private.json` | `ebc3db609060179f632fbf5fe41166f9a75e38642685d842e80beef37a2a8973` |
| `pi-native-repair5-arc-juncture-evals/seg2-u64-stageb/stage-b/summary.json` | `pi-native-repair5-arc-juncture-evals/seg2-u64-stageb/stage-b-plan/plan-private.json` | `695a17bf34cae260dca0212498ee55db1f99eaec8d91090d0421a01b95001541` |
| `pi-native-repair5-arc-juncture-evals/seg3-u96-stageb/stage-b/summary.json` | `pi-native-repair5-arc-juncture-evals/seg3-u96-stageb/stage-b-plan/plan-private.json` | `329dd73b70bce3da95bd2ebf7789ec301c988dd8284ebbed0d8ffdacda078678` |
| `pi-native-repair5-arc-juncture-evals/seg4-u128-stageb/stage-b/summary.json` | `pi-native-repair5-arc-juncture-evals/seg4-u128-stageb/stage-b-plan/plan-private.json` | `d2276376d43c5b573cec2374b6b52b164bd39d7d38d17b8fe085196cb79e5f9d` |
| `pi-native-repair5-arc-u128-dual-gate-v1/stage-b/summary.json` | `pi-native-repair5-arc-u128-dual-gate-v1/stage-b-plan/plan-private.json` | `d2276376d43c5b573cec2374b6b52b164bd39d7d38d17b8fe085196cb79e5f9d` |
| `pi-native-repair6-arc-juncture-evals/seg1-u32/stage-b/summary.json` | `pi-native-repair6-arc-juncture-evals/seg1-u32/stage-b-plan/plan-private.json` | `d03e34d3b3b122cafb13f786f53f4e9258f91c4429201047a872ebfce5737991` |
| `pi-native-repair6-arc-juncture-evals/seg2-u64/stage-b/summary.json` | `pi-native-repair6-arc-juncture-evals/seg2-u64/stage-b-plan/plan-private.json` | `d5c82ec1681d31e6ecf31573877e39d6d4183b4d0a9f3ed2ecf3f79f80e8376f` |
| `pi-native-repair6-arc-juncture-evals/seg3-u96/stage-b/summary.json` | `pi-native-repair6-arc-juncture-evals/seg3-u96/stage-b-plan/plan-private.json` | `d81464982c3ebc0d72769a87e079068bf535d6ca2010185dc1bb03ce264b8f5b` |
| `pi-native-repair6-arc-juncture-evals/seg4-u128/stage-b/summary.json` | `pi-native-repair6-arc-juncture-evals/seg4-u128/stage-b-plan/plan-private.json` | `b61dfb54e82665721b35d7e418b1e7ee90c9bd1307c3280974b321cbf9e72043` |
| `pi-native-repair6-arc-u128-dual-gate-v1/stage-b/summary.json` | `pi-native-repair6-arc-u128-dual-gate-v1/stage-b-plan/plan-private.json` | `b61dfb54e82665721b35d7e418b1e7ee90c9bd1307c3280974b321cbf9e72043` |
| `pi-native-repair6-arc-u96-dual-gate-v1/stage-b/summary.json` | `pi-native-repair6-arc-u96-dual-gate-v1/stage-b-plan/plan-private.json` | `d81464982c3ebc0d72769a87e079068bf535d6ca2010185dc1bb03ce264b8f5b` |
| `pi-native-repair7-arc-juncture-evals/seg1-u32/stage-b/summary.json` | `pi-native-repair7-arc-juncture-evals/seg1-u32/stage-b-plan/plan-private.json` | `06b42f93ffd307f703f30e5ce6d1fba13379f1641294c06039cb78c66d6cca29` |
| `pi-native-repair7-arc-juncture-evals/seg2-u64/stage-b/summary.json` | `pi-native-repair7-arc-juncture-evals/seg2-u64/stage-b-plan/plan-private.json` | `0c966bfae34737ad8bf939c77a33a12dd8de29204bda10300a2085c47249ac07` |
| `pi-native-repair7-arc-juncture-evals/seg3-u96/stage-b/summary.json` | `pi-native-repair7-arc-juncture-evals/seg3-u96/stage-b-plan/plan-private.json` | `1e2f940ef827dae6e66a31f7f62567a2447a3fa34e11fafa9a271952bd96dbcb` |
| `pi-native-repair8-arc-juncture-evals/seg1-u32/stage-b/summary.json` | `pi-native-repair8-arc-juncture-evals/seg1-u32/stage-b-plan/plan-private.json` | `5abd3e6d567bca8ceb37989df0ceb2edf608b65eed0903bc9c5c3204b73fb073` |
| `pi-native-repair8-arc-juncture-evals/seg2-u64/stage-b/summary.json` | `pi-native-repair8-arc-juncture-evals/seg2-u64/stage-b-plan/plan-private.json` | `601b54f65a5527d8f04df011640c65d5170144825c0f36865e39a789ebfefe73` |
| `pi-native-repair8-arc-juncture-evals/seg3-u96/stage-b/summary.json` | `pi-native-repair8-arc-juncture-evals/seg3-u96/stage-b-plan/plan-private.json` | `324a3559c3464477ba96fb897fda07f266cb687738bf2dfefd3e8ac2c0ab5e98` |
| `pi-native-repair9-dose-screen-probe-d0-v1/stage-b/summary.json` | `pi-native-repair9-dose-screen-probe-d0-v1/stage-b-plan/plan-private.json` | `bd348fbb134223ffad5c288e949a3b7a1bc3448c0d48268eec2e5fa1fdddcb30` |
| `pi-native-repair9-dose-screen-probe-d100-v1/stage-b/summary.json` | `pi-native-repair9-dose-screen-probe-d100-v1/stage-b-plan/plan-private.json` | `4d67b9325a5afda6b4813dfc0d73c55e055c175774ff07d623514383cd21e669` |
| `pi-native-repair9-dose-screen-probe-d25-v1/stage-b/summary.json` | `pi-native-repair9-dose-screen-probe-d25-v1/stage-b-plan/plan-private.json` | `d2d8daf804ffea84b75ef20922e81e16ff4eb93def98e142755c3b836b20ab76` |
| `pi-native-repair9-dose-screen-probe-d50-v1/stage-b/summary.json` | `pi-native-repair9-dose-screen-probe-d50-v1/stage-b-plan/plan-private.json` | `dad170ed280995805142b949ca98cb612b04befdb91bc4630b88c6ff0db316ca` |
| `pi-native-repair9-dose-screen-probe-d75-v1/stage-b/summary.json` | `pi-native-repair9-dose-screen-probe-d75-v1/stage-b-plan/plan-private.json` | `a5914a6bde69d5d343f53d11b5671cbf8ffd338d7e2e50c95e9700112575530b` |
| `pi-native-repair9-full-arc-juncture-evals/seg1-u128/stage-b/summary.json` | `pi-native-repair9-full-arc-juncture-evals/seg1-u128/stage-b-plan/plan-private.json` | `426fe4db7028eaa60f462a32de8bdd643c932393aa8ee8f32136d1cb6e077c00` |
| `pi-native-repair9r-full-arc-juncture-evals/seg1-u128/stage-b/summary.json` | `pi-native-repair9r-full-arc-juncture-evals/seg1-u128/stage-b-plan/plan-private.json` | `26fd9da1e32d4478736fd562bf759ea4a6fce7a55719e0d7e2bc2e723de05d6a` |
| `pi-native-repair9r-full-arc-u256-dual-gate-v1/stage-b/summary.json` | `pi-native-repair9r-full-arc-u256-dual-gate-v1/stage-b-plan/plan-private.json` | `bda054f07dc5a81877e0bf87643abb246a9c59ca6d671e49af028b52ee83d9ae` |
| `pi-native-repair9r-full-arc-v3-juncture-evals/seg1-u128/stage-b/summary.json` | `pi-native-repair9r-full-arc-v3-juncture-evals/seg1-u128/stage-b-plan/plan-private.json` | `9f6e0db98230f17095ea473095ecc65475467fefee3b6a1bcebc7aa8d9c52a7b` |
| `pi-native-repair9r-full-arc-v3-u256-dual-gate-v1/stage-b/summary.json` | `pi-native-repair9r-full-arc-v3-u256-dual-gate-v1/stage-b-plan/plan-private.json` | `9ed499fad81d65ff5a8f5cc7cc11bf435fc122e0ca34b718f577d3fdeb8dade0` |
| `pi-native-soup-eval-v1/stage-b/summary.json` | `pi-native-soup-eval-v1/stage-b-plan/plan-private.json` | `201593105c9f17fdfba9be6135ffdc23d58f826c4d5cc2ac4bcda5fe35262ee5` |
| `pi-native-task-vector-sweep-v1/stage-b-eta0.1/summary.json` | `pi-native-task-vector-sweep-v1/stage-b-eta0.1-plan/plan-private.json` | `475bd22b3256111127458491554c14755aa13cc7d84c772c787b597c5b2ecdc9` |
| `pi-native-task-vector-sweep-v1/stage-b-eta0.25/summary.json` | `pi-native-task-vector-sweep-v1/stage-b-eta0.25-plan/plan-private.json` | `81a8bbfb5f9d02f2e04441eaccd9c1234d5862e34c762f611a499cbd827423d0` |
| `pi-native-task-vector-sweep-v1/stage-b-eta0.5/summary.json` | `pi-native-task-vector-sweep-v1/stage-b-eta0.5-plan/plan-private.json` | `bb2c98c2161b64028a881726a9f41e47733e33e4adf469fe4ec2abeac63b3702` |

## E6: repeat prevention and development isolation
- `refresh_pool` indexes global `task_sha256` across pending, live/stale claims, and done, updating the index within the pull to catch aliases. Round IDs no longer authorize implicit replay.
- `MAX_TASK_REPLAYS = 0` explicitly prohibits replay. Each newly frozen task and pull receipt records its own `replay_count` (legacy records default to zero); the pull receipt also records the policy limit. The limit regression temporarily declares one replay and proves the counter stops the third refresh.
- Non-train bundles are skipped before building/extracting a body, with `POOL_SKIPPED ... reason=non-training-split`; they are also retained in the pull receipt’s skipped list. The freeze boundary refuses receipt-ineligible/development tasks.
- Claim-time filtering backstops already-frozen development/receipt-ineligible tasks and round-scoped train stragglers whose content is already claimed/retired. It logs `POOL_SKIPPED` (`receipt-ineligible` or `replay-limit`) without moving/deleting skipped evidence.
- `pool_status.pending_training` counts executable pending tasks, not preserved ineligible/replay files. The coordinator refreshes only when this count and both live/stale claim counts are zero. Total `pending` remains an honest on-disk count.

### Final repeat-prevention accounting
Observed historical receipt: `e97-rl-loop-v1/bank/state/pool-pull-round-3729.json`.
Every one of the four legacy hashes has **3,729 done records** (14,916 total), and none of these four identities remain pending or claimed. These are task retirements, NOT receipts or gradient-exposure counts.

| Legacy identity prefix | Split | Content task SHA256 | Next refresh |
|---|---|---|
| `9f036dbba4b8` | train | `1f501faed3d65e1b51c4cc7bff06dff1cc44c2d18da5d291f1db9e09974b493d` | skip: replay-limit, replay_count=0 |
| `ae2c8b347e9e` | train | `92e4748d768f444ded10d869c3cc8f376153308b48227867cddd1d0eafa9606e` | skip: replay-limit, replay_count=0 |
| `d25ccf0087f9` | development | `5b17d6b44bc56e015738ff6eea1a8b0db434aa4f3455dd33cbe3b01c59335f69` | skip: non-training-split (no body construction) |
| `fa889a723d02` | development | `7abab294b00022a4baeeef88274ee9f7da1e69bdef5588d3a4e962a912e19092` | skip: non-training-split (no body construction) |

Validated this outcome against the REAL sealed lake and pinned validator, with observed-hash tombstones only in a temporary pool: **0 frozen, 2 replay-limit skips, 2 development skips**. The real bank was only read.
The wider paused pool contains **312 pending, 157 executable training pending, 155 development pending, 0 claims, 28520 done**, with 13,920 global content identities. The 155 pre-existing development files remain untouched as frozen evidence; claim-time filtering prevents their execution after any separately authorized resume. A surviving already-retired training copy likewise logs replay-limit and remains untouched.

## E7: supply backoff and recovery
- `LANE_MIN=8` remains the normal operating target; the independent emergency `BACKOFF_FLOOR=2` permits severe-event reductions 8 → 6 → 4 → 2 → 2. Already-below-floor states never increase during backoff.
- Probe degradation and failed-tranche transport reductions use the same floor. Bank-failure clamping never raises concurrency. All unhealthy health events reset the clean streak.
- Recovery remains the existing gradual +2 policy after two healthy clean tranches to soft cap 16; four clean tranches with low-water pressure permit catch-up to hard cap 24. Severe/degraded probes cannot immediately undo backoff by counting as clean.

## Regression tests
`tests/test_eval_e97_pi_native_stage_b.py`:
- `test_stage_b_run_metadata_uses_non_default_frozen_checkpoint` (new; existing two grading tests retained).
`tests/test_rl_bank_pool_supply.py` (new):
- `test_refresh_pool_twice_has_zero_duplicate_content`
- `test_refresh_pool_skips_legacy_content_in_done_or_claims`
- `test_refresh_pool_development_never_enters_pending`
- `test_freeze_pool_task_refuses_development`
- `test_refresh_pool_deduplicates_aliases_within_one_pull`
- `test_coordinator_refresh_waits_for_empty_pending_and_claims`
- `test_supply_severe_backoff_is_monotonic_and_recovers_on_clean_streaks`
- `test_supply_degraded_and_failed_tranche_backoff_stay_below_normal_min`
- `test_supply_bank_failure_resets_clean_streak_without_increasing_lanes`
- `test_refresh_pool_declared_replay_limit_has_its_own_counter`
- `test_supply_daemon_recovers_only_after_healthy_clean_tranches`
- `test_claim_pool_task_preserves_and_skips_existing_development`
- `test_claim_pool_task_blocks_already_retired_round_scoped_straggler`

## Validation and limitations
- Host: lambda01, project `.venv/bin/python` (Python 3.12.3); not Frontier, so no Frontier module activation or Slurm queue evidence applies.
- `.venv/bin/python -m pytest tests/test_eval_e97_pi_native_stage_b.py tests/test_rl_bank_pool_supply.py -q` — **21 passed** (3 E4-file cases; 18 E6/E7 cases).
- `.venv/bin/python -m py_compile scripts/rl-loop-v1/rl_bank.py scripts/rl-loop-v1/rl_bank_coordinator.py scripts/rl-loop-v1/standing_supply.py` — passed.
- `.venv/bin/python /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts/rl_bank_unit_test.py --scratch /tmp/emender-f2-bank-unit.5kNBvg` — **67 checks passed**, including real process claim races, sealed-lake admission, merge/trigger/recovery checks, and concurrently delivered F1 channel/exposure regressions. No modifications to that unit-test source by F2.
- Read-only historical inventory: 65 mismatches, 0 unmatched plans; real-lake temporary refresh: 0 new tasks; three deployed/repository copy comparisons: identical.
- `git diff --check` and scoped cached diff checks passed. Commits used scoped adds, `git pull --rebase origin main` before push, no force push.
- Initial historical-inventory harness rejected a list-valued non-Stage-B summary; fixed by checking dictionary type. Initial claim regression used a nonexistent retirement helper; corrected to `retire_task`. All final validations pass.
- Architecture authority read: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` (ADR-003 decision and retained research authority), companion gap matrix. Applicable safety intents: R07 identity/immutable evidence, R14/NDP13 bounded discovery/backoff, R16 evidence discipline. This narrow prototype-bank repair does NOT claim production ADR-003, elastic/native, V21S, ISP, overlap, or scale conformance.

## Approved deviations / remaining boundaries
- Supervisor approved the new E6/E7 regression file instead of editing F1-owned `rl_bank_unit_test.py`.
- Supervisor approved waiting for all claims as well as executable pending work before discovery; this changes pool-buffering timing versus refreshing merely when claimable work reaches zero.
- After discovering 155 existing development tasks, supervisor approved the claim-time filter and `pending_training` accounting. Frozen files remain in place; supply’s historical raw-pending telemetry/watermarks remain unchanged by this repair.
- The global historical index is rebuilt from durable pool evidence (observed ~1.5 seconds for 13,920 content identities). Preserve done records: deleting/archiving them without a retained identity index would remove dedupe authority. Refresh publication is still owned by the single coordinator; this is not a new multi-producer semantic-lock protocol.
- No evaluation for development tasks was implemented; no model capability gain or resume authorization is claimed. External review remains required before restart.
