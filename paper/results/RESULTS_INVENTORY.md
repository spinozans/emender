# Results durability inventory — P13

Branch: `paper-1p3b-spine`. Snapshot before inventory commit:
`738236d6b7ad4517d35815c18b8ea0c1ce92e367` (P13 Parts 1/2). All paths are repository-relative
unless absolute. SHA-256 is of the exact file bytes, not a JSON normalization.
Sizes are bytes; **>100KB** means >100,000 bytes (decimal KB).
Each TRACKED entry was checked against `git ls-files` on this branch and the
working-tree bytes matched `git show HEAD:<path>`. This is a durability
manifest, not a revalidation of every result or a replacement for the current
manuscript's scopes, negatives, or protocol qualifications.

152 tracked artifacts; 14 tracked files exceed 100KB.
The large files are retained exceptions, not described as tens-of-KB artifacts.
No input result files were copied, reformatted or modified for this inventory.

## Tracked load-bearing inputs and provenance

| Path | Bytes | SHA-256 | Status | Role |
|---|---:|---|---|---|
| `docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md` | 12047 | `015e7631bf30339e3d85227678e139b25a2a2014912f56fed762bb2885913a05` | TRACKED | E97 run/checkpoint identity, S3 durability receipt and launch recipe |
| `docs/GDN2_E97_MATCHED_LOSS_COMPARISON_20260723.md` | 5047 | `93a01fefdc65d1067c4f587d63797e802e4252137bf16eef248ec795f0bd449b` | TRACKED | Early matched-token loss table and averaging-window method |
| `docs/GDN2_OPS_SNAPSHOT_20260728.md` | 6438 | `38addfa75ed2d92ecfe1f98eb90ad03db91a628a0d79324435a34a5b363e5b4a` | TRACKED | Final matched-cutoff loss table and rank-0 alignment statistics |
| `docs/repro/cmaes_1300m_anchors/anchors_corrected.json` | 3052 | `2e889795d4835fc86760102c4978f389ca4a5419442bd31a822716224f432fdc` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/cmaes_1300m_anchors/anchors_gdn2_primary.json` | 404 | `6bd8c03a35779b3daa01baa42810626cf2fd36b271828fe24d14262c2924dbfc` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/cmaes_1300m_anchors/anchors_missing_20260531.json` | 1476 | `c2ef27965647c40ddfac6b2bf4af8d8c420907686e1c8054e1f715dad12321aa` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/lb_emender_mix_20260612/anchors_emender.json` | 613 | `1aea22713c85b3751308de25180c6c6fc977c40636d9cc3ec6d5d4e9704f989a` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/lb_emender_mix_20260612/best.json` | 1090 | `c685478b82c842af5d49a31d7bf27a44b5a9ce91ab38833aae578d792d23dc0b` | TRACKED | CMA winner geometry/fitness binding; family identified by parent path |
| `docs/repro/lb_emender_mix_20260612/emender_mix_generations.jsonl` | 9715 | `bc1ebaadeda22a558ad3aa1f9172e1f0fdc8075d2968d5900039b5b6ee58c08a` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/lb_emender_mix_20260612/emender_mix_results.json` | 49629 | `539c1157da530b2e9efd2ecd58b97c38eb009af56ff9de1f841a80fcb6928b43` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/lb_gdn2_mlp_20260612/anchors_gdn2_mlp.json` | 500 | `24f7348ec966edc7d6dab6cae7c7b1ae809cae1ab202141dbe77df120d347c9d` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/lb_gdn2_mlp_20260612/best.json` | 533 | `866bd81066140c25e827da0da4ebbf76c7b980f59084f371c61f4d814bbd4ee3` | TRACKED | CMA winner geometry/fitness binding; family identified by parent path |
| `docs/repro/lb_gdn2_mlp_20260612/gdn2_mlp_generations.jsonl` | 10086 | `86f271bd88b21a5a8ef9ad4b43159306421b9cc23036f7a5299e0146cd0ece17` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/lb_gdn2_mlp_20260612/gdn2_mlp_results.json` | 51495 | `f9f03bc91b83c01ee670d406d5964611188d146b3b256ddf53aa89f4b541fd18` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/lb_m2rnn2_20260612/anchors_m2rnn.json` | 671 | `0fe964e4dff2fa6c8dc2d0377b0c81af0418e038cb99b364745f5462dbe01186` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/lb_m2rnn2_20260612/best.json` | 971 | `59d055257126e0c5e3b0976241ce6ac6b0dd9d64752922791e884a720a35cdb8` | TRACKED | CMA winner geometry/fitness binding; family identified by parent path |
| `docs/repro/lb_m2rnn2_20260612/m2rnn_generations.jsonl` | 9299 | `baf09053d1f5b501f0b974e50c8769eb793fcbdc7fad0b05ed69f3716dd65cad` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/repro/lb_m2rnn2_20260612/m2rnn_results.json` | 47220 | `9b5113e0ea553695669b7a0664c7d8baaf872a07d8b872cb54ac57443a134630` | TRACKED | CMA retained results/anchors/generations supporting winner and sweep binding |
| `docs/validation/e97-huggingface-dense-base-and-cli-agent-release.md` | 3788 | `07dce5ff8f186818c29f2213cd868b9cfb7840ade9ed353d932ed8deeb0dc287` | TRACKED | E97 terminal source checkpoint SHA-256 and release validation binding |
| `docs/validation/gdn2-huggingface-152b-control-release.md` | 4498 | `624330099ecebcc9af50d18e9bf0dd1f3d30f72cdbc2df0d34902ce10040fb74` | TRACKED | GDN2-MLP terminal source checkpoint SHA-256 and release validation binding |
| `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md` | 2206 | `92ae96d2e962cb2ab121f577a9726b0a879e920a4519b47087a9a0157e05434d` | TRACKED | Six-task 8 M sweep table and length-extrapolation summary |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__e97__seed123.json` | 5195 | `2fe0f6543cccc6215f5d0516175083f7241e990f60fdef4888b1b91a60421763` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__e97__seed42.json` | 5195 | `5d2f1d7368de0d002fd88550a37a6243a5f94e5bf61de831609278e5c2f750b4` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__e97__seed456.json` | 5181 | `71b622d7c749db42555a60d3f0f5d9c807add413d628dde7fcb862569b37ddc2` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__e97-linear__seed123.json` | 5207 | `1bb1f3d58aa78f1fe22d189d81e82a37029cf1dccac08e2c4f71f31bcdcc7a99` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__e97-linear__seed42.json` | 5207 | `03cbc8f8d03ff78d2d7c654b7622964fd607e9a32267f392a315a62b018b748c` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__e97-linear__seed456.json` | 5200 | `6d609017573ee36cf740029204675c523f0b6291823fddfb8a796b4d07c07836` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__e97-raw__seed123.json` | 5166 | `88e2a7024ed5c887c5d438599fc2d7d87936dd0d62f86f4325dd1526e68a754b` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__e97-raw__seed42.json` | 5180 | `f5fdb3e8418d77b5c29c4f2f55ac3e74e0c7e21f0a85b6c36850e89dd8fcb239` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__e97-raw__seed456.json` | 5148 | `9107e2d304db92b356978434d30e0a771f744c23d6e04a47a4c58a22db93e3be` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__gdn__seed123.json` | 5201 | `f0ba90d0e9a354bb2fc11619bdcd04fc98be6d2c01037c532e5b4a8ae1174de5` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__gdn__seed42.json` | 5188 | `57fdff732fb916b283dead715226becc0434e7e213fd13e783315c7da6fde640` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__gdn__seed456.json` | 5186 | `335a33761aa342445c34b5a3f9c754cb5214d248aa408123602e85f803693544` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__gdn-8m__seed123.json` | 5642 | `8b23ff388b068d69be3f4afbfd783203a3e06d4acc0751d836bdb3ebe401ce49` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__gdn-8m__seed42.json` | 5629 | `3e99fb91670c1de1bfb43a9c5530a1531c3b2cec484026bada01a0fac4f7a736` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/expressivity_tasks/results/e97raw_iterated_nonlinear_map__gdn-8m__seed456.json` | 5650 | `5818c8691a6966cbe8313fea06125c396d70df7494f6d19d8322e04f2371ad09` | TRACKED | 8 M contractive-control per-arm/per-seed source record |
| `experiments/grok_symmetric_width/confirm_signature.json` | 10868 | `7ef4ec87ca2d51e2e27aa59535e63f645121b8b35f9779a26faf80c7a662af7e` | TRACKED | Small-scale symmetric-width / eight-seed confirmation aggregate and throughput record |
| `experiments/grok_symmetric_width/CONFIRM.md` | 26057 | `8f9c6e8772c03942c2006e739bc50a3a931adafb542ae6c74126711004a9f986` | TRACKED | Eight-seed confirmation table and protocol interpretation |
| `experiments/grok_symmetric_width/RESULTS.md` | 36294 | `56e2289102e9e0804cdcf822f5d710704def8c5d9f9a5c6c4c999b1d1b85e0e3` | TRACKED | Small-scale modular-quadratic and two negative-family tables / conditional seed selection |
| `experiments/grok_symmetric_width/signature.json` | 13499 | `7995fb14d758abba0c7ddf34fcfef0d37e266076c7f75c17c34834c290c2f6be` | TRACKED | Small-scale symmetric-width / eight-seed confirmation aggregate and throughput record |
| `experiments/grok_symmetric_width/throughput_clean.json` | 1295 | `7f690196ce77eb68ecb0dedd098e98bf68739050cdb4d273caaf1976638733e8` | TRACKED | Small-scale symmetric-width / eight-seed confirmation aggregate and throughput record |
| `experiments/grok_symmetric_width/throughput_confirm.json` | 1288 | `916dd4f93bb9c87b65ffec4312917de0026bb461cf9e29a0cebecf290d4df903` | TRACKED | Small-scale symmetric-width / eight-seed confirmation aggregate and throughput record |
| `experiments/lb_compare_20260613/bpb_results_avgonly_run1.json` | 3266 | `c30afe3a8fd014a2884862714055aa6ec9bba65345b9936206b2ec0dfb6546ab` | TRACKED | CMA-stage five-family held-out BPB/separator measurement; distinguish averaged-only diagnostic |
| `experiments/lb_compare_20260613/bpb_results.json` | 3623 | `e5f6e79b8b7b26282fa7f7bfda95ddd900c2a7497ab36a2b99a177d2ccd65a9c` | TRACKED | CMA-stage five-family held-out BPB/separator measurement; distinguish averaged-only diagnostic |
| `experiments/lb_compare_20260613/LEADERBOARD.md` | 9205 | `0abd55410d62c09816ef704ca5d27c0e6b9441604d5c43866ef1e7ab3af93178` | TRACKED | Five-family CMA-stage held-out table (search-budget, not 150 B pair BPB) |
| `experiments/lb_compare_20260613/sep_results.json` | 22599 | `fdff4771390fcb0f685d0e7b6150e790949d8609dc48d9bd11b371ae81a9b565` | TRACKED | CMA-stage five-family held-out BPB/separator measurement; distinguish averaged-only diagnostic |
| `experiments/lb_emender_mlp_20260612/RESULT/generations.jsonl` | 11908 | `dc22720b27ec50872f343e39674c0752de954138bb88aeed27e5dbf013daa582` | TRACKED | E97-MLP carried-in CMA winner binding / retained search records (no separate tracked best.json) |
| `experiments/lb_emender_mlp_20260612/RESULT/results.json` | 60428 | `a7e9062cef56551e489c0e08d01b6ed3d5ae915fb284a7f9dec3deede1e541c6` | TRACKED | E97-MLP carried-in CMA winner binding / retained search records (no separate tracked best.json) |
| `formal/lean/ElmanProofs/PaperCore.lean` | 6018 | `9a8d60f7134a2385ff660b4cae775bc897562f9e3d1da9585d0ab11749e151db` | TRACKED | Trusted-core entry point; inventory maps its recursive local closure |
| `formal/lean/scripts/check_paper_core.sh` | 595 | `949f2672f543370e4d65c789273a8f576d7569ea03a7a0d061103411be4b08f4` | TRACKED | Recursive project-source trust gate; dependency assumptions not covered |
| `paper/data/lean-trust-gate-theorems.json` | 3266540 **>100KB** | `2e3ed61744c20b6f3d7056a63ef69e8c9b83795a2a3d1abbd3b3e87ffee263b3` | TRACKED | 857-declaration recursive trust-gate inventory (17 files); contains full source quotations |
| `paper/ndmpapernotes.md` | 13255 | `8d92a3cf0b086acd3901b1a17a0b4335234bca69e07b088f5ad2b9cbc5f1d658` | TRACKED | Only retained rounded historical 8 M S3/S5 mean/min/max source; no exact three-seed observations |
| `paper/results/cma_flop_rate/overlay.csv` | 155735 **>100KB** | `412158bbc60e3cbcb6e17f7cead4e73a01ac8acc2e73374764df9b6fa1e10533` | TRACKED | Predecessor CMA convergence/threshold trajectory source |
| `paper/results/cma_flop_rate/README.md` | 11657 | `7934565edf6f2197427ab5baa82c4242d283312db0709a315405a2c05bdd2a6d` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/cma_flop_rate/SOURCES.md` | 5421 | `6e4bd691c5fe53d7479ac212ecd1628ec52a3d7c705cf360a3de0a892bab40e3` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/cma_flop_rate/summary.csv` | 691 | `8b868022028cf5659155611867c82568e45f9179a8433d6a6651a55bbb66955c` | TRACKED | Predecessor CMA convergence/threshold trajectory source |
| `paper/results/cma_flop_rate/thresholds.csv` | 615 | `9d175b8553bf6160df12318fe02ee2df54c5750610de77fa30d23be18ed3f81b` | TRACKED | Predecessor CMA convergence/threshold trajectory source |
| `paper/results/cma_flop_rate/trajectory_e1.csv` | 24156 | `eed1fb39e7b0547e85c09e87f2cc9c994afe47a2e9903fc2d072748b3a09af47` | TRACKED | Predecessor CMA convergence/threshold trajectory source |
| `paper/results/cma_flop_rate/trajectory_e88.csv` | 50660 | `9756db3d85148738d47638bb39c56afc252f21eac84d55fb1a1e9de7716691ca` | TRACKED | Predecessor CMA convergence/threshold trajectory source |
| `paper/results/cma_flop_rate/trajectory_fla-gdn.csv` | 52868 | `a7b9148f78328b0b5b4b9b3c5158acbfd8766385c68d7959df458c5a76fc2f05` | TRACKED | Predecessor CMA convergence/threshold trajectory source |
| `paper/results/cma_flop_rate/trajectory_mamba2.csv` | 52977 | `7cf4bc554f07189cbddea796174e7f868e2ba8a441b40d0431bcea304dfb6b54` | TRACKED | Predecessor CMA convergence/threshold trajectory source |
| `paper/results/cmaes_burst_v2/cmaes_eval_summary.csv` | 1226023 **>100KB** | `5105d9db56dc86c80aeef1a9ca7aad5b126e7923d82cb0cc7867152f7fb0a572` | TRACKED | Retained CMA candidate/generation/normalization figure and sweep-table source |
| `paper/results/cmaes_burst_v2/cmaes_generation_summary.csv` | 51457 | `333a246cac56d96f9ec657dc5f019e67b7945afe1e8a54cb906f8cc81f4b65e9` | TRACKED | Retained CMA candidate/generation/normalization figure and sweep-table source |
| `paper/results/cmaes_burst_v2/cmaes_log_manifest.json` | 39318 | `2fa19e9c3ffdc81c155342d93310db41de62e30ad164e0156a0d57d3dbf27d20` | TRACKED | Retained CMA candidate/generation/normalization figure and sweep-table source |
| `paper/results/cmaes_burst_v2/cmaes_normalization_manifest.json` | 33880 | `b9629433e3377fd4d67eae073910d132ec64a44dd928fd6cfee88e31f65381ce` | TRACKED | Retained CMA candidate/generation/normalization figure and sweep-table source |
| `paper/results/cmaes_burst_v2/cmaes_trajectory_points.csv.gz` | 7545894 **>100KB** | `de28ed8625667bbb7d3d54e8a253733aecfdf36acbaa5a5ed9e0e3d0c07674ed` | TRACKED | Retained CMA candidate/generation/normalization figure and sweep-table source |
| `paper/results/evidence/DOSSIER.json` | 24653 | `7ac6514fee5505ad7fbb4104f2f4240a87ecce398a5a60eba616c50c6e8d2691` | TRACKED | Claim-to-artifact provenance dossier (historical snapshot; not current claim authority) |
| `paper/results/evidence/DOSSIER.md` | 8538 | `ed22a64b816e04861526788d712594b54b7511ce934237723f700c1a95d86eb8` | TRACKED | Human-readable historical claim/provenance evidence manifest; outdated claim framing disclosed |
| `paper/results/expressivity_1p3b/capacity/budget.json` | 453 | `b868b35ff14d3a5f67475830e0e8cf711e7439e1eb06800b81a324c58744d5fe` | TRACKED | S5 capacity boundary record, budget, held-out split or failed-stage receipt |
| `paper/results/expressivity_1p3b/capacity/cap_e97_s5_heldout.json` | 315 | `0fedac79eb03e67f9698caa4c4a266ab1706cd13ace59f57f07e5229e9d28c3d` | TRACKED | S5 capacity boundary record, budget, held-out split or failed-stage receipt |
| `paper/results/expressivity_1p3b/capacity/cap_e97_s5_sweep.json` | 1375 | `d2a90da970d63effe4e47f79e4e5e2a7aebe25ccadb0a63688ad7b83a2e0948a` | TRACKED | S5 capacity boundary record, budget, held-out split or failed-stage receipt |
| `paper/results/expressivity_1p3b/capacity/cap_e97_s5_train.json` | 840931 **>100KB** | `d545b11e5953531cadbba41a54fb963c64b421fe013a09871eced0bb8f76c9d7` | TRACKED | S5 capacity boundary record, budget, held-out split or failed-stage receipt |
| `paper/results/expressivity_1p3b/capacity/cap_gdn2_s5_heldout.json` | 316 | `2de49fd84aa091f8bcae8d6243e377ad50d2b806764d59c82c187663d7708d61` | TRACKED | S5 capacity boundary record, budget, held-out split or failed-stage receipt |
| `paper/results/expressivity_1p3b/capacity/cap_gdn2_s5_sweep.json` | 1377 | `5ba59735b035172e2d4f394fbc229cc2d41f7d67ab64d43c83d038013fadcc6d` | TRACKED | S5 capacity boundary record, budget, held-out split or failed-stage receipt |
| `paper/results/expressivity_1p3b/capacity/cap_gdn2_s5_train_crash2_partial.json` | 659653 **>100KB** | `cbcf6493ae6764c89f2d7cf740e778f25e787089110ea847182616eb20938664` | TRACKED | S5 capacity boundary record, budget, held-out split or failed-stage receipt |
| `paper/results/expressivity_1p3b/capacity/cap_gdn2_s5_train_crash3_partial.json` | 660594 **>100KB** | `ca7962acb4c71e3c21447057595bd145d0fd7dce541eaf9e37a103b8f7541e47` | TRACKED | S5 capacity boundary record, budget, held-out split or failed-stage receipt |
| `paper/results/expressivity_1p3b/capacity/cap_gdn2_s5_train.json` | 661639 **>100KB** | `5a38fa21377a745012a2bd5c3a5ccab34591cca763b4706ee89b9b23c01a2b93` | TRACKED | S5 capacity boundary record, budget, held-out split or failed-stage receipt |
| `paper/results/expressivity_1p3b/capacity/PLAN.md` | 6582 | `8b89bd3850341b25256cbdaa8f678a626a3217f07f2d3f3bc141b3d2dfe7edef` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/expressivity_1p3b/capacity/SUMMARY.md` | 3482 | `32a64d1a92531246fedc74892da5758427ea77fc60e0027a5253c173e03c9525` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/expressivity_1p3b/e97_args.json` | 778 | `27e4beafac1a80d58d63edb816b3a5b20e4c268540cc738fe40470bfeb2ddec2` | TRACKED | Current-pair initial S3/S5 fine-tune or zero-shot record / architecture arguments |
| `paper/results/expressivity_1p3b/ft_e97_s3_permutation.json` | 2717 | `b018e09af03815e5893e09da012c6c4db177682229d05fc00ef6a8c0d9600c4f` | TRACKED | Current-pair initial S3/S5 fine-tune or zero-shot record / architecture arguments |
| `paper/results/expressivity_1p3b/ft_e97_s5_permutation.json` | 4486 | `738da71fe2b4a82e6ad6edd47ff9ba032d1b74efc1f86cf7838ada654a258854` | TRACKED | Current-pair initial S3/S5 fine-tune or zero-shot record / architecture arguments |
| `paper/results/expressivity_1p3b/ft_gdn2_s3_permutation.json` | 2715 | `2d798799f4b326ab7d2b96144c964785251cdde98052f9bc27c74288905bfc33` | TRACKED | Current-pair initial S3/S5 fine-tune or zero-shot record / architecture arguments |
| `paper/results/expressivity_1p3b/ft_gdn2_s5_permutation.json` | 4489 | `a712d1ed5e58d8a55396e7d58487de68be3b2261ff7e1ab3fbb375779c249c39` | TRACKED | Current-pair initial S3/S5 fine-tune or zero-shot record / architecture arguments |
| `paper/results/expressivity_1p3b/gdn2_args.json` | 662 | `634e8242d3a4fadf82f35c8e47253fc0e3cc5c2353c0fd0e058ea55edfa7ad7a` | TRACKED | Current-pair initial S3/S5 fine-tune or zero-shot record / architecture arguments |
| `paper/results/expressivity_1p3b/iterated-map/budget.json` | 611 | `fb728e3256849c576360850f705c37521cf0827ed9bc7b7c56c936969b2078e3` | TRACKED | Current-pair depth-wall/control table input or pre-registered budget |
| `paper/results/expressivity_1p3b/iterated-map/ft_e97_modular_quadratic_sweep.json` | 1097 | `e88a996f807a30d7072da2f2332ac4a55266f297fd188e4af3a0a43e0a387565` | TRACKED | Current-pair depth-wall/control table input or pre-registered budget |
| `paper/results/expressivity_1p3b/iterated-map/ft_e97_modular_quadratic.json` | 349541 **>100KB** | `e4fc5eab028a738f9a5f40d983e766893460a70f233d8b1849788fef5adda282` | TRACKED | Current-pair depth-wall/control table input or pre-registered budget |
| `paper/results/expressivity_1p3b/iterated-map/ft_gdn2_modular_quadratic_sweep.json` | 1101 | `dbaaef60987c419c928c8ccbd539834ad06612b0fe1229987354c28d18d0356b` | TRACKED | Current-pair depth-wall/control table input or pre-registered budget |
| `paper/results/expressivity_1p3b/iterated-map/ft_gdn2_modular_quadratic.json` | 308001 **>100KB** | `d9578fa5554f579601af97d2d6d5997fd572c30d7d1e7805fcc13f507b99f3c3` | TRACKED | Current-pair depth-wall/control table input or pre-registered budget |
| `paper/results/expressivity_1p3b/iterated-map/lyapunov_logistic_map.json` | 1161 | `a3e02ceec8b82b41c479da015ea176c1fb3a0f4287a3e709f485f4eab5509ccf` | TRACKED | Current-pair depth-wall/control table input or pre-registered budget |
| `paper/results/expressivity_1p3b/iterated-map/phase1_8m_control_table.json` | 12768 | `6cc15e2db44b6d5ed10adc6e8f03ca522b84c32e728473ada42edf85bd398be0` | TRACKED | Current-pair depth-wall/control table input or pre-registered budget |
| `paper/results/expressivity_1p3b/iterated-map/PLAN.md` | 7896 | `2ff1f5904efafc76de55873fd0ee6213b588011b2f944b8416612eabed44a8d8` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/expressivity_1p3b/PLAN.md` | 6330 | `50b4a23378390d538035aebfa9d3876210548390376542aeb8e19b6fe37a02d6` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/expressivity_1p3b/SUMMARY.md` | 7879 | `7559262e388fbf35881871a5677ef85326fea2933d72e862d307e16b2fddcb99` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/expressivity_1p3b/zeroshot_e97.json` | 2742 | `9b4b4ee6b0e25e180aa25a3201dc9e0a5c46b9ba9dce21da4a3ada9256d1d45b` | TRACKED | Current-pair initial S3/S5 fine-tune or zero-shot record / architecture arguments |
| `paper/results/expressivity_1p3b/zeroshot_gdn2.json` | 2749 | `c3c5e01a0cdfee0956a3ffe34df8b9b4580fe34e38b236d8c7d9d4f00c95d005` | TRACKED | Current-pair initial S3/S5 fine-tune or zero-shot record / architecture arguments |
| `paper/results/figure_2/AS_OF.md` | 9174 | `9c226cc29ed154d6e7b4f98b36dc64128fc142a66ea78d2786bcde5c030908ab` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/figure_2/E88_NDM.csv` | 7636683 **>100KB** | `e9fdb7e8e84000576f4342f1ca8e3bce89aaf0d3495a1efe5aa813b1ecf559c6` | TRACKED | Predecessor training-loss trajectory source CSV |
| `paper/results/figure_2/FLA_GDN.csv` | 10180560 **>100KB** | `d7f398acea572ea7725e0c196623d1cf308b2d201807d37f58d9ce5eef5c406d` | TRACKED | Predecessor training-loss trajectory source CSV |
| `paper/results/figure_2/M2RNN_CMA.csv` | 7463857 **>100KB** | `767c6a6b959e3e1c6d9295c1998997acd77bd4b116941d01a73d8fcd99a33898` | TRACKED | Predecessor training-loss trajectory source CSV |
| `paper/results/figure_2/Mamba2.csv` | 9923584 **>100KB** | `1aeb404d725c3074aa8c20c8f3890ae788299236cfd421cf45792d953de7356d` | TRACKED | Predecessor training-loss trajectory source CSV |
| `paper/results/figure_2/raw/README.md` | 1129 | `2347b75cd78cdb790a9dbc10bd0301df9d54bc88d89167c42d0b7a0ad6ead718` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/figure_2/SOURCES.md` | 3393 | `7f91d342dbacd7a0aa2f8268c3446609cc7ee9c1a7c6270a19444f7b40b54ecb` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/figure_4_hybrid/canon_hybrid_AABB__fsm_tracking__seed123.json` | 4268 | `ca962a586f1f379299f20ace028c71532f99bb6c6257074fa8b4629ff10a7775` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_hybrid_AABB__fsm_tracking__seed42.json` | 4246 | `b1b5e5e5f42d7bed85d3f08fc2fea7547575c70e4cfaa5da88218408cd6b52a7` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_hybrid_AABB__fsm_tracking__seed456.json` | 4262 | `654b58eb02a02647cc7885ce6d34bc86515e4f95a8c886bccbfeb90672c3246b` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_hybrid_AABB__modular_counter__seed123.json` | 4242 | `4a2301e20f621cdb5d94ef4663d8c9086dfe2c296bc327360a527fa340e232ee` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_hybrid_AABB__modular_counter__seed42.json` | 4239 | `2fcba04ef48bf07c9b675ecdc474ac4509c2038d414774492979668ee5fbfe4b` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_hybrid_AABB__modular_counter__seed456.json` | 4244 | `36b4fb53ad8115b14a425295e9f8e21cf27b76c6ee62f9e6c5da70bf763e4b70` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_E88__fsm_tracking__seed123.json` | 3961 | `72732c119d1809a670618cc3d9082ee704e6bcc1e0e1fd240fbfbbf71128a0e2` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_E88__fsm_tracking__seed42.json` | 4018 | `14fdc9dfd8bf062f730367a7d047c96a5383cf13559a73f4bfd5cd09600d6e46` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_E88__fsm_tracking__seed456.json` | 4113 | `77625bc3a1bb84e5a7ef29c923f616947f2919d7933ea1639f7c24ffb25e8b16` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_E88__modular_counter__seed123.json` | 4231 | `5a8ba6557a44ebaaa87e13cb2d894e821b786e6751746015ac5862967fc1b565` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_E88__modular_counter__seed42.json` | 4218 | `8e45f0a6a96571012ea003bfe576516c2cc148dceb11563056f931b841219ac9` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_E88__modular_counter__seed456.json` | 4238 | `c665fd37f5cab69b758d28fbd269c275e992f39299ecfd5ab5296bd3c43fd883` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_FLA__fsm_tracking__seed123.json` | 4269 | `222253cf5c763b281bd925dc359a48b9bd93c6bdf0e90d0443cc7a6250eac154` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_FLA__fsm_tracking__seed42.json` | 4269 | `60c64eec21e2634e3c8d118dcaec69700d6b44b545b2798b16e6f512699166ec` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_FLA__fsm_tracking__seed456.json` | 4259 | `8799a85058fa61a01a538f8afc5313477b685d539c63453508bffd2fad71e599` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_FLA__modular_counter__seed123.json` | 4245 | `54c7d83a32c1bd53d53cfecba4241cf54857c8300d287228f107bc8a0ff49e39` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_FLA__modular_counter__seed42.json` | 4251 | `2fd2836b6f65ec6ea6aebd5282ceb385c4f6b10e090197c687e4c66739bea138` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/canon_pure_FLA__modular_counter__seed456.json` | 4256 | `6ca47663b3e9ae4fc63eb9a559c60d9639eed1c24ed99ab37e6745c2702b0d40` | TRACKED | 8 M canonical hybrid/pure per-seed figure input |
| `paper/results/figure_4_hybrid/SOURCES.md` | 1381 | `6c9aa9549fd0da281e643316e693f053adfd2e1014ebc7a2f7b07070e73756b2` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/results/qa_reasoning/fact_panel_latest.csv` | 2149 | `89f2aceaf5630b0f42fc712d521e94fc3bca90a1ebad48fb02950eba0ea51c9c` | TRACKED | Predecessor fact/reasoning panel table source CSV |
| `paper/results/qa_reasoning/knowledge_probe_40item.csv` | 616 | `1889a8b937b23a317ce45ce1ab8e048b4fa408341ff0d6f100afff8028c89c2e` | TRACKED | Predecessor fact/reasoning panel table source CSV |
| `paper/results/qa_reasoning/racer_panel_300item_progression.csv` | 2781 | `ba4e38c7c1c432b2a8eaa38565ebd42b608730b0a1ebcc7327bd695caf6e210e` | TRACKED | Predecessor fact/reasoning panel table source CSV |
| `paper/results/qa_reasoning/reasoning_panel_latest.csv` | 1985 | `913132abc32fec5110a130ca0348f88a1338db91c3c5cdbdd776c708d7d6e580` | TRACKED | Predecessor fact/reasoning panel table source CSV |
| `paper/results/qa_reasoning/SOURCES.md` | 2934 | `4b6e704a5ec5239d3c456b16037dc46e70199270a6c34b71c5d19acb026e7524` | TRACKED | Source/protocol ledger for the adjacent figure/table result records |
| `paper/review/BPB_FULL_TABLE.md` | 25594 | `4abaeea335385c5b23fe7c2d4e4915829e8121319c96e5525b992b688abf47a9` | TRACKED | Consolidated BPB reference-table source; current manuscript qualifications take priority |
| `paper/review/COMMA_PILE_BPB.md` | 10078 | `4c26265f19cdd938c87a7bcad1a49c9cf30b0866cfdd48cec1beef572aec1432` | TRACKED | Common-Pile contamination-control per-model table source |
| `paper/review/comma_slice.json` | 1026 | `56e017b66b3db4403c3b71f1116ef49c42484a451f53631904b10b120bc5046c` | TRACKED | Predecessor held-out byte/slice manifest or measured NLL/BPB record |
| `paper/review/E88_HELDOUT_HARNESS.md` | 8213 | `3a118e1cc0cd390aa513aa06aa6937a1ad49ecd913608000e0f83f664e87488c` | TRACKED | Predecessor schedule-free inference-weight recovery / NLL validation receipt |
| `paper/review/heldout_harness_json/e88.json` | 630 | `542241bec94ccd79d4a63b864c89003681fb2659a33b67590e8d52000435bb4d` | TRACKED | Predecessor held-out byte/slice manifest or measured NLL/BPB record |
| `paper/review/heldout_harness_json/fla-gdn.json` | 636 | `2fcc787698e0e97fb249b51b8a8786eb3761d444cc978cec7a062979759f540a` | TRACKED | Predecessor held-out byte/slice manifest or measured NLL/BPB record |
| `paper/review/heldout_harness_json/m2rnn.json` | 632 | `5205d92990b4e7698a6cf87ee1e8d56f04bfaf92dc2057c36f9ec242b13dc21b` | TRACKED | Predecessor held-out byte/slice manifest or measured NLL/BPB record |
| `paper/review/heldout_multislice_slices.json` | 2086 | `36843fcc994933be1911970b4938621217427e026f7762e62756016627c6c412` | TRACKED | Predecessor held-out byte/slice manifest or measured NLL/BPB record |
| `paper/review/HELDOUT_MULTISLICE.md` | 5230 | `cd9dcd5d8aedeb8abbd55ac1446cf7f15d8448518a87fedd5ce0d6cc9d791aee` | TRACKED | Predecessor multi-slice per-model BPB table |
| `paper/review/heldout_slice.json` | 671 | `9087218bce8699cd82fe4867be665bed8e293d855cea3618bf58b3cc959e8c34` | TRACKED | Predecessor held-out byte/slice manifest or measured NLL/BPB record |
| `paper/review/s3_s5_finetune_v03_data_matched/e88.json` | 14604 | `8bea7ee9cefa0da0d52f75ec1e9f9fbe376001f174e39243c2f0cb1083bd315a` | TRACKED | Predecessor S3/S5 protocol-specific table source; not current-pair or historical 8 M data |
| `paper/review/s3_s5_finetune_v03_data_matched/gdn.json` | 14892 | `48bbf91e85e97e0e81ba84208e606a888811f550eb12e3f955078a2a61f93c1a` | TRACKED | Predecessor S3/S5 protocol-specific table source; not current-pair or historical 8 M data |
| `paper/review/s3_s5_finetune_v03_data_matched/m2rnn.json` | 14842 | `9d291d2af170e32f2a2226db3b42f8c55344deb46267a92e44eca87448cd6019` | TRACKED | Predecessor S3/S5 protocol-specific table source; not current-pair or historical 8 M data |
| `paper/review/s3_s5_finetune_v03_data_tocomp/e88.json` | 27723 | `39dcd9ad11965928115846df9f1e177e7ced67cbc5345cf7fd7d6fbedd60ba91` | TRACKED | Predecessor S3/S5 protocol-specific table source; not current-pair or historical 8 M data |
| `paper/review/s3_s5_finetune_v03_data_tocomp/gdn.json` | 28477 | `53d1d34fd362b9ed0348727db1cdb1af5d14d36f22836001f769aa26dd92a9a1` | TRACKED | Predecessor S3/S5 protocol-specific table source; not current-pair or historical 8 M data |
| `paper/review/s3_s5_finetune_v03_data_tocomp/m2rnn.json` | 57125 | `322640479b7a39921b34466e0b1acb636062140da92dc7c91af93e11c3f4054b` | TRACKED | Predecessor S3/S5 protocol-specific table source; not current-pair or historical 8 M data |
| `paper/review/S5_SYMMETRIC_BUDGET.md` | 15021 | `766cb4d67d93abcca41f33a80ae402481c3a89435d67332dd03444c9b2499188` | TRACKED | Predecessor symmetric-budget S5 table and protocol qualifications |
| `paper/review/s5_symmetric_data/e88.json` | 26534 | `f8fd46b220cd9c994c7d479e5a711133dacefbc99dcaf38691b1758dfd03d4e9` | TRACKED | Predecessor S3/S5 protocol-specific table source; not current-pair or historical 8 M data |
| `paper/review/s5_symmetric_data/gdn.json` | 26327 | `b853b1fb58be29d1cd514e15f6559df56f649b6c93ac4409b98b4c0ce1e04196` | TRACKED | Predecessor S3/S5 protocol-specific table source; not current-pair or historical 8 M data |
| `paper/review/s5_symmetric_data/m2rnn.json` | 26472 | `60a03677fda9303693f300bf4f570c27ac485f81a8558c110d905d1e1754e698` | TRACKED | Predecessor S3/S5 protocol-specific table source; not current-pair or historical 8 M data |
| `paper/tmlr/canonical-baseline.json` | 520 | `d2f44023859c43a65bec30d07fe796761a829a769f82a88e444e7c3a0b7fb17f` | TRACKED | Operator-directed P13 source digest / accepted-snapshot lineage |
| `paper/tmlr/validation/SHA256SUMS` | 1230 | `e650cec72d70469d493089738f21e9cd041fa72c544f717572e2d3cb0d1f9c9b` | TRACKED | Rebuilt PDF/bundle/style/source durability digests |
| `scripts/launch_emender_8gpu_diloco.sh` | 2821 | `40d60b3ad1e1f5348f300cd5917ca47f30653218c5066c468a7647e610cbb0ba` | TRACKED | E97 launch manifest defining common LM recipe |
| `scripts/launch_gdn2_mlp_8gpu_diloco.sh` | 2121 | `e54f8eafc8da1ffdc92ec086b1e7f409655f336608869a414efde38d1b1a8cff` | TRACKED | GDN2-MLP launch manifest defining common LM recipe |

## Untracked, scratch-only, or missing requested/source artifacts

These are findings, not durable inputs. External storage was not copied or
rehashed at checkpoint scale. Missing paths have no fabricated size/digest.

| Path | Bytes | SHA-256 | Status | Role |
|---|---:|---|---|---|
| `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606/args.json` | 3981 | `2ecf4d84f9657512102bec7b2031c01ce2e1548bc5e35bac974dcd644167a652` | SCRATCH-ONLY | External original run/probe metadata; not tracked on paper branch |
| `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606/run_manifest.json` | 652 | `a24cdf883db50c313d2ef49c2fbbd522a268138ebb917d52abe4f35165499b11` | SCRATCH-ONLY | External original run/probe metadata; not tracked on paper branch |
| `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/launch_manifest.json` | 1394 | `dcff77a69100ed484c39f3543f18697f44b7ba4f652face7949637f766e21ae1` | SCRATCH-ONLY | External original run/probe metadata; not tracked on paper branch |
| `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444/args.json` | 3910 | `601dc417846233d7c0942583ef614c55548cf750b2b3e0ed92e7ae0f665270b7` | SCRATCH-ONLY | External original run/probe metadata; not tracked on paper branch |
| `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444/run_manifest.json` | 550 | `c1db7ec816bdbf56256f57a43022deeecbcf8d2d0a9ab98e2908cebd2c56f252` | SCRATCH-ONLY | External original run/probe metadata; not tracked on paper branch |
| `/tmp/racer_generations_20260518/knowledge_probe_systematic_E88.json` | not available | `unavailable` | MISSING | External original run/probe metadata; not tracked on paper branch |
| `/tmp/racer_generations_20260518/knowledge_probe_systematic_FLA-GDN.json` | not available | `unavailable` | MISSING | External original run/probe metadata; not tracked on paper branch |
| `/tmp/racer_generations_20260518/knowledge_probe_systematic_M2RNN.json` | not available | `unavailable` | MISSING | External original run/probe metadata; not tracked on paper branch |
| `/tmp/racer_generations_20260518/knowledge_probe_systematic_Mamba2.json` | not available | `unavailable` | MISSING | External original run/probe metadata; not tracked on paper branch |
| `docs/repro/lb_emender_mlp_20260612/best.json` | not available | `unavailable` | MISSING | Requested E97-MLP best.json spelling is absent; use tracked RESULT/results.json best_params/best_loss |

## Durability limits / coverage boundaries

- The 1.3 B depth-wall sweeps, paired fine-tune records, throughput-derived
  budget, 8 M control aggregate and its 15 per-seed source records, S5
  capacity records (including both partial crashes), architecture args,
  trust inventory, historical evidence dossier, and all source CSV/JSON
  under paper/results are inventoried above.
- GDN2-MLP's `docs/repro/lb_gdn2_mlp_20260612/best.json` binds its search
  winner. E97-MLP has no corresponding tracked `best.json`: its carried-in
  winner is retained as `experiments/lb_emender_mlp_20260612/RESULT/results.json`
  (best_params/best_loss), alongside generations.jsonl. Other family's
  best.json files are context, not substitutes for either current-pair winner.
- Historical 8 M S3/S5 exact three-seed records remain unlocated; the rounded
  summaries in ndmpapernotes.md are the durable source. Do not substitute
  newer 1.3 B S5 records. Hybrid figure per-seed records are retained separately.
- Grok signature/confirmation aggregates and source tables are included.
  The larger per-run grok archive remains tracked under
  experiments/grok_symmetric_width/runs/ but is not expanded into this compact
  key-artifact inventory. Likewise unrelated ablation batteries are out of scope.
- The evidence DOSSIER is a historical snapshot (21 claims, 17 bound, four
  unbound), with old section identifiers and stronger claim wording. Hashing
  it does not endorse that framing or fill its missing bindings. The current
  manuscript's O1/G1–G3/U3.1 boundaries remain the claim authority.
- The final loss table currently relies on committed operational Markdown
  parsing receipts; a standalone current-pair aligned-loss CSV/JSON is not
  present in paper/results. Predecessor figure_2 CSVs are not substitutes.
- QA/reasoning summary CSVs are durable, but their original per-item
  eval.json/manifest.json trees are cited outside Git in
  paper/results/qa_reasoning/SOURCES.md (~/racer_eval_runs/). No claim is made
  that those full snapshots or the historical /tmp probes are durably tracked.
- The local .pt checkpoints and corpus slices are intentionally not small
  Git artifacts. Source checkpoint hashes are bound in the release receipts
  and manuscript identity table. Slice manifests retain byte counts/SHAs,
  not the underlying 10MB slice payloads. S3 paths alone are not Git tracking.
- Pre-existing untracked P4/P6/P7/P8/P10/P12 review files in this worktree
  are outside the result-input set and were left untouched.

## Independent verification

Parse the Path/Bytes/SHA-256/Status table rows, then for every TRACKED row:
(1) require exact path membership in git ls-files; (2) require file size;
(3) hash the file with sha256sum; (4) compare with git show HEAD:<path>.
Absolute SCRATCH-ONLY paths are optional external provenance, never required
for the standalone paper build. Missing entries remain explicit durability gaps.
