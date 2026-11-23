# K15 bookmark execution ledger

0c8ca0a4 CPU pytest collection failed: repository pytest collects x45 as top-level; switch cross-package import to absolute. No card time.
76c8bef6 CPU pytest passed 1 test in 1.57s (native two-context append/locate/byte/retire and delayed gradient). Reserve attempt A: one card, 1700s timeout +15s kill grace, capped 0.5h.

Attempt A completed steps 1..4; owner 3462180 (PPID=1, starttime 550158662), GPU0 heartbeat 1795435598 -> 1795435606. Lease absent by 1795435620; conservative charge 22s = 0.0062h. Source bd0c0239. Artifacts /mnt/nvme2n1/erikg/k15_bookmarks_A; no continuing run.
```
gpu_lease: granted GPUs: 0 (pid 3462180)
{"anchor": {"base": "k15", "seed": 3401, "anchor": 1.3811104467936925, "state": "/mnt/nvme2n1/erikg/xsa_anchor_replays_v1/k15/final.pt", "sha256": "accb73ce3ad0cbe5b88d439dad8073d17a0ab3ec25a9f3f52b6cfdf7f968ecd0", "engine_sha256": "e769a26157ca3da36569258289476f0c1e74e9b2e33e3d187e8b185076349d2c"}, "identity": {"stream": "flagship_beyond_100gb", "source": "/mnt/nvme2n1/erikg/pile.txt", "source_size": 1307910802540, "manifest": "/home/erikg/emender/experiments/expressivity_tasks/x43_fork_resolved/receipts/flagship_stream/documents.jsonl.gz", "manifest_sha256": "48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414", "registration": "/home/erikg/emender/experiments/expressivity_tasks/x43_fork_resolved/receipts/flagship_stream/registration.json", "registration_sha256": "c36f84cfd1a59530a7991a4be6c1f68b424a79c5b7de169eabfc90d3cd31cc31", "boundary_bytes": 100000000000, "lattice_registration": "/home/erikg/emender/experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_150M_LATTICE.md", "lattice_sha256": "99bed43706538c553ac57b279b1ce28fa78dfefe6a4808dbcedd973af6e1797c"}, "exe": "87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f", "params": 158726502}
{"anchor_evaluation": {"score": 1.3811104467936925, "nll_heldout": 1.3811104467936925, "eval_split": "calibration", "eval_windows": 7, "eval_targets": 14336, "units": "nats/byte"}}
{"step": 1, "doc": 100000000071, "start": 0, "bytes_sha": "1ad1b22f9162aa8088b6f06942a876145c2b85b1ed4394bb911662f0c848f688", "forced": true, "delayed_gradient": 0.001220703125, "norm": 0.38352295756340027, "nll": 2.0011203289031982, "calls": [[0, 1, 2], [1, 0, 2], [0, 2, 3], [1, 2, 3], [0, 3, 0], [0, 1, 2], [1, 2, 3]], "seconds": 0.5089743183925748, "tokens_per_second": 251.48616614733177}
{"step": 2, "doc": 100000000524, "start": 0, "bytes_sha": "5e84cbc087aed301a1c00a76197088797cafd50320f1ff8c246e1acc56fec2ac", "forced": false, "delayed_gradient": null, "norm": 63.554447174072266, "nll": 1.8212409019470215, "calls": [[1, 1, 2], [1, 1, 2], [1, 2, 3], [0, 1, 2], [1, 0, 2], [1, 1, 2], [1, 3, 0]], "seconds": 0.2707315478473902, "tokens_per_second": 472.79307128311797}
{"step": 3, "doc": 100000002302, "start": 0, "bytes_sha": "a5ec7e42567b89e26fa73b207fb667aa06794be3775b1964d88d4846bf829cea", "forced": false, "delayed_gradient": null, "norm": 52.74989318847656, "nll": 2.216906785964966, "calls": [[0, 1, 2], [0, 0, 2], [0, 2, 3], [1, 0, 2], [1, 0, 2], [1, 2, 3], [1, 2, 3]], "seconds": 0.21792434249073267, "tokens_per_second": 587.3598081657319}
{"step": 4, "doc": 100000007119, "start": 0, "bytes_sha": "ff4600a937df3054f6bcec2632f37ac7c5756a7537b6b11cba247eee04954d0e", "forced": false, "delayed_gradient": null, "norm": 23.642343521118164, "nll": 1.9217945337295532, "calls": [[0, 0, 2], [0, 1, 2], [1, 0, 2], [0, 3, 0], [1, 1, 2], [0, 0, 2], [1, 2, 3]], "seconds": 0.19935828540474176, "tokens_per_second": 642.0600966753474}
{"checkpoint": "746ba8513b7c9cd2559c74112ed0e9835b512a8c90b03a3046ecd1cfe0dc2f3a"}
94e51d5ca3d40299bdface4b78816e672df78c5150be3ac1489ac06afbd65580  /mnt/nvme2n1/erikg/k15_bookmarks_A/launch.json
516dc2c4952f96e921c0e2dfcc32b6ea710968dd6f101a13fd27968e5a1018b4  /mnt/nvme2n1/erikg/k15_bookmarks_A/owner_check
f327bce5334b4cd5c22f3d4524f87f3f9cccc641613506701f1b166dcf84e4b3  /mnt/nvme2n1/erikg/k15_bookmarks_A/console
42c489e40e3c5b323768707d09f3ec6dd49c1764124d7869eba127ed86e8e814  /mnt/nvme2n1/erikg/k15_bookmarks_A/forced_trace.json
```
fc01aed0 CPU pytest 1 passed in 1.74s with shared composed mixer geometry. Attempt B uses fresh directory, new source, same <=0.5h timeout.

Attempt B f2cf39aa: anchor, four full steps, delayed gradient and exact chosen native action/reply prefix replay passed. Variable-shape end-logit parity FAILED at prefix 33 (max 0.17676 > 0.0625); matched forward timer not reached. PPID=1 GPU0 owner 3466029/starttime 550173030, renewed 1795435742 -> 1795435744; lease absent at 1795435768. Charge <=26s. Do not relax tolerance.
```
gpu_lease: granted GPUs: 0 (pid 3466029)
{"anchor": {"base": "k15", "seed": 3401, "anchor": 1.3811104467936925, "state": "/mnt/nvme2n1/erikg/xsa_anchor_replays_v1/k15/final.pt", "sha256": "accb73ce3ad0cbe5b88d439dad8073d17a0ab3ec25a9f3f52b6cfdf7f968ecd0", "engine_sha256": "e769a26157ca3da36569258289476f0c1e74e9b2e33e3d187e8b185076349d2c"}, "identity": {"stream": "flagship_beyond_100gb", "source": "/mnt/nvme2n1/erikg/pile.txt", "source_size": 1307910802540, "manifest": "/home/erikg/emender/experiments/expressivity_tasks/x43_fork_resolved/receipts/flagship_stream/documents.jsonl.gz", "manifest_sha256": "48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414", "registration": "/home/erikg/emender/experiments/expressivity_tasks/x43_fork_resolved/receipts/flagship_stream/registration.json", "registration_sha256": "c36f84cfd1a59530a7991a4be6c1f68b424a79c5b7de169eabfc90d3cd31cc31", "boundary_bytes": 100000000000, "lattice_registration": "/home/erikg/emender/experiments/expressivity_tasks/x43_fork_resolved/PREREGISTRATION_150M_LATTICE.md", "lattice_sha256": "99bed43706538c553ac57b279b1ce28fa78dfefe6a4808dbcedd973af6e1797c"}, "exe": "87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f", "params": 158790278}
{"anchor_evaluation": {"score": 1.3811104467936925, "nll_heldout": 1.3811104467936925, "eval_split": "calibration", "eval_windows": 7, "eval_targets": 14336, "units": "nats/byte"}}
{"step": 1, "doc": 100000000071, "start": 0, "bytes_sha": "1ad1b22f9162aa8088b6f06942a876145c2b85b1ed4394bb911662f0c848f688", "forced": true, "delayed_gradient": 0.0011940469266846776, "norm": 0.23711992800235748, "nll": 1.9876651763916016, "calls": [[0, 1, 2], [1, 0, 2], [0, 2, 3], [1, 2, 3], [0, 3, 0], [0, 1, 2], [1, 2, 3]], "seconds": 0.49880014453083277, "tokens_per_second": 256.6158037512113}
{"step": 2, "doc": 100000000524, "start": 0, "bytes_sha": "5e84cbc087aed301a1c00a76197088797cafd50320f1ff8c246e1acc56fec2ac", "forced": false, "delayed_gradient": null, "norm": 14.14575481414795, "nll": 1.8186670541763306, "calls": [[0, 1, 2], [0, 0, 2], [0, 0, 2], [0, 1, 2], [0, 0, 2], [0, 1, 2], [1, 1, 2]], "seconds": 0.2567654363811016, "tokens_per_second": 498.5094637504763}
{"step": 3, "doc": 100000002302, "start": 0, "bytes_sha": "a5ec7e42567b89e26fa73b207fb667aa06794be3775b1964d88d4846bf829cea", "forced": false, "delayed_gradient": null, "norm": 22.28959083557129, "nll": 2.215632438659668, "calls": [[0, 1, 2], [0, 2, 3], [0, 2, 3], [0, 3, 0], [1, 0, 2], [0, 1, 2], [0, 2, 3]], "seconds": 0.20859395805746317, "tokens_per_second": 613.6323467467774}
{"step": 4, "doc": 100000007119, "start": 0, "bytes_sha": "ff4600a937df3054f6bcec2632f37ac7c5756a7537b6b11cba247eee04954d0e", "forced": false, "delayed_gradient": null, "norm": 9.168537139892578, "nll": 1.9225044250488281, "calls": [[1, 1, 2], [0, 1, 2], [0, 0, 2], [1, 3, 0], [1, 0, 2], [0, 3, 0], [0, 1, 2]], "seconds": 0.2152174385264516, "tokens_per_second": 594.7473442504892}
{"checkpoint": "0b4eeb8208b2a44f366d935a3bbaa702eed13e980258f545973b77a56c2a76f5"}
{"prefix_replay_max_abs": [0.046875, 0.1767578125, 0.015625, 0.0625, 0.0625, 0.0, 0.0], "greedy_trace": [[1, 0], [1, 2], [1, 0], [0, 1], [1, 0], [0, 2], [0, 3]]}
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/erikg/emender/experiments/expressivity_tasks/x45_matrix_emender/k15_bookmark_probe.py", line 102, in <module>
    main(Path(sys.argv[1]))
  File "/home/erikg/emender/experiments/expressivity_tasks/x45_matrix_emender/k15_bookmark_probe.py", line 92, in main
    assert max(differences) <= 0.0625  # one bf16 ulp around logit magnitude 8
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
86ec39b46ec18260813322b48d66e12fb640f9e6366e6844e4833f94af176e2b  /mnt/nvme2n1/erikg/k15_bookmarks_B/console
42c489e40e3c5b323768707d09f3ec6dd49c1764124d7869eba127ed86e8e814  /mnt/nvme2n1/erikg/k15_bookmarks_B/forced_trace.json
7d7ebc1bffe9dcb602170900567baa64dd8550ecc7cea8f7d6276be51e54615f  /mnt/nvme2n1/erikg/k15_bookmarks_B/launch.json
428349af37061e41d8aac5df5284f17d8213d3f6bd5a275cea1c5d3a0e6c0b88  /mnt/nvme2n1/erikg/k15_bookmarks_B/owner_check
```

C (5d32c41a): reloaded B checkpoint, greedy chosen-only greedy RPCs identical on all seven prefixes. Padded: all logits on all visible positions exactly equal. Unpadded variable-shape differences also present with doors off on same updated backbone (max 0.34375), so no claim of variable-shape logit equivalence. Warmed B=1 T=128 throughput integrated 1920.98 vs off 2294.68 input bytes/s; not incremental decode. GPU0 PPID=1 owner 3468731 starttime 550180937 renewed 1795435821 -> 1795435823, lease absent 1795435838. Charge <=17s. Total shared conservative charge A+B+C = 65s (0.01806h); no running allocation.
```
gpu_lease: granted GPUs: 0 (pid 3468731)
{"doors": false, "tokens_s": 2294.683794948707}
{"doors": false, "end": 17, "prefix_max": [0.0625, 0.0]}
{"doors": false, "end": 33, "prefix_max": [0.203125, 0.0]}
{"doors": false, "end": 49, "prefix_max": [0.203125, 0.0]}
{"doors": false, "end": 65, "prefix_max": [0.34375, 0.0]}
{"doors": false, "end": 81, "prefix_max": [0.310546875, 0.0]}
{"doors": false, "end": 97, "prefix_max": [0.0, 0.0]}
{"doors": false, "end": 113, "prefix_max": [0.0, 0.0]}
{"doors": true, "tokens_s": 1920.975126378167}
{"doors": true, "end": 17, "prefix_max": [0.0625, 0.0]}
{"doors": true, "end": 33, "prefix_max": [0.1767578125, 0.0]}
{"doors": true, "end": 49, "prefix_max": [0.1767578125, 0.0]}
{"doors": true, "end": 65, "prefix_max": [0.28125, 0.0]}
{"doors": true, "end": 81, "prefix_max": [0.1748046875, 0.0]}
{"doors": true, "end": 97, "prefix_max": [0.0, 0.0]}
{"doors": true, "end": 113, "prefix_max": [0.0, 0.0]}
87e739b0565c6596e01a5d3dc54884e2a97596fb00b56234db6711578568f361  /mnt/nvme2n1/erikg/k15_bookmarks_C/console
97dde8b2a93480a5acbe8e23852eeb2dde8eb551e242b92c6360dc63ad37e701  /mnt/nvme2n1/erikg/k15_bookmarks_C/launch.json
60bf365ab807a13adf29530429b54151e7d2d67f9df8b68c34ac4beb16c1c77a  /mnt/nvme2n1/erikg/k15_bookmarks_C/owner_check
```
