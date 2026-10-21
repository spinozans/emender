# X15b amendment-2 checkpoint restart — launch report

Admission785dd40c committed/pushed before training. Failed replay admission
974320fc remains INFRA-INTERRUPTED/FAIL; interruption evidence6bbab420 preserved.
Supervisor explicitly authorized this separate checkpoint branch, not a tolerance
or automatic retry. Original433000 checkpoint hash unchanged; model, complete
Muon+AdamW optimizer and numpy/torch/CUDA RNG restored and compared exactly.
checkpoint_restored.json explicitly says at_death_restoration_claimed=false.

567,000 fresh updates numbered433001–1000000; original1M cosine and1M ceiling.
Historical433001–433296 superseded, not claimed bitwise reproduced. New stream
has explicit433000 boundary event and approved provenance on every update.
Concession #9 (task's incorrect500K endpoint) remains recorded.

Physical card7 GPU-8f63894f-5b64-755c-060f-7022c603e7d2 acquired by gpu_lease;
worker1332577, trainer1341549. At launch verification:1,855 consecutive finite
fresh updates through434855; all LRs equal original cosine exactly, every row
carries branch provenance. Validation consecutive500-step series through434500.
New checkpoint434500 SHAe571e3710e97d2e7f66b9ae4aa13768198945cfe3aa3d2e9619b84eabff4f827
verified with optimizer+RNG state; original checkpoint retained. Root free
1,652,081,332,224 bytes, above fail-closed200GiB guard. Launch verification is a
snapshot, not terminal completion evidence. GPU snapshot:1,118MiB/35% utilization.

Local detached proc watcher1464262 running. Supervisor explicitly confirmed
**native arming**: process proc_41b4, PID1618938, first poll
2026-10-21T21:26:02Z RUNNING muon3x_seed0. Supervisor says its exit wakes the
attended session natively. Command from /home/erikg/emender:
`.venv/bin/python -m scripts.watch_invertible_stage15_muon_restart --interval 900`.
One-shot watcher returned2 pending; GDN2 amendment DONE. Old replay watcher
was stopped with identity verification. No WorksGood completion machinery.

Validation:28 tests passed/1 opt-in GPU test skipped; actual card7 restoration,
new checkpoint roundtrip state inspection, live finite/consecutive/LR/provenance
checks and hash-bound committed registration checks passed. Scoped staged
whitespace check passed. Standalone expressivity: Frontier/resilient-DiLoCo
requirement IDs not applicable. Existing unrelated dirty files left untouched.

X15G-growth-report-addendum.md is an interim censored snapshot only. On native
wake, watcher writes full all64 count/CE and exact-marginal trajectory with the
registered any-position>=50/512 escape verdict and fixed-final retention.
1M terminal verdict remains pending; no claim of escape or exact-marginal
optimizer-robust failure is made now. Supervisor will launch independent review
on completion; that acceptance gate remains outstanding.
