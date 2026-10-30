# Source inspection (before GPU conclusions)

`kernels.py:287-292` writes History via journal_store and immediately reloads
128 history lanes, then reduces requests and caps. It has no barrier between
that store and reload. This is the exact operation and missing barrier location
in part4/pressure_barrier_diagnostic.patch, inherited from X31 kernels.py:189-194.
It shares scan_kernel with the r_live sum and lazy-decay helper sums named in
part 15. The nearest preceding unconditional barrier is at kernels.py:270,
before the history store; it cannot order that later store against its reload.
This identifies a shared ordering-risk site, not proof that it causes the
int32-add assert. The assertion source is still a generated shared combiner.

X40_STABLE_MAPPING is consulted only by backward.py:128 in rebase_owned.
Its alternate mapping_runtime uses byte pointers/runtime sizes to avoid
specialization on allocator address order. model.scan under torch.no_grad()
directly uses wired_forward, bypassing autograd_scan and rebase_owned.
Consequently this flag has no direct control over the failing inference
scan. Its historical presence establishes prior sensitivity in another path,
not an established forward-kernel address-order defect. The layout experiment
records both flag settings rather than inferring equivalence from the name.

forward.py:45-46 explicitly rejects experimental batched nonrecording reads
after a checkpoint heap assertion. All experiments retain X40_BATCHED_READS=0.
No production source has been edited, and no training or search was launched.
