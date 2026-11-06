# Round8 revised portable fusion design

Supersedes the earlier compiler-escape brief: no driver changes and no new
CUDA C++ kernels. Existing certified X31 CUDA remains untouched. Round6's
180,075,776 bytes/head conditional analytic estimate remains the working model;
no physical counter result is claimed. Read round7 a346ffe9, efe195a2,
680068c5 and round6 component_fixed.json before this revision.

First measure identical warmed R5 forward store invocations eagerly and through
Torch graph replay. Round6's 6145 launches were explicitly an eager,
CUPTI-instrumented forward store invocation, not the already captured backward.
Capture initialization and scratch inside the graph, compare all six output
buffers bytewise, alternate seven eager/replay timing samples, then separately
profile actual device kernels and host kernel/graph launches. Measure B1 and
B16/T1024. These are preprojected forward store times, not training step times.
Do not subtract instrumented kernel sum from uninstrumented wall time to claim
an exact overhead decomposition.

Primary fusion path remains split Triton/Gluon. The inherited two-kernel probe
(pre-control + score + normalize + ordered READ; post-control) failed LLVM.
Next isolate control from the fused READ: three kernels per position,
pre-control; score + normalize + ordered READ; post-control, plus initialization.
Keep the unchanged arithmetic helpers, reductions, masked behavior and two
parity chains. First compile nonzero B1/T32 under an exclusive physical4 lease.
A failed compiler probe is a failure, never a launch-count or speed result.

Only a successfully compiled byte-exact candidate advances to compositional
fixtures, ten pairs, T512/T1024/T2048 masked resets, 109 X31 and 108 operator
tests, then timing. Intended three-kernel count is 3*T+1, not measured until
execution. Global-scratch scaffolding still has round7's incomplete analytic
4,347,396,096-byte subtotal at T1024; no shared staging claim.

Portability evidence must distinguish source API from execution: new code is
Torch and Triton/Gluon only, with no CUDA-C++ additions. NVIDIA execution alone
does not certify AMD/ROCm or other vendors. torch.cuda graph API is backend
specific; cross-vendor graph support and Gluon layouts require actual testing.
No driver or protected storage changes, production promotion or training.

Tool discovery returned no contact_supervisor capability; no contact claimed.
