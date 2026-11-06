# Round8 compiler escape probe

Read round7 design/result and its three terminal receipts first. Choose the
explicitly permitted two-kernel Triton fallback as the smallest experiment:
pre-control, score, unchanged normalization and ordered READ in the first
kernel; post-control and persisted metadata in the second. This isolates the
phase boundary at the already persisted score/product/scalar/parity carries.
At T1024 the intended count is 2049 versus 6145 (not yet measured).

This first probe retains global-scratch value updates. It is scaffolding,
not the requested shared-page implementation and cannot qualify for GO.
CUDA C++ remains the named alternative if pre-control plus READ still exceeds
compiler tolerance; its cost includes reproducing the exact fixed reduction
tree, transcendental lowering and signed-zero arithmetic, not merely porting
control syntax. nvcc exists at /usr/local/cuda/bin/nvcc.

Do not time or scale a failed compilation. First compile a nonzero B1/T32 pair
under the physical4 exclusive lease; then exactness gates before timing.
No production or registration mutation. No inference of measured launches or
physical traffic from this source schedule. The inherited global-scratch
traffic subtotal remains round7's 4,347,396,096 bytes at T1024/BS2048;
shared-page target accounting remains unimplemented.

Tool discovery returned no contact_supervisor capability. No supervisor
message or approval is claimed. Work stays serial and attended.
