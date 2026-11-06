# Phase 1 launch-shape experiment decision

Warp16/32 candidates improved fixed-weight first-batch compute from 2.74447
steps/s to 4.18228/4.24765, but both failed strict fingerprints (80 differing
leaves including loss and gradients). Do not promote them. The 99-test scoped
C14 suite passes for warp16; tolerance parity does not override byte identity.

Next, test the previously designed safe reach bound with the original 8 warps:
from an empty store, at most two allocations per token implies physical indices
<2*T, so for T128 the cap257 backing can use a BS256 compute tile. Keep S257
in all policy/occupancy arithmetic. The unchanged kernel leaves physical tail
slot256 unwritten: initialize its state to exact empty physical bits in the
wrapper after the launch. The unchanged kernel's padded-lane empty-read POP
counter is observable in raw trace fingerprints: reconstruct that counter from
pre-read cardinality in the wrapper. No kernel body changes are proposed.

This is an experimental wrapper/configuration implementation, not pre-certified.
Keep old compiled reduction order as far as the original 8-warps launch permits;
byte fingerprints decide whether the smaller tile is numerically identical.
Test all physical state, raw traces, gradients and loss before any promotion.
If exactness fails, retain the failure and defer arithmetic-preserving tiling to
a full kernel-internal phase with the 109-oracle/graph/2M-token battery.
No matched-twin registration or training is changed by this experiment.
