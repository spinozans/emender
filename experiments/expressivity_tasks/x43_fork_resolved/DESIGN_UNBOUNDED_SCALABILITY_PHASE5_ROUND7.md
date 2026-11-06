# Round7 fused op-cycle design — before implementation

Diagnostic candidate only; production and 1.3B registration remain unchanged.
Hardware-counter permission is explicitly waived by the round7 work order;
source traffic is analytic, never reported as physical DRAM traffic.

## Dependency and launch plan

One CTA owns one independent batch/head, across the complete context. Keep
initialization as a separate launch, then one persistent op-cycle launch:
expected two launches versus 6145 at T1024/BS2048. No cross-CTA synchronization
is required. Control metadata persists in the physical final arrays between
phases. A CTA barrier is mandatory after each producer before any consumer that
changes its layout. No state crosses a native invocation or document reset.

The logical page is 128 slots (64 FP32 keys and 128 FP32 values per slot:
98,304 bytes combined). Keys and values must be staged separately to bound
shared capacity: 32,768 and 65,536 bytes respectively. The intended final
implementation reuses this page allocation across slots and context positions.
The first implementation milestone fuses the existing global-scratch phases;
it is explicitly scaffolding, not evidence that shared-page staging exists.
Reject that milestone for production regardless of its launch count.

For each valid position, run CORRECT, expiration, array APPEND/duplicate/cap
MERGE; compute scores in ascending pages; normalize with the unchanged
full-slot reduction tree; READ with the unchanged two parity chains; POP;
stream APPEND/CORRECT/cap MERGE; DECAY and death; write all trace fields.
Do not reorder control around READ, reassociate normalization, or retain stale
values across CORRECT/POP/MERGE. Masked positions write zero output and retain
state. The exact existing arithmetic (including signed-zero additions) is binding.

## Carry and persistence

Dedicated per-call scratch: scores and normalized strength products, each
B*BS FP32; eight int32 scalars per head (including bitcast maximum); two ordered
FP32 value chains of 128 lanes (1024 bytes/head). The even and odd chains must
continue across page boundaries without tree reassociation. Live mask, strength,
birth, order, owner, prediction, prediction time and expiry retain the existing
seven-field physical layout. Dirty keys/values and all metadata must persist
before the next position, including mid-context POP and slot reuse.

## Gates and accounting

First compile and run one small nonzero parity probe under an exclusive broker
lease on an idle permitted physical card. A compiler failure or bit mismatch
is a retained failure. Then compositional-carry fixtures, ten native shape pairs,
T512/T1024/T2048 masked/reset fixtures, 109 X31 and 108 operator tests. Only
an exact candidate is timed on all ten shapes. Count actual launches, not just
Python dispatches. Report B1 stress and B16 separately; no extrapolation of B1
as the batched training number.

Analytic full staging payload per position is live key/value loads plus dirty
persistence, query/control inputs, output, and carry/scratch traffic. Enumerate
all terms per implemented pass. Compare with the occupancy-conditional
180,075,776-byte/head T1024 receipt, preserving its cache-tier caveat. Shared
staging alone does not remove recurrent reads or imply lower physical DRAM.

GO requires all standing exactness and speed gates plus the standing model-level
training gates. Launch reduction alone is insufficient. Unrun gates stay unrun.
