# X43 C14-B1 enabled integration

The enabled store door is implemented independently of the frozen C14 arms.
`c14_b_model.StoreIntegrated` retains exactly 2,133,700 trainable parameters
and their names/shapes. Answers never enter the embedding hook. Their complete
260-scalar contents pass through the standing `device_input` and `W_v` maps.
Each head shares its 32/64 physical slots between stream and array ownership.
Array APPEND reuses the actual detached native stream key at that position.
Full suffix bytes remain metadata; they never enter numeric addressing.

The B-only Triton forward implements shared softmax/strength reads, ordinary
stream POP/write/capacity routing, metadata duplicate consolidation, delayed
observed-byte residual correction and stale ownership retirement. A correction
means the originating array's smallest-index argmax byte differed from the
subsequently observed input byte. It cannot change the prediction already made.
A capacity delta replaces the value and transfers ownership to the last writer,
while preserving the destination key and strength/birth law. Native event traces
report operations and actual nonzero effective array read weights, with explicit
slot-opportunity denominators. A hit means nonzero float32 contribution weight,
not top-1 retrieval or demonstrated causal benefit.

Autograd reconstructs the complete independent-window state with immutable
value/strength versions and differentiates that replay; reentrant block
checkpointing also replays native forward. This deliberately does not claim to
reuse the frozen X40 all-field undo/reverse kernel. There is no cross-window
continuation to roll back: keys, values, strength, birth/order and suffix owner,
prediction and expiry metadata are reconstructed from the same empty reset.
Every backward checks replay/native read agreement and fails closed on a
mismatch. The replay cost is included in measured training time.

Launch evidence is in `receipts/c14_b/launch_gates.json` and
`receipts/c14_b/launch_battery.txt`: 65 tests passed, including real leased GPU
native/eager operation and gradient checks under 1/32/64-slot budgets, stream-only
parity with the standing native scan, force-disabled exact forward, enabled
checkpoint gradient equality, and BF16 batch-16/context-128 finite backward.
First/middle/last registered training shards passed complete float32 feature-bit
and raw metadata identity against the original decoder. The unchanged frozen
registration audit passed. Source/cache hashes are checked again before training
and every consumed raw source document is matched against its corpus receipt.

An initial GPU test invocation had two test-import failures; the corrected run
and the full battery passed. The failure log is retained. No failed numerical
native or gradient result has been omitted.

Training and PB1–PB4 results are pending. Passing launch gates alone makes no
scientific claim about NLL, management usefulness or throughput. Conditional
ablations remain dependent on the measured PB1 delta. Resilient DiLoCo was not
changed or tested; gap-matrix requirement IDs are inapplicable.

The runtime exposes no Intercom/contact_supervisor tool (searched tool names
and descriptions for contact_supervisor, supervisor and Intercom). No contact,
reply or approval is claimed. Work is attended and single-threaded, without wg
services or workers. No frozen receipts, P3 run, union service, source mount or
committed checkpoint is changed by the B implementation.
