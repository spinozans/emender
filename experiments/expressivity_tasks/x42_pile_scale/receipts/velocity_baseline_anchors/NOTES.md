# Queue and validation notes

These anchors use the current committed velocity_regime.json B64/T2048 regime,
not the superseded B1/T512 prose in VELOCITY_PROTOCOL.md. The search harness
and all registered compute sources remain unchanged. The baseline update loop
is AST-compared with velocity_train.train in tests; only the detached emender
gate offset is removed. Data window helper, synchronized timing, whole-update
overshoot, optimizer, seed, and evaluation window starts are shared/matched.
LR 0.00025180085693680385 and warmup 122 are fixed to c06 for both arms.

Part8's builders assume vocabulary 32768. Constructor-only module patches
supply the velocity GPT-2 vocabulary 50257; no certified source is edited.
Capacity matches c06's 150003984 trainable parameters within 0.1%, rather than
its total 227317104 including 77313120 frozen parameters. This is a trainable
capacity anchor, and frozen-parameter asymmetry remains a comparison limit.

The existing c06 full-budget receipt supplies the emender row. Its exact score
5.3733475944725795 differs materially from raw NLL 5.314661979675293: consult
c06_result.json for occupancy and degeneracy components. It is not a fresh
rerun. Card 0 runs transformer then GDN2; no card fallback is permitted.
The launcher waits indefinitely through the broker under systemd supervision.
Runtime source hashes fail closed if queued code changes. Failure receipts
and a nonzero service exit preserve failures; a failed transformer does not
prevent the independent GDN2 attempt. No checkpoints or .pt files are used.

Residual risks: no completed baseline GPU measurements at delivery; full-batch
GPU memory/kernel feasibility is unproven until execution. OOM fails the arm
without silently changing the B64 regime. BF16 fused baseline CE differs in
numerics from the emender FP32 ChunkedLoss, as in the part8 best-practice
baseline surface. No claim about the strength of 5.37 is justified yet.
Systemd user service survives this session but is not configured for reboot.
Runtime receipts update locally after launch; later measurements are not
part of the initial pushed snapshot. No resilient DiLoCo behavior is changed;
RESILIENT_DILOCO_GAP_MATRIX requirement IDs are not applicable.

c06 score components: {"nll_heldout": 5.314661979675293, "occ_frac": 0.8685614797286689, "occupancy_penalty": 0.00868561479728669, "degeneracy_flags": {"dead_store": false, "pressure_runaway": true, "slope_one": false}, "degeneracy_penalty": 0.05, "train_seconds": 2701.2769256266765}
