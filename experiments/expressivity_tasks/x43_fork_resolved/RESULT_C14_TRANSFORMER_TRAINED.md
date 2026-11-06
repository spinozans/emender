# C14 transformer consultation arm trained

The registered 2700-active-second training is complete. Paired evaluation and the head-asymmetry verdict are pending; no quality result is claimed in this phase.

The apparent short cache was a continuation-only console counter: 225,880,702 shared Emender-prefix targets + 746,495,031 continuation targets = **972,375,733**; 1,785,536 + 5,900,112 = **7,685,648 windows**. Exhaustive reconciliation passed every shard hash, source-order window identity, window/target count and float32 entropy parity check. Both journal hashes and cache sizes match their union phase-boundary receipts, including continuation SHA256 `0e7cefde3246a3b8524f5013fcb12414af694e5b5c431db20b044c354ed6edc3`. Total shared-plus-continuation cache: 100,137,603,932 bytes. The service was neither restarted nor modified.

The first reconciliation attempt rejected the continuation's current audit hash because it required the shared prefix's historical hash. The retained failed attempt was followed by an explicit check that both audit versions have identical frozen stream fields; the exhaustive rerun passed. This was a receipt-version check, not missing cache data. No training preceded the passing receipt.

| Measured item | Value |
| --- | ---: |
| Steps | 281,741 |
| Independent windows | 4,507,856 |
| Real byte targets | 570,344,976 |
| Active seconds | 2700.010380685 |
| Training wall seconds | 2707.113914 |
| Checkpoint save/hash stalls excluded from active time | 1.995516 |
| Keep fraction on real targets | 0.499944539 |

Seed 3401, batch 16, context 128, BF16, AdamW lr 3e-4/decay .01/clip 1; 2,133,748 registered transformer parameters. Natural raw byte inputs and shifted targets remain in one hash-checked source document. Fresh state per window; truncated tails are padded and loss-masked. Source order starts at window zero, shared prefix first and continuation second. Independent keep=.5 per position zeros all 260 consultation inputs jointly, without rescaling. Frozen dense controls, existing arms and phase 1–3 kernel receipts remain unchanged; their evidence hashes validate. The registered transformer attention head uses the shared v2 fast feature decoder; Emender-specific store kernels do not execute in this head.

Broker card 2; detached Popen/start_new_session launcher with DEVNULL stdin and retained logs. The PPID=1 owner, matching lease starttime, heartbeat renewal and advancing steps are receipted in `receipts/c14_transformer/training_survival.json`. Final atomic checkpoint outside git: `/home/erikg/c14_dense_bytes/transformer_consultation_v2_resume.pt`, SHA256 `2720a3cd8dc8a8be90b14c710a2f0484c992827615f5f826150a2373dfd10272`. Model/optimizer tensors are finite, parameter shapes load strictly, and RNG/cursor/step/target/active-time resume state is retained. Source-order target accounting recomputes exactly. Twelve document-boundary/feature-decoder tests pass.

Limitations: one seed and fixed active time produce unequal exposure; this phase establishes training, not paired quality or a head-asymmetry verdict. Native-query parity reuses extraction attestations; no fresh extractor rerun is claimed. Cache extraction and lease wait are excluded from active training and separately receipted. No resilient DiLoCo behavior changed or ran; gap-matrix requirement IDs are inapplicable.
