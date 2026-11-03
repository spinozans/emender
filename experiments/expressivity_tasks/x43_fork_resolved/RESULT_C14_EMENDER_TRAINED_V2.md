# C14 consultation v2 feature pathway

The 304-batch CPU profile measured 8.721s sparse dictionary/entropy decoding,
6.927s Python feature assembly, and 0.016s reads. Native ordered-double conversion
took 0.162s. All 4,864 windows (including padding) were float32 bit-identical,
with shard SHA256 and dense output SHA256 receipts in feature_profile.json.
The registered dropout was already vectorized; it was not the bottleneck.

The 304-step GPU profile measured 0.303s native read/decode, 0.114s sampling and
transfer, 0.249s feature transfer/dropout, and 12.312s training. Including startup,
it achieved 23.415 steps/s; steps 100–300 achieved 32.996 steps/s, within 1.26x
of frozen dense 41.3. The oracle profile had a substantially larger first-100
training cost; do not attribute that difference to feature decoding. Reported
losses at steps 1/100/200/300, target counts and dropout keep fractions matched
exactly. The oracle instrumentation note records a source-hash limitation.

Native conversion holds one 17,039,360-byte dense shard in memory; it writes no
dense corpus copy. Native library/checkpoints stay outside git. The disk guard
fails closed below 150 GB free. Each consumed shard is rehashed, and every
window is identity/length matched against frozen raw-byte document offsets.
The decoder rejects unsupported counts beyond exact double-integer support.

The v2 primary-arm amendment is registered in paired_arm_amendment.json before
training. V1 and dense evidence remain frozen. Paired quality evaluation is
pending. The registered 2700-active-second v2 training is pending at this commit.

The transformer exposure is 4.305x the Emender target count. Its future training
must bind the same shared decoder for prefix and continuation shards. This is a
committed launch requirement, not evidence of a transformer run or of an existing
transformer training runner. The union service and ~/sxgc were not changed.

Twenty selected boundary/sparse/batching/decoder tests pass. No resilient DiLoCo
behavior was changed, tested or run; gap-matrix requirements are not applicable.
