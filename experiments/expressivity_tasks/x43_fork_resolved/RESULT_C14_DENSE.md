# C14 dense control — failed source preflight; no training

The operator authorized the dense arm independently of the blocked consultation
bridge. The exact raw shard and document-index hashes in c14_byte_source.json
were checked before an exhaustive, source-ordered reconstruction attempt.

**Failure:** the bound 00.jsonl.zst source exhausted after 7,021,438 raw rows.
Only 5,800,191 of the 7,831,213 selected document receipts matched, materializing
32,199,938,547 natural bytes. The next required receipt is source_index 5899215,
SHA256 db821729e50019cc6294aa9876814a540b18a17123ef47f9b0d185ebb8ec96f5.
2,031,022 selected receipts remain unmatched in this ordered reconstruction.
No token-truncated document was reached/admitted. The explicit complete-corpus
prerequisite failed independently of xsa. The prefix is not a registered dense
training corpus and was not substituted for one.

Receipt: receipts/c14_dense/source_preflight.json; full progress log:
receipts/c14_dense_source_console.txt. The partial byte corpus and offsets remain
outside git at /mnt/nvme2n1/erikg/c14_dense_bytes/*.partial* for diagnosis only.
No .pt file was produced, no GPU allocated for C14, and no training curve or
held-out NLL exists. No support/fork strata or shared completed eval set is claimed.
The paired evaluation construction and dropout-free dense recipe are committed
in PREREGISTRATION_C14_DENSE.md before model measurements; the gated runner
c14_dense_train.py remains unexecuted.

Residual risk: completing C14 requires an explicitly bound extension of the raw
source set or a separately registered corpus scope. Other shards are present,
but using them would violate this task's instruction to use exactly the source
bound by c14_byte_source.json. No source substitution or xsa query was attempted.
The previously registered P1/P2/P3 training predictions remain untested.

Manual note: contact_supervisor is unavailable (tool discovery by name and
supervisor/intercom description returned no matches), so no supervisor decision
could be requested about the newly demonstrated raw-source insufficiency.
