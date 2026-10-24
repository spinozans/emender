# X26b → X26c candidate divergence ledger

Only the complete bundle is identified by this control; no factorial attribution.

|Seam|X26b reconciled native-task control|X26c shipped raw pipeline|
|---|---|---|
|Training implementation|our runner calling native components|their complete unmodified bash launcher + trainer main/run_epoch|
|Batch|128 exact; carry epoch tails|256; native per-bucket partial tails, drop_last=False|
|LR|constant1e-4|3e-4, warmup1000, cosine to10% by30k|
|Weight decay|1e-4|GRU .01; Delta .005|
|Budget|60k|30k OR500epochs OR patience100/60|
|Validation|update0/every500|every2epochs/cap/final epoch; no update0|
|Checkpoint|pooled ID accuracy, BCE tie-break|mean-of-batch ID finalAcc, strictly better by1e-12, no tie loss|
|Stop|3consecutive100% ID or60k|native patience and epoch/step caps|
|Curriculum|eligible-pool shuffles seed23000; fixed128/tail carry|native bucket4096, seed0+epoch, epoch1 start, partial batches|
|Dropout RNG|our sampling/forward schedule|their loader/scheduler/validation/RNG schedule|
|Library environment|torch2.9.1+cu128, numpy2.4.6, FLA.5.1|shipped unpinned upgrades in isolated venvs; numpy2.5.4, Delta FLA.6.0 (full freeze receipts)|
|Logging/eval raw accuracy|sample-weighted|upstream averages batch means; external pooled metrics added only after training|

UNCHANGED: native frozen X26b data hashes/counts/seeds, all task-contract fields,
BOS+MAT encoding and no STATE teacher forcing, GRU2x256/drop.3, Delta1x256/h4/
drop.2/negative-eigenvalues OFF, shipped final BCE+.3aux BCE, AdamW betas/eps,
clip1, BF16, curriculum length formula5->100/150epochs. `CHUNK_SIZE=64` is a
shipped argument accepted but unused by the real FLA model constructor; retain
it rather than 'repair' upstream. No settings inferred from unused shell knobs.

Paper caption at `results_stage26_replication/sources/paper-layout.txt:581–585`
defines matrix-length ID[1,100]/OOD[101,200]/[201,300]. Its raw accuracy bars
must not be compared without native-prior/balanced/recall disclosure. Source
revision-to-figure provenance is still unverified; this reproduces a public
pipeline at a pinned revision, not a certified reconstruction of Figure2.
