# X26 reconciled-source discrepancy ledger (before training)

Claim: **partial replication under reconciled sources**, NOT exact Figure 2 reproduction.
Sources: local `sources/paper.pdf`, `sources/paper-layout.txt` (Poppler `pdftotext -layout`),
public https://arg-git.informatik.uni-kl.de/pub/LinearRNN.git at
`dbfa365ad680f425dfb79a9e78b5e0d7915757bb`. Selected upstream Python/shell/data-generator
files are byte-for-byte snapshots in `sources/LinearRNN/`; `sources/manifest.json` binds them.
No source proves which public revision generated Figure 2; 'stale table' is a hypothesis,
not an established erratum. Sarrof et al. is the paper's attribution, not a verified
identity between its implementation and this public repo's clamped RNN / GRU.

## Exact Appendix Table 1 (PDF p31, layout lines1898–1911)

|Task|Model|dm|L|dh|h|dstate|LR|steps|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|DGC|RNN|256|2|256|—|—|3e-4|60000|
|DGC|Transformer|256|2|—|4|—|3e-4|60000|
|DGC|Mamba|256|2|—|—|128|3e-4|60000|
|DGC|RWKV-7|256|2|—|4|—|3e-4|60000|
|DGC|DeltaNet|256|2|—|4|—|3e-4|60000|
|IMM|RNN|256|2|256|—|—|3e-4|60000|
|IMM|Transformer|256|2|—|4|—|3e-4|60000|
|IMM|Mamba|256|2|—|—|64|3e-4|60000|
|IMM|RWKV-7|256|2|—|4|—|3e-4|60000|
|IMM|DeltaNet|256|2|—|4|—|3e-4|60000|

Here `dm` is model dimension, `dh` hidden dimension (not number of heads), `h`
heads, `dstate` SSM expansion factor; **m is not an additional model-size column**.
All rows: AdamW, batch128, norm clip1.0; table does not specify weight decay.

## Rulings (supervisor-approved before implementation/optimizer updates)

|Seam|Conflicting receipt|Chosen primary and reason|
|---|---|---|
|LR / WD|Section6 p9/layout589–598: lr1e-4, wd1e-4; all Table1 rows lr3e-4; repo scripts vary|Section6 lr/wd1e-4, constant; no warmup. Explicit operator protocol and supervisor precedence.|
|Batch|Section6=64; Appendix F/layout1882–1887 and Table1 caption=128; repo DGC-RNN/IMM=256|128: two appendix statements agree, approved ruling, not claimed identical to actual Figure2 run.|
|Budget / stopping|Section6=100% ID three consecutive evaluations or60k; DGC/IMM scripts often30k, patience/epoch caps|60k ceiling; validation every500 updates inherited X23E; three100% validations. No patience/epoch cap; OOD never selects checkpoint or stop.|
|DGC RNN|Table1 dm256/L2; dgc/rnn.sh emb128,hidden256,L1,drop.1,hh_shrink.3; train_rnn.py clamped ReLU, mean pool, GELU head, 2 logits|Repo architecture: emb128->256, ONE clamped-ReLU layer, clamp6, shrink.3, masked mean+LayerNorm+256/GELU/drop.1/2 head. Retain mean pool under paper silence. BCE on logit1-logit0 (mathematically equivalent to two-class CE). Do not call it a Table1-sized L2 model.|
|DGC DeltaNet|Table1 L2/dm256; delta.sh emb128,hidden256,L1,h4,drop.1; script defaults WD.01, short conv OFF|Repo architecture: emb128->256, ONE actual FLA DeltaNet layer, 4heads (64 each), expand_k/v1, beta+gate, no short conv, negative eigenvalues OFF; residual/drop/LayerNorm and same mean pool/head. Override optimizer/budget/batch only.|
|IMM RNN|Paper calls nonlinear RNN; imm_unbound/rnn.sh GRU2x256/drop.3, wd.01, batch256, aux.3, curriculum5->100/150epochs; model class has no extra MLP despite unused script knobs|Actual repo GRU core2x256/drop.3+LayerNorm+linear scalar head. Replace native type+9-vector encoder with scalar embedding4x256/drop.3; retain X23E9N tokens and >0 labels. Aux.3 every completed intermediate matrix (every ninth scalar), exact big integers and >0; no prefix-state input.|
|IMM DeltaNet|Table1 L2; imm_unbound/delta.sh L1/h4/d256/drop.2, wd.005, aux.3/curriculum150|Repo ONE real FLA DeltaNet layer256/h4, expand_k/v1, beta, gate OFF, short conv ON(size4), negative eigenvalues OFF, RMSNorm/drop.2/scalar head; same scalar encoder adaptation and aux/curriculum as above. Current upstream code explicitly replaces an older mislabeled additive linear-attention model.|
|Curriculum epochs|Repo epoch=eligible-pool pass; our pool70k, not their dataset size|One shuffled pass of eligible frozen X23E rows per epoch; maxT=max(5,round(5+95*min(epoch/150,1))). Carry epoch tail across updates to maintain128 examples; recorded epoch/maxT. No stream regeneration/truncation.|
|Task label|Main Def12 P[0,0]>0 vs Appendix F/layout1875–1881 and repo P[0,0]==0|Retain reviewed X23E main-definition exact >0. Aux labels likewise >0. This is a substantive task mismatch, not a claimed proven bug in either source.|
|IMM input|Repo BOS+N matrix-vector tokens (or explicitly leaky teacher-forced STATE option); X23E9N scalar tokens|Retain X23E scalar stream; no BOS, header, product state, or extra target input. Only the model changes relative to our accepted task contract. Native tokenization is a conditional follow-up control, NOT run now.|
|DGC input|Repo character tokenizer vocab PAD,0,1,;,- vs X23E typed unary/source/edge/pair/target/EOS/SOS IDs|Retain exact X23E reviewed tokens/vocab8. This encoder adaptation is disclosed; not an exact repo-default replication.|
|RWKV optional|Both repo configs available:256/depth2/head_dim64,emb128 forDGC,drop.1,4xMLP; IMM native encoder as above|Both max-bin zero-update probes failed upstream layer1 `v_first=None` at torch.lerp on installed FLA. Raw `.txt`/exit receipts retained. No new global lerp monkeypatch or different recurrence silently substituted; neither optional arm authorized for launch.|
|Secondary repo-default RNN|dgc/rnn.sh uses `--post_act_clip`, but argparse only defines `--act_clip`; additional repo defaults differ in data/precision/CE/budget|Cannot truthfully run script exactly as-is. Deferred, not launched or described as unmodified; optional triangulation requires a separately registered correction/control.|
|Nonfinite handling|Upstream DGC scripts nan_to_num logits before loss|Fail closed as software/numerical error, never replace logits or interpret as inability.|

## Approved interpretation / contingency

Qualitative target: RNN DGC ~100% ID and strong OOD, DeltaNet high ID/degraded OOD,
both learn IMM ID. Numerical thresholds are registered in REGISTRATION.md, not inferred
from a training result. **A failure of their models on our contract does NOT by itself
prove a generator bug; it triggers the native-tokenization control and the field-by-field
comparison.** Compare distributions, graph numbering/encoding, product order, labels,
clipping/overflow, balancing/rejection, sequence headers, and teacher forcing first.
Do not attribute failures to architecture/theory or retrospectively change this task.

No Figure2 numeric values are reconstructed from the scrambled markdown; only the
paper's stated qualitative pattern is targeted. No resilient DiLoCo behavior changes
(R01–R16/NDP01–NDP17/V21S01–V21S17/ISP01–ISP07 not applicable); local lambda01 CUDA,
no Frontier scheduler queue binding claimed.

---

## Clarification (X26R review, parent-receipted 2026-10-23)

Per the X26R acceptance condition: the registered predictions are QUALITATIVE
(the Figure 2 pattern: RNN generalizes OOD on DGC; DeltaNet degrades OOD on
DGC; both learn IMM in-dist) - no numerical success thresholds or Figure 2
accuracy coordinates were registered, and none may be applied post hoc. The
wording above ('numerical thresholds are registered') referred to the
deterministic data/label checks in the test suite, not to evaluation
thresholds. The hash-bound registration is unchanged; this clarification is
receipted separately. No exact-Figure-2 claim and no task-bug attribution are
licensed by the registration; the failure-triggered native-tokenization control
procedure stands as registered.
