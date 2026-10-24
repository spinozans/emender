# Field-by-field native IMM vs X23E/X26 scalar contract

Sources: unchanged LinearRNN@dbfa365a `imm_unbound/gen.py`, `train_rnn.py`,
`train_deltanet.py`, shell configs; X23E `tasks/merrill26_separators.py`;
X26 `run_merrill26_replication.py`. Native control executes those upstream
classes/collate/loss directly. Scope is IMM, not a native DGC experiment.

|Field|X23E / failed X26 scalar adaptation|Native control as shipped|
|---|---|---|
|Matrix entries|P(-1,0,+1)=(.45,.10,.45)|IID uniform(-1,0,+1), each1/3|
|Distribution/numbering|3x3 row-major matrices, no graph numbering|Same3x3/row-major; DGC numbering not applicable|
|Product order|P=I; P=P@M_t|Same right multiplication, not reversed/transposed|
|Label|1 iff exact P[0,0]>0|1 iff int64 P[0,0]==0; negative and positive BOTH class0|
|Arithmetic|Python arbitrary-precision integers, no clipping/modulo|NumPy int64 wraparound; no explicit clipping, overflow deliberately preserved|
|Sequence|9N categorical scalar tokens (-1,0,+1 mapped0,1,2), pad3; no BOS|BOS+N MAT tokens; each MAT has raw9-vector and learned type; pad type0|
|Length header|N implicit in scalar count|Text T header parsed by dataset, NOT given as model input; BOS only extra token|
|Encoder|Embedding(4,256) replaces their native encoder|Original type embedding(4,256)+Linear(9,256); no positional embedding|
|Final prediction|Final scalar (ninth entry of last matrix)|Final MAT token at indexN, not BOS|
|Aux placement/semantics|.3 BCE at completed intermediate ninth scalars, exact >0|.3 BCE at intermediate MAT1..N-1, ==0 of signed_log1p(int64 prefix) (zero preserved)|
|Loss|Mean final BCE +.3 mean prefix BCE, global effective-batch normalization|Same mathematical reductions, direct upstream functions, now native labels/positions|
|Teacher forcing|No prefix-product inputs|Default false: NO STATE inputs; stored prefixes are output-target-only|
|Balancing/rejection|Global exact50/50, independently sampled N uniform, rejection conditioned on desired >0 label|Fixed700/200/100 rows PER LENGTH, no balance/rejection; singular/zero products retained|
|Splits/seeds|70k/20k/10k each bin; independent seeds23001..23005|Same counts/bins, single Python Random0 sequential split generation|
|Curriculum|X26 eligible shuffled epochs5->100/150, carry tails128|Retained exactly; native formula agrees, native bucket/partial batches not adopted|
|Optimizer/precision/stop|Reconciled128/1e-4/1e-4/clip1/bf16/no warmup/60k/eval500/3perfect|Retained; NOT raw scripts256/lr3e-4/wd.01 GRU,.005 Delta/1000warmup/30k/epoch eval/patience|
|Architecture core|GRU2x256/drop.3 or real FLA DeltaNet1x256/h4/drop.2|Same cores/heads/dropout; original9-vector encoder restored (params793857/270401 vs791297/267841)|

Native majority baselines: train52.4129%, validation52.655%, ID53.21%,
OOD1 76.54%, OOD2 89.61%. Thus high native OOD accuracy can be explained by
imbalance without learning; class recalls/balanced accuracy are compulsory.
All native distribution/overflow differences are deliberate, supervisor-approved.

Finding BEFORE outcomes: this is NOT a tokenization-only intervention: input
probabilities, label predicate, arithmetic and balance also change. Success on
native vs failure on scalar would identify the load-bearing contract BUNDLE,
not uniquely tokenization, label semantics, or overflow. Identifying a single
causal field needs separately registered factorial controls; no such launch here.
