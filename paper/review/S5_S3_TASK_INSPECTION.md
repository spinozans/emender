# S5 / S3 permutation-composition task — inspection, worked examples, difficulty analysis

**Task:** `inspect-s5-s3`. READ-ONLY diagnostic. Goal: show *exactly* what the model
must solve in the S5 / S3 state-tracking tasks, with a **real** generated worked
example (nothing invented — every sequence below is produced by running the actual
task code), and an honest "is it too easy?" analysis.

All code references are to files in this repo. Every number / sequence in this doc
was produced by importing and running the real task classes
(`experiments/expressivity_tasks/tasks/s5_permutation.py`); none is hand-written.

---

## 1. Exact task specification (with code citations)

Both tasks are instances of one class,
`PermutationCompositionTask` in
[`experiments/expressivity_tasks/tasks/s5_permutation.py:31`](../../experiments/expressivity_tasks/tasks/s5_permutation.py),
with `S3PermutationTask` = `n=3` (`:71`) and `S5PermutationTask` = `n=5` (`:76`).
Registered as `s3_permutation` / `s5_permutation` in
[`tasks/__init__.py:36-37`](../../experiments/expressivity_tasks/tasks/__init__.py).

### 1.1 Vocabulary / alphabet — what tokens go IN

The **input alphabet is the set of adjacent transposition *generators*** of the
symmetric group $S_n$ — **not** the 120 (resp. 6) group elements.

```python
self.generators = [(i, i + 1) for i in range(n - 1)]   # s5_permutation.py:40
self.n_gen = len(self.generators)                       # :43
```

- **S5 (n=5):** `n_gen = 4` input tokens — the adjacent transpositions
  `(0,1), (1,2), (2,3), (3,4)`. Input ids ∈ {0,1,2,3}.
- **S3 (n=3):** `n_gen = 2` input tokens — `(0,1), (1,2)`. Input ids ∈ {0,1}.

These adjacent transpositions are the standard Coxeter generators of $S_n$; they
generate the whole group. So the input stream is a word over a 4-letter (resp.
2-letter) alphabet, and the output is the running product of that word in $S_n$.

`vocab_size = max(n_gen+1, n_classes)` (`:46`) = 120 for S5, 6 for S3 — sized so the
*same* embedding table covers both input generator ids and output permutation ids.

### 1.2 How a sequence is generated

[`generate_batch`, `s5_permutation.py:48-65`](../../experiments/expressivity_tasks/tasks/s5_permutation.py):

```python
gen_ids = rng.integers(0, self.n_gen, size=(B, T))     # :49  i.i.d. uniform generators
...
cur = identity                                          # :54
for t in range(T):
    cur = _apply_swap(cur, self.generators[gen_ids[b,t]])  # :56 apply transposition
    targets[b, t] = self.perm_to_id[cur]                   # :57 running product id
```

- Each input token is an **i.i.d. uniform draw over the generators** (`:49`).
  There is **no curriculum**, no special tokens, no identity token: *every* input
  is a genuine adjacent transposition (an odd permutation).
- The state starts at the identity and is updated by *swapping two adjacent
  entries of the current permutation tuple* at each step
  (`_apply_swap`, `:23-28`).

### 1.3 What the model must PREDICT at each position

The target at position `t` is the **running prefix-product** — the composition of
*all* generators seen so far (positions `0..t`), encoded as the **lexicographic id**
of the resulting permutation:

```python
self.perms = _all_perms(n)                  # :41  lex-ordered list of all n! perms
self.perm_to_id = {p: i for i,p in ...}     # :42  permutation tuple -> class id 0..n!-1
targets[b, t] = self.perm_to_id[cur]        # :57  cur = product of gen_ids[b, 0..t]
```

This is **true state-tracking**: the label is a deterministic function of the entire
prefix via the recurrence `state_t = swap(state_{t-1}, g_t)`. It is *not* "predict the
next element" and *not* a single final classification (unless `mode="final"`, see §1.5).

### 1.4 T, K, classes, chance, metric

| | S5 | S3 |
|---|---:|---:|
| n (= K) | 5 | 3 |
| input alphabet size (`n_gen`) | 4 | 2 |
| output classes (`n_classes = n!`) | **120** | **6** |
| naive chance `random_baseline_acc = 1/n!` ([`:67-68`]) | **0.00833** | **0.16667** |
| **effective chance per position** (see §3.2) | **0.01667** (1/60) | **0.33333** (1/3) |
| train length `seq_len` ([`run_separation_suite.py:38,46`]) | 128 | 128 |
| training steps ([`:45,37`]) | 20000 | 10000 |
| eval lengths ([`:41,49`]) | {128, 256, 512, 1024} | {128, 256, 512, 1024} |

**Metric** — per-position accuracy over the masked positions, computed in
[`train_hybrid.py:33-34`](../../experiments/expressivity_tasks/train_hybrid.py):

```python
preds = logits.argmax(dim=-1)
correct += ((preds == y) & m).sum().item()      # masked, per-position
total   += m.sum().item()
```

So the reported accuracy is **token-level (per-position), averaged over all
supervised positions in the batch** — not last-token-only, not sequence-exact.

### 1.5 `running` vs `final` mode — the supervision mask

[`s5_permutation.py:60-64`](../../experiments/expressivity_tasks/tasks/s5_permutation.py):

```python
if self.mode == "running":   mask = np.ones((B, T), bool)      # supervise EVERY position
else:                        mask[:, T-1] = True               # supervise only the last
```

The separation suite uses the **default `running` mode** (the task is constructed
with no `mode` override in `train_hybrid.py:120`), so **every position 0..T-1 is a
supervised state target** — the densest possible supervision. (Verified at runtime:
`mask.all() == True`.)

---

## 2. REAL worked examples (generator actually run)

Produced by importing the real classes and calling `generate_batch`. Reproduce with:

```python
import numpy as np
from experiments.expressivity_tasks.tasks.s5_permutation import S3PermutationTask, S5PermutationTask
inp, tgt, mask = S5PermutationTask(mode="running").generate_batch(B=1, T=10, rng=np.random.default_rng(1))
```

### 2.1 S5 worked example (n=5, T=10, seed=1) — short enough to trace by hand

`n_gen=4`, generators `[(0,1),(1,2),(2,3),(3,4)]`; `n_classes=120`, chance 0.0083.
Read it as: start at identity `(0,1,2,3,4)`; at each step swap the two adjacent
positions named by the generator; the target id is the lex rank of the running perm.

```
 t in_id  generator      running_perm  target_id
 0     1     (1, 2)   (0, 2, 1, 3, 4)          6   <- swap pos1,pos2 of identity
 1     2     (2, 3)   (0, 2, 3, 1, 4)          8   <- swap pos2,pos3 of previous
 2     3     (3, 4)   (0, 2, 3, 4, 1)          9
 3     3     (3, 4)   (0, 2, 3, 1, 4)          8   <- (3,4) again undoes step 2
 4     0     (0, 1)   (2, 0, 3, 1, 4)         50
 5     0     (0, 1)   (0, 2, 3, 1, 4)          8   <- (0,1) again undoes step 4
 6     3     (3, 4)   (0, 2, 3, 4, 1)          9
 7     3     (3, 4)   (0, 2, 3, 1, 4)          8
 8     0     (0, 1)   (2, 0, 3, 1, 4)         50
 9     1     (1, 2)   (2, 3, 0, 1, 4)         60
```

Input id sequence: `[1,2,3,3,0,0,3,3,0,1]` → target id sequence
`[6,8,9,8,50,8,9,8,50,60]`. The model must emit the **target_id** column given only
the **in_id** column. Note positions 3 and 5: applying the same generator twice in a
row returns to the prior state — the model must carry the *full* running state to
know that, not just the last token.

### 2.2 S3 worked example (n=3, T=10, seed=0) — the control

`n_gen=2`, generators `[(0,1),(1,2)]`; `n_classes=6`, chance 0.1667.

```
 t in_id  generator    running_perm  target_id
 0     1     (1, 2)      (0, 2, 1)          1
 1     1     (1, 2)      (0, 1, 2)          0   <- back to identity
 2     1     (1, 2)      (0, 2, 1)          1
 3     0     (0, 1)      (2, 0, 1)          4
 4     0     (0, 1)      (0, 2, 1)          1
 5     0     (0, 1)      (2, 0, 1)          4
 6     0     (0, 1)      (0, 2, 1)          1
 7     0     (0, 1)      (2, 0, 1)          4
 8     0     (0, 1)      (0, 2, 1)          1
 9     1     (1, 2)      (0, 1, 2)          0
```

Input `[1,1,1,0,0,0,0,0,1]`... → target `[1,0,1,4,1,4,1,4,1,0]`. Same machinery,
smaller group.

---

## 3. Honest difficulty analysis — "is it too easy?"

### 3.1 The formal point: a fixed-length S5 instance is in TC⁰/AC⁰

The headline theory claim (paper `main.typ:524`; reviewed at length in
[`PRECISION_NONLINEARITY_RESEARCH.md`](PRECISION_NONLINEARITY_RESEARCH.md) §1, §6)
is **linear-state ⊆ TC⁰ ⊊ NC¹**, with $S_5$ the canonical NC¹-complete witness via
Barrington's theorem. **But that separation is asymptotic.**
`PRECISION_NONLINEARITY_RESEARCH.md:109-113`:

> "Finite training length makes the asymptotic separation moot. TC⁰ ⊊ NC¹ is an
> *asymptotic* statement; every fixed-length instance of the $S_5$ word problem is
> ... trivially solvable."

So **high accuracy at the training length T=128 is fully expected and is *not*
evidence against the theory.** A fixed length-128 $S_5$ product is a finite circuit
a constant-depth net can fit. The discriminating signal is **length
generalization**, not T=128 accuracy. The repo's own data shows exactly this
shape (`PRECISION_NONLINEARITY_RESEARCH.md:34`): the linear winner is **0.9997 at
T=128** but decays **0.75 → 0.39 → 0.20** at T=256/512/1024 — "the predicted
asymptotic failure showing through" (`:422`). So the right reading of "too easy at
T=128" is: *yes, and it is supposed to be*; the task earns its keep at T≥256.

### 3.2 Is there a shortcut? Two findings, one reassuring and one caveat

**(a) No fixed-window heuristic suffices — the target is genuinely the running
product.** Empirically (real run, B=200, T=128): for window sizes k ∈ {1,2,4,8},
**98.5–99.3% of input-windows that recur map to *different* targets**. A predictor
that looked only at the last k input tokens would be wrong almost always. The label
truly depends on the entire prefix — this is real state-tracking, no local shortcut.

**(b) CAVEAT — position parity is a free bit, so the effective chance level is
DOUBLE the advertised one.** Every input token is a transposition (an *odd*
permutation), and there is no identity token. Therefore after `t+1` tokens the
running product has parity `(t+1) mod 2` — **the parity is fully determined by the
position index alone.** Verified at runtime (real generator, 500 sequences, every
position): the set of target parities at position `t` is exactly `{(t+1) mod 2}`,
with **0/64 positions violating it**, for both S3 and S5.

Consequences:
- At any fixed position only **half** the group is reachable: the alternating group
  $A_n$ at even positions, its coset at odd positions. **S5: 60 classes per
  position, not 120. S3: 3 classes per position, not 6.**
- The honest **per-position chance level is 1/60 = 0.0167 for S5** and **1/3 =
  0.333 for S3**, i.e. **2× the `random_baseline_acc()` reported by the task**
  (which returns `1/n!` = 0.0083 / 0.1667, [`:67-68`]). A model that learns "use the
  position index to halve the candidate set" gets this bit for free; the
  `random (1/120)` row in the results tables is therefore a *conservative
  (too-low)* baseline by 2×. This does not change the qualitative separation but it
  inflates the apparent "× above chance" headline numbers by 2×, and it makes the
  S3 control easier than 1/6 suggests (only 3 live classes).

  *This is the single most important "too easy" caveat in this doc.* It is not a
  correctness bug — the targets are right — but any claim of the form "N× above
  chance" should use 1/60 (S5) / (1/3) (S3) as the floor.

### 3.3 Why high at T=128 but decaying with length

The base task *is* learnable at the trained length (a fixed circuit fits T=128), but
maintaining the running composition is **not length-robust for a linear-state
recurrence**: it must implement true non-abelian group multiplication step after
step, and a linear (TC⁰) recurrence provably cannot do so at unbounded length
without precision/width growing with T. So the model fits the training-length
distribution and then **decays toward (the effective) chance** as T exceeds what it
saw — the textbook signature of "learned the length-128 instance, did not learn the
*algorithm*" (`train_hybrid.py:221-224`; `PRECISION_NONLINEARITY_RESEARCH.md:36-40`).
That is *why* the protocol trains at 128 and evaluates out to 1024: the gap between
T=128 and T=1024 accuracy is the actual measured quantity.

### 3.4 Distributional notes — are some positions easier than worst-case?

- **No easy/identity tokens injected.** Inputs are uniform over the *non-identity*
  generators (`:49`), so the model never gets a "do nothing" step that trivially
  preserves state. Each step is a genuine swap.
- **The running product mixes toward uniform on $A_n$/its coset.** At the last
  position of T=128 (real run, B=200) the S5 targets covered 58 of the 60 reachable
  ($A_5$) ids with identity appearing only ~3.5% of the time — i.e. the long-T state
  is close to uniform over the reachable half, *not* concentrated on easy elements.
  Early positions (small t) are easier simply because few generators have been
  applied (fewer reachable states), but per-position accuracy averages over all t,
  so the bulk of positions are in the well-mixed regime.
- The only structural freebie is the parity bit of §3.2.

### 3.5 Is the S3 control doing its job?

**Yes.** S3 is the same machinery at `n=3` (`S3PermutationTask`, `:71`): same input
format (transposition generators), same running-prefix-product target, same dense
supervision, same train/eval lengths {128,256,512,1024}. The *only* differences are
group size (6 vs 120) and — crucially — **$S_3$ is solvable while $S_5$ is not.**
That is exactly the intended control: under the Barrington/Liu-et-al framing
(`PRECISION_NONLINEARITY_RESEARCH.md:400-405`) a linear-state / constant-depth model
*can* track a solvable group's word problem with length robustness but *cannot*
track $S_5$. So a clean result is "model length-generalizes on S3 but collapses on
S5" — and indeed the from-scratch and 1.3B-finetune results
([`S3_S5_FINETUNE.md`](S3_S5_FINETUNE.md)) report the linear/M2RNN models holding up
better on the solvable parity/S3 ladder than on S5. Caveat carried over from §3.2:
S3's per-position chance is 1/3, not 1/6, so "above chance" on S3 should be judged
against 0.333.

---

## 4. Bottom line

- **What goes in:** a uniform i.i.d. stream of adjacent-transposition generator ids
  (S5: 4 symbols, S3: 2 symbols). **What comes out:** at *every* position, the lex
  id of the running group product of all generators seen so far. True state-tracking,
  per-position accuracy, train at T=128, eval at {128,256,512,1024}.
- **Too easy?** At T=128, *expectedly* easy — fixed-length $S_5$ is TC⁰, so high
  T=128 accuracy is predicted and not a theory violation; the task discriminates via
  **length extrapolation** to 256/512/1024. There is **no fixed-window shortcut**
  (target needs the full prefix). The one genuine easiness is the **parity-from-
  position freebie**, which halves the live class count and means the honest chance
  floor is **1/60 (S5) / 1/3 (S3)** — double the task's reported `1/n!`. Report
  "× above chance" against those.
- **S3 control:** correct and well-matched — identical setup, solvable group, the
  intended foil to the non-solvable $S_5$.
