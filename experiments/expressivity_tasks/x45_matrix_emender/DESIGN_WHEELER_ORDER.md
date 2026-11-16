# DESIGN — the Wheeler-proper fork: an ordered-key store as an online maintained index

Design document (registration input, **not** a registration). Status: design
confirmed by the operator lane (three semantics decisions on record, see
§3–§5); the pre-registered predictions live in
[`RESULT_WHEELER_CAPACITY.md`](RESULT_WHEELER_CAPACITY.md) and are committed
before any training run of the SPAN-RUN probe.

## 0. Frame (the operator's compression-theoretic argument)

The corpus spine (chiSA suffix array, `x43_fork_resolved/DESIGN_SUFFIX_ATTENTION.md`)
is a STATIC compressed index over the corpus: order is the address (the BWT /
suffix order), intervals are the retrieval (a query opens `[L,R)` and the
interval's width and branch histogram are the answer's support), and repetition
is what makes the index small (runs are the compression). The identity-cache
diagnosis at 150 M (`CMAES_TUNING.md`, confirmed at probe scale by the SEP-KV
lane's collision lemma) is the degeneration of exactly this machinery to **RLE
with run length 1**: the store caches the latest write per content and never
builds runs, so it holds `r` memories and compresses nothing.

The claim under test: a **pointwise** store holds `r` memories; an **ordered,
run-coalescing** store holds `Theta(r x avg-run-length)` of context mass — the
same mathematics as the static spine, maintained ONLINE by the model over its
own context: order is the address, intervals are the retrieval, MERGE is
run-length coalescing. The probe below (`wheeler_capacity_probe.py`,
[`RESULT_WHEELER_CAPACITY.md`](RESULT_WHEELER_CAPACITY.md)) is the capacity
witness at oracle level.

## 1. DEPENDENCY — the parallel SEP-KV lane

The SEP-KV lane (cards 2/3, running: `key_separation_probe.py`, its committed
pre-registration and receipts) tests the orthogonal axis: whether **key-value
separation is necessary at all** (free learned keys vs fused content keys).
This stage tests **ordered/interval vs pointwise/free**, not separation: both
arms here are built on the SEP-KV **minimal separated-key form** as the base
(the pointwise arm IS that form verbatim at rank 8; the ordered arm adds only
the order axis). Consequences, stated as a hard dependency:

* **No files of the SEP-KV lane are touched** — this stage imports
  `separated_oracle_scan` and the `SeparatedModel` modules read-only, exactly
  as the separator lane imported the scan before it.
* **Reclassification clause.** If SEP-KV refutes separation entirely (the
  fused arm learns its task families), these results RECLASSIFY: the ordered
  axis would then be evaluated over fused, content-derived keys. The frame
  survives that reclassification in compressed form — the static spine's
  order is content-derived, so a fused ordered variant remains meaningful —
  but the arm definitions of this probe (separated keys at the base) would no
  longer be the minimal forms, and the capacity numbers below would be
  re-interpreted as the ordered axis measured on a non-minimal base.

## 2. The order mechanism: DERIVED vs MAINTAINED — recommendation: DERIVED

**(a) Derived order** — the store's keys are scalars in `(-1, 1)` (a `tanh`
squash of a learned projection); the order is `argsort` of the key values.
Insertion is *writing a key between its neighbours*: the new key takes its
sort position with **no relabeling** of any row — slot indices are stable
storage addresses, the order is a derived view, and POP simply removes a value
from the sort.

**(b) Maintained order** (Dietz–Sleator order-maintenance labels) — an explicit
linked list between successive writes with amortised O(1) insert-between and
O(1) comparison, at the price of a pointer structure whose labels change
under insertion.

**Trade analysis and recommendation.** At probe scale the decision is easy, and
it is the same decision at kernel scale:

* *Cost.* Derived order pays an `argsort` of `r` scalars per decision point
  (`r = 8` here; `O(r log r)` at scale, a batched radix sort over the state's
  key channel). Maintained order pays nothing per comparison but everything
  in representation: the labels are positions in a host-side list, not tensor
  coordinates. The program's whole scan doctrine (`algebra.py`,
  `formal/lean/OPERATOR_ALGEBRA_STATUS.md`: the schedule is a fixed,
  detached, tensor-shaped decision channel consumed by an affine scan) needs
  the state to stay `(r, d_v)` with stable slot axes; a maintained order
  reintroduces the sequential dependency and the relabeling storm that the
  matrix fork removed, and the K8 growth machinery (prefix-key zero-injection)
  composes with a derived order for free — new rows enter the sort by their
  key, growth reveals slots, nothing renumbers.
* *Differentiability.* Neither order channel is differentiated (the doctrine:
  decisions are detached). The derived order's *gradient channel* is the key
  value itself: training moves keys, hence the order, without ever touching a
  permutation — which is exactly how the Wheeler-ness diagnostic (§6) can be
  measured on trained weights.
* *Fidelity to the frame.* The static spine's order is derived from content
  (the suffix order is a function of the bytes), never maintained by labels;
  a derived order is the honest online analogue.

**Derived order is recommended and implemented.** Maintained order remains the
fallback if a future registration needs O(1) neighbour queries inside a
decision (e.g., interval ops at sub-sort granularity); nothing in this
probe's semantics depends on the choice.

## 3. Interval-read semantics (confirmed by the operator lane)

The read of the ordered store is **one interval read**: the query's key `q`
opens `[q - w, q + w]` in the learned key order (`w` a registered constant; the
whole probe uses `w = 0.25` over keys in `(-1, 1)`).

* **Weights** — the **full-population** softmax over ALL live rows of the
  score `-read_scale * |q - k_j|` (registered `read_scale = 2.0`); the
  denominator spans every live row. Rows **outside the interval receive
  numerator weight exactly 0**. So the read is
  `out = sum_{j live, |k_j - q| <= w} softmax_j * strength_j * S_j` with the
  softmax normalised over all live rows — the K7 split-softmax precedent
  (`K7_HIGH_RANK.md`: the denominator spans slot blocks under the Lean law).
* **Why this form (the Lean truncation law).** With the full-population
  denominator, survivor coefficients are invariant under interval widening
  (`AttentionOperator.full_coefficients_truncation_invariant`), omitted mass
  drops the output mass monotonically
  (`full_truncation_mass_monotone`, `full_truncation_decomposition`), and the
  output norm stays bounded (`readout_norm_le`: `norm(out) <= alpha * B` for
  values bounded by `B`). Retained (interval-only) normalisation is
  **rejected** per `Obstructions.truncation_sharpens_survivor`: it would make
  a row look *sharper* when its neighbours are omitted — exactness here is
  re-earned by the full denominator, not assumed by renormalisation (the K7
  doctrine verbatim).
* **Omitted mass is carried, not hidden.** Every interval read receipts the
  mass it omitted: `omitted = sum_{j live, outside} softmax_j * strength_j`
  and the retained share, recorded per read in the probe's channel receipts.
  An empty interval is a value: the read is exactly 0 (the R2.4 rule — empty
  support never renormalises into noise).

Registration-level constants of the ordered read: `w = 0.25`,
`read_scale = 2.0` (inherited), keys not decayed (an address is categorical —
the SEP-KV lane's flagged open question, inherited here), values decayed by
the registered DECAY gate.

## 4. MERGE as run-length coalescing — the five ops in index terms

**REGISTRATION-LEVEL SEMANTIC (marked per the operator lane's confirmation):
the accumulate-vs-overwrite distinction below is the ordered variant's
DEFINING new semantic — it is not inherited from the matrix registration.**

The registered five ops, restated for an ordered key store (`k_t` the write's
scalar key, `k_j` a live row's key, `S_j` its value, `||S_j||` its strength):

| op | index-term form | ordered-store semantics |
| --- | --- | --- |
| APPEND | **insert** | a free slot takes the write: `S_j := p_t`, `k_j := k_t`, run length `:= 1`. The key takes its derived sort position; nothing relabels. |
| CORRECT | **update-in-run** | a write with `|k_t - k_j| <= tau_match_dist` for its nearest live row `j` **accumulates**: `S_j := S_j + p_t` (mass-preserving, strength-weighted by the payload's own norm), run length `j := +1`. The row **keeps its key** (a quantisation centre; see the trade below). |
| POP | **close-interval** | `S_j := 0`, liveness off — the interval at `k_j` closes, its mass is lost. |
| MERGE | **coalesce** | under capacity pressure, the tightest adjacent live pair with key gap `<= tau_run` coalesces: `S_i := S_i + S_j` (mass-preserving), run lengths add, row `j` closes. **MERGE has priority over POP**: pressure is resolved losslessly first. |
| DECAY | **age** | the registered multiplicative gate on values only; keys are addresses and do not age. |

**Legality (what makes this an emender store and not a new algebra).** The
accumulate is an *additive* rank-1 update — literally the registered APPEND
coefficient form `S[j] += v` applied in-run — so it is within the algebra's
existing scan-legal form (`algebra.apply_lane` with `erase=False`); MERGE is
two such updates plus a POP, the registration's "MERGE = two rank-1 ops" with
the blend replaced by a mass-preserving sum; POP and DECAY are verbatim. The
pointwise arm's registered CORRECT (erase-and-replace) stays verbatim in that
arm. The pair (accumulate in-run, MERGE-before-POP priority) is the
operational content of "coalescing": the pointwise row holds the latest write
only (**run length 1 — the identity cache**), the ordered row holds the
**run's mass**.

**Fixed centre vs drifting centre (the update-in-run trade).** On
update-in-run the row keeps its key. The alternative — drifting the row's key
toward each new write (a running mean) — fails for exactly the reason the
frame predicts: a drifting row walks the key arc as the run extends, and the
mass ends up parked at the run's *last* address, where no earlier member's
query can open it (the interval read at an interior member's key misses the
row entirely). Fixed centres make the store an **adaptive quantisation of the
address space**: rows are centres, a write joins its nearest centre, and a
query at address `a` reads the local cell — the online image of an interval
in the static spine. The Wheeler-ness of the learned solution is then a
measurable property of the trained key projection (§6), never an assumption.

Registered constants of the ordered policy (new for this variant, not
inherited — the registered `tau_*` are cosine thresholds in `R^16`; these are
key-space radii in `(-1, 1)`): `tau_match_dist = 0.10` (a write joins a row's
run within this radius), `tau_run = 0.25 = w` (adjacent live rows this close
coalesce under pressure), `tau_pop_dist = 0.5` (POP fires only if the
farthest live row is at least this dissimilar, else the write is dropped —
the NONE branch). Two ordering facts are load-bearing and stated: **`tau_run`
must exceed `tau_match_dist`** — with `tau_run <= tau_match_dist` a
coalescible pair can never exist (the second write of any pair closer than
`tau_match_dist` matches the first row and accumulates instead of inserting
two rows), the same self-deduplication impossibility the separator lane
receipted for the registered MERGE's `tau_merge > tau_match` ordering; and
**`tau_run = w` is the losslessness rule** — two rows closer than the
interval width are never resolved apart by any read, so coalescing them under
pressure loses nothing the read could have expressed. Sensitivity of these
radii to the learned key scale is an open registration question; the probe's
receipts (run-length histograms, key-scale statistics) are designed to make
that sensitivity legible rather than to tune it away.

## 5. The arms (both on the SEP-KV minimal separated-key base)

* **(a) POINTWISE** — the SEP-KV minimal separated-key form **verbatim** at
  rank `r = 8`: free 16-D learned keys projected from the write context,
  matching by key cosine, the registered five-op `RoutingPolicy` (whose
  `tau_merge = 0.85` double-match makes MERGE effectively inert — the lane's
  receipts; "no coalescing" is the registered policy's own behaviour, not a
  new construction), CORRECT = erase-and-replace (run length 1), the certified
  read `softmax * strength` over all live rows. One address per written item.
* **(b) ORDERED** — the same base (same block shape, same write-context key
  projections, same DECAY gate, same value/outhead) with exactly three
  deltas: scalar 1-D keys (§2), the interval read (§3), and the run-coalescing
  op semantics of §4.

Both arms are eager per-token torch oracles (the K6 doctrine: decisions from
the detached state, read weights attached), rank 8, probe-scale models
(~70k params), triton/torch-native, **no kernel work, no edits to
`k6_*`/`k7_*`/`k8_*`/`k9_*`/`granularity_separator.py`/SEP-KV files**.

## 6. The Wheeler-ness diagnostic (operational, never an assumption)

Post-training, on the trained weights, per evaluation window and layer: rank
the window's token writes by their emitted scalar keys; let the SPAN's binding
writes have ranks `R` (there are `S` of them). **Contiguity** is

```
contiguity = S / (max(R) - min(R) + 1)
```

— 1.0 when the span's writes occupy one contiguous block of the learned order
(same-role contexts sort together: the Wheeler property), lower when foreign
writes interleave. The probe receipts the mean contiguity, the interleaving
writer classes (BOS/PAD/distractor/probe), the run-length histogram
(writes absorbed per live row — does the store actually build runs?), and the
span's row count, so "the learned order is Wheeler-like" is a measured claim
with a measured failure mode, not a design assumption.

## 7. Interactions

* **Growth.** New slots enter by the K8 zero-injection (legal per
  `injection_propagation`): a grown row is dead until a write claims it, and
  it joins the derived order by its key at that moment. Order and growth
  compose with no relabeling — the sort is over live keys. This probe runs
  fixed rank (r = 8); the composition is stated because the registration
  this design feeds will grow.
* **Run boundaries.** A run boundary is a key gap `> tau_run = w` between
  adjacent live rows. Coalescing never crosses a boundary (the pressure path only
  merges pairs inside one); the **interval read may cross boundaries** — `w`
  is a query radius, not a run predicate — and the omitted-mass receipt
  discloses exactly what crossing cost. Boundary semantics are therefore
  write-side tight, read-side permissive, and the receipts separate the two.
* **SEP-KV.** §1: the arms share the separated-key base; a SEP-KV refutation
  reclassifies this stage's arm definitions (the order-axis results carry,
  the base does not).

## 8. What is provable now vs what needs a separator

**Provable now (machine-checked or Lean-covered):**

* *The algebra lift.* An ordered key store with derived order IS the matrix
  store plus a detached permutation view: the schedule still records stable
  slots, the ops are the registered coefficient updates (`S[j] += v`,
  `S[j] := 0`, `S *= lambda`), so every fixed-schedule Lean theorem
  (`MemoryAlgebra.AffineOperator`, the scan/monoid results) applies verbatim;
  the accumulate is the registered `erase=False` lane, hence scan-legal.
* *The interval omitted-mass bound.* The interval read is exactly the Lean
  `AttentionOperator` structure with the interval as the returned-candidate
  set and full normalisation: survivor-coefficient invariance, monotone mass
  loss, and `norm(out) <= alpha * B` all hold for it; the K7 precedent
  already carried this numerically at rank 2048.
* *The pointwise slot law.* Distinct bindings need distinct live rows in the
  pointwise arm (the SEP-KV collision lemma: content-identical or
  key-identical rows are mechanically indistinguishable to the read), so a
  pointwise store holds at most `r` distinct bindings — the floor of the
  capacity argument, machine-checked in the SEP-KV lane's tests.

**The free-key collapse basin (receipted mid-sweep; supervisor-directed
addition, 2026-11-16).** The probe's ordered arm trains a FREE scalar key
projection, and one seed in three fell into a degenerate basin the static
spine never faces because its order is built offline: the key projection
collapsed to a single quantisation centre (`span_key_std -> 0`), every write
matched row 0, one run absorbed the whole window (run length ~498 at S=2
scale), and accuracy fell to chance — the mass-preserving accumulate
faithfully preserved everything into one address nothing could resolve. This
receipt is DIRECT EVIDENCE for the constrained key family over free keys: a
key space restricted to a learned rotation of an orthonormal basis cannot
collapse to a single centre BY CONSTRUCTION — spread is an invariant of the
family, not a training outcome — at the cost of address expressivity. The
free-key form's price (a receipted degenerate basin) and the constrained
form's price (bounded expressivity) are now a REGISTRATION DECISION WITH
RECEIPTS ON BOTH SIDES; the probe measures the free-key form and reports the
basin wherever it occurs, per-seed, in every aggregate.

**Needs a separator (not claimed here):**

* *The Theta(r x run) mass claim as a theorem.* The mass arithmetic
  (`r` rows x `avg-run-length` writes per row) is a counting statement about
  the coalescing semantics; turning it into a capacity lower bound against
  all pointwise strategies needs a task-level adversary argument (the
  collision lemma is the mechanism, not the bound). The SPAN-RUN probe is the
  empirical witness; a formal statement is future separator work.
* *That trained models place same-role contexts contiguously.* This is
  exactly the Wheeler-ness diagnostic — a measurable property of trained
  weights, decided by the run, never assumed.

## 9. Pointers

* The probe: [`wheeler_capacity_probe.py`](wheeler_capacity_probe.py)
  (task family SPAN-RUN, both arms, sweep, receipts);
  tests: [`test_wheeler_capacity_probe.py`](test_wheeler_capacity_probe.py).
* The pre-registered predictions and the verdict:
  [`RESULT_WHEELER_CAPACITY.md`](RESULT_WHEELER_CAPACITY.md).
* The static-index theory this unifies with:
  `x43_fork_resolved/DESIGN_SUFFIX_ATTENTION.md`,
  `x43_fork_resolved/SPINE_DOCUMENT_IDENTITY.md`. The truncation law:
  `formal/lean/OPERATOR_ALGEBRA_STATUS.md`. The growth machinery this
  composes with: `K8_GROWTH.md`. The base arms: `keys.py`,
  `PREREGISTRATION_MATRIX_EMENDER.md` (in `x43_fork_resolved/`),
  `key_separation_probe.py` (SEP-KV, read-only).
