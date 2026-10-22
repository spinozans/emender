# X20 terminal design finding: NO LICENSED CONSTRUCTION

The tracked theory licenses no expressivity separation between the experimental
architecture pair: E97 with `state_activation="tanh"` versus linear-state
`fla-gdn` with `allow_neg_eigval=True`, including the requested wrapped
32×32×4 shape. This is an absence of a proved separator in these sources,
**not** a proof of equal capacity, universal wrapper compensation, or a training
result. No candidate was selected, no training was launched, and no cards were
consumed by X20.

Authority: `/home/erikg/emender-scratch/fix-fanout/taskX20-separationdesign.md`,
followed by two live supervisor decisions on 2026-10-22. The final decision
explicitly approved this terminal finding and prohibited launch without a
citable wrapper-proof. Source hashes and the base commit are in
`receipts/terminal.json`; registration and the four inactive outcome categories
are in `registration.json`.

## What the affine-fold theorem actually proves

`CompositionalDepthDuality.gdnStep_eq_affine`
(`formal/lean/ElmanProofs/Expressivity/CompositionalDepthDuality.lean:177–181`)
writes a single GDN core step as

```
H' = L(p) H + B(p)
L(p) = splitGatedTransition(1, k, b) decay
B(p) = outerKV(k, e97WriteValue(w, v))
```

For **every fixed finite parameter schedule** `steps`,
`gdn_state_channel_depth_bounded_in_time` (lines 188–228) proves
`∃ A, C, ∀ init, gdnStateFold steps init = A init + C`.
The quantifier order matters: the schedule, and therefore `A,C`, are fixed
before the initial state varies. The companion
`gdn_state_channel_affine_identity` (lines 230–242) proves
`F(H1+H2)+F(0)=F(H1)+F(H2)` for this raw state map.

An arbitrary nonlinear **input-only** encoder may produce the schedule; the
conditional affine characterization still applies if that schedule is
independent of the varied initial state. It does **not** say the token-word
function is affine, or that `decode(A encode(init)+C)` is affine under arbitrary
nonlinear encoders/decoders. A nonlinear decoder invalidates the raw-state
second-difference inference. If schedule generation reads the varied state,
its state-independence premise is lost. A stacked production network also has
nonlinear interlayer processing of history; the theorem does not characterize
the whole stack as one fixed-schedule affine map of all its states.

For E97, `e97Step_eq_tanh_comp` (lines 77–83),
`e97StateAfter_eq_compose` (lines 96–106), and
`e97_compositional_depth_grows_in_time` (lines 118–130) describe the executed
composition: one more `tanh`-bearing step is appended per token. They do not
prove that the resulting input/output function has no shallower realization,
or a width/depth lower bound against nonlinear wrappers. The paper states this
conditional structural distinction at `paper/main.typ:1486–1503`.

## Enumeration and wrapper audit of the tracked constructions

1. **Temporal composition / affine-state identity.** The theorems above separate
   the *form of the raw state path*, not the representable wrapped task classes.
   A literal scalar specialization is `K=V=1`, `lambda=1`, zero key, `T=1`:
   E97 maps `h` to `tanh(h)`. No scalar affine map agrees at all of
   `{0, 1/2, 1}`, since `tanh(1)-2 tanh(1/2)+tanh(0) != 0`.
   Identity state propagation followed by a `tanh` decoder computes exactly the
   same one-step output. This is a bare continuous-state structural regression,
   not a token/exact-match temporal separator; it is **rejected**, not promoted
   to an auxiliary separation experiment. This specialization and counterexample
   are elementary deductions from the definitions, not a new Lean theorem.

2. **S5 prefix-product tracking and routing.**
   `paper/main.typ:350–363` explicitly constructs a 5×5 permutation-matrix
   linear-state tracker: `R0=I5`, `Rt=R(t-1) P(token)`. Right multiplication
   is linear after vectorization and tracks noncommuting products exactly.
   Thus noncommutativity and NC1-completeness do not rule out linear state.
   `S5GDNBridge.gdn_s5_realization_requires_per_token_action` is a necessary
   routing condition under an all-words decoder plus injectivity on reachable
   states, **not** a nonrealizability theorem. Its module disclaimer
   (`formal/lean/ElmanProofs/Expressivity/S5GDNBridge.lean:49–80`) explicitly
   disclaims depth/size lower bounds. Only **token-independent** schedules are
   excluded by `gdn_s5_constant_parameters_cannot_realize`; the experimental
   arm has token-dependent projections. The general linear realization fits
   well below 32×32 state, but is not asserted here to satisfy production GDN
   gate/decay restrictions. That exact constrained realization/obstruction is
   open G3 (`paper/main.typ:2008–2016`, `formal/lean/GAPS-D2.md`). Neither a
   bare-GDN nor a wrapped-GDN separation follows.

3. **Finite-state capacity and lookup/generator memory.**
   `S5Inseparability.capacity_lower_bound_120` and
   `small_recognizer_cannot_realize_s5` require at least 120 distinguishable
   states in *any* exact recognizer, regardless of architecture or decoder.
   A 32×32 state is not shown to have fewer than 120 states; the lower bound
   distinguishes neither arm. `S5NDMRealization`'s 480 state/input-key table
   and `NDMRealizesS5.emender_realizes_s5_tracker`'s `d=12` preloaded generator
   memory are not a separation from GDN. In the latter, group multiplication is
   **external**, with the current group state supplied; it is not a recurrent
   E97 all-words realization (`paper/main.typ:1530–1583`). O1 remains open
   (`paper/main.typ:2004–2006`). A nonlinear wrapper is not excluded by any
   of these results.

4. **Delta versus fixed-right/raw-write resource separation.**
   `RecurrentResourceFormalism.emender_m2rnn_one_step_resource_separation`
   (`formal/lean/ElmanProofs/Architectures/RecurrentResourceFormalism.lean:940–963`)
   excludes the specific `FixedRightRawExternalForget2` resource. Its embedded
   version (lines 1127–1150) applies for `K>=2,V>=1`.
   `MultiStepSeparation.emender_m2rnn_k_step_separation`
   (`formal/lean/ElmanProofs/Architectures/MultiStepSeparation.lean:447–472`)
   extends the 2×2 witness to every finite `k>=1`: raw-write row-zero
   preservation conflicts with NDM's nonzero `tanhIter k (-1)` entry.
   These are genuine exact resource separations, but the excluded class has
   a fixed right transition, raw write, and external row/column/cell forget;
   **it is not fla-gdn's input-dependent left-transition delta core**.
   They neither exclude arbitrary wrappers nor compare the requested pair.

5. **Conditional saturation / hold / release.**
   `EmenderLatching.emender_saturation_insensitivity`,
   `emender_latch_holds_by_default`, and
   `emender_latch_releases_on_counter_delta` give sufficient slot-level tanh
   bounds (`formal/lean/ElmanProofs/Architectures/EmenderLatching.lean:94–105`,
   `157–167`, `268–275`; `paper/main.typ:1634–1679`). Hold assumes
   `lambda>=1` and an adequate write-budget margin. The paper explicitly
   notes production decay below one as an outstanding applicability gap
   (`paper/main.typ:2027–2031`). No theorem here proves that GDN with a decoder
   cannot perform the same discrete hold/release task. These are not capacity
   impossibility results for either bare or wrapped GDN.

## Selection and launch disposition

**Selected construction: none. Wrapper-proof: unavailable.** No tracked
construction establishes a task that requires nonlinearity in the accumulated
state transition despite every allowed choice of nonlinear input/output
wrappers at the requested dimensions. The structural theorem survives an
input-only nonlinear schedule generator, but its *raw-state* affine identity
does not survive arbitrary nonlinear decoding as an output constraint.

Neither primary arms nor bare-model auxiliary arms were launched. No optimizer,
staging, scaffold, task length, vocabulary, exact-match threshold, annealing
schedule, watcher, native supervisor arming, or censoring process was activated.
Without a valid target these would create an unlicensed experiment, not satisfy
the original design gate. There is no result to censor. The four requested
outcome names are retained in registration, all unassigned; numerical ceiling
criteria are deliberately unset rather than fabricated for a nonexistent task.
A future experiment needs a separately reviewed construction/proof and complete
pre-registration before any data collection.

## Corrections

- **Concession #16 (supervisor correction):** the task's “triangle-map / 3g1”
  names were cited from memory and do not name constructions in the tracked
  theory. They are retracted as source claims; neither is a selected candidate.
- **Subsequent S5 correction:** the supervisor's suggestion that noncommutative
  S5 accumulation is impossible for bounded-dimensional linear-state folds
  contradicts the tracked paper's explicit 5×5 linear realization. That
  suggestion is also retracted, including as a proposed bare-model separator.
- “No separation is proven here” is not “no separation exists.” Empirical ties
  are compatible with this finding but do not prove representational equality.

## Validation and limits

The receipt records source hashes, host/base commit, zero submitted runs, and
CPU-only checks of the scalar wrapper counterexample and all 480 S5
state/generator transitions. Those checks illustrate the source reading; they
are not new formal proofs or training results. No Lean sources, model code,
existing training records, or project instructions were changed. Lean was not
rebuilt: this is a source/quantifier audit, not a refreshed trust-gate claim.
Resilient DiLoCo and Frontier scheduler requirements are not applicable: no
resilient behavior or scheduler submission was changed or executed.
