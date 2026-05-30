# E97 Split Erase/Right-Direction Witness Blueprint

Task: `formal-design-split`

Date: 2026-05-30

Status: design artifact only. The Lean names below are target statements for
`lean-proof-e97` and the later trust-gate review. This file does not claim that
the new 2D witness or E88 collapse theorem has already been proved.

## Goal

The existing E97 formal surface proves algebraic equality of direct and expanded
split-gated forms, all-one-gate specialization to E88, and a 1 x 1 write-gate
witness. The next target should distinguish the part of split control that the
1 x 1 witness cannot see: in two key dimensions, E97/GDN-2-style split control
can use one key-axis direction for the rank-one write and a different key-axis
direction for erase/read. E88's all-one/coupled delta transition can only use a
self-outer-product direction.

In this note, `writeDir` means the key-axis row/write direction `k` in the
transition factor `k eraseDir^T`, not the value-axis payload `w*v`. The
value-axis write gate remains part of the E97/GDN-2 spec, but the finite witness
sets `v = 0` so the theorem isolates the transition-direction factorization.

Scope this as a one-step transition/factorization result. It is not a theorem
about learned semantics, training speed, loss, BPB, optimizer behavior, kernel
throughput, or empirical task superiority.

## Existing Surface To Extend

Extend, do not duplicate, these existing files:

- `formal/lean/ElmanProofs/Architectures/SplitGatedDelta.lean`
  - already defines `hadamard`, `onesVec`, `e97ReadKey`, `e97WriteValue`,
    `splitGatedTransition`, `e97UpdateExpanded`, `gdn2LinearCore`, and the
    all-one E97-to-E88 specialization theorems.
  - add the 2D split-direction witness definitions and algebraic target
    theorems here.
- `formal/lean/ElmanProofs/Architectures/M2RNNComparison.lean`
  - already defines `Vec`, `MatState`, `outerKV`, `matrixTanh`,
    `e88DeltaTransition`, and `e88DeltaUpdateExpanded`.
  - use these definitions; avoid a second vector or matrix formalism.
- `formal/lean/ElmanProofs/Architectures/RecurrentResourceFormalism.lean`
  - already has a "Concrete One-Step Transition Separation" section with
    `TwoVec`, `TwoMat`, and 2D witness style for E88 versus fixed-transition
    M2RNN.
  - the split-direction theorem can follow the same proof style, but should live
    in `SplitGatedDelta.lean` unless moving shared 2D helpers is cleaner.
- After Lean proof and trust-gate review only, update `PaperCore.lean`,
  `TRUSTED_PROOF_SURFACE.md`, `E97_GDN2_FORMALISM_FINDINGS.md`, and
  `PROOF_INVENTORY.md` to include the new theorem surface.

## Weakest Assumptions

The intended theorem should avoid unnecessary analytic or empirical assumptions:

- Key dimension `K = 2` is enough.
- Work at the expanded transition/preactivation level:
  `lambda I - writeDir eraseDir^T`.
- No unit-norm assumption is needed.
- No positivity assumption is needed for the impossibility direction.
- Gate bounds are not needed, but the concrete witness uses a 0/1 erase gate.
- To realize an arbitrary erase direction through E97's pointwise gate, require
  only `eraseDir = hadamard b writeDir`. Equivalently, every nonzero coordinate
  of `eraseDir` must lie in the support of `writeDir`. The numerical witness
  satisfies this directly.
- Compare transition factors, or equivalently compare updates uniformly over
  states with the additive value write fixed to zero. Do not claim that one
  selected output matrix cannot be matched if E88 is also allowed an arbitrary
  value write term to absorb the difference.
- Lifting a preactivation equality through the E97/E88 nonlinear update may use
  the existing `Activation.tanh_injective`, as in the current one-step
  transition separation proofs.

## Lean-Friendly Helper Targets

These names are suggestions, not current proved theorems.

```lean
namespace SplitGatedDelta

abbrev TwoVec := M2RNNComparison.Vec 2
abbrev TwoMat := Matrix (Fin 2) (Fin 2) Real

/-- Explicit split transition from separate write and erase/read directions. -/
def splitTransitionFromDirs
    (lambda : Real) (writeDir eraseDir : TwoVec) : TwoMat
-- body target: lambda-smul identity minus
-- M2RNNComparison.outerKV writeDir eraseDir

/-- In 2D, "parallel" means zero determinant. -/
def parallel2 (a b : TwoVec) : Prop :=
  a 0 * b 1 = a 1 * b 0

theorem splitGatedTransition_eq_splitTransitionFromDirs
    (lambda : Real) (k b : TwoVec) :
    splitGatedTransition (K := 2) lambda k b =
      splitTransitionFromDirs lambda k (hadamard b k)

theorem e88_coupled_transition_forces_parallel_split_dirs
    (lambdaSplit lambdaE88 : Real) (writeDir eraseDir p : TwoVec) :
    splitTransitionFromDirs lambdaSplit writeDir eraseDir =
      M2RNNComparison.e88DeltaTransition (K := 2) lambdaE88 p ->
    parallel2 writeDir eraseDir

theorem e88_cannot_realize_nonparallel_split_transition
    (lambdaSplit : Real) (writeDir eraseDir : TwoVec)
    (hnot : Not (parallel2 writeDir eraseDir)) :
    Not (Exists fun lambdaE88 : Real =>
      Exists fun p : TwoVec =>
        splitTransitionFromDirs lambdaSplit writeDir eraseDir =
          M2RNNComparison.e88DeltaTransition (K := 2) lambdaE88 p)

end SplitGatedDelta
```

Interpretation: E88's coupled transition is `lambdaE88 I - p p^T`; its
off-diagonal entries are necessarily symmetric. A split transition
`lambdaSplit I - u r^T` can have asymmetric off-diagonal entries. Equality with
an E88 transition forces `u 0 * r 1 = u 1 * r 0`, so the split write direction
`u` and erase/read direction `r` are parallel. The all-one E88 collapse is the
special case `r = u`, already covered by the existing E97-to-E88 specialization.

Parallel is a necessary escape hatch, not a full sufficiency theorem. If
`r = c u`, E88 still needs scalar/sign compatibility to represent the same
rank-one factor as `p p^T`. The downstream Lean target only needs the necessary
condition for the nonparallel witness.

## Concrete 2D Witness

Use key dimension 2 and value dimension 2 for the finite update witness. Let:

```text
writeDir u       = (1, 1)
eraseGate b      = (1, 0)
eraseDir r=b*u   = (1, 0)
writeGate w      = (1, 1)       -- irrelevant when v = 0
value v          = (0, 0)
lambda           = 1
state H          = I_2
```

Then `u` and `r` are not parallel:

```text
u0*r1 = 1*0 = 0
u1*r0 = 1*1 = 1
```

E97 realizes the split transition because `r = hadamard b u`:

```text
splitTransitionFromDirs(1, u, r)
  = I - u r^T
  = [[0, 0],
     [-1, 1]]
```

At `H = I_2` and `v = 0`, the expanded E97 preactivation is exactly this
matrix, and the nonlinear E97 state is the elementwise `tanh` of it:

```text
e97 preactivation = [[0, 0],
                     [-1, 1]]

e97 update        = [[0, 0],
                     [tanh(-1), tanh(1)]]
```

Any E88 coupled transition in 2D has the form, for `p = (a, c)`:

```text
e88DeltaTransition(mu, p)
  = mu I - p p^T
  = [[mu - a*a, -a*c],
     [-a*c,     mu - c*c]]
```

The off-diagonal entries are equal. The split witness has off-diagonal entries
`0` and `-1`, so no `mu`, `a`, and `c` can match it.

Recommended target names for the finite witness:

```lean
def splitWitnessWriteDir : TwoVec := fun _ => 1
def splitWitnessEraseGate : TwoVec := fun i => if i = 0 then 1 else 0
def splitWitnessEraseDir : TwoVec :=
  hadamard splitWitnessEraseGate splitWitnessWriteDir
def splitWitnessValue : TwoVec := 0

-- Same split transition also describes the GDN-2-style linear core on
-- identity decay / decayed state; E97 then applies matrixTanh.
theorem splitWitness_dirs_not_parallel :
    Not (parallel2 splitWitnessWriteDir splitWitnessEraseDir)

theorem e97_realizes_splitWitness_transition :
    splitGatedTransition (K := 2) 1
        splitWitnessWriteDir splitWitnessEraseGate =
      splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir

theorem splitWitness_transition_entries :
    And
      (splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir 0 0 = 0)
      (And
        (splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir 0 1 = 0)
        (And
          (splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir 1 0 = -1)
          (splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir 1 1 = 1)))

theorem e88_cannot_realize_splitWitness_transition :
    Not (Exists fun lambdaE88 : Real =>
      Exists fun p : TwoVec =>
        splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir =
          M2RNNComparison.e88DeltaTransition (K := 2) lambdaE88 p)
```

Optional nonlinear-update lift, if `lean-proof-e97` wants the witness stated as
a one-step update rather than only a transition factor:

```lean
theorem e97_splitWitness_update_at_identity_zero :
    e97UpdateExpanded (K := 2) (V := 2) 1 (1 : TwoMat)
        splitWitnessWriteDir splitWitnessEraseGate
        (onesVec 2) (0 : TwoVec) =
      M2RNNComparison.matrixTanh
        (splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir)
```

If a matching E88 nonlinear-update impossibility is added, state it as a
uniform-over-states or zero-write transition theorem so the result remains a
factorization separation. Avoid the stronger and false-looking claim that E88
cannot match one chosen output matrix when its value write term is unconstrained.

## What This Would Prove

If implemented and trust-gate verified, the target would prove:

- E97's pointwise split erase gate can realize a one-step transition with
  nonparallel write and erase/read directions in 2D.
- E88's all-one/coupled delta transition cannot realize that nonparallel
  split-direction transition; equality with an E88 transition forces the two
  split directions to be parallel, with exact all-one collapse as `eraseDir =
  writeDir`.
- The witness is finite and explicit: `u=(1,1)`, `b=(1,0)`, `r=(1,0)`, `H=I_2`,
  `v=0`, `lambda=1`.

It would not prove:

- E97 has greater semantic capacity in general.
- E97 learns tasks that E88 cannot learn.
- E97 or GDN-2 is empirically better, faster, or lower loss.
- Any statement about full language-model blocks, residual wrappers, optimizer
  behavior, gate-generation networks, kernels, or training dynamics.
