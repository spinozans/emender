# Abstract Synthesis — "Emending Nonlinear Recurrence"

One proposed abstract, synthesized from the six candidates and the reporter's
analysis, with the author's settled positioning decisions applied. For the
author to review before anything is spliced into `main.typ`. `main.typ` is not
touched here.

## Gap closed

The reporter found all six candidates agreed on the existence proof, the
held-out tie, the two-axis framing, and the Lean one-step separation, with no
can/can't error and full style compliance. Three gaps remained. First, the
deployed-1.3B / S5 result was under-stated everywhere: every candidate carried
length-generalization scale-agnostically, none named S5 (only #5 surfaced a
bare "S5") and none pinned the converged-ceiling / under-reach contrast to the
actual released production weights (`tab_s5_1p3b`). Second, the candidates split
their weight across affordability, the racer, and held-out loss rather than
placing it where the author wants it. This synthesis fixes both: it opens on
purity as the existence proof (pure nonlinear-in-time recurrence reaching
competitive scale at all was the slot the field assumed intractable, the slot
the concurrent M2RNN authors declined by going hybrid-with-attention), states
bulk loss only as a control, and makes the named, deployed-1.3B S5 separation
load-bearing across both axes: linear collapses to a converged ceiling
(computability), the delta update length-generalizes far past its training
length while raw-write under-reaches (efficiency, not can/can't). It rejects the
GPT-2 / named-model anchor and the exact bpb number, and cites the Lean one-step
function-class separation. It borrows #1's compression and "able in principle"
safeguard and #2/#3's clean computability-vs-efficiency labeling.

## Proposed abstract

Pure nonlinear-in-time recurrence, in which the state passes through a
nonlinearity at every step and so forecloses the time-axis parallel scan that
linear recurrences rely on for throughput, was the slot assumed intractable at
competitive scale; a concurrent attempt chose hybridisation with attention
instead. We fill it, attention-free, made trainable by width-axis
multi-programming: hundreds of small recurrent programs per token, each running
its time loop serially. The model reaches below one bit per byte on The Pile on
one workstation-class GPU, but bulk loss is a control: on held-out bytes the
delta-correcting, raw-write, and linear-recurrent models are a statistical tie.
They separate on two axes loss is blind to, both shown on the deployed 1.3B
weights fine-tuned on the S5 word problem. Computability: the linear recurrence
provably cannot track non-solvable S5 state at length, a converged ceiling, not
undertraining. Learning efficiency at matched budget: the delta-correcting
update length-generalizes S5 far past its training length while the raw-write
update, able in principle, under-reaches. A Lean 4 trusted core proves the delta
update reaches a strictly larger one-step function class than raw-write at
matched compute.
