TASK X28E - THE BUDGETED-GROWTH TAPE HEAD: the unbounded fork with reduction
(the memory-head engineering leg; DRAFT directive, companion to theory_x28t
and the X21T both-impossible theorems). Attribution discipline: the
operator's EXACT words appear in quotation marks; supervisor condensations
and interpretations are labeled as such and carry no quotation marks. The
operator DELEGATED the detailed design decisions to the supervisor:
"Okay, this is all a bit wild, but I'm happy for you to do these
experiments if you think that they're useful."
The resolutions section of REGISTRATION.md records each delegation.

- CENTRAL HYPOTHESIS. Operator, verbatim: "I think the key idea is that we
  want to be able to escape the trained regime, and states should be able
  to grow to sizes that are appropriate for a given task, even if those
  are outside of the regime to which we've trained. I think that's the key
  concept." SUPERVISOR OPERATIONALIZATION (not an operator quotation):
  states grow to the size the task demands, even outside the training
  regime — tested as depth generalization on the a^k b^k ladder. The
  ladder's asymptotic certificate (exact separation at all depths requires
  no finite total-state abstraction; neither the slot-count breakpoint nor
  finite OOD success is theorem-certified) is in REGISTRATION.md's
  finite-depth scope.

- GROWTH AND REDUCTION ARE BOTH REQUIRED. Operator, verbatim: "I just want
  to make sure that we have both growth and reduction, both thought
  through and feasible. Because if it's only growth, obviously, it's
  intractable. We can't have that." The head therefore ships THREE
  architectural reduction mechanisms - POP BY CONTENT (a read carries a
  pop strength marking the matched slot dead; task-driven shrinkage; the
  true stack op; the per-b popping on the ladder is a registered MECHANISM
  PREDICTION, receipted by traces, not an architectural guarantee),
  USAGE-DECAY (per-slot age/usage counters, decaying strength, free below
  threshold; the cache policy; registered inert within the ladder), MERGE
  UNDER BUDGET (append pressure degrades into a delta-rule merge onto best
  match when the budget cap is hit; at budget the head becomes a bounded
  slot store using delta-rule corrections - GDN2-LIKE as an analogy, not
  an equivalence - graceful degradation). Zero learned management: all
  three reduction ops are architectural; gradient learns only
  keys/values/queries/strengths
  and magnitudes, never routing.

- FIXED-SIZE PREFERENCE MEASURED BY THE BUDGET ARMS. Operator, verbatim:
  "the whole point was to escape an unbounded state, to have a fixed-size
  state as possible, or a small state, a compressed state... if we start
  making indefinite growing memories, we've failed in that regard... a
  fixed-size memory could be big, but that would be good." The budget
  sweep (B in {GDN2-equivalent fixed-size control, 4x, 16x,
  unbounded-with-decay}) measures what a capped head with the merge policy
  buys; the append-only control arm (no pop - predicted to FAIL a^k b^k)
  tests whether pop is load-bearing; native controls (GDN2 and E97 at the
  same sizes) carry the receipted X25/X27 OOD failures as empirical
  regularities (NOT theorem exclusions at finite depth).

- TURING-COMPLETENESS FRAMING. Operator, verbatim: "on this basis, you
  could argue that no computer is actually turning complete. I think
  that's a bit ridiculous." SUPERVISOR INTERPRETATION (not an operator
  quotation): the class follows the resource axis the way any real
  computer's does - no instantiation has the unbounded resource. The
  impossibility theorems bound distinct whole-channel STATE VALUES; growing
  slot count is the head's chosen supply of that resource - the supply
  whose unbounded sufficiency the X28T construction exhibits - and the
  ladder tests the finite-depth form of the growth claim empirically.
  No claim above one level below Turing; no theorem certification of
  finite-depth results.

Engineering: the complete five-case routing rule (R-EMPTY / R-NOMATCH /
R-ZERO-RESIDUAL / R-CONFLICT / R-CAP; append when there is nothing to
correct), fp32 cells; curriculum stage 1 pure associative recall, stage 2
append-necessity tasks, stage 3 the ladder; fused-kernel equivalence gate
vs eager fp32 oracle (worst-case gradient error < 0.08, X24 standard),
BPTT through the growing store via sparse checkpoints + reverse-replay
per ndm/triton/mlp_mem_fused.py WITH the discrete routing trace saved in
checkpoint payloads (replay follows the trace; decisions are never
recomputed from rounded state), fp32 live state / bf16 replay checkpoints
(X21E receipt), 90,904-token numerics smoke, cap-stress smoke, nonfinite
death = CENSORED infra never a training verdict (X27 softplus-small
receipt). Seeds n=1 per cell, right-censoring at the update cap. Card 7
carries the live big-softplus run - leave it alone; cards 0-6 only,
leases at launch time. Two milestones: M1 = registration acceptance
(engineering only); M2 = reviewed, pushed engineering freeze before ANY
curriculum optimizer update. Nothing trains before both.
