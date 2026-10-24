import ElmanProofs.Expressivity.TapeMemory

-- X28T axiom audit: every delivered theorem must show ONLY
-- propext, Classical.choice, Quot.sound (or a subset).
#print axioms TapeMemory.tapePhase_card
#print axioms TapeMemory.replicate_true_cons
#print axioms TapeMemory.fold_pushTape
#print axioms TapeMemory.stateAfter_push
#print axioms TapeMemory.fold_garbage
#print axioms TapeMemory.fold_popping_sees_a
#print axioms TapeMemory.step_pop_nil
#print axioms TapeMemory.step_pop_cons
#print axioms TapeMemory.fold_popChain
#print axioms TapeMemory.fold_trichotomy
#print axioms TapeMemory.tapeCount_slots
#print axioms TapeMemory.tapeCount_exceeds_bound
#print axioms TapeMemory.tapeCount_twin_countModel
#print axioms TapeMemory.stateAfter_balanced
#print axioms TapeMemory.tapeModel_accepts_balanced
#print axioms TapeMemory.takeWhile_balanced
#print axioms TapeMemory.dropWhile_push
#print axioms TapeMemory.takeWhile_replicate_false
#print axioms TapeMemory.dropWhile_replicate_false
#print axioms TapeMemory.isBalancedAB_iff
#print axioms TapeMemory.tapeModel_decides_balanced
#print axioms TapeMemory.tapeModel_accepts_iff
#print axioms TapeMemory.tapeModel_rejects_leading_b
#print axioms TapeMemory.tapeModel_rejects_count_mismatch
#print axioms TapeMemory.tapeModel_rejects_trailing_a
#print axioms TapeMemory.tapeModel_cells_unary
#print axioms TapeMemory.tapeModel_separates_balanced
#print axioms TapeMemory.tapeModel_no_finite_abstraction
#print axioms TapeMemory.tape_memory_pair
