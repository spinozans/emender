# Follow-up specialization decision (before implementation)

Streaming v1 improves T129+ but regresses T32..128. Keep reference BS<=256,
streaming BS>=512. This is explicit experimental shape dispatch only.
The first static profile also attributes9224 expanded instructions apiece to
each full-context key-row store. Its mask writes only scalar dst when alloc:
replace the expanded row address/value tensors with a scalar-dst vector store,
leaving the same key bits and alloc mask. This removes another context-sized
intermediate rather than adding global state passes. Run the same ten shapes
and all exact gates on the final selected path. Keep v1 sources and receipts
immutable. No concurrency, promotion, or training is introduced.
