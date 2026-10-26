# Read-only reference provenance

The naive implementation and routing-trace definitions were copied verbatim,
not imported from the live training package. Their SHA-256 sums are identical
to the source files:

- `naive_reference.py` / X28E `eng/tape_head.py`:
  `68fe25baa015031df86ee810beb2b41474b89612f74d3d51519589bd46d4e119`
- `routing_trace.py` / X28E `eng/routing_trace.py`:
  `d741aaa94c6ea82de60a166090c7fdb14d020497861fa8e24e1c9fc2bf341a8d`
- X28E `REGISTRATION.md`, read through F12/F12g/F12h:
  `50963a4c3002de5770b56cf16126d44a3c749bd5aed23b8063c2a7d4eaa45cd2`
- X28E `generators/stage1_mqar.py` (read-only import):
  `83396bf12828d0a098d895e62b75e0021ce2ad087faf5f4105538e89add83417`
- X28E `generators/stage3_ladder.py` (read-only import):
  `c20b589d8315d5c7d4d011ea12903af78667692009083acb84e3b3b0b2705a63`

Sorted path-and-content hash of all X28E package files, excluding pre-existing
`__pycache__` files, observed before implementation measurements:
`233117bca17483a4e7836a784801b74d07659e7880589e011ddfee036bf19bf4`.
The final report records the corresponding end check. All commands set
`PYTHONDONTWRITEBYTECODE=1`; imports cannot create cache files in the freeze.

Actual executable semantics resolve two ambiguities in the task shorthand:

1. POP multiplies strength by `(1-pop_strength)` and kills only on threshold
   crossing. Unconditional binary kill would fail native parity and remove
   the pop-magnitude gradient. See the copied oracle's POP block.
2. The certified trainer freezes the stage embedding (its F12f explanation).
   `WordModel` follows that behavior. Learned embeddings with detached W_k
   would still move the addressing geometry between optimizer updates.

Neither ambiguity was silently interpreted as a change to X28E. No supervisor
contact tool was present in this session's tool catalog, so the alternatives
are explicitly documented instead of attributing approval to a supervisor.
