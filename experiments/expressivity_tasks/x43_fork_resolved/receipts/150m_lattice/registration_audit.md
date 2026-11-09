# 150M lattice registration audit — 2026-11-09

Registration-only review: seven arms, same flagship stream/cache, 2700-active-second
stopping rule, actual exposure columns, four claims with null allowance, transfer/
stream/seed/budget caveats, two firing gates, zero contention and pre-flagship
sign-off requirements are explicit. No training or service command was run.
No changes were made to ~/sxgc, /mnt/nvme2n1 or live services.

Registration bytes were committed first (12fd6fbb) and then pinned by full commit
and SHA256 under C14, following the existing 1P3B convention. This does not
promote C-A through C-D to measured outcomes. git diff --check passed.

The initial read-only validator invocation against the alternate existing
/home/erikg/sxgc-byte-fixes checkout failed: missing xsa/src/branches.inc.rs.
This failure is retained; it was not a successful validation. To avoid reading
or modifying the protected ~/sxgc working tree, the three historical upstream
bindings were fetched from that alternate repository into an isolated temporary
git repository and materialized from their exact pinned commits. The full,
unmodified validator then passed with --sxgc-repo pointing at that audit copy:
17 claims, 464 bindings, 10 measured, 4 refuted, 3 registered. Exact JSON output
is retained in program_validation.json. This audits provenance, not scientific
entailment, training feasibility or extraction completion.

Residual risks: byte nominations and extraction completion remain future gates;
exact nominated configs/capacity and executable qualification must be receipted
before launch. A single seed and stream/geometry differences limit cross-scale
inference. Common-exposure checkpoints may be unavailable within 2700 seconds;
that absence must remain explicit. No live readiness check was attempted.
