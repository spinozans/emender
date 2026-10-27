# Remaining X36 work — part 1 is not complete

1. **Persistent forward integration with exact training parity.** Reuse X35's
   device-token-loop core and two causal chunk-read launches in an X36 wrapper.
   Preserve logical S reduction layout in reverse replay. X35's compact backing
   must not silently change gradient roundoff. Require unchanged 109 oracle
   tests plus exact X31-vs-X36 LM gradients and matched loss curve before timing.
   Receipt core/read/materialization/rebase launch counts separately, <=10 lane
   launches including setup/copies as explicitly scoped. Current 1536 token
   launches fail. No claim of a new persistent implementation in this part.
2. **Replay memory layout.** The landed live masks remove dead-row adjoint
   bandwidth. Remaining scalar DK/DV loops are strided and still expensive.
   Tile/coalesce with the original accumulation order. Prefix reduction was
   rejected because gradients changed: retain the failed receipt. Reverse
   replay is already one kernel/chunk; reducing launches cannot alone fix it.
3. **Complete attribution.** Independently calibrate host launch overhead and
   capture-only kernel cost; current journal column is an ablation estimate
   and may be negative under timing noise. Split read_out backward/optimizer
   from remainder. Keep additive wall table and clearly separate overlapping
   CPU API, GPU busy and launch-gap measurements.
4. **Zero-sync training.** Eval-only tape telemetry is landed. X34's original
   clip_grad_norm_(error_if_nonfinite=True) and optimizer still synchronize.
   Replace loop diagnostics with device flags consumed outside steady steps;
   verify finite-gradient behavior and optimizer parity. Profile the entire
   training loop; zero lane telemetry does not establish global zero syncs.
5. **T4096 and asymptotics.** Complete B64/T4096 ratio <=2; memory-safe LM loss
   projection may be needed on a 48GB card. Capacity sweep alone cannot establish
   lack of sequential T dependence: measure multiple T, report live occupancy,
   and distinguish reachable-prefix bounds from full-store attention work.
6. **Targets.** B64/T512 tape/base <=3 and <=10 launches remain gates, never
   waived by a large speedup against a slow tape baseline. No further scientific
   LM pilot/report work is authorized by this engineering slice.
