# C14-B1 cross-arm occupancy and compute receipt

Training receipts only; sampled live = array + stream, equally weighted across retained samples, not a whole-training time average. Full per-head sampled traces and source SHA256 bindings are in `receipts/c14_b_v2_unbudgeted/occupancy_scaling.json.gz`. PB3 budgeted telemetry is retained unchanged.

| Arm | Steps | Targets | Active seconds | Steps/s | ms/step | Per-head peak live (0/0,0/1,1/0,1/1) | Per-head sampled mean live |
|---|---:|---:|---:|---:|---:|---|---|
| c14_b_v2 | 35419 | 71677042 | 2700.032370 | 13.117991 | 76.231 | 32/64/32/64 | 11.565 / 12.817 / 11.223 / 10.521 |
| c14_b_v2_append_only | 37254 | 75394452 | 2700.045113 | 13.797547 | 72.477 | 32/64/32/64 | 25.288 / 36.009 / 25.088 / 35.142 |
| c14_b_v2_unbudgeted | 7142 | 14455380 | 2700.288256 | 2.644903 | 378.086 | 81/110/81/96 | 11.565 / 12.817 / 11.231 / 10.527 |

Unbudgeted step cost is 4.9597x budgeted and 5.2167x append-only; target exposure is respectively 4.9585x and 5.2157x lower. Budgeted and append-only share 32/64 caps; unbudgeted cap 257 never binds and all four heads have zero capacity merges.

The physical budget is a compute-envelope control as well as a curation signal: removing it accompanies roughly fivefold step cost and fivefold less exposure at the same 2700-second active budget. Natural churn permits peaks up to 110 live slots. The receipts establish 110 as a maximum, not a stationary mean or a demonstrated steady state; each independent document-bounded window resets state.

The budgeted and unbudgeted sampled means are nearly identical (early retained traces); they do not show a fivefold live-occupancy increase. These three arm-level points show a compute penalty for removing the cap, but cannot identify a linear/power scaling law or isolate live occupancy from allocated capacity (257 versus 32/64), lifecycle operations, and training trajectory. Append-only is slightly faster despite its occupancy; occupancy alone is not a sufficient predictor. No per-head timing is measured. Single seed and unequal exposure preclude an isolated causal quality claim.

For the program’s unbounded-state thesis, the receipted 81/110/81/96 peaks without capacity merges and ~5x step cost show why allowing more retained state must be evaluated together with its compute envelope.

Manual notes: contact_supervisor is absent from the enabled tool catalog; no supervisor transport available. No WorksGood service used. No DiLoCo behavior involved; requirement IDs inapplicable.
