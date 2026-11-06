# X43 P100 out-of-index evaluation — coverage versus transfer

Frozen-arm evaluation completed 4096 windows /515134 raw masked targets from 170 documents fully beyond byte100e9.0 training steps; no gate calibration or weight updates.

Every selected full-document SHA256 was checked against all frozen offset identities; repeated content was excluded. Selection hashes and spans are in selection.json. Independent17-query full100GB brute-force verification passed before selection; all256 branch counts and EOF were compared, not merely interval closure.

| arm | out-of-index NLL |
|---|---:|
|emender_dense_off|2.316299673075341|
|emender_consulted_on|1.279200776268499|
|emender_consulted_off|2.352360810212475|
|transformer_dense_off|1.4619428990668006|
|transformer_consulted_on|1.1488696351668612|
|transformer_consulted_off|1.519654393938848|
|unbounded_on|1.3142320577700763|
|unbounded_off|2.432560937658321|
|array_alone|Infinity|
|emender_dense_stacked_soft|1.3928217627532276|
|emender_dense_stacked_literal|Infinity|
|emender_consulted_stacked_soft|1.2464771875752192|
|emender_consulted_stacked_literal|Infinity|
|transformer_dense_stacked_soft|1.1876237218980579|
|transformer_dense_stacked_literal|Infinity|
|transformer_consulted_stacked_soft|1.123847505701048|
|transformer_consulted_stacked_literal|Infinity|

| family | paired transfer gain (off minus on, nats/byte) |
|---|---:|
|emender_consulted|1.073160034|
|transformer_consulted|0.370784759|
|unbounded|1.118328880|
|emender_stacked|1.105883623|
|transformer_stacked|0.395806888|

Paired in-region comparisons, where existing off tables permit, are retained in decomposition.json. The preregistered positive25–75% survival prediction is descriptive; report the measured fractions without significance claims. Array-alone contradictions retain infinite loss; soft stacked numbers correspond to1.5059/1.2636, while literal forced-walk variants remain separate. Support-length strata and zero-target counts are in eval.json.

The frozen in-region features were from Wikipedia, while this experiment queries Pile-head100. Excluding source spans removes direct inclusion of these document occurrences; repeated substrings can remain indexed. Consequently the in-region minus out-of-index gain is an operational difference, not an identified pure coverage component. New documents and index composition change together. A clean causal comparison requires matched index inclusion/exclusion on the same heldout documents, beyond this experiment.

Residual risks: one frozen seed per arm; source-order prefix of beyond-boundary candidates;4096 windows is much smaller than frozen full evaluation; no confidence intervals; sampled query verification is not exhaustive SA/BWT certification. Frozen weights/gates and source bindings were checked. No DiLoCo behavior changed; gap IDs inapplicable.
