# X15G growth-report addendum — X15b amendment-2 checkpoint restart

RUNNING muon3x_seed0
Provenance: restart-from-checkpoint; historical433001–433296 superseded by re-execution.
Exact checkpoint model/optimizer/RNG restoration; NO at-death-restoration claim. Originals preserved.
Concession #9: original 1M ceiling/cosine retained; task text incorrectly said 500K.
Observed through 433500; fixed-final1M complete: False.
Any-position >=50/512 escape crossing: none observed (right-censored until completion).
Latest material positions: [].
Latest CE 8.158906043; CE-ln3721=-0.062841686.
Near-marginal CE is NOT proof of no learning; no post-score equivalence tolerance invented.
Full per-position exact-marginal signature in marginal_signature.json.

Terminal verdict pending/censored; no 1M conclusion.

|Position|Latest /512|Max /512|First>=50|First512|Latest CE|
|---|---|---|---|---|---|
|P1|10|19|censored@433500|censored@433500|4.198784463|
|P2|0|2|censored@433500|censored@433500|8.222066833|
|P3|0|3|censored@433500|censored@433500|8.221747275|
|P4|0|3|censored@433500|censored@433500|8.221136596|
|P5|0|3|censored@433500|censored@433500|8.222946411|
|P6|0|3|censored@433500|censored@433500|8.221720546|
|P7|0|2|censored@433500|censored@433500|8.222286144|
|P8|0|2|censored@433500|censored@433500|8.221591916|
|P9|0|2|censored@433500|censored@433500|8.220854532|
|P10|0|2|censored@433500|censored@433500|8.222186608|
|P11|0|3|censored@433500|censored@433500|8.221616553|
|P12|0|3|censored@433500|censored@433500|8.221471291|
|P13|0|2|censored@433500|censored@433500|8.222183697|
|P14|0|2|censored@433500|censored@433500|8.222425463|
|P15|0|2|censored@433500|censored@433500|8.221938984|
|P16|0|2|censored@433500|censored@433500|8.222127138|
|P17|1|2|censored@433500|censored@433500|8.222546846|
|P18|0|2|censored@433500|censored@433500|8.221925752|
|P19|0|3|censored@433500|censored@433500|8.221796541|
|P20|0|2|censored@433500|censored@433500|8.221020762|
|P21|0|2|censored@433500|censored@433500|8.221330088|
|P22|0|2|censored@433500|censored@433500|8.221513234|
|P23|0|2|censored@433500|censored@433500|8.222348480|
|P24|0|2|censored@433500|censored@433500|8.221795270|
|P25|1|2|censored@433500|censored@433500|8.222305791|
|P26|0|4|censored@433500|censored@433500|8.221689265|
|P27|1|2|censored@433500|censored@433500|8.220760964|
|P28|1|2|censored@433500|censored@433500|8.222426601|
|P29|0|3|censored@433500|censored@433500|8.221416922|
|P30|1|3|censored@433500|censored@433500|8.221700674|
|P31|0|3|censored@433500|censored@433500|8.221441377|
|P32|0|2|censored@433500|censored@433500|8.222067697|
|P33|0|2|censored@433500|censored@433500|8.222146211|
|P34|0|2|censored@433500|censored@433500|8.221625494|
|P35|1|3|censored@433500|censored@433500|8.221338451|
|P36|0|2|censored@433500|censored@433500|8.221346023|
|P37|0|3|censored@433500|censored@433500|8.222068265|
|P38|0|2|censored@433500|censored@433500|8.221173201|
|P39|0|3|censored@433500|censored@433500|8.221686564|
|P40|0|2|censored@433500|censored@433500|8.222079754|
|P41|0|2|censored@433500|censored@433500|8.221667368|
|P42|0|3|censored@433500|censored@433500|8.221408220|
|P43|1|2|censored@433500|censored@433500|8.220885083|
|P44|0|2|censored@433500|censored@433500|8.221266754|
|P45|0|2|censored@433500|censored@433500|8.222444179|
|P46|1|2|censored@433500|censored@433500|8.221946497|
|P47|0|2|censored@433500|censored@433500|8.222440567|
|P48|0|3|censored@433500|censored@433500|8.222215118|
|P49|0|3|censored@433500|censored@433500|8.221368356|
|P50|0|2|censored@433500|censored@433500|8.221689090|
|P51|0|2|censored@433500|censored@433500|8.221901370|
|P52|0|2|censored@433500|censored@433500|8.221510861|
|P53|1|2|censored@433500|censored@433500|8.221369747|
|P54|0|2|censored@433500|censored@433500|8.221219128|
|P55|0|3|censored@433500|censored@433500|8.222372022|
|P56|1|2|censored@433500|censored@433500|8.222052654|
|P57|0|3|censored@433500|censored@433500|8.221294090|
|P58|0|3|censored@433500|censored@433500|8.222101815|
|P59|0|3|censored@433500|censored@433500|8.221445417|
|P60|0|3|censored@433500|censored@433500|8.222135885|
|P61|0|3|censored@433500|censored@433500|8.221287370|
|P62|0|2|censored@433500|censored@433500|8.221825168|
|P63|0|3|censored@433500|censored@433500|8.221945614|
|P64|1|2|censored@433500|censored@433500|8.221559934|

## Training loss trajectory (full per-update records in bound streams)
Original rows after433000 are superseded, not stitched into this branch.
/home/erikg/emender/experiments/expressivity_tasks/results_invertible_stage15_probes_8m/muon3x_seed0/training.jsonl: update1 loss=250.876220703; update433000 loss=8.158800125.
/home/erikg/emender/experiments/expressivity_tasks/results_invertible_stage15_probes_8m/muon_restart/muon3x_seed0/training.jsonl: update433001 loss=8.159132004; update433860 loss=8.160170555.
