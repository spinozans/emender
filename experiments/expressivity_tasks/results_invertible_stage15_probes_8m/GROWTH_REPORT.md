# X15 probe growth report

Single-seed conditional probes; transients descriptive; fixed-final1M primary for Muon.
Missing crossings right-censored at last observed step, not zero/failure.
Marginal reference CE ln3721=8.221747728; joint chance1/3721; expected0.1376/512 per position.
Near-marginal CE is NOT proof of no learning. Materiality >=50/512; all64 columns carry verdict.


## Muon@3x scratch T64
RUNNING muon3x_seed0
Observed through8000; complete fixed-final1M: False.
|Position|Latest /512|Max /512|First>=50|First512|Latest CE|
|---|---|---|---|---|---|
|P1|10|11|censored@8000|censored@8000|4.739171|
|P2|1|1|censored@8000|censored@8000|8.219529|
|P3|0|1|censored@8000|censored@8000|8.224186|
|P4|1|1|censored@8000|censored@8000|8.221052|
|P5|1|1|censored@8000|censored@8000|8.217414|
|P6|0|1|censored@8000|censored@8000|8.227341|
|P7|0|1|censored@8000|censored@8000|8.226276|
|P8|0|2|censored@8000|censored@8000|8.224617|
|P9|0|1|censored@8000|censored@8000|8.226463|
|P10|0|1|censored@8000|censored@8000|8.226071|
|P11|0|1|censored@8000|censored@8000|8.221900|
|P12|0|1|censored@8000|censored@8000|8.222237|
|P13|0|1|censored@8000|censored@8000|8.222381|
|P14|0|1|censored@8000|censored@8000|8.220905|
|P15|0|1|censored@8000|censored@8000|8.225681|
|P16|0|1|censored@8000|censored@8000|8.215445|
|P17|0|1|censored@8000|censored@8000|8.224902|
|P18|0|1|censored@8000|censored@8000|8.229418|
|P19|0|1|censored@8000|censored@8000|8.221532|
|P20|1|2|censored@8000|censored@8000|8.223039|
|P21|0|2|censored@8000|censored@8000|8.222058|
|P22|0|1|censored@8000|censored@8000|8.220801|
|P23|0|1|censored@8000|censored@8000|8.225095|
|P24|1|1|censored@8000|censored@8000|8.226066|
|P25|1|1|censored@8000|censored@8000|8.227352|
|P26|0|1|censored@8000|censored@8000|8.226004|
|P27|1|2|censored@8000|censored@8000|8.219365|
|P28|0|1|censored@8000|censored@8000|8.227425|
|P29|0|1|censored@8000|censored@8000|8.226268|
|P30|0|1|censored@8000|censored@8000|8.222068|
|P31|0|2|censored@8000|censored@8000|8.224255|
|P32|0|1|censored@8000|censored@8000|8.221549|
|P33|0|1|censored@8000|censored@8000|8.223374|
|P34|0|1|censored@8000|censored@8000|8.223819|
|P35|0|0|censored@8000|censored@8000|8.223007|
|P36|1|1|censored@8000|censored@8000|8.223943|
|P37|0|1|censored@8000|censored@8000|8.222693|
|P38|0|1|censored@8000|censored@8000|8.228297|
|P39|0|1|censored@8000|censored@8000|8.223472|
|P40|0|1|censored@8000|censored@8000|8.226588|
|P41|0|2|censored@8000|censored@8000|8.227707|
|P42|1|1|censored@8000|censored@8000|8.220144|
|P43|0|1|censored@8000|censored@8000|8.223155|
|P44|0|1|censored@8000|censored@8000|8.222410|
|P45|0|1|censored@8000|censored@8000|8.238022|
|P46|1|1|censored@8000|censored@8000|8.220957|
|P47|0|0|censored@8000|censored@8000|8.226593|
|P48|0|1|censored@8000|censored@8000|8.223413|
|P49|0|2|censored@8000|censored@8000|8.228331|
|P50|0|2|censored@8000|censored@8000|8.223450|
|P51|0|2|censored@8000|censored@8000|8.223204|
|P52|1|1|censored@8000|censored@8000|8.232203|
|P53|0|1|censored@8000|censored@8000|8.220872|
|P54|0|0|censored@8000|censored@8000|8.218515|
|P55|0|0|censored@8000|censored@8000|8.223836|
|P56|0|1|censored@8000|censored@8000|8.224387|
|P57|0|1|censored@8000|censored@8000|8.221612|
|P58|0|1|censored@8000|censored@8000|8.226264|
|P59|0|2|censored@8000|censored@8000|8.222262|
|P60|0|1|censored@8000|censored@8000|8.224395|
|P61|0|2|censored@8000|censored@8000|8.221022|
|P62|0|1|censored@8000|censored@8000|8.225482|
|P63|0|1|censored@8000|censored@8000|8.224517|
|P64|0|1|censored@8000|censored@8000|8.223752|

### Loss trajectory / exact-marginal signature (full500-step series in marginal_signature.json)
|Step|Validation CE|CE - ln3721|Max position /512|Material positions|
|---|---|---|---|---|
|0|251.382983208|+243.161235479|2|[]|
|5000|8.202960670|-0.018787058|10|[]|
|8000|8.169524491|-0.052223238|10|[]|
Prediction(a) material crossing observed: none, right-censored.
Prediction(b) optimizer-robust horizon failure remains conditional on full1M completion and loss/count signature; near marginal alone is insufficient. AdamW comparison is historical, not a new matched-seed causal trial.
Training loss: update1=250.876220703; update8259=8.171090126; full per-update trajectory in training.jsonl.

## GDN2 attended amendment1 — aggregate-best482500
Original fail-closed attempt preserved; supervisor task conflated best P5=333 with final P5=318.
Amendment was committed/pushed before T64 evaluation; no tuning or checkpoint change.
DONE gdn2_zeroshot
Zero training updates; all historical T16 counts reproduced exactly; model state unchanged.
|Position|T16 champion /512|T64 zero-shot /512|T64 CE|
|---|---|---|---|
|P1|512|512|0.000221|
|P2|512|512|0.000287|
|P3|512|512|0.001247|
|P4|512|511|0.026011|
|P5|333|295|1.317887|
|P6|3|2|6.734287|
|P7|0|0|8.222616|
|P8|0|0|8.222125|
|P9|0|0|8.221429|
|P10|0|0|8.221766|
|P11|0|0|8.221744|
|P12|0|0|8.221563|
|P13|0|0|8.222149|
|P14|0|0|8.221613|
|P15|0|0|8.221914|
|P16|0|0|8.221779|
|P17|not trained|0|8.222118|
|P18|not trained|1|8.221255|
|P19|not trained|0|8.284809|
|P20|not trained|1|8.867669|
|P21|not trained|0|9.233450|
|P22|not trained|0|10.670828|
|P23|not trained|0|11.589474|
|P24|not trained|0|14.460153|
|P25|not trained|0|14.337055|
|P26|not trained|0|17.756334|
|P27|not trained|0|18.141914|
|P28|not trained|0|18.757886|
|P29|not trained|0|20.769962|
|P30|not trained|0|21.718806|
|P31|not trained|2|22.130459|
|P32|not trained|0|22.489678|
|P33|not trained|0|22.487200|
|P34|not trained|1|23.071236|
|P35|not trained|0|23.402160|
|P36|not trained|0|23.330027|
|P37|not trained|0|22.871522|
|P38|not trained|0|22.980412|
|P39|not trained|0|22.756757|
|P40|not trained|0|23.337801|
|P41|not trained|0|23.061473|
|P42|not trained|0|23.133344|
|P43|not trained|0|22.459864|
|P44|not trained|0|23.536427|
|P45|not trained|0|24.098944|
|P46|not trained|0|24.133177|
|P47|not trained|0|23.734248|
|P48|not trained|0|24.326538|
|P49|not trained|1|24.471776|
|P50|not trained|0|23.930145|
|P51|not trained|0|23.974658|
|P52|not trained|0|23.270302|
|P53|not trained|0|23.345108|
|P54|not trained|0|22.809636|
|P55|not trained|1|24.169959|
|P56|not trained|0|22.546157|
|P57|not trained|0|22.839726|
|P58|not trained|0|23.394585|
|P59|not trained|0|23.451141|
|P60|not trained|0|23.300011|
|P61|not trained|0|22.972259|
|P62|not trained|1|23.804720|
|P63|not trained|0|23.479324|
|P64|not trained|0|23.615592|
Prediction(a)/(b): trained P1-P4=[512, 512, 512, 511], compared with512 each. Report mixed/noncollapse outcomes without forcing a binary verdict.
Prediction(c): near-learned P5=333 -> 295/512 zero-shot; P5+ columns reported, not a new training result.
