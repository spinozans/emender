# X16 ignition sweep and curriculum growth

Fixed-final500K primary; transient crossings descriptive. None = right-censored at observed phase step.
Historical endpoints are context, not matched causal points; seed0 only, no monotonicity assumed.

## e97_w64_seed0 (explicit P1 share 0.015625)
TERMINAL e97_w64_seed0: complete

T16: observed through phase update 500000; any-position first50=None; fixed-final=True.
|Position|Latest /512|Max /512|First50|First508|First512|
|---|---|---|---|---|---|
|P1|11|20|None|None|None|
|P2|0|2|None|None|None|
|P3|0|2|None|None|None|
|P4|0|3|None|None|None|
|P5|0|3|None|None|None|
|P6|0|2|None|None|None|
|P7|0|2|None|None|None|
|P8|0|2|None|None|None|
|P9|0|2|None|None|None|
|P10|0|2|None|None|None|
|P11|0|2|None|None|None|
|P12|0|3|None|None|None|
|P13|0|2|None|None|None|
|P14|0|3|None|None|None|
|P15|0|2|None|None|None|
|P16|1|2|None|None|None|

## e97_w32_seed0 (explicit P1 share 0.03125)
TERMINAL e97_w32_seed0: complete

T16: observed through phase update 500000; any-position first50=None; fixed-final=True.
|Position|Latest /512|Max /512|First50|First508|First512|
|---|---|---|---|---|---|
|P1|6|19|None|None|None|
|P2|0|2|None|None|None|
|P3|0|2|None|None|None|
|P4|0|3|None|None|None|
|P5|0|2|None|None|None|
|P6|0|2|None|None|None|
|P7|0|2|None|None|None|
|P8|1|3|None|None|None|
|P9|2|2|None|None|None|
|P10|0|2|None|None|None|
|P11|0|2|None|None|None|
|P12|0|3|None|None|None|
|P13|1|3|None|None|None|
|P14|0|3|None|None|None|
|P15|0|3|None|None|None|
|P16|0|2|None|None|None|

## gdn2_w512_seed0 (explicit P1 share 0.001953125)
TERMINAL gdn2_w512_seed0: complete

T16: observed through phase update 500000; any-position first50=None; fixed-final=True.
|Position|Latest /512|Max /512|First50|First508|First512|
|---|---|---|---|---|---|
|P1|0|2|None|None|None|
|P2|0|2|None|None|None|
|P3|0|2|None|None|None|
|P4|0|2|None|None|None|
|P5|0|2|None|None|None|
|P6|0|2|None|None|None|
|P7|0|2|None|None|None|
|P8|0|2|None|None|None|
|P9|1|2|None|None|None|
|P10|0|2|None|None|None|
|P11|0|2|None|None|None|
|P12|0|3|None|None|None|
|P13|0|2|None|None|None|
|P14|0|3|None|None|None|
|P15|0|2|None|None|None|
|P16|1|2|None|None|None|

## gdn2_w2048_seed0 (explicit P1 share 0.00048828125)
TERMINAL gdn2_w2048_seed0: complete

T16: observed through phase update 500000; any-position first50=None; fixed-final=True.
|Position|Latest /512|Max /512|First50|First508|First512|
|---|---|---|---|---|---|
|P1|0|2|None|None|None|
|P2|0|3|None|None|None|
|P3|0|2|None|None|None|
|P4|0|2|None|None|None|
|P5|0|3|None|None|None|
|P6|0|2|None|None|None|
|P7|1|2|None|None|None|
|P8|0|2|None|None|None|
|P9|0|2|None|None|None|
|P10|0|3|None|None|None|
|P11|0|2|None|None|None|
|P12|0|2|None|None|None|
|P13|0|2|None|None|None|
|P14|0|2|None|None|None|
|P15|1|2|None|None|None|
|P16|0|2|None|None|None|

## e97_curriculum_seed0 (explicit P1 share None)
TERMINAL e97_curriculum_seed0: criterion_not_reached

T16: observed through phase update 500000; any-position first50=26500; fixed-final=True.
|Position|Latest /512|Max /512|First50|First508|First512|
|---|---|---|---|---|---|
|P1|512|512|26500|40000|49000|
|P2|512|512|231000|242500|251000|
|P3|512|512|320000|339500|343000|
|P4|5|11|None|None|None|
|P5|0|2|None|None|None|
|P6|1|2|None|None|None|
|P7|0|2|None|None|None|
|P8|0|3|None|None|None|
|P9|0|2|None|None|None|
|P10|0|2|None|None|None|
|P11|0|2|None|None|None|
|P12|0|2|None|None|None|
|P13|0|3|None|None|None|
|P14|0|3|None|None|None|
|P15|0|2|None|None|None|
|P16|0|2|None|None|None|

