# Native LR screen v1: execution plan

`configs/pi/e97-native-lr-screen-v1.json` freezes five new rates, in order:
`2e-4, 0.00047431158698290157, 1e-3, 1e-5, 2e-6`. Reuse the completed `5e-5`
control. Every new case starts fresh SR optimizer state from the same parent
live/y point, uses the same eight updates and exact sample schedule, and
consumes 1,490,478 total / 415,097 native-agent targets.

`prepare_e97_native_lr_screen.py` verifies retained baseline inventories and
renders the proven smoke launcher with exact-once anchors. Numerical training
code and flags other than LR are unchanged. Each case has separate outputs,
Triton caches and a fresh eight-rank process set. The wrapper additionally
captures lease acquisition into a checked assignment before `eval`, so a failed
acquisition cannot become a successful empty `eval`. No retry, promotion or
automatic sustained expansion is permitted. Any process or identity failure
stops the chain and retains its artifacts.

Evaluate two completed rates together, each in x and y, on the exact previously
frozen examples. The final pair repeats the existing 5e-5 control; this repeat
is retained separately, not substituted for the original baseline measurement.
The retained control worker and evaluator use the already qualified immutable
source exports. Per-case deadlines are 2,400 seconds with 30-second kill grace.
The whole bounded screen has a six-hour ceiling, not an unbounded controller.

This is a **small LR screen**, not the substantive training budget. Results are
for shortlisting acquisition/retention trade-offs, not final LR selection or
agent capability claims. Subsequent larger-exposure work targets tens of millions
of agent targets, with intermediate learning/retention/execution checks.

Architecture scope remains ADR-003 safety intent R07/R12, R14/NDP13, R16 and
NDP15 checkpoint atomicity. No elastic/native-data-plane/async/Frontier/ROCm
or communicator-shrink claim. No source commands are executed by this screen.
