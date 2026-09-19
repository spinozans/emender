# Dose-screen execution slice — the fixed 24-case subset (STAGED definition)

The frozen 96-case execution panel (the panel used at the v10 and v3 u256
dual gates; identical case sets, byte-verified) partitions into SIX id-prefix
groups x FOUR families:

| group (id prefix) | cohort field | cases | per family |
|---|---|---:|---:|
| regression-old | prior-regression | 8 | 2 |
| regression-fresh | prior-regression | 8 | 2 |
| fresh | prior-fresh | 16 | 4 |
| transfer | prior-transfer | 16 | 4 |
| bridge-fresh | fresh | 32 | 8 |
| bridge-composition | composition | 16 | 4 |

Slice = ONE case per (group, family) cell = 24 cases: every group represented
(4 cases each), every family balanced (6 cases each). Selection method
(fixed, shared by all dose probes and identical across them):

- candidates per cell = the source panel's cases with that (id-prefix, family),
  sorted by case id;
- cell visiting order = groups in the table order x families
  (lookup, sum, edit, recovery);
- `random.Random(240977).choice(candidates)` per cell.

Source panel: `pi-native-repair9r-full-arc-v3-u256-dual-gate-v1/execution/panel.json`
sha256 `cdcd3ac9e4bde2325f2dc28e082efd8e74a04cd03a29fe34d2663882cd59aa74`
(upstream panel sha `7f03244fca9437ebdfbc7044287c9ddeff61d1953c8a80f062032e46f097ad67`
recorded inside it). The 24 chosen ids live in `slice-definition.json` and are
re-verified (membership + 6x4 balance) by `freeze_execution_slice_panel.py`
each time a probe's slice panel is regenerated for its exact u32 checkpoint.

The slice is a SCREEN, not the gate: the frozen `>= 64/96` execution gate stays
bound to the full 96-case panel. Slice panels bind FOUR models per the dual-gate
convention: dose-probe-y (train), dose-probe-x (saved), bridge-y-control,
bridge-x-control — 96 episodes per probe run, with the bridge controls as the
same-run window reference.
