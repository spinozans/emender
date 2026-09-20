# E97 Dose Screen v1 — five-point curve at u32 + the u128/u256 arc anchors

Five 32-update probes from the bridge parent, identical protocol, varying only the
restored-pool dose (grounded-authored + representation-bridge records): d0 (v2 prep,
0%), d25, d50, d75, d100 (v3 prep). Evaluations: the 14-case Stage-B panel (Pi-native
dialect) and the fixed 24-case execution slice (OpenHands-compatible runtime), with
bridge controls in-run for every slice. All probes admitted (32-update proposals,
audited, nonce-searched), trained, and evaluated under the frozen machinery; all
artifacts under pi-native-repair9-dose-screen-v1/ and pi-native-repair9-dose-screen-probe-d{0,25,50,75,100}-v1/.

## The u32 curve

| Dose | Stage-B valid/correct @u32 | Exec slice @u32 (bridge control) |
|---|---|---|
| 0% | 1/14, 1/14 | 9/24 (16/24) |
| 25% | 1/14, 1/14 | 13/24 (16/24) |
| 50% | 0/14, 0/14 | 11/24 (16/24) |
| 75% | 0/14, 0/14 | 11/24 (16/24) |
| 100% | 1/14, 1/14 | 12/24 (16/24) |

## Findings

1. THE HEADLINE: the u32 Stage-B collapse is DOSE-INDEPENDENT and mixture-wide. Even
   d0 (zero restored records) collapses to 1/14 — yet the v2 ARC recovered to 10/14
   by u128. Every mixture of this family passes through an early dialect-chaos
   transient (~32 updates) and re-consolidates over the following ~100 updates.
   Consequence: checkpoints below ~u64 are uninterpretable for Stage-B; the arc
   protocol's 128-update juncture cadence is the correct instrument.
2. The slice shows a presence effect, not a dose effect: d0 = 9/24; every restored
   dose lands 11-13/24 (within noise of each other). The OH-runtime competence
   requires the restored data but saturates by d25.
3. The real dose signals live at u128/u256 (from the arcs): Stage-B 0%=10/14 vs
   100%=5/14 at u128; execution 0%=0/96 vs 100%=59-60/96 at u256. Note honestly:
   even at 100% restored, u256 execution (59-60/96) remains below the gate floor
   (64/96) — the restored dose alone does not clear the execution leg at u256; the
   arc's later gates (u512+), the longer training, and the intermingled public
   tool-talk corpus are the remaining levers.

## V4 recommendation (for operator sign-off)

Run the v4 arc at d25 (25% restored pools: 646 authored + 556 bridge records):
- Stage-B preservation: the u128 endpoint trend says lower dose preserves the
  Pi-native dialect (0%→10/14, 100%→5/14); d25 is the lowest dose that still
  carries restored data.
- Slice: d25 measured best of all doses at u32 (13/24).
- The intermingled public tool-talk corpus (Toucan SFT 135M + Nemotron 48M +
  Tool-Reasoning 31K ~8M assistant tokens, overlap-audited, rendered Pi-native)
  enters the same prep — teaching tool vocabulary through the Pi frame, which
  may relieve the dialect competition itself.
- Uncertainty, stated: u256 execution at d25 is unknown; the u256 gate is the
  arc's first decision point, not its promise. Fail-closed rules unchanged.
