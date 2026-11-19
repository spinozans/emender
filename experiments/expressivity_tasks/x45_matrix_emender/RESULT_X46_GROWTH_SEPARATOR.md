# RESULT X46 — THE GROWTH SEPARATOR ON GAS: fixed machines vs growth where the state provably must alias

**Status: the construction, the derived ceilings and the pre-stated
predictions are COMMITTED (`ff9098b5` + this registration's commit) BEFORE
any headline arm.** The pre-stated predictions live in
`PREREGISTRATION_X46_GROWTH_SEPARATOR.md` (DRAFT — PENDING OPERATOR
SIGN-OFF; the arm runner is fail-closed on the `OPERATOR-SIGNED` mark).
Results are appended below per part as they land; nothing pre-stated is
edited after sign-off.

## 0. What is committed (the pre-registration evidence)

* **The construction** (`x46_growth_separator.py`, `DESIGN_X46_GROWTH
  _SEPARATOR.md` §1-§2): the GAS family — a document-bounded
  growing-associations stream, `n` distinct keys bound to iid values
  (pool 128), all introduced then all probed at lag ~n; one symbolic
  construction, two encodings (combined pair tokens for the ours
  machines, 2-byte-key byte patterns for GDN2). The `n` dial
  {64, 320, 640}.
* **The derived ceilings** (machine-checked:
  `receipts/x46_selftest.json`, 19 CPU tests green):
  * OURS: the slot-count bound (live rows ≤ R at every token; every
    post-fill APPEND paid for by a POP; ≤ R distinct directions in the
    state) + the aliasing swap argument (an aliased row's read outputs
    are one shared direction up to a positive scalar — no fixed linear
    head answers both probes; 200 random heads, zero answer both).
    `C(N) = min(1, 128/N) + (1−…)/128`.
  * GDN2: the key-space bound (orthonormal keys retrieve with error
    exactly 0.0 — the per-head exact bound 128; the Welch
    frame-potential bound holds for every key configuration, realized
    rms 0.0882 ≥ floor 0.0686 at N=320; crosstalk/signal closed form
    `sqrt((n−d)/d)` = exactly 1 at n=2d, 1.22 at the floor and 1.57
    realized at N=320). The pre-stated effective capacity H·d_k = 256.
  * The registered ours trigger fires mechanically on the constructed
    overflow (rate 0.0 → 1.0 across the fill boundary, fire at 128, snap
    64→128) — no calibration needed on this family.
* **The ceiling table** (chance 1/128, band +0.05):

| machine | N=64 | N=320 | N=640 |
|---|---:|---:|---:|
| ours-fixed (128 slots) | no ceiling | **0.4047** | 0.2063 |
| gdn2-fixed (2×128) | no ceiling | **0.8016** | 0.4047 |
| ours-growable (grown 512) | — | escape | 0.8016 |
| gdn2-growth (grown H=4 = 512) | — | escape | 0.8016 |

## 1. RESULTS — pending the operator sign-off and the headline arms

To be appended per part as it lands: the N=64 learnability control (P-0),
the N=320 headline (P-1/P-2: the fixed machines pinned at their derived
ceilings vs the growables escaped, with the mechanistic stratification
P-4/P-5), the N=640 matched-grown-state cell (P-3: the slots-vs-
superposition prize, no directional prediction), and the verdict against
the pre-stated refutation clauses.

## 2. Provenance

New files only: `x46_growth_separator.py` (the construction, the arms, the
fail-closed runner), `test_x46_growth_separator.py` (19 CPU tests),
`DESIGN_X46_GROWTH_SEPARATOR.md`,
`PREREGISTRATION_X46_GROWTH_SEPARATOR.md` (DRAFT pending sign-off), this
file, `run_x46_leased.sh` (cards 2/3), and the receipts
(`x46_selftest.json`, `x46_smoke.json`, `x46_gdn2_calibration.json`, the
`x46_arm_*`/`x46_sweep_*`/`x46_summary_*` cells as they land). The
committed machinery is IMPORTED ONLY, never edited: the K8 growth
machinery (`k8_growth.py`), the GDN2 wrappers
(`gdn2_growth_c3.py`/`gdn2_growth_smoke.py`), the spectra sidecar, the
`k10_*`/`k11_*`/`k12_*` sources, every PREREGISTRATION, and `lattice.py`.
Card discipline: cards 2/3 via the broker lease; card 4 (the detached
emender run) never touched; card 7 forbidden.
