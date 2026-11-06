# Round4 design, before implementation

Read R1/R2/R3 results and exact-resume receipts. No production promotion.
Tool discovery finds no supervisor/intercom transport; no contact claimed.

Source inspection finds R3 already imports R2's register discipline in substance:
its update_value streams 128-row tiles; read_page maintains only two FP32x128
parity rows, and previous_value addresses one selected row. Therefore the missing
synthesis is page-granularity tuning, not introducing another context-sized
value tensor. Reuse these exact JIT functions and change the launch composition.
Sweep READ pages 256/512/1024 on matched T1024 inputs; retain full samples and
parity/repeat hashes. Score stays128, update128. Every page streams pairs rather
than materializing PAGE x128. Choose lowest measured median, then validate ten
shapes and masked-reset long fixtures. Compare R2 directly on identical inputs.

## T2048 validation strategy: composition proof plus independent primitives

Use the operator-authorized T1024 monolithic anchor plus carry-composition
proof, explicitly not an unchanged T2048 monolithic result. Keep R3 PAGE256 as
a validation decomposition at T2048. No weaker S/BS, arithmetic or masks.
R4 shares initialization/control/score/normalization with R3 byte-for-byte.
For a READ let p_i=FP32(FP32(w_i*strength_i)*value_i). Reference's distributed
row reduction has two ascending parity chains (independently checked by the
existing adversarial primitive reference). First pair initializes A0=p0,A1=p1;
then A_j <- FP32(A_j+p_(2r+j)), in increasing r. No page partial sums.
At an even page boundary storing/loading FP32 carries preserves their bits;
induction on pairs proves partition independence. Only final page reduces the
two rows with the unchanged fixed_sum. PAGE256/512/1024 divide BS>=1024;
smaller BS uses the existing guarded loop. Invalid tokens produce zero reads
and no state mutations. Different documents initialize distinct empty state.

Anchor the full pipeline to the unchanged monolithic reference at all ten
shapes including T1024. Independently test READ/previous at BS128..4096 with
random/cancellation/signed-zero/NaN data for each page size. At T2048 compare
complete state/read/trace/replay/gradient fingerprints with PAGE256 and reset.
This is compositional certification; report its scope honestly. It does not
prove all operation-rich cases or full-model training equivalence.

Standing109+108 suites,8 short state pairs,4 replay/gradient pairs, T512/1024
masked reset and T2048 composition gate remain required. Record native-only
22layer x2head x2pass projection over2B targets; <=0.502691ms/head is the24h
bar. Slower spill cuts rejected. Do not start training for hundreds-of-days
projections. Cards3/4/5/6 only, approved venv interpreter, serial attended work.
Seed3402 terminals harvested read-only if complete; never disturb card2.
No DiLoCo behavior change, gap-matrix IDs inapplicable.
