"""Render the final failed/partial part-7 result without waiving gates."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/'part7_arithmetic.json').read_text())

def ranges(profile):return [v for k,v in profile.items() if k.startswith('x36::')]
def activities(profile):return sum(x['activity_count'] for x in ranges(profile))
def syncs(profile):return len(profile['dtoh']),len([e for e in profile['steady_sync_runtime'] if e['name']!='cudaDeviceSynchronize'])
def load(name):return json.loads((ROOT/name).read_text())

fresh=[next(r for r in load(f'part7_fresh_t{t}.json') if r['arm']=='tape_lm') for t in (512,4096)]
continued=p['continued_profile']
# Explicit end-of-profile cudaDeviceSynchronize is outside registered ranges;
# steady cudaStreamSynchronize and DtoH entries are retained verbatim.
fresh_counts=[activities(r['profile']) for r in fresh]
continued_count=activities(continued)
continued_syncs=syncs(continued)
wall=p['fresh_gates']
long=p['long_context']
gpus=sorted({str(load(f'part7_live{live}.json')['physical_gpu']) for live in (17,256,1024,2800)})
failed_arithmetic=[]
for r in p['capacity']:
    if r['mode']=='continued' and r['status']=='failed':
        a,b=r['slots'];d=r['device_delta_ms'];q=r['required_savings_ms']
        failed_arithmetic.append(f"| {r['live']} | {a}→{b} | {d['forward']:.3f} | {d['replay']:.3f} | {q['step_ms']:.3f} | {q['replay_ms']:.3f} |")
rows=[]
for live in (17,256,1024,2800):
    gates=[r for r in p['capacity'] if r['live']==live and r['mode']=='continued']
    vals=[]
    for slots in ([2944,5888],[5888,11776],[8192,8224]):
        r=next(g for g in gates if g['slots']==slots)
        vals.append(f"{r['ratios']['step_ms']:.3f}/{r['ratios']['replay_ms']:.3f} {r['status'].upper()}")
    rows.append('| '+str(live)+' | '+' | '.join(vals)+' |')
absolute=[]
for live in (17,256,1024,2800):
    records={r['slots']:r for r in load(f'part7_live{live}.json')['results'] if r['mode']=='continued'}
    vals=[f"{records[s]['median_ms']['step_ms']:.3f}/{records[s]['median_ms']['replay_ms']:.3f}" for s in (2944,5888,11776)]
    absolute.append('| '+str(live)+' | '+' | '.join(vals)+' |')
convergence=(ROOT.parent/'REPORT_PART6.md').read_text().split('## Parts 1–6 convergence\n\n',1)[1].split('\n\nReproduction',1)[0]
values=[
    f"{wall[0]['ratio']:.3f}x {wall[0]['status'].upper()}",
    f"{wall[1]['ratio']:.3f}x {wall[1]['status'].upper()}",
    'Part 6 PASS carried; not rerun',
    'See new matched-entry table; failed ratios remain FAIL',
    f"Fresh {fresh_counts[0]}/{fresh_counts[1]}, continued {continued_count} FAIL",
    'Fresh 7 / continued 8 PASS (unchanged phases)',
    f"Fresh PASS; continued {continued_syncs[1]} syncs/{continued_syncs[0]} DtoH FAIL",
    'Part 6 PASS carried; fresh path unchanged by pooled optimization',
    '179 PASS, including 25 raising-unpack byte controls',
    '40/40 fixed-live points survive; no near-budget rerun',
]
lines=convergence.splitlines()
lines[0]=lines[0]+' Part 7 |'
lines[1]=lines[1]+'---|'
for i,value in enumerate(values):lines[i+2]=lines[i+2]+' '+value+' |'
convergence='\n'.join(lines)

text=f'''# X36 part 7 — event-driven adjoint liveness; closure FAILS

The requested all-gates-green or established-floors closure is **not achieved**.
The named dense-scatter hypothesis is disproved for the measured workload:
**zero dense native-gradient unpack calls** execute at every fixed-live point.
Replacing that absent operation cannot improve either capacity doubling or T4096.
The implemented improvement uses journal decisions to suppress **unused packed
value-adjoint traffic**. All **179 tests pass**, including cold signed zeros,
continued-capacity byte fixtures and a new raising-unpack guard in all 25
capacity cases (both imported and continued).

The two host sizing synchronizations remain; inclusive launches remain above
10. Failed capacity/long-context gates retain their measured ratios and required
savings in [part7_arithmetic.json](receipts/part7_arithmetic.json).
**No hardware/irreducible floor waiver is claimed.** This is a failed performance
closure with a narrower validated optimization and a corrected diagnosis.

## What the journal actually routes

`reverse_program` already reads `Cases`, `Matched` and `Alloc` and routes the
packed value/key adjoint directly into the writing token's `DVout`/`DKout`.
Corrections read only their matched row; allocation reads and clears only its
allocated row. There is no dense cold-gradient pass in this token loop.
Pop derivatives use strength adjoints and logical reductions, not a W_v/W_k
payload scatter. Read adjoints still cover resident live rows because query
gradients depend on their contents.

`native_grad` calls `unpack` only if the corresponding native-state **input**
requires a gradient. The capacity protocol requires query/value **projection
input** gradients and no native-state input gradients: `[false,false,false]`.
The new regression patches `unpack` to raise; all 25 shapes still match oracle
bytes in both modes. Receipts assert those three flags at every sample.
Even for native-gradient consumers, unpack runs once per chunk, outside the
reverse token loop. T4096's empty-origin fallback does not use pooled unpack.

The new device-only event-consumer mask scans recorded correction/allocation
decisions once per pooled store. If no writing token can consume value adjoints
and there is no native-value gradient consumer, it suppresses GValues loads and
stores during read replay. It preserves the exact per-token DVout zero stores.
Chunks with writes and native-value consumers keep the original recurrence.
For live17/256/1024/2800 this removes logically
{', '.join(f"{256*512*live*128*8:,}" for live in (17,256,1024,2800))} bytes
of packed read/write accesses per chunk. These are logical accesses, **not
measured DRAM bytes**. The branch remains on device and adds no launch or sync.

Cold output-adjoint behavior is unchanged: when native gradients are requested,
untouched key/value rows retain any supplied output adjoints, and cold strength
rows execute the existing `+0` recurrence. They cannot universally be replaced
by zero without losing nonzero output adjoints or signed-zero semantics.
The full cold-signed-zero fixtures passed; no exactness break was observed.

## Capacity remeasurement

B64/H4/T512, entry live rows spread to S-1, frozen keys, zero writes/pops,
decay=1, production allocation. Two warmups and five raw samples per point;
construction/snapshots/GC remain outside samples. Imported controls and all
medians, compiler statistics, pool counts, native-gradient flags and peaks are
retained in `part7_live17/256/1024/2800.json`. All 40 points completed.

Each cell below is **step/replay** ratio; limit <=1.5 for both.

| Entry live | 2944→5888 | 5888→11776 | 8192→8224 |
|---|---|---|---|
{chr(10).join(rows)}

Absolute continued **step/replay milliseconds** (no baseline slowing):

| Entry live | S2944 | S5888 | S11776 |
|---|---:|---:|---:|
{chr(10).join(absolute)}

Per-point ratios to the part-6 medians are also receipted in JSON; the
removed logical traffic does not imply proportional wall savings.

Zero native unpack traffic contributes **0 ms** to every failed gate. The
remaining capacity-dependent operations are concrete and separately counted:

| S | Forward tile probes/chunk | Full logical lane width | Public freed backing bytes |
|---|---:|---:|---:|
'''
for r in p['fixed_live_work']:
    text+=f"| {r['slots']} | {r['forward_tile_probes']:,} | {r['logical_reduction_width']:,} | {r['public_freed_bytes']:,} |\n"
text+=f'''
The forward loops over `ceil(S/32)` tiles at saturated logical reach, including
empty tiles. It evaluates 32 floating lanes per nonempty tile. Uniform live256
entries populate all 92 then all 184 tiles on the first doubling: **2x padded
read work at fixed live**. Uniform live1024 entries populate all 184 then all
368 tiles on the second doubling: another **2x padded read work at fixed live**.
Live17 occupies 17 tiles at every capacity but still probes the full tile grid.
The JSON receipts these nonempty-tile counts and padded FLOP arithmetic for
every capacity/live pair; they are source-level lane counts, not ISA counters. The reverse reconstructs full logical operands for DK+4=68
reductions per store/token (64 query sums plus pop, maximum, denominator and
dp sum). The first doubling changes the logical width 4096→8192 and the oracle
warp policy 4→16, even when PS stays 17 or 256. `freed` is dense public int8
telemetry initialized once per chunk; its public transpose can require another
allocation/copy of the same size. Private metadata and inverse maps also retain
logical-S shapes. None is a dense projection-gradient scatter.

For every pair, the JSON retains `forward_delta`, `replay_delta` (under
`device_delta_ms`) and `max(0,large-1.5*small)` separately for wall/replay.
Those identify how much of the measured increment is forward versus backward;
source counts identify the surviving O(S) mechanisms. For the continued misses:

| Live | S pair | Forward delta ms | Replay delta ms | Wall savings required ms | Replay savings required ms |
|---|---|---:|---:|---:|---:|
{chr(10).join(failed_arithmetic)}

Required savings are `max(0,large-1.5*small)`; a zero replay requirement
means only the whole-step ratio fails. These are separately measured ranges. Their individual
subexpression timings have **not** been causally isolated. These are current
implementation costs, not an established irreducible floor. No failure is
silently waived because its packed pool is small.

## Long context: useful work is not constant per token here

The fresh matched T512/T4096 ratios are **{wall[0]['ratio']:.3f}x** and
**{wall[1]['ratio']:.3f}x**. T4096 requires
**{long['required_wall_savings_ms']:.3f} ms** of wall savings to meet <=2x.
The proposed dense-unpack removal contributes exactly zero. The new pooled
liveness specialization does not execute for this fresh long-context fallback.

The routing trace counts {long['occupied_pairs'][0]:,}→{long['occupied_pairs'][1]:,}
occupied slot-token pairs: **{long['occupied_pair_ratio']:.0f}x** for 8x tokens.
Mean live occupancy is 128→1024. Just score-dot plus weighted-value read work is
`2*(64+128)*pairs` = {long['payload_read_flops'][0]:,}→{long['payload_read_flops'][1]:,}
FLOPs. This is flat at 384 FLOPs **per live-slot/token pair**; per input token
it rises 8x. Padding, nonlinearities, recurrence and backward are additional.
These kernels use vector reductions; the count is not a calibrated GEMM floor.

| Separate profile | T512 ms | T4096 ms |
|---|---:|---:|
| Scan kernel | {long['scan_kernel_busy_ms'][0]:.3f} | {long['scan_kernel_busy_ms'][1]:.3f} |
| Reverse kernel | {long['reverse_kernel_busy_ms'][0]:.3f} | {long['reverse_kernel_busy_ms'][1]:.3f} |
| Other scan/replay device activities | {long['non_scan_reverse_device_busy_ms'][0]:.3f} | {long['non_scan_reverse_device_busy_ms'][1]:.3f} |

Scan ns per occupied pair is {long['scan_ns_per_occupied_pair'][0]:.3f}→{long['scan_ns_per_occupied_pair'][1]:.3f};
reverse is {long['reverse_ns_per_occupied_pair'][0]:.3f}→{long['reverse_ns_per_occupied_pair'][1]:.3f}.
Inclusive fresh activities are {fresh_counts[0]}→{fresh_counts[1]}, not 8x as many.
Most device busy time is inside scan/reverse; the full per-name launch ledger
and other activity totals are in the JSON. Profile GPU busy times are not
additive wall samples. Launch/telemetry wall costs are not independently
calibrated, so this does not establish an honest compute lower bound or fully
explain the wall residual. The long-context gate remains failed when above 2x.

## Sync and launch residuals

The continued profile has **{continued_count} inclusive activities** versus <=10
(excess {continued_count-10}), **{continued_syncs[1]} steady stream syncs** and
**{continued_syncs[0]} DtoH transfers** versus zero. Their complete names/counts
and busy-time decomposition are retained in `part7_profile.json` and the
losslessly archived trace. Explicit terminal and profiler teardown device synchronizations are outside
the registered ranges; no steady transfer is excluded.

The current eager exact-allocation path has two serial shape dependencies:
forward needs device entry maximum/cold-tail metadata before host journal
reservation/dispatch; backward needs the post-scan union maximum before host
`torch.empty` allocates exact PS. Arithmetic is **2 > 0**, independent of event
scatter. This is a conditional constraint of the present host-sized allocation
strategy, not a hardware floor: device allocation, bounded reservation or a
changed specialization strategy could alter it. No such redesign or launch
fusion was implemented here. The inclusive-launch gate has no irreducible-floor
proof. Both requested improvements remain unmet.

## Parts 1–7 convergence

{convergence}

## Reproduction and remaining risk

`receipts/part7_run.sh` runs the complete parity suite before all timed work,
then the four fixed-live sweeps, continued profile and matched T512/T4096
profiles/routing traces. Only `/home/erikg/emender/.venv/bin/python` and
`run_leased.sh` were used for GPU work, on physical card {', '.join(gpus)}
from the broker-approved 0/1/5/6 set.
No .pt files, forbidden/certified cards, WG, workers or subagents were used.
Run archival as `/home/erikg/emender/.venv/bin/python -m
experiments.expressivity_tasks.x36_tape_perf.receipts.part7_archive` from the
repository root. It verifies byte-identical gzip round trips and profile-summary
equality; `part7_analysis.py` regenerates arithmetic, then `part7_report.py`
regenerates this report. Part 6 is archived in [REPORT_PART6.md](REPORT_PART6.md).

At S2944, live1024 replay improves 293.803→121.747 ms and live2800 replay
847.677→442.912 ms versus part 6. These absolute gains do not close the small
entry gates or justify a floor waiver.

The delivered change removes unused packed adjoint traffic; it does not replace
a dense scatter on the measured path because that operation is absent. Failed
ratios, two sizing syncs and excess launches remain risks and unmet work. The
historical curve and fresh capacity gates are explicitly carried from part 6;
they were not rerun. Supervisor escalation was unavailable because no callable
`contact_supervisor` tool was exposed; coordination attempts and discoveries
are in `part7_coordination.txt`. This report must not be used to attest that
X36 reached all-gates-green or a proven-floor closure.
'''
(ROOT.parent/'REPORT.md').write_text(text)
print('Wrote',ROOT.parent/'REPORT.md')
