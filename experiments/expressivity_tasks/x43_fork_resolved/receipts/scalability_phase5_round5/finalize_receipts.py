"""Finalize fresh R5 evidence; assertions fail closed. No GPU work."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
root=Path('experiments/expressivity_tasks/x43_fork_resolved');r=root/'receipts/scalability_phase5_round5'
def read(name):return json.loads((r/name).read_text())
def write(name,obj):
 p=r/name;assert not p.exists();p.write_text(json.dumps(obj,indent=2)+'\n')
failed={'profile_terminal.json','profile_fixed_terminal.json','reduction_terminal.json'}
terminals={p.name:json.loads(p.read_text())['exit_code'] for p in sorted(r.glob('*terminal.json'))}
assert len(terminals)==17 and {k for k,v in terminals.items() if v}==failed
sweep=read('sweep.json')['rows'];assert len(sweep)==10
shape=read('shape.json');gradient=read('gradient.json')
assert len(shape)==8 and len(gradient)==4
assert all(x['exact'] and x['repeat_exact'] for x in sweep+shape+gradient)
primitives=read('reduction_fixed.json');assert len(primitives)==24
assert all(x['read_exact'] and x['prev_exact'] for x in primitives)
for t in (512,1024,2048):
 x=read(f'long_{t}.json');assert x['exact'] and x['reset_exact'] and x['masked_tail']==7
for label,count in [('oracle',109),('battery',108)]:
 assert re.search(rf'{count} passed', (r/f'{label}_console.txt').read_text())
for label in ('historical_phase2','historical_phase5','historical_round2'):
 assert read(label+'_guard.json')['matches']
old=read('implementation_sources.json')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in old.items())
projection=read('projection.json');a=read('matched_r4.json')['rows'][0]
profile={arm:read(f'cuda_{arm}.json')['kernels'] for arm in ('r4','r5')}
assert all(profile[arm] for arm in profile)
rows='\n'.join(f"|{x['batch']}|{x['tokens']}|{x['reference_median_ms']:.6f}|{x['median_ms']:.6f}|{x['read_registers']} / {x['read_spills']}|{x['registers']} / {x['spills']}|" for x in sweep)
report=f'''# Phase5 round5: imported attention tiling, real1.3B NO-GO

**T1024 improves from {a['reference_median_ms']:.6f} to {a['median_ms']:.6f}ms/head
({projection['time_reduction']*100:.2f}% less wall time, {projection['speedup']:.6f}x).
The native-only1.3B projection is {projection['native_only_days']:.6f}days: NO-GO.**
No production kernel changed and no training launched.

## Profile and the attention import

Design/algebra a8fb1ac5 committed/pushed before implementation; implementation
4b9aa02d committed/pushed before the complete validation battery. Read algebra
is scaled dot-product SOFTMAX over currently live slots, then strength-weighted
values. match_coefs/match_wire affect post-read write enable, not addressing.
The exact algebra and the correct online m,d,U equations are in DESIGN.md.
[FlashAttention](https://arxiv.org/html/2205.14135v2) supplies register-sized tiling
and streaming accumulators. Direct online-softmax rescaling changes native FP32
parenthesization/division placement, so it cannot simply replace the required
byte semantics. R5 imports tiles while preserving normalization's denominator
tree and the exact two parity accumulation chains. No full context x slots
matrix is materialized. Normalization's BS-vector scratch remains.

The early CUDA-event profile correctly selected READ, but included enqueue gaps.
Subsequent CUPTI profiling successfully measures actual kernel durations,
without claiming Nsight hardware spill counters or register-lifetime tracing:

|Stage|R4 kernel ms|R5 kernel ms|
|---|---:|---:|
|initialization|0.002336|0.002368|
|pre/post control,2048 calls|56.561090|56.662388|
|address scores,1024 calls|3.729571|3.728535|
|normalization,1024 calls|1.855219|1.856075|
|READ including gather/carries,2048 calls|189.907448|27.859426|
|sum of kernel durations|252.055664|90.108792|

READ fell6.82x in this instrumented comparison, from75.34% of device time
to30.92%; control is now62.88%. Uninstrumented matched wall time remains
191.888684ms. These are separate profiled and unprofiled iterations: their
difference is NOT a certified additive measurement of launch overhead. It does
show that further work must measure host enqueue/launch gaps and control rather
than assuming addressing matrix arithmetic dominates. Score/normalization are
only5.58ms of the R4 device profile. Carry/gather costs within READ are not
individually resolved. The profile receipts retain exact samples and counts.

R5 loads32x128 products into a register tile, exposes16 pairs by bit-preserving
reshape/permute/split, then appends pairs in exact increasing order. No tile
subtotal. PAGE1024 boundary stores/loads preserve FP32 carries; first pair
initializes without adding zero. T1024 READ42 registers/0 compiler spills versus
R4's28/0; increased tile registers fit without observed compiler spilling.
See COMPOSITION_PROOF.md for pair/tile/page/token induction and scope.

R4 partitions launches but does not implement PagedAttention page tables.
[Its source literature](https://arxiv.org/abs/2309.06180) addresses physical KV
allocation/sharing; no measured layout/fragmentation cost here justifies a table.
Context transitions cannot run as independent immutable KV queries: CORRECT,
expiry, array writes, POP and MERGE mutate earlier slots. The existing ordered
state machine is retained. No O(1) total metadata claim: scalar control and
normalization remain BS-wide.

## Sparse read decision

PB3 reports all sampled array effective weights nonzero (findability1.0), not
their mass. Its existing counters cannot establish top16/32 concentration.
Fresh synthetic snapshots at t256/512/1023 have top16 masses
.687718/.667267/.661266 and top32 .790909/.839760/.810231, with256/257/257
nonzero products. These are RANDOM-FIXTURE measurements, not trained PB3
statistics. There is no byte-exact pruning result. Trained top-k mass remains
unmeasured. Top-k is not chosen or implemented; any future variant requires
registration and quality predictions before training. Exact top-k selection
itself still scans occupancy unless an index is introduced.

## Ten independent monolithic pairs

Broker physical5, five synchronized allocation/wrapper-inclusive wall samples
per arm on identical nonzero inputs. All ten complete read/final physical state/
raw trace fingerprints and repeats match. Small shapes still regress strongly;
no unconditional selector is added. Timing varies across cards/runs; independent
sweep T1024 is197.998854ms versus the separate matched R4 comparison191.888684ms.
Compiler resource columns describe READ and post-control only.

|B|T|monolithic ms/head|R5 ms/head|READ regs/spills|post regs/spills|
|---:|---:|---:|---:|---:|---:|
{rows}

## Gates, failures and projection

- 109 X31 tests pass in15.57s;108 C14/operator tests pass in37.71s; no skips.
- Eight operation-rich short state pairs and four native/replay/gradient pairs
  pass full byte fingerprints and repeats.
- All24 independent matrix READ/previous-value adversarial pairs pass through
  BS4096: random, cancellation, signed zeros and NaN payloads.
- T512/T1024 unchanged monolithic long gates and T2048 compositional gate pass
  complete native/state/read/trace/replay/gradient fingerprints. All have peak257,
  seven masked-tail positions and distinct-document fresh-state reset equality.
- T2048 reference is PAGE256 decomposition plus the T1024 monolithic anchor and
  carry proof. No repaired unchanged T2048 LLVM compilation is claimed.
- Two profile harness failures (wrong positional index, then missing proxy
  attribute) and one candidate SliceLayout compilation failure are retained
  as exit1 with logs/source snapshot. Corrected final profile/reduction and all
  final gates exit0. No failed receipt qualifies a result.

At22 layers*2heads*2native passes, T1024 and2B raw-byte scenario targets,
{projection['seconds_target']:.9f}s/target, {projection['targets_second']:.6f}targets/s,
{projection['native_only_days']:.6f}native-only days. Required23148.148148targets/s,
<=.502691ms/head with every other cost free. Dense/replay/optimizer/source/
checkpoint cost is excluded: this is a lower-bound scenario, not training.
No150M/1.3B throughput or memory certification or eventual speed promise.

Residual risks: maximum-live operation-rich long cases; full trained-model
fingerprints; >=2M paired real-target optimizer/RNG/cursor/checkpoint-resume
identity; sustained full-width throughput/memory; BS-wide control; enqueue gaps;
trained top-k concentration. Diagnostic wrapper temporarily patches a shared
READ function under try/finally and assumes serial execution; not thread safe
or suitable for production installation. All are explicit non-promotion limits.

Manual notes: attended single-thread work, broker cards4/5 only, approved venv
Python only. No training/checkpoint, WG, subagents, protected-card workload,
corpus change, sxgc or /mnt/nvme2n1 access, or .pt commit. AGENTS.md/CLAUDE.md
unchanged. No resilient DiLoCo change; gap-matrix IDs inapplicable. Supervisor/
intercom discovery exposed no callable transport, so no contact is claimed.
Unrelated workspace changes preserved. Historical phase2/phase5/R2 source guards
match; source/battery hashes, terminal exits and exact resume are retained here.
'''
p=root/'RESULT_UNBOUNDED_SCALABILITY_PHASE5_ROUND5.md';assert not p.exists();p.write_text(report)
resume='''# Round5 exact resume: attention register tiling, real1.3B NO-GO

No production selection, training or training checkpoint exists. Do not start
the1.3B scenario from this result. Best internal diagnostic now uses
c14_phase5r5_tiled_pages.py:32x128 register tiles, PAGE1024, exact normalized
parity carries. Matched191.888684ms/head versus R4 255.924206; projection381.723005
native-only days. All ten native/repeat pairs,109+108 suites,8 short state pairs,
4 replay/gradient pairs,24 primitive adversarial pairs and T512/T1024/T2048
masked-reset fixtures pass. T2048 is compositional, not repaired monolithic.

Pre-implementation design a8fb1ac5; implementation4b9aa02d. Read DESIGN.md,
COMPOSITION_PROOF.md, result report and validation_summary.json first.
profile_final.json is event attribution; cuda_r4/r5.json are CUPTI actual kernel
durations, NOT spill counters. READ189.907448->27.859426ms device time; control
56.66ms now dominates device time. Uninstrumented wall191.89ms includes host
submission and gaps; profile/unprofiled difference is not an exact subtraction.
Next structural candidate should separately measure/control those gaps and
control, retaining mutable-store order. Do not blindly re-import online softmax:
its real-arithmetic equivalence does not retain the native FP32 reduction tree.
Do not assume trained reads are top-k sparse: PB3 lacks weight masses, synthetic
top32 is only .79-.84. No top-k implementation/registration/training exists.

Only diagnostic install is c14_phase5r5_validate.install. It is serial-only;
shared READ temporarily patched in try/finally. Fresh runs require NEW output
paths/wrapper receipt directory, never overwrite existing completed receipts.
Use only /home/erikg/emender/.venv/bin/python and a broker lease restricted to
physical3/4/5/6 (this round used4/5); never occupied/protected cards. The source
wrappers deliberately refuse their completed terminals. New candidate needs a
new committed/pushed design and explicit source bindings before validation.
Push every commit. Preserve historical guards and failed attempt evidence.

Mandatory before promotion: operation-rich maximum-live long/full trained-model
fingerprints, >=2M paired real-target optimizer/RNG/cursor/resume identity,
150M/1.3B sustained throughput/memory. No other training/evaluator was touched.
No DiLoCo changes; IDs inapplicable. No supervisor transport exposed/contact
claimed. Unrelated worktree changes preserved. Fresh process/device/lease checks
and final source hashes are retained separately. No own GPU process remains.
'''
assert not (r/'RESUME.md').exists();(r/'RESUME.md').write_text(resume)
paths={Path(p) for p in old}|set(root.glob('*phase5r5*.py'))|set(root.glob('*phase5r5*.sh'))|{Path(__file__),p}
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
write('validation_summary.json',dict(utc=datetime.now(timezone.utc).isoformat(),design_commit='a8fb1ac5',implementation_commit='4b9aa02d',terminals=terminals,sources=sources,counts=dict(oracle=109,battery=108,shape=8,gradient=4,primitive=24,native_sweep=10,long=3),projection=projection,decision='NO-GO',training_started=False,manual_notes='Serial attended work. Failed attempts retained, final gates passed. No exposed supervisor transport.',residual_risks=['BS-wide metadata/control','host enqueue overhead unresolved','trained top-k concentration unmeasured','full model/resume/full-width throughput gates unqualified']))
print('All final evidence assertions passed; reports written.')
