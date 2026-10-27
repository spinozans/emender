"""Build an explicitly incomplete part-5 adjudication from retained receipts."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from .validate_part5 import source_hashes

ROOT = Path(__file__).resolve().parent


def main():
    gates = json.loads((ROOT/'gates_part5.json').read_text())
    parity = json.loads((ROOT/'parity_part5.json').read_text())
    assert gates['source_sha256'] == parity['source_sha256'] == source_hashes()
    suites = list(ET.parse(ROOT/'validation_part5.xml').getroot().iter('testsuite'))
    counts = {name:sum(int(s.get(name,0)) for s in suites)
              for name in ('tests','failures','errors','skipped')}
    assert counts['tests'] >= 160 and all(counts[n] == 0 for n in ('failures','errors','skipped'))
    floors = {(r['component'],r['heads']):r for r in gates['structural_cost']}
    floor_ratios = {name:floors[name,16]['wall_ms_per_token']/floors[name,1]['wall_ms_per_token']
                   for name in ('prefix_native_initialization','exact_fresh_preparation',
                                'sequential_store_core','fresh_scan_writes_disabled')}
    summary = dict(task='X35 v3 part 5',scope_complete=False,v3_closed=False,
        result='FAIL: live/cold pooling is not implemented. Diagnostic isolation and six-gate remeasurement only; production representation is unchanged from part 4.',
        validation=dict(**counts,unchanged_x31_tests=109,unchanged_prior_x35_tests=49,
                        new_diagnostic_tests=2,raw='validation_part5.txt',junit='validation_part5.xml',source_receipt='parity_part5.json'),
        registered_gates=gates['gates'],forward_speedup=gates['forward_speedup'],
        heads_ratio=gates['heads_ratio'],heads_ratio_s736=gates['heads_ratio_s736'],
        component_h16_h1_ratios=floor_ratios,
        residual_cost_adjudication='Ordinary fresh T16 scans have physical S=16 for either architectural budget, not S=11776. Exact production fresh preparation and the sequential store kernel are isolated independently. The core grid grows from 64 to 1024 store-owning programs with H1 to H16. These measurements do not establish an irreducible hardware floor, an additive decomposition, or a completed pooling design.',
        zero_live_continuation_probe=gates['zero_live_continuation_probe'],
        residual_risks=['Mandatory live/cold row pools, stable nonidentity logical ownership, and occupancy-proportional continuation/replay are unimplemented.',
                        'Touched dead rows and prefix metadata remain resident; the prefix grows with chunk history at zero occupancy.',
                        'Public promotion, imports, snapshots, and explicit journals remain dense.',
                        'Heads timing failures remain failures; pooling alone cannot be assumed to remove store-core scaling.',
                        'Timer wheel remains separate from the live heap-based scan.'],
        operational_notes='Single-threaded attended execution, exact project Python, X35 GPU broker only, cards 0-6. No GPU7, .pt output, WG machinery, delegation, Frontier/Slurm or resilient training changes. X28E/X30/X31/X33 sources and tests were not edited. Existing unrelated workspace changes preserved. contact_supervisor was searched first and is unavailable.',
        retained_unsuccessful_attempts=['validation_part5_diagnostic_failure.txt: 158 passed, one new diagnostic fixture failed because its argument index was wrong.',
            'gates_part5_accounting_failure.json/txt: complete-cell samples retained; diagnostic output accounting failed before completion.',
            'gates_part5_duplicate_accounting_failure.json/txt: complete-cell samples retained; duplicate physical_slots field in diagnostic accounting failed before completion.'],
        raw_gate_receipt='gates_part5.json',raw_gate_stdout='gates_part5.txt',
        structural_cost_receipt='structural_cost_part5.json',
        measurement_note='Only the final completed run determines the part-5 verdict. Earlier incomplete runs are retained, not selected for favorable thresholds. No initialization subtraction or H1 slowdown. Independent diagnostic samples are not registered cell samples.')
    (ROOT/'summary_part5.json').write_text(json.dumps(summary,indent=2)+'\n')
    structural = dict(source_sha256=gates['source_sha256'],physical_gpu=gates['physical_gpu'],
                      measurements=gates['structural_cost'],component_h16_h1_ratios=floor_ratios,
                      note=summary['residual_cost_adjudication'])
    (ROOT/'structural_cost_part5.json').write_text(json.dumps(structural,indent=2)+'\n')
    prep = [floors['exact_fresh_preparation',h]['wall_ms_per_token'] for h in (1,16)]
    core = [floors['sequential_store_core',h]['wall_ms_per_token'] for h in (1,16)]
    probe = gates['zero_live_continuation_probe']
    report = f'''\n## Part 5 — FAIL: diagnostic adjudication, pooling still unimplemented

**This does not complete the work order or close v3.** No production cell,
backing, kernel, or replay implementation was changed. The delivered work is
an exact fresh-preparation/store-core isolation diagnostic and a six-gate
remeasurement, with a successful final parity receipt. The required live/cold
pools and nonidentity logical map remain unimplemented. This is an incomplete
delivery, not the permitted adjudication of a floor after pooling completion.

Final validation: **{counts['tests']} passed**, zero failures/errors/skips;
the unchanged 109 X31 tests and 49 existing X35 tests, plus two diagnostic
regressions. The regressions compare captured production-core decisions and
payloads with the complete scan and exercise diagnostic accounting without
collecting timings during parity. The initial diagnostic fixture indexing
failure and two subsequent accounting failures are preserved with their
intermediate validation/hash receipts. They are not final-source attestation.

| Registered gate | Final measurement | Verdict |
|---|---|---|
| C64/S2944/H4 forward >=2x | {gates['forward_speedup']:.6f}x | {'PASS' if gates['gates']['c64_s2944_h4_forward_ge_2x'] else 'FAIL'} |
| S736/T16 H16/H1 <=2 | {gates['heads_ratio_s736']:.6f}x | {'PASS' if gates['gates']['s736_t16_heads_ratio_le_2'] else 'FAIL'} |
| S11776/T16 H16/H1 <=2 | {gates['heads_ratio']:.6f}x | {'PASS' if gates['gates']['s11776_t16_heads_ratio_le_2'] else 'FAIL'} |
| Occupancy-proportional backing | Prefix backing unchanged | FAIL |
| Live-proportional memory | No live/cold pools | FAIL |
| Full oracle parity | {counts['tests']} tests, zero failures/errors/skips | PASS |

### Residual cost isolation

The premise that an O(architectural S) fresh-state allocation explains the
remaining heads ratio is not supported by the existing implementation. In an
ordinary empty-origin T16 call, `backing.scan` passes `budget=min(S,16)` to
`_tape_scan`: native payloads, metadata, free words, death diagnostics and
version maps all have physical S=16. Both architectural S736 and S11776 share
that same physical horizon. This says nothing about long continuations,
imports, explicit journals or public promotion, which still have residuals.

`structural_cost_part5.py` builds its preparation function directly from the
production `_tape_scan` AST, stopping immediately before the core launch. It
runs the actual native initialization, metadata setup, layout copies, trace
allocation, free-word initialization and causal-version preparation. The
independent core measurement prepares a NEW exact state before every timed
interval and times just the real core dispatch. Its preparation is explicitly
outside that diagnostic timing, never outside registered whole-cell timing.
The diagnostic uses independent random K=Q/V inputs, writes enabled, POP and
append logits zero; it is not a matched additive attribution of registered
whole-cell samples. A separate writes-disabled scan is an ablation, not a gate.

Exact fresh preparation: H1 **{prep[0]:.6f}**, H16 **{prep[1]:.6f} ms/token**.
Sequential store core: H1 **{core[0]:.6f}**, H16 **{core[1]:.6f} ms/token**,
ratio **{floor_ratios['sequential_store_core']:.6f}x**. The core launches one
program per store: 64 programs at H1 versus 1024 at H16, each running the
sequential T16 routing/scheduling loop. All allocations in this fresh diagnostic
use 16 physical rows. Every sample is retained. These are measured component
costs, not an irreducible hardware floor; they are neither added nor subtracted
to relabel a failed registered gate. Further kernel work could change them.

At **zero live rows**, {len(probe)} successive T16 chunks grow base-state
bytes from **{probe[0]['base_state_bytes']:,}** to **{probe[-1]['base_state_bytes']:,}**,
and physical rows from {probe[0]['physical_slots']} to {probe[-1]['physical_slots']}.
This remains a direct counterexample to occupancy-proportional backing.
No favorable fresh-start footprint overrides either structural failure.

### Parts 1–5 convergence

| Part | Integrated parity | C64/S2944/H4 forward >=2x | S11776/T16 heads <=2 | Live/cold pools / memory |
|---|---|---|---|---|
| 1 | Counterexamples only, 12 tests | NOT RUN | NOT RUN | NOT IMPLEMENTED |
| 2 | Components; no changed representation | NOT RUN | NOT RUN | NOT IMPLEMENTED |
| 3 | 141 passed | 1.879757x, FAIL | 7.407990x, FAIL | FAIL / FAIL |
| 4 | 158 passed | 2.194722x, PASS | 2.349044x, FAIL | FAIL / FAIL |
| 5 | {counts['tests']} passed; production unchanged | {gates['forward_speedup']:.6f}x, {'PASS' if gates['gates']['c64_s2944_h4_forward_ge_2x'] else 'FAIL'} | {gates['heads_ratio']:.6f}x, {'PASS' if gates['gates']['s11776_t16_heads_ratio_le_2'] else 'FAIL'} | FAIL / FAIL |

Part 1's original pure-chunk >=3x and mixed-training >=2x targets were not run;
this table tracks the later structural-forward preregistration without
retroactively changing those original verdicts. Part-5 timing differences are
remeasurements, not speedups from a new production implementation.

Residual risks: no live/cold row pools, no occupancy-driven compaction or
journal-aware nonidentity map; history-dependent continuation capacity;
dense imports/public materialization/snapshots/explicit replay; separate,
unintegrated timer wheel. No closure, new pooling parity, or universal
impossibility claim is made. Summary and raw receipts remain authoritative.

Operational scope: only X35 artifacts changed. Exact project Python and X35
broker, physical GPU {gates['physical_gpu']}; no card 7 or .pt output. The LM and
certified-arm reservations were respected by the broker. No WG service,
workers, delegation, Frontier/Slurm or resilient training changes; scheduler
Partition/QOS and resilient requirement IDs are inapplicable. Unrelated
workspace edits are preserved. `contact_supervisor` was searched first in the
callable catalog and is unavailable; no supervisor coordination was possible.
'''
    path = ROOT/'REPORT.md'
    original = path.read_text().split('\n## Part 5 —')[0]
    path.write_text(original.rstrip()+'\n'+report)
    print(json.dumps({k:summary[k] for k in ('scope_complete','registered_gates','forward_speedup','heads_ratio','heads_ratio_s736')},indent=2))


if __name__ == '__main__':
    main()
