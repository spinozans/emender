"""Rebuild the final part-6 ledger from retained receipts; stdlib only."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
CUSTOM = {'initialize_free_words', 'scan_kernel', 'materialize_state',
          'setup_kernel', 'rebase', 'reverse_kernel', 'index_rows'}


def main():
    load = lambda name: json.loads((ROOT / name).read_text())
    cases = list(ET.parse(ROOT / 'part6_final_source_parity.xml').getroot().iter('testcase'))
    parity = dict(tests=len(cases), failures=sum(c.find('failure') is not None for c in cases),
                  errors=sum(c.find('error') is not None for c in cases),
                  skipped=sum(c.find('skipped') is not None for c in cases))
    fresh = load('part6_final_fresh.json')
    for slots in (8192, 8224):
        fresh.extend(load(f'part6_final_fresh_s{slots}.json'))
    assert all(r['status'] == 'completed' for r in fresh)
    by_shape = {(r['arm'], r['tokens'], r['slots']): r for r in fresh}
    gates = []

    def gate(name, value, limit, scope):
        gates.append(dict(name=name, value=value, limit=limit, scope=scope,
                          status='passed' if value <= limit else 'failed'))

    gate('exactness', parity['failures'] + parity['errors'] + parity['skipped'], 0,
         f'{parity["tests"]} collected tests; original 143 plus 36 regressions')
    for tokens, limit in ((512, 3.), (4096, 2.)):
        tape = by_shape['tape_lm', tokens, 2944]
        base = by_shape['base_lm', tokens, 2944]
        gate(f'B64/T{tokens}', tape['median_ms'] / base['median_ms'], limit, 'matched whole-LM wall')
    for a, b in ((2944, 5888), (5888, 11776), (8192, 8224)):
        first, second = (by_shape['tape_lm', 512, s] for s in (a, b))
        gate(f'fresh step {a}->{b}', second['median_ms'] / first['median_ms'], 1.5, 'B64/T512 fresh')
        gate(f'fresh replay {a}->{b}', second['component_median_ms']['replay_backward'] /
             first['component_median_ms']['replay_backward'], 1.5, 'B64/T512 fresh')
    decomposition = []

    def profile_gates(profile, label):
        names = ('x36::scan_forward', 'x36::replay_backward')
        activities = sum(profile[name]['activity_count'] for name in names)
        custom = sum(count for name in names for kernel, count in profile[name]['counts'].items()
                     if kernel in CUSTOM)
        gate(f'inclusive {label}', activities, 10, 'inclusive scan/replay')
        gate(f'custom {label}', custom, 10, 'narrow custom kernels')
        unexpected = [e for e in profile['steady_sync_runtime'] if e['name'] != 'cudaDeviceSynchronize']
        boundaries = sum(e['name'] == 'cudaDeviceSynchronize' for e in profile['steady_sync_runtime'])
        gate(f'steady sync/DtoH {label}', len(unexpected) + len(profile['dtoh']) + abs(boundaries - 2),
             0, 'two measurement boundaries excluded; metadata transfers retained')
        return dict(label=label, inclusive=activities, custom=custom,
                    unexpected_syncs=unexpected, dtoh=profile['dtoh'],
                    components={name: profile[name]['component_device_busy_ms'] for name in names})

    for r in fresh:
        if r['arm'] != 'tape_lm':
            continue
        d = profile_gates(r['profile'], f'fresh T{r["tokens"]}/S{r["slots"]}')
        d.update(tokens=r['tokens'], slots=r['slots'],
                 allocator_cpu_ms=r['scan_replay_host_allocation_cpu_ms'],
                 journal_ablation_ms=r['journal_increment_estimate_ms'],
                 compiler=r['reverse_compiler_stats'])
        decomposition.append(d)
    continued = []
    for label in ('live17', 'live256', 'live1024', 'live2800', 'near_budget'):
        receipt = load(f'part6_hot_{label}.json')
        assert len(receipt['results']) == 10 and len(receipt['gates']) == 6
        for g in receipt['gates']:
            scope = 'fixed-live cell control' if g['occupancy_matched'] else 'growing-occupancy stress; literal gate retained'
            for metric, value in g['ratios'].items():
                gate(f'{label}/{g["mode"]} {metric} {g["slots"]}', value, 1.5, scope)
        continued.append(dict(label=label, gates=receipt['gates'],
            peak_allocated_bytes=max(s['peak_allocated_bytes'] for r in receipt['results'] for s in r['raw_ms']),
            peak_reserved_bytes=max(s['peak_reserved_bytes'] for r in receipt['results'] for s in r['raw_ms']),
            points=len(receipt['results']),
            rows=[dict(mode=r['mode'], slots=r['slots'], live=r['live'],
                       median_ms=r['median_ms'], compiler=r['compiler'],
                       allocated_rows=r['pool']['allocated_rows'],
                       adjoint_bytes=r['pool']['adjoint_bytes'],
                       hot_payload_bytes=r['pool']['hot_payload_bytes']) for r in receipt['results']]))
    profile_receipt = load('part6_hot_profile.json')
    continued_profile = next(r for r in profile_receipt['results'] if 'profile' in r)
    decomposition.append(profile_gates(continued_profile['profile'], 'continued L256/T512/S11776'))
    curves = load('part6_final_training_slice.json')
    for arm in ('base_lm', 'tape_lm'):
        gate(f'2M curve {arm}', curves[arm]['max_loss_error'], 1e-5, '512 steps / 2,097,152 tokens')
    return dict(part=6, status='failed' if any(g['status'] == 'failed' for g in gates) else 'passed',
                parity=parity, gates=gates, continued=continued, decomposition=decomposition,
                limitations=['Nonempty-entry sizing uses two host metadata transfers per short chunk.',
                             'Nonempty-entry chunks longer than 512 retain dense replay.',
                             'Native cold backing, required native-gradient backing, integer maps and logical-S reductions remain dense.',
                             'No hardware or irreducible structural floor is established.'])


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
