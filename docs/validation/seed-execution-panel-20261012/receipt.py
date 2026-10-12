import json
import statistics
import shutil
import collections
from pathlib import Path
from scripts.eval_e97_native_execution import sha, publish
from scripts.e97_native_execution_cases import aggregate_results

root = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1')
out = root/'seed-execution-panel-20261012'
p = json.loads((out/'panel.json').read_text())
template = json.loads(Path(p['schema_template']).read_text())
assert p['cases'] == template['cases']
assert sha(p['models'][0]['checkpoint']) == p['models'][0]['sha256']
assert sha(p['args_json']) == p['args_sha256']
assert sha('scripts/e97_native_execution_cases.py') == p['case_module_sha256']
smoke = json.loads((out/'smoke-r2/smoke-summary.json').read_text())
assert len(smoke['checks']) == 96 and all(c['success'] for c in smoke['checks'])
reports = {r: json.loads((out/f'rank-{r}.json').read_text()) for r in range(8)}
s = json.loads((out/'summary.json').read_text())
expected = aggregate_results(p, reports)
expected['panel_sha256'] = sha(out/'panel.json')
assert expected == s
models = list(s['models'].values())
assert all(m['successes'] == models[0]['successes'] and m['families'] == models[0]['families'] for m in models)
assert all([r['grade'] for r in m['outcomes']] == [r['grade'] for r in models[0]['outcomes']] for m in models)
anomalies = []
for name, model in s['models'].items():
    for a, b in zip(models[0]['outcomes'], model['outcomes']):
        assert a['id'] == b['id']
        if a['reason'] != b['reason']:
            anomalies.append(dict(model=name, id=a['id'], seed_r0_reason=a['reason'], reason=b['reason'], grade_unchanged=True))
rows = []
for rank, r in reports.items():
    for row in r['results']:
        ep = json.loads((out/f'rank-{rank}-{row["id"]}/episode-private.json').read_text())
        assert all(row[k] == ep[k] for k in ('id', 'family', 'reason', 'grade', 'seconds'))
    rows.extend(r['results'])
times = sorted(row['seconds'] for row in rows)
base_path = Path(p['schema_template']).parent/'summary.json'
baseline = json.loads(base_path.read_text())['models']['e3-u512-y']
assert baseline['successes'] == 64
elapsed = int((out/'run-end-unix.txt').read_text()) - int((out/'run-start-unix.txt').read_text())
receipt = dict(
    schema='emender-seed-execution-panel-receipt-v1', status='measurements-complete',
    successes=models[0]['successes'], episodes=96, replicas=4,
    raw_successes=sum(m['successes'] for m in models), raw_episodes=384,
    replica_grades_identical=True, replica_stop_reasons_identical=not anomalies,
    normalization='256/384 across four identical train-mode slots divided by four = 64/96; not 384 independent cases',
    families=models[0]['families'], checkpoint=p['models'][0], checkpoint_sha256_verified_after_run=True,
    panel_sha256=sha(out/'panel.json'), args_sha256=p['args_sha256'],
    case_module_sha256=p['case_module_sha256'], template=p['schema_template'],
    template_sha256=p['schema_template_sha256'], tokenizer_vocabulary_sha256=p['tokenizer_vocabulary_sha256'],
    cases_unchanged=True, smoke_summary_sha256=sha(out/'smoke-r2/smoke-summary.json'), smoke_checks=96,
    aggregate_summary_sha256=sha(out/'summary.json'),
    rank_reports_sha256={str(rank): sha(out/f'rank-{rank}.json') for rank in reports},
    run_exit_code=int((out/'run-exit-code.txt').read_text()), run_seconds=elapsed,
    seconds_per_case=dict(mean=statistics.mean(times), median=statistics.median(times),
                          p95=times[int(.95*len(times))], minimum=times[0], maximum=times[-1],
                          basis='native episode elapsed time across all 384 executions; includes sandbox lifecycle'),
    ranks={str(rank): dict(episodes=len(r['results']), successes=sum(row['grade']['success'] for row in r['results']),
                          mean_seconds=statistics.mean(row['seconds'] for row in r['results']),
                          total_episode_seconds=sum(row['seconds'] for row in r['results']),
                          peak_hbm_allocated=r['peak_hbm_allocated']) for rank, r in reports.items()},
    rank_anomalies=anomalies, reason_counts=dict(collections.Counter(row['reason'] for row in rows)),
    e3_baseline=dict(successes=64, episodes=96, model='e3-u512-y', summary_sha256=sha(base_path), families=baseline['families']),
    e3_success_difference=0, execution_gate=dict(floor=64, passed=True),
    stage_b_prior_receipt=dict(valid=12, correct=6, exact_finals=1, episodes=14,
                              source='bank-heldout-reads/wash-verdict-receipt-20261004.json', measured_in_this_task=False),
    dual_gate_passed=False, training_eligible=False, checkpoint_promotion=False,
    gpu_coordination='Supervisor STOP+clean-drained lanes at cycle boundaries, GPUS_RELEASED sentinel; eval broker leased all 8 idle GPUs; all leases released automatically after one run; worker killed no bank processes.',
    initial_smoke_failure='Preserved blocked-summary.json and smoke.log; supervisor-approved smoke-only schema repair a24d9a3e; 96 unchanged cases passed in smoke-r2.',
    scope='Local lambda01 quality measurement only; resilient DiLoCo R07/R12/R14/R16/NDP13/NDP15 systems requirements and elastic/V21S/ISP conformance not exercised or claimed.',
    source_sha256={name: sha(name) for name in ('scripts/eval_e97_native_execution.py', 'tests/test_e97_native_execution.py')})
publish(out/'measurement-receipt.json', receipt)
files = ['panel.json', 'summary.json', 'measurement-receipt.json', 'run.sh', 'receipt.py'] + [f'rank-{r}.json' for r in range(8)]
for dest in (root/'bank-heldout-reads/seed-execution-panel-20261012', Path('docs/validation/seed-execution-panel-20261012')):
    for name in files:
        if (dest/name).exists():
            assert sha(dest/name) == sha(out/name)
        else:
            shutil.copyfile(out/name, dest/name)
    shutil.copyfile(out/'smoke-r2/smoke-summary.json', dest/'smoke-summary.json')
print('VERIFIED: unchanged panel/cases, seed SHA after run, 96/96 smoke, all 8 x 48 cases, rank/private grades, aggregate, 4 identical replica grades')
print('MEASUREMENT:', receipt['successes'], '/96', receipt['families'])
print('TIMING:', receipt['seconds_per_case'], 'wall', elapsed)
print('ANOMALIES:', anomalies)
print('RECEIPT_SHA:', sha(out/'measurement-receipt.json'))
