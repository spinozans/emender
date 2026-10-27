"""CPU-only protocol/completeness audit; hypotheses are outcomes, not gates."""
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parent
root = R.parent
load = lambda p: json.loads(p.read_text())
sweep = load(R/'sweep.json')
selected = sweep['selected']
initial = []
checks = {}
for arm in ('0','0.001','0.01','0.1'):
    final_step = 3052 if arm == selected else 611
    train = load(R/f'train_{arm}.json')
    rows = [json.loads(line) for line in (root/'runs'/arm/'curve.jsonl').read_text().splitlines()]
    checks[f'{arm}:completed'] = train['status'] == 'completed' and train['steps'] == final_step
    checks[f'{arm}:steps'] = [row['step'] for row in rows] == list(range(1,final_step+1))
    checks[f'{arm}:finite_replay'] = all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) and row['replay_node']=='ReplayBackward' for row in rows)
    checks[f'{arm}:tokens'] = train['tokens'] == final_step*32768
    checks[f'{arm}:broker'] = train['physical_gpu'] in ('0','1','5','6')
    parameters = load(R/f'parameters_{arm}.json')
    checks[f'{arm}:parameters'] = parameters['relative_total_difference'] < .001
    checkpoints = (0,64,512,611,1536,3052) if arm == selected else (0,64,512,611)
    for step in checkpoints:
        row = load(R/f'eval_{arm}_{step}.json')
        checks[f'{arm}:{step}:telemetry'] = len(row['stories'])==16 and all(
            story['telemetry'][0]['slot_usage_status'].startswith('shadow replay parity passed') and
            len(story['telemetry'][0]['slot_usage_js'])==6 and
            len(story['telemetry'][0]['read_cosine_pairs'])==6 for story in row['stories'])
    row = load(R/f'eval_{arm}_0.json')
    initial.append((row['validation']['nll'],[s['telemetry'][0]['read_cosine_pairs'] for s in row['stories']]))
checks['matched_initialization'] = all(row==initial[0] for row in initial)
base = sweep['rows']['0']['validation']['perplexity']
eligible = [a for a,row in sweep['rows'].items() if row['validation']['perplexity']<=1.02*base]
def correlation(arm):
    return sum(s['telemetry'][0]['read_cosine_mean'] for s in sweep['rows'][arm]['stories'])/16
checks['selection_rule'] = selected == min(eligible,key=lambda a:(correlation(a),float(a)))
probe = load(R/'winner_probe.json')
checks['final_probes'] = (probe['arm']==selected and len(probe['stories'])==16 and
    [row['k'] for row in probe['depth']]==[16,32,64,128,256,512,1152] and
    [row['context'] for row in probe['context']]==[512,1024,2048,4096] and
    all(row['windows']==2 for row in probe['context']))
result = dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
              hypotheses=load(R/'summary.json'),note='Prediction failures are retained as scientific results, not protocol successes.')
(R/'artifact_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
assert all(checks.values())
