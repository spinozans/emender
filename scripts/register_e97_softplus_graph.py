"""One-shot X24 registration; refused after launch/output or failed zero-update gates."""
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import socket
import subprocess
import sys

from experiments.expressivity_tasks.run_e97_softplus_graph import ROOT, COMPARISON_ROOT
from experiments.expressivity_tasks.run_merrill26_separators import digest, write_json


def register():
    if ((ROOT / 'registration.json').exists() or (ROOT / 'datasets').exists()
            or (ROOT / 'receipts' / 'launch.json').exists() or (ROOT / 'e97_graph_seed0').exists()):
        raise ValueError('registration/data/launch/output exists; refuse overwrite')
    baseline_path = COMPARISON_ROOT / 'registration.json'
    baseline = json.loads(baseline_path.read_text())
    smoke = json.loads((ROOT / 'receipts' / 'smoke_e97_graph.json').read_text())
    kernel = json.loads((ROOT / 'receipts' / 'kernel_match.json').read_text())
    assert smoke['status'] == kernel['status'] == 'passed'
    assert smoke['optimizer_updates'] == kernel['optimizer_updates'] == 0
    assert smoke['tokens'] == 90904 and smoke['n'] == 300 and smoke['parameters'] == 8950272
    assert (ROOT / 'receipts' / 'tests_exit_code.txt').read_text().strip() == '0'
    sources = set(baseline['source_sha256'])
    sources.update(subprocess.check_output(['git','ls-files','ndm'],text=True).splitlines())
    sources.update(['experiments/expressivity_tasks/run_e97_softplus_graph.py',
        'scripts/launch_e97_softplus_graph.sh','scripts/watch_e97_softplus_graph.py',
        'scripts/register_e97_softplus_graph.py','tests/test_e97_softplus.py',
        'tests/test_e97_softplus_graph.py',str(baseline_path)])
    sources.update(str(p) for p in ROOT.rglob('*') if p.is_file())
    row = {**baseline, 'task':'X24', 'registered_utc':datetime.now(timezone.utc).isoformat(),
        'host':socket.gethostname(), 'python':sys.version,
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol':str(ROOT / 'REGISTRATION.md'), 'runs':['e97_graph_seed0'],
        'cards':{'e97_graph_seed0':7}, 'primary':['graph'], 'secondary_adapted':[],
        'optimizer_updates_before_registration':0,
        'comparison':{'registration':str(baseline_path),'sha256':digest(baseline_path),
            'datasets':str(COMPARISON_ROOT / 'datasets'),
            'dataset_receipts_at_registration':{str(p):digest(p)
                for p in sorted((COMPARISON_ROOT / 'datasets' / 'graph').glob('*.json'))}},
        'activation_choice':{'chosen':'softplus','softplus_c_supported_by_e97':False,
            'before_amendment_fused':['tanh','identity'],
            'before_amendment_eager':['tanh','identity','relu','softplus'],
            'gdn_phi_table_not_e97_support':['identity','tanh','relu','softplus_c','softplus'],
            'authorization':'supervisor approved fused plain-softplus extension after support audit',
            'concession_18':'softplus_c bounded below at -log(2)/2; unbounded above suffices for counting',
            'precision':'exact X23E legacy: FP32 live accumulation; bf16 replay checkpoints/scratch'},
        'architecture':{**baseline['architecture'],'e97_state':'softplus',
            'only_shape_change':'none relative to X23E graph; activation-only fused amendment'},
        'predictions':{
            'activation-binding':'softplus >=99% ID/OOD where tanh drops >=5pp ID-to-far-OOD; supports practical activation bottleneck, candidate host, mixed-ladder escape; O10 realization still open',
            'not-the-activation':'softplus also drops >=5pp OOD despite ID learning; wrapper/training/depth implicated, O10 open, tanh boundedness not practical bottleneck',
            'fails-to-train':'nonfinite logits/gradients or exploding-gradient instability; retain honest failure, separately registered X16E staging retry indicated, never automatic',
            'other':'ID failure separate; exact counts for all bins; infra and incomplete outcomes right-censored'},
        'max_bin_zero_update_smokes':{'e97_graph_seed0':smoke}, 'kernel_match':kernel,
        'versions':{p:metadata.version(p) for p in baseline['versions']},
        'source_sha256':{p:digest(p) for p in sorted(sources)},
        'dependency_sha256':{p:digest(p) for p in baseline['dependency_sha256']}}
    write_json(ROOT / 'registration.json',row)
    print(json.dumps({'registration':str(ROOT / 'registration.json'),
        'sha256':digest(ROOT / 'registration.json'),'registered_utc':row['registered_utc']}))


if __name__ == '__main__':
    register()
