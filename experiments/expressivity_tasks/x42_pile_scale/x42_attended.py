"""X42 attended launch adapters; registered compute sources stay byte-identical.

Only the train function's physical-card admission tuple is extended for the
explicitly authorized ablations. Every other AST node remains unchanged.
"""
import argparse
import ast
from dataclasses import asdict, replace
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

from . import velocity_train as train
from . import velocity_parent as parent
from .velocity_candidate import Policy
from .velocity_state import atomic_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'receipts/lineage_suspects'
REGISTRATION = ROOT / 'velocity_registration.json'


def admit_ablation_card():
    source = inspect.getsource(train.train)
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
               and ast.unparse(node.left) == "os.environ.get('GPU_LEASE_HELD')"
               and len(node.comparators) == 1
               and ast.literal_eval(node.comparators[0]) == ('1', '5', '6')]
    if len(matches) != 1:
        raise RuntimeError('expected exactly one physical-card admission guard')
    matches[0].comparators[0] = ast.Tuple(elts=[ast.Constant('2'), ast.Constant('4')], ctx=ast.Load())
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, str(Path(train.__file__)), 'exec'), train.__dict__, namespace)
    return namespace['train']


class Budget4Policy(Policy):
    @property
    def budgets(self):
        return tuple(4 * value for value in super().budgets)


def ablate(variant):
    train.verify_registration(REGISTRATION.relative_to(Path.cwd()))
    audit = json.loads((ROOT/'receipts/pure_v2_architecture_audit/audit.json').read_text())
    baseline = Policy(**audit['best_policy'])
    policy = Budget4Policy(**asdict(baseline)) if variant == 'budgets4' else replace(baseline, w_in_width=256)
    out = OUT/variant
    amendment = dict(variant=variant, status='running', started=time.time(), pid=os.getpid(),
        registered_compute_sha256=train.sha(train.__file__), registration_sha256=train.sha(REGISTRATION),
        adapter_sha256=train.sha(__file__), baseline_result=audit['best_result'],
        baseline_result_sha256=train.sha(audit['best_result']), baseline_score=audit['best_score'],
        baseline_budgets=baseline.budgets, effective_budgets=policy.budgets,
        policy=asdict(policy), physical_gpu=os.environ['GPU_LEASE_HELD'],
        admission_only_amendment="train physical-card guard ('1','5','6') -> ('2','4'); all other AST nodes unchanged",
        budget_serialization_note='budgets4 keeps baseline policy fields; subclass multiplies each rounded per-head budget exactly by four',
        preregistered_material_improvement_nll=0.1,
        interpretation='Single seed wall-clock comparison; reduced throughput may offset capacity gains.')
    atomic_json(OUT/f'{variant}_amendment.json', amendment)
    try:
        result = admit_ablation_card()(policy, REGISTRATION, out)
        result.update(effective_budgets=policy.budgets, attended_amendment_sha256=train.sha(OUT/f'{variant}_amendment.json'))
        atomic_json(out/'result.json', result)
        base = json.loads(Path(audit['best_result']).read_text())
        atomic_json(OUT/f'{variant}_comparison.json', dict(status='completed',
            baseline_score=base['score'], score=result['score'], score_improvement=base['score']-result['score'],
            baseline_nll=base['nll_heldout'], nll=result['nll_heldout'],
            nll_improvement=base['nll_heldout']-result['nll_heldout'],
            materially_improved=base['nll_heldout']-result['nll_heldout']>=0.1,
            per_head=result['per_head'], effective_budgets=policy.budgets,
            refusal_definition='R-CAP append requests merged at capacity; per_head.r_cap_fire_rate',
            result_sha256=train.sha(out/'result.json')))
    except BaseException:
        import traceback
        atomic_json(OUT/f'{variant}_failure.json',dict(status='failed',traceback=traceback.format_exc()))
        raise


def resume():
    original = parent.launch
    def launch(policy, path, registration):
        # Reuse original attempt management, replacing only its shell launcher.
        # The module-local subprocess proxy leaves all other processes alone.
        path.mkdir(parents=True, exist_ok=True)
        attempt=1
        while (path/f'attempt{attempt}').exists(): attempt+=1
        output=path/f'attempt{attempt}'
        atomic_json(path/'policy.json', asdict(policy))
        env=dict(os.environ, X42_LAUNCH_RECEIPTS=str(path/f'launch{attempt}'))
        with (path/f'parent_console{attempt}.txt').open('w') as log:
            child=subprocess.Popen(['bash',str(ROOT/'x42_attended_leased.sh'),'1',
                '-m','experiments.expressivity_tasks.x42_pile_scale.velocity_train',
                '--policy',str(path/'policy.json'),'--registration',str(registration),'--output',str(output)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return child, output
    parent.launch=launch
    state=ROOT/'runs/velocity_production_v1/generation.json'
    atomic_json(OUT/'resume.json',dict(status='starting',generation_before_sha256=train.sha(state),
        registration_sha256=train.sha(REGISTRATION),physical_card_allowlist=[1],
        note='Registered parent still reports legacy cards [1,5,6]; every child is restricted to broker card 1 by this adapter.',pid=os.getpid()))
    parent.run(REGISTRATION.relative_to(Path.cwd()), ROOT/'runs/velocity_production_v1', generations=12)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['budgets4','read_width256','resume'])
    args=parser.parse_args()
    if args.mode=='resume': resume()
    else: ablate(args.mode)
