"""Independent, locked, atomic CMA-ES instance for one baseline family.

One outstanding lease request per family; failed candidates stop this instance,
remain unscored, and retry from scratch on restart. Pure parent is untouched.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from .cmaes import CMAES
from .velocity_state import atomic_json, save_generation, restore_generation
from .velocity_parent import cleanup_previous, process_receipt
from .velocity_train import sha
from .velocity_baseline_search import ROOT, center, decode, verify_search


def launch(arm, policy, path, registration):
    path.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (path/f'launch{attempt}').exists(): attempt += 1
    output = path/f'attempt{attempt}'
    receipts = path/f'launch{attempt}'
    receipts.mkdir()
    atomic_json(path/'policy.json', asdict(policy))
    env = dict(os.environ, X42_LAUNCH_RECEIPTS=str(receipts))
    command = ['bash', str(ROOT/'baseline_search/run_leased.sh'), '-m',
        'experiments.expressivity_tasks.x42_pile_scale.velocity_baseline_train',
        '--arm', arm, '--policy', str(path/'policy.json'),
        '--registration', str(registration), '--output', str(output)]
    atomic_json(receipts/'queued.json', dict(arm=arm, command=command,
        registration_sha256=sha(registration), queued=time.time(),
        status='waiting_for_broker', eligible_cards=[1,5,6], conditional_card=0))
    with (path/f'parent_console{attempt}.txt').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    return child, output


def wait_result(child, output, arm, digest, policy):
    status = child.wait()
    if status: raise RuntimeError(f'{arm} candidate failed: exit={status} output={output}')
    result = json.loads((output/'result.json').read_text())
    if (result.get('status') != 'completed' or result.get('score_eligible') is not True
        or result.get('train_seconds', 0) < 2700 or result.get('train_budget_seconds') != 2700
        or result.get('batch') != 64 or result.get('context') != 2048
        or result.get('eval_windows') != 24 or result.get('arm') != arm
        or result.get('registration_sha256') != digest or result.get('policy') != asdict(policy)
        or result.get('physical_gpu') not in ('0','1','5','6')
        or result.get('score') != result.get('nll_heldout')):
        raise RuntimeError('ineligible baseline endpoint')
    import math
    if not math.isfinite(result['score']) or result['score'] < 0:
        raise RuntimeError('invalid baseline score')
    return result


def anchor_reference(arm, out):
    path = ROOT/'receipts/velocity_baseline_anchors'/arm/'result.json'
    if not path.exists():
        atomic_json(out/'generation0.json', dict(status='pending', source=str(path)))
        return
    result = json.loads(path.read_text())
    if (result.get('status') != 'completed' or not result.get('score_eligible')
        or result.get('train_seconds',0) < 2700):
        raise RuntimeError('anchor exists but is not eligible')
    atomic_json(out/'generation0.json', dict(status='reference_only',
        source=str(path), sha256=sha(path), result=result))


def run(arm, registration, output, generations=12):
    if generations < 1: raise ValueError('positive absolute generation cap required')
    registration = Path(registration)
    protocol = verify_search(registration, arm)
    digest = sha(registration)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    with (out/'parent.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cleanup_previous(out)
        anchor_reference(arm, out)
        state = out/'generation.json'
        if state.exists(): es, completed = restore_generation(state, digest)
        else:
            es = CMAES(center(arm), population=protocol['population'], seed=protocol['cma_seed'])
            es.ask(); es.phase='asked'; completed={}
            save_generation(state, es, digest, completed)
        while es.generation < generations:
            if es.phase == 'told':
                es.ask(); es.phase='asked'; completed={}
                save_generation(state, es, digest, completed)
            generation = es.generation+1
            atomic_json(out/'parent.json', dict(pid=os.getpid(), arm=arm,
                status='running', generation=generation, generations=generations,
                registration_sha256=digest, updated=time.time()))
            for i in range(es.population):
                if str(i) in completed: continue
                anchor_reference(arm, out)
                policy = decode(arm, es.x[i])
                child, endpoint = launch(arm, policy, out/f'g{generation}_c{i:02}', registration)
                atomic_json(out/'active.json', process_receipt(child, endpoint))
                try:
                    result = wait_result(child, endpoint, arm, digest, policy)
                    completed[str(i)] = result
                    save_generation(state, es, digest, completed)
                    atomic_json(out/'active.json', {})
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait()
                    raise
            es.tell([completed[str(i)]['score'] for i in range(es.population)])
            es.phase='told'
            save_generation(state, es, digest, completed)
            atomic_json(out/f'g{generation}_summary.json', dict(generation=generation,
                winner=min(completed, key=lambda i: completed[i]['score']), completed=completed))
        anchor_reference(arm, out)
        atomic_json(out/'parent.json', dict(pid=os.getpid(), arm=arm,
            status='generation_cap_reached', generation=es.generation,
            generations=generations, updated=time.time()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm', choices=('transformer','gdn2'), required=True)
    p.add_argument('--registration', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--generations', type=int, required=True)
    a=p.parse_args()
    if os.sys.executable != '/home/erikg/emender/.venv/bin/python': raise RuntimeError('approved interpreter required')
    def terminate(signum, frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, terminate)
    try: run(a.arm, a.registration, a.output, a.generations)
    except BaseException:
        atomic_json(Path(a.output)/'parent_failure.json', dict(status='failed',
            arm=a.arm, traceback=traceback.format_exc(), updated=time.time()))
        raise

if __name__=='__main__': main()
