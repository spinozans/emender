#!/usr/bin/env python3
"""Disjoint training-only authored discovery/recovery preparation, not SFT admission.

No evaluation trajectory is reused, no model is sampled, and no optimizer is
loaded. These are verified demonstration receipts, not a packed training dataset.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tiktoken

from scripts.e97_pi_repository_probes import repository_probes
from scripts.qualify_e97_pi_repository_probes import TEST_COMMAND, verifier
from scripts.qualify_e97_pi_native_transport import run_control
from scripts.eval_e97_native_execution import publish, sha

SEED='e97-training-only-directory-discovery-20260915-v1'


def fixtures():
    rows=[]
    for family in ('sum_all','scale_all'):
        for index in range(4):
            tag=hashlib.sha256(f'{SEED}:{family}:{index}'.encode()).hexdigest()[:10]
            module='logic_'+tag; implementation=module+'.py'; test='tests/test_'+tag+'.py'
            values=[7+index*3, -5-index, 11+index*2]
            if family=='sum_all':
                broken='def total(values):\n    return sum(values[1:])\n'
                repaired='def total(values):\n    return sum(values)\n'
                spec='total(values) returns the sum of every input value, including the first. Empty input returns zero.'
                checks=f'from {module} import total\nimport unittest\n\nclass Tests(unittest.TestCase):\n    def test_all_values(self):\n'
                for v in (values,[],[index+17],[-8,-4]):
                    checks+=f'        self.assertEqual(total({v!r}), {sum(v)!r})\n'
            else:
                broken='def scale(values, factor):\n    return [value + factor for value in values]\n'
                repaired='def scale(values, factor):\n    return [value * factor for value in values]\n'
                spec='scale(values, factor) multiplies every value by factor and preserves order. Empty input returns an empty list.'
                checks=f'from {module} import scale\nimport unittest\n\nclass Tests(unittest.TestCase):\n    def test_scaling(self):\n'
                for v,f in ((values,3+index),([],4),([-2,5],-3),([3,7],0)):
                    checks+=f'        self.assertEqual(scale({v!r}, {f}), {[x*f for x in v]!r})\n'
            recovery=index%2==1
            files={'README.md':f'# Project {tag}\n\nImplementation: {implementation}\n{spec}\nRun: {TEST_COMMAND}\n',
                   implementation:broken,test:checks}
            rows.append(dict(id=f'training-discovery-{family}-{tag}',family=family,implementation=implementation,
                files=files,expected_source=repaired,recovery=recovery,
                prompt='Repair the Python project in /testbed. Discover the actual files, use its tests to identify the problem, fix the implementation, and verify it. Leave tests and documentation unchanged.',
                origin='authored-training-only-discovery-preparation',training_eligible=False,
                authored_failure_prefix_turns=1 if recovery else 0,
                supervise_assistant_from=1 if recovery else 0,
                genuine_model_failure=False,evaluation_trajectory_reused=False))
    return rows


def validate_disjoint(candidates, protected):
    evaluation_paths={p for c in protected for p in c['files'] if p!='README.md'}
    evaluation_contents={v for c in protected for v in c['files'].values()}
    if len(candidates)!=8 or len({c['id'] for c in candidates})!=8:
        raise ValueError('fixed training preparation coverage')
    for c in candidates:
        if c['prompt'] in {x['prompt'] for x in protected}:
            raise ValueError('evaluation_prompt_reuse')
        if (set(c['files'])-{'README.md'}) & evaluation_paths or set(c['files'].values()) & evaluation_contents:
            raise ValueError('evaluation_file_reuse')
        if c['expected_source'] in {x['expected_source'] for x in protected}:
            raise ValueError('evaluation_repair_reuse')
    return dict(exact_prompt_file_and_repair_overlap=False,shared_convention='README.md',
                universal_corpus_independence_claim=False)


def control(c):
    test=next(p for p in c['files'] if p.startswith('tests/'))
    steps=[]
    if c['recovery']:
        steps.append(('str_replace_editor',dict(command='view',path='/testbed/unavailable_'+c['id'][-10:]+'.py')))
    steps.extend([
        ('execute_bash',dict(command='ls -a; cat README.md; '+TEST_COMMAND)),
        ('str_replace_editor',dict(command='view',path='/testbed/'+test)),
        ('str_replace_editor',dict(command='view',path='/testbed/'+c['implementation'])),
        ('str_replace_editor',dict(command='str_replace',path='/testbed/'+c['implementation'],
                                  old_str=c['files'][c['implementation']],new_str=c['expected_source'])),
        ('execute_bash',dict(command=TEST_COMMAND)),
        ('finish',dict(message='The implementation is fixed and the project tests pass.')),
    ])
    return dict(name=c['id'],prompt=c['prompt'],files=deepcopy(c['files']),steps=steps,
                private_analysis=None,commentaries=[None]*len(steps),
                expected={**c['files'],c['implementation']:c['expected_source']})


def main():
    p=argparse.ArgumentParser(); p.add_argument('--panel',type=Path,required=True); p.add_argument('--panel-sha',required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--pi',default=shutil.which('pi'))
    a=p.parse_args(); os.umask(0o077)
    def interrupted(signum,frame): raise TimeoutError('discovery preparation interrupted')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or sha(a.panel)!=a.panel_sha:
        raise ValueError('CPU/panel identity')
    panel=json.loads(a.panel.read_text()); cases=fixtures(); overlap=validate_disjoint(cases,repository_probes())
    a.output.mkdir(parents=True,mode=0o700,exist_ok=False)
    publish(a.output/'plan-private.json',dict(seed=SEED,cases=cases,overlap=overlap,training_eligible=False,
            model_generations=0,optimizer_updates=0,not_an_sft_dataset_authority=True))
    rows=[]
    try:
        for c in cases:
            row=run_control(panel,control(c),tiktoken.get_encoding('p50k_base'),a.output/c['id'],a.pi,
                            Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts')
            episode=json.loads((a.output/c['id']/'episode-private.json').read_text())
            test_exits=[x['result']['exit_code'] for x in episode['calls'] if x['request']['name']=='execute_bash'
                        and x['request']['arguments']['command'].endswith(TEST_COMMAND)]
            if test_exits!=[1,0]: raise ValueError('discovery preparation test sequence')
            if c['recovery'] and not episode['calls'][0]['result']['message']['content'].startswith('ERROR:'):
                raise ValueError('authored failure prefix did not actually fail')
            actual=json.loads((a.output/c['id']/'sandbox/reader.stdout').read_text())
            if verifier(panel,c,actual,a.output/(c['id']+'-verifier'))!=0 or verifier(panel,c,c['files'],a.output/(c['id']+'-negative-verifier'))!=1:
                raise ValueError('semantic verifier control failed')
            row.update(authored_failure_prefix_turns=c['authored_failure_prefix_turns'],
                       supervise_assistant_from=c['supervise_assistant_from'],genuine_model_failure=False,
                       training_eligible=False,evaluation_trajectory_reused=False)
            rows.append(row)
        publish(a.output/'summary.json',dict(status='training-only-discovery-preparation-passed',results=rows,
                model_generations=0,optimizer_updates=0,training_eligible=False,not_an_sft_dataset_authority=True,
                admission_changed=False,checkpoint_promotion=False))
        print('TRAINING_DISCOVERY_PREPARATION_PASSED',len(rows),flush=True)
    except BaseException as exc:
        publish(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),completed=rows,optimizer_updates=0))
        raise


if __name__=='__main__': main()
