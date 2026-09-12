#!/usr/bin/env python3
"""Training-only sampled rollouts, immutable rewards, and same-state repairs.

This collects candidates. It neither admits training data nor applies RL updates.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time

from scripts.build_e97_grounding_correction import fixtures
from scripts.e97_native_execution_cases import context, grade
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.e97_open_swe_native_codec import compact, encode, native_turn, render, vocabulary
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode
from scripts.eval_e97_native_execution import episode, publish, sha


def tasks(seed, count=16):
    rows = fixtures(seed, count)
    for i, c in enumerate(rows):
        # New namespace, selector keys/order, and edit deltas. No evaluation case reuse.
        c = json.loads(json.dumps(c).replace('train/', 'onpolicy/'))
        c['id'] = f'onpolicy-task-{i:03d}'
        c.pop('source_style')
        raw = json.loads(c['files'][c['path']])
        if c['family'] == 'lookup':
            keys = ['amber', 'violet', 'silver']
            j = i//4
            keys = keys[j % 3:] + keys[:j % 3]
            values = {k: hashlib.sha256(f'{seed}:{i}:{k}'.encode()).hexdigest()[:20] for k in keys}
            raw = dict(active=keys[(j+1) % 3], values=values)
            c['files'][c['path']] = json.dumps(raw)
            c['answer'] = values[raw['active']]
        c['delta'] = (1, 3, 7, 11)[i//4 % 4]
        c['expected_output'] = ({**raw, 'count': raw['count']+c['delta']} if c['family']=='edit' else None)
        if c['family']=='edit':
            c['prompt'] = c['prompt'].replace('increased by 7', f'increased by {c["delta"]}')
        c['output_path'] = c['base']+'/result.json'
        c['missing_path'] = '/testbed/'+c['base']+'/missing.json'
        c['files'] = {p.removeprefix('/testbed/'): v for p,v in c['files'].items()}
        rows[i] = c
    return rows


def freeze(args):
    cfg = json.loads(args.recipe.read_text())
    if (cfg['tasks'], cfg['rollouts_per_task'], cfg['optimizer_updates'], cfg['temperature'], cfg['top_k'], cfg['top_p']) != (16,2,0,1.,0,0.):
        raise ValueError('canary budget/sampling contract')
    if sha(cfg['base_panel']) != cfg['base_panel_sha256']:
        raise ValueError('source panel identity')
    panel = json.loads(Path(cfg['base_panel']).read_text())
    generated = tasks(cfg['seed'], cfg['tasks'])
    forbidden = {c['answer'] for c in panel['cases'] if c['family']!='edit'}
    forbidden.update(c['answer'] for c in fixtures('grounding-correction-train-20260912-v1',1024) if c['family']!='edit')
    if any(c['answer'] in forbidden for c in generated if c['family']!='edit'):
        raise ValueError('training/evaluation answer overlap')
    panel.update(schema=cfg['schema'], cases=generated, sample_untruncated_policy=True,
                 models=[dict(name='correction-y',checkpoint=cfg['checkpoint'],sha256=cfg['checkpoint_sha256'],mode=cfg['weight_mode'])],
                 config=cfg, config_sha256=sha(args.recipe), training_eligible=False,
                 scope='Training-only canary; same-family variants, not an independent benchmark')
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    publish(args.output/'panel.json',panel)
    print('ONPOLICY_PANEL_FROZEN',sha(args.output/'panel.json'),flush=True)


def load(args):
    if sha(args.panel)!=args.panel_sha:
        raise ValueError('panel identity')
    panel=json.loads(args.panel.read_text())
    if panel['schema']!='emender-e97-native-onpolicy-canary-v1' or panel['training_eligible'] is not False:
        raise ValueError('training-only candidate schema')
    return panel


def candidate(messages, tools, prefix_assistants, enc):
    pieces, meta = render(dict(messages=messages,tools=tools),enc)
    if not 0 <= prefix_assistants < meta['assistant_units']:
        raise ValueError('teacher suffix coverage')
    kept=[]; seen=0
    for text,target in pieces:
        kept.append((text,target and seen>=prefix_assistants))
        if target: seen+=1
    lengths,_=vocabulary(enc)
    tokens,mask,text=encode(kept,enc,lengths)
    return dict(schema='emender-native-onpolicy-masked-candidate-v1',training_eligible=False,
                prefix_assistants_unsupervised=prefix_assistants,assistant_units=seen,
                token_ids=tokens.tolist(),assistant_mask=mask.tolist(),messages=messages,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),targets=int(mask.sum()))


def observed_json(text, path):
    """Parse the real full cat-n observation without altering stored evidence."""
    header=f"Here's the result of running `cat -n` on {path}:\n"
    if not text.startswith(header):
        raise ValueError('unexpected editor view header')
    lines=[]
    for i,line in enumerate(text[len(header):].splitlines(),1):
        match=re.fullmatch(r'\s*([0-9]+)\t(.*)',line)
        if match is None or int(match[1])!=i:
            raise ValueError('incomplete or noncontiguous editor view')
        lines.append(match[2])
    return json.loads('\n'.join(lines))


def repair(sandbox, messages, original, case, panel, enc, output):
    """Never reset the sandbox, replace an observation, or change original reward."""
    calls=copy.deepcopy(original['calls']); prefix=copy.deepcopy(messages)
    if original['grade']['success']:
        record=candidate(prefix,panel['tools'],0,enc)
        publish(output/'candidate-private.json',record)
        return dict(kind='autonomous-success',verified=True,candidate_sha256=sha(output/'candidate-private.json'))
    if (not original['grade']['checks']['source_files_unchanged'] or
            original.get('reason') in ('protocol_error','host_tool_deadline','invalid_tool_request') or
            any('dispatch_error' in c for c in calls)):
        return dict(kind='excluded',verified=False,reason='changed input or unresolved dispatch')
    # A rejected finish changes no tool state. Replace only that unexecuted final
    # decision, retaining the original finish in the autonomous episode evidence.
    replaced_finish=original['final'] is not None
    if replaced_finish:
        if prefix[-1]['role']!='assistant' or 'Action: finish\n' not in native_turn(prefix[-1]):
            raise ValueError('rejected finish boundary')
        prefix.pop()
    n_prefix=sum(m['role']=='assistant' for m in prefix)
    e=NativeEpisode(panel['tools'],enc)
    for m in prefix:e.append_source_message(m)
    sandbox.resume_for_continuation()
    teacher_calls=[]; deadline=time.monotonic()+panel['config']['teacher_seconds']
    final=None; verified=False; reason=None
    def action(name,args):
        nonlocal final
        if time.monotonic()>deadline:raise ValueError('teacher deadline')
        message=dict(role='assistant',content=None,reasoning_content=None,think=None,
                     tool_calls=[dict(type='function',function=dict(name=name,arguments=compact(args)))])
        turn=e.accept_generated_turn(native_turn(message))
        if name=='finish':final=args['message'];return ''
        reply=sandbox.request('execute',call=turn.backend_call())
        if 'dispatch_error' in reply:raise ValueError('teacher dispatch')
        item=dict(request=turn.backend_call(),result=reply['result'],origin='verified-teacher')
        teacher_calls.append(item); e.append_observation(reply['result']['message'])
        return reply['result']['message']['content']
    def view(path):return action('str_replace_editor',dict(command='view',path=path))
    try:
        if case['family']=='recovery' and not calls:
            if not view(case['missing_path']).startswith('ERROR:'):raise ValueError('expected missing file')
        raw=case['files'][case['path'].removeprefix('/testbed/')]
        observation=view(case['path'])
        if observation.startswith('ERROR:') or raw not in observation:
            raise ValueError('observed input mismatch')
        observed=json.loads(observation[observation.index(raw):observation.index(raw)+len(raw)])
        if case['family']=='lookup':answer=observed['values'][observed['active']]
        elif case['family']=='recovery':
            dest=observed['active_path'];raw=case['files'][dest.removeprefix('/testbed/')]
            text=view(dest)
            if text.startswith('ERROR:') or raw not in text:raise ValueError('observed recovery mismatch')
            answer=json.loads(text[text.index(raw):text.index(raw)+len(raw)])['value']
        else:
            program=f'import json,os; d=json.load(open({case["path"]!r})); '
            if case['family']=='sum':
                answer=str(observed['left']+observed['right']);program+='print(d["left"]+d["right"])'
            else:
                dest='/testbed/'+case['output_path'];answer='done'
                program+=(f'd["count"]+={case["delta"]}; '
                          f'f=os.fdopen(os.open({dest!r},os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600),"w"); '
                          'json.dump(d,f); f.close(); print("written")')
            text=action('execute_bash',dict(command='python -c '+shlex.quote(program),timeout=10))
            if teacher_calls[-1]['result']['exit_code']!=0:raise ValueError('teacher shell failed')
            if case['family']=='sum' and text.splitlines()[0]!=answer:raise ValueError('sum output mismatch')
            if case['family']=='edit':
                if observed_json(view(dest),dest)!=case['expected_output']:
                    raise ValueError('written output observation mismatch')
        if answer!=case['answer']:raise ValueError('independent oracle answer mismatch')
        action('finish',dict(message=answer))
        snapshot=sandbox.snapshot(list(case['files'])+([case['output_path']] if case['expected_output'] is not None else []),label='teacher')
        verdict=grade(case,final,calls+teacher_calls,snapshot)
        if not verdict['success']:raise ValueError('teacher continuation did not satisfy original task')
        record=candidate(e.source_messages(),panel['tools'],n_prefix,enc)
        publish(output/'candidate-private.json',record)
        verified=True
    except ValueError as exc:
        reason=str(exc)
    publish(output/'teacher-private.json',dict(verified=verified,reason=reason,calls=teacher_calls,
            messages=e.source_messages(),replaced_rejected_finish=replaced_finish,
            original_reward=int(original['grade']['success']),same_container_id=sandbox.identity))
    return dict(kind='teacher-repair',verified=verified,reason=reason,
                candidate_sha256=sha(output/'candidate-private.json') if verified else None)


def preflight(args):
    import tiktoken
    panel=load(args);enc=tiktoken.get_encoding('p50k_base');results=[]
    for c in panel['cases']:
        out=args.output/c['id'];e=NativeEpisode(panel['tools'],enc)
        e.append_source_message(context('system',panel['system']));e.append_source_message(context('user',c['prompt']))
        calls=[]
        with NativeSandbox(panel,out) as sandbox:
            sandbox.request('setup',files=c['files'])
            path=c['missing_path'] if c['family']=='recovery' else '/testbed/absent-for-teacher-preflight.json'
            for _ in range(2):
                msg=dict(role='assistant',content=None,reasoning_content=None,think=None,
                         tool_calls=[dict(type='function',function=dict(name='str_replace_editor',arguments=compact(dict(command='view',path=path))))])
                t=e.accept_generated_turn(native_turn(msg));r=sandbox.request('execute',call=t.backend_call())['result']
                calls.append(dict(request=t.backend_call(),result=r));e.append_observation(r['message'])
            final='wrong-preflight-answer'
            msg=dict(role='assistant',content=None,reasoning_content=None,think=None,
                     tool_calls=[dict(type='function',function=dict(name='finish',arguments=compact(dict(message=final))))])
            e.accept_generated_turn(native_turn(msg))
            snapshot=sandbox.snapshot(list(c['files'])+([c['output_path']] if c['expected_output'] is not None else []))
            original=dict(calls=calls,final=final,snapshot=snapshot,grade=grade(c,final,calls,snapshot))
            publish(out/'preflight-original-private.json',dict(messages=e.source_messages(),**original))
            result=repair(sandbox,e.source_messages(),original,c,panel,enc,out)
            if not result['verified']:raise ValueError('authored failed-state teacher preflight failed')
            results.append(dict(id=c['id'],**result))
    publish(args.output/'summary.json',dict(status='passed',authored_interventions=True,autonomous_successes=0,results=results))
    print('ONPOLICY_TEACHER_PREFLIGHT_PASSED',len(results),flush=True)


def run(args):
    import torch
    import tiktoken
    from ndm.e97 import load_e97_checkpoint
    panel=load(args);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('eight fixed actors required')
    target=panel['models'][0]
    if sha(target['checkpoint'])!=target['sha256'] or sha(panel['args_json'])!=panel['args_sha256']:
        raise ValueError('model input identity')
    torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=panel['args_json'],device=torch.device('cuda',local),
                               dtype=torch.bfloat16,weight_mode=target['mode'],use_triton=True,mmap=True)
    loaded.model.eval();enc=tiktoken.get_encoding('p50k_base');results=[]
    for i,c in enumerate(panel['cases']):
        for attempt in range(2):
            index=2*i+attempt
            if index%8!=rank:continue
            case={**c,'id':c['id']+f'-sample-{attempt}'};out=args.output/case['id']
            seed=int.from_bytes(hashlib.sha256(f'{panel["config"]["seed"]}:{index}'.encode()).digest()[:4],'little')
            torch.manual_seed(seed)
            def continuation(sandbox,messages,original):
                return repair(sandbox,messages,original,case,panel,enc,out)
            row=episode(loaded,case,panel,enc,out,continuation=continuation)
            results.append(dict(id=case['id'],task_id=c['id'],seed=seed,reward=int(row['autonomous_success']),
                                family=c['family'],reason=row['reason'],continuation=row['continuation'],
                                episode_sha256=sha(out/'episode-private.json')))
            print('ONPOLICY_ROLLOUT',case['id'],int(row['autonomous_success']),row['continuation']['kind'],flush=True)
    publish(args.output/f'rank-{rank}.json',dict(rank=rank,panel_sha256=args.panel_sha,results=results))


def aggregate(args):
    import math
    panel=load(args);rows=[]
    for rank in range(8):
        r=json.loads((args.output/f'rank-{rank}.json').read_text())
        if r['rank']!=rank or r['panel_sha256']!=args.panel_sha:raise ValueError('rank identity')
        expected={c['id']+f'-sample-{a}' for i,c in enumerate(panel['cases']) for a in range(2) if (2*i+a)%8==rank}
        if {x['id'] for x in r['results']}!=expected or len(r['results'])!=len(expected):raise ValueError('rank coverage')
        rows.extend(r['results'])
    for row in rows:
        out=args.output/row['id'];ep=json.loads((out/'episode-private.json').read_text())
        if sha(out/'episode-private.json')!=row['episode_sha256'] or ep['assisted'] or ep['supplied_calls']:
            raise ValueError('autonomous attribution')
        if row['reward']!=int(ep['grade']['success']):raise ValueError('reward changed by teacher')
        for turn in ep['generations']:
            trace=turn['sampling'];logs=trace.get('selected_logprobs',[])
            if len(logs)!=len(turn['token_ids']) or any(not math.isfinite(v) or v>1e-6 for v in logs):
                raise ValueError('sampled log probability coverage')
        for f in ('cleanup.json','reader-cleanup.json'):
            if not json.loads((out/f).read_text())['removed']:raise ValueError('container cleanup')
        correction=row['continuation']
        if correction['verified'] and sha(out/'candidate-private.json')!=correction['candidate_sha256']:
            raise ValueError('candidate binding')
        if correction['kind']=='teacher-repair':
            t=json.loads((out/'teacher-private.json').read_text())
            before=json.loads((out/'container-before.json').read_text())
            if t['same_container_id']!=before['Id'] or t['original_reward']!=row['reward']:
                raise ValueError('same-state/reward identity')
            if correction['verified']:
                if not json.loads((out/'teacher/reader-cleanup.json').read_text())['removed']:
                    raise ValueError('teacher snapshot reader cleanup')
    groups={c['id']:[r['reward'] for r in rows if r['task_id']==c['id']] for c in panel['cases']}
    mixed=sum(len(set(v))>1 for v in groups.values())
    repairs=sum(r['continuation']['kind']=='teacher-repair' and r['continuation']['verified'] for r in rows)
    result=dict(schema=panel['schema'],status='measurements-complete',panel_sha256=args.panel_sha,
                autonomous_rollouts=len(rows),autonomous_successes=sum(r['reward'] for r in rows),
                verified_failed_state_repairs=repairs,mixed_reward_task_groups=mixed,reward_groups=groups,
                family_successes={f:sum(r['reward'] for r in rows if r['family']==f) for f in ('lookup','sum','edit','recovery')},
                rollout_signal_ready=mixed>=1 and repairs>=1,rl_optimizer_ready=False,
                rl_blocker='actor/trainer log-probability, masking and optimizer path not qualified',
                training_eligible=False,optimizer_updates=0,automatic_expansion=False,checkpoint_promotion=False,results=rows)
    publish(args.output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'},sort_keys=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('freeze');q.add_argument('--recipe',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    for command in ('preflight','run','aggregate'):
        q=s.add_parser(command);q.add_argument('--panel',type=Path,required=True);q.add_argument('--panel-sha',required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)
