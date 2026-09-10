"""Run only inside the restricted native-runtime qualification container.

Fixtures are authored here, never commands from a training trajectory.
"""
import asyncio
import base64
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import time
import tiktoken

from scripts.e97_openhands_native_backend import OpenHandsNativeBackend
from scripts.e97_open_swe_native_codec import native_turn,vocabulary
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode


def numbered_rows(text):
    return [(int(m[1]),m[2]) for line in text.splitlines()
            if (m:=re.match(r'^\s*(\d+)[\t ]+(row\d{4})$',line))]


def context(role,text):
    return {'role':role,'content':text,'reasoning_content':None,'think':None,'tool_calls':None}


async def qualify(bundle):
    expected=json.loads((bundle/'expected-tools.json').read_text())
    config=json.loads((bundle/'tokenizer.json').read_text())
    encoding=tiktoken.Encoding(name='p50k_base-frozen-native',pat_str=config['pat_str'],
        mergeable_ranks={base64.b64decode(k):v for k,v in config['mergeable_ranks']},special_tokens=config['special_tokens'])
    if vocabulary(encoding)[1]!=expected['tokenizer_vocabulary_sha256']:raise ValueError('tokenizer_identity')
    backend=OpenHandsNativeBackend();calls=[];checks=[]
    actual={t['function']['name']:t for t in backend.declared_tools()}
    wanted={t['function']['name']:t for t in expected['tools']}
    if actual!=wanted:raise ValueError('upstream_tool_schema_mismatch: '+str([k for k in wanted if actual.get(k)!=wanted[k]]))
    if importlib.metadata.version('openhands-aci')!='0.3.1':raise ValueError('editor_version')
    source_hashes={}
    for obj in (type(backend.executor),type(backend.formatter)):
        p=Path(inspect.getfile(obj));source_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    episode=NativeEpisode(expected['tools'],encoding)
    episode.append_source_message(context('system','Use the declared original source tools.'))
    episode.append_source_message(context('user','Exercise the fixture files and persistent terminal; then finish.'))
    async def call(name,args):
        source={'role':'assistant','reasoning_content':'Private fixture note.','content':'Checking source-native tools.',
                'think':None,'tool_calls':[{'type':'function','function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False)}}]}
        turn=episode.accept_generated_turn(native_turn(source));request=turn.backend_call()
        if request!={'name':name,'arguments':args}:raise ValueError('adapter_changed_arguments')
        start=time.monotonic();result=await backend.execute(**request)
        episode.append_observation(result['message'])
        calls.append({'name':name,'arguments':args,'seconds':time.monotonic()-start,**result})
        return result
    try:
        await backend.start()
        path=Path('/testbed/lines.txt');path.write_text(''.join(f'row{i:04d}\n' for i in range(1,501)))
        full=await call('str_replace_editor',{'command':'view','path':str(path)})
        rows=numbered_rows(full['message']['content'])
        assert rows==[(i,f'row{i:04d}') for i in range(1,501)]
        checks.append('full_file_view_500_lines')
        opened=await call('str_replace_editor',{'command':'view','path':str(path),'view_range':[400,-1]})
        assert numbered_rows(opened['message']['content'])==[(i,f'row{i:04d}') for i in range(400,501)]
        checks.append('open_ended_view_to_eof')
        finite=await call('str_replace_editor',{'command':'view','path':str(path),'view_range':[21,23]})
        assert numbered_rows(finite['message']['content'])==[(i,f'row{i:04d}') for i in range(21,24)]
        checks.append('finite_inclusive_view')
        old=path.read_bytes()
        bad=await call('str_replace_editor',{'command':'view','path':str(path),'view_range':[0,2]})
        assert bad['message']['content'].startswith('ERROR:') and 'view_range' in bad['message']['content'] and path.read_bytes()==old
        checks.append('authentic_invalid_range_error')
        directory=await call('str_replace_editor',{'command':'view','path':'/testbed'})
        assert directory['action_type']=='FileReadAction' and 'lines.txt' in directory['message']['content']
        checks.append('directory_view_not_rewritten_to_bash')
        # The edit argument is taken from the real returned observation beyond line 200.
        await call('str_replace_editor',{'command':'str_replace','path':str(path),'old_str':rows[-1][1],'new_str':'observed-from-read'})
        assert path.read_text().endswith('observed-from-read\n')
        checks.append('observation_dependent_edit_beyond_line_200')
        created=Path('/testbed/new.txt');payload='  alpha\n\nβeta\n'
        await call('str_replace_editor',{'command':'create','path':str(created),'file_text':payload})
        assert created.read_bytes()==payload.encode()
        checks.append('create_exact_payload_whitespace_unicode')
        rejected=await call('str_replace_editor',{'command':'create','path':str(created),'file_text':'MUST NOT OVERWRITE'})
        assert rejected['message']['content'].startswith('ERROR:') and created.read_bytes()==payload.encode()
        checks.append('create_never_overwrites_existing_file')
        await call('str_replace_editor',{'command':'str_replace','path':str(created),'old_str':'alpha','new_str':'ALPHA'})
        assert created.read_text()==payload.replace('alpha','ALPHA')
        await call('str_replace_editor',{'command':'undo_edit','path':str(created)})
        assert created.read_text()==payload
        checks.append('replace_and_undo_persist_editor_history')
        await call('str_replace_editor',{'command':'insert','path':str(created),'insert_line':1,'new_str':'INSERTED'})
        assert created.read_text()=='  alpha\nINSERTED\n\nβeta\n'
        checks.append('insert_after_requested_line')
        ambiguous=Path('/testbed/ambiguous.txt');ambiguous.write_text('same\nsame\n')
        rejected=await call('str_replace_editor',{'command':'str_replace','path':str(ambiguous),'old_str':'same','new_str':'different'})
        assert rejected['message']['content'].startswith('ERROR:') and ambiguous.read_text()=='same\nsame\n'
        checks.append('ambiguous_replace_errors_without_mutation')
        setup=await call('execute_bash',{'command':'mkdir -p /testbed/nested && cd /testbed/nested && export E97_NATIVE_FLAG=persistent-fixture','timeout':10})
        assert setup['exit_code']==0
        state=await call('execute_bash',{'command':'printf "state:%s:%s\\n" "$PWD" "$E97_NATIVE_FLAG"','timeout':10})
        assert state['exit_code']==0 and 'state:/testbed/nested:persistent-fixture' in state['message']['content']
        checks.append('persistent_cwd_and_environment')
        waiting=await call('execute_bash',{'command':'read -r answer; printf "got:%s\\n" "$answer"'})
        assert waiting['exit_code']==-1 and 'no new output' in waiting['message']['content']
        checks.append('default_soft_timeout_preserves_live_process')
        busy=await call('execute_bash',{'command':'printf must-not-start-new','is_input':'false','timeout':2})
        assert busy['exit_code']==-1 and 'NOT executed' in busy['message']['content']
        checks.append('busy_shell_rejects_fresh_command')
        received=await call('execute_bash',{'command':'native-input','is_input':'true','timeout':3})
        assert received['upstream_is_input'] is True and received['exit_code']==0 and 'got:native-input' in received['message']['content']
        checks.append('stdin_continuation_reaches_existing_process')
        long_budget=await call('execute_bash',{'command':'printf long-budget-accepted','timeout':3600})
        assert long_budget['exit_code']==0
        checks.append('explicit_timeout_above_120_accepted')
        timed=await call('execute_bash',{'command':'sleep 5','timeout':0.5})
        assert timed['exit_code']==-1 and 'timed out' in timed['message']['content']
        interrupted=await call('execute_bash',{'command':'C-c','is_input':'true','timeout':3})
        assert interrupted['exit_code']!=-1
        alive=await call('execute_bash',{'command':'printf "after:%s\\n" "$E97_NATIVE_FLAG"','timeout':3})
        assert alive['exit_code']==0 and 'after:persistent-fixture' in alive['message']['content']
        checks.append('timeout_interrupt_and_same_shell_recovery')
        finish={'role':'assistant','reasoning_content':None,'content':'Fixture complete.','think':None,
                'tool_calls':[{'type':'function','function':{'name':'finish','arguments':'{"message":"  Done.\\n"}'}}]}
        turn=episode.accept_generated_turn(native_turn(finish))
        assert episode.finished and turn.backend_call() is None and turn.public_events()[-1]['text']=='  Done.\n'
        checks.append('public_finish_and_episode_termination')
        return {'schema':'emender-openhands-native-execution-qualification-v1','status':'passed',
                'checks':checks,'calls':calls,'sandbox':backend.sandbox,'source_hashes':source_hashes,
                'dataset_manifest_sha256':expected['dataset_manifest_sha256'],'tool_schemas_exact':True,
                'model_loaded':False,'training_eligible':False,'historical_runtime_identity_verified':False,
                'scope':'scripted isolated upstream shell/editor execution and native episode integration; not model capability or historical environment replay'}
    except Exception as error:
        print('E97_NATIVE_EXECUTION_FAILURE '+json.dumps({'status':'failed','error':repr(error),'completed_checks':checks,'calls':calls},ensure_ascii=False),flush=True)
        raise
    finally:backend.close()


if __name__=='__main__':
    result=asyncio.run(qualify(Path(os.environ['E97_NATIVE_BUNDLE'])))
    print('E97_NATIVE_EXECUTION_RESULT '+json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
