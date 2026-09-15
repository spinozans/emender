"""Real Pi RPC lifecycle with scripted native turns; zero model/executor evidence."""
from pathlib import Path
import shutil
import tiktoken
from scripts.e97_native_execution_cases import context
from scripts.e97_open_swe_native_codec import compact,native_turn
from scripts.e97_pi_native_session import NativeTaskSession
from scripts.e97_pi_native_session_transport import serve_pi_session
from scripts.qualify_e97_pi_native_session import scripted


def turn(name,args):
    return native_turn(dict(role='assistant',content=None,reasoning_content='PRIVATE_SESSION_SENTINEL',think=None,
        tool_calls=[dict(type='function',function=dict(name=name,arguments=compact(args)))]))


def test_real_sandbox_script_is_two_explicit_bounded_tasks():
    tasks,turns=scripted()
    assert [t['task_id'] for t in tasks]==['workspace-create','workspace-followup']
    assert len(turns)==8 and turns[2].splitlines()[3]=='Action: finish' and turns[-1].splitlines()[3]=='Action: finish'
    assert all('PRIVATE_SESSION_ANALYSIS_' in t for t in turns)


def test_real_pi_two_explicit_tasks_retain_executor_not_model_context(tmp_path):
    pi=shutil.which('pi')
    if not pi: return
    panel=dict(system='native session test',tools=[dict(type='function',function=dict(name=n)) for n in
        ('execute_bash','str_replace_editor','think','finish')],max_turns=4,generation_budget=4096,
        episode_generation_budget=8192,episode_seconds=30)
    enc=tiktoken.get_encoding('p50k_base'); queue=iter([
        turn('execute_bash',dict(command='set')),turn('finish',dict(message='one done')),
        turn('execute_bash',dict(command='get')),turn('finish',dict(message='two done'))])
    state={}; prompts=[]
    def generate(prompt,budget,deadline):
        prompts.append(prompt); text=next(queue); return text,enc.encode_ordinary(text),'valid'
    def execute(call):
        if call['arguments']['command']=='set': state['value']='persistent'
        return dict(result=dict(message=context('tool',state['value']),exit_code=0))
    s=NativeTaskSession(panel,enc,generate,execute,max_tasks=2,session_tokens=8192,session_seconds=120)
    tasks=[dict(task_id='one',prompt='First task'),dict(task_id='two',prompt='Second task')]
    out=tmp_path/'session'
    result=serve_pi_session(s,tasks,out,pi_bin=pi,
        extension=Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts',seconds=60)
    assert result['pi_exit']==0 and result['tasks_begun']==result['successful_task_closures']==2
    assert result['failed_task_settlements']==0
    assert [r['op'] for r in result['requests']]==[
        'begin_task','next','execute','next','execute','settle_task',
        'begin_task','next','execute','next','execute','settle_task','close_session']
    assert len(prompts)==4 and 'First task' in prompts[0] and 'First task' in prompts[1]
    assert 'First task' not in prompts[2] and 'First task' not in prompts[3]
    assert 'Second task' in prompts[2] and state==dict(value='persistent')
    public=(out/'pi-events-private.jsonl').read_text(); assert 'PRIVATE_SESSION_SENTINEL' not in public
    assert len(s.completed_history)==10 and len(s.tasks)==2


def test_real_pi_failed_first_task_is_retained_before_successful_second(tmp_path):
    pi=shutil.which('pi')
    if not pi: return
    panel=dict(system='native failure session',tools=[dict(type='function',function=dict(name=n)) for n in
        ('execute_bash','str_replace_editor','think','finish')],max_turns=2,generation_budget=4096,
        episode_generation_budget=8192,episode_seconds=30)
    enc=tiktoken.get_encoding('p50k_base'); calls=0; prompts=[]
    finish=turn('finish',dict(message='second done'))
    def generate(prompt,budget,deadline):
        nonlocal calls
        prompts.append(prompt); calls+=1
        if calls==1: return None,[1],'invalid_opening'
        return finish,enc.encode_ordinary(finish),'valid'
    def execute(call): raise AssertionError('unexpected external call')
    s=NativeTaskSession(panel,enc,generate,execute,max_tasks=2,session_tokens=8192,session_seconds=120)
    tasks=[dict(task_id='fails',prompt='First fails'),dict(task_id='works',prompt='Second works')]
    out=tmp_path/'failed-session'
    result=serve_pi_session(s,tasks,out,pi_bin=pi,
        extension=Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts',seconds=60,
        require_successful_tasks=False)
    assert result['failed_task_settlements']==1 and result['successful_task_closures']==1
    assert [r['op'] for r in result['requests']]==['begin_task','next','settle_task',
        'begin_task','next','execute','settle_task','close_session']
    assert s.tasks[0].failed and s.tasks[0].closed and not s.tasks[0].close_verified
    assert s.tasks[1].close_verified and s.closed
    assert 'First fails' not in prompts[1] and 'Second works' in prompts[1]
    assert len(s.completed_history)==5 and s.completed_history[1]['stopReason']=='error'
