from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys
import pytest
import tiktoken
from scripts.e97_grounded_curriculum import fixtures,teacher_record,FAMILIES
from scripts.e97_native_execution_cases import context,grade

TOOLS=[dict(type='function',function=dict(name=n)) for n in ('execute_bash','str_replace_editor','think','finish')]


def test_paired_curriculum_counts_and_observation_dependence():
    rows=fixtures('test',dict(lookup=128,sum=128,edit=384,recovery=384),'train')
    assert len(rows)==2048 and Counter(c['family'] for c in rows)==dict(lookup=256,sum=256,edit=768,recovery=768)
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['prompt']==b['prompt'] and set(a['files'])==set(b['files']) and a['files']!=b['files']
        assert a['expected_output']!=b['expected_output'] if a['family']=='edit' else a['answer']!=b['answer']
        assert all(not p.startswith('/') and '..' not in Path(p).parts for p in a['files'])
        if a['family']!='edit':assert a['answer'] not in a['prompt']
    assert Counter(c['family'] for c in rows if c['authored_failure_prefix'])==dict(edit=384,recovery=384)


def test_eval_cohorts_are_not_training_records():
    groups={cohort:fixtures('split-'+cohort,dict.fromkeys(FAMILIES,2),cohort) for cohort in ('train','fresh','transfer')}
    assert all(not c['authored_failure_prefix'] for c in groups['fresh']+groups['transfer'])
    assert not ({p for c in groups['train'] for p in c['files']} & {p for c in groups['fresh']+groups['transfer'] for p in c['files']})
    assert any('active_id' in c['prompt'] for c in groups['transfer'])
    assert any('number in change' in c['prompt'] for c in groups['transfer'])


def test_representation_bridge_pairs_and_observed_deltas():
    from scripts.e97_representation_bridge import bridge_cases
    rows=bridge_cases('bridge-unit')
    assert len(rows)==768
    assert Counter(c['family'] for c in rows)==dict.fromkeys(FAMILIES,192)
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['prompt']==b['prompt'] and set(a['files'])==set(b['files']) and a['files']!=b['files']
        if a['family']=='edit':
            assert a['expected_output']!=b['expected_output']
            if a['bridge_layout'] in (1,2):
                for c in (a,b):
                    before=json.loads(c['files'][c['path'].removeprefix('/testbed/')]);after=c['expected_output']
                    for key in c['recipe']['field_path']:before,after=before[key],after[key]
                    assert 2<=abs(after-before)<=100 and after-before!=1
        else:assert a['answer']!=b['answer'] and a['answer'] not in a['prompt']
    assert sum(c['source_style'] for c in rows)==384
    for naming in range(4):
        for layout in (1,2):
            amounts=set()
            for c in rows:
                if c['family']!='edit' or c['naming_index']!=naming or c['bridge_layout']!=layout:continue
                d=json.loads(c['files'][c['path'].removeprefix('/testbed/')])
                for key in c['recipe']['delta_path']:d=d[key]
                amounts.add(d)
            assert len(amounts)>=5


def test_bridge_pointer_selection_is_unique_and_row_order_independent():
    from scripts.e97_representation_bridge import selected_value
    spec=dict(kind='rows',selector=['active'],table=['records'],key_field='id',value_field='path')
    d={'active':'chosen','records':[{'id':'other','path':'/testbed/other'},{'id':'chosen','path':'/testbed/yes'}]}
    assert selected_value(d,spec)=='/testbed/yes'
    d['records'].reverse();assert selected_value(d,spec)=='/testbed/yes'
    d['records'].append(d['records'][0])
    with pytest.raises(ValueError,match='unique'):selected_value(d,spec)


class LocalExecutor:
    """Test-only real file IO/subprocess arithmetic in pytest's private tmpdir."""
    def __init__(self,root,files):
        self.root=root
        for name,text in files.items():
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    def request(self,op,call):
        assert op=='execute';args=call['arguments'];code=0
        if call['name']=='str_replace_editor':
            p=self.root/args['path'].removeprefix('/testbed/')
            if args['command']=='create':
                if p.exists():text='ERROR: File already exists.'
                else:p.write_text(args['file_text']);text='Created.'
            elif not p.exists():text='ERROR: File not found.'
            else:text=f"Here's the result of running `cat -n` on {args['path']}:\n"+''.join(f'{i:6d}\t{line}\n' for i,line in enumerate(p.read_text().splitlines(),1))
        else:
            parts=shlex.split(args['command']);assert parts[:2]==['python','-c']
            preamble=('import builtins,os\nreal_open=builtins.open; real_osopen=os.open\n'
                f'root={str(self.root)!r}\n'
                'def mapped(p): return root+p[len("/testbed"):] if isinstance(p,str) and p.startswith("/testbed/") else p\n'
                'builtins.open=lambda p,*a,**k:real_open(mapped(p),*a,**k)\n'
                'os.open=lambda p,*a,**k:real_osopen(mapped(p),*a,**k)\n')
            result=subprocess.run([sys.executable,'-c',preamble+parts[2]],capture_output=True,text=True,timeout=10)
            text=result.stdout+result.stderr;code=result.returncode
        return dict(result=dict(message=context('tool',text),exit_code=code))


@pytest.mark.parametrize('cohort',['train','composition'])
@pytest.mark.parametrize('index',[0,1,59])
@pytest.mark.parametrize('family',FAMILIES)
def test_representation_bridge_teacher_actual_io(tmp_path,cohort,index,family):
    from scripts.e97_representation_bridge import bridge_cases
    from scripts.audit_e97_grounded_expansion import check_teacher,reconstruct
    cases=bridge_cases('bridge-io',60,cohort)
    c=next(c for c in cases if c['family']==family and c['id'].endswith(f'{index:04d}-world-1'))
    panel=dict(tools=TOOLS,system='Use observations.',models=[{},dict(system_message=context('system','Source system.'))])
    enc=tiktoken.get_encoding('p50k_base');executor=LocalExecutor(tmp_path,c['files'])
    record,receipt=teacher_record(c,panel,panel['models'][1]['system_message'],executor,enc)
    check_teacher(c,record,receipt,panel);reconstruct(record,TOOLS,enc)
    if family=='edit':
        commands=[x['request']['arguments'].get('command','') for x in receipt['calls']]
        assert any('assert actual==expected' in command for command in commands)
        if 'delta_path' in c['recipe'] or 'delta_spec' in c['recipe']:
            assert any('d[' in command and ('request' in command or 'change' in command or 'step' in command) for command in commands)


@pytest.mark.parametrize('cohort',['train','transfer'])
@pytest.mark.parametrize('family',FAMILIES)
def test_native_teacher_real_io_masks_and_oracle(tmp_path,cohort,family):
    rows=fixtures('io-'+family,dict.fromkeys(FAMILIES,4),cohort)
    c=next(c for c in rows if c['family']==family and c['id'].endswith('0003-world-1'))
    executor=LocalExecutor(tmp_path,c['files']);enc=tiktoken.get_encoding('p50k_base')
    record,evidence=teacher_record(c,dict(tools=TOOLS,system='Use observations.'),context('system','Source system.'),executor,enc)
    snapshot={p:(tmp_path/p).read_text() for p in c['files']}
    if c['expected_output'] is not None:snapshot[c['output_path']]=(tmp_path/c['output_path']).read_text()
    assert grade(c,c['answer'],evidence['calls'],snapshot)['success']
    assert evidence['source_bytes_unchanged'] and record['training_eligible'] is False
    selected=enc.decode([t for t,m in zip(record['token_ids'],record['assistant_mask']) if m])
    assert 'Action: finish\n' in selected
    if c['authored_failure_prefix']:
        assert record['prefix_assistants_unsupervised'] in (1,2)
        assert '"command":"create"' not in selected
        if family=='recovery':assert c['missing_path'] not in selected
    assert all(snapshot[p]==v for p,v in c['files'].items())
    from scripts.audit_e97_grounded_expansion import check_teacher,reconstruct
    audit_panel=dict(tools=TOOLS,system='Use observations.',models=[{},dict(system_message=context('system','Source system.'))])
    check_teacher(c,record,evidence,audit_panel)
    _,mask=reconstruct(record,TOOLS,enc)
    assert list(mask)==record['assistant_mask']
    record['assistant_mask'][0]=1
    with pytest.raises(ValueError,match='independent token/mask'):reconstruct(record,TOOLS,enc)
