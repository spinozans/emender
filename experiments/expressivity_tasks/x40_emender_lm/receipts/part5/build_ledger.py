"""Collect executed commands and validate final receipt qualification."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

R=Path(__file__).resolve().parent
ROOT=R.parent.parent
RUN='experiments/expressivity_tasks/x40_emender_lm/run_leased.sh'
MOD='experiments.expressivity_tasks.x40_emender_lm.'
REL='experiments/expressivity_tasks/x40_emender_lm/receipts/part5/'
FAST='X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1 X40_FUSED_INIT=1 '
commands=[]
def add(command,result,summary):commands.append(dict(command=command,result=result,summary=summary))
def read(p):return json.loads(p.read_text())

add('ps -p 634691 -o pid,args; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv','passed','Depth process active on physical 0; wrapper excludes 0/2/3/4/7. See depth_process.txt and gpu_inventory.csv.')
add(RUN+' -m '+MOD+'scale_probe --output '+REL+'scale_initial --steps 5','failed','T512 functional probe passed after pressure-only exclusion; T2048 oracle rebase grid.y exceeded CUDA launch limit. No timing accepted.')
add(RUN+' -m '+MOD+'scale_probe --gate-only --output '+REL+'scale_gate_rebased','passed','Both retention-policy contexts pass; integer axis-transpose control passes small exact mapping and 65537 log blocks.')
for name in ('production_parity','production_T2048','reads_parity','backward_parity','fused_parity','fused_parity_after_guard'):
    proof=read(R/(name+'.json'));flags=proof.get('flags',{})
    env=' '.join(k+'='+v for k,v in flags.items() if v!='0')
    args=''
    if name=='production_T2048':args=' --batch 1 --context 2048 --receipt production_T2048.json'
    elif name=='fused_parity_after_guard':args=' --receipt fused_parity_after_guard.json'
    add((env+' ' if env else '')+RUN+' -m '+MOD+'part5_gate'+args,proof['status'],'Exact native production gate including pressure; '+name+'.json')
for directory,phase in (('surface','qualified final surface'),('surface_pre_long_gate','earlier OOM diagnostic before independent long gate')):
    for p in sorted((R/directory).glob('*.json')):
        row=read(p)
        cmd=RUN+' -m '+MOD+f"scale_surface --scale {row['scale']} --arm {row['arm']} --batch {row['batch']} --context {row['context']} --steps 20"
        msg=phase+': '+(f"{row['tokens_per_second']:.3f} tok/s; 20 full training updates" if row['status']=='passed' else row.get('error','failed'))
        add(cmd,row['status'],msg)
for label in sorted(p.name for p in R.glob('pilot_*') if p.is_dir()):
    p=R/label/'wall.json';failure=R/label/'failure.json'
    if p.exists():
        row=read(p);batch=row['batch'];status=row['status']
        env=' '.join(k+'='+v for k,v in row.get('flags',{}).items() if v!='0')
        if not env and label.startswith('pilot_reads'):env='X40_BATCHED_READS=1'
        args=f' --label {label.removesuffix("_pre_guard")} --batch {batch} --steps {row["updates"]}'
        if row.get('bounded_projection'):args+=' --bounded-loss'
        if label=='pilot_B128_bounded':args+=' --gate fused_parity_after_guard.json'
        msg=f"{row['tokens_per_second']:.3f} tok/s; {row['updates']} measured updates"
        if label.endswith('_pre_guard'):msg+='; historical artifact archived after adding the guard, not eligible for final best'
        if label=='pilot_reads_cpu':msg+='; one-sample CPU diagnostic, not sustained acceptance'
    elif failure.exists():
        row=read(failure);status='failed';env=FAST.strip()
        batch=256 if '256' in label else 128
        args=f' --label {label} --batch {batch}'
        if label=='pilot_B256_checkpoint':args+=' --bounded-loss --checkpoint'
        msg=row['error']
        if label=='pilot_B256_checkpoint':msg+='; rejected checkpoint prototype reverted and patch retained'
        if label=='pilot_B128_release':msg+='; ineffective early-release change reverted and patch retained'
    else:continue
    add((env+' ' if env else '')+RUN+' -m '+MOD+'part5_pilot'+args,status,msg)
for p in sorted(R.glob('warp*_parity.json')):
    v=read(p);args=f" --warps {v['warps']}"
    if v.get('min_slots'):args+=f" --min-slots {v['min_slots']}"
    assert v['original_launch_restored'] and v['status']=='failed'
    add(FAST+RUN+' -m '+MOD+'part5_warp_probe'+args,'failed',v['error']+'; original launch restored, no timing')
for name,flags in (('curve',''),('curve_bounded',' --bounded-loss --steps 64'),('curve_bounded_pre_guard',' --bounded-loss --steps 64')):
    v=read(R/name/'summary.json')
    add(FAST+RUN+' -m '+MOD+'train_slice --output '+REL+name+flags,v['status'],f"{v['tokens']} matched v1 tokens; max loss difference {v['max_loss_difference']:.9g}; tolerance 1e-5"+('; archived pre-guard run' if name.endswith('_pre_guard') else ''))
for name,tests in (('oracle','experiments/expressivity_tasks/x31_tape_cell_fused/tests'),('replay','experiments/expressivity_tasks/x36_tape_perf/tests')):
    xml=ET.parse(R/(name+'.xml')).getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
    count=sum(int(x.get('tests',0)) for x in suites);fail=sum(int(x.get('failures',0))+int(x.get('errors',0)) for x in suites)
    assert fail==0
    add(FAST+RUN+' -m pytest -q -p '+MOD+'oracle_adapter '+tests+' --junitxml='+REL+name+'.xml','passed',f'{count} native fixtures passed')
assert '9 passed' in (R/'local_tests.log').read_text()
add('/home/erikg/emender/.venv/bin/python -m pytest -q experiments/expressivity_tasks/x40_emender_lm/tests','passed','9 local tests passed, including full-stack capacity matching and fail-closed nonrecording reads')
add('git diff --cached --check','passed','Code/structured artifacts pass; narrow .gitattributes preserves raw log, patch-context and profiler formatting')
(R/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')

proof=read(R/'fused_parity_after_guard.json')
core=('model.py','forward.py','kernels.py','backward.py','parity.py','fused_init.py','exact_decay.py')
current={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in core}
assert all(current[n]==proof['sources'][n] for n in core)
curve=read(R/'curve_bounded/summary.json')
assert all(current[n]==curve['sources'][n] for n in core)
bench=read(R/'pilot_B128_bounded/wall.json')
assert all(current[n]==bench['sources'][n] for n in core)
points=[read(p) for p in (R/'surface').glob('*.json')]
assert len(points)==24 and sum(x['status']=='passed' for x in points)==6
assert len((R/'long_surface_exit_codes.txt').read_text().splitlines())==12
assert read(R/'production_T2048.json')['pressure_gate'] is True
assert len(list((R/'surface_pre_long_gate').glob('*.json')))==12
assert not any(p.suffix in ('.pt','.pth','.pyc') for p in R.rglob('*') if p.is_file())
(R/'validation.json').write_text(json.dumps(dict(status='passed',final_core_sha256=current,
    post_guard_production_benchmark_and_curve_sources_match=True,surface_arms=24,passed_arms=6,failed_arms=18,
    long_arms_repeated_after_gate=12,native_tests=179,local_tests=9,forbidden_artifacts=[]),indent=2)+'\n')
print('Receipt consistency passed; performance acceptance remains failed.')
