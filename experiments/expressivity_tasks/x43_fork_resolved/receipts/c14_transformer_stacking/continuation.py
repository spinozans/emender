import os, subprocess, time, json
from pathlib import Path
root=Path('/home/erikg/emender')
out=root/'experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_transformer_stacking'
while not (out/'terminal.json').exists(): time.sleep(10)
assert json.loads((out/'terminal.json').read_text())['exit_code']==0
prefix='experiments.expressivity_tasks.x43_fork_resolved.'
for module,log in [('c14_transformer_stacking_score','scoring_console.txt'),('validate_c14_transformer_stacking','validation_console.txt'),('c14_first_pass_report','report_console.txt')]:
    with (out/log).open('w') as f:
        result=subprocess.run(['/home/erikg/emender/.venv/bin/python','-m',prefix+module],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        (out/'continuation_terminal.json').write_text(json.dumps(dict(module=module,exit_code=result.returncode))+'\n')
        raise SystemExit(result.returncode)
(out/'continuation_terminal.json').write_text('{"exit_code":0}\n')
