"""Launch the registered leased train/eval wrapper outside the attended session."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/erikg/emender')
REL = Path('experiments/expressivity_tasks/x43_fork_resolved')
OUT = ROOT / REL / 'receipts/c14_transformer'


def main():
    assert sys.executable == str(ROOT / '.venv/bin/python')
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('train', 'eval'))
    args = parser.parse_args()
    global OUT
    if args.phase == 'train':
        state=json.loads((OUT/'cache_reconciliation.json').read_text())
        assert state['status']=='passed' and state['training_authorized']
    else:
        state=json.loads((OUT/'training.json').read_text())
        assert state['status']=='trained_registered_budget'
        OUT=OUT.with_name('c14_transformer_eval')
    OUT.mkdir(exist_ok=True)
    wrapper='run_c14_transformer_leased.sh'
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                argv = (proc / 'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            assert not any(Path(a.decode(errors='replace')).name == wrapper for a in argv), 'live wrapper exists'
    command = ['bash', str(REL / wrapper), args.phase]
    with (OUT / (args.phase + '_detached_launcher_console.txt')).open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    record = dict(pid=process.pid, command=command,
                  launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  method='Popen start_new_session=True; stdin DEVNULL; file stdout/stderr')
    (OUT / (args.phase + '_detached_launch.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
