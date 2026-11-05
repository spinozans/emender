"""Launch the registered leased train/eval wrapper outside the attended session."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/erikg/emender')
REL = Path('experiments/expressivity_tasks/x43_fork_resolved')
OUT = ROOT / REL / 'receipts/c14_b_matched'


def main():
    assert sys.executable == str(ROOT / '.venv/bin/python')
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('train', 'eval'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.phase == 'eval':
        state = json.loads((OUT / 'training.json').read_text())
        assert state['status'] == 'completed_matched_exposure'
        assert state['steps'] == 35419 and state['actual_bytes_targets'] == 71677042
    # Refuse a duplicate live wrapper, including one launched by another session.
    wrapper = 'run_c14_b_matched' + ('_eval' if args.phase == 'eval' else '') + '_leased.sh'
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                argv = (proc / 'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            assert not any(Path(a.decode(errors='replace')).name == wrapper for a in argv), 'live wrapper exists'
    command = ['bash', str(REL / wrapper)] + (['--resume'] if args.resume else [])
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
