#!/usr/bin/env python3
"""External W3G window/occupancy guard; never changes the frozen read instrument."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.w3_matcheddose_preflight import REPO, ROOT, PARENT_SHA, checked, file_sha

RUN = ROOT / 'w3-matcheddose'


def process_fields(pid):
    # /proc comm may contain spaces/parentheses. Fields after the last ')' start
    # with state; PPID and starttime are offsets 1 and 19 respectively.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return int(fields[1]), fields[19]


def belongs_to(pid, root, starttime):
    original_pid = pid
    try:
        seen = set()
        while pid > 1 and pid not in seen:
            seen.add(pid)
            parent, start = process_fields(pid)
            if pid == root:
                return start == starttime
            pid = parent
    except (FileNotFoundError, ProcessLookupError):
        # A process which exited between nvidia-smi and /proc is not an occupant.
        return not Path(f'/proc/{original_pid}').exists()
    return False


def snapshot():
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=30)
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=30)
    rows = [[v.strip() for v in line.split(',')] for line in gpu.splitlines() if line.strip()]
    if len(rows) != 8 or {r[0] for r in rows} != set(map(str, range(8))):
        raise ValueError('missing/different eight-card GPU inventory')
    processes = [dict(uuid=v[0], pid=int(v[1]), command=v[2], memory=v[3])
                 for line in apps.splitlines() if line.strip()
                 for v in [[part.strip() for part in line.split(',')]]]
    return dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                gpus=rows, processes=processes)


def window_free(state, grantable):
    return (not state['processes'] and set(grantable.split(',')) == set(map(str, range(8)))
            and all(int(row[2]) < 256 and int(row[3]) < 10 for row in state['gpus']))


def verify_entries(run):
    """Re-hash existing immutable training pins before the runner pins read points."""
    records = []
    for arm in ('A', 'B'):
        directory = run / 'arms' / arm
        complete = json.loads((directory / 'complete-receipt.json').read_text())
        terminal = json.loads((run / 'arms' / f'{arm}-terminal.json').read_text())
        if (complete['status'], complete['updates'], complete['inputs'], complete['targets'],
                terminal['exit_code']) != ('complete', 512, 3571182, 154386, 0):
            raise ValueError('incomplete/different arm; no read')
        checked(directory / 'start-receipt.json', complete['start_receipt_sha256'])
        checked(directory / 'training.jsonl', complete['training_log_sha256'])
        for point, expected in (('entry', PARENT_SHA), ('final', complete['checkpoint_sha256'])):
            pin = directory / f'{point}-pinned/pin-receipt.json'
            receipt = json.loads(pin.read_text())
            if receipt['sha256'] != expected or not receipt['pinned_before_gate_read']:
                raise ValueError('pin identity mismatch')
            records.append(dict(arm=arm, point=point, pin_receipt=checked(pin, file_sha(pin)),
                                checkpoint=checked(pin.parent / receipt['checkpoint'], expected)))
    return records


def stop_group(child):
    # Frozen runner installs its own lease EXIT trap. TERM lets that trap run;
    # KILL bounds teardown if a rank or a CLI child ignores TERM.
    try:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=35)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def write_once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=300)
    parser.add_argument('--guard-interval', type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0 or args.guard_interval <= 0:
        parser.error('positive polling intervals required')
    evidence = RUN / 'gate-supervision'
    evidence.mkdir(mode=0o700, exist_ok=False)
    runner = REPO / 'scripts/w3_run_gate_reads.sh'
    write_once(evidence / 'launcher.json', dict(pid=os.getpid(), starttime=process_fields(os.getpid())[1],
        runner=checked(runner, file_sha(runner)), interval=args.interval,
        guard_interval=args.guard_interval, automatic_retry=False,
        source=checked(Path(__file__), file_sha(__file__))))
    child = None
    status, code = 'failed-before-read', 1
    try:
        with (evidence / 'occupancy.jsonl').open('x', buffering=1) as log:
            while True:
                state = snapshot()
                grantable = subprocess.check_output(['bash', 'scripts/gpu_lease.sh', 'list-free'],
                                                    cwd=REPO, text=True, timeout=30).strip()
                state.update(phase='waiting', grantable=grantable)
                log.write(json.dumps(state, sort_keys=True) + '\n')
                print(state['time'], 'WAIT eight free GPUs', grantable, flush=True)
                if window_free(state, grantable):
                    break
                time.sleep(args.interval)
            entries = verify_entries(RUN)
            write_once(evidence / 'read-time-training-pins.json', dict(
                verified_before_any_read=True, records=entries))
            # Hashing pins can take minutes: verify the window again afterwards.
            while True:
                state = snapshot()
                grantable = subprocess.check_output(['bash', 'scripts/gpu_lease.sh', 'list-free'],
                                                    cwd=REPO, text=True, timeout=30).strip()
                state.update(phase='verified-pins-waiting', grantable=grantable)
                log.write(json.dumps(state, sort_keys=True) + '\n')
                if window_free(state, grantable):
                    break
                time.sleep(args.interval)
            write_once(evidence / 'window-before-read.json', state)
            with (evidence / 'runner-console.txt').open('x') as console:
                child = subprocess.Popen(['bash', str(runner)], cwd=REPO, start_new_session=True,
                                         stdout=console, stderr=subprocess.STDOUT)
                start = process_fields(child.pid)[1]
                write_once(evidence / 'reader.json', dict(pid=child.pid, starttime=start,
                           runner_sha256=file_sha(runner)))
                while child.poll() is None:
                    state = snapshot()
                    foreign = [p for p in state['processes'] if not belongs_to(p['pid'], child.pid, start)]
                    state.update(phase='reading', foreign=foreign)
                    log.write(json.dumps(state, sort_keys=True) + '\n')
                    if foreign:
                        stop_group(child)
                        status = 'aborted-occupancy-wait-no-retry'
                        # Preserve partial evidence; never replay partially read checkpoints.
                        # Return only after a new clean window, for attended continuation.
                        while True:
                            state = snapshot()
                            state['phase'] = 'abort-wait'
                            log.write(json.dumps(state, sort_keys=True) + '\n')
                            if not state['processes']:
                                break
                            time.sleep(args.interval)
                        raise RuntimeError('foreign occupant interrupted reads; attended decision required')
                    time.sleep(args.guard_interval)
                code = child.wait()
                status = 'measurements-complete' if code == 0 else 'frozen-runner-failed-no-retry'
    except BaseException as exc:
        if child is not None and child.poll() is None:
            stop_group(child)
        write_once(evidence / 'error.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        write_once(evidence / 'terminal.json', dict(status=status, exit_code=code,
                   automatic_retry=False, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
