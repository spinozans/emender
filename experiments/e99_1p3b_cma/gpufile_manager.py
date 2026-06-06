#!/usr/bin/env python3
"""Maintain the CMA gpu_file: a GPU is AVAILABLE iff it is idle OR occupied only
by the running search's own descendant processes. Zero cooldown, never grabs a
GPU held by a foreign job. Exits when the search PID dies.

Used after the sibling controls finished — replaces the conservative idle-timer
orchestrator so my own evals' GPUs re-qualify immediately between generations.
"""
import os, sys, time, subprocess, argparse


def gpu_index_by_uuid():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout
    m = {}
    for line in out.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(',')]
        m[uuid] = int(idx)
    return m


def gpu_occupants():
    """{gpu_index: set(pids)} from compute-apps."""
    idx_by_uuid = gpu_index_by_uuid()
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout
    occ = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid = [x.strip() for x in line.split(',')]
        gi = idx_by_uuid.get(uuid)
        if gi is not None:
            occ.setdefault(gi, set()).add(int(pid))
    return occ


def descendants(root):
    seen = {root}
    frontier = [root]
    while frontier:
        nxt = []
        for p in frontier:
            out = subprocess.run(['ps', '--ppid', str(p), '-o', 'pid=', '--no-headers'],
                                 capture_output=True, text=True).stdout
            for line in out.split():
                try:
                    c = int(line)
                except ValueError:
                    continue
                if c not in seen:
                    seen.add(c); nxt.append(c)
        frontier = nxt
    return seen


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--search_pid', type=int, required=True)
    ap.add_argument('--gpu_file', required=True)
    ap.add_argument('--all_gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--poll_sec', type=int, default=20)
    args = ap.parse_args()
    all_gpus = [int(g) for g in args.all_gpus.split(',')]
    logp = os.path.join(os.path.dirname(args.gpu_file), 'gpufile_manager.log')
    while True:
        if not alive(args.search_pid):
            with open(logp, 'a') as f:
                f.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} search {args.search_pid} dead; stop\n")
            break
        mine = descendants(args.search_pid)
        occ = gpu_occupants()
        avail = []
        for g in all_gpus:
            owners = occ.get(g, set())
            if not owners or owners.issubset(mine):
                avail.append(g)
        if avail:
            with open(args.gpu_file, 'w') as f:
                f.write(','.join(str(g) for g in avail) + '\n')
        with open(logp, 'a') as f:
            f.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} avail={avail} "
                    f"occ={ {g: sorted(occ.get(g,set())) for g in all_gpus if occ.get(g)} }\n")
        time.sleep(args.poll_sec)


if __name__ == '__main__':
    main()
