"""Measure a native XSA session through its actual synchronous pipe protocol.
CPU systems benchmark, not LM training. Supply a single document >= 1,000,512 bytes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def timed(fn):
    t = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns() - t) / 1e6


def stats(xs):
    s = sorted(xs)
    return {k: s[math.ceil(len(s)*p)-1] for k, p in [('p50', .5), ('p99', .99), ('max', 1)]}


def bench(exe, data, n, depth, steps):
    with subprocess.Popen([exe, 'session', str(depth)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE) as proc:
        def rpc(cmd):
            proc.stdin.write(cmd.encode('ascii') + b'\n')
            proc.stdin.flush()
            reply = proc.stdout.readline()
            if not reply:
                raise RuntimeError(f'XSA session exited during {cmd[:12]}')
            return json.loads(reply)

        def ingest():
            for i in range(0, n, 65536):
                end = min(n, i+65536)
                if rpc('append '+data[i:end].hex()) != {'visible': end}:
                    raise RuntimeError('prefill offset mismatch')

        _, prefill = timed(ingest)
        def seal(pos):
            if rpc('seal') != {'sealed': pos}:
                raise RuntimeError('seal offset mismatch')

        _, first_seal = timed(lambda: seal(n))
        latencies = []
        for i in range(n, n+steps):
            def step():
                if rpc('append '+data[i:i+1].hex()) != {'visible': i+1}:
                    raise RuntimeError('decode offset mismatch')
                return rpc('branch '+data[i+1-depth:i+1].hex())
            answer, ms = timed(step)
            latencies.append(ms)
            if i in (n, n+steps-1):
                p = data[i+1-depth:i+1]
                counts = [0]*256
                j = data.find(p, 0, i+1)
                while j >= 0 and j+depth <= i:
                    counts[data[j+depth]] += 1
                    j = data.find(p, j+1, i+1)
                if answer != {'counts': counts, 'end': 0}:
                    raise RuntimeError('causal counts mismatch')
        _, second_seal = timed(lambda: seal(n+steps))
        reread, cold = timed(lambda: rpc('branch '+data[n+steps-depth:n+steps].hex()))
        if reread != answer:
            raise RuntimeError('seal changed answer')
        status = Path(f'/proc/{proc.pid}/status').read_text()
        peak_kib = int(next(x for x in status.splitlines() if x.startswith('VmHWM:')).split()[1])
        proc.stdin.close()
        if proc.wait(timeout=15):
            raise RuntimeError('nonzero session exit')
    return dict(prefix=n, steps=steps, depth=depth, ingest_ms=prefill,
                synchronous_seals_ms=[first_seal, second_seal], first_append_query_ms=latencies[0],
                warm_append_query_ms=stats(latencies[1:]), first_query_after_reseal_ms=cold,
                server_peak_rss_kib=peak_kib)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('binary', type=Path)
    ap.add_argument('document', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    data = args.document.read_bytes()[:1_000_512]
    if len(data) < 1_000_512:
        ap.error('document too short')
    cases = {'document': data, 'repetitive': (b'abababba'*125064)}
    result = dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  clock='client elapsed milliseconds, includes two synchronous IPCs and JSON decode per byte',
                  input_name=args.document.name, rows=[])
    for name, text in cases.items():
        for size in [100_000, 500_000, 1_000_000]:
            row = dict(case=name, input_sha256=hashlib.sha256(text).hexdigest(),
                       **bench(args.binary.resolve(), text, size, 8, 512))
            result['rows'].append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
