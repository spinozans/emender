"""B2 host-side wire and vote-order contract; no index, corpus, or GPU."""
import json
import sys
from types import SimpleNamespace

import pytest

from .b1_mount import Span
from .b2_worker import PositionalWorker, WorkerRegistry


ANSWER = json.dumps({'rows': [{'raw': 1, 'lf': [0], 'hits': [[0, 0, 0]],
                               'status': 'complete'}]}).encode()


def pipe_worker(tmp_path, response):
    program = tmp_path/'worker'
    program.write_text(f'''#!{sys.executable}
import sys
print('{{"ready":1}}', flush=True)
r = sys.stdin.buffer
r.read(int.from_bytes(r.read(4), 'little'))
sys.stdout.buffer.write({response!r}); sys.stdout.buffer.flush()
''')
    program.chmod(0o700)
    descriptor = SimpleNamespace(name='fixture', version='v1', index='unused', text='unused',
                                 starts=[0], size=1)
    return PositionalWorker(descriptor, program)


@pytest.mark.parametrize('response', [
    ANSWER, ANSWER+b'\n', ANSWER+b'\n{"send_ns":1}', ANSWER+b'\n{}\n',
    ANSWER+b'\n{"rows":[],"send_ns":1}\n', ANSWER+b'\n{"send_ns":-1}\n',
    ANSWER+b'\n{"send_ns":true}\n', ANSWER+b'\n{"send_ns":0.5}\n',
])
def test_invalid_response_is_terminal(tmp_path, response):
    worker = pipe_worker(tmp_path, response)
    with pytest.raises(ValueError):
        worker.request([b'a'])
    assert worker.proc.poll() is not None and worker.proc.stdin.closed and worker.proc.stdout.closed
    with pytest.raises(ValueError):
        worker.request([b'a'])


def test_complete_response_with_timing_fence(tmp_path):
    worker = pipe_worker(tmp_path, ANSWER+b'\n{"send_ns":0}\n')
    try:
        rows, record = worker.request([b'a'])
        assert rows == [[Span('fixture', 'v1', 0, 0, 1)]] and record['send_ns'] == 0
    finally:
        worker.close()


def test_resolve_coalesces_only_within_each_request():
    a, b, c = (Span('fixture', 'v1', 0, i, i+1) for i in range(3))
    calls = []

    def request(patterns):
        calls.append(patterns)
        return ([{b'a': [a,b], b'b': [c]}[p] for p in patterns],
                {'rows': [{'status': 'complete'} for _ in patterns]})

    registry = WorkerRegistry({}, None, None, {'fixture'})
    registry.workers['fixture'] = SimpleNamespace(request=request)
    for _ in range(2):
        rows = registry.resolve('fixture', [[b'b',b'a'],[b'a',b'a']])
        assert [[span for span,_ in row] for row in rows] == [[c,a,b],[a,b,a,b]]
    assert calls == [[b'b',b'a'],[b'b',b'a']]
