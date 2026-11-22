"""Black-box replay of the native `xsa session` protocol (CPU, stdlib only).

Usage: python -m experiments.expressivity_tasks.x43_fork_resolved.xsa_session_smoke /path/to/xsa
No corpus or training process required. Captures no timing claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def check(exe: Path) -> dict:
    fixture = (b'abcXabc', b'c\x00\x1e\xffabcYabc', b'aaaa')
    queries = (b'abc', b'a', b'aa', b'ca', b'cc', b'\x00\x1e', b'\xff', b'absent')
    checks = 0
    for stride in (1, 2, 5, 1000):
        docs = [bytearray()]
        commands, expected = [], []
        visible = 0

        def command(line, reply):
            commands.append(line)
            expected.append(reply)

        def probe():
            for p in queries:
                counts, end = [0] * 256, 0
                for doc_id, doc in enumerate(docs):
                    i = doc.find(p)
                    while i >= 0:
                        q = i + len(p)
                        if q < len(doc):
                            counts[doc[q]] += 1
                        elif doc_id < len(docs) - 1:
                            end += 1
                        i = doc.find(p, i + 1)
                command(f'branch {p.hex()}', dict(counts=counts, end=end))

        for doc in fixture:
            for byte in doc:
                docs[-1].append(byte)
                visible += 1
                command(f'append {byte:02x}', dict(visible=visible))
                probe()
                if visible % stride == 0:
                    command('seal', dict(sealed=visible))
                    probe()
            command('end', dict(ended=True))
            docs.append(bytearray())
            probe()
        command('seal', dict(sealed=visible))
        probe()
        run = subprocess.run([exe, 'session', '8'], text=True, input='\n'.join(commands)+'\n',
                             capture_output=True, timeout=60)
        if run.returncode:
            raise RuntimeError(run.stderr[-2000:])
        replies = run.stdout.splitlines()
        if len(replies) != len(expected):
            raise ValueError(f'missing/extra replies for stride {stride}')
        for i, (line, reply) in enumerate(zip(replies, expected)):
            if json.loads(line) != reply:
                raise ValueError(f'stride {stride}, request {i}: {commands[i]} mismatch')
        checks += sum(line.startswith('branch ') for line in commands)
    return dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                seal_strides=[1, 2, 5, 1000], branch_replies_verified=checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exe', type=Path)
    print(json.dumps(check(parser.parse_args().exe.resolve()), indent=2))
