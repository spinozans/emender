"""Trusted reader in a separate PID-namespace helper while the agent is paused.

No oracle is supplied. The source root is /proc/1/root, not the helper's root.
No model process runs in the helper's cgroup. Never import from agent storage.
"""
import json
import os
from pathlib import PurePosixPath
import stat
import sys


def read_regular_files(root, names):
    if not isinstance(names, list) or not 1 <= len(names) <= 8 or len(set(names)) != len(names):
        raise ValueError('snapshot names')
    for name in names:
        if not isinstance(name, str) or not name or name.startswith('/') or '..' in PurePosixPath(name).parts:
            raise ValueError('snapshot path')
    # This one trusted magic link intentionally enters the paused agent's root.
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    result = {}
    try:
        for name in names:
            opened = []
            try:
                parent = root_fd
                parts = ('testbed', *PurePosixPath(name).parts)
                for part in parts[:-1]:
                    parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                    opened.append(parent)
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                opened.append(fd)
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 65536:
                    raise ValueError('nonregular or oversized file')
                data = bytearray()
                while len(data) <= 65536:
                    chunk = os.read(fd, 65537-len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                if len(data) > 65536:
                    raise ValueError('file exceeded size bound')
                result[name] = data.decode('utf-8')
            except (OSError, ValueError):
                result[name] = None
            finally:
                for fd in reversed(opened):
                    os.close(fd)
    finally:
        os.close(root_fd)
    return result


if __name__ == '__main__':
    print(json.dumps(read_regular_files('/proc/1/root', json.loads(sys.argv[1]))), flush=True)
