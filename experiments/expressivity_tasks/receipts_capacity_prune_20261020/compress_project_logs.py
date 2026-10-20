"""Gzip only positively identified inactive Emender pi-bash logs in /tmp."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from prune import OUT, digest, entry_hash, write_json


def classify(path):
    stat = path.lstat()
    if path.is_symlink() or stat.st_uid != os.getuid() or stat.st_nlink != 1:
        return 'retain: ownership/link ambiguity'
    if stat.st_size > 32 * 1024 * 1024:
        return 'retain: large/unverified project identity; task example cab7545f335f075e has bioinformatics content'
    text = path.read_text(errors='replace')
    if re.search(r'1gbwt|1khash|bioinformatics|source_gap_bp|oriented_continuation|genome|genomics', text, re.I):
        return 'retain: other-project/bioinformatics evidence'
    markers = [m for m in ('/home/erikg/emender/', 'experiments/expressivity_tasks/',
                           'emender-scratch/fix-fanout', 'emender/.venv/') if m in text]
    if not markers:
        return 'retain: no positive Emender project identity'
    # No open descriptor is permitted, including read-only tails. An uncertain
    # lsof result fails closed instead of risking an active log replacement.
    opened = subprocess.run(['lsof', '-t', '--', str(path)], text=True, capture_output=True)
    if opened.returncode != 1 or opened.stdout.strip() or opened.stderr.strip():
        return 'retain: open descriptor or lsof uncertainty'
    return markers


def main():
    manifest_path = OUT / 'tmp_logs.json'
    if manifest_path.exists():
        raise RuntimeError('Refusing to overwrite compression manifest')
    manifest = {'scope': '/tmp/pi-bash-*.log positively identified inactive Emender logs only',
                'retained': [], 'compress': []}
    for p in sorted(Path('/tmp').glob('pi-bash-*.log')):
        st = p.lstat()
        identity = classify(p)
        if isinstance(identity, str) or Path(str(p) + '.gz').exists():
            manifest['retained'].append({'path': str(p), 'bytes': st.st_size,
                                         'mtime_ns': st.st_mtime_ns,
                                         'reason': identity if isinstance(identity, str) else 'gzip destination exists'})
            continue
        entry = {'path': str(p), 'gzip_path': str(p) + '.gz', 'bytes': st.st_size,
                 'inode': st.st_ino, 'device': st.st_dev, 'mtime_ns': st.st_mtime_ns,
                 'sha256': digest(p), 'project_identity_markers': identity}
        entry['manifest_entry_sha256'] = entry_hash(entry)
        manifest['compress'].append(entry)
    write_json(manifest_path, manifest)
    freed = 0
    with (OUT / 'tmp_logs.execution.jsonl').open('x') as log:
        for e in manifest['compress']:
            p = Path(e['path'])
            st = p.lstat()
            assert (st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns) == (e['inode'], e['device'], e['bytes'], e['mtime_ns'])
            opened = subprocess.run(['lsof', '-t', '--', str(p)], text=True, capture_output=True)
            assert opened.returncode == 1 and not opened.stdout.strip() and not opened.stderr.strip()
            dest = Path(e['gzip_path'])
            with dest.open('xb') as output, gzip.GzipFile(filename=p.name, mode='wb', fileobj=output, mtime=int(st.st_mtime)) as zipped, p.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    zipped.write(block)
            os.chmod(dest, 0o600)  # Never widen access to private session logs.
            check = hashlib.sha256()
            with gzip.open(dest, 'rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    check.update(block)
            assert check.hexdigest() == e['sha256']
            assert (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) == (e['bytes'], e['mtime_ns'], e['inode'])
            p.unlink()
            event = {'path': str(p), 'manifest_entry_sha256': e['manifest_entry_sha256'],
                     'gzip_path': str(dest), 'gzip_sha256': digest(dest),
                     'gzip_bytes': dest.stat().st_size, 'freed_bytes': e['bytes'] - dest.stat().st_size,
                     'uncompressed_sha256_verified': True}
            freed += event['freed_bytes']
            log.write(json.dumps(event) + '\n')
            log.flush()
            os.fsync(log.fileno())
    manifest['freed_bytes'] = freed
    manifest['compressed_count'] = len(manifest['compress'])
    write_json(manifest_path, manifest)
    print(json.dumps({'freed_bytes': freed, 'compressed_count': manifest['compressed_count'],
                      'retained_count': len(manifest['retained'])}))


if __name__ == '__main__':
    main()
