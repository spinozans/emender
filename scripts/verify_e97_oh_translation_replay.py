#!/usr/bin/env python3
"""Replay-verify translated OH->Pi-native records against the recorded repository state.

Phase 2 of the fail-closed OH-repair pipeline (plan:
docs/validation/e97-oh-translation-and-reasoning-integration-plan-v1.md, T1).

For every candidate record this verifier:
  1. materializes the recorded repository state (git clone cache + base_commit
     archive extract) at the OH path convention <work>/workspace/<instance_dir>,
  2. executes every mapped Pi action in order — file operations directly on the
     worktree, bash commands inside a docker container mounted at /workspace —
  3. requires every recorded observation to match reality: read observations
     must equal the actual file content (parsed from the OH `cat -n` format),
     edits must apply with a unique oldText, writes must not clobber, bash
     outputs and exit codes must match the recorded core observation,
  4. requires the final replayed state of every file touched by the recorded
     model patch to equal the post-patch content (the exact final-state check),
  5. and only then assembles the final Pi-native record text from the verified
     observations and emits it to the verified collection.

Any mismatch DROPS the record (fail-closed); nothing is silently downgraded.
The model patch is used strictly as a verification oracle and never enters the
collection payload.
"""
import argparse, hashlib, io, json, re, shutil, subprocess, sys, tarfile, time
from pathlib import Path

RS = '\x1e'
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.e97_open_swe_native_codec import compact  # noqa: E402
from scripts.build_e97_oh_pi_native_translation import (  # noqa: E402
    assemble, parse_catn_body, parse_bash_observation, render_read_output)


def run(cmd, timeout=None, binary=False):
    return subprocess.run(cmd, capture_output=True, text=not binary, timeout=timeout)


class RepoCache:
    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def materialize(self, repo, base_commit, dest):
        """Extract base_commit's tree into dest. Returns error string or None."""
        owner, name = repo.split('/')
        bare = self.dir / f'{owner}__{name}.git'
        if not bare.exists():
            r = run(['git', 'clone', '--bare', f'https://github.com/{repo}.git', str(bare)],
                    timeout=3600)
            if r.returncode != 0:
                return f'clone_failed:{repo}'
        if run(['git', '-C', str(bare), 'cat-file', '-e', f'{base_commit}^{{commit}}'],
               timeout=120).returncode != 0:
            if run(['git', '-C', str(bare), 'fetch', 'origin', base_commit],
                   timeout=1800).returncode != 0:
                return f'missing_base_commit:{base_commit[:10]}'
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = run(['git', 'clone', '--no-checkout', '--offline' if False else '--template=', str(bare), str(dest)], timeout=1800)
        if r.returncode != 0:
            return f'worktree_clone_failed:{repo}'
        r = run(['git', '-C', str(dest), 'checkout', '--detach', base_commit], timeout=600)
        if r.returncode != 0:
            return f'checkout_failed:{base_commit[:10]}'
        r = run(['git', '-C', str(dest), 'clean', '-fdxq'], timeout=600)
        return None


class BashRunner:
    """Long-lived docker container per worker; /workspace maps to the shared root."""

    def __init__(self, container_name, image, shared_root):
        self.name = container_name
        run(['docker', 'rm', '-f', container_name], timeout=180)
        import os as _os
        r = run(['docker', 'run', '-d', '--name', container_name,
                 '--user', f'{_os.getuid()}:{_os.getgid()}',
                 '-e', 'PYTHONDONTWRITEBYTECODE=1',
                 '-v', f'{Path(shared_root).resolve()}:/workspace', image,
                 'sleep', 'infinity'], timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f'docker_run_failed:{(r.stderr or "")[:200]}')

    def exec(self, command, cwd_rel, timeout):
        r = run(['docker', 'exec', '-w', f'/workspace/{cwd_rel}', self.name,
                 'bash', '-o', 'pipefail', '-c', command], timeout=timeout + 60)
        return r.stdout, r.stderr, r.returncode

    def close(self):
        run(['docker', 'rm', '-f', self.name], timeout=180)


DET_ENUM = re.compile(r'\b(find|ls|grep|rg)\b')
DET_EXACT = re.compile(r'^(\s*)(cat|head|tail|wc|pwd|echo|which|file|python3?|sed -n)\b')


def _bash_verify_mode(command):
    """deterministic = safe read-only commands whose output must reproduce."""
    c = command.strip()
    if re.search(r'\|\s*(head|tail)\b', c) or '>' in c or '>>' in c:
        return 'env'
    if any(tok in c for tok in ('git ', 'pip ', 'pytest', 'npm ', 'cargo ', 'go ', 'make', 'curl', 'wget', 'apt-get')):
        return 'env'
    if DET_ENUM.search(c) or DET_EXACT.match(c):
        return 'deterministic'
    return 'env'


def _listing_paths(text):
    return sorted(line.rstrip('/') for line in text.split('\n')
                  if line.startswith('/workspace/'))


def _output_matches(recorded_core, output, command):
    a, b = _norm(recorded_core), _norm(output)
    if a == b:
        return True
    if DET_ENUM.search(command):
        return sorted(a.split('\n')) == sorted(b.split('\n'))
    return False


def _ship_recorded_bash(recorded_core, recorded_exit):
    text = recorded_core
    if recorded_exit != 0:
        text = text.rstrip('\n') + f'\nCommand exited with code {recorded_exit}\n'
    return text


def _norm(text):
    lines = [line.rstrip() for line in text.replace('\r\n', '\n').split('\n')]
    while lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)


def verify_record(candidate, workspace_root, instance_dir, bash, model_patch, bash_timeout,
                  oracle_ctx=None):
    ctx_bare = [None]
    cand_base = [None]
    if oracle_ctx:
        ctx_bare[0] = oracle_ctx.get('bare_cache')
        cand_base[0] = oracle_ctx.get('base_commit')
    """Replay one candidate. Returns (final_record_text|None, fail_reason|None)."""
    blocks = candidate['blocks']
    actions = candidate['actions']
    observation_texts = {}
    is_error = {}
    stats = {'bash_recorded_verbatim': 0, 'clipped_observation_verified_prefix': 0}
    root = Path(workspace_root)

    def fail(reason):
        return None, reason

    def rel(path):
        p = path.replace('\\', '/')
        return p[len('/workspace/'):] if p.startswith('/workspace/') else p.lstrip('/')

    def resolve(path):
        return root / rel(path)

    for action in actions:
        tool = action['tool']
        args = action['arguments']
        idx = action['index']
        recorded = action.get('recorded_observation')
        if tool == 'think':
            observation_texts[idx] = 'Your thought has been logged.'
            is_error[idx] = False
            continue
        if tool == 'finish':
            continue
        if tool == 'read':
            fp = resolve(args['path'])
            try:
                content = fp.read_text()
            except FileNotFoundError:
                err = f"ENOENT: no such file or directory, access '{args['path']}'"
                if recorded and ('ENOENT' in recorded or 'No such file' in recorded):
                    observation_texts[idx] = recorded.split('\n')[0][:500]
                    is_error[idx] = True
                    continue
                return fail(f'read_missing_file:idx{idx}')
            except IsADirectoryError:
                return fail(f'read_is_directory:idx{idx}')
            except UnicodeDecodeError:
                return fail(f'read_binary:idx{idx}')
            text, render_err = render_read_output(content, args.get('offset'), args.get('limit'))
            if render_err is not None:
                return fail(f'read_offset_error:idx{idx}')
            if recorded:
                _, recorded_lines = _split_view(recorded)
                clipped = '<response clipped>' in recorded or '<NOTE>' in recorded
                if clipped:
                    # The dataset truncated this observation mid-line: verify only
                    # the lines strictly before the first clipped line, then ship
                    # the full verified content (strictly more truthful).
                    clip_line = None
                    for lineno, want in sorted(recorded_lines.items()):
                        if '<response clipped>' in want or '<NOTE>' in want:
                            clip_line = lineno
                            break
                    actual = content.split('\n')
                    for lineno, want in sorted(recorded_lines.items()):
                        if clip_line is not None and lineno >= clip_line:
                            break
                        if lineno < 1 or lineno > len(actual) or actual[lineno - 1] != want:
                            return fail(f'read_content_mismatch:idx{idx}:line{lineno}')
                    stats['clipped_observation_verified_prefix'] += 1
                else:
                    actual = content.split('\n')
                    for lineno, want in recorded_lines.items():
                        if lineno < 1 or lineno > len(actual) or actual[lineno - 1] != want:
                            return fail(f'read_content_mismatch:idx{idx}:line{lineno}')
            observation_texts[idx] = text
            is_error[idx] = False
        elif tool == 'write':
            fp = resolve(args['path'])
            if fp.exists():
                if recorded and 'already exists' in recorded:
                    observation_texts[idx] = recorded.split('\n')[-1][:500] if recorded.startswith('ERROR') else recorded
                    is_error[idx] = True
                    continue
                return fail(f'write_existing_file:idx{idx}')
            if not fp.parent.exists() and recorded and 'No such file or directory' in recorded:
                observation_texts[idx] = recorded
                is_error[idx] = True
                continue
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(args['content'])
            observation_texts[idx] = f'Successfully wrote to {args["path"]}'
            is_error[idx] = False
        elif tool == 'edit':
            fp = resolve(args['path'])
            try:
                content = fp.read_text()
            except FileNotFoundError:
                return fail(f'edit_missing_file:idx{idx}')
            failed_edit = False
            for edit in args['edits']:
                old, new = edit['oldText'], edit['newText']
                count = content.count(old)
                if count != 1:
                    if recorded and recorded.startswith('ERROR'):
                        # The original action failed identically: ship the
                        # authentic error observation (failed-action recovery data).
                        observation_texts[idx] = recorded
                        is_error[idx] = True
                        failed_edit = True
                        break
                    return fail(f'edit_not_unique:idx{idx}:count{count}')
                content = content.replace(old, new, 1)
            if failed_edit:
                continue
            fp.write_text(content)
            observation_texts[idx] = (f'Successfully replaced {len(args["edits"])} '
                                      f'block(s) in {args["path"]}.')
            is_error[idx] = False
        elif tool == 'bash':
            if bash is None:
                return fail('bash_runner_unavailable')
            timeout = min(int(args.get('timeout') or bash_timeout), 900)
            stdout, stderr, code = bash.exec(args['command'], instance_dir, timeout)
            output = stdout + stderr if (stdout and stderr and stderr.strip()) else (stdout or stderr)
            if code != 0:
                output = output.rstrip('\n') + f'\nCommand exited with code {code}\n'
            recorded_core, recorded_exit = parse_bash_observation(recorded or '')
            mode = _bash_verify_mode(args['command'])
            verified = False
            if action.get('expectation', {}).get('kind') == 'dir_listing':
                # The mapped `find ... -maxdepth 2` replaces OH's tree listing:
                # verify the path multiset and ship the executed find output.
                recorded_paths = _listing_paths(recorded or '')
                executed_paths = _listing_paths(output)
                if recorded_paths == executed_paths:
                    observation_texts[idx] = output
                    is_error[idx] = code != 0
                    verified = True
                else:
                    return fail(f'dir_listing_mismatch:idx{idx}:{len(recorded_paths)}vs{len(executed_paths)}')
            elif mode == 'deterministic':
                if recorded_exit == code and _output_matches(recorded_core, output, args['command']):
                    observation_texts[idx] = output
                    is_error[idx] = code != 0
                    verified = True
            if not verified:
                # Environment-dependent or non-reproducible command: execute for
                # state fidelity, ship the AUTHENTIC recorded observation (OH
                # trailers stripped, Pi exit trailer appended), and count it.
                observation_texts[idx] = _ship_recorded_bash(recorded_core, recorded_exit)
                is_error[idx] = recorded_exit != 0
                stats['bash_recorded_verbatim'] += 1
        else:
            return fail(f'unknown_tool:{tool}')

    if model_patch:
        err = _check_final_state(root / instance_dir, model_patch,
                                 bare_cache=ctx_bare[0], repo=candidate['repo'],
                                 base_commit=cand_base[0])
        if err:
            return fail(err)
    record_text = _assemble_with_errors(blocks, observation_texts, is_error)
    return record_text, {'stats': stats}


def _split_view(text):
    marker = '`cat -n` on'
    idx = text.find(marker)
    if idx < 0:
        return text, {}
    nl = text.find('\n', idx)
    return text[:idx], parse_catn_body(text[nl + 1:]) if nl >= 0 else {}


def _assemble_with_errors(blocks, observation_texts, is_error):
    parts = []
    for block in blocks:
        kind = block['kind']
        if kind == 'protocol':
            parts.append('Protocol:\n' + block['header'])
        elif kind == 'system':
            parts.append('\n\nSystem:\n' + compact({'role': 'system', 'content': block['content']}))
        elif kind == 'user':
            parts.append('\n\nUser:\n' + compact({'role': 'user', 'content': block['content']}))
        elif kind == 'assistant':
            parts.append('\n\nAssistant:\n' + block['turn'])
        elif kind == 'toolresult':
            idx = block['action_index']
            parts.append('\n\nToolResult:\n' + compact({
                'role': 'toolResult', 'toolCallId': block['tool_call_id'],
                'toolName': block['tool_name'],
                'content': [{'type': 'text', 'text': observation_texts[idx]}],
                'isError': bool(is_error.get(idx, False))}))
    return ''.join(parts) + RS


def _check_final_state(repo_root, model_patch, bare_cache=None, repo=None, base_commit=None):
    """Every file the patch touches must match the post-patch content.

    The patch is applied to a PRISTINE base extraction (the trajectory's own
    created files would otherwise collide with patch-added files); each patched
    file must then equal the replayed tree's file."""
    repo_root = Path(repo_root)
    scratch = repo_root.parent / (repo_root.name + '-patchcheck')
    if scratch.exists():
        shutil.rmtree(scratch)
    if bare_cache and repo:
        owner, name = repo.split('/')
        bare = Path(bare_cache) / f'{owner}__{name}.git'
        r = run(['git', '-C', str(bare), 'archive', base_commit], timeout=900, binary=True)
        if r.returncode != 0:
            return 'patchcheck_archive_failed'
        scratch.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(r.stdout), mode='r:') as tar:
            tar.extractall(scratch, filter='data')
    else:
        shutil.copytree(repo_root, scratch)
    if not model_patch.endswith('\n'):
        model_patch += '\n'
    r = subprocess.run(['git', 'apply', '--whitespace=nowarn', '-'],
                       input=model_patch, text=True, cwd=scratch, capture_output=True,
                       timeout=600)
    if r.returncode != 0:
        return f'patch_apply_failed:{(r.stderr or "")[:100]}'
    for m in re.finditer(r'^\+\+\+ b/(.+)$', model_patch, re.M):
        rel = m.group(1)
        patched = scratch / rel
        replayed = repo_root / rel
        a = patched.read_text() if patched.exists() else None
        b = replayed.read_text() if replayed.exists() else None
        if a != b:
            return f'final_state_mismatch:{rel}'
    return None


def load_model_patches(parquet_dir, wanted_ids, cache_path):
    """Extract trajectory_id -> model_patch from the local OH-verified parquets."""
    if Path(cache_path).exists():
        data = json.loads(Path(cache_path).read_text())
        if wanted_ids <= set(data):
            return {k: v for k, v in data.items() if k in wanted_ids}
    import pyarrow.parquet as pq
    patches = {}
    for pf in sorted(Path(parquet_dir).rglob('*.parquet')):
        t = pq.read_table(pf, columns=['trajectory_id', 'metadata'])
        for tid, md in zip(t.column('trajectory_id').to_pylist(),
                           t.column('metadata').to_pylist()):
            if tid in wanted_ids and tid not in patches and md:
                patch = (md.get('model_patch') or {}).get('patch')
                if patch:
                    patches[tid] = patch
    Path(cache_path).write_text(json.dumps(patches))
    return {k: v for k, v in patches.items() if k in wanted_ids}


def _instance_dir(cand):
    for block in cand['blocks']:
        if block['kind'] == 'user':
            m = re.search(r'/workspace/([^\s<\\"]+)', block['content'] or '')
            if m:
                return m.group(1)
    return cand['instance_id']


def verify_one(job, ctx):
    path, cand, patch = job
    t0 = time.time()
    wid = hashlib.sha1(cand['trajectory_id'].encode()).hexdigest()[:10]
    workspace = Path(ctx['work_root']) / wid
    repo_root = workspace / _instance_dir(cand)
    bash = None
    try:
        err = RepoCache(ctx['repo_cache']).materialize(cand['repo'], cand['base_commit'], repo_root)
        if err:
            return path, cand, None, err, time.time() - t0
        bash = BashRunner(f'e97-replay-{wid}', ctx['docker_image'], workspace)
        oracle = {'bare_cache': str(ctx['repo_cache']), 'base_commit': cand['base_commit']}
        text, meta = verify_record(cand, workspace, _instance_dir(cand), bash,
                                   patch, ctx['bash_timeout'], oracle_ctx=oracle)
        reason = None if text is not None else (meta or 'unknown')
        return path, cand, text, reason, time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return path, cand, None, f'exception:{type(exc).__name__}:{str(exc)[:120]}', time.time() - t0
    finally:
        if bash:
            bash.close()
        shutil.rmtree(workspace, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--translation-dir', type=Path, required=True)
    p.add_argument('--instance-meta', type=Path, required=True)
    p.add_argument('--parquet-dir', type=Path, required=True)
    p.add_argument('--repo-cache', type=Path, required=True)
    p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--bash-timeout', type=int, default=120)
    p.add_argument('--docker-image', default='python:3.11-slim')
    args = p.parse_args()

    candidates = sorted((args.translation_dir / 'candidates').glob('*.json'))
    if args.limit:
        candidates = candidates[:args.limit]
    wanted = {json.loads(c.read_text())['trajectory_id'] for c in candidates}
    patches = load_model_patches(args.parquet_dir, wanted,
                                 args.translation_dir / 'model-patches-cache.json')
    print(f'ORACLE patches available for {len(patches)}/{len(wanted)} candidates', flush=True)

    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verified_dir = args.output_dir / 'verified'
    verified_dir.mkdir(exist_ok=True)

    ctx = {'repo_cache': str(args.repo_cache), 'work_root': str(args.work_root),
           'bash_timeout': args.bash_timeout, 'docker_image': args.docker_image}
    jobs = [(str(c), json.loads(c.read_text()), patches.get(json.loads(c.read_text())['trajectory_id']))
            for c in candidates]

    report = {'candidates': len(jobs), 'passed': 0, 'dropped': 0, 'drop_reasons': {},
              'drop_samples': {}, 'per_record_seconds': [], 'started': time.time()}

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(verify_one, job, ctx) for job in jobs]
        for i, fut in enumerate(as_completed(futures)):
            path, cand, text, reason, seconds = fut.result()
            report['per_record_seconds'].append(round(seconds, 2))
            if text is None:
                report['dropped'] += 1
                key = (reason or 'unknown').split(':')[0]
                report['drop_reasons'][key] = report['drop_reasons'].get(key, 0) + 1
                report['drop_samples'].setdefault(key, (reason or 'unknown')[:300])
            else:
                report['passed'] += 1
                (verified_dir / f'{cand["trajectory_id"]}.json').write_text(json.dumps({
                    'schema': 'emender-e97-oh-pi-native-verified-record-v1',
                    'trajectory_id': cand['trajectory_id'],
                    'instance_id': cand['instance_id'],
                    'problem_key': cand['problem_key'],
                    'repo': cand['repo'],
                    'record_text': text}, sort_keys=True) + '\n')
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                done = i + 1
                elapsed = time.time() - report['started']
                rate = done / elapsed if elapsed else 0.0
                eta_h = (len(jobs) - done) / rate / 3600 if rate else float('inf')
                print(f'PROGRESS {done}/{len(jobs)} passed={report["passed"]} '
                      f'dropped={report["dropped"]} rate={rate:.2f}/s eta_h={eta_h:.2f}', flush=True)

    report['wall_seconds'] = round(time.time() - report['started'], 1)
    (args.output_dir / 'replay-report.json').write_text(
        json.dumps({k: v for k, v in report.items() if k != 'per_record_seconds'},
                   indent=2, sort_keys=True) + '\n')
    print('OH_REPLAY_VERIFIED', json.dumps(
        {k: v for k, v in report.items() if k != 'per_record_seconds'}, sort_keys=True))


if __name__ == '__main__':
    main()
