"""Source-named era8 schema/mechanics/compatibility tests, run in stage cwd."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import tiktoken

import teacher_author as author
import standing_supply as supply
from scripts import e97_diversity as mechanics
from scripts import e97_diversity_author as engine
from scripts.e97_pi_native_codec import PiNativeEpisode, native_turn
from scripts.e97_pi_scripted_v8 import ScriptedEpisodeV8

ROOT = Path(os.environ['F9_STAGE_ROOT'])
NOTE = 'Operations notes: this depot keeps careful records of each transfer and reviews all posted summaries before the morning shift. The daily ledger is authoritative; obsolete notes should not be used.\n'


def terminal_fixture(tag='a'):
    return {'family': 'terminal', 'workspace': 'natural',
            'prompt': f'Read the daily ledger in `data/ledger-{tag}.csv`, double each numeric cargo quantity and post exact lines north:14 and south:22 (in that order, newline after each) in reports/summary.txt. Remove notes/obsolete.txt. Verify the posted report is nonempty.',
            'token': hashlib.sha256(tag.encode()).hexdigest()[:20],
            'files': [{'path': f'data/ledger-{tag}.csv', 'content': NOTE + 'north,7\nsouth,11\n'},
                      {'path': 'notes/obsolete.txt', 'content': NOTE}],
            'solution': {'actions': [{'tool': 'read', 'arguments': {'path': f'data/ledger-{tag}.csv'}},
                                     {'tool': 'bash', 'arguments': {'command': f"mkdir -p reports; awk -F, '/^[a-z]+,[0-9]+$/{{print $1 \":\" $2*2}}' data/ledger-{tag}.csv > reports/summary.txt; rm notes/obsolete.txt"}}],
                         'assertions': [{'kind': 'exact', 'path': 'reports/summary.txt', 'content': 'north:14\nsouth:22\n'},
                                        {'kind': 'absent', 'path': 'notes/obsolete.txt'},
                                        {'kind': 'command', 'argv': ['test', '-s', 'reports/summary.txt'], 'exit_code': 0, 'timeout': 2}]}}


def conversation_fixture():
    return {'family': 'conversation', 'workspace': 'natural',
            'prompt': 'Help prepare the depot notice. Ask which depot before using workspace tools. When I supply it, acknowledge with only a JSON object containing depot. Follow any subsequent correction before posting the notice, then finish with done.',
            'token': 'afaec09bef1839b74610',
            'files': [{'path': 'notes/operations.md', 'content': NOTE}, {'path': 'notes/dispatch.md', 'content': NOTE.replace('depot', 'dispatch')}],
            'solution': {'user_script': [
                {'on': {'question': 'Use north. Acknowledge with exactly {"depot":"north"}.'}, 'reply_pattern': 'Which depot should I use\\?', 'allow_tools': False},
                {'on': {'json': 'Correction: use south instead. Read notes/operations.md, post notices/depot.txt containing exactly depot=south followed by a newline, read it back, and finish with done.'}, 'reply_pattern': '\\{"depot":"north"\\}', 'allow_tools': False}],
                'final_pattern': 'done',
                'actions': [{'tool': 'finish', 'arguments': {'message': 'Which depot should I use?'}},
                            {'tool': 'finish', 'arguments': {'message': '{"depot":"north"}'}},
                            {'tool': 'read', 'arguments': {'path': 'notes/operations.md'}},
                            {'tool': 'write', 'arguments': {'path': 'notices/depot.txt', 'content': 'depot=south\n'}},
                            {'tool': 'read', 'arguments': {'path': 'notices/depot.txt'}},
                            {'tool': 'finish', 'arguments': {'message': 'done'}}],
                'assertions': [{'kind': 'exact', 'path': 'notices/depot.txt', 'content': 'depot=south\n'}]}}


@pytest.mark.parametrize('fixture', [terminal_fixture, conversation_fixture])
def test_new_family_schema_and_seed_instructions(fixture):
    spec = fixture()
    author.guards(spec, set(), set())
    instruction = author.author_instruction('depot', False, spec['family'], seed={'kind': 'error', 'payload': 'broken config', 'provenance': {'origin': 'invented'}})
    assert 'SEED' in instruction and 'solution' in instruction
    assert spec['family'] in instruction


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(workspace='thin'),
    lambda s: s['solution'].update(assertions=[]),
    lambda s: s['solution']['assertions'][0].update(path='../escape'),
    lambda s: s['solution']['actions'][0].update(tool='download'),
    lambda s: s['files'][0].update(content='stub\n'),
    lambda s: s['solution']['assertions'][0].update(content=''),
])
def test_terminal_schema_rejects_invalid(mutate):
    spec = terminal_fixture()
    mutate(spec)
    with pytest.raises((ValueError, KeyError)):
        author.guards(spec, set(), set())


def test_conversation_schema_requires_clarification_and_bounded_script():
    spec = conversation_fixture()
    spec['solution']['user_script'][0]['allow_tools'] = True
    with pytest.raises(ValueError):
        author.guards(spec, set(), set())
    with pytest.raises(ValueError):
        mechanics.script_schema(conversation_fixture()['solution']['user_script'] * 2)


def test_end_state_exact_regex_absent_command_and_no_token_echo(tmp_path):
    (tmp_path / 'answer.txt').write_text('count=42\n')
    checks = [{'kind': 'regex', 'path': 'answer.txt', 'pattern': 'count=[0-9]+\\n'},
              {'kind': 'absent', 'path': 'forbidden.txt'},
              {'kind': 'command', 'argv': ['test', '-s', 'answer.txt'], 'exit_code': 0, 'timeout': 1}]
    receipt = mechanics.capture_end_state(tmp_path, checks, 'a'*64)
    assert mechanics.check_end_state(checks, receipt, 'a'*64)
    assert not mechanics.check_end_state(checks, receipt, 'b'*64)
    (tmp_path / 'forbidden.txt').write_text('token=42\n')
    assert not mechanics.check_end_state(checks, mechanics.capture_end_state(tmp_path, checks, 'a'*64), 'a'*64)
    receipt['files']['answer.txt'] = 'token=42\n'
    assert not mechanics.check_end_state(checks, receipt, 'a'*64)


@pytest.mark.parametrize('component', ['leaf', 'parent'])
def test_end_state_rejects_symlink_escape(tmp_path, component):
    external = tmp_path / 'outside'
    external.mkdir()
    (external / 'text').write_text('real\n')
    root = tmp_path / 'workspace'
    root.mkdir()
    if component == 'leaf':
        (root / 'text').symlink_to(external / 'text')
        path = 'text'
    else:
        (root / 'directory').symlink_to(external, target_is_directory=True)
        path = 'directory/text'
    with pytest.raises((ValueError, OSError)):
        mechanics.capture_end_state(root, [{'kind': 'exact', 'path': path, 'content': 'real\n'}], 'a'*64)


def test_user_script_determinism_and_before_acting_gate():
    script = conversation_fixture()['solution']['user_script']
    assert mechanics.user_turn(script, 0, 'Which depot should I use?', False) == mechanics.user_turn(script, 0, 'Which depot should I use?', False)
    with pytest.raises(ValueError):
        mechanics.user_turn(script, 0, 'Which depot should I use?', True)
    with pytest.raises(ValueError):
        mechanics.user_turn(script, 0, 'I used north.', False)


def finish(message):
    return {'role': 'assistant', 'content': None, 'reasoning_content': None, 'tool_calls': [{'id': 'f', 'type': 'function', 'function': {'name': 'finish', 'arguments': mechanics.canonical({'message': message})}}]}


def test_versioned_script_continuation_and_legacy_finish_close():
    enc = tiktoken.get_encoding('p50k_base')
    tools = [{'name': 'read', 'label': 'read', 'description': 'Read text', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}]
    old = PiNativeEpisode(tools, enc)
    old.append_context({'role': 'user', 'content': 'opening'})
    frame = native_turn(finish('done'), tools)
    old.accept_generated_turn(frame)
    assert old.finished
    with pytest.raises(ValueError):
        old.append_context({'role': 'user', 'content': 'more'})
    assert old.text().endswith(frame)
    new = ScriptedEpisodeV8(tools, enc, conversation_fixture()['solution']['user_script'])
    new.append_context({'role': 'user', 'content': 'opening'})
    new.accept_generated_turn(native_turn(finish('Which depot should I use?'), tools))
    _, reply = new.append_scripted_user('Which depot should I use?', False)
    assert not new.finished and 'Use north' in reply
    assert new.prompt().endswith('Assistant:\n')


def test_morph_revalidation_dedupe_and_split_isolation():
    base = terminal_fixture()
    for trick in mechanics.MORPHS:
        with pytest.raises(ValueError):
            engine.validate_morph(base, deepcopy(base), trick, author, set(), set(), set())
    candidate = terminal_fixture('b')
    candidate['solution']['assertions'] = []
    with pytest.raises(ValueError):
        engine.validate_morph(base, candidate, 'distractor', author, set(), set(), set())
    candidate = terminal_fixture('b')
    candidate['split'] = 'development'
    with pytest.raises(ValueError):
        engine.validate_morph(base, candidate, 'entity_rename', author, set(), set(), set())
    candidate = terminal_fixture('b')
    assert engine.validate_morph(base, candidate, 'entity_rename', author, set(), set(), set())['morph']['trick'] == 'entity_rename'
    with pytest.raises(ValueError):
        engine.check_dedupe(candidate, set(), identity='a'*64, known_identities={'a'*64})


def test_rotation_invariant_exact_family_shares():
    from collections import Counter
    expected = {'first_action': 16, 'edit': 6, 'lookup': 6, 'recovery': 6, 'sum': 6, 'chat': 6, 'writing': 6, 'core': 1, 'terminal': 4, 'conversation': 3}
    assert supply.ROTATION_CYCLE == 60
    assert len(supply.ROTATION) == 60
    assert supply.ROTATION_SHARES == expected == dict(Counter(supply.ROTATION))


def test_validator_era4_verdict_and_proof_transcript_compatibility(tmp_path):
    from scripts import e97_first_party_validator_protocol_breadth as old
    from scripts import e97_first_party_validator_diversity as new
    # Authentic era4-shaped proof terminal; identical receipt checks in v8.
    text = 'Grounded notes from an office ledger.'
    spec = {'schema': 'emender-e97-first-party-validator-v2', 'task_identity': 'a'*64,
            'fixture_tree_digest': 'b'*64, 'archive_sha256': 'c'*64, 'program_sha256': 'd'*64,
            'interpreter_sha256': 'e'*64, 'expected_token': 'bookkeeping', 'required_read_path': 'notes.txt',
            'minefield': {'allowed_tools': ['list_files', 'read'], 'forbidden_paths': ['/', '..']},
            'expected_final': 'done', 'required_grounded_reads': [{'path': 'notes.txt', 'expected_text': text}]}
    raw, effective = author._gym_read_receipt('notes.txt', text)
    terminal = author._write_terminal('Read notes.txt and finish with done.', [{'sequence': 0, 'tool_name': 'read', 'arguments': {'path': 'notes.txt'}, 'is_error': False, 'arguments_json': '{"path":"notes.txt"}', 'raw_observation': raw, 'effective_observation': effective}], 'Final: done')
    assert old._spec(spec) == new._diversity_spec(spec)[0]
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(mechanics.canonical(spec))
    for final, expected in [('Final: done', 0), ('Final: incorrect', 1)]:
        terminal['messages'][-1]['content'] = final
        terminal_path = tmp_path / 'terminal.json'
        terminal_path.write_text(mechanics.canonical(terminal))
        for mode in ('focused', 'regression'):
            verdicts = [subprocess.run([sys.executable, str(ROOT / f'scripts/{program}.py'), '--mode', mode, '--spec', str(spec_path), '--terminal', str(terminal_path)], capture_output=True).returncode for program in ('e97_first_party_validator_protocol_breadth', 'e97_first_party_validator_diversity')]
            assert verdicts[0] == verdicts[1]
            assert (verdicts[0] != 0) == (expected != 0 and mode == 'focused')


def test_archive_retains_old_validator_bytes():
    from ndm.e97_first_party_source_archive import verified_source_member_payload
    payloads = author._authority_payloads()
    member, _ = verified_source_member_payload(payloads['generator-manifest.json'], payloads['source-archive.tar'], author.ERA4_VALIDATOR_MEMBER)
    assert member == (Path('/home/erikg/emender') / author.ERA4_VALIDATOR_MEMBER).read_bytes()


def test_real_pi_new_families_and_morph_all_build_gates(tmp_path):
    # No model/API/network request: reference actions use the actual owned Pi.
    fixtures = [terminal_fixture('proof'), conversation_fixture()]
    variant = terminal_fixture('morph')
    engine.validate_morph(fixtures[0], variant, 'entity_rename', author, set(), set(), set())
    fixtures.append(variant)
    for i, spec in enumerate(fixtures):
        spec['accepted_index'] = i
        spec['split'] = 'train'  # independent development isolation is schema tested above
    tranche = 'test-diversity-' + tmp_path.name
    directory = author.WORK / 'teacher-authored' / f'tranche-{tranche}'
    directory.mkdir(parents=True)
    for i, spec in enumerate(fixtures):
        (directory / f'authored-{i:04d}.json').write_text(json.dumps(spec))
    quarantine = author.build_tranche(tranche)
    proofs = json.loads((directory / 'proofs.json').read_bytes())
    assert len(proofs) == 3
    assert all(p['native_close_verified'] and p['degeneracy_screen'] == 'pass' for p in proofs)
    from ndm.e97_first_party_read_observe import validate_generated_quarantine
    validate_generated_quarantine(quarantine)
    from ndm.e97_protected_overlap import check_protected_overlap
    from ndm.e97_task_lake import validate_source_registry
    from lake_expand import PANELS
    overlap = check_protected_overlap(registry=validate_source_registry(json.loads((quarantine / 'source-registry.json').read_bytes())), candidate_collection=quarantine / 'tasks.jsonl', candidate_root=quarantine, protected_panels=PANELS)
    assert overlap['status'] == 'pass' and not any(overlap['collision_counts'].values())
    assert not (quarantine / 'admission-receipt.json').exists()


@pytest.mark.parametrize('command', ['curl https://example.org', 'rm /etc/passwd', 'cat ../secret', 'mkdir allowed\npython evil.py', 'cat $HOME/secret', 'cat $(pwd)', 'cat `pwd`', 'cat input &', 'awk \'BEGIN{system("touch sentinel")}\' data.csv', 'git config alias.pwn "!touch sentinel"', 'ln -s /etc data'])
def test_offline_shell_reference_gate_rejects(command):
    spec = terminal_fixture()
    spec['solution']['actions'][1]['arguments']['command'] = command
    with pytest.raises(ValueError):
        author.guards(spec, set(), set())


def test_offline_shell_policy_gate_before_pi_dispatch_retains_failure_for_pg():
    from scripts.e97_pi_scripted_v8 import SafeWorkspaceBridgeV8
    from scripts.e97_pi_native_tool_bridge import MODEL, PROVIDER, BridgeStopped
    import rl_loop_driver as driver
    enc = tiktoken.get_encoding('p50k_base')
    curriculum = driver.load_curriculum()
    tools = driver.tool_manifest(curriculum)['model_visible_tools']
    frame = native_turn({'role': 'assistant', 'content': None, 'reasoning_content': 'This unsafe action should be rejected without execution.', 'tool_calls': [{'id': 'bad', 'type': 'function', 'function': {'name': 'bash', 'arguments': mechanics.canonical({'command': 'touch /tmp/f9-policy-sentinel'})}}]}, tools)
    bridge = SafeWorkspaceBridgeV8(driver.make_panel(curriculum.SYSTEM, tools, driver.POLICY_PANEL), 'opening', enc, lambda *a: (frame, enc.encode_ordinary(frame), 'valid'))
    request = {'systemPrompt': curriculum.SYSTEM, 'model': MODEL, 'provider': PROVIDER, 'tools': tools, 'messages': deepcopy(bridge.history)}
    with pytest.raises(BridgeStopped):
        bridge.next(request)
    assert bridge.failed and bridge.reason == 'invalid_native_turn' and bridge.pending is None
    assert len(bridge.history) == 1  # Pi never received an executable call
    assert bridge.generations[-1]['reason'] == 'valid'
    assert bridge.episode.text().endswith(frame)
    from scripts.build_e97_pi_native_curriculum import encode_candidate
    _, mask, _ = encode_candidate(bridge.episode.text(), bridge.generations, 0, enc)
    assert sum(mask) > 0  # valid unsafe-action failure remains PG evidence


def test_assertion_commands_cannot_mutate_workspace():
    spec = terminal_fixture()
    spec['solution']['assertions'][2]['argv'] = ['rm', 'reports/summary.txt']
    with pytest.raises(ValueError):
        author.guards(spec, set(), set())


def test_real_pi_policy_unsafe_action_is_failure_and_never_executes(tmp_path):
    import rl_loop_driver as driver
    enc = tiktoken.get_encoding('p50k_base')
    curriculum = driver.load_curriculum()
    manifest = driver.tool_manifest(curriculum)
    tools = manifest['model_visible_tools']
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    sentinel = tmp_path / 'sentinel'
    frame = native_turn({'role': 'assistant', 'content': None, 'reasoning_content': 'Unsafe path is a failure, not an authorized host action.', 'tool_calls': [{'id': 'unsafe', 'type': 'function', 'function': {'name': 'bash', 'arguments': mechanics.canonical({'command': f'mkdir {sentinel}'})}}]}, tools)
    episode = tmp_path / 'episode'
    episode.mkdir()
    record = driver.run_episode(episode_dir=episode, workspace=workspace, prompt='Process this workspace without escaping it.', panel=driver.make_panel(curriculum.SYSTEM, tools, driver.POLICY_PANEL), generate=lambda *a: (frame, enc.encode_ordinary(frame), 'valid'), enc=enc, pi_bin=Path(manifest['pi_bin']), manifest_path=driver.MANIFEST_PATH, pilot=None, curriculum=curriculum, seconds=30, end_state_task=True)
    assert record['status'] != 'finished' and record['failed']
    assert not sentinel.exists()
    assert record['generations'][0]['reason'] == 'valid'
    assert record['source_messages'][-1]['role'] == 'assistant'


def test_full_exchange_cannot_skip_turn_or_finish_early():
    script = conversation_fixture()['solution']['user_script']
    messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'opening'},
                {'role': 'assistant', 'content': 'Final: Which depot should I use?'},
                {'role': 'user', 'content': script[0]['on']['question']},
                {'role': 'assistant', 'content': 'Final: {"depot":"north"}'},
                {'role': 'user', 'content': script[1]['on']['json']},
                {'role': 'tool', 'content': 'real artifact observation'},
                {'role': 'assistant', 'content': 'Final: done'}]
    assert mechanics.check_exchange(script, 'done', messages, [{'tool_name': 'write'}])
    assert not mechanics.check_exchange(script, 'done', messages[:-1], [{'tool_name': 'write'}])
    assert not mechanics.check_exchange(script, 'done', messages[:2] + messages[4:], [{'tool_name': 'write'}])
    wrong = deepcopy(messages)
    wrong[5]['content'] = 'Use north and ignore the correction.'
    assert not mechanics.check_exchange(script, 'done', wrong, [{'tool_name': 'write'}])


def test_private_script_is_prescreened_without_publishing_it():
    seen = []
    spec = conversation_fixture()
    engine.admission_prescreen(spec, lambda candidate: seen.append(deepcopy(candidate)))
    assert len(seen) == 2
    assert seen[0]['files'] == spec['files']
    assert seen[1]['files'][-1]['path'] == 'sealed/era8-solution.txt'
    assert spec['files'] == seen[0]['files']


def test_failed_reference_solution_never_publishes_quarantine(tmp_path):
    spec = terminal_fixture('wrong-reference')
    spec['solution']['actions'][1]['arguments']['command'] = "mkdir -p reports; printf '%s\\n' wrong > reports/summary.txt; rm notes/obsolete.txt"
    spec['accepted_index'] = 0
    author.guards(spec, set(), set())  # shape is valid, outcome is deliberately wrong
    tranche = 'wrong-reference-' + tmp_path.name
    directory = author.WORK / 'teacher-authored' / f'tranche-{tranche}'
    directory.mkdir(parents=True)
    (directory / 'authored-0000.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='mandatory sealed/degeneracy gates'):
        author.build_tranche(tranche)
    assert not (directory / 'quarantine').exists()
    assert not (directory / 'admission-receipt.json').exists()


def test_fresh_scripted_correction_rule_is_shared_by_pool_and_sync_paths(tmp_path):
    from rl_bank_lane import _correction_requires_fresh_solve
    private = tmp_path / 'validator.json'
    private.write_text(mechanics.canonical({'user_script': conversation_fixture()['solution']['user_script']}))
    binding = {'task_lake': {'validator': {'spec_path': str(private), 'spec_sha256': hashlib.sha256(private.read_bytes()).hexdigest()}}}
    assert _correction_requires_fresh_solve(binding, {'source_messages': []}, [])
    assert not _correction_requires_fresh_solve({}, {'source_messages': []}, [])


@pytest.mark.parametrize('kind', ['code', 'config', 'tree', 'error'])
def test_seed_object_schema_for_invented_fragments(kind):
    seed = {'kind': kind, 'payload': 'local invented fragment', 'provenance': {'origin': 'invented'}}
    assert 'SEED' in author.author_instruction(None, False, 'terminal', seed=seed)
    seed['provenance']['origin'] = 'external-benchmark'
    with pytest.raises(ValueError):
        engine.seed_instruction(seed)


def test_seed_harvest_is_admitted_solved_train_and_reverified():
    seeds = engine.harvest_solved(Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1'), author.TASK_LAKE, supply.verify_admitted, count=2)
    assert len(seeds) == 2
    for seed in seeds:
        assert seed['kind'] == 'solved_bundle' and seed['provenance']['split'] == 'train'
        assert engine.verify_seed(seed, supply.verify_admitted)['split'] == 'train'
        mutated = deepcopy(seed)
        mutated['payload']['prompt'] += ' drift'
        with pytest.raises(ValueError):
            engine.verify_seed(mutated, supply.verify_admitted)


@pytest.mark.parametrize('check', [{'kind': 'exact', 'path': 'result.txt', 'content': 'token=secret\n'}, {'kind': 'exact', 'path': 'result.txt', 'content': '0123456789abcdefabcd\n'}, {'kind': 'regex', 'path': 'result.txt', 'pattern': 'token=.*'}])
def test_new_families_reject_token_echo_artifacts(check):
    with pytest.raises(ValueError):
        mechanics.assertions_schema([check])


# F9R's exact validation-only repro: never send any command to a transport.
@pytest.mark.parametrize('command', ['printf %s {1..999999999}', 'printf %s ~root', 'printf %s *', 'printf %s file?', 'printf %s [ab]', 'printf %s {a,b}', 'pip install x', 'python evil.py', 'sh evil.sh', 'env cat file', 'bash -c cat'])
def test_f9r_expansion_rejected_by_reference_and_policy(command):
    from scripts.e97_offline_shell import shell_allowed
    from scripts.e97_pi_scripted_v8 import SafeWorkspaceBridgeV8
    from scripts.e97_pi_native_tool_bridge import MODEL, PROVIDER, BridgeStopped
    import rl_loop_driver as driver
    with pytest.raises(ValueError):
        shell_allowed(command)
    spec = terminal_fixture()
    spec['solution']['actions'][1]['arguments']['command'] = command
    with pytest.raises(ValueError):
        author.guards(spec, set(), set())
    enc = tiktoken.get_encoding('p50k_base')
    c = driver.load_curriculum()
    tools = driver.tool_manifest(c)['model_visible_tools']
    frame = native_turn({'role': 'assistant', 'content': None,
        'reasoning_content': 'Check admission only; do not execute.',
        'tool_calls': [{'id': 'probe', 'type': 'function', 'function': {
            'name': 'bash', 'arguments': mechanics.canonical({'command': command})}}]}, tools)
    b = SafeWorkspaceBridgeV8(driver.make_panel(c.SYSTEM, tools, driver.POLICY_PANEL),
        'opening', enc, lambda *a: (frame, enc.encode_ordinary(frame), 'valid'))
    with pytest.raises(BridgeStopped):
        b.next({'systemPrompt': c.SYSTEM, 'model': MODEL, 'provider': PROVIDER,
            'tools': tools, 'messages': deepcopy(b.history)})
    assert b.failed and b.pending is None and len(b.history) == 1
    assert b.episode.text().endswith(frame)


def test_raw_quote_boundary_preserves_authorized_literals():
    from scripts.e97_offline_shell import shell_allowed
    shell_allowed(terminal_fixture()['solution']['actions'][1]['arguments']['command'])
    shell_allowed("printf '%s\\n' '{a,b} ~root * [ab] ?' > report.txt")
    shell_allowed('printf %s "{a,b} ~root * [ab] ?"')
    for command in ['printf %s "$(pwd)"', 'printf %s \\*', 'printf %s x # ignored',
                    terminal_fixture()['solution']['actions'][1]['arguments']['command'].replace("'", '"')]:
        with pytest.raises(ValueError):
            shell_allowed(command)


def test_sealed_validator_regeneration_and_both_archive_closures():
    import runpy
    source = Path(__file__).resolve().parent
    generated = runpy.run_path(str(source / 'sync_validator.py'))['validator_source']()
    assert generated.encode() == (source / 'scripts/e97_first_party_validator_diversity.py').read_bytes()
    namespace = {}
    exec(compile(generated, '<sealed-validator>', 'exec'), namespace)
    for command in ['printf %s {1..999999999}', 'printf %s ~root', 'printf %s *']:
        with pytest.raises(ValueError):
            namespace['shell_allowed'](command)
    from ndm.e97_first_party_source_archive import _manifest_components_payload
    old = json.loads(Path('/home/erikg/emender/configs/pi/e97-firstparty-generator-manifest-v1.json').read_bytes())
    _manifest_components_payload(mechanics.canonical(old).encode())
    additive = deepcopy(old)
    additive['components'].append({'path': 'scripts/e97_first_party_validator_diversity.py',
        'sha256': hashlib.sha256(generated.encode()).hexdigest()})
    additive['components'].sort(key=lambda c: c['path'])
    _manifest_components_payload(mechanics.canonical(additive).encode())
    additive['components'].append({'path': 'scripts/unauthorized.py', 'sha256': 'a'*64})
    with pytest.raises(ValueError):
        _manifest_components_payload(mechanics.canonical(additive).encode())


def test_integrated_current_correction_sources_and_supply_knobs():
    import rl_bank_lane
    assert supply.TRANCHE_SIZE == 96
    assert (supply.LANE_MIN, supply.LANE_START, supply.LANE_SOFT_CAP, supply.LANE_HARD_CAP) == (24, 24, 48, 48)
    assert rl_bank_lane.CorrectionService.__module__ == 'rl_correction_service'
    repo = Path('/home/erikg/emender/scripts/rl-loop-v1')
    for name in ('rl_correction_service.py', 'rl_correction_pool.py', 'rl_correction_config.py', 'rl_bank.py'):
        assert (ROOT / 'loop' / name).read_bytes() == (repo / name).read_bytes()


def test_rollback_boundary_is_read_only_and_refuses_inflight_work(tmp_path):
    import runpy
    gate = runpy.run_path(str(Path(__file__).resolve().parent / 'rollback_gate.py'))['rollback_inventory']
    for directory in ('pool/pending', 'pool/claims', 'lanes'):
        (tmp_path / directory).mkdir(parents=True)
    assert gate(tmp_path)['old_only_restore_allowed']
    spec = tmp_path / 'sealed.json'
    spec.write_text(mechanics.canonical({'user_script': conversation_fixture()['solution']['user_script']}))
    task = {'task_id': 'new', 'body': {'task_lake': {'validator': {'spec_path': str(spec), 'spec_sha256': hashlib.sha256(spec.read_bytes()).hexdigest()}}}}
    pending = tmp_path / 'pool/pending/new.json'
    pending.write_text(mechanics.canonical(task))
    assert not gate(tmp_path)['old_only_restore_allowed']
    pending.rename(tmp_path / 'pool/claims/new.claim')
    assert not gate(tmp_path)['old_only_restore_allowed']
    job = tmp_path / 'lanes/lane-00/corrections/job-new'
    job.mkdir(parents=True)
    (job / 'spec.json').write_text(mechanics.canonical({'task': task}))
    assert len(gate(tmp_path)['unpublished_correction_jobs']) == 1
    (tmp_path / 'pool/claims/new.claim').unlink()  # fixture only: simulate compatible retirement
    assert not gate(tmp_path)['old_only_restore_allowed']
    (job / 'published.json').write_text(mechanics.canonical({'published_unix': 1, 'outcome': {'task_id': 'new'}}))
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert gate(tmp_path)['old_only_restore_allowed']
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    pending.write_text(mechanics.canonical(task))
    spec.write_text('corrupted')
    with pytest.raises(ValueError):
        gate(tmp_path)


def _seed_corpus_fixture(tmp_path):
    seeds = [{'kind': 'error', 'payload': 'invented broken config', 'provenance': {'origin': 'invented'}},
             {'kind': 'solved_bundle', 'payload': {'prompt': 'admitted training task'}, 'provenance': {'split': 'train'}}]
    corpus = tmp_path / 'seed-corpus.json'
    corpus.write_text(mechanics.canonical(seeds))
    manifest = {'schema': 'emender-era8-local-seed-corpus-v1', 'seeds': len(seeds),
                'corpus_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest(),
                'items': [{'kind': s['kind'], 'seed_sha256': engine.digest(s), 'provenance': s['provenance']} for s in seeds]}
    path = tmp_path / 'seed-manifest.json'
    path.write_text(mechanics.canonical(manifest))
    return seeds, corpus, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_intake_checks_manifest_corpus_and_authority_before_use(tmp_path, monkeypatch):
    seeds, corpus, manifest, pin = _seed_corpus_fixture(tmp_path)
    verified, logs = [], []
    admitted = object()
    def authority(seed, verify_admitted):
        assert verify_admitted is admitted
        verified.append(seed)
    monkeypatch.setattr(engine, 'verify_seed', authority)
    assert engine.load_verified_seed_corpus(corpus, pin, admitted, logs.append) == seeds
    assert verified == [seeds[1]] and logs[0].startswith('SEED_INTAKE verified:')


@pytest.mark.parametrize('fault', ['no_pin', 'no_manifest', 'manifest_bytes', 'no_corpus', 'corpus_bytes', 'count', 'entry', 'authority', 'external_fragment'])
def test_seed_intake_explicitly_disables_absent_corrupt_or_unauthorized(tmp_path, monkeypatch, fault):
    seeds, corpus, manifest, pin = _seed_corpus_fixture(tmp_path)
    checked, logs = [], []
    def authority(seed, verify_admitted):
        checked.append(seed)
        if fault == 'authority':
            raise ValueError('admission authority drift')
    monkeypatch.setattr(engine, 'verify_seed', authority)
    if fault == 'no_pin': pin = None
    elif fault == 'no_manifest': manifest.unlink()
    elif fault == 'manifest_bytes': manifest.write_text('{}')
    elif fault == 'no_corpus': corpus.unlink()
    elif fault == 'corpus_bytes': corpus.write_text('[]')
    elif fault in {'count', 'entry', 'external_fragment'}:
        m = json.loads(manifest.read_bytes())
        if fault == 'count': m['seeds'] += 1
        elif fault == 'entry': m['items'][1]['seed_sha256'] = '0'*64
        else:
            seeds[0]['provenance']['origin'] = 'external-benchmark'
            corpus.write_text(mechanics.canonical(seeds))
            m['corpus_sha256'] = hashlib.sha256(corpus.read_bytes()).hexdigest()
        manifest.write_text(mechanics.canonical(m))
        pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert engine.load_verified_seed_corpus(corpus, pin, object(), logs.append) == []
    assert len(logs) == 1 and logs[0].startswith('SEED_INTAKE disabled:')
    if fault != 'authority': assert not checked


def test_retained_real_seed_corpus_reverified_for_supply():
    corpus = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-final/author-work/seed-corpus.json')
    logs = []
    seeds = engine.load_verified_seed_corpus(corpus, 'bfa5e4418afd8f20aeaaccbe44cafcec413baa568cc904c988e5878b5d950a13', supply.verify_admitted, logs.append)
    assert len(seeds) == 12 and sum(s['kind'] == 'solved_bundle' for s in seeds) == 8
    assert logs[0].startswith('SEED_INTAKE verified:')


def test_supply_seed_intake_logs_disable_at_production_callsite(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(supply, 'WORK', tmp_path)
    monkeypatch.setenv('F9_AUTHOR_WORK', str(tmp_path))
    monkeypatch.delenv('F9_SEED_MANIFEST_SHA256', raising=False)
    monkeypatch.setattr(author, 'lake_token_tree_index', lambda: (set(), set()))
    monkeypatch.setattr(engine, 'lake_dedupe_index', lambda lake: (set(), set()))
    def no_teacher(*args, **kwargs):
        raise AssertionError('no teacher API authorized in this test')
    monkeypatch.setattr(author, 'lunaroute_call', no_teacher)
    result = supply.author_parallel(0, count=0, lanes=0, state={})
    assert result['calls'] == 0
    assert 'SEED_INTAKE disabled:' in capsys.readouterr().out
    assert 'F9_SEED_MANIFEST_SHA256 absent/invalid' in (tmp_path / 'teacher-authored/tranche-000/seed-intake.log').read_text()


def test_repaired_archive_pins_and_legacy_validator_bytes():
    import tarfile
    from ndm.e97_first_party_source_archive import verify_archive_members
    manifest = ROOT / 'configs/pi/e97-firstparty-generator-manifest-v1.json'
    archive = ROOT / 'configs/pi/e97-firstparty-source-v1.tar'
    digest = verify_archive_members(manifest, archive)
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    registry = json.loads((ROOT / 'configs/pi/e97-onpolicy-source-registry-v1.json').read_bytes())
    pins = [s['receipts']['source_archive_sha256'] for s in registry['sources'] if s['kind'] == 'first-party' and s['status'] == 'admitted']
    assert len(pins) == 2 and pins == [digest, digest]
    with tarfile.open(archive) as tar:
        for name in ['scripts/e97_first_party_validator.py', 'scripts/e97_first_party_validator_first_action.py', 'scripts/e97_first_party_validator_protocol_breadth.py']:
            assert tar.extractfile(name).read() == (Path('/home/erikg/emender') / name).read_bytes()
        name = 'scripts/e97_first_party_validator_diversity.py'
        assert tar.extractfile(name).read() == (ROOT / name).read_bytes()


@pytest.mark.parametrize('identity', ['18c99597ae7d714c927c6362f59339e241abe6310299e971b00a22e6d9fc0cb9', 'b839a3659911beee5feeadf86726fbc63d6a044ae9aef098cb68dff92745e805', 'aae2fc331246ad5b7f332493206bc0ee8e9ceb988a1eea90c9c900819d486aa2'])
def test_retained_terminals_regraded_under_repaired_sealed_bytes(identity, tmp_path):
    historical = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-lake-expansion-v1/F9-staged/qualified-final/author-work/teacher-authored/tranche-test-diversity-test_real_pi_new_families_and_0/quarantine')
    spec = historical / 'private_validators' / f'{identity}.json'
    terminal = historical / 'era8-proofs' / identity / 'gym-terminal-projection-proof.json'
    command = [sys.executable, str(ROOT / 'scripts/e97_first_party_validator_diversity.py')]
    for mode in ('focused', 'regression'):
        cp = subprocess.run([*command, '--mode', mode, '--spec', str(spec), '--terminal', str(terminal)], capture_output=True, timeout=30)
        assert cp.returncode == 0, cp.stderr.decode()
    mutant = json.loads(terminal.read_bytes())
    private = json.loads(spec.read_bytes())
    check = next(c for c in private['end_state_assertions'] if c['kind'] in {'exact', 'regex'})
    mutant['sealed_end_state']['files'][check['path']] = None
    path = tmp_path / 'missing-output.json'
    path.write_text(mechanics.canonical(mutant))
    cp = subprocess.run([*command, '--mode', 'focused', '--spec', str(spec), '--terminal', str(path)], capture_output=True, timeout=30)
    assert cp.returncode != 0 and b'sealed end state failed' in cp.stderr
