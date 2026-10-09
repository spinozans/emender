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
