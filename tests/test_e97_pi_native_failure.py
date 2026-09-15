"""Actual Pi error termination with scripted model failures; zero model loads."""
import json
from pathlib import Path
import shutil
import pytest
import tiktoken
from scripts.e97_open_swe_native_codec import native_turn, compact
from scripts.e97_pi_native_bridge import NativePiBridge, BridgeStopped
from scripts.e97_pi_native_transport import serve_pi
from scripts.e97_pi_native_failure import verify_model_failure


@pytest.mark.parametrize('reason', ['turn_budget','invalid_opening'])
def test_real_pi_failed_task_is_not_success_or_transport_corruption(tmp_path, reason):
    pi = shutil.which('pi')
    if not pi: pytest.skip('Pi CLI unavailable')
    panel = dict(system='native failure control', tools=[dict(type='function', function=dict(name=n)) for n in
                 ('execute_bash','str_replace_editor','think','finish')], max_turns=1,
                 generation_budget=4096, episode_generation_budget=8192, episode_seconds=30)
    enc = tiktoken.get_encoding('p50k_base'); generated=[]
    text = native_turn(dict(role='assistant', content=None, reasoning_content='PRIVATE_FAILURE_SENTINEL', think=None,
                       tool_calls=[dict(type='function', function=dict(name='think', arguments=compact(dict(thought='PRIVATE_THOUGHT_SENTINEL'))))]))
    def generate(prompt, budget, deadline):
        generated.append(prompt)
        return (text, enc.encode_ordinary(text), 'valid') if reason=='turn_budget' else (None,[1],'invalid_opening')
    def execute(call): raise AssertionError('unexpected external tool')
    b = NativePiBridge(panel, 'Failure control', enc, generate, execute)
    out=tmp_path/'pi'
    with pytest.raises(BridgeStopped, match='pi_transport_not_qualified'):
        serve_pi(b,out,pi_bin=pi,extension=Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts',seconds=45)
    evidence=verify_model_failure(b,out)
    assert evidence['model_failure_transport_verified'] and not evidence['task_succeeded']
    assert len(generated)==1 and b.final is None
    public=(out/'pi-events-private.jsonl').read_text()
    assert 'PRIVATE_FAILURE_SENTINEL' not in public and 'PRIVATE_THOUGHT_SENTINEL' not in public
    path=out/'requests-private.jsonl'; rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[-1]['messages'][0]['content'][0]['text']='changed'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='failed_history_changed'): verify_model_failure(b,out)
