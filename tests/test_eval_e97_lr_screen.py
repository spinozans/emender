import json
import tiktoken
from scripts.e97_lr_screen_panel import make_panel
from scripts.eval_e97_lr_screen import evaluate_task


class FakeEngine:
    def __init__(self, outputs):
        self.encoding = tiktoken.get_encoding('p50k_base')
        self.outputs = iter(outputs)
        self.histories = []

    def encode(self, text):
        return self.encoding.encode_ordinary(text)

    def decode(self, ids):
        return self.encoding.decode(ids)

    def advance(self, ids, cache=None):
        history = (cache or []) + ids
        self.histories.append(self.decode(history))
        return history

    def generate(self, cache, **kwargs):
        assert kwargs['max_new_tokens'] == 4096 and kwargs['temperature'] == 0
        return self.encode(next(self.outputs)), cache


def test_all_scripted_fixture_replays_and_reasoning_history():
    panel = make_panel()
    for task in panel['tasks']:
        outputs = []
        for path in task['required_missing_reads'] + task['required_reads']:
            outputs.append('Analysis: "Read the actual fixture. λ"\nAction: read\nArguments: ' + json.dumps({'path': path, 'offset': 1, 'limit': 10}))
        outputs.append('Analysis: "Use observed values."\nFinal: ' + task['answer'])
        engine = FakeEngine(outputs)
        result = evaluate_task(engine, engine.encoding, panel, task)
        assert result['success'] and result['first_turn_protocol_valid']
        if task['required_reads']:
            assert 'Read the actual fixture. λ' in engine.histories[-1]
            assert '\n\nTool:\n' in engine.histories[-1]
            assert result['turns'][-1]['cache_event'] == 'hit'


def test_missing_analysis_wrong_final_and_unobserved_final_fail():
    panel = make_panel()
    task = next(t for t in panel['tasks'] if t['kind'] == 'read')
    for text in ('Final: '+task['answer'], 'Analysis: "x"\nFinal: wrong',
                 'Analysis: "x"\nFinal: '+task['answer']):
        engine = FakeEngine([text])
        assert not evaluate_task(engine, engine.encoding, panel, task)['success']


def test_repeat_in_unchanged_fixture_stops():
    panel = make_panel()
    task = next(t for t in panel['tasks'] if t['kind'] == 'read')
    action = 'Analysis: "Inspect."\nAction: read\nArguments: ' + json.dumps({'path': task['required_reads'][0], 'offset': 1, 'limit': 10})
    engine = FakeEngine([action, action])
    result = evaluate_task(engine, engine.encoding, panel, task)
    assert not result['success'] and result['error'].startswith('no_progress')
