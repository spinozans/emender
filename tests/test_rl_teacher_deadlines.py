"""F7 deploy-at-relaunch correction timeout contract; no endpoint requests."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

SPEC = importlib.util.spec_from_file_location("rl_teacher_deadlines",
    Path(__file__).resolve().parents[1] / "scripts/rl-loop-v1/rl_teacher_deadlines.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def pilot_fixture():
    entries = []
    metrics = SimpleNamespace(record=entries.append)
    legacy = []
    pilot = SimpleNamespace(lunaroute_call=lambda *args: legacy.append(args) or ("legacy", {}))
    def make(model, tools, enc, metrics):
        def generate(prompt, budget, deadline):
            return pilot.lunaroute_call(model, [], .3, budget, metrics)
        generate.state = {"fixture": True}
        return generate
    pilot.make_teacher_generate = make
    module.harden_teacher_deadlines(pilot)
    return pilot, metrics, entries, legacy


def test_retry_timeout_and_sleep_use_remaining_episode_budget():
    pilot, metrics, entries, legacy = pilot_fixture()
    clock = [100.0]
    timeouts, sleeps = [], []
    def call(request, *, timeout):
        timeouts.append(timeout)
        clock[0] += 1
        raise TimeoutError("endpoint stalled")
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    with patch.object(module.Path, "read_text", return_value='{"lunaroute":{"access":"test"}}'), \
         patch.object(module.time, "monotonic", side_effect=lambda: clock[0]), \
         patch.object(module.time, "sleep", side_effect=sleep), \
         patch.object(module.urllib.request, "urlopen", side_effect=call):
        generate = pilot.make_teacher_generate("model", [], None, metrics)
        with pytest.raises(TimeoutError, match="deadline exhausted"):
            generate("prompt", 32, 105)
    assert timeouts == [5]
    assert sleeps == [4]
    assert len(entries) == 1 and "error" in entries[0]
    assert generate.state == {"fixture": True}
    assert pilot.lunaroute_call("judge", [], 0, 32, metrics)[0] == "legacy"
    assert len(legacy) == 1  # Context deadline never leaks into judges.


def test_success_one_metric_no_deadline_leak_and_idempotent_binding():
    pilot, metrics, entries, legacy = pilot_fixture()
    bound = pilot.make_teacher_generate
    module.harden_teacher_deadlines(pilot)
    assert pilot.make_teacher_generate is bound
    response = io.StringIO(json.dumps({"choices": [{"message": {"content": " frame "}}],
                                      "usage": {"completion_tokens": 7}}))
    with patch.object(module.Path, "read_text", return_value='{"lunaroute":{"access":"test"}}'), \
         patch.object(module.time, "monotonic", return_value=100), \
         patch.object(module.urllib.request, "urlopen", return_value=response) as call:
        assert pilot.make_teacher_generate("model", [], None, metrics)("prompt", 32, 110)[0] == "frame"
        assert call.call_args.kwargs["timeout"] == 10
    assert len(entries) == 1 and entries[0]["usage"]["completion_tokens"] == 7
    assert pilot.lunaroute_call("judge", [], 0, 32, metrics)[0] == "legacy"


def test_late_response_rejected_without_success_metric():
    pilot, metrics, entries, _ = pilot_fixture()
    response = io.StringIO(json.dumps({"choices": [{"message": {"content": "late"}}]}))
    times = iter([100, 100, 111, 111, 111, 111])
    with patch.object(module.Path, "read_text", return_value='{"lunaroute":{"access":"test"}}'), \
         patch.object(module.time, "monotonic", side_effect=lambda: next(times)), \
         patch.object(module.urllib.request, "urlopen", return_value=response):
        with pytest.raises(TimeoutError, match="deadline exhausted"):
            pilot.make_teacher_generate("model", [], None, metrics)("prompt", 32, 110)
    assert len(entries) == 1 and "error" in entries[0] and "usage" not in entries[0]
