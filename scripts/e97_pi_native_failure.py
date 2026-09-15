"""Read-only validation of a naturally failed model turn through real Pi.

A failed task never becomes a verified finish. This only distinguishes an
expected empty Pi error termination from a transport/history failure.
"""
import json
from pathlib import Path
from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_bridge import API, MODEL, PROVIDER, normalized_messages

MODEL_STOP_REASONS = frozenset({'turn_budget', 'episode_generation_budget', 'episode_deadline',
    'context_budget', 'generation_budget', 'invalid_opening', 'invalid_frame', 'empty',
    'separator_before_valid_turn'})


def verify_model_failure(bridge, output):
    output = Path(output)
    if (bridge.reason not in MODEL_STOP_REASONS or not bridge.failed or not bridge.closed or
            bridge.close_verified or bridge.final is not None or bridge.pending is not None):
        raise ValueError('not_a_verified_model_failure_path')
    terminal = json.loads((output/'transport-terminal.json').read_text())
    if (terminal['pi_exit'] != 0 or not terminal['closed'] or not terminal['bridge_failed'] or
            terminal['close_verified'] or terminal['reason'] != bridge.reason):
        raise ValueError('failed_transport_terminal_identity')
    requests = [json.loads(s) for s in (output/'requests-private.jsonl').read_text().splitlines()]
    if not requests or requests[-1]['op'] != 'close':
        raise ValueError('missing_failure_close')
    messages = requests[-1]['messages']
    if not messages:
        raise ValueError('missing_pi_failure_message')
    error = messages[-1]
    if (error.get('role') != 'assistant' or error.get('content') != [] or error.get('stopReason') != 'error' or
            error.get('api') != API or error.get('model') != MODEL or error.get('provider') != PROVIDER or
            error.get('errorMessage') != 'Native compatibility transport stopped; inspect private owner receipt.'):
        raise ValueError('unexpected_pi_failure_message')
    if compact(normalized_messages(messages[:-1])) != compact(bridge.history):
        raise ValueError('failed_history_changed')
    expected = ['next','execute'] * sum(m['role']=='toolResult' for m in bridge.history) + ['next','close']
    if [r['op'] for r in requests] != expected or [r['op'] for r in terminal['requests']] != expected:
        raise ValueError('failure_path_extra_or_missing_requests')
    if any(r['peer_pid'] != terminal['pi_pid'] for r in terminal['requests']):
        raise ValueError('failure_path_peer_identity')
    return dict(model_failure_transport_verified=True, task_succeeded=False,
                native_finish_verified=False, reason=bridge.reason, automatic_retry=False)
