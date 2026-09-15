"""Explicit task boundaries over the qualified single-native-episode bridge.

A source-native finish ends a record. This coordinator does NOT reopen that
record, compact it, or claim conversational memory across tasks. An operator
must explicitly begin each fresh task. Complete histories are retained, and the
same executor can retain the workspace/shell. Ordinary follow-up input is not a
new-task authorization. No Pi lifecycle transport is implemented in this module.
"""
from copy import deepcopy
import hashlib
import time

from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_bridge import BridgeStopped, NativePiBridge, normalized_messages


def history_sha(messages):
    return hashlib.sha256(compact(normalized_messages(messages)).encode()).hexdigest()


class NativeTaskSession:
    def __init__(self, panel, encoding, generate, execute, *, max_tasks=2, session_tokens=16384, session_seconds=1200):
        if type(max_tasks) is not int or not 1 <= max_tasks <= 4:
            raise ValueError('bounded_task_count_required')
        if type(session_tokens) is not int or not 1 <= session_tokens <= 32768:
            raise ValueError('bounded_session_tokens_required')
        if type(session_seconds) is not int or not 1 <= session_seconds <= 2400:
            raise ValueError('bounded_session_seconds_required')
        self.panel = deepcopy(panel)
        self.encoding = encoding
        self.generate = generate
        self.executor = execute
        self.max_tasks = max_tasks
        self.session_tokens = session_tokens
        self.deadline = time.monotonic() + session_seconds
        self.tasks = []
        self.boundaries = []
        self.completed_history = []
        self.current = None
        self.closed = False

    def begin_task(self, task_id, prompt, *, expected_history_sha, acknowledge_fresh_record=False):
        """Explicit control-plane action, never inferred from a model/user message."""
        if self.closed or time.monotonic() >= self.deadline:
            raise BridgeStopped('session_not_active')
        if acknowledge_fresh_record is not True:
            raise BridgeStopped('explicit_fresh_record_acknowledgement_required')
        if self.current is not None:
            raise BridgeStopped('previous_task_not_settled')
        if (not isinstance(task_id, str) or not task_id or len(task_id) > 128 or
                any(row['task_id'] == task_id for row in self.boundaries)):
            raise BridgeStopped('invalid_or_duplicate_task_id')
        if not isinstance(prompt, str) or not prompt.strip():
            raise BridgeStopped('nonempty_task_prompt_required')
        if len(self.tasks) >= self.max_tasks:
            raise BridgeStopped('session_task_budget')
        if expected_history_sha != history_sha(self.completed_history):
            raise BridgeStopped('session_history_identity')
        used = sum(t.tokens for t in self.tasks)
        if used >= self.session_tokens:
            raise BridgeStopped('session_token_budget')
        panel = deepcopy(self.panel)
        panel['episode_generation_budget'] = min(panel['episode_generation_budget'], self.session_tokens-used)
        bridge = NativePiBridge(panel, prompt, self.encoding, self.generate, self.executor)
        bridge.deadline = min(bridge.deadline, self.deadline)
        boundary = dict(task_id=task_id, task_index=len(self.tasks),
                        prior_public_history_sha256=expected_history_sha,
                        fresh_native_record=True, previous_tasks_in_model_context=False,
                        executor_reused=True, prior_tasks_retained=len(self.tasks))
        self.tasks.append(bridge)
        self.boundaries.append(boundary)
        self.current = bridge
        return deepcopy(boundary)

    def _local_history(self, messages):
        normalized = normalized_messages(messages)
        n = len(self.completed_history)
        if compact(normalized[:n]) != compact(self.completed_history):
            raise BridgeStopped('completed_session_history_changed')
        return normalized[n:]

    def next(self, request):
        if self.closed or self.current is None:
            raise BridgeStopped('explicit_task_boundary_required')
        local = deepcopy(request)
        local['messages'] = self._local_history(request['messages'])
        return self.current.next(local)

    def execute(self, request):
        if self.closed or self.current is None:
            raise BridgeStopped('no_active_native_task')
        return self.current.execute(request)

    def settle_task(self, messages):
        if self.closed or self.current is None:
            raise BridgeStopped('no_active_native_task')
        local = self._local_history(messages)
        if self.current.failed:
            from scripts.e97_pi_native_failure import verify_model_failure_messages
            normalized = verify_model_failure_messages(self.current, local)
            self.current.closed = True
            self.completed_history.extend(deepcopy(normalized))
            self.boundaries[-1].update(task_succeeded=False, failure_reason=self.current.reason,
                                       native_finish_verified=False)
            result = dict(task_settled=True, task_succeeded=False, failure_reason=self.current.reason)
        else:
            closed = self.current.close(local)
            if closed['verified'] is not True:
                raise BridgeStopped('failed_task_cannot_be_continued')
            self.completed_history.extend(deepcopy(self.current.history))
            self.boundaries[-1].update(task_succeeded=True, failure_reason=None,
                                       native_finish_verified=True)
            result = dict(task_settled=True, task_succeeded=True, failure_reason=None)
        self.current = None
        return dict(**result, tasks=len(self.tasks), public_history_sha256=history_sha(self.completed_history))

    def close(self, *, expected_history_sha):
        if self.closed or self.current is not None:
            raise BridgeStopped('session_not_settled')
        if expected_history_sha != history_sha(self.completed_history):
            raise BridgeStopped('session_history_identity')
        self.closed = True
        return dict(closed=True, tasks=len(self.tasks), generated_tokens=sum(t.tokens for t in self.tasks),
                    private_native_records_retained=len(self.tasks), conversational_memory_claim=False)
