"""Versioned deterministic user continuation. Ordinary finish remains closed."""
from copy import deepcopy
import json
from scripts.e97_diversity import canonical

from scripts.e97_diversity import script_schema, user_turn
from scripts.e97_pi_native_codec import PiNativeEpisode
from scripts.e97_pi_native_tool_bridge import NativePiToolBridge


class ScriptedEpisodeV8(PiNativeEpisode):
    def __init__(self, tools, encoding, script):
        super().__init__(tools, encoding)
        header = json.loads(self._header[len("Protocol:\n"):])
        header['profile'] = 'emender-e97-pi-native-scripted-user-v8'
        header['instructions'] = header['instructions'].replace('or continue after finish.', 'or invent user turns. A finish ends the current assistant response; only a sealed follow-up user turn permits another response.')
        for pseudo in header['pseudo_actions']:
            if pseudo['name'] == 'finish':
                pseudo['description'] = 'End the current assistant response; a sealed user follow-up may continue this versioned dialogue.'
        self._header = 'Protocol:\n' + canonical(header)
        self._text = self._header
        self.script = deepcopy(script_schema(script))
        self.user_index = 0

    def append_scripted_user(self, reply, acted):
        if not self.finished or self.user_index >= len(self.script):
            raise ValueError("script continuation not authorized")
        kind, text = user_turn(self.script, self.user_index, reply, acted)
        # Only this versioned episode may continue a turn-ending finish, and
        # only after the sealed deterministic branch admits the reply.
        self._finished = False
        self.append_context({"role": "user", "content": text})
        self.user_index += 1
        return kind, text


class SafeWorkspaceBridgeV8(NativePiToolBridge):
    """Reject unsafe native actions before dispatch to Pi, retaining failure frames."""
    def next(self, request):
        result = super().next(request)
        if self.pending is not None:
            from scripts.e97_offline_shell import action_allowed
            try:
                action_allowed(self.pending['name'], self.pending['arguments'])
            except ValueError:
                # Pi has not received the call; public history must stay at the
                # pre-generation prefix. The valid native failure frame remains
                # in source_messages/generations for policy-gradient evidence.
                self.history.pop()
                self.pending = None
                self.stop('invalid_native_turn')
        return result


class ScriptedPiBridgeV8(SafeWorkspaceBridgeV8):
    def __init__(self, panel, prompt, encoding, generate, script):
        super().__init__(panel, prompt, encoding, generate)
        episode = ScriptedEpisodeV8(self.tools, encoding, script)
        for message in self.episode.source_messages():
            episode.append_context(message)
        self.episode = episode
        self.turn_start = len(episode.source_messages())
        self.exchanges = []

    def next(self, request):
        result = super().next(request)
        if self.final is None or self.episode.user_index == len(self.episode.script):
            return result
        if self.pending is not None or self.reason != "finished" or self.failed:
            self.stop("script_turn_not_finished")
        acted = any(m.get("role") == "toolResult" and m.get("toolName") != "think"
                    for m in self.episode.source_messages()[self.turn_start:])
        reply = self.final
        try:
            kind, text = self.episode.append_scripted_user(reply, acted)
        except ValueError:
            self.stop("sealed_user_script_rejected")
        self.history.append({"role": "user", "content": [{"type": "text", "text": text}]})
        self.exchanges.append({"behavior_class": kind, "assistant": reply, "user": text})
        self.turn_start = len(self.episode.source_messages())
        self.final, self.reason = None, None
        result['continue_user'] = text
        return result

    def close(self, messages):
        if self.episode.user_index != len(self.episode.script):
            self.stop('script_incomplete_at_native_close')
        return super().close(messages)
