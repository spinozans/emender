#!/usr/bin/env python3
"""e97-rl-loop-v1: the single-GPU continuous reinforcement cycle driver.

One program runs the loop the operator directed:

  (a) SERVE    load the current policy checkpoint (starts at E3-u256
               3770621c...) through the agent-server ENGINE path
               (ndm.e97_agent_server.TorchE97AgentEngine +
               ndm.e97.load_e97_checkpoint, weight_mode='saved');
  (b) ATTEMPT claim the next seed task from the filesystem queue and run one
               real Pi-native tool episode in a fresh workspace (the
               teacher-pilot bridge + real Pi child executing the eleven
               frozen tools);
  (c) GRADE   the pilot's mechanical run_check verifier over the episode
               workspace (file/command/output checks) produces a grade
               receipt (pass/fail + per-check trace);
  (d) TEACHER-CORRECT  on failure, GLM-5.3 (lunaroute, pilot Pi-transport
               pattern) continues the SAME workspace from the policy's
               partial trajectory; the teacher continuation must ALSO pass
               the same mechanical grader (verified-only, no exceptions);
               an authored deterministic fixture teacher is available as an
               explicit fallback stub;
  (e) QUEUE    a verified-only training receipt (on-policy success or
               teacher-corrected continuation) is appended to the
               digest-linked filesystem receipt stream (NO SQLite);
  (f) TRAIN    once enough receipts exist, rl_build_pack.py converts them to
               a masked-SFT authority + boundary-aware packs and
               rl_train_step.py runs a short single-GPU masked-SFT step
               (chunk-2048 config) on ONE leased GPU;
  (g) RE-SERVE the updated checkpoint is probed (load + one policy frame);
               the loop continues with it.  An empty queue ends the run --
               the work-driven contract.

Subcommands:
  init        bind the starting checkpoint into workspace state
  collect     GPU stage: attempt/grade/correct/receipt for queued tasks
  probe       GPU stage: re-serve proof for a checkpoint (load + one frame)
  status      CPU: queue + receipt-stream state
  verify      CPU: verify the whole receipt chain (digests + re-encode)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import tiktoken

from ndm.e97_atomic import publish_bytes_no_replace
from ndm.e97_onpolicy_records import canonical_json

from rl_common import (REPO_ROOT, claim_next_task, ensure_layout,
                       finish_task, next_cycle_number, requeue_task,
                       set_current_checkpoint, sha256_file, sweep_orphans,
                       workspace_paths)
from rl_receipts import (append_receipt, build_receipt, grade_receipt,
                         grade_receipt_digest, stream_tail_digest,
                         walk_stream)

PILOT_PATH = REPO_ROOT / "scripts" / "collect_e97_teacher_generation_pilot.py"
MANIFEST_PATH = REPO_ROOT / "configs" / "pi" / "e97-active-tool-surface-v1.json"
PROVIDER_V2 = REPO_ROOT / "configs" / "pi" / "e97-pi-native-frozen-tools-v2.ts"

# Policy episode budgets (deviation from the pilot's TERMINAL_PANEL: the 4B
# on-device policy decodes far slower than an API teacher, so per-turn and
# episode budgets are tuned for a single-GPU prototype; system prompt and
# tool surface stay byte-identical to the pilot records).
POLICY_PANEL = {
    "max_turns": 12,
    "generation_budget": 1024,
    "episode_generation_budget": 12288,
    "episode_seconds": 1500,
}
# Teacher episodes keep the pilot's own panel budgets (fast API generation).
TEACHER_PANEL = {
    "max_turns": 16,
    "generation_budget": 3072,
    "episode_generation_budget": 24576,
    "episode_seconds": 480,
}
DEFAULT_TEACHER_MODEL = "glm-5.3-flash-background"


def load_pilot():
    """Import the teacher pilot module (authoring, grader, transport glue)."""
    spec = importlib.util.spec_from_file_location("e97_teacher_pilot", PILOT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_curriculum():
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts import build_e97_pi_native_curriculum

    return build_e97_pi_native_curriculum


def tool_manifest(curriculum) -> dict[str, Any]:
    import hashlib

    observed = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    if observed != curriculum.MANIFEST_SHA:
        raise ValueError("tool surface manifest identity drifted")
    manifest = json.loads(MANIFEST_PATH.read_text())
    pi_bin = Path(manifest["pi_bin"])
    if sha256_file(pi_bin) != manifest["pi_bin_sha256"]:
        raise ValueError("frozen pi runtime identity drifted")
    return manifest


def make_panel(system: str, tools: list[dict[str, Any]], budgets: Mapping[str, int]) -> dict[str, Any]:
    panel = {"system": system, "tools": tools}
    panel.update(budgets)
    return panel


# ------------------------------------------------------------------ serving
def load_policy_engine(checkpoint: Path, args_json: Path, *, device: str):
    import torch

    from ndm.e97 import load_e97_checkpoint
    from ndm.e97_agent_server import TorchE97AgentEngine

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch.cuda.set_device(0)
    loaded = load_e97_checkpoint(
        checkpoint, args_json=args_json, device=device, dtype=dtype,
        weight_mode="saved", use_triton=(device == "cuda"), mmap=True)
    engine = TorchE97AgentEngine(
        loaded, ingest_mode="segment", weight_mode="saved", device=device,
        dtype="bfloat16" if device == "cuda" else "float32",
        use_triton=(device == "cuda"))
    return engine


def make_policy_generate(engine, tools, enc):
    """Greedy on-policy next-frame generation (eval_e97_native_execution shape).

    Whole causal prompt replay per turn, strict five-line canonical frame
    validation; any deviation is an honest model stop reason, never a
    synthesized continuation.
    """
    import torch

    from ndm.e97 import advance_e97_cache_segment, generate_e97_from_cache
    from scripts.e97_pi_native_codec import validate_generated_turn

    state = {"turns": 0, "tokens": 0, "reasons": {}}

    def generate(prompt: str, budget: int, deadline: float):
        prefix = enc.encode_ordinary(prompt)
        if len(prefix) + budget > 65536:
            return None, [], "context_budget"
        ids: list[int] = []
        reason = "generation_budget"
        with torch.no_grad():
            cache = advance_e97_cache_segment(engine.loaded, prefix)
            for _ in range(budget):
                if time.monotonic() >= deadline:
                    reason = "episode_deadline"
                    break
                new, cache = generate_e97_from_cache(
                    engine.loaded, cache, max_new_tokens=1, temperature=0.0,
                    top_k=0, top_p=0.0, stop_token_ids=(218,))
                if not new:
                    reason = "empty"
                    break
                ids.extend(new)
                text = enc.decode(ids)
                if not ("Analysis: ".startswith(text) or text.startswith("Analysis: ")):
                    reason = "invalid_opening"
                    return None, ids, reason
                if text.count("\n") > 4:
                    reason = "invalid_frame"
                    return None, ids, reason
                if text.count("\n") == 4:
                    try:
                        validate_generated_turn(text, tools, enc)
                    except ValueError:
                        pass
                    else:
                        state["turns"] += 1
                        state["tokens"] += len(ids)
                        state["reasons"]["valid"] = state["reasons"].get("valid", 0) + 1
                        return text, ids, "valid"
                if new[-1] == 218:
                    reason = "separator_before_valid_turn"
                    break
        state["reasons"][reason] = state["reasons"].get(reason, 0) + 1
        return None, ids, reason

    generate.state = state
    return generate


def make_teacher_generate(mode: str, model: str, tools, enc, pilot, metrics,
                          body: Mapping[str, Any] | None = None):
    """Live GLM-5.3 through the pilot's Pi-transport frame contract, or the
    explicit authored fixture stub."""
    if mode == "live":
        return pilot.make_teacher_generate(model, tools, enc, metrics)
    if mode == "fixture":
        from rl_teacher_fixture import make_fixture_generate

        if body is None:
            raise ValueError("fixture teacher requires the task body")
        return make_fixture_generate(body, tools, enc)
    raise ValueError(f"unknown teacher mode: {mode}")


# ----------------------------------------------------------------- episodes
def retained_prefix(source_messages, tools):
    """Which policy frames a spliced correction may replay (GAP #1).

    The canonical codec forbids content after finish, so a trailing finish
    frame is dropped from the replayed prefix; a trailing tool call whose
    observation never arrived (transport failure between call and result)
    is dropped as well.  Both remain immutable attempt evidence (the
    attempt_link in the receipt carries the full attempt digests).

    Returns (retained_source_indices, dense_actions, dropped_finish,
    dropped_unresolved) where dense_actions counts replayed tool-call frames
    (think/finish excluded) for the sealed per-episode turn budgets.
    """
    from scripts.e97_pi_native_codec import semantic_turn

    assistants = [index for index, message in enumerate(source_messages)
                  if message.get("role") == "assistant"]
    retained = list(assistants)
    dropped_finish = dropped_unresolved = False
    if retained:
        last = source_messages[retained[-1]]
        name = semantic_turn(last, tools)["name"]
        if name == "finish":
            retained.pop()
            dropped_finish = True
        elif not any(item.get("role") == "toolResult"
                     for item in source_messages[retained[-1] + 1:]):
            retained.pop()
            dropped_unresolved = True
    dense = sum(1 for index in retained
                if semantic_turn(source_messages[index], tools)["name"]
                not in ("think", "finish"))
    return retained, dense, dropped_finish, dropped_unresolved


def replay_prefix(bridge, source_messages, *, policy_text):
    """Replay the retained policy prefix into the correction bridge episode.

    Byte-exact: native_turn(source_message) reproduces the policy's own
    canonical frame and accept_generated_turn re-validates it, so the
    correction transcript embeds the policy prefix verbatim and the
    teacher generates given that exact prefix as conversation context
    (the episode prompt handed to the teacher IS the spliced transcript).
    Fail closed if the replay is not a byte-exact prefix of the policy's
    own episode text.
    """
    from scripts.e97_pi_native_codec import native_turn

    retained, dense, dropped_finish, dropped_unresolved = \
        retained_prefix(source_messages, bridge.tools)
    retained_set = set(retained)
    for index, message in enumerate(source_messages):
        role = message.get("role")
        if role == "assistant":
            if index in retained_set:
                bridge.episode.accept_generated_turn(
                    native_turn(message, bridge.tools))
        elif role == "toolResult":
            bridge.episode.append_context(message)
    replayed = bridge.episode.text()
    if policy_text is not None and not policy_text.startswith(replayed):
        raise ValueError("replayed prefix is not a byte-exact prefix of "
                         "the policy episode")
    return {
        "retained_frames": len(retained),
        "dense_actions": dense,
        "dropped_trailing_finish": dropped_finish,
        "dropped_unresolved_call": dropped_unresolved,
        "replayed_text_sha256": hashlib.sha256(replayed.encode()).hexdigest(),
    }


def prepare_workspace(directory: Path, body: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    for name, text in body["workspace_files"].items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for argv in body.get("setup", []):
        import subprocess

        completed = subprocess.run(argv, cwd=directory, capture_output=True,
                                   text=True, timeout=120)
        if completed.returncode != 0:
            raise RuntimeError(f"seed setup failed: {argv}: {completed.stderr[:200]}")
    return directory


def sealed_user_script(body):
    binding = body.get("task_lake")
    if binding is None:
        return None
    pin = binding["validator"]
    if sha256_file(pin["spec_path"]) != pin["spec_sha256"]:
        raise ValueError("sealed user spec identity drift")
    return json.loads(Path(pin["spec_path"]).read_bytes()).get("user_script")


def is_end_state_task(body):
    binding = body.get("task_lake")
    if binding is None:
        return False
    pin = binding["validator"]
    if sha256_file(pin["spec_path"]) != pin["spec_sha256"]:
        raise ValueError("end-state spec identity drift")
    return "end_state_assertions" in json.loads(Path(pin["spec_path"]).read_bytes())


def run_episode(*, episode_dir: Path, workspace: Path, prompt: str, panel: dict,
                generate, enc, pi_bin: Path, manifest_path: Path,
                pilot, curriculum, seconds: int,
                prefix_messages=None, prefix_policy_text=None, user_script=None, end_state_task=False) -> dict[str, Any]:
    """Drive one real Pi tool episode; return the full owner-side record.

    With ``prefix_messages`` (the retained source messages of a failed
    policy episode) the bridge's canonical transcript is SEEDED with that
    exact prefix before the first generate call, so the episode is one
    continuous spliced transcript: policy prefix + this episode's suffix.
    """
    from scripts.e97_pi_native_bridge import BridgeStopped
    from scripts.e97_pi_native_tool_bridge import NativePiToolBridge
    from scripts.e97_pi_native_tool_transport import serve_pi_native_tools

    if not episode_dir.is_dir():
        raise RuntimeError("episode directory must exist before the episode")
    if user_script is not None:
        if prefix_messages is not None:
            raise ValueError("scripted conversations require fresh correction")
        from scripts.e97_pi_scripted_v8 import ScriptedPiBridgeV8
        bridge = ScriptedPiBridgeV8(panel, prompt, enc, generate, user_script)
    elif end_state_task:
        from scripts.e97_pi_scripted_v8 import SafeWorkspaceBridgeV8
        bridge = SafeWorkspaceBridgeV8(panel, prompt, enc, generate)
    else:
        bridge = NativePiToolBridge(panel, prompt, enc, generate)
    prefix_info = None
    if prefix_messages is not None:
        prefix_info = replay_prefix(bridge, prefix_messages,
                                   policy_text=prefix_policy_text)
    record: dict[str, Any] = {
        "schema": "emender-rl-loop-episode-v1",
        "started_unix": time.time(),
        "prefix": prefix_info,
    }
    try:
        terminal = serve_pi_native_tools(
            bridge, episode_dir / "pi", pi_bin=pi_bin,
            provider_extension=(Path(os.environ["F9_STAGE_ROOT"]) / "configs/pi/e97-pi-native-scripted-v8.ts" if user_script is not None else PROVIDER_V2), pi_extensions=curriculum.EXTENSIONS,
            cwd=workspace, seconds=seconds,
            extra_env={"E97_PI_TOOL_MANIFEST": str(manifest_path.resolve()), **({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TEMPLATE_DIR": ""} if end_state_task else {})})
    except BridgeStopped as exc:
        record.update(status="stopped", reason=bridge.reason or str(exc),
                      close_verified=False, final=None)
        _publish_episode(episode_dir, record, bridge, terminal=None)
        return record
    except Exception as exc:  # transport/runtime failure is honest evidence
        record.update(status="error", reason=f"{type(exc).__name__}: {exc}",
                      close_verified=False, final=None)
        _publish_episode(episode_dir, record,
                         bridge if "bridge" in locals() else None, terminal=None)
        return record
    finished = terminal["close_verified"] and bridge.final is not None
    record.update(status="finished" if finished else "incomplete",
                  reason=bridge.reason, close_verified=terminal["close_verified"],
                  final=bridge.final,
                  turns=len(bridge.generations),
                  tokens=sum(len(item["token_ids"]) for item in bridge.generations),
                  calls=sum(1 for m in bridge.history if m["role"] == "toolResult"),
                  tool_errors=sum(m["isError"] for m in bridge.history
                                  if m["role"] == "toolResult"))
    _publish_episode(episode_dir, record, bridge, terminal)
    return record


def _publish_episode(episode_dir: Path, record: dict[str, Any], bridge, terminal) -> None:
    """Bind the owner-side bridge evidence into the record and publish it."""
    if bridge is not None:
        record.update(native_record=bridge.episode.text(),
                      generations=bridge.generations,
                      public_history=bridge.history,
                      source_messages=bridge.episode.source_messages(),
                      failed=bridge.failed,
                      prompt_sha256=hashlib.sha256(
                          bridge.history[0]["content"][0]["text"].encode()).hexdigest())
    payload = (canonical_json(record) + "\n").encode("utf-8")
    publish_bytes_no_replace(episode_dir / "episode-private.json", payload, mode=0o400)
    if terminal is not None:
        publish_bytes_no_replace(
            episode_dir / "terminal.json",
            (canonical_json(terminal) + "\n").encode("utf-8"), mode=0o400)


def grade_episode(body: Mapping[str, Any], workspace: Path, pilot, task: Mapping[str, Any],
                  *, record: Mapping[str, Any] | None = None,
                  episode_dir: Path | None = None, stage: str = "policy",
                  system: str | None = None,
                  judge_metrics: Any = None,
                  judge_model: str | None = None) -> tuple[bool, dict[str, Any]]:
    """The mechanical grader dispatch; returns (passed, grade receipt).

    Sealed gym tasks (task-lake bundles) are graded by executing the ACTUAL
    sealed first-party validator program (sha-verified against the private
    validator spec's program_sha256) over a faithful dense-format terminal
    projection of the real Pi-native episode. Authored seed tasks keep the
    pilot run_check mechanical verifier. Both graders are fail-closed and
    both grade receipts bind the queue task identity.
    """
    if "task_lake" in body:
        if record is None or episode_dir is None:
            raise RuntimeError("gym grading requires the episode record")
        floor_passed, grade = grade_gym_episode(
            body, task=task, record=record, episode_dir=episode_dir,
            stage=stage, system=system or "", workspace=workspace)
        # era-6 writing family (operator directive 2026-10-02): the sealed
        # validator grades the MECHANICAL FLOOR; draft quality is judged by
        # GLM-5.3. A writing task passes only when BOTH legs pass. The
        # judge runs only when the floor passed (fail-closed order) and its
        # verdict is receipted evidence inside the grade.
        from rl_writing_judge import is_writing_task, judge_writing_episode
        if is_writing_task(body):
            if not floor_passed:
                grade["judge"] = {"kind": "writing-judge", "passed": False,
                                  "reason": "mechanical floor failed; judge not run"}
                return False, grade
            judge_passed, evidence = judge_writing_episode(
                body=body, workspace=workspace,
                metrics=judge_metrics, model=judge_model)
            grade["judge"] = evidence
            return bool(floor_passed and judge_passed), grade
        return floor_passed, grade
    results = []
    for check in body["verify"]:
        ok, detail = pilot.run_check(check, workspace)
        results.append((check, ok, detail))
    passed = bool(results) and all(ok for _, ok, _ in results)
    grade = grade_receipt(task["task_id"], task["task_sha256"], results,
                          passed=passed, stage=stage)
    return passed, grade


def gym_terminal_projection(record: Mapping[str, Any], system: str,
                            prompt: str) -> dict[str, Any]:
    """Project one real Pi-native episode into the dense terminal format.

    Format-only translation (no content invention): every action is a tool
    call the episode really made, every observation text is the real Pi tool
    result, and the final answer is the episode's real finish message. The
    dense-format projection exists because the sealed first-party validator
    is spec-digest-bound to the dense read-observe controller terminal,
    while this loop runs the Pi-native five-frame protocol. Think frames are
    private pseudo-actions with no dense counterpart and are omitted.

    Dense read receipts (lake-expansion-v1, 2026-09-27, supervisor-approved
    live patch): every SUCCESSFUL read action additionally carries the dense
    controller's raw_observation receipt ({"ok": true, "tool": "read",
    "path", "offset": 1, "lines": [{"line": n, "text": …}], "truncated":
    false}) with effective_observation as its exact canonical encoding. Every
    line is taken verbatim from the real Pi read observation text — no
    content invention, no consultation of the spec or expected token. The
    current sealed validator (72820d63…, the pin on every NEW admitted
    collection's private spec) derives the expected token from exactly this
    receipt, so without the enrichment no new-era collection can pass
    focused grading regardless of episode quality; the original validator
    (344a1209…, the old lake's pin) ignores the added fields, so old-lake
    grading is unchanged (proven against both eras on a real episode:
    e97-lake-expansion-v1/scripts/prove_projection_fix.py).
    """
    from scripts.e97_open_swe_native_codec import strict_json

    messages: list[dict[str, Any]] = [{"role": "system", "content": system},
                                     {"role": "user", "content": prompt}]
    actions: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    sequence = 0
    finished_final = None
    for message in record["source_messages"]:
        role = message.get("role")
        if role == "assistant":
            call = message["tool_calls"][0]["function"]
            name = call["name"]
            if name == "think":
                continue
            if name == "finish":
                finished_final = strict_json(call["arguments"])["message"]
                messages.append({"role": "assistant",
                                 "content": "Final: " + finished_final})
                continue
            pending = {"tool_name": name,
                       "arguments": strict_json(call["arguments"])}
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": f"e97-{sequence}", "type": "function",
                                "function": {"name": name,
                                             "arguments": call["arguments"]}}]})
        elif role == "user":
            if message.get("content") != prompt:
                messages.append({"role": "user", "content": message["content"]})
        elif role == "toolResult":
            if pending is None:
                continue  # the skipped think frame's private acknowledgement
            text = "".join(block["text"] for block in message["content"])
            action = {"sequence": sequence, **pending,
                     "is_error": bool(message["isError"])}
            if (action["tool_name"] == "read" and not action["is_error"]):
                # faithful dense read receipt: every line is the real Pi
                # observation text, verbatim (see docstring)
                body = text[:-1] if text.endswith("\n") else text
                raw = {"ok": True, "tool": "read",
                       "path": action["arguments"]["path"], "offset": 1,
                       "lines": [{"line": index, "text": line}
                                 for index, line in enumerate(body.split("\n"), start=1)],
                       "truncated": False}
                action["raw_observation"] = raw
                action["effective_observation"] = canonical_json(raw)
            actions.append(action)
            messages.append({"role": "tool",
                             "tool_call_id": f"e97-{sequence}",
                             "content": text})
            sequence += 1
            pending = None
    if pending is not None:
        # A trailing tool call whose observation never arrived has no dense
        # terminal counterpart; drop the unanswered call entry (the frame
        # itself remains immutable episode evidence).
        messages.pop()
    if record.get("final") is not None and finished_final != record["final"]:
        raise ValueError("projection final does not reproduce the episode final")
    status = "success" if (record.get("status") == "finished"
                           and record.get("final") is not None) else "failed"
    return {
        "schema": "emender-e97-dense-terminal-projection-v1",
        "status": status,
        "messages": messages,
        "actions": actions,
        "projection_note": ("faithful dense-format projection of the real "
                            "Pi-native episode; every action and observation "
                            "is real; format-only translation"),
    }


def degeneracy_screen(record: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Mechanical screen for the degenerate-mode collapse (2026-10-02).

    Operator directive (verbatim): "we shpuld reward flecible and normal
    andnjon repetitive behavior!" The sealed validator grades mechanical
    outcomes; this screen guards the channels it cannot see:
      1. repetitive analysis — with >=2 assistant turns carrying analysis
         text, ALL of them identical is the verified collapse signature
         (2026-10-02: 297/297 session-4 and 825/825 session-3 receipts
         carried the same constant boilerplate analysis on every turn).
      2. finish over an unobserved error — the last tool result before
         finish being an error is the verified m14 pathology (reads a
         file that was never created, then finishes anyway).
    Fail-closed: violations reject the grade, so no receipt forms from a
    degenerate episode; the task then falls through to the teacher-corrected
    channel, which shifts the diet toward clean teacher-guided receipts
    while the policy is degenerate.
    """
    analyses: list[str] = []
    results: list[bool] = []
    for message in record.get("source_messages", []):
        role = message.get("role")
        if role == "assistant":
            text = message.get("reasoning_content")
            if isinstance(text, str) and text.strip():
                analyses.append(text.strip())
        elif role == "toolResult":
            results.append(bool(message.get("isError")))
    violations: list[str] = []
    if len(analyses) >= 2 and len(set(analyses)) == 1:
        violations.append("repetitive analysis across turns")
    if results and results[-1]:
        violations.append("finish over unobserved error")
    return (not violations), violations


def grade_gym_episode(body: Mapping[str, Any], *, task: Mapping[str, Any],
                      record: Mapping[str, Any], episode_dir: Path,
                      stage: str, system: str, workspace: Path | None = None) -> tuple[bool, dict[str, Any]]:
    """Execute the sealed first-party validator over the projected terminal."""
    binding = body["task_lake"]
    spec_path = Path(binding["validator"]["spec_path"])
    program = Path(binding["validator"]["program_path"])
    if sha256_file(program) != binding["validator"]["program_sha256"]:
        raise SystemExit("sealed validator program identity drift")
    if sha256_file(spec_path) != binding["validator"]["spec_sha256"]:
        raise SystemExit("sealed validator spec identity drift")
    projection = gym_terminal_projection(record, system, body["prompt"])
    private_spec = json.loads(spec_path.read_bytes())
    if "end_state_assertions" in private_spec:
        from scripts.e97_diversity import capture_end_state
        if workspace is None:
            raise ValueError("end-state grading needs owner workspace")
        try:
            projection["sealed_end_state"] = capture_end_state(workspace, private_spec["end_state_assertions"], private_spec["task_identity"])
        except (ValueError, OSError, subprocess.TimeoutExpired):
            projection["sealed_end_state"] = None  # explicit failing receipt; never fallback to claims
    projection_path = Path(episode_dir) / f"gym-terminal-projection-{stage}.json"
    publish_bytes_no_replace(
        projection_path, (canonical_json(projection) + "\n").encode("utf-8"),
        mode=0o400)
    checks: list[tuple[dict[str, Any], bool, str]] = []
    passed = True
    for mode in ("focused", "regression"):
        completed = subprocess.run(
            [sys.executable, str(program), "--mode", mode,
             "--spec", str(spec_path), "--terminal", str(projection_path)],
            capture_output=True, text=True, timeout=120)
        ok = completed.returncode == 0
        passed = passed and ok
        detail = (completed.stdout.strip() or completed.stderr.strip())[:200]
        checks.append(({"type": f"sealed-validator-{mode}"}, ok, detail))
    # degeneracy screen (operator directive 2026-10-02): the reward must see
    # the reasoning channel, not just mechanical outcomes.
    screen_ok, screen_violations = degeneracy_screen(record)
    checks.append(({"type": "degeneracy-screen"}, screen_ok,
                   "; ".join(screen_violations) if screen_violations
                   else "non-repetitive analysis; no finish-over-error"))
    passed = passed and screen_ok
    grade = grade_receipt(task["task_id"], task["task_sha256"], checks,
                          passed=passed, stage=stage)
    grade["grader"] = {
        "kind": "sealed-first-party-validator",
        "program_path": str(program),
        "program_sha256": binding["validator"]["program_sha256"],
        "spec_path": str(spec_path),
        "spec_sha256": binding["validator"]["spec_sha256"],
        "spec_digest": binding["validator"]["spec_digest"],
        "terminal_projection_sha256": sha256_file(projection_path),
        "interpreter_pinned_sha256": binding["validator"]["interpreter_pinned_sha256"],
        "interpreter_executed_sha256": sha256_file(sys.executable),
        "interpreter_note": ("the sealed spec pins a hermetic interpreter that "
                             "is not present on this host; the byte-verified "
                             "sealed program executes under the project "
                             "interpreter (recorded per grade)"),
    }
    return passed, grade


def correction_prompt(task_prompt: str, policy_record: Mapping[str, Any],
                      policy_episode_text: str | None) -> str:
    """Legacy fresh-episode correction prompt (kept for the CPU dry-run)."""
    tail = ""
    native = policy_episode_text
    if native:
        tail = native[-6000:]
    reason = policy_record.get("reason") or "unknown"
    return (f"{task_prompt}\n\n"
            "A previous agent attempted this task in the current workspace but "
            f"did not complete it (stop reason: {reason}). Its partial "
            "transcript tail:\n\n"
            f"{tail}\n\n"
            "The workspace still contains whatever changes it made. Complete "
            "the task now. Verify your work and finish when the task is done.")


# ------------------------------------------------------------------ collect
def collect(args: argparse.Namespace) -> None:
    paths = workspace_paths(args.workspace)
    ensure_layout(paths)
    pilot = load_pilot()
    curriculum = load_curriculum()
    manifest = tool_manifest(curriculum)
    pi_bin = Path(manifest["pi_bin"])
    tools = manifest["model_visible_tools"]
    enc = tiktoken.get_encoding("p50k_base")

    state = json.loads((paths["state"] / "current_checkpoint.json").read_text())
    checkpoint = Path(state["checkpoint_path"])
    if sha256_file(checkpoint) != state["checkpoint_sha256"]:
        raise SystemExit("current checkpoint identity mismatch")
    cycle = args.cycle if args.cycle else next_cycle_number(paths)
    cycle_dir = paths["episodes"] / f"cycle-{cycle:04d}"
    cycle_dir.mkdir(parents=True, exist_ok=False)

    print(f"COLLECT cycle={cycle} loading checkpoint {checkpoint.name}", flush=True)
    engine = load_policy_engine(checkpoint, Path(args.args_json), device=args.device)
    policy_generate = make_policy_generate(engine, tools, enc)
    print("COLLECT policy engine ready", flush=True)

    metrics = pilot.Metrics()
    if args.teacher != "fixture":
        teacher_generate = make_teacher_generate(
            args.teacher, args.teacher_model, tools, enc, pilot, metrics)
    else:
        teacher_generate = None  # per-task authored fixture (built per claim)

    policy_panel = make_panel(curriculum.SYSTEM, tools, POLICY_PANEL)
    teacher_panel = make_panel(curriculum.SYSTEM, tools, TEACHER_PANEL)

    receipts = 0
    prev_digest = stream_tail_digest(paths["stream"])
    attempts = 0
    seen: set[str] = set()
    outcomes: list[dict[str, Any]] = []
    while receipts < args.max_receipts and attempts < args.max_tasks:
        task = claim_next_task(paths, skip=seen)
        if task is None:
            print("COLLECT queue empty; idling (work-driven contract)", flush=True)
            break
        seen.add(task["task_id"])
        attempts += 1
        body = task["body"]
        task_dir = cycle_dir / task["task_id"]
        print(f"COLLECT task={task['task_id']} template={body['template']}", flush=True)
        gym = body.get("task_lake")
        eligible = bool(body.get("receipt_eligible", True))

        # ---- (b) ATTEMPT: one on-policy episode in a fresh workspace.
        # Sealed gym tasks honor the bundle's per-episode turn limit; the
        # wall-clock/token budgets stay the Pi-native lane's own (the sealed
        # seconds/completion_tokens accounting binds the dense controller).
        attempt_panel = policy_panel
        if gym is not None:
            attempt_panel = make_panel(curriculum.SYSTEM, tools, {
                **POLICY_PANEL,
                "max_turns": min(POLICY_PANEL["max_turns"],
                                 int(gym["limits"]["turns"]))})
        attempt_dir = task_dir / "attempt"
        workspace = prepare_workspace(attempt_dir / "workspace", body)
        policy_record = run_episode(
            episode_dir=attempt_dir, workspace=workspace, prompt=body["prompt"],
            panel=attempt_panel, generate=policy_generate, enc=enc, pi_bin=pi_bin,
            manifest_path=MANIFEST_PATH, pilot=pilot, curriculum=curriculum,
            seconds=attempt_panel["episode_seconds"], user_script=sealed_user_script(body), end_state_task=is_end_state_task(body))
        # ---- (c) GRADE
        passed, grade = grade_episode(
            body, workspace, pilot, task, record=policy_record,
            episode_dir=attempt_dir, stage="policy", system=curriculum.SYSTEM)
        grade_path = attempt_dir / "grade.json"
        publish_bytes_no_replace(
            grade_path, (canonical_json(grade) + "\n").encode("utf-8"), mode=0o400)
        print(f"COLLECT policy grade passed={passed} status={policy_record['status']}",
              flush=True)

        episode_artifacts = {
            "attempt_grade_sha256": grade_receipt_digest(grade),
            "attempt_episode_sha256": sha256_file(attempt_dir / "episode-private.json"),
            "attempt_status": policy_record["status"],
            "attempt_reason": policy_record.get("reason"),
        }
        outcome: dict[str, Any] = {
            "task_id": task["task_id"], "template": body.get("template"),
            "split": body.get("split", "seed"), "receipt_eligible": eligible,
            "attempt": {"status": policy_record["status"],
                        "grade_passed": passed,
                        "reason": policy_record.get("reason")},
            "correction": None, "receipt": None, "targets": 0,
        }

        if passed and policy_record["status"] == "finished":
            # ---- (e) QUEUE: verified on-policy success receipt
            if eligible:
                receipt = _receipt_from_episode(
                    cycle=cycle, task=task, kind="on-policy-success",
                    policy_checkpoint=state, record=policy_record,
                    supervise_from=0, grade=grade, teacher=None,
                    attempt_link=None, enc=enc, prev=prev_digest)
                prev_digest = append_receipt(paths, receipt)
                receipts += 1
                outcome.update(receipt=receipt["receipt_sha256"],
                               targets=receipt["episode"]["targets"])
                print(f"COLLECT receipt=on-policy-success "
                      f"{receipt['receipt_sha256'][:16]}", flush=True)
            else:
                print("COLLECT dev-split task measured (no receipt: sealed "
                      "train/development split isolation)", flush=True)
            outcome["receipt_kind"] = "on-policy-success" if eligible else None
            finish_task(paths, task)
            outcomes.append(outcome)
            continue

        # ---- (d) TEACHER-CORRECT (GAP #1 splice): the correction bridge
        #      replays the policy's retained partial trajectory into its own
        #      canonical transcript (byte-exact prefix), the live teacher
        #      generates given that exact prefix as conversation context,
        #      and ONLY the teacher's suffix frames are supervised targets.
        correction_dir = task_dir / "correction"
        correction_dir.mkdir(parents=True, exist_ok=False)
        if args.teacher == "fixture":
            teacher_generate = make_teacher_generate(
                args.teacher, args.teacher_model, tools, enc, metrics, body=body)
        prefix_messages = policy_record.get("source_messages")
        if sealed_user_script(body) is not None:
            prefix_messages = None
            workspace = prepare_workspace(correction_dir / "workspace", body)
        policy_generations = [item for item in (policy_record.get("generations") or [])
                              if item.get("reason") == "valid"]
        teacher_panel_eff = teacher_panel
        if gym is not None:
            dense = 0
            if prefix_messages:
                _, dense, _, _ = retained_prefix(prefix_messages, tools)
            teacher_panel_eff = make_panel(curriculum.SYSTEM, tools, {
                **TEACHER_PANEL,
                "max_turns": max(1, int(gym["limits"]["turns"]) - dense)})
        teacher_record = run_episode(
            episode_dir=correction_dir, workspace=workspace, prompt=body["prompt"],
            panel=teacher_panel_eff, generate=teacher_generate, enc=enc,
            pi_bin=pi_bin, manifest_path=MANIFEST_PATH, pilot=pilot,
            curriculum=curriculum, seconds=teacher_panel_eff["episode_seconds"],
            prefix_messages=prefix_messages,
            prefix_policy_text=(None if sealed_user_script(body) is not None else policy_record.get("native_record")), user_script=sealed_user_script(body), end_state_task=is_end_state_task(body))
        passed, grade = grade_episode(
            body, workspace, pilot, task, record=teacher_record,
            episode_dir=correction_dir, stage="teacher", system=curriculum.SYSTEM)
        publish_bytes_no_replace(
            correction_dir / "grade.json",
            (canonical_json(grade) + "\n").encode("utf-8"), mode=0o400)
        print(f"COLLECT teacher grade passed={passed} status={teacher_record['status']}",
              flush=True)
        splice = teacher_record.get("prefix") or {}
        prefix_frames = int(splice.get("retained_frames", 0))
        teacher_identity = (
            {"mode": "live", "model": args.teacher_model,
             "api": "lunaroute /v1/chat/completions"}
            if args.teacher == "live" else
            {"mode": "fixture",
             "note": "authored deterministic continuation stub (not GLM-5.3)"})
        teacher_identity.update(
            continuation=("fresh-episode" if sealed_user_script(body) is not None else "same-transcript-splice"),
            prefix_frames=prefix_frames,
            dropped_trailing_finish=bool(splice.get("dropped_trailing_finish")),
            dropped_unresolved_call=bool(splice.get("dropped_unresolved_call")))
        outcome["correction"] = {
            "status": teacher_record["status"], "grade_passed": passed,
            "reason": teacher_record.get("reason"),
            "prefix_frames": prefix_frames,
        }
        if passed and teacher_record["status"] == "finished":
            if eligible:
                merged = _merged_correction_record(
                    teacher_record, policy_generations[:prefix_frames])
                receipt = _receipt_from_episode(
                    cycle=cycle, task=task, kind="teacher-corrected",
                    policy_checkpoint=state, record=merged,
                    supervise_from=prefix_frames, grade=grade,
                    teacher=teacher_identity,
                    attempt_link={**episode_artifacts,
                                   "policy_generations": len(policy_generations),
                                   "retained_prefix_frames": prefix_frames,
                                   "dropped_trailing_finish": bool(
                                       splice.get("dropped_trailing_finish")),
                                   "dropped_unresolved_call": bool(
                                       splice.get("dropped_unresolved_call"))},
                    enc=enc, prev=prev_digest)
                prev_digest = append_receipt(paths, receipt)
                receipts += 1
                outcome.update(receipt=receipt["receipt_sha256"],
                               targets=receipt["episode"]["targets"],
                               receipt_kind="teacher-corrected",
                               supervise_from=prefix_frames)
                print(f"COLLECT receipt=teacher-corrected(spliced, "
                      f"supervise_from={prefix_frames}) "
                      f"{receipt['receipt_sha256'][:16]}", flush=True)
            else:
                print("COLLECT dev-split correction measured (no receipt: sealed "
                      "train/development split isolation)", flush=True)
            finish_task(paths, task)
        else:
            if gym is not None:
                # per-cycle gym freezes refresh every sealed task each cycle;
                # retire instead of requeue so the queue stays cycle-scoped
                requeued = requeue_task(paths, task, max_attempts=1)
            else:
                requeued = requeue_task(paths, task, max_attempts=args.max_attempts)
            print(f"COLLECT task failed both stages; requeued={requeued}", flush=True)
        outcomes.append(outcome)

    if metrics.calls:
        publish_bytes_no_replace(
            cycle_dir / "teacher-metrics.json",
            (canonical_json(metrics.dump()) + "\n").encode("utf-8"), mode=0o400)
    publish_bytes_no_replace(
        cycle_dir / "collect-summary.json",
        (canonical_json({
            "schema": "emender-rl-loop-collect-summary-v1",
            "cycle": cycle, "attempts": attempts, "receipts": receipts,
            "policy_checkpoint_sha256": state["checkpoint_sha256"],
            "teacher_mode": args.teacher,
            "policy_generate_state": getattr(policy_generate, "state", {}),
            "outcomes": outcomes,
        }) + "\n").encode("utf-8"), mode=0o400)
    print(f"COLLECT_DONE cycle={cycle} attempts={attempts} receipts={receipts}",
          flush=True)


def _merged_correction_record(teacher_record: Mapping[str, Any],
                              prefix_generations: list[Mapping[str, Any]]) -> dict[str, Any]:
    """One spliced record: policy prefix generations + teacher suffix.

    The correction episode's native_record already contains the replayed
    policy prefix (byte-exact) followed by the teacher's own frames; the
    merged generation list walks that transcript in order so encode_candidate
    can mask exactly the teacher suffix (supervise_from = len(prefix)).
    """
    merged = dict(teacher_record)
    generations = [dict(item) for item in prefix_generations] + \
        [dict(item) for item in teacher_record["generations"]]
    for index, item in enumerate(generations):
        item["turn"] = index
    merged["generations"] = generations
    return merged


def _receipt_from_episode(*, cycle: int, task: Mapping[str, Any], kind: str,
                          policy_checkpoint: Mapping[str, Any], record: Mapping[str, Any],
                          supervise_from: int, grade: Mapping[str, Any],
                          teacher: Mapping[str, Any] | None,
                          attempt_link: Mapping[str, Any] | None, enc, prev) -> dict:
    from scripts.build_e97_pi_native_curriculum import encode_candidate

    ids, mask, supervised = encode_candidate(
        record["native_record"], record["generations"], supervise_from, enc)
    return build_receipt(
        cycle=cycle, task=task, kind=kind,
        policy_checkpoint={
            "checkpoint_path": policy_checkpoint["checkpoint_path"],
            "checkpoint_sha256": policy_checkpoint["checkpoint_sha256"]},
        episode_text=record["native_record"], generations=record["generations"],
        supervise_from=supervise_from, tokens=len(ids), targets=int(sum(mask)),
        grade=grade, teacher=teacher, attempt_link=attempt_link,
        prev_receipt_sha256=prev)


# ------------------------------------------------------------------- probe
def probe(args: argparse.Namespace) -> None:
    """RE-SERVE proof: load a checkpoint through the serving path and emit one
    policy frame on a fixed protocol prompt (no tool execution)."""
    paths = workspace_paths(args.workspace)
    ensure_layout(paths)
    curriculum = load_curriculum()
    manifest = tool_manifest(curriculum)
    tools = manifest["model_visible_tools"]
    enc = tiktoken.get_encoding("p50k_base")

    checkpoint = Path(args.checkpoint)
    observed = sha256_file(checkpoint)
    if args.checkpoint_sha256 and observed != args.checkpoint_sha256:
        raise SystemExit("probe checkpoint identity mismatch")
    print(f"PROBE loading {checkpoint.name}", flush=True)
    engine = load_policy_engine(checkpoint, Path(args.args_json), device=args.device)
    generate = make_policy_generate(engine, tools, enc)

    probe_record = {"schema": "emender-rl-loop-probe-v1",
                    "checkpoint": str(checkpoint), "checkpoint_sha256": observed}
    from scripts.e97_pi_native_codec import PiNativeEpisode

    episode = PiNativeEpisode(tools, enc)
    episode.append_context({"role": "user", "content": args.prompt})
    prompt = episode.prompt()
    t0 = time.monotonic()
    text, ids, reason = generate(prompt, min(POLICY_PANEL["generation_budget"], 1024),
                                time.monotonic() + 600)
    probe_record.update(latency_s=round(time.monotonic() - t0, 2),
                        generate_reason=reason, prompt_tokens=len(enc.encode_ordinary(prompt)),
                        frame_text=text, frame_tokens=len(ids))
    if text is not None:
        from scripts.e97_pi_native_codec import validate_generated_turn

        try:
            validate_generated_turn(text, tools, enc)
            probe_record["frame_valid"] = True
        except ValueError as exc:
            probe_record["frame_valid"] = False
            probe_record["frame_invalid_reason"] = str(exc)
    else:
        probe_record["frame_valid"] = False
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes_no_replace(
        output, (canonical_json(probe_record) + "\n").encode("utf-8"), mode=0o400)
    print(f"PROBE_DONE valid={probe_record['frame_valid']} reason={reason} "
          f"tokens={len(ids)}", flush=True)


# ------------------------------------------------------------------ status
def status(args: argparse.Namespace) -> None:
    paths = workspace_paths(args.workspace)
    ensure_layout(paths)
    swept = sweep_orphans(paths) if args.sweep_orphans else None
    report = {
        "schema": "emender-rl-loop-status-v1",
        "queue": {
            "pending": len(list(paths["queue_pending"].glob("*.json"))),
            "active": len(list(paths["queue_active"].glob("*.json"))),
            "done": len(list(paths["queue_done"].glob("*.json"))),
            "swept_orphans": swept,
        },
        "checkpoint_state": (json.loads((paths["state"] / "current_checkpoint.json")
                                        .read_text())
                             if (paths["state"] / "current_checkpoint.json").is_file()
                             else None),
    }
    from rl_receipts import stream_summary

    report["stream"] = stream_summary(paths)
    print(json.dumps(report, sort_keys=True, indent=1))


# -------------------------------------------------------------------- init
def init(args: argparse.Namespace) -> None:
    paths = workspace_paths(args.workspace)
    ensure_layout(paths)
    checkpoint = Path(args.checkpoint)
    observed = sha256_file(checkpoint)
    if observed != args.checkpoint_sha256:
        raise SystemExit("checkpoint sha256 mismatch")
    set_current_checkpoint(paths, checkpoint, observed, cycle=0,
                           note="operator GO: RL loop v1 start (E3-u256)")
    print("INIT", observed)


def verify(args: argparse.Namespace) -> None:
    paths = workspace_paths(args.workspace)
    enc = tiktoken.get_encoding("p50k_base")
    receipts = walk_stream(paths, enc=enc)
    print(json.dumps({
        "schema": "emender-rl-loop-verify-v1",
        "verified_receipts": len(receipts),
        "chain_head": receipts[-1]["receipt_sha256"] if receipts else None,
        "targets": sum(item["episode"]["targets"] for item in receipts),
        "kinds": sorted({item["kind"] for item in receipts}),
    }, sort_keys=True, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path,
                        default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)

    p = sub.add_parser("collect")
    p.add_argument("--args-json", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--teacher", choices=("live", "fixture"), default="live")
    p.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--max-receipts", type=int, default=2)
    p.add_argument("--max-tasks", type=int, default=4)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--cycle", type=int, default=0)

    p = sub.add_parser("probe")
    p.add_argument("--args-json", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt",
                   default="List the files in the current directory, then report "
                           "the count and finish.")
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("status")
    p.add_argument("--sweep-orphans", action="store_true")

    sub.add_parser("verify")

    args = parser.parse_args()
    signal.signal(signal.SIGTERM,
                  lambda s, f: (_ for _ in ()).throw(TimeoutError("interrupted")))
    if args.command == "init":
        init(args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "probe":
        probe(args)
    elif args.command == "status":
        status(args)
    elif args.command == "verify":
        verify(args)


if __name__ == "__main__":
    main()
