"""Era-8 task mechanics; first-party patterns, never benchmark data.

Terminal tasks follow Terminal-Bench's seeded-workspace/end-state pattern.
Conversation tasks follow slot clarification, constrained dialogue and user
correction patterns. The deterministic user is sealed, not an LLM simulator.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

FAMILIES = ("terminal", "conversation")
MORPHS = ("entity_rename", "chain_depth", "distractor", "failure_inversion")
INSTRUCTIONS = {
    "terminal": """Author one newly invented Terminal-Bench-pattern terminal task, not a benchmark copy.
Use a natural seeded workspace (2-6 realistic text/code/config files, each
150-4096 bytes). Require multiple real bash/read/write operations: transform
files, repair a failing program/config, process data, set up an offline
environment or perform local git operations. Never use a token echo.
solution.actions is a 2-10 item reference plan using read/write/edit/bash,
with actual tool arguments; it must solve the stated goal through real Pi.
solution.assertions is a nonempty bounded list of end-state checks:
{"kind":"exact","path":relative,"content":text},
{"kind":"regex","path":relative,"pattern":fullmatch_regex},
{"kind":"absent","path":relative}, or
{"kind":"command","argv":[strings],"exit_code":integer,"timeout":1..10}.
Include at least one exact/regex check on an output that the plan changes.
All assertions must be true after the reference plan and at least one must
be false initially. State all requirements in the prompt, not just privately.
Do not access network, install/download packages, or use absolute paths.
""",
    "conversation": """Author one newly invented conversation-agent-pattern task.
Use a natural seeded workspace (2-6 files, each 150-4096 bytes). The agent
must ask for a missing parameter BEFORE acting, then produce constrained
responses and/or recover from a user correction. There are 2-4 user turns
INCLUDING the opening prompt. No free-form user simulator.
solution.user_script is a 1-3 item list. Each step has exactly:
{"on": {behavior_class: user_text}, "reply_pattern": fullmatch_regex,
 "allow_tools": boolean}. Behavior classes are question (reply ending ?),
json (valid JSON object/array), statement (other reply), acted (any workspace
tool since the prior user turn). Use explicit deterministic branches; any
missing branch fails closed. The first step must allow_tools=false and have
only a question branch, giving the initially missing parameter. A later turn
may correct it. reply_pattern must constrain the reply at that step.
solution.actions is the full 2-12 item reference plan using read/write/edit/bash
and finish; intermediate finish.message replies drive the script, last finish
ends the episode. solution.final_pattern constrains the last reply, not a token.
solution.assertions is the same end-state schema as terminal; include a real
exact/regex artifact reflecting the FULL exchange, not just the opening.
User-script text MUST NOT appear in public fixtures or be appended to the
opening prompt. Agent can see only the opening and successive user turns.
""",
}
JSON_HINT = ('Strict JSON: {"family": "terminal" or "conversation", '
             '"workspace":"natural", "prompt":str, "files":[{"path":str,"content":str}], '
             '"token":20-lowercase-hex-bookkeeping-only, "solution":{"actions":'
             '[{"tool":str,"arguments":object}], "assertions":[objects], '
             '"user_script":[objects] (conversation only), "final_pattern":str '
             '(conversation only)}, "notes":str}.')


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe_path(value: Any) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value)
            or Path(value).is_absolute() or ".." in Path(value).parts
            or value != Path(value).as_posix() or value == "."):
        raise ValueError("unsafe end-state path")
    return value


def pattern(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise ValueError("invalid bounded pattern")
    # Keep authored patterns simple: no backreferences, lookarounds or nested
    # repetition. This is a validator, not an arbitrary regex runtime.
    if "(" in value or "\\1" in value:
        raise ValueError("pattern groups are not supported")
    re.compile(value)
    return value


def assertions_schema(value: Any) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("end-state assertions missing or unbounded")
    file_paths = set()
    content_check = False
    for check in value:
        if not isinstance(check, dict):
            raise ValueError("invalid assertion")
        kind = check.get("kind")
        keys = {"exact": {"kind", "path", "content"},
                "regex": {"kind", "path", "pattern"},
                "absent": {"kind", "path"},
                "command": {"kind", "argv", "exit_code", "timeout"}}.get(kind)
        if keys is None or set(check) != keys:
            raise ValueError("invalid assertion fields")
        if kind == "command":
            argv = check["argv"]
            if (not isinstance(argv, list) or not 1 <= len(argv) <= 16
                    or any(not isinstance(a, str) or not a or len(a) > 1024 or "\x00" in a for a in argv)
                    or type(check["exit_code"]) is not int
                    or type(check["timeout"]) is not int or not 1 <= check["timeout"] <= 10):
                raise ValueError("invalid command check")
        else:
            path = safe_path(check["path"])
            if path in file_paths:
                raise ValueError("duplicate assertion path")
            file_paths.add(path)
            if kind == "exact":
                text = check["content"]
                if not isinstance(text, str) or not text or len(text.encode()) > 8192 or "\x00" in text:
                    raise ValueError("invalid expected content")
                if re.fullmatch(r'(?:token=[^\r\n]+|[0-9a-f]{20})\n?', text):
                    raise ValueError('end-state artifact cannot be a token echo')
                content_check = True
            if kind == "regex":
                pattern(check["pattern"])
                if 'token=' in check['pattern']:
                    raise ValueError('end-state regex cannot grade a token echo')
                content_check = True
    if not content_check:
        raise ValueError("end state needs an exact/regex artifact, not command/token only")
    return value


def script_schema(value: Any) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError("user script must have 2-4 user turns including opening")
    for step in value:
        if (not isinstance(step, dict) or set(step) != {"on", "reply_pattern", "allow_tools"}
                or type(step["allow_tools"]) is not bool or not isinstance(step["on"], dict)
                or not step["on"] or not set(step["on"]) <= {"question", "json", "statement", "acted"}
                or any(not isinstance(t, str) or not 1 <= len(t) <= 2048 for t in step["on"].values())):
            raise ValueError("invalid sealed user step")
        pattern(step["reply_pattern"])
    if value[0]["allow_tools"] or set(value[0]["on"]) != {"question"}:
        raise ValueError("script must require clarification before acting")
    return value


def behavior_class(reply: str, acted: bool) -> str:
    if acted:
        return "acted"
    if reply.strip().endswith("?"):
        return "question"
    try:
        if isinstance(json.loads(reply), (dict, list)):
            return "json"
    except ValueError:
        pass
    return "statement"


def user_turn(script: list[dict], index: int, reply: str, acted: bool) -> tuple[str, str]:
    step = script_schema(script)[index]
    kind = behavior_class(reply, acted)
    if acted and not step["allow_tools"]:
        raise ValueError("agent acted before clarification")
    if not re.fullmatch(step["reply_pattern"], reply):
        raise ValueError("conversation constrained reply mismatch")
    if kind not in step["on"]:
        raise ValueError("unscripted agent behavior")
    return kind, step["on"][kind]


def guards(spec: dict, common) -> None:
    if spec.get("workspace") != "natural":
        raise ValueError("era8 requires natural workspace")
    common(spec, set(), set(), minimum_files=2, token_in_files=False)
    from scripts.e97_offline_shell import workspace_path
    for file in spec['files']:
        workspace_path(file['path'])
    solution = spec["solution"]
    if spec['token'] in canonical(solution):
        raise ValueError('bookkeeping token cannot enter reference outcome')
    required = {"actions", "assertions"}
    if spec["family"] == "conversation":
        required |= {"user_script", "final_pattern"}
        script_schema(solution["user_script"])
        pattern(solution["final_pattern"])
    if set(solution) != required:
        raise ValueError("era8 solution fields")
    assertions_schema(solution["assertions"])
    actions = solution["actions"]
    if not isinstance(actions, list) or not 2 <= len(actions) <= 12:
        raise ValueError("reference action bounds")
    for action in actions:
        if (not isinstance(action, dict) or set(action) != {"tool", "arguments"}
                or action["tool"] not in {"read", "write", "edit", "bash", "finish"}
                or not isinstance(action["arguments"], dict)):
            raise ValueError("invalid reference tool action")
        if "path" in action["arguments"]:
            safe_path(action["arguments"]["path"])
        from scripts.e97_offline_shell import action_allowed, assertion_argv_allowed
        action_allowed(action['tool'], action['arguments'])
    for check in solution['assertions']:
        if check['kind'] == 'command':
            assertion_argv_allowed(check['argv'])
    if spec["family"] == "terminal" and any(a["tool"] == "finish" for a in actions):
        raise ValueError("terminal reference finish is appended by proof")
    if spec["family"] == "conversation":
        index, acted = 0, False
        for action in actions:
            if action["tool"] != "finish":
                acted = True
            elif index < len(solution["user_script"]):
                user_turn(solution["user_script"], index, action["arguments"]["message"], acted)
                index += 1
                acted = False
            else:
                if not re.fullmatch(solution["final_pattern"], action["arguments"]["message"]):
                    raise ValueError("reference final mismatch")
        if index != len(solution["user_script"]) or actions[-1]["tool"] != "finish":
            raise ValueError("reference script incomplete")


def _read_beneath(root_fd: int, path: str) -> str | None:
    parts = safe_path(path).split("/")
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        try:
            if not stat.S_ISREG(os.fstat(leaf).st_mode):
                raise ValueError("end-state file is not regular")
            text = os.read(leaf, 8193)
            if len(text) > 8192:
                raise ValueError("end-state file exceeds bound")
            return text.decode("utf-8")
        finally:
            os.close(leaf)
    except FileNotFoundError:
        return None
    finally:
        os.close(fd)


def capture_end_state(workspace: Path, checks: list[dict], identity: str) -> dict:
    """Owner-produced receipt; never inferred from model read-backs/claims."""
    assertions_schema(checks)
    from scripts.e97_offline_shell import assertion_argv_allowed
    for check in checks:
        if check["kind"] == "command":
            assertion_argv_allowed(check["argv"])
    fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        files, commands = {}, []
        for check in checks:
            if check["kind"] != "command":
                files[check["path"]] = _read_beneath(fd, check["path"])
            else:
                cp = subprocess.run(check["argv"], cwd=f"/proc/self/fd/{fd}", pass_fds=(fd,),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=check["timeout"], check=False,
                                    env={**os.environ, 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_TEMPLATE_DIR': ''})
                commands.append({"argv": check["argv"], "exit_code": cp.returncode})
        return {"kind": "workspace-end-state-v1", "task_identity": identity,
                "files": files, "commands": commands}
    finally:
        os.close(fd)


def check_end_state(checks: list[dict], receipt: dict, identity: str) -> bool:
    assertions_schema(checks)
    paths = {c["path"] for c in checks if c["kind"] != "command"}
    commands = [c for c in checks if c["kind"] == "command"]
    if (not isinstance(receipt, dict) or set(receipt) != {"kind", "task_identity", "files", "commands"}
            or receipt["kind"] != "workspace-end-state-v1" or receipt["task_identity"] != identity
            or not isinstance(receipt["files"], dict) or set(receipt["files"]) != paths
            or not isinstance(receipt["commands"], list) or len(receipt["commands"]) != len(commands)):
        return False
    for check in checks:
        kind = check["kind"]
        if kind == "command":
            actual = receipt["commands"][commands.index(check)]
            if actual != {"argv": check["argv"], "exit_code": check["exit_code"]}:
                return False
        else:
            actual = receipt["files"][check["path"]]
            if kind == "absent" and actual is not None:
                return False
            if kind == "exact" and actual != check["content"]:
                return False
            if kind == "regex" and (not isinstance(actual, str) or len(actual) > 8192
                                     or re.fullmatch(check["pattern"], actual) is None):
                return False
    return True


def check_exchange(script: list[dict], final_pattern: str, messages: list[dict], actions: list[dict]) -> bool:
    """Replay sealed branches over the entire faithfully projected exchange."""
    script_schema(script)
    index, acted, last = 0, False, None
    finished = False
    for message in messages[2:]:  # projection starts system + opening user
        if finished or (last is not None and message.get('role') != 'user'):
            return False
        if message.get("role") == "tool":
            acted = True
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            content = message["content"]
            if not content.startswith("Final: "):
                return False
            reply = content[7:]
            if index < len(script):
                try:
                    _, last = user_turn(script, index, reply, acted)
                except ValueError:
                    return False
            else:
                last = None
                if not re.fullmatch(final_pattern, reply):
                    return False
                finished = True
        elif message.get("role") == "user":
            if last is None or message.get("content") != last:
                return False
            index += 1
            acted, last = False, None
    return index == len(script) and last is None and finished and bool(actions)

OFFLINE_DISCIPLINE = """
Mandatory offline grammar for both reference AND policy execution (no arbitrary interpreters):
- Commands: mkdir [-p] paths; rm [-f|-r|-rf] paths; cp/mv source dest; cat/sort/uniq/wc/head/tail/paste paths.
- awk exactly -F, '/^[a-z]+,[0-9]+$/{print $1 ":" $2*INTEGER}' path (also + or -); sed -n 'START,ENDp' path.
- printf '%s\\n' literal; test -f|-s|-d path; git init; git add paths; git status --porcelain; git log --oneline.
- Bounded semicolon/pipeline chains and > safe-relative-path redirects only. No shell expansion, substitutions, newline commands, globs, backgrounding, networking, absolute/traversal/.git-metadata paths or executable scripts.
- Command assertions use ONLY read-only commands from this grammar, not mkdir/rm/cp/mv/git init/add.
- Python/program execution, package installation and arbitrary environment setup are NOT authorized; create/repair text or code as artifacts, process data, and initialize local git using these permitted operations.
"""
