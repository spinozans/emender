"""Deploy-at-relaunch adapter; the F7 endpoint probe does NOT use this code.

The pilot remains byte-identical for live B. Correction generators in the staged
lane bind urllib timeouts and retry sleeps to the actual episode deadline.
Calls outside a correction generation (e.g. writing judges) retain their policy.
"""
from __future__ import annotations
from contextvars import ContextVar
import json
import os
from pathlib import Path
import time
import urllib.request


def harden_teacher_deadlines(pilot):
    if getattr(pilot, "_correction_deadlines_bound", False):
        return
    deadline_var = ContextVar("correction_deadline", default=None)
    original_call = pilot.lunaroute_call
    original_make = pilot.make_teacher_generate

    def call(model, messages, temperature, max_tokens, metrics):
        deadline = deadline_var.get()
        if deadline is None:
            return original_call(model, messages, temperature, max_tokens, metrics)
        token = json.loads(Path(os.path.expanduser("~/.pi/agent/auth.json")).read_text())["lunaroute"]["access"]
        url = os.environ.get("LUNAROUTE_ROUTING_URL", "https://gw.lunaroute.com/v1") + "/chat/completions"
        body = json.dumps({"model": model, "messages": messages,
                           "temperature": temperature, "max_tokens": max_tokens}).encode()
        last = None
        for attempt in range(4):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError("correction episode deadline exhausted") from last
            request = urllib.request.Request(url, data=body, headers={
                "Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            start = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=min(900, remaining)) as response:
                    payload = json.load(response)
                content = payload["choices"][0]["message"].get("content")
                if content is None:
                    raise ValueError("empty content")
                if time.monotonic() >= deadline:
                    raise TimeoutError("response arrived after correction deadline")
                entry = {"model": model, "latency_s": round(time.monotonic()-start, 3),
                         "attempt": attempt, "usage": payload.get("usage")}
                metrics.record(entry)
                return content.strip(), entry
            except Exception as exc:
                last = exc
                metrics.record({"model": model, "latency_s": round(time.monotonic()-start, 3),
                                "attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
                remaining = deadline-time.monotonic()
                if attempt < 3 and remaining > 0:
                    time.sleep(min(20*(attempt+1), 60, remaining))
        raise RuntimeError(f"lunaroute api exhausted for {model}: {last}")

    def make(*args, **kwargs):
        generate = original_make(*args, **kwargs)
        def bounded(prompt, budget, deadline):
            token = deadline_var.set(deadline)
            try:
                return generate(prompt, budget, deadline)
            finally:
                deadline_var.reset(token)
        bounded.state = generate.state
        return bounded
    pilot.lunaroute_call = call
    pilot.make_teacher_generate = make
    pilot._correction_deadlines_bound = True
