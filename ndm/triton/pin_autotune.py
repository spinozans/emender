"""Pin Triton ``@triton.autotune`` config selection — kill the init-wedge / autotune storm.

WHY THIS EXISTS
---------------
The fused kernels on BOTH race arms (the E97/emender split-edit path and the
gdn2-mlp FLA gated-delta path) lean on ``@triton.autotune`` config sweeps that
live in the FLA library (``fla.modules.layernorm`` fwd/bwd, the gated-delta
chunk kernels, the short causal conv, ...). The compiled-binary cache
(~/.triton/cache) persists across processes, but the autotune *config
SELECTION* is in-process only: Triton re-benchmarks every config of every
multi-config kernel on the FIRST call in each fresh process (``do_bench`` ->
launch+sync storm).

* On an idle box that storm costs ~1-2 min per process.
* Under box contention it has been observed to DEADLOCK: 52-minute silent
  init-wedges, GPU pinned 100%, zero training steps, the log frozen right after
  ``[fused-guard]``.
* On Frontier, ~512 GCDs each run their OWN storm simultaneously at every job
  start / chained-resume. Catastrophic. Pinned configs => every rank starts
  instantly and identically.

WHAT THIS DOES
--------------
We do NOT edit the FLA site-packages (not version-controlled, lost on
reinstall). Instead we monkeypatch ``triton.runtime.autotuner.Autotuner.run``
at the *class* level (so it covers instances FLA already constructed at import
time). When pinning is enabled (the default), a multi-config kernel's config is
chosen from a committed registry of MEASURED-BEST configs keyed by
``(kernel_name, autotune_key)`` instead of being re-benchmarked. No ``do_bench``
is ever called => no storm => instant, identical init on every rank.

The config controls *scheduling / occupancy only* (block sizes, num_warps,
num_stages), NOT the math. Pinning is therefore numerically identical to the
autotuned path for the shapes we measured (same Config object => same launched
kernel => byte-identical output). For shapes/kernels we did not measure we fall
back to a single deterministic config (``configs[0]``) — still correct, still
no storm, just possibly not the throughput optimum.

ENV KNOBS (escape hatches)
--------------------------
* ``NDM_PIN_TRITON_AUTOTUNE`` (default ``1``): master switch. Set to ``0`` to
  fully RESTORE the original autotune sweep — use this when you move to a
  DIFFERENT shape regime and want Triton to re-benchmark (then optionally
  re-capture a registry for the new shapes, see ``NDM_PIN_TRITON_RECORD``).
* ``NDM_PIN_TRITON_REGISTRY``: path to the pinned-config JSON registry
  (default: ``pinned_autotune_configs.json`` next to this file).
* ``NDM_PIN_TRITON_STRICT`` (default ``0``): when ``1``, a multi-config kernel
  whose ``(name, key)`` is NOT in the registry falls back to the REAL autotune
  sweep (instead of ``configs[0]``). Lets you optimize a newly-introduced
  kernel/shape while keeping everything measured pinned. Off by default because
  the whole point is "no storm, ever".
* ``NDM_PIN_TRITON_VERBOSE`` (default ``0``): print one line per pin/fallback
  decision (first time each ``(name, key)`` is seen).
* ``NDM_PIN_TRITON_RECORD``: path to write a freshly-MEASURED registry. When
  set, pinning is forced OFF, the real autotuner runs, and every cached winner
  is recorded and dumped (merged) to that path at process exit. This is how the
  committed registry was generated — see ``scripts/capture_pinned_autotune.py``.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading

# Only triton + stdlib. Do NOT import sibling ndm.triton kernel modules here
# (this module is imported from ndm/triton/__init__.py and must stay acyclic).
import triton
from triton.runtime import autotuner as _autotuner_mod

Autotuner = _autotuner_mod.Autotuner

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REGISTRY = os.path.join(_HERE, "pinned_autotune_configs.json")

# Sentinel attribute so install() is idempotent across the several entry points
# (train.py, eval scripts, cmaes) that may each call it.
_INSTALLED_FLAG = "_ndm_pin_autotune_installed"

_lock = threading.Lock()
_seen_decisions: set = set()  # (name, keystr) we've already logged in verbose mode


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #
def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


def _pin_enabled() -> bool:
    # Recording mode forces pinning OFF so the real autotuner produces winners.
    if os.environ.get("NDM_PIN_TRITON_RECORD"):
        return False
    return _env_truthy("NDM_PIN_TRITON_AUTOTUNE", "1")


def _strict() -> bool:
    return _env_truthy("NDM_PIN_TRITON_STRICT", "0")


def _verbose() -> bool:
    return _env_truthy("NDM_PIN_TRITON_VERBOSE", "0")


# --------------------------------------------------------------------------- #
# (de)serialization of triton.Config <-> plain dict
# --------------------------------------------------------------------------- #
def config_to_dict(cfg) -> dict:
    """Serialize a ``triton.Config`` to a JSON-able dict.

    ``pre_hook`` cannot be serialized; callers must skip configs that carry one.
    """
    return {
        "kwargs": dict(cfg.kwargs),
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
        "num_ctas": getattr(cfg, "num_ctas", 1),
        "maxnreg": getattr(cfg, "maxnreg", None),
    }


def config_from_dict(d: dict):
    """Rebuild a ``triton.Config`` from :func:`config_to_dict` output."""
    return triton.Config(
        dict(d["kwargs"]),
        num_warps=d["num_warps"],
        num_stages=d["num_stages"],
        num_ctas=d.get("num_ctas", 1),
        maxnreg=d.get("maxnreg", None),
    )


def _keystr(key) -> str:
    """Stable string form of the autotune key tuple (ints + dtype strings)."""
    return json.dumps([str(k) for k in key], separators=(",", ":"))


def _kernel_name(self) -> str:
    base = getattr(self, "base_fn", None) or getattr(self, "fn", None)
    return getattr(base, "__name__", None) or repr(base)


def _compute_key(self, args, kwargs):
    """Replicate Triton's autotune key computation (triton 3.5.x Autotuner.run)."""
    nargs = dict(zip(self.arg_names, args))
    all_args = {**nargs, **kwargs}
    _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
    key = [_args[k] for k in self.keys if k in _args]
    for _, arg in _args.items():
        if hasattr(arg, "dtype"):
            key.append(str(arg.dtype))
    return tuple(key)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def load_registry(path: str | None = None) -> dict:
    """Load the pinned-config registry: ``{kernel_name: {keystr: config_dict}}``."""
    path = path or os.environ.get("NDM_PIN_TRITON_REGISTRY") or _DEFAULT_REGISTRY
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as fh:
        raw = json.load(fh)
    # tolerate the wrapper format produced by the capture script
    return raw.get("kernels", raw)


_REGISTRY: dict | None = None
_REGISTRY_CONFIGS: dict = {}  # (name, keystr) -> triton.Config (lazily materialized)


def _registry() -> dict:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def _pinned_config_for(name: str, key):
    """Return the pinned ``triton.Config`` for (name,key), or ``None`` to defer.

    * exact (name, keystr) match -> measured-best config for this shape
    * name measured but this shape not -> first measured config for that kernel
      (correct + no storm; not necessarily optimal)
    * name not in registry -> ``None`` (caller decides: configs[0] or autotune)
    """
    reg = _registry()
    entries = reg.get(name)
    if not entries:
        return None
    ks = _keystr(key)
    cache_key = (name, ks)
    cfg = _REGISTRY_CONFIGS.get(cache_key)
    if cfg is not None:
        return cfg
    d = entries.get(ks)
    if d is None:
        # measured kernel, unmeasured shape: reuse any measured config (correct).
        d = next(iter(entries.values()))
    cfg = config_from_dict(d)
    _REGISTRY_CONFIGS[cache_key] = cfg
    return cfg


# --------------------------------------------------------------------------- #
# patched Autotuner.run (pinning) and recording wrapper
# --------------------------------------------------------------------------- #
_ORIG_RUN = None  # set on first install()


def _maybe_log(name, key, decision):
    if not _verbose():
        return
    tag = (name, _keystr(key))
    if tag in _seen_decisions:
        return
    _seen_decisions.add(tag)
    print(f"[pin-autotune] {name} key={tag[1]} -> {decision}", file=sys.stderr, flush=True)


def _pinned_run(self, *args, **kwargs):
    # Single-config kernels never benchmark; let the original handle them.
    if not _pin_enabled() or len(self.configs) <= 1:
        return _ORIG_RUN(self, *args, **kwargs)

    self.nargs = dict(zip(self.arg_names, args))
    key = _compute_key(self, args, kwargs)

    if key not in self.cache:
        name = _kernel_name(self)
        cfg = _pinned_config_for(name, key)
        if cfg is None:
            # Unmeasured kernel.
            if _strict():
                self.nargs = None
                _maybe_log(name, key, "STRICT: real autotune")
                return _ORIG_RUN(self, *args, **kwargs)
            cfg = self.configs[0]  # deterministic, no benchmark, no storm
            _maybe_log(name, key, f"fallback configs[0]={cfg}")
        else:
            _maybe_log(name, key, f"pinned {cfg}")
        self.cache[key] = cfg

    config = self.cache[key]
    self.best_config = config
    if config.pre_hook is not None:
        full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
        config.pre_hook(full_nargs)
    ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
    self.nargs = None
    return ret


# --- recording (measurement) mode ------------------------------------------ #
_RECORD: dict = {}
_RECORD_PATH = None
_record_lock = threading.Lock()


def _recording_run(self, *args, **kwargs):
    ret = _ORIG_RUN(self, *args, **kwargs)  # the REAL autotuner picks winners
    try:
        if len(self.configs) > 1 and self.cache:
            name = _kernel_name(self)
            with _record_lock:
                d = _RECORD.setdefault(name, {})
                for k, cfg in self.cache.items():
                    if cfg.pre_hook is not None:
                        continue  # un-pinnable; leave it to autotune
                    d[_keystr(k)] = config_to_dict(cfg)
    except Exception:  # recording must never break a training run
        pass
    return ret


def _dump_record():
    if not _RECORD_PATH:
        return
    with _record_lock:
        merged = {}
        if os.path.isfile(_RECORD_PATH):
            try:
                with open(_RECORD_PATH) as fh:
                    prev = json.load(fh)
                merged = prev.get("kernels", prev)
            except Exception:
                merged = {}
        for name, entries in _RECORD.items():
            merged.setdefault(name, {}).update(entries)
        out = {
            "_comment": "Pinned Triton autotune configs (kernel_name -> autotune_key -> config). "
                        "Generated by ndm.triton.pin_autotune RECORD mode.",
            "kernels": merged,
        }
        tmp = _RECORD_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        os.replace(tmp, _RECORD_PATH)
    n = sum(len(v) for v in merged.values())
    print(f"[pin-autotune] recorded {len(merged)} kernels / {n} (name,key) configs -> {_RECORD_PATH}",
          file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #
def install() -> bool:
    """Idempotently patch ``Autotuner.run``. Returns True if a patch is active.

    Honors env: RECORD mode installs the recorder; otherwise (and only if the
    master switch is on) installs the pinning wrapper. With the master switch
    off and no RECORD, leaves Triton untouched (original sweep).
    """
    global _ORIG_RUN, _RECORD_PATH
    with _lock:
        if _ORIG_RUN is None:
            _ORIG_RUN = Autotuner.run

        record_path = os.environ.get("NDM_PIN_TRITON_RECORD")
        if record_path:
            _RECORD_PATH = record_path
            if getattr(Autotuner, _INSTALLED_FLAG, None) != "record":
                Autotuner.run = _recording_run
                setattr(Autotuner, _INSTALLED_FLAG, "record")
                atexit.register(_dump_record)
                print(f"[pin-autotune] RECORD mode: real autotune + capture -> {record_path}",
                      file=sys.stderr, flush=True)
            return True

        if not _env_truthy("NDM_PIN_TRITON_AUTOTUNE", "1"):
            # Master switch off: restore original if we had patched.
            if getattr(Autotuner, _INSTALLED_FLAG, None):
                Autotuner.run = _ORIG_RUN
                setattr(Autotuner, _INSTALLED_FLAG, None)
            return False

        if getattr(Autotuner, _INSTALLED_FLAG, None) != "pin":
            Autotuner.run = _pinned_run
            setattr(Autotuner, _INSTALLED_FLAG, "pin")
            reg = _registry()
            nconf = sum(len(v) for v in reg.values())
            print(f"[pin-autotune] pinned mode: {len(reg)} kernels / {nconf} (name,key) configs "
                  f"from registry (set NDM_PIN_TRITON_AUTOTUNE=0 to restore sweep)",
                  file=sys.stderr, flush=True)
        return True


def maybe_install_from_env() -> bool:
    """Auto-install unless explicitly disabled. Safe to call many times.

    Disabled entirely (no patch, no print) only if BOTH the master switch is off
    AND no RECORD path is set — i.e. the user asked for the stock autotuner.
    """
    if not os.environ.get("NDM_PIN_TRITON_RECORD") and not _env_truthy("NDM_PIN_TRITON_AUTOTUNE", "1"):
        return False
    return install()
