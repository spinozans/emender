#!/usr/bin/env python3
"""Regenerate each staged modeling_ndm.py from the pristine HF snapshot and apply a
CORRECTLY-INDENTED packaging patch (vendored-import fallback + transformers-robust
tying). CPU only; verifies each result with py_compile."""
import os, shutil, py_compile

STAGE_ROOT = "/home/erikg/ndm/.wg-worktrees/agent-757/hf_v03_fix_staging"
REPOS = {"emender-e88-1.3b": "poietic-pbc/emender-e88-1.3b",
         "gdn-1.3b": "poietic-pbc/gdn-1.3b",
         "m2rnn-cma-1.3b": "poietic-pbc/m2rnn-cma-1.3b"}
HDR = "# [fix-hf-v03] packaging patch: vendored-import fallback + transformers-robust tying\n"

def snap(repo):
    c = open(f"/home/erikg/.cache/huggingface/hub/models--{repo.replace('/','--')}/refs/v0.3").read().strip()
    return f"/home/erikg/.cache/huggingface/hub/models--{repo.replace('/','--')}/snapshots/{c}/modeling_ndm.py"

REPL = [
    ('        module = importlib.import_module("ndm.models.m2rnn_baseline")\n',
     '        try:\n'
     '            module = importlib.import_module("ndm.models.m2rnn_baseline")\n'
     '        except ModuleNotFoundError:\n'
     '            module = importlib.import_module("elman.models.m2rnn_baseline")\n'),
    ('    module = importlib.import_module("ndm.models.ladder_lm")\n',
     '    try:\n'
     '        module = importlib.import_module("ndm.models.ladder_lm")\n'
     '    except ModuleNotFoundError:\n'
     '        module = importlib.import_module("elman.models.ladder_lm")\n'),
    ('    _tied_weights_keys = ["model.lm_head.weight"]\n',
     '    _tied_weights_keys = ["model.lm_head.weight"]\n'
     '    all_tied_weights_keys = ["model.lm_head.weight"]\n'),
    ('    def tie_weights(self):\n',
     '    def tie_weights(self, *args, **kwargs):\n'),
]

for name, repo in REPOS.items():
    dst = os.path.join(STAGE_ROOT, name)
    if not os.path.isdir(dst):
        print(f"{name}: no staged dir, skip"); continue
    p = os.path.join(dst, "modeling_ndm.py")
    src = open(os.path.realpath(snap(repo))).read()          # pristine
    for old, new in REPL:
        assert old in src, f"{name}: pattern not found: {old[:50]!r}"
        assert src.count(old) == 1, f"{name}: pattern not unique: {old[:50]!r}"
        src = src.replace(old, new)
    open(p, "w").write(HDR + src)
    py_compile.compile(p, doraise=True)                       # verify it parses
    print(f"{name}: patched + compiles OK -> {p}")
print("ALL MODELING PATCHED")
