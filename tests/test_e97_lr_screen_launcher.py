import hashlib
import json
from pathlib import Path
import sys
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/launch_e97_private_analysis_lr_screen.sh'
GUARD = SCRIPT.read_text().split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]


def prepare(root):
    (root/'recipe.json').write_text('{}')
    (root/'dev-panel.json').write_text('{}')
    (root/'freeze.json').write_text(json.dumps({'files':{},'panel_tests_passed':11}))
    (root/'novelty-audit.json').write_text(json.dumps({'status':'passed','colliding_markers':0}))
    pre=root/'evals/e97-lr5e5-screen-parent-train-core-preflight-v1'
    (pre/'results').mkdir(parents=True); (pre/'identity').mkdir()
    (pre/'results/summary.json').write_text(json.dumps({'tasks':120,'passed':120}))
    (pre/'identity/weight-mode.txt').write_text('train')
    (pre/'identity/checkpoint.sha256').write_text('aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39 checkpoint')
    return pre


def execute(root,previous,monkeypatch):
    monkeypatch.setattr(sys,'argv',['guard',str(root),str(previous)])
    exec(compile(GUARD,str(SCRIPT),'exec'),{})


def test_q8_requires_exact_parent_retention(tmp_path,monkeypatch):
    pre=prepare(tmp_path)
    execute(tmp_path,0,monkeypatch)
    (pre/'results/summary.json').write_text(json.dumps({'tasks':120,'passed':119}))
    with pytest.raises(AssertionError):
        execute(tmp_path,0,monkeypatch)


def test_q8_receipt_needed_before_u64(tmp_path,monkeypatch):
    prepare(tmp_path)
    with pytest.raises(FileNotFoundError):
        execute(tmp_path,8,monkeypatch)
    terminal=tmp_path/'runs/e97-private-lr5e5-v1-q8/terminal'; terminal.mkdir(parents=True)
    (terminal/'checkpoint.reload.json').write_text(json.dumps({'mmap_load':'passed','sft_updates':8}))
    execute(tmp_path,8,monkeypatch)


def test_u128_cannot_bypass_behavior_gate(tmp_path,monkeypatch):
    prepare(tmp_path)
    terminal=tmp_path/'runs/e97-private-lr5e5-v1-u64/terminal'; terminal.mkdir(parents=True)
    (terminal/'checkpoint.reload.json').write_text(json.dumps({'mmap_load':'passed','sft_updates':64}))
    with pytest.raises(FileNotFoundError):
        execute(tmp_path,64,monkeypatch)


def test_frozen_file_mutation_rejected(tmp_path,monkeypatch):
    prepare(tmp_path)
    (tmp_path/'freeze.json').write_text(json.dumps({'files':{'recipe.json':'a'*64},'panel_tests_passed':11}))
    with pytest.raises(AssertionError):
        execute(tmp_path,0,monkeypatch)
