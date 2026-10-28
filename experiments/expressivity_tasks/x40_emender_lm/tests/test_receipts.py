"""Evaluation/depth imports must honor the CLI run's isolated receipt path."""
import json
from experiments.expressivity_tasks.x40_emender_lm import runner,evaluate,depth

def test_stage_writers_share_run_directory(tmp_path,monkeypatch):
    monkeypatch.setenv('X40_RECEIPT_DIR',str(tmp_path))
    for stage in (runner,evaluate,depth):
        name=stage.__name__.split('.')[-1]+'.json'
        stage.write(name,{'isolated':True})
        assert json.loads((tmp_path/name).read_text())=={'isolated':True}
