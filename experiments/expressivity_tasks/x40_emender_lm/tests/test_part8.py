"""Acceptance validation must never silently omit missing/invalid arms."""
import json
import pytest
from experiments.expressivity_tasks.x40_emender_lm import part8_report as report
from experiments.expressivity_tasks.x40_emender_lm.part8_surface import GDNModel
from experiments.expressivity_tasks.x40_emender_lm.scale_surface import SHAPES, SurfaceLM, counts
import torch


def test_missing_surface_retains_all_36_arms(tmp_path, monkeypatch):
    out=tmp_path/'receipts';out.mkdir()
    monkeypatch.setattr(report,'OUT',out);monkeypatch.setattr(report,'ROOT',tmp_path)
    report.main()
    summary=json.loads((out/'summary.json').read_text())
    assert summary['surface_arms']==36
    assert summary['successful_arms']==0
    assert len(summary['failures'])==36
    assert all(v[k] is None for v in summary['ratios'] for k in ('pure_over_transformer','pure_over_gdn2','gdn2_over_transformer'))
    assert summary['status']=='failed'


def test_short_timing_cannot_be_accepted(tmp_path,monkeypatch):
    out=tmp_path/'receipts';(out/'surface').mkdir(parents=True)
    monkeypatch.setattr(report,'OUT',out);monkeypatch.setattr(report,'ROOT',tmp_path)
    (out/'surface/22M_transformer_B64_T512.json').write_text(json.dumps(dict(status='passed',seconds=[1.]*19,warmups=3)))
    with pytest.raises(AssertionError): report.main()


@pytest.mark.parametrize('scale',SHAPES)
def test_registered_gdn2_family_matches_total_capacity(scale):
    with torch.device('meta'):
        target=counts(SurfaceLM(scale,'pure'))['total']
        model=GDNModel(scale,target)
    assert abs(counts(model)['total']/target-1)<.001
    assert all(type(b.layer.gdn2.gdn2).__name__=='GatedDeltaNet2' for b in model.blocks)
    assert all(b.layer.gdn2.gdn2.mode=='chunk' for b in model.blocks)


@pytest.mark.parametrize('context',(512,2048))
def test_checkpoint_preflight_uses_existing_strict_b32_receipt(context,monkeypatch,tmp_path):
    import subprocess
    from pathlib import Path
    from experiments.expressivity_tasks.x40_emender_lm import part8_surface as surface
    from experiments.expressivity_tasks.x40_emender_lm.part8_surface import validate_parity,FLAGS
    root=surface.ROOT
    # Historical evidence binds to the source at its receipt commit, never HEAD.
    paths=[root/f'receipts/part8/production_T{context}.json',
           root/f'receipts/part6/checkpoint_gate_B32_T{context}_packed_expandable.json']
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    for receipt in paths:
        rel=receipt.relative_to(repo)
        commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(rel)],text=True).strip()
        payload=subprocess.check_output(['git','show',f'{commit}:{rel}'])
        saved=tmp_path/receipt.relative_to(root);saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(payload)
        names=('model.py','forward.py','backward.py','kernels.py','scale_surface.py') if 'part8' in str(rel) else ('part6_checkpoint.py','part6_journal.py')
        for name in names:
            (tmp_path/name).write_bytes(subprocess.check_output(['git','show',f'{commit}:{root.relative_to(repo)/name}']))
    monkeypatch.setattr(surface,'ROOT',tmp_path)
    monkeypatch.setattr(surface,'OUT',tmp_path/'receipts/part8')
    for key in FLAGS:monkeypatch.setenv(key,'1')
    validate_parity(context,dict(checkpoint=True,packed=False,offload=False))


@pytest.mark.parametrize('batch',(64,128))
def test_packing_is_never_carried_into_a_smaller_microbatch(batch):
    from experiments.expressivity_tasks.x40_emender_lm.part8_surface import envelope_specs
    specs=envelope_specs(batch,'pure')
    for i,spec in enumerate(specs):
        if spec['packed'] and not spec['offload']:
            assert any(old['microbatch']==spec['microbatch'] and old['checkpoint'] and not old['packed'] for old in specs[:i])
    micros=[s['microbatch'] for s in specs if s['checkpoint'] and not s['packed'] and not s['offload']]
    assert micros==([64,32,16,8,4,2,1] if batch==64 else [128,64,32,16,8,4,2,1])


def test_missing_gate_repair_uses_traceback_filename():
    from experiments.expressivity_tasks.x40_emender_lm.part8_repair import missing_gate_preflight
    row=dict(status='failed',error="FileNotFoundError(2, 'No such file or directory')",traceback='FileNotFoundError: checkpoint_gate_B1_T512_packed_expandable.json')
    assert missing_gate_preflight(row)
    row['traceback']='FileNotFoundError: corpus.bin'
    assert not missing_gate_preflight(row)
    row.update(error='RuntimeError: native assertion',traceback='checkpoint_gate_ source bound')
    assert not missing_gate_preflight(row)
