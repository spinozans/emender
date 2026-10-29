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
def test_checkpoint_preflight_uses_existing_strict_b32_receipt(context,monkeypatch):
    from experiments.expressivity_tasks.x40_emender_lm.part8_surface import validate_parity,FLAGS
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
