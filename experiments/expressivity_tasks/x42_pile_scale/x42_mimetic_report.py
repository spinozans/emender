"""Fail-closed report for the two full-budget attended experiments."""
import json
import math
from pathlib import Path
from statistics import mean
from .velocity_train import sha
from .velocity_state import atomic_json
from .velocity_score import score_metrics
from .x42_mimetic_geometry import ROOT, OUT


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    baseline=json.loads(Path(protocol['baseline_result']).read_text())
    rows={}
    for arm in ('donor','mimetic','unfrozen'):
        result=json.loads((OUT/arm/'result.json').read_text())
        assert result['status']=='completed'
        assert 2700<=result['train_seconds']<2700+max(
            json.loads(line)['step_seconds'] for line in (OUT/arm/'training.jsonl').read_text().splitlines())+.01
        assert (result['batch'],result['context'],result['eval_windows'],result['eval_context'])==(64,2048,24,2048)
        assert math.isfinite(result['score'])
        assert json.loads((OUT/f'{arm}_terminal.json').read_text())['exit_code']==0
        if arm!='donor':
            computed=score_metrics(nll_heldout=result['nll_heldout'],occ_frac=result['occ_frac'],
                ever_touched_fractions=[r['ever_touched_fraction'] for r in result['per_head']],
                r_cap_fire_rate=result['r_cap_fire_rate'],occupancy_slope=max(result['occupancy_slopes']))
            assert computed['score']==result['score']
        row={k:result[k] for k in ['score','nll_heldout','train_seconds','tokens_consumed','tokens_per_second','steps','peak_allocated_bytes']}
        row.update(result_sha256=sha(OUT/arm/'result.json'))
        if arm!='donor':
            row.update(score_improvement=baseline['score']-result['score'],nll_improvement=baseline['nll_heldout']-result['nll_heldout'],
                materially_improved=baseline['nll_heldout']-result['nll_heldout']>=.1,
                occ_frac=result['occ_frac'],gate_mean=mean(r['gate_mean'] for r in result['per_head']),
                event_density=mean(r['event_density'] for r in result['per_head']),r_cap_fire_rate=result['r_cap_fire_rate'],
                degeneracy_flags=result['degeneracy_flags'],per_head=result['per_head'])
        rows[arm]=row
    gradients=json.loads((OUT/'unfreeze_gradients.json').read_text())['parameters']
    assert all(r['present'] and r['finite'] for r in gradients)
    assert any(r['norm']>0 for r in gradients if '.key.' in r['name'])
    assert any(r['norm']>0 for r in gradients if '.features.' in r['name'])
    mapping=json.loads((OUT/'mapping.json').read_text());assert mapping['frozen_unchanged']
    assert mapping['donor_sha256']==sha(OUT/'donor_weights.pt')
    mimic=rows['mimetic']['materially_improved'];unfreeze=rows['unfrozen']['materially_improved']
    readings={
        (True,True):'Both interventions improve raw NLL materially: early optimization and frozen geometry are both plausible contributors. Their interaction is unmeasured.',
        (True,False):'Mimicry improves materially while unfreezing does not: early-training optimization is the stronger lead at this horizon; geometry learning may require more time.',
        (False,True):'Unfreezing improves materially while this transplant does not: learned geometry is the stronger lead; partial-transplant mismatch prevents ruling out early-training effects.',
        (False,False):'Neither intervention improves raw NLL materially: the results are consistent with short-horizon selectivity cost, but do not identify it causally. Lossy transplant and slower geometry training remain alternative explanations.'}
    risks=['One seed, one 2700s horizon; wall-clock comparisons consume different token counts.',
        'Mimetic arm receives an extra 2700s of donor pretraining; this is not equal end-to-end training compute.',
        'Partial transplant leaves token embeddings, vocabulary head, queries, norms and frozen semantic basis at baseline initialization; representation mismatch and lossy geometry limit any null conclusion.',
        'Unfreezing adds trainable semantic/key parameters without shrinking MLP; capacity and throughput confound a pure frozen-basis-price estimate.',
        'The requested 2x2 is an interpretation matrix, not a completed factorial experiment: joint mimicry+unfreezing was not run.',
        'No proxy result establishes scale transfer or authorizes 1.3B production training; diagnostic is never production.']
    recommendation=('At 1.3B, validate a geometry-aware donor warm start on trainable content/value/read-out/MLP paths, preserving the frozen spine, with mapped-representation calibration before a scale run.' if mimic else
        'Keep the baseline frozen-spine initialization as the certified 1.3B default pending a matched-compute scale-transfer test; this partial donor transplant is not justified as the default.')
    if unfreeze:recommendation+=' Keep learned-basis work in a separate research branch; require capacity-matched controls and new certification before considering any production change.'
    summary=dict(status='completed',baseline_score=baseline['score'],baseline_nll=baseline['nll_heldout'],arms=rows,
        observed_quadrant=dict(mimetic_material_nll_improvement=mimic,unfreeze_material_nll_improvement=unfreeze),
        verdict=readings[mimic,unfreeze],recommendation_1_3B=recommendation,residual_risks=risks,
        resilient_diloco='No resilient training behavior changed; normative requirement IDs not applicable.')
    atomic_json(OUT/'summary.json',summary)
    lines=['# X42 mimetic initialization and unfrozen-basis verdict','',summary['verdict'],'',
        'Standing reference: score %.9f; raw NLL %.9f. Positive deltas below mean improvement.'%(baseline['score'],baseline['nll_heldout']),'',
        '| Arm | Score | NLL | Score improvement | NLL improvement | Tokens | Tokens/s |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,row in rows.items():
        lines.append('| %s | %.6f | %.6f | %s | %s | %d | %.1f |'%(arm,row['score'],row['nll_heldout'],
            '—' if arm=='donor' else '%.6f'%row['score_improvement'],'—' if arm=='donor' else '%.6f'%row['nll_improvement'],row['tokens_consumed'],row['tokens_per_second']))
    lines+=['','Donor score is raw NLL. Candidate scores retain the unchanged occupancy/degeneracy penalties. Full per-head telemetry is in summary.json and each arm/result.json. The diagnostic score is for comparison only, excluded from certified promotion.','',
        '| Mimicry NLL improvement ≥0.1 | Unfreeze NLL improvement ≥0.1 | Read |','|---|---|---|']
    for (m,u),reading in readings.items():lines.append('| %s | %s | %s |'%('yes' if m else 'no','yes' if u else 'no',reading))
    lines+=['','## 1.3B initialization recommendation','',recommendation,'','## Evidence','',
        'protocol.json binds baseline policy and compute sources; donor_provenance.json identifies the fresh donor recipe and corpus hashes; mapping.json lists every copied tensor, geometric projection errors and frozen-parameter checks; unfreeze_scope.json reports added trainable capacity; unfreeze_gradients.json proves finite nonzero key and semantic gradients. Each arm records synchronized training time, final-step overshoot, prefix tokens, score and terminal exit. Donor weights stay local and are not committed.','',
        'The frozen baseline is not fully frozen effective keys: its shared content projection learns coordinates within fixed per-head semantic/key geometry. The diagnostic additionally learns that basis. Management remains frozen. No certified source was edited.','',
        '## Residual risks','']+['- '+risk for risk in risks]+['',summary['resilient_diloco']]
    (OUT/'VERDICT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='completed',scores={k:v['score'] for k,v in rows.items()},verdict=summary['verdict'])))

if __name__=='__main__':main()
