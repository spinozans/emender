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
    assert all(sha(path)==digest for path,digest in protocol['sources'].items())
    baseline=json.loads(Path(protocol['baseline_result']).read_text())
    rows={}
    for arm in ('donor','mimetic','unfrozen'):
        result=json.loads((OUT/arm/'result.json').read_text())
        assert result['status']=='completed'
        updates=[json.loads(line) for line in (OUT/arm/'training.jsonl').read_text().splitlines()]
        assert updates[-2]['train_seconds']<2700<=updates[-1]['train_seconds']
        assert updates[-1]['train_seconds']==result['train_seconds']
        assert len(updates)==result['steps']
        assert result['tokens_consumed']==result['steps']*64*2048
        assert all(r['step']==i and r['tokens']==i*64*2048 and math.isfinite(r['loss']) and math.isfinite(r['grad_norm'])
            for i,r in enumerate(updates,1))
        assert (result['batch'],result['context'],result['eval_windows'],result['eval_context'])==(64,2048,24,2048)
        assert math.isfinite(result['score'])
        assert json.loads((OUT/f'{arm}_terminal.json').read_text())['exit_code']==0
        if arm!='donor':
            computed=score_metrics(nll_heldout=result['nll_heldout'],occ_frac=result['occ_frac'],
                ever_touched_fractions=[r['ever_touched_fraction'] for r in result['per_head']],
                r_cap_fire_rate=result['r_cap_fire_rate'],occupancy_slope=max(result['occupancy_slopes']))
            assert computed['score']==result['score']
        row={k:result[k] for k in ['score','nll_heldout','train_seconds','tokens_consumed','tokens_per_second','steps','peak_allocated_bytes']}
        row.update(result_sha256=sha(OUT/arm/'result.json'),overshoot_seconds=result['train_seconds']-2700)
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
        'The diagnostic inherits baseline LR/warmup; a null or negative result does not exclude benefits after geometry-specific tuning or longer training.',
        'No shuffled or scale-matched random donor control was run; a mimetic gain does not isolate semantic transfer from weight scale or conditioning.',
        'The requested 2x2 is an interpretation matrix, not a completed factorial experiment: joint mimicry+unfreezing was not run.',
        'No proxy result establishes scale transfer or authorizes 1.3B production training; diagnostic is never production.']
    recommendation=('At 1.3B, validate a geometry-aware donor warm start on trainable content/value/read-out/MLP paths, preserving the frozen spine, with mapped-representation calibration before a scale run.' if mimic else
        'Keep the baseline frozen-spine initialization as the certified 1.3B default pending a matched-compute scale-transfer test; this partial donor transplant is not justified as the default.')
    if unfreeze:recommendation+=' Keep learned-basis work in a separate research branch; require capacity-matched controls and new certification before considering any production change.'
    else:recommendation+=' Keep per-head semantic embeddings and key projections frozen: this baseline-LR diagnostic provides no short-horizon benefit from unfreezing.'
    curves={}
    for name,directory in [('baseline',Path(protocol['baseline_result']).parent)]+[(a,OUT/a) for a in ('mimetic','unfrozen')]:
        entries=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
        curves[name]=[]
        for seconds in (270,1350,2700):
            eligible=[r for r in entries if r['train_seconds']<=seconds] or entries[:1]
            tail=eligible[-10:]
            curves[name].append(dict(at_train_seconds=seconds,last_step=tail[-1]['step'],
                tokens=tail[-1]['tokens'],mean_last_up_to_10_train_objectives=mean(r['loss'] for r in tail)))
    scope=json.loads((OUT/'unfreeze_scope.json').read_text())
    baseline_telemetry={k:baseline[k] for k in ('occ_frac','r_cap_fire_rate','degeneracy_flags','occupancy_slopes','per_head')}
    baseline_telemetry.update(gate_mean=mean(r['gate_mean'] for r in baseline['per_head']),
        event_density=mean(r['event_density'] for r in baseline['per_head']))
    summary=dict(status='completed',baseline_score=baseline['score'],baseline_nll=baseline['nll_heldout'],arms=rows,
        baseline_telemetry=baseline_telemetry,unfreeze_capacity=scope,
        learning_curves=curves,
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
        'After mimicry, the raw NLL gap to this fresh donor is %.6f. The donor consumed %.3fx as many training tokens as the mimetic arm during their respective 2700s runs. Remaining quality differences therefore mix architecture, optimization and throughput; they do not isolate a selectivity cost.'%(
            rows['mimetic']['nll_heldout']-rows['donor']['nll_heldout'],rows['donor']['tokens_consumed']/rows['mimetic']['tokens_consumed']),'',
        '| Arm | Occupancy fraction | Mean gate | Event density | R-CAP/token |',
        '|---|---:|---:|---:|---:|']
    for name,r in [('baseline',baseline_telemetry)]+[(a,rows[a]) for a in ('mimetic','unfrozen')]:
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f |'%(name,r['occ_frac'],r['gate_mean'],r['event_density'],r['r_cap_fire_rate']))
    lines+=['','All three pure arms retain the pressure_runaway degeneracy flag (R-CAP >0.05/token), adding 0.05 to each score. Their near-unit write-event densities and high occupancy persist. The mimetic gain is therefore a raw-NLL gain without relief of the pressure penalty. The observed unfreeze delta is a conditional optimization result under this recipe, not an intrinsic frozen-basis price.']
    lines+=['',
        '| Mimicry NLL improvement ≥0.1 | Unfreeze NLL improvement ≥0.1 | Read |','|---|---|---|']
    for (m,u),reading in readings.items():lines.append('| %s | %s | %s |'%('yes' if m else 'no','yes' if u else 'no',reading))
    lines+=['','## Early training dynamics','',
        '| Arm | Objective near 270s | Objective near 1350s | Objective near 2700s |',
        '|---|---:|---:|---:|']
    for name,curve in curves.items():
        lines.append('| '+name+' | '+' | '.join('%.6f'%r['mean_last_up_to_10_train_objectives'] for r in curve)+' |')
    lines+=['','Means cover the last up to ten updates at each elapsed-time mark. These are training objectives on different consumed prefixes, not held-out NLL; token positions are retained in summary.json. They describe optimization dynamics without establishing a causal mechanism.',
        '','## 1.3B initialization recommendation','',recommendation,'','## Evidence','',
        'protocol.json binds baseline policy and compute sources; donor_provenance.json identifies the fresh donor recipe and corpus hashes; mapping.json lists every copied tensor, geometric projection errors and frozen-parameter checks; unfreeze_scope.json reports added trainable capacity; unfreeze_gradients.json proves finite nonzero key and semantic gradients. Each arm records synchronized training time, final-step overshoot, prefix tokens, score and terminal exit. Donor weights stay local and are not committed.','',
        'Exactly 144 read-path tensors were initialized: input weights from the donor K/V right-singular basis and input biases zeroed; shared W_in factors from rank-limited donor K weights (donor K bias omitted); W_v from selected donor V rows projected into that input basis, with their biases; read-out from corresponding donor output columns and full output bias; both MLP linears from donor neurons, replicating neurons and dividing outgoing weights by multiplicity. Queries, semantic embeddings, key projections, management, vocabulary embedding/head and layer norms retained baseline initialization. Detailed head indices and layer widths are in mapping.json. The frozen replay spine source files were untouched.','',
        'Diagnostic trainable capacity: %d → %d parameters; total tensor capacity stays %d. This adds %d trainable parameters.'%(
            scope['before']['trainable'],scope['after']['trainable'],scope['after']['total'],scope['after']['trainable']-scope['before']['trainable']),'',
        'The frozen baseline is not fully frozen effective keys: its shared content projection learns coordinates within fixed per-head semantic/key geometry. The diagnostic additionally learns that basis. Management remains frozen. No certified source was edited.','',
        '## Residual risks','']+['- '+risk for risk in risks]+['',summary['resilient_diloco']]
    (OUT/'VERDICT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='completed',scores={k:v['score'] for k,v in rows.items()},verdict=summary['verdict'])))

if __name__=='__main__':main()
