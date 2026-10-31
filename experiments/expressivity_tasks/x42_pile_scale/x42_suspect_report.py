"""Summarize X42 attended ablation endpoints without upgrading pending runs."""
import json
import statistics
import subprocess
from pathlib import Path

from .velocity_train import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/lineage_suspects'


def summarize(result):
    heads=result['per_head']
    return dict(score=result['score'],nll=result['nll_heldout'],
        occupancy=result['occ_frac'],mean_gate=statistics.mean(r['gate_mean'] for r in heads),
        r_cap_fire_rate=result['r_cap_fire_rate'],train_seconds=result['train_seconds'],
        tokens=result['tokens_consumed'],tokens_per_second=result['tokens_per_second'],
        score_eligible=result['score_eligible'],per_head=heads)


def main():
    audit=json.loads((ROOT/'receipts/pure_v2_architecture_audit/audit.json').read_text())
    base=json.loads(Path(audit['best_result']).read_text())
    variants={}
    for variant in ('budgets4','read_width256'):
        endpoint=OUT/variant/'result.json'
        if endpoint.exists():
            result=json.loads(endpoint.read_text())
            amendment=OUT/f'{variant}_amendment.json'
            saved=json.loads(amendment.read_text())
            assert saved['adapter_sha256']==sha(ROOT/'x42_attended.py')
            assert saved['registered_compute_sha256']==sha(ROOT/'velocity_train.py')
            assert result['attended_amendment_sha256']==sha(amendment)
            assert result['train_seconds']>=2700 and result['score_eligible']
            variants[variant]=dict(status='completed',**summarize(result),
                score_improvement=base['score']-result['score'],
                nll_improvement=base['nll_heldout']-result['nll_heldout'],
                materially_improved_nll=base['nll_heldout']-result['nll_heldout']>=.1,
                materially_improved_score=base['score']-result['score']>=.1,
                effective_budgets=result['effective_budgets'],result_sha256=sha(endpoint))
        elif (OUT/f'{variant}_failure.json').exists():
            variants[variant]=json.loads((OUT/f'{variant}_failure.json').read_text())
        else: variants[variant]=dict(status='pending')
    pure_state=json.loads((ROOT/'runs/velocity_production_v1/generation.json').read_text())
    resumed=json.loads((OUT/'resume.json').read_text())
    parent=json.loads((ROOT/'runs/velocity_production_v1/parent.json').read_text())
    service=subprocess.check_output(['systemctl','--user','show','emender-x42-pure-resumed',
        '-p','MainPID','-p','ActiveState','-p','SubState'],text=True)
    assert resumed['generation_before_sha256']==audit['preserved_state_sha256']['generation.json']
    assert pure_state['registration_hash']==resumed['registration_sha256']
    active=json.loads((ROOT/'runs/velocity_production_v1/active.json').read_text())
    starts=[str(Path(v['output'])/'start.json') for v in active.values() if (Path(v['output'])/'start.json').exists()]
    pure=dict(status='resumed_training' if starts else 'resumed_waiting_for_card1',training_started=bool(starts),started_receipts=starts,service=service,registration_unchanged=True,paused_state_restored=True,
        parent=parent,card_allowlist=[1],resume_receipt=resumed)
    report=dict(status='completed' if all(v['status']=='completed' for v in variants.values()) else 'incomplete',
        baseline=summarize(base),variants=variants,pure_search=pure,
        material_improvement_threshold_nll=.1,material_improvement_threshold_score=.1,
        refusal_definition='R-CAP rate: an append request cannot allocate and merges at capacity; not a dropped-token count.',
        residual_risks=['One seed and one 2700-second wall-clock candidate per ablation; no significance estimate.',
            'Budget and width changes can alter throughput; width uses existing capacity solver and changes MLP hidden size.',
            'Part25 existing budget-transfer verdict is incomplete and was not modified.',
            'Event-density information-flow cost remains static analysis only.',
            'Historical v1 source differs from current source; exact registered historical Block was verified.',
            'Supervisor contact tool unavailable in this runtime.',
            'Existing baseline launchers also request card 1 and can reacquire it; pure parent is resumed but may remain queued without new training progress.'])
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# X42 suspect ablations','',
        '| Variant | Status | Score | NLL | NLL improvement | Occupancy | R-CAP rate | Mean gate |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,r in [('baseline',report['baseline']),*variants.items()]:
        if 'score' in r: lines.append(f"| {name} | {r.get('status','completed')} | {r['score']:.6f} | {r['nll']:.6f} | {r.get('nll_improvement',0):.6f} | {r['occupancy']:.6f} | {r['r_cap_fire_rate']:.6f} | {r['mean_gate']:.6f} |")
        else: lines.append(f"| {name} | {r['status']} | | | | | | |")
    lines += ['', 'Material improvement thresholds were fixed before endpoints: score decrease >=0.1 and NLL decrease >=0.1, reported separately. The NLL threshold was fixed at launch; the score rule is additionally receipted in comparison_rule.json. Per-head telemetry, timing, token counts, source hashes and state attestation are retained in summary.json and the individual result/amendment receipts.',
        '', 'Pure search restored the exact paused gen-4 state with unchanged registration. Its attended launcher requests only card 1 through the broker and queues behind existing holders. Check pure_search.status/training_started in summary.json: parent restart alone does not prove a new training update.',
        '', 'Residual risks: '+ ' '.join(report['residual_risks'])]
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],variants={k:v['status'] for k,v in variants.items()},pure_service=service)))


if __name__=='__main__': main()
