"""Static block-composition and receipt-source attribution; no torch/GPU imports."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

BASE=Path('experiments/expressivity_tasks')
OUT=BASE/'x42_pile_scale/receipts/lineage_suspects'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    registration_paths=[BASE/'x42_pile_scale/velocity_registration.json',
        BASE/'x42_pile_scale/velocity_regime.json',
        BASE/'x40_emender_lm/receipts/ignition_v1/registration.json',
        BASE/'x40_emender_lm/receipts/part25_budget/source_manifest.json',
        BASE/'x34_lm_pilot/receipts/validation_sources.json',
        BASE/'x40_emender_lm/receipts/part8/production_T512_expandable.json']
    registrations={str(p):json.loads(p.read_text()) for p in registration_paths}
    models=[
        ('X40 v1 22M','x40_emender_lm/model.py','Block','memory read residual -> LayerNorm/Linear/GELU/Linear residual',
         ['x40_emender_lm/receipts/ignition_v1/long_context.json','x40_emender_lm/receipts/ignition_v1/compression_slopes.json','x40_emender_lm/receipts/ignition_v1/depth_partial.json']),
        ('Part25 budget-scaled probe','x40_emender_lm/part25_budget.py',None,'imports X40 LM; same memory + MLP blocks; budget policy changed by part25_depth',
         ['x40_emender_lm/receipts/part25_budget/preregistration.json','x40_emender_lm/receipts/part25_budget/verdict.json']),
        ('X34 tape arm','x34_lm_pilot/model.py','Block','all six blocks: attention + MLP; block index 2 adds tape residual using same pre-MLP norm2 state',
         ['x34_lm_pilot/receipts/preregistration.json','x34_lm_pilot/receipts/validation_sources.json','x34_lm_pilot/RECEIPTS.md']),
        ('Part8/Part6 pure surface','x40_emender_lm/scale_surface.py','MemoryBlock','memory residual + LayerNorm/Linear/GELU/Linear residual; SurfaceLM replaces ff Identity during construction',
         ['x40_emender_lm/receipts/part8/summary.json','x40_emender_lm/receipts/part6/summary.json','x40_emender_lm/receipts/part8/surface/1.3B_pure_B64_T2048.json','x40_emender_lm/receipts/part6/baseline/surface/1.3B_pure_B64_T2048.json']),
        ('X37 differentiated heads','x37_head_differentiation/model.py','Block','X34 attention + MLP + tape residual at block index 2; other blocks attention + MLP', ['x37_head_differentiation/cli.py']),
        ('X38 diversity','x38_diversity_objective/model.py','LM','inherits X37 composition; hook and diversity objective change training, not branches', ['x38_diversity_objective/model.py']),
        ('X39 usage/control/part9','x36_tape_perf/model.py','Block','inherits X34 attention + MLP; integrated persistent tape at block index 2', ['x39_memory_usage/receipts/part6_control_evaluation.json','x40_emender_lm/receipts/ignition_v1/depth_control.json','x40_emender_lm/depth_control.py']),
        ('X42 velocity pure 150M','x42_pile_scale/velocity_candidate.py','Block','memory residual + LayerNorm/Linear/GELU/Linear residual; Candidate replaces ff Identity with capacity-matched MLP', ['x42_pile_scale/receipts/pure_v2_architecture_audit/audit.json','x42_pile_scale/runs/velocity_production_v1/g3_c10/attempt1/result.json'])]
    rows=[]
    for name,relative,cls,composition,receipts in models:
        path=BASE/relative; text=path.read_text(); tree=ast.parse(text)
        excerpt=''
        if cls:
            node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            excerpt=ast.get_source_segment(text,node)
        bindings=[]
        for reg,contents in registrations.items():
            sources=contents.get('sources',contents)
            expected=sources.get(str(path),sources.get(relative,sources.get(path.name) if relative.startswith('x40_emender_lm/') else None))
            if isinstance(expected,str): bindings.append(dict(registration=reg,registered_sha256=expected,matches=sha(path)==expected))
        rows.append(dict(model=name,source=str(path),source_sha256=sha(path),composition=composition,
            full_class=excerpt,registered_bindings=bindings,
            dependent_receipts=[dict(path=str(BASE/r),sha256=sha(BASE/r)) for r in receipts if (BASE/r).exists()]))
    historical=json.loads((OUT/'historical_v1_block.json').read_text())
    raw=subprocess.check_output(['git','show',historical['commit']+':'+historical['path']])
    assert hashlib.sha256(raw).hexdigest()==historical['sha256']
    for row in rows:
        for binding in row['registered_bindings']:
            if not binding['matches'] and binding['registered_sha256']==historical['sha256'] and row['source']==historical['path']:
                binding['historical_resolution']='Exact registered blob at '+historical['commit']+' verified; full Block includes MLP'
    fingerprint_path=BASE/'x34_lm_pilot/runs/pilot_tape_lm/preregistration.json'
    fingerprint=json.loads(fingerprint_path.read_text())['fingerprint']
    assert fingerprint['model']==sha(BASE/'x34_lm_pilot/model.py')
    result=dict(x34_training_fingerprint=dict(path=str(fingerprint_path),sha256=sha(fingerprint_path),model_sha256=fingerprint['model'],matches=True),historical_registered_v1=historical,method='Static full-class and constructor audit plus registered hashes; no GPU touched',
        conclusion='No audited pure LM lacks its MLP. The two lineage families are pure memory+MLP versus hybrid attention+MLP+tape, not one-branch versus two-branch pure.',
        models=rows,supervisor_channel='contact_supervisor not present in ALL_TOOLS; unavailable',
        resilient_diloco='No resilient training behavior changed; requirement IDs not applicable')
    (OUT/'lineage.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# X42 lineage composition audit','',result['conclusion'],'',
        '| Model | Block composition | Dependent receipts |','|---|---|---|']
    for r in rows: lines.append('| '+r['model']+' | '+r['composition']+' | '+'; '.join(x['path'] for x in r['dependent_receipts'])+' |')
    lines += ['', 'Source SHA-256, full audited classes, and registered hash comparisons are retained in `lineage.json`. The ignition registration binds historical model SHA-256 ed3824dc9a49fac175a1a4bd3bbfa4ff59a322a06a407c0719f01e8770804259 at commit 173858ac. Its full Block also contains the MLP. The current model hash differs after later performance changes; historical attribution is supported by the exact registered blob in `historical_v1_block.json`.',
        '', '## Consequence ledger', '',
        '- Long-context PASS and compression slopes (mean 0.2539159908219952): X40 ignition-v1, six memory + MLP blocks; no attention. These are TinyStories v1 findings, not Pile proxy quality findings.',
        '- Depth bins: the pure ignition-v1 depth.py Probe uses X40 memory + MLP; depth_partial.json is failed_incomplete_report_card. The matched depth_control.py imports X36 LM and therefore uses the X39 hybrid attention + MLP + one tape lane; depth_control.json is completed. These are different compositions. Validation artifacts retain their own statuses; branch attribution does not upgrade incomplete evidence.',
        '- Part25 budget probe: imports the same MLP-equipped X40 LM, loads step-3052 pretrained weights, and changes budgets in part25_depth. It is a valid budget-transfer experiment for memory + MLP. Existing verdict currently says incomplete with failed scales 1/4/16; no live part25 process was observed. No probe process, checkpoint, source or receipt was modified here.',
        '- X34 pilot tape telemetry and occupancy: hybrid attention + MLP, with one tape lane at block 2. X37/X38 retain this hybrid lineage through X34 inheritance; X39 usage/control/part9 import X36 LM and retain it as well and cannot be labeled all-memory-stack evidence.',
        '- Proxy velocity standings: X42 Candidate memory + MLP, twelve layers, 150M trainable capacity solver. Best prior score 4.802025388578573 is a two-branch pure candidate. Transformer/GDN2 standings retain their own registrations.',
        '- Throughput surface: part8/part6 instantiate SurfaceLM, whose constructor installs the MLP in every pure block. This measures memory + MLP under its own batch/context/kernel settings. The requested ~1.6x pure speedup is not substantiated by part8 summary: at 1.3B, pure/transformer ratios are 0.6308 (T512 B64), 0.7591 (T512 B128), 0.2925 (T2048 B64), and 0.2866 (T2048 B128). The ~1.585 reciprocal of 0.6308 favors transformer. Part8 surface_status is passed but overall status is failed; part6 overall status is also failed. No performance qualification is upgraded here.',
        '', '## Suspect 4: event granularity (static only)', '',
        '`velocity_candidate.Cell` sends every token through the scan (beta initially ones, token 0 is content). `velocity_kernels.scan_kernel` computes a policy wire and thresholds its sigmoid against Enable; disabled tokens receive routing case 0. `event_density` is mean(case != 0), not read density. Reads and residual/MLP processing still occur for disabled writes. Thus density ~0.5 does not alone establish a fixed parity alternation or that half the tokens bypass memory reads. It indicates fewer store mutations: omitted writes cannot introduce that token\'s new value into the future store at that layer/head, although existing reads and the residual stream continue. Layerwise correlations, lost content and effective information capacity require trace analysis/controlled experiments. No event policy or kernel was changed.',
        '', '## Operational observations and residual risks', '',
        'The initial snapshot was stale: card 1 held a transformer lease, card 5 a GDN2 lease; no live part25 was found. The resumed pure launcher is restricted to broker card 1 and may queue until its existing holder finishes. Protected jobs were untouched. Ablations use broker cards 2 and 4; card 3 remains excluded regardless of observed idleness. Card 7 is excluded.',
        'Registered compute source files remain byte-identical. The attended ablation adapter recompiles the train function with only its physical-card admission tuple changed; same-compiler bytecode and constant comparisons are receipted in admission_validation.json. Budgets4 multiplies the rounded budgets exactly, [69,189] -> [276,756]. Read width is set to its existing maximum 256; the existing 150M solver adjusts MLP hidden width to maintain the registered capacity target. W_in is two consecutive linear maps (256 -> w_in_width -> 64); maxing w_in_width changes the factorization bottleneck, while the 64-dimensional key/query features and 128-dimensional values per head remain fixed. Its effective linear rank ceiling rises only from 58 to 64. This is a single-seed, wall-clock-budget comparison and may mix capacity benefit with throughput cost. Material improvement is prospectively defined as NLL decrease >=0.1.',
        'No resilient DiLoCo behavior changed; architecture requirement IDs are not applicable. No .pt files are part of this change.']
    (OUT/'LINEAGE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'models':len(rows),'registered_bindings':sum(len(r['registered_bindings']) for r in rows),'mismatches':[(r['model'],b['registration']) for r in rows for b in r['registered_bindings'] if not b['matches']]}))


if __name__=='__main__': main()
