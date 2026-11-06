"""Descriptive transfer report with explicit index-change limitations."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/p100_heldout'

def main():
    e=json.loads((OUT/'eval.json').read_text());assert e['status']=='complete'
    s=json.loads((OUT/'selection.json').read_text());f=json.loads((OUT/'features.json').read_text())
    lines=['# X43 P100 out-of-index evaluation — coverage versus transfer','',f"Frozen-arm evaluation completed {e['windows']} windows /{e['targets']} raw masked targets from {e['documents']} documents fully beyond byte100e9.0 training steps; no gate calibration or weight updates.",'', 'Every selected full-document SHA256 was checked against all frozen offset identities; repeated content was excluded. Selection hashes and spans are in selection.json. Independent17-query full100GB brute-force verification passed before selection; all256 branch counts and EOF were compared, not merely interval closure.','', '| arm | out-of-index NLL |','|---|---:|']
    for k,v in e['table'].items():
        if k.endswith('/total'):lines.append(f"|{k[:-6]}|{v['nll']}|")
    lines+=['','| family | paired transfer gain (off minus on, nats/byte) |','|---|---:|']
    for k,v in e['gains'].items():lines.append(f"|{k}|{v['transfer_gain']:.9f}|")
    # In-region comparisons use the same off definition, with index shift explicit.
    sources={'emender_consulted':('c14_paired_eval','v2_on','v2_off'),'transformer_consulted':('c14_transformer_eval','v2_on','v2_off'),'unbounded':('c14_b_matched','B1_on','B1_off')}
    comparisons={}
    for name,(folder,on,off) in sources.items():
        p=ROOT/'receipts'/folder/('eval_table.json' if folder=='c14_b_matched' else 'table.json')
        if not p.exists():continue
        table=json.loads(p.read_text())
        if f'{off}/final_test/total' not in table:continue
        inside=table[f'{off}/final_test/total']['nll']-table[f'{on}/final_test/total']['nll'];outside=e['gains'][name]['transfer_gain']
        comparisons[name]=dict(in_region_gain=inside,out_of_index_gain=outside,survival_fraction=outside/inside if inside else None,operational_coverage_difference=inside-outside)
    for kind,folder in [('emender','c14_stacking'),('transformer','c14_transformer_stacking')]:
        old=json.loads((ROOT/'receipts'/folder/'table.json').read_text())
        base=comparisons[kind+'_consulted']
        inside=old['v2_off/final_test/total']['nll']-old['v2_on_soft/final_test/total']['nll'] if 'v2_off/final_test/total' in old else base['in_region_gain']+old['v2_on/final_test/total']['nll']-old['v2_on_soft/final_test/total']['nll']
        outside=e['gains'][kind+'_stacked']['transfer_gain']
        comparisons[kind+'_stacked']=dict(in_region_gain=inside,out_of_index_gain=outside,survival_fraction=outside/inside if inside else None,operational_coverage_difference=inside-outside)
    (OUT/'decomposition.json').write_text(json.dumps(dict(comparisons=comparisons,causal_identification=False,reason='Frozen C14 index Wikipedia100GB; new serving index Pile100GB. Distribution, index corpus and substring duplication confound a pure causal decomposition.'),indent=2)+'\n')
    lines+=['', 'Paired in-region comparisons, where existing off tables permit, are retained in decomposition.json. The preregistered positive25–75% survival prediction is descriptive; report the measured fractions without significance claims. Array-alone contradictions retain infinite loss; soft stacked numbers correspond to1.5059/1.2636, while literal forced-walk variants remain separate. Support-length strata and zero-target counts are in eval.json.','', 'The frozen in-region features were from Wikipedia, while this experiment queries Pile-head100. Excluding source spans removes direct inclusion of these document occurrences; repeated substrings can remain indexed. Consequently the in-region minus out-of-index gain is an operational difference, not an identified pure coverage component. New documents and index composition change together. A clean causal comparison requires matched index inclusion/exclusion on the same heldout documents, beyond this experiment.','', 'Residual risks: one frozen seed per arm; source-order prefix of beyond-boundary candidates;4096 windows is much smaller than frozen full evaluation; no confidence intervals; sampled query verification is not exhaustive SA/BWT certification. Frozen weights/gates and source bindings were checked. No DiLoCo behavior changed; gap IDs inapplicable.']
    (ROOT/'RESULT_P100_OUT_OF_INDEX.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
