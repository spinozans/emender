"""Receipt-driven four-family first-pass ladder and signed stacking report."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from .c14_documents import sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'receipts/c14_transformer_stacking'


def main():
    read = lambda p: json.loads(p.read_text())
    tr = read(OUT/'table.json')
    em = read(ROOT/'receipts/c14_stacking/table.json')
    rows = []
    for family, table in [('Transformer', tr), ('Emender', em)]:
        for arm, label in [('dense','dense'), ('dense_soft','dense + soft'),
                           ('dense_c13','dense + literal walk (total ∞)'),
                           ('v2_on','array consultation'), ('v2_on_soft','array consultation + soft'),
                           ('v2_on_c13','array + literal walk (total ∞)')]:
            rows.append((family+' '+label, table, arm))
    oracle = em['array_alone/final_test/supported_target']['nll']
    def row(t, arm, stratum): return t[f'{arm}/final_test/{stratum}']
    def fmt(x): return '∞' if x == 'Infinity' else f'{x:.9f}'
    lines = ['# C14: the complete first-pass four-family table', '',
        'Frozen final-test: 214,097,022 real byte targets; 158,891,199 supported targets. NLL is nats/byte. All frozen arms are unchanged. Literal-walk rows are failed total-LM recipes, retained as measured rungs. Each soft row uses its own calibration-only C13 logistic gate, fitted on nonempty forks and applied to all nonempty support without forced walks. Empty support abstains to the head. No final-test fitting, checkpoint changes, new extraction or dropped targets.', '',
        '| Family / rung | Total | Supported | Supported − array | Zero target | Reset 1–16 | Interior 17–128 | Fork |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    records = []
    for label,t,arm in rows:
        supported=row(t,arm,'supported_target')['nll']
        values=[row(t,arm,s)['nll'] for s in ['total','supported_target','zero_target','window_reset_1_16','window_interior_17_128','fork']]
        lines.append('| '+label+' | '+' | '.join([fmt(values[0]),fmt(supported),f'{supported-oracle:+.9f}']+[fmt(v) for v in values[2:]])+' |')
        records.append(dict(label=label,total=values[0],supported=supported,distance=supported-oracle))
    lines += [f'| Array alone (conditional oracle) | ∞ | {oracle:.9f} | +0.000000000 | ∞ | ∞ | ∞ | ∞ |', '',
        'The array-alone row is the unsmoothed maximal-supported-suffix empirical distribution. “Oracle” is conditional shorthand: selecting targets with q(target)>0 is answer-conditioned. It is neither a fair total-LM baseline nor a mathematical lower bound for a trained head. Total loss is infinite on zero-target positions, including empty support. Distances above use only the identical supported-target denominator.', '',
        '![Every measured four-family rung, supported-target distance](C14_FIRST_PASS_LADDER.svg)', '',
        '## Transformer P3: all signs', '',
        '| Stratum | Targets | Consulted | Consulted + soft | Soft − consulted | Dense + soft | Consulted soft − dense soft |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    strata=['total','window_reset_1_16','window_interior_17_128','wikipedia_support','empty_support','supported_target','nonempty_zero_target','zero_target','fork','nonfork','forced','forced_zero_target']
    for s in strata:
        a=row(tr,'v2_on',s); b=row(tr,'v2_on_soft',s); c=row(tr,'dense_soft',s)
        lines.append(f"| {s} | {a['targets']:,} | {a['nll']:.9f} | {b['nll']:.9f} | {b['nll']-a['nll']:+.9f} | {c['nll']:.9f} | {b['nll']-c['nll']:+.9f} |")
    a=row(tr,'v2_on','total'); b=row(tr,'v2_on_soft','total'); c=row(tr,'dense_soft','total')
    gain=a['total_nll']-b['total_nll']
    dense_fusion_gain=row(tr,'dense','total')['nll']-c['nll']
    residual_gain=a['nll']-b['nll']
    reset=row(tr,'v2_on','window_reset_1_16')['total_nll']-row(tr,'v2_on_soft','window_reset_1_16')['total_nll']
    both=b['nll']<a['nll'] and b['nll']<c['nll']
    lines += ['',f"Soft P3 **{'PASS' if both else 'FAIL'}**: beats consulted alone **{b['nll']<a['nll']}**; beats dense+calibration **{b['nll']<c['nll']}**. Total change is **{b['nll']-a['nll']:+.9f}** ({(b['nll']/a['nll']-1)*100:+.4f}%). Reset positions contribute **{reset/gain:.3%}** of the net gain (signed ratio, not a causal attribution).", '',
        f"Fusion changes the transformer’s total by {b['nll']-a['nll']:+.9f}; the residual gain is {residual_gain:+.9f}, versus the dense transformer’s fusion gain {dense_fusion_gain:+.9f}. The residual is {residual_gain/dense_fusion_gain:.3%} of the dense-fusion gain (a descriptive ratio, not an identified causal effect).", '',
        'The Emender comparison was 1.509260010 → 1.505908651 (−0.003351359; 93.981% of net gain at reset). The small post-consultation fusion residual replicates on the better reader: transformer −0.002496858 versus Emender −0.003351359. Its reset concentration differs (58.452% versus 93.981%); the transformer also gains in the interior. Both soft stacks worsen zero-target and nonfork losses. This is consistent with consultation already capturing most of this gate recipe’s benefit, but does not prove absorption into weights.', '',
        f"Literal C13 forced walks remain **FAIL** for both transformer dense and consulted arms: total ∞, {row(tr,'v2_on_c13','total')['infinite_losses']:,} forced-zero final targets. Supported consulted+walk NLL is {row(tr,'v2_on_c13','supported_target')['nll']:.9f}; this conditional finite score cannot rescue the total. Emender exact-walk total also fails on the same 30,751,940 targets. No smoothing or target-conditioned escape.", '',
        '## Measured head verdict and its limits', '',
        'The tier-1 C13 asymmetry inverts at Pile scale. The transformer is the better array-reader at this registered budget: consulted total 1.266063829 vs 1.509260010, supported 0.717143862 vs 0.811639725. Its supported distance from the array is +0.053709860 vs +0.148205723. The larger Emender gain over dense does not reverse this absolute ranking. Transformer P1 PASS, P2 FAIL; both consulted heads worsen zero-target loss. The retained Emender claims are operator causality, unbounded-store data-efficiency and bounded-state theory, each under its own receipt scope.', '',
        'This is a first-pass single-seed, equal-active-time comparison with unequal training exposure. No uncertainty estimate, matched-target architecture effect, semantic mechanism or universal head ranking is established. Tiny BF16 replay differences from historical dense batch-16 losses remain separate: current transformer paired batch-128 total is 1.404428829 vs historical 1.404434206. The main table uses current paired/stacking replays; Emender raw/stacked rows use its prior retained stacking receipts.', '',
        'Evidence: [transformer verdict](RESULT_C14_TRANSFORMER.md), [Emender stacking](RESULT_C14_STACKING.md), transformer [gates](receipts/c14_transformer_stacking/frozen_gates.json), [freeze](receipts/c14_transformer_stacking/calibration_freeze.json), [full calibration and final table](receipts/c14_transformer_stacking/table.json), [independent replay](receipts/c14_transformer_stacking/validation.json). Calibration has 220,107,062 targets, with 101,006,646 nonempty fork fitting targets; document split, windows, checkpoint, prediction and code hashes are retained.', '',
        'No resilient DiLoCo behavior was changed or run; requirement IDs from RESILIENT_DILOCO_GAP_MATRIX.md are inapplicable. No Frontier/Slurm, WorksGood, subagents, new training, frozen-arm edits, or .pt commits. Only broker cards 2/4 are eligible; card 1, ~/sxgc and /mnt/nvme2n1 remain unchanged.']
    (ROOT/'RESULT_C14_FIRST_PASS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'first_pass.json').write_text(json.dumps(dict(rows=records,array_supported=oracle,soft_P3='PASS' if both else 'FAIL',soft_change=b['nll']-a['nll'],reset_gain_share=reset/gain,inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'table.json',ROOT/'receipts/c14_stacking/table.json']}),indent=2)+'\n')
    # Repository-native static SVG, every measured main-table rung and baseline.
    height=80+len(records)*38+110
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
         f'<rect width="1200" height="{height}" fill="white"/>',
         '<g font-family="sans-serif" font-size="14" fill="#18212b">',
         '<text x="20" y="28" font-size="20">C14 supported-target distance from the array baseline</text>',
         '<text x="20" y="52">Signed nats/byte relative to 0.663434002; same 158,891,199 targets; negative values can occur</text>']
    scale=430/max(x['distance'] for x in records)
    svg += [f'<line x1="500" x2="500" y1="70" y2="{80+len(records)*38}" stroke="#8995a0"/>']
    for i,x in enumerate(records):
        y=80+i*38
        svg += [f'<text x="20" y="{y+17}">{escape(x["label"])}</text>',f'<rect x="{500+min(0,x["distance"]*scale):.3f}" y="{y}" width="{abs(x["distance"]*scale):.3f}" height="24" fill="{"#2474a6" if i<6 else "#a45332"}"/>',f'<text x="{510+max(0,x["distance"]*scale):.3f}" y="{y+17}">{x["distance"]:+.6f}</text>']
    svg += [f'<text x="20" y="{height-68}">Array alone: +0.000000 (conditional); full total ∞. Literal walks also have infinite total loss.</text>', f'<text x="20" y="{height-38}">Single seed, fixed active time, unequal exposure. A conditional baseline is not a total-LM bound.</text>','</g></svg>']
    (ROOT/'C14_FIRST_PASS_LADDER.svg').write_text('\n'.join(svg)+'\n')


if __name__ == '__main__': main()
