"""CPU-only summaries; upstream experiment artifacts remain read-only."""
import json
from pathlib import Path

from ..x34_lm_pilot.data import sha
from ..x34_lm_pilot.cli import write

ROOT = Path(__file__).resolve().parent
REC = ROOT/'receipts'


def main():
    training = json.loads((REC/'part6_control_training.json').read_text())
    controls = {step: json.loads((REC/f'control_b64_{step}.json').read_text())
                for step in (611,3052)}
    rows = []
    for family, arms, step in (
            ('x37_head_differentiation',('b','c','d'),3052),
            ('x38_diversity_objective',('0','0.001','0.01','0.1'),611),
            ('x38_diversity_objective',('0.01',),3052)):
        folder = ROOT.parent/family/'receipts'
        for arm in arms:
            evaluation, run = folder/f'eval_{arm}_{step}.json', folder/f'train_{arm}.json'
            if not evaluation.exists() or not run.exists():
                rows.append(dict(family=family,arm=arm,status='missing completed endpoint'))
                continue
            e, t = json.loads(evaluation.read_text()), json.loads(run.read_text())
            fp, control_fp = t['fingerprint'], training['fingerprint']
            matches = dict(batch=fp['batch']==control_fp['batch'],
                seed=fp['seed']==control_fp['seed'],context=fp['context']==control_fp['context'],
                tokenizer=fp['tokenizer']==control_fp['tokenizer'],
                train_bin=fp['train_bin']==control_fp['train'],
                valid_bin=fp['valid_bin']==control_fp['valid'],
                tokens=e['tokens']==controls[step]['tokens'],
                run_reached_evaluation=t['tokens']>=e['tokens'])
            ratio = e['validation']['perplexity']/controls[step]['validation']['perplexity']
            rows.append(dict(family=family,arm=arm,status=t['status'],step=step,
                tokens=e['tokens'],run_tokens=t['tokens'],protocol_matches=matches,
                evaluation_sha256=sha(evaluation),training_sha256=sha(run),
                arm_perplexity=e['validation']['perplexity'],
                b64_control_perplexity=controls[step]['validation']['perplexity'],
                perplexity_ratio=ratio,
                no_more_than_2pct_worse=ratio<=1.02 if all(matches.values()) else None,
                more_than_5pct_worse=ratio>1.05 if all(matches.values()) else None))
    write(REC/'part6_family_comparability.json', dict(rows=rows,
        caveats=['Historical B64-vs-B8 quality adjudications were not batch matched.',
                 'X38 pilots require the new 611-step control, not its 3052-step endpoint.',
                 'This checks recorded batch, seed, context, tokenizer, data and tokens; '
                 'it does not re-certify archived model implementations or change upstream reports.']))


if __name__ == '__main__':
    main()
