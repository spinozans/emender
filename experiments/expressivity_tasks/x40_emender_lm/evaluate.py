"""Five-length context/compression receipts from the trained in-memory model."""
import math
import json
import numpy as np
import torch
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from ..x34_lm_pilot.cli import validate
from ..x34_lm_pilot.probes import context_probe,story_probe
from .runner import ROOT,write


@torch.no_grad()
def evaluate(model):
    model.eval()
    valid=np.memmap(data.RUNS/'valid.bin',mode='r',dtype=np.uint16)
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    contexts=context_probe(model,valid,'cuda',lengths=(512,1024,2048,4096,8192))
    for row in contexts:
        row['nll_bins128']=[float(np.mean(row['per_position_nll'][i:i+128]))
            for i in range(0,row['context'],128)]
        # Counts and fractions are both retained; routing denominator is each
        # head's actual batch*time, rather than a global 24-head denominator.
        for window in row['telemetry']:
            for block in window:
                for head in block:
                    counts=head['routing_per_head'][0]
                    total=sum(counts.values())
                    head['routing_fractions']={k:v/total for k,v in counts.items()}
    slopes=[]
    for b in range(6):
        heads=[]
        for h in range(4):
            peaks=[max(w[b][h]['peak_live_per_head'][0] for w in r['telemetry']) for r in contexts]
            heads.append(None if min(peaks)<=0 else float(np.polyfit(
                np.log([r['context'] for r in contexts]),np.log(peaks),1)[0]))
        slopes.append(heads)
    stories=story_probe(model,tok,data.RUNS/'valid.txt','cuda',stories=16)
    equal=[sum(len(set(head['peak_live_per_head'][0] for head in row['telemetry'][b]))==1
        for row in stories) for b in range(6)]
    quality=validate(model,'cuda')
    references={}
    for name in ('base_lm','tape_lm'):
        path=ROOT.parent/'x34_lm_pilot/receipts'/f'probe_pilot_{name}.json'
        receipt=json.loads(path.read_text())
        references[name]=dict(path=str(path.relative_to(ROOT.parent)),sha256=data.sha(path),
            contexts=receipt.get('contexts',receipt.get('context')),
            protocol_limit='Historical X34 B8 receipts, not batch-matched X40 B64.')
    write('evaluation.json',dict(status='completed',validation=quality,reference_receipts=references,
        reference_ppl=6.8564175640,ppl_ratio=quality['perplexity']/6.8564175640,
        contexts=contexts,stories=stories,slopes_per_block_head=slopes,
        equal_four_head_peak_stories_per_block=equal,
        long_context_ppl_ratio=contexts[-1]['perplexity']/contexts[0]['perplexity'],
        comparison_limit='Existing B64 control is a hybrid; no new transformer quality baseline trained.'))
    return contexts
