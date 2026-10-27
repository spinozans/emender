"""Zero-shot depth discrimination and held-out context decay."""
import math
import torch
from torch.nn import functional as F

DEPTHS = [16,32,64,128,256,512,1152]


def bin_name(k):
    return 'ID' if k <=128 else {256:'2x',512:'4x',1152:'9x'}[k]


@torch.no_grad()
def score(model, ids, device):
    x = torch.tensor([ids], device=device)
    logits, telemetry = model(x[:, :-1])
    losses = F.cross_entropy(logits.flatten(0,1), x[:, 1:].flatten(), reduction='none')
    return dict(mean_logprob=-losses.mean().item(), total_logprob=-losses.sum().item(),
                target_tokens=len(ids)-1, telemetry=telemetry)


@torch.no_grad()
def depth_probe(model, tokenizer, device, depths=DEPTHS):
    rows = []
    eos = tokenizer.token_to_id('<|endoftext|>')
    for k in depths:
        scores = {}
        delta = max(1, round(k/8))
        for j in (k, k-delta, k+delta):
            ids = tokenizer.encode('a'*k+'b'*j).ids+[eos]
            scores[j] = score(model, ids, device)
        margins = [scores[k]['mean_logprob']-scores[j]['mean_logprob'] for j in (k-delta,k+delta)]
        rows.append(dict(k=k, bin=bin_name(k), scores=scores, margins=margins,
                         accuracy=sum(1 if m>1e-6 else .5 if abs(m)<=1e-6 else 0 for m in margins)/2))
    return rows


@torch.no_grad()
def context_probe(model, tokens, device, lengths=(512,1024,2048,4096), windows=2):
    rows=[]
    for length in lengths:
        sums=torch.zeros(length, device=device)
        telemetry=[]
        count=0
        for i in range(windows):
            ids=tokens[i*length:(i+1)*length+1]
            if len(ids) != length+1:
                break
            x=torch.tensor([ids.tolist()], device=device, dtype=torch.long)
            logits, metrics=model(x[:, :-1])
            sums+=F.cross_entropy(logits.flatten(0,1),x[:,1:].flatten(),reduction='none')
            telemetry.append(metrics)
            count+=1
        if not count:
            raise ValueError('insufficient held-out tokens')
        losses=(sums/count).tolist()
        nll=sum(losses)/length
        rows.append(dict(context=length,windows=count,nll=nll,perplexity=math.exp(nll),
                         per_position_nll=losses,telemetry=telemetry))
    return rows


@torch.no_grad()
def story_probe(model, tokenizer, path, device, stories=16, cap=4096):
    rows=[]
    story=[]
    with open(path) as src:
        for line in src:
            if '<|endoftext|>' not in line:
                story.append(line)
                continue
            text=''.join(story)+line.split('<|endoftext|>')[0]
            story=[]
            ids=tokenizer.encode(text).ids+[tokenizer.token_to_id('<|endoftext|>')]
            if len(ids)<2:
                continue
            result=score(model,ids[:cap+1],device)
            rows.append(dict(story_index=len(rows),original_tokens=len(ids),
                             truncated=len(ids)>cap+1,**result))
            if len(rows)==stories:
                break
    if not rows:
        raise ValueError('no held-out stories')
    return rows


def aggregate_depth(rows):
    bins={}
    for name in ('ID','2x','4x','9x'):
        group=[r for r in rows if r['bin']==name]
        if not group:
            continue
        valid=[r['scores'][r['k']]['mean_logprob'] for r in group]
        corrupt=[s['mean_logprob'] for r in group for j,s in r['scores'].items() if j!=r['k']]
        bins[name]=dict(depths=[r['k'] for r in group],pairs=2*len(group),
                        accuracy=sum(r['accuracy'] for r in group)/len(group),
                        valid_mean_logprob=sum(valid)/len(valid),
                        corrupted_mean_logprob=sum(corrupt)/len(corrupt))
    return bins
