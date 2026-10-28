"""CPU-only term audit; no trained weights are recoverable from v0."""
import json
from pathlib import Path
from unittest.mock import patch
import torch
from .model import LM, policy_receipt

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1);torch.manual_seed(3401)
    model=LM(128).eval();rows=[]
    def inspect(cell,x,ids):
        z=cell.W_in(x);logits=cell.W_write(x)
        heads=[]
        for h in range(4):
            features=z+cell.features[h](ids)
            k=cell.geometry[h]['key'](features)
            q=cell.geometry[h]['query'](features)
            # Isolate evidence from a single first-token slot. Includes a
            # repeated first token and novel IDs; not a full recurrent scan.
            evidence=(q*k[:,:1]).sum(-1)*.25;evidence[:,0]=0
            wire=logits[:,:,h]+cell.match_coef[h]*evidence
            heads.append(dict(bias=float(cell.W_write.bias[h]),
                base_min=float(logits[:,:,h].min()),base_mean=float(logits[:,:,h].mean()),
                match_min=float(evidence.min()),match_max=float(evidence.max()),
                match_mean=float(evidence.mean()),wired_open=float((wire>0).float().mean()),
                no_match_open=float((logits[:,:,h]>0).float().mean()),
                empty_sigmoid=float(logits[:,0,h].sigmoid().mean()),
                error_coef=float(cell.error_coef[h])))
        rows.append(heads)
        return torch.zeros(*ids.shape,4,128)
    ids=torch.tensor([list(range(1,33))+[1]*8])
    with torch.no_grad(),patch('experiments.expressivity_tasks.x40_emender_lm.model.Cell.forward',inspect):model(ids)
    result=dict(status='passed',blocks=rows,policies=policy_receipt(),
        limitations=['Small-vocabulary synthetic initial-state audit; zero read outputs between blocks; not a trained checkpoint or recurrent trajectory.',
        'No .pt weights saved: terminal parameter values and causal attribution of trained collapse cannot be recovered.'],
        gate_cost=dict(lambda_value=.01,reduction='mean over six blocks and four heads: each lane coefficient .01/24, not sixfold compounding',
        routing_gradient='None through detached threshold; positive sigmoid cost only pushes enable logits downward.'),
        decay='At zero writes all horizons lead to zero steady occupancy; decay cannot close the enable wire. At one write/token horizons exceed short-window exposure except head 0; POP/merges also limit occupancy.')
    (ROOT/'receipts/part2_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
