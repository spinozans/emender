"""Attended X42 transplant and nonproduction geometry diagnostic.

Registered sources are read-only. AST amendments are explicitly receipted.
"""
import argparse
import ast
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import traceback
from unittest.mock import patch
import torch
from . import velocity_train as pure
from . import velocity_baseline_train as donor_train
from .velocity_baseline_search import SearchBaseline, SearchPolicy, verify_search
from .velocity_candidate import Candidate, Cell, Policy
from .velocity_state import atomic_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'receipts/mimetic_geometry'
REG = ROOT/'velocity_registration.json'
DONOR = OUT/'donor_weights.pt'


def amended(function, module, *, donor=False):
    tree = ast.parse(inspect.getsource(function))
    changes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and ast.unparse(node.left) == "os.environ.get('GPU_LEASE_HELD')":
            node.comparators[0] = ast.Tuple(elts=[ast.Constant('2'), ast.Constant('4')], ctx=ast.Load())
            changes += 1
    assert changes == 1
    if donor:
        # Persist after full-budget training and evaluation, outside the timer.
        index = next(i for i, n in enumerate(tree.body[0].body) if isinstance(n, ast.Return))
        tree.body[0].body[index:index] = ast.parse(
            "torch.save({'state_dict': model.state_dict(), 'policy': asdict(policy)}, DONOR)").body
    ast.fix_missing_locations(tree)
    namespace = dict(module.__dict__, DONOR=DONOR)
    exec(compile(tree, str(Path(__file__)), 'exec'), namespace)
    atomic_json(OUT/('donor_ast.json' if donor else 'pure_ast.json'), dict(
        original_sha256=pure.sha(module.__file__), amended_ast=ast.dump(tree),
        changes=['physical-card allowlist -> 2,4'] + (['save completed donor after evaluation'] if donor else [])))
    return namespace[function.__name__], namespace


def tensor_sha(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def transplant(model, checkpoint):
    state = checkpoint['state_dict']
    rows = []
    allowed = set()
    before = {n:tensor_sha(p) for n,p in model.named_parameters()}
    for i, block in enumerate(model.blocks):
        prefix = f'model.blocks.{i}.'
        qkv = state[prefix+'qkv.weight']; bias = state[prefix+'qkv.bias']
        d = qkv.shape[1]
        # Capacity geometry: shared orthogonal input coordinates chosen from
        # donor K/V, then rank-limited factorization of donor K content map.
        _, _, vh = torch.linalg.svd(torch.cat((qkv[d:2*d], qkv[2*d:]),0), full_matrices=False)
        basis = vh[:256]
        block.input.weight.copy_(basis); block.input.bias.zero_()
        kmap = qkv[d: d+64] @ basis.T
        u,s,v = torch.linalg.svd(kmap, full_matrices=False)
        rank = min(block.cell.policy.w_in_width,64)
        block.cell.W_in[0].weight.zero_(); block.cell.W_in[1].weight.zero_()
        block.cell.W_in[0].weight[:rank].copy_(s[:rank,None].sqrt()*v[:rank])
        block.cell.W_in[1].weight[:,:rank].copy_(u[:,:rank]*s[:rank].sqrt())
        # Evenly spaced complete 64-d donor heads; two per 128-d memory value.
        heads = torch.linspace(0,11,2*model.policy.N).round().long()
        indices = (heads[:,None]*64+torch.arange(64)).flatten()
        value = qkv[2*d:][indices]
        block.cell.W_v.weight.copy_(value@basis.T)
        block.cell.W_v.bias.copy_(bias[2*d:][indices])
        block.read_out.weight.copy_(state[prefix+'out.weight'][:,indices])
        block.read_out.bias.copy_(state[prefix+'out.bias'])
        donor_hidden = state[prefix+'ff.0.weight'].shape[0]
        recipient_hidden = block.ff[0].out_features
        neurons = torch.arange(recipient_hidden)%donor_hidden
        multiplicity = torch.bincount(neurons,minlength=donor_hidden).clamp_min(1)
        block.ff[0].weight.copy_(state[prefix+'ff.0.weight'][neurons])
        block.ff[0].bias.copy_(state[prefix+'ff.0.bias'][neurons])
        block.ff[2].weight.copy_(state[prefix+'ff.2.weight'][:,neurons]/multiplicity[neurons])
        block.ff[2].bias.copy_(state[prefix+'ff.2.bias'])
        names = ['input.weight','input.bias','cell.W_in.0.weight','cell.W_in.1.weight',
                 'cell.W_v.weight','cell.W_v.bias','read_out.weight','read_out.bias',
                 'ff.0.weight','ff.0.bias','ff.2.weight','ff.2.bias']
        allowed.update(f'blocks.{i}.'+n for n in names)
        rows.append(dict(layer=i,donor_width=d,recipient_width=block.input.in_features,
            donor_heads=12,donor_head_indices=heads.tolist(),memory_heads=model.policy.N,
            input_rank=256,content_rank=rank,donor_mlp_hidden=donor_hidden,
            recipient_mlp_hidden=recipient_hidden,
            value_relative_projection_error=float(torch.linalg.vector_norm(value-(value@basis.T)@basis)/torch.linalg.vector_norm(value)),
            content_relative_projection_error=float(torch.linalg.vector_norm(qkv[d:d+64]-kmap@basis)/torch.linalg.vector_norm(qkv[d:d+64]))))
    after = {n:tensor_sha(p) for n,p in model.named_parameters()}
    changed = [n for n in before if before[n]!=after[n]]
    assert set(changed)<=allowed
    assert all(before[n]==after[n] for n,p in model.named_parameters() if not p.requires_grad)
    atomic_json(OUT/'mapping.json',dict(donor=str(DONOR),donor_sha256=pure.sha(DONOR),
        copied=sorted(allowed),changed=changed,layers=rows,frozen_unchanged=True,
        untouched='queries, semantic embeddings, key projections, management, embedding/head, layer norms',
        scope='partial read-path transplant; geometry projection is lossy; not function-equivalent attention'))


class DiagnosticCandidate(Candidate):
    def __init__(self, policy):
        super().__init__(policy)
        before = self.counts()
        for block in self.blocks:
            block.cell.features.requires_grad_(True)
            for geometry in block.cell.geometry: geometry['key'].requires_grad_(True)
        atomic_json(OUT/'unfreeze_scope.json',dict(before=before,after=self.counts(),
            scope='diagnostic only; outside certified spine; never production; no parity claim',
            changes=['per-head semantic embeddings trainable','per-head keys trainable','key detach removed'],
            capacity='same tensor geometry and MLP as baseline; added trainable parameters reported, not capacity-rematched'))


def diagnostic_forward():
    source=inspect.getsource(Cell.forward)
    import textwrap
    tree=ast.parse(textwrap.dedent(source)); changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='k' for t in node.targets):
            unsqueeze=node.value
            assert ast.unparse(unsqueeze.func.value).endswith('.detach()')
            unsqueeze.func.value=unsqueeze.func.value.func.value
            changed+=1
    assert changed==1
    ast.fix_missing_locations(tree)
    namespace={};exec(compile(tree,__file__,'exec'),Cell.forward.__globals__,namespace)
    return namespace['forward']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=['donor','mimetic','unfrozen'])
    a=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    assert os.sys.executable=='/home/erikg/emender/.venv/bin/python'
    try:
        pure.verify_registration(REG.relative_to(Path.cwd()))
        audit=json.loads((ROOT/'receipts/pure_v2_architecture_audit/audit.json').read_text())
        policy=Policy(**audit['best_policy'])
        if a.arm=='donor':
            reg=ROOT/'velocity_transformer_registration.json'
            verify_search(reg,'transformer')
            results=[]
            for path in (ROOT/'runs/velocity_transformer_v1').glob('g*_c*/attempt*/result.json'):
                r=json.loads(path.read_text())
                if r.get('score_eligible') and r.get('status')=='completed':results.append((r['score'],str(path),r))
            best=min(results,key=lambda x:(x[0],x[1]))
            donor_policy=SearchPolicy(**best[2]['policy'])
            atomic_json(OUT/'donor_provenance.json',dict(source='fresh 2700s donor; original runners never save weights',
                best_available_recipe=best[1],best_recipe_score=best[0],policy=asdict(donor_policy),
                searched='X42 tree (all pt/pth/safetensors), /tmp and local user storage mounts; no attributable X42 weights located',
                corpus_train_sha256=pure.sha(ROOT/'data/pile24/train.bin'),corpus_valid_sha256=pure.sha(ROOT/'data/pile24/valid.bin')))
            run,_=amended(donor_train.train,donor_train,donor=True)
            run('transformer',donor_policy,reg,OUT/'donor')
        else:
            run,namespace=amended(pure.train,pure)
            if a.arm=='mimetic':
                checkpoint=torch.load(DONOR,map_location='cpu',weights_only=True)
                def factory(p):
                    model=Candidate(p);transplant(model,checkpoint);return model
                namespace['Candidate']=factory
                result=run(policy,REG,OUT/a.arm)
                result.update(experiment_arm='mimetic',initialization='partial trained donor read-path transplant',
                    protocol_sha256=pure.sha(OUT/'protocol.json'),donor_sha256=pure.sha(DONOR))
                atomic_json(OUT/a.arm/'result.json',result)
            else:
                models=[]
                def factory(p):
                    model=DiagnosticCandidate(p);models.append(model);return model
                namespace['Candidate']=factory
                original_step=torch.optim.AdamW.step
                audited=False
                def audited_step(optimizer,*args,**kwargs):
                    nonlocal audited
                    if not audited:
                        rows=[]
                        for name,p in models[0].named_parameters():
                            if '.cell.features.' in name or ('.cell.geometry.' in name and '.key.' in name):
                                rows.append(dict(name=name,present=p.grad is not None,
                                    finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),
                                    norm=None if p.grad is None else float(p.grad.norm())))
                        atomic_json(OUT/'unfreeze_gradients.json',dict(step=1,parameters=rows))
                        assert all(r['present'] and r['finite'] for r in rows), 'missing/nonfinite diagnostic gradients'
                        assert any(r['norm']>0 for r in rows if '.key.' in r['name']), 'no key gradient'
                        assert any(r['norm']>0 for r in rows if '.features.' in r['name']), 'no semantic gradient'
                        audited=True
                    return original_step(optimizer,*args,**kwargs)
                with patch.object(Cell,'forward',diagnostic_forward()), patch.object(torch.optim.AdamW,'step',audited_step):
                    result=run(policy,REG,OUT/a.arm)
                result.update(experiment_arm='unfrozen_basis_diagnostic',score_eligible=False,
                    comparison_protocol_complete=True,production_eligible=False,
                    scope='outside certified spine; parity gates do not apply; never production',
                    protocol_sha256=pure.sha(OUT/'protocol.json'))
                atomic_json(OUT/a.arm/'result.json',result)
    except BaseException:
        atomic_json(OUT/(a.arm+'_failure.json'),dict(status='failed',traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
