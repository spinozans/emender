"""CPU integration fixtures, not full E97/GPU training qualification."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from ndm.e97 import _apply_schedulefree_train_weights
from ndm.schedulefree_sr_candidate import ScheduleFreeSRCandidate
from scripts.train_e97_4b_pi_sft import (
    SCHEMA, build_optimizer, configure_precision, load_resume_optimizer,
    validate_precision_world,
)


def args(**changes):
    values=dict(optimizer_precision='bf16-sr-candidate',sr_seed=927413,
                offload_schedulefree_state=True,schedulefree_offload_pin_memory=0,
                schedulefree_offload_release_gradients=0,schedulefree_offload_bucket_numel=7,
                loss_logits_fp32=True,loss_chunk_size=128,checkpoint_loss_chunks=True,
                disable_bf16_reduced_precision_reduction=True,
                disable_diloco_merge=True,gradient_checkpoint_group_size=3,
                mlp_checkpoint_chunk_size=4096,lr=.03,weight_decay=.01,warmup_steps=0)
    values.update(changes)
    return SimpleNamespace(**values)


def model():
    result=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Linear(8,3)).bfloat16()
    result.loss_chunk_size=0
    return result


@pytest.fixture(autouse=True)
def restore_matmul_policy():
    previous=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    yield
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=previous


def prepared(*, cancellation_edge=False):
    torch.manual_seed(7241)
    m=model();a=args();policy=configure_precision(m,a)
    opt=build_optimizer(m.parameters(),a,named_parameters=m.named_parameters())
    opt.train()
    # Synthetic arithmetic fixtures only; no E97 model, source data or CUDA.
    for step in range(4):
        for p in m.parameters():p.grad=torch.randn_like(p)
        opt.step()
    if cancellation_edge:
        # A deliberate finite BF16 checkpoint edge, independent of whether the
        # random update history happens to survive inverse interpolation.
        # With beta1=.9, y=2^-8 and z=1 export to rounded x; reconstructing
        # beta1*x+(1-beta1)*z cannot recover the recorded y exactly.
        with torch.no_grad():
            p=next(m.parameters())
            p.view(-1)[0]=2**-8
            opt.state[p]['z'].view(-1)[0]=1
    y={n:p.detach().clone() for n,p in m.named_parameters()}
    opt.eval()
    return m,opt,y,policy


def test_selected_policy_is_applied_and_named_factory_is_used():
    m=model();a=args();policy=configure_precision(m,a)
    assert m.loss_logits_fp32 and m.loss_chunk_size==128 and m.checkpoint_loss_chunks
    assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    opt=build_optimizer(m.parameters(),a,named_parameters=m.named_parameters())
    assert isinstance(opt,ScheduleFreeSRCandidate)
    assert policy['optimizer_schema']==opt.state_schema
    with pytest.raises(ValueError,match='layout'):
        build_optimizer(list(m.parameters())[::-1],a,named_parameters=m.named_parameters())


@pytest.mark.parametrize('change',[
    {'loss_chunk_size':None},{'loss_chunk_size':0},{'loss_chunk_size':8192},
    {'offload_schedulefree_state':False},{'optimizer_precision':'unrecognized'},
    {'checkpoint_loss_chunks':False},
])
def test_invalid_precision_choices_fail(change):
    with pytest.raises(ValueError):configure_precision(model(),args(**change))


@pytest.mark.parametrize('world,disabled',[(64,True),(64,False),(8,False),(1,True)])
def test_unqualified_topologies_fail(world,disabled):
    with pytest.raises(ValueError,match='full-world'):
        validate_precision_world(args(disable_diloco_merge=disabled),world)


def test_only_full_world_ddp_candidate_is_available():
    validate_precision_world(args(),8)
    validate_precision_world(args(optimizer_precision='legacy',disable_diloco_merge=False),64)


def test_loader_restores_exact_recorded_y_not_inverse_interpolation():
    m,opt,y,policy=prepared(cancellation_edge=True)
    saved=deepcopy(m.state_dict());state=deepcopy(opt.state_dict())
    rebuilt=model().float();rebuilt.load_state_dict(saved)
    assert _apply_schedulefree_train_weights(rebuilt,{'optimizer_state_dict':state,'sft_precision':policy},{'optimizer':'schedulefree'})
    assert all(torch.equal(p,y[n]) for n,p in rebuilt.named_parameters())
    assert all(p.dtype==torch.bfloat16 for p in rebuilt.parameters())
    # Show this test discriminates between exact restoration and a fresh inverse.
    beta=state['param_groups'][0]['betas'][0]
    ids=state['param_groups'][0]['params']
    approximate=[saved[n].float().lerp(state['state'][i]['z'].float(),1-beta).bfloat16()
                 for (n,_),i in zip(m.named_parameters(),ids)]
    assert any(not torch.equal(value,expected) for value,expected in zip(approximate,y.values()))


@pytest.mark.parametrize('corruption',['backup','state','identity','optimizer_config','legacy_policy'])
def test_partial_or_mislabelled_state_cannot_restore_train_weights(corruption):
    m,opt,_,policy=prepared();state=deepcopy(opt.state_dict());config={'optimizer':'schedulefree'}
    first=state['param_groups'][0]['params'][0]
    if corruption=='backup':del state['eval_live_y'][first]
    if corruption=='state':del state['state'][first]
    if corruption=='identity':del state['precision_identity']
    if corruption=='optimizer_config':config['optimizer']='adamw'
    if corruption=='legacy_policy':policy['optimizer']='legacy'
    with pytest.raises(ValueError):
        _apply_schedulefree_train_weights(m,{'optimizer_state_dict':state,'sft_precision':policy},config)


def test_unused_first_step_parameter_still_has_complete_checkpoint_state():
    m=model();a=args();opt=build_optimizer(m.parameters(),a,named_parameters=m.named_parameters())
    opt.train();next(m.parameters()).grad=torch.ones_like(next(m.parameters()))
    opt.step();opt.eval()
    state=opt.state_dict()
    assert set(state['state'])==set(state['param_groups'][0]['params'])
    rebuilt=model();rebuilt.load_state_dict(m.state_dict())
    other=build_optimizer(rebuilt.parameters(),a,named_parameters=rebuilt.named_parameters())
    other.load_state_dict(state)


def test_resume_binds_entire_precision_policy(tmp_path):
    m,opt,_,policy=prepared();path=tmp_path/'checkpoint.pt'
    payload=dict(schema=SCHEMA,optimizer_state_dict=opt.state_dict(),sft_precision=policy,
                 sft_updates=4,sft_total_tokens=80,assistant_target_tokens=24)
    torch.save(payload,path)
    rebuilt=model();rebuilt.load_state_dict(m.state_dict())
    other=build_optimizer(rebuilt.parameters(),args(),named_parameters=rebuilt.named_parameters())
    assert load_resume_optimizer(path,other,{'sft_precision':policy})['updates']==4
    for key,value in [('learning_rate',.1),('loss_logits_fp32',False),('loss_chunk_size',256),
                      ('sr_seed',5),('bf16_reduced_precision_reduction',True),
                      ('checkpoint_loss_chunks',False),
                      ('mlp_checkpoint_chunk_size',256)]:
        changed={**policy,key:value}
        with pytest.raises(RuntimeError,match='sft_precision'):
            load_resume_optimizer(path,other,{'sft_precision':changed})
    del payload['sft_precision'];torch.save(payload,path)
    with pytest.raises(RuntimeError,match='sft_precision'):
        load_resume_optimizer(path,other,{'sft_precision':policy},allow_legacy_precision=True)
