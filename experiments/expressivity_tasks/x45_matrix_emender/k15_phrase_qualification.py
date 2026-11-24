"""Pre-update CPU qualification; no fit or quality scoring."""
import copy, json, os, time, resource
import torch
from .k15_phrase_recurrent import PhraseK15, RecurrentReplay
from .k15_bookmarks import BookmarkSession
from .k15_historical_training import ANCHOR, ANCHOR_SHA, j
from .k10_separated_keys import convert_model_to_separated_keys
from .matched_arms import matrix_config, matrix_model_for, RANK, RoutingPolicy

def backbone():
    assert j.digest(ANCHOR)==ANCHOR_SHA and j.digest(j.EXE)=="87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f"
    config = j.incumbent._tuned_winner()
    policy = RoutingPolicy(ops=True, **{k:config[k] for k in ('tau_match','tau_merge','tau_pop','read_scale')})
    base = matrix_model_for(matrix_config(rank=RANK), policy, RANK, compile_scan=False)
    base = j.convert_model_to_composition(convert_model_to_separated_keys(base))
    base.load_state_dict(torch.load(ANCHOR,map_location='cpu',weights_only=True),strict=True)
    base.requires_grad_(False).eval()
    return base

PREFIX = b'z'*124 + b'ABAB'


def note(**data):
    print(json.dumps(data), flush=True)


@torch.no_grad()
def trace(model, size):
    session = BookmarkSession(j.EXE)
    try:
        replay = RecurrentReplay(model, session, PREFIX)
        session.rpc('seal'); result = []
        for _ in range(size):
            law = replay.predict(chosen=True); edge, = replay.pending[0]
            y = int(law.argmax()); replay.observe(y)
            result.append((edge.action, edge.state.distance, y, len(edge.state.rows)))
        note(trace=result, phrase_accesses=replay.reads, forwards=replay.calls)
        return result
    finally: session.close()


def main():
    assert __debug__ and os.environ['CUDA_VISIBLE_DEVICES']=='' and torch.get_num_threads()==4
    torch.manual_seed(3411); base = backbone(); model = PhraseK15(base); tic = time.monotonic()
    expected = [('start',2,65,128),('continue',2,66,128)]
    compiled = trace(model,3); assert compiled[:2]==expected and compiled[2][-1]==129
    torch.manual_seed(3411); random_model = PhraseK15(base,compiled=False)
    assert trace(random_model,2) != expected
    with torch.no_grad(): model.operand.weight[0,0].neg_()
    assert trace(model,1)[0][2]==66
    with torch.no_grad(): model.operand.weight[0,0].neg_()
    session = BookmarkSession(j.EXE)
    try:
        replay = RecurrentReplay(model, session, PREFIX)
        base.blocks[0].norm1.weight.requires_grad_(True)
        p = replay.predict(); assert abs(float(p.sum().detach())-1)<1e-5 and p.min()>0
        # K15 receives gradients through feedback; frozen base is never updated.
        first = replay.pending[0]; replay.observe(65)
        q = replay.predict(); assert abs(float(q.sum().detach())-1)<1e-5
        # Groups are the first COPY vs LITERAL histories consistent with A.
        by_origin = {}
        for st, mass in replay.paths:
            single = copy.copy(replay)
            single.paths = [(st, torch.ones_like(mass))]; single.pending = None; single.reads = []
            with torch.no_grad(): by_origin[st.distance] = float(single.predict()[66])
        note(second_byte_given_first_history=by_origin)
        assert abs(by_origin[2]-by_origin[0])>1e-5
        replay.observe(66); live = len(replay.paths); r = replay.predict()
        later = -r[65].log()
        grads = torch.autograd.grad(later,(first[0].weight, first[-1].payload, base.blocks[0].norm1.weight))
        note(delayed_gradients=[float(g.norm()) for g in grads], paths_before_third=live, forwards=replay.calls)
        assert all(g.isfinite().all() and g.norm()>0 for g in grads)
        replay.observe(65); note(paths=len(replay.paths), forwards=replay.calls)
    finally:
        note(seconds=time.monotonic()-tic, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        session.close()


if __name__=='__main__': main()
