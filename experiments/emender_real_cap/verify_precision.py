import os, sys, math, torch
sys.path.insert(0, os.getcwd())
from ndm.models.ladder_lm import LadderLM
from ndm.models.typed_head_mixture import allocate_types

# TYPE_NAMES: gdn2_recall,e97_track,count,latch,nonlin,gdn2_nonlin_shell,e97_raw,e97_delta,refit
# Emender: sea of gdn2_recall + sparse e97_delta. 8/64 nonlin -> logits
import numpy as np
def logits_for(n_heads, frac_delta):
    # fraction of e97_delta; rest gdn2_recall
    import math
    f = frac_delta
    L = [-30.0]*9
    L[0] = math.log(1-f)   # gdn2_recall
    L[7] = math.log(f)     # e97_delta
    return L

n_heads=16
L = logits_for(n_heads, 8/64)
print("alloc:", allocate_types(n_heads, L)['counts'])

def try_dtype(name, cast):
    torch.manual_seed(0)
    m = LadderLM(vocab_size=512, dim=256, depth=2, level='typed-gdn2-lm',
                 n_heads=n_heads, n_state=16, expansion=1.0,
                 layer_kwargs=dict(head_type_logits=L, gdn_allow_neg_eigval=True,
                                   lam_max=1.585, beta_max=2.747,
                                   e97_state_nonlin='tanh')).cuda()
    if cast is not None:
        m = m.to(cast)
    m.train()
    x = torch.randint(0, 512, (2, 128), device='cuda')
    try:
        if cast is None:  # autocast path
            with torch.autocast('cuda', dtype=torch.float16):
                loss = m(x, return_loss=True)
        else:
            loss = m(x, return_loss=True)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        print(f"[{name}] OK loss={float(loss):.4f} finite={torch.isfinite(loss).item()} gradnorm={float(gn):.3f}")
    except Exception as e:
        print(f"[{name}] RAISED: {type(e).__name__}: {str(e)[:300]}")

try_dtype("bf16-cast", torch.bfloat16)
try_dtype("fp16-cast", torch.float16)
try_dtype("fp16-autocast", None)
