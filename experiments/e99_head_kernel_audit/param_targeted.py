"""Targeted real param counts: fixed-shape mixture spread + per-mixture dim derived
to the 1270M handoff convention and to 1.3B. Real LadderLM builds, narrow ranges.

Audit task e99-head-kernel-audit Q4. CPU-only construction (no kernel launch).
"""
import sys, os, json, math
sys.path.insert(0, "/home/erikg/ndm/.wg-worktrees/agent-1141")
import torch
from ndm.models.ladder_lm import LadderLM

VOCAB = 50281
DENSE = [12, -12, -12, -12, -12]
M5 = [math.log(5), 0, 0, 0, 0]
M2 = [math.log(2), 0, 0, 0, 0]
UNI = [0, 0, 0, 0, 0]
NON = [-12, -12, -12, -12, 12]


def count(dim, depth, nh, ns, g):
    m = LadderLM(vocab_size=VOCAB, dim=dim, depth=depth, level='typed-gdn2-lm',
                 n_heads=nh, n_state=ns, expansion=1.0,
                 layer_kwargs=dict(head_type_logits=g, gdn_allow_neg_eigval=True,
                                   lam_max=1.585, beta_max=2.747))
    t = sum(p.numel() for p in m.parameters())
    del m
    return t


def row(label, dim, depth, nh, ns, g):
    t = count(dim, depth, nh, ns, g)
    return dict(label=label, dim=dim, depth=depth, n_heads=nh, n_state=ns, total=t,
                params_b=round(t / 1e9, 4),
                dev_1p3=round((t - 1.3e9) / 1.3e9 * 100, 2),
                dev_1270=round((t - 1.270e9) / 1.270e9 * 100, 2))


def derive(label, depth, nh, ns, g, target, dims):
    best = None
    for d in dims:
        r = row(label, d, depth, nh, ns, g)
        if best is None or abs(r['total'] - target) < abs(best['total'] - target):
            best = r
    best['label'] = label
    return best


if __name__ == '__main__':
    rows = []
    for lab, g in [("fixedshape_dense", DENSE), ("fixedshape_5to1", M5),
                   ("fixedshape_2to1", M2), ("fixedshape_uniform", UNI),
                   ("fixedshape_all_nonlin", NON)]:
        rows.append(row(lab, 3328, 17, 102, 32, g))
        print("ok", lab, rows[-1]['params_b'], flush=True)
    rows.append(derive("dim->1270M_dense", 17, 102, 32, DENSE, 1.270e9, range(3456, 3841, 64)))
    print("ok dense1270", rows[-1]['params_b'], flush=True)
    rows.append(derive("dim->1270M_5to1", 17, 102, 32, M5, 1.270e9, range(3200, 3585, 64)))
    print("ok 5to1_1270", rows[-1]['params_b'], flush=True)
    rows.append(derive("dim->1270M_2to1", 17, 102, 32, M2, 1.270e9, range(3264, 3649, 64)))
    print("ok 2to1_1270", rows[-1]['params_b'], flush=True)
    rows.append(derive("dim->1.3B_dense", 17, 102, 32, DENSE, 1.300e9, range(3520, 3905, 64)))
    print("ok dense1.3", rows[-1]['params_b'], flush=True)
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), 'param_targeted.json'), 'w'), indent=2)
    print("WROTE param_targeted.json", flush=True)
