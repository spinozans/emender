"""Real parameter counts for E99 1.3B-class LadderLM (typed-gdn2-lm).

Audit task e99-head-kernel-audit, Q4: the logged redo config (dim3328/depth17/
n_heads102/n_state32) was described as ~1.1B. Count exactly, then derive corrected
shapes hitting 1.3B +/-2%. Uses the REAL redo implementation in the agent-1141
worktree (TypedHeadMixtureLayer + gdn2_nonlin_shell-aware ladder_lm). No kernel
launch — params are counted on CPU (meta-free construction).
"""
import sys, os, json, argparse

REDO_ROOT = "/home/erikg/ndm/.wg-worktrees/agent-1141"
sys.path.insert(0, REDO_ROOT)

import torch
from ndm.models.ladder_lm import LadderLM

VOCAB = 50281  # p50k_base, matches screen_worker default

# Dense-GDN logits (all weight on gdn2_recall) and a 5:1 mixture for the
# 5-type typed head (gdn2_recall, e97_track, count, latch, nonlin).
DENSE = [12.0, -12.0, -12.0, -12.0, -12.0]
MIX_5to1 = [1.6094, 0.0, 0.0, 0.0, 0.0]  # softmax -> ~5:1 gdn:rest (ln5)

def count(dim, depth, n_heads, n_state, logits, expansion=1.0, label=""):
    lk = dict(head_type_logits=logits, gdn_allow_neg_eigval=True,
              lam_max=1.585, beta_max=2.747)
    m = LadderLM(vocab_size=VOCAB, dim=dim, depth=depth, level='typed-gdn2-lm',
                 n_heads=n_heads, n_state=n_state, expansion=expansion,
                 layer_kwargs=lk)
    total = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n, p in m.named_parameters()
              if 'embed' in n.lower() or 'wte' in n.lower() or 'tok' in n.lower())
    head = sum(p.numel() for n, p in m.named_parameters()
               if 'lm_head' in n.lower() or 'to_logits' in n.lower() or n.lower().endswith('head.weight'))
    del m
    return dict(label=label, dim=dim, depth=depth, n_heads=n_heads, n_state=n_state,
                expansion=expansion, total=total, params_b=round(total/1e9, 4),
                embed=emb, lm_head=head)

if __name__ == '__main__':
    rows = []
    # 1) logged redo config
    rows.append(count(3328, 17, 102, 32, DENSE, label="redo_logged_dense"))
    rows.append(count(3328, 17, 102, 32, MIX_5to1, label="redo_logged_5to1"))
    print(json.dumps(rows[-2], indent=0))
    print(json.dumps(rows[-1], indent=0))
    out = os.path.join(os.path.dirname(__file__), 'param_counts.json')
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print("WROTE", out)
