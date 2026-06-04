"""gdn-vs-e88-transition: empirical per-step transition-operator spectra.

Compares the realized eigenvalue spectra of the per-token state-transition
operator A_t for the two S5-symmetric winners:

  * e88-linear  (E88FLAHybrid, linear_state=1, use_gate=1)
        A_t = decay_t * I  -  k_norm_t k_norm_t^T
        eigenvalues: decay_t  (mult n-1, perp to k)   in (0,1)
                     decay_t - 1  (mult 1, along k)   in (-1,0)  <-- NEGATIVE
        (e88_fla_hybrid.py:1732-1733 serial; :1881-1886 affine-scan form)

  * gdn  (fla GatedDeltaNet, allow_neg_eigval=False default)
        A_t = g_t * (I  -  beta_t k_t k_t^T)
        eigenvalues: g_t          (mult n-1, perp to k)  in (0,1)
                     g_t(1-beta_t) (mult 1, along k)     in (0,g_t)  POSITIVE
        (fla/layers/gated_deltanet.py:266-270; ops/.../fused_recurrent.py:104-115)

Both are rank-1 (single-reflection) updates per token; the architectural
difference probed here is the *sign reachable* by the along-k eigenvalue
(Grazzi 2025, negative eigenvalues unlock state-tracking).

No saved checkpoints of the winners exist (only eval JSON), so each winner
config is trained on S5 on ONE GPU for --steps, then the realized transition
spectra are extracted on real S5 batches via forward-pre-hooks on the inner
recurrent modules. The sign structure is parameterization-invariant (holds at
init and after training); we report the trained-model accuracy for context.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F

from ndm.models.hybrid_ladder import HybridLadderLM
from ndm.models.e88_fla_hybrid import E88FLAHybrid
from experiments.expressivity_tasks.tasks import ALL_TASKS

try:
    from fla.layers.gated_deltanet import GatedDeltaNet as FLAGatedDeltaNet
except Exception:
    FLAGatedDeltaNet = None

WINNERS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results/s5_symmetric_20260603/winners')


def build_model(arm, task, device):
    w = json.load(open(os.path.join(WINNERS, f'{arm}.args.json')))
    p = w['params']
    level = {'e88': 'E88', 'fla-gdn': 'fla-gdn', 'm2rnn': 'm2rnn'}[w['model']]
    e88_kw = {}
    if w['model'] == 'e88':
        if 'linear_state' in p: e88_kw['linear_state'] = bool(p['linear_state'])
        if 'use_gate' in p:     e88_kw['use_gate'] = bool(p['use_gate'])
    layer_kwargs = [dict(e88_kw) if level.startswith('E88') else {}]
    model = HybridLadderLM(
        vocab_size=task.vocab_size,
        dim=int(p['dim']), depth=int(p['depth']),
        layer_pattern=[level], layer_kwargs=layer_kwargs,
        n_state=int(p['n_state']), n_heads=int(p['n_heads']),
        expansion=1.0,
    ).to(device)
    return model, w, float(p['lr'])


def train(model, task, lr, steps, seq_len, B, seed, device):
    import schedulefree
    rng = np.random.default_rng(seed)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr,
                                         weight_decay=0.01, betas=(0.9, 0.95))
    model.train(); opt.train()
    for step in range(steps):
        inp, tgt, mask = task.generate_batch(B, seq_len, rng)
        x = torch.from_numpy(inp).to(device)
        y = torch.from_numpy(tgt).to(device)
        m = torch.from_numpy(mask).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        lp = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                             y.view(-1), reduction='none').view_as(m)
        loss = (lp * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss={loss.item():.4f}", flush=True)
    opt.eval()
    return opt


@torch.no_grad()
def evaluate(model, task, B, T, n_batches, seed, device):
    model.eval()
    rng = np.random.default_rng(seed + 9999)
    correct = total = 0
    for _ in range(n_batches):
        inp, tgt, mask = task.generate_batch(B, T, rng)
        x = torch.from_numpy(inp).to(device); y = torch.from_numpy(tgt).to(device)
        m = torch.from_numpy(mask).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        preds = logits.argmax(-1)
        correct += ((preds == y) & m).sum().item(); total += m.sum().item()
    return correct / max(total, 1)


def e88_eigs(mod, x):
    """A_t = decay*I - kk^T. Returns (perp_eig, alongk_eig) flat over B,T,H."""
    x = x.float()
    alpha = F.linear(x, mod.a_proj.weight.float())          # [B,T,H]
    g = -mod.A_log.float().exp() * F.softplus(alpha + mod.dt_bias.float())
    decay = g.exp()                                          # [B,T,H] in (0,1)
    perp = decay
    alongk = decay - 1.0
    return perp.reshape(-1).cpu().numpy(), alongk.reshape(-1).cpu().numpy()


def gdn_eigs(mod, x):
    """A_t = g*(I - beta*kk^T). Returns (perp_eig, alongk_eig)."""
    x = x.float()
    a = F.linear(x, mod.a_proj.weight.float())              # [B,T,HV]
    g = (-mod.A_log.float().exp() * F.softplus(a + mod.dt_bias.float())).exp()
    beta = torch.sigmoid(F.linear(x, mod.b_proj.weight.float()))
    if getattr(mod, 'allow_neg_eigval', False):
        beta = beta * 2.0
    perp = g
    alongk = g * (1.0 - beta)
    return perp.reshape(-1).cpu().numpy(), alongk.reshape(-1).cpu().numpy()


def extract(model, arm, task, B, T, n_batches, seed, device):
    is_e88 = arm == 'e88-linear'
    target_cls = E88FLAHybrid if is_e88 else FLAGatedDeltaNet
    mods = [m for m in model.modules() if isinstance(m, target_cls)]
    captured = {id(m): [] for m in mods}
    hooks = []
    for m in mods:
        def pre(mod, inp, _store=captured, _id=id(m)):
            _store[_id].append(inp[0].detach())
        hooks.append(m.register_forward_pre_hook(pre))

    perp_all, alongk_all = [], []
    model.eval()
    rng = np.random.default_rng(seed + 31337)
    with torch.no_grad():
        for _ in range(n_batches):
            for v in captured.values():
                v.clear()
            inp, _, _ = task.generate_batch(B, T, rng)
            x = torch.from_numpy(inp).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model(x)
            for m in mods:
                xin = captured[id(m)][-1]
                p, a = (e88_eigs(m, xin) if is_e88 else gdn_eigs(m, xin))
                perp_all.append(p); alongk_all.append(a)
    for h in hooks:
        h.remove()
    return np.concatenate(perp_all), np.concatenate(alongk_all)


def summ(name, arr):
    pct = np.percentile(arr, [0, 1, 5, 50, 95, 99, 100])
    return {
        'name': name, 'n': int(arr.size),
        'min': float(arr.min()), 'max': float(arr.max()),
        'mean': float(arr.mean()), 'std': float(arr.std()),
        'p0': float(pct[0]), 'p1': float(pct[1]), 'p5': float(pct[2]),
        'p50': float(pct[3]), 'p95': float(pct[4]), 'p99': float(pct[5]),
        'p100': float(pct[6]),
        'frac_negative': float((arr < 0).mean()),
        'frac_lt_-0.25': float((arr < -0.25).mean()),
        'frac_lt_-0.5': float((arr < -0.5).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['e88-linear', 'gdn'])
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--extract_batches', type=int, default=4)
    ap.add_argument('--extract_T', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    task = ALL_TASKS['s5_permutation']()
    arm_file = 'e88-linear' if args.arm == 'e88-linear' else 'gdn'
    model, w, lr = build_model(arm_file, task, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.arm}] model={w['model']} params={n_params:,} lr={lr}", flush=True)

    # Spectra at INIT (sign structure is parameterization-invariant)
    perp0, along0 = extract(model, args.arm, task, args.batch_size,
                            args.extract_T, 2, args.seed, device)

    print(f"[{args.arm}] training {args.steps} steps on S5 ...", flush=True)
    train(model, task, lr, args.steps, args.seq_len, args.batch_size, args.seed, device)
    acc128 = evaluate(model, task, args.batch_size, 128, 8, args.seed, device)
    acc256 = evaluate(model, task, args.batch_size, 256, 8, args.seed, device)
    print(f"[{args.arm}] trained S5 acc T128={acc128:.4f} T256={acc256:.4f}", flush=True)

    perp, along = extract(model, args.arm, task, args.batch_size,
                          args.extract_T, args.extract_batches, args.seed, device)

    out = {
        'arm': args.arm, 'model': w['model'], 'params': n_params, 'lr': lr,
        'steps': args.steps, 'seed': args.seed,
        'trained_acc': {'T128': acc128, 'T256': acc256},
        'transition_form': (
            'A_t = decay_t*I - k k^T   (eig: decay, decay-1)' if args.arm == 'e88-linear'
            else 'A_t = g_t*(I - beta_t k k^T)   (eig: g, g*(1-beta))'),
        'reflections_per_token': 1,
        'state_rank_update_per_token': 1,
        'allow_neg_eigval': (None if args.arm == 'e88-linear'
                             else bool(getattr(
                                 [m for m in model.modules()
                                  if isinstance(m, FLAGatedDeltaNet)][0],
                                 'allow_neg_eigval', False))),
        'init_alongk_eig': summ('alongk_init', along0),
        'init_perp_eig': summ('perp_init', perp0),
        'trained_alongk_eig': summ('alongk_trained', along),
        'trained_perp_eig': summ('perp_trained', perp),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2)
    print(json.dumps({k: out[k] for k in
                      ['arm', 'trained_acc', 'trained_alongk_eig', 'trained_perp_eig']},
                     indent=2), flush=True)
    print(f"Saved {args.out}", flush=True)


if __name__ == '__main__':
    main()
