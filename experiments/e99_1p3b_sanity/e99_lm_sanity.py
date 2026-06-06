#!/usr/bin/env python3
"""E99 / E98 / GDN-2 — 1.3B-class LM wiring sanity + checkpoint ROUND-TRIP.

Task: wire-e99-e98. This drives the REAL production LM path (the same
`ndm/models/ladder_lm.py` LadderLM + FLA-GDN code train.py uses) on REAL Pile
data for a short window and verifies, per candidate:

  1. loss decreases over the window, NaN-free, dtype path is what we claim;
  2. throughput (tok/s) -> projected GPU-days for a token budget;
  3. checkpoint ROUND-TRIP LOSS CONSISTENCY: after a few steps we save the model
     (production `model_state_dict` convention), record the loss on a FIXED
     held batch, then reload the checkpoint in a FRESH PROCESS and recompute the
     loss on the SAME batch. A strict-clean load is NOT enough (PILE_BPB_MEASURED
     documents an E88 checkpoint that loaded 0-missing/0-unexpected yet produced
     ~17.6 nats/token from a forward/recurrence mismatch). We treat a reload that
     does not reproduce the loss within tolerance as a HARD BLOCKER.

All training is REAL (real Pile tokens, real fwd/bwd, schedule-free AdamW). No
mock data. `paper/main.typ` is never touched; no checkpoint is published.

Usage:
  python e99_lm_sanity.py --config typed-gdn2-lm --gpu 5 --steps 20
  python e99_lm_sanity.py --phase reload --reload_ckpt /path/ckpt.pt   # internal
"""
import os, sys, json, time, argparse, subprocess, datetime

# Repo root on path (this file is experiments/e99_1p3b_sanity/)
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

DATA = '/home/erikg/elman/data/pile.txt'
TOKENIZER = 'p50k_base'
CHUNK = 2048          # ctx, matched to the prior CMA / THROUGHPUT convention
ROUNDTRIP_SEED = 1234  # fixed -> deterministic held batch across processes

# --------------------------------------------------------------------------
# Candidate roster. Each entry is the EXACT production-path config: the level
# string train.py resolves, the LadderLM shape, and the per-layer candidate
# knobs forwarded as layer_kwargs. Shapes chosen to sit at 1.3B-class with a
# depth comparable to the E88 (12) / FLA-GDN (21) baselines, preserving the
# DISCOVERED FORM of each candidate (E99: 5:1 GDN2:nonlinear head ratio via the
# typed-gdn-2-head winner logits; E98-CMA: cma-capability meta-config knobs).
# --------------------------------------------------------------------------
CONFIGS = {
    # E99 typed Emender: native GDN-2 + nonlinear mix. Logits are the
    # typed-gdn-2-head CMA winner [gdn2_recall, e97_track, count, latch, nonlin];
    # softmax -> ~82% GDN-2 / ~17% nonlinear == the discovered 5:1 ratio.
    'typed-gdn2-lm': dict(
        level='typed-gdn2-lm', dim=3072, depth=22, n_heads=96, n_state=32,
        expansion=1.0, bf16=False, lr=9.95e-4, knob_lr_mult=1.0,
        layer_kwargs=dict(
            head_type_logits=[3.9995, -1.9008, -0.9211, -2.8866, 2.4146],
            gdn_allow_neg_eigval=True, lam_max=1.585, beta_max=2.747),
        note='E99 typed Emender (native GDN-2 heads + nonlinear specialist, 5:1)'),
    # Dense native GDN-2 / GatedDeltaNet control == the FLA GatedDeltaNet level,
    # the SAME native delta-memory backbone the typed GDN-2 heads run, and the
    # measured 1.352B FLA-GDN baseline from paper/review/THROUGHPUT.md.
    'fla-gdn': dict(
        level='fla-gdn', dim=2688, depth=21, n_heads=44, n_state=64,
        expansion=2.0, bf16=True, lr=8.63e-4, knob_lr_mult=1.0,
        layer_kwargs=None,
        note='Dense native GDN-2 / GatedDeltaNet control (FLA chunked kernel)'),
    # E98-CMA / unified Emender control: cma-capability winner meta-config
    # (split-gate + spread-init + corner_mixture + knob-LR), shape re-derived to
    # 1.3B-class (the report calls this the e98-scale-run starting point).
    'e98-cma-lm': dict(
        level='e98-cma-lm', dim=3072, depth=17, n_heads=192, n_state=16,
        expansion=1.0, bf16=False, lr=9.79e-4, knob_lr_mult=5.38,
        layer_kwargs=dict(
            corner_mixture=[0.4015, 0.2821, 0.0089, 0.3075],
            lam_max=1.585, beta_max=2.747, igain_max=2.0),
        note='E98-CMA unified Emender control (cma-capability meta-config)'),
}


def build_model(cfg, vocab_size, device):
    from ndm.models.ladder_lm import LadderLM
    m = LadderLM(
        vocab_size=vocab_size, dim=cfg['dim'], depth=cfg['depth'],
        level=cfg['level'], n_heads=cfg['n_heads'], n_state=cfg['n_state'],
        expansion=cfg['expansion'], layer_kwargs=cfg['layer_kwargs'])
    m = m.to(device)
    if cfg['bf16']:
        m = m.bfloat16()
    return m


def make_dataset(vocab_size, seed, batch_size):
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    ds = TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1,
                                seed=seed, tokenizer_name=TOKENIZER)
    return ds


def fixed_batch(vocab_size, batch_size, device):
    """A deterministic held batch (fixed seed) for round-trip loss comparison."""
    ds = make_dataset(vocab_size, ROUNDTRIP_SEED, batch_size)
    chunks, _, _ = ds.get_batch(batch_size, device=device)
    return chunks  # [B, CHUNK+1]


def loss_on(model, chunks, bf16):
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
    return float(loss.item())


def phase_reload(args):
    """Fresh-process reload: build, strict-load, recompute loss on saved batch."""
    ckpt = torch.load(args.reload_ckpt, map_location='cpu')
    cfg = ckpt['cfg']
    device = 'cuda'
    vocab_size = ckpt['vocab_size']
    model = build_model(cfg, vocab_size, device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    chunks = ckpt['eval_batch'].to(device)
    l_post = loss_on(model, chunks, cfg['bf16'])
    out = dict(l_post=l_post, n_missing=len(missing), n_unexpected=len(unexpected),
               missing=list(missing)[:8], unexpected=list(unexpected)[:8])
    print('RELOAD_RESULT ' + json.dumps(out))
    return out


def phase_run(args):
    cfg = CONFIGS[args.config]
    device = 'cuda'
    torch.manual_seed(0)
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    # vocab from tokenizer
    probe = TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1, seed=0,
                                   tokenizer_name=TOKENIZER)
    vocab_size = probe.vocab_size if hasattr(probe, 'vocab_size') else 50281

    model = build_model(cfg, vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())

    # Optimizer with knob-LR group (matches train.py wiring).
    import schedulefree
    KNOB_SUFFIXES = ('lam_raw', 'beta_raw', 'igain_raw', 'gamma_raw')
    knob, base = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (knob if any(name.endswith(s) for s in KNOB_SUFFIXES) else base).append(p)
    if cfg['knob_lr_mult'] != 1.0 and knob:
        groups = [{'params': base, 'lr': cfg['lr']},
                  {'params': knob, 'lr': cfg['lr'] * cfg['knob_lr_mult']}]
    else:
        groups = model.parameters()
    opt = schedulefree.AdamWScheduleFree(groups, lr=cfg['lr'],
                                         weight_decay=0.01, betas=(0.9, 0.95))

    train_ds = TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1,
                                      seed=42, tokenizer_name=TOKENIZER)
    bs = args.batch_size

    model.train(); opt.train()
    losses, toks_per_s = [], []
    nan_seen = False
    warmup_done_t = None
    tokens_after_warmup = 0
    for step in range(1, args.steps + 1):
        chunks, _, _ = train_ds.get_batch(bs, device=device)
        t0 = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        lv = float(loss.item())
        losses.append(lv)
        if not (lv == lv):  # NaN
            nan_seen = True
            print(f'NaN at step {step}', flush=True)
            break
        # tok/s: skip first 3 steps (compile/warmup ramp)
        if step > 3:
            toks_per_s.append(bs * (CHUNK + 1) / dt)
            tokens_after_warmup += bs * (CHUNK + 1)
        print(f'step {step:3d} | loss {lv:.4f} | dt {dt*1000:.0f}ms | '
              f'tok/s {bs*(CHUNK+1)/dt:.0f}', flush=True)

    sustained_tok_s = (sum(toks_per_s) / len(toks_per_s)) if toks_per_s else 0.0

    # ---- ROUND-TRIP: save (production convention), record fixed-batch loss ----
    held = fixed_batch(vocab_size, bs, device)
    l_pre = loss_on(model, held, cfg['bf16'])
    model.train(); opt.train()  # restore (irrelevant to saved weights)

    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, f'{args.config}_sanity.pt')
    torch.save({
        'step': args.steps,
        'model_state_dict': model.state_dict(),   # production key name
        'cfg': cfg,
        'vocab_size': vocab_size,
        'eval_batch': held.detach().cpu(),
        'l_pre': l_pre,
    }, ckpt_path)

    # Free THIS process's GPU memory before spawning the reload subprocess —
    # otherwise two 1.3B models share one card and OOM. l_pre is already
    # computed and the held batch is on CPU inside the checkpoint.
    import gc
    del opt
    model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Fresh process reload.
    rt = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--phase', 'reload',
         '--reload_ckpt', ckpt_path],
        env={**os.environ}, capture_output=True, text=True)
    reload_out = None
    for line in rt.stdout.splitlines():
        if line.startswith('RELOAD_RESULT '):
            reload_out = json.loads(line[len('RELOAD_RESULT '):])
    if reload_out is None:
        print('RELOAD STDERR:\n' + rt.stderr[-2000:], flush=True)
        raise RuntimeError('reload phase produced no result')

    l_post = reload_out['l_post']
    delta = abs(l_post - l_pre)
    tol = 1e-2  # nats/token; forward must reproduce within this
    roundtrip_ok = delta < tol and reload_out['n_missing'] == 0 and reload_out['n_unexpected'] == 0

    # projection: GPU-days for a token budget at sustained tok/s
    def gpu_days(tokens):
        return tokens / sustained_tok_s / 86400 if sustained_tok_s else None

    result = dict(
        config=args.config, level=cfg['level'], note=cfg['note'],
        params=n_params, params_b=round(n_params / 1e9, 4),
        dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'],
        n_state=cfg['n_state'], expansion=cfg['expansion'], lr=cfg['lr'],
        knob_lr_mult=cfg['knob_lr_mult'], batch_size=bs, chunk_size=CHUNK,
        dtype=('bf16' if cfg['bf16'] else 'fp32'),
        steps=args.steps, loss_first=losses[0] if losses else None,
        loss_last=losses[-1] if losses else None,
        loss_decreased=(len(losses) >= 2 and losses[-1] < losses[0]),
        nan_seen=nan_seen,
        sustained_tok_s=round(sustained_tok_s, 1),
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
        roundtrip_l_pre=round(l_pre, 6),
        roundtrip_l_post=round(l_post, 6),
        roundtrip_delta=round(delta, 8),
        roundtrip_n_missing=reload_out['n_missing'],
        roundtrip_n_unexpected=reload_out['n_unexpected'],
        roundtrip_ok=roundtrip_ok,
        gpu_days_per_1B_tok=gpu_days(1e9),
        gpu_days_per_10B_tok=gpu_days(10e9),
        timestamp=datetime.datetime.utcnow().isoformat() + 'Z',
    )
    out_json = os.path.join(args.outdir, f'{args.config}_result.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print('SANITY_RESULT ' + json.dumps(result), flush=True)

    # The sanity checkpoint has served its round-trip purpose. Delete it — task
    # constraint: do NOT publish/stage/upload any generated checkpoint.
    try:
        os.remove(ckpt_path)
    except OSError:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=list(CONFIGS.keys()))
    ap.add_argument('--phase', choices=['run', 'reload'], default='run')
    ap.add_argument('--reload_ckpt', type=str, default=None)
    ap.add_argument('--gpu', type=str, default=None)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--outdir', type=str,
                    default=os.path.join(_THIS, 'results'))
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.phase == 'reload':
        phase_reload(args)
    else:
        phase_run(args)


if __name__ == '__main__':
    main()
