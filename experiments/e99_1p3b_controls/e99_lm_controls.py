#!/usr/bin/env python3
"""E99 1.3B LM — matched head-to-head CONTROLS (task run-matched-1-3b).

Bounded controls (NOT a search) interpreting the E99 LM-CMA result. Same
production LM path, same Pile/tokenizer/ctx, same budget caps, same fitness
(handoff AvgLoss / Final, nats/token) and same checkpoint ROUND-TRIP method as
the E99 sanity (experiments/e99_1p3b_sanity/e99_lm_sanity.py) and the prior
E97/GDN-2 CMA-ES batch (docs/HANDOFF_E97_GDN2_CMAES_20260528.md).

Each arm is trained WALLCLOCK-MATCHED to the handoff `--train_minutes` budget
(15 min default) on REAL Pile tokens, real fwd/bwd, schedule-free AdamW. We
record, per (arm, seed):

  * the full loss-vs-tokens curve (so a TOKEN-MATCHED cross-walk can be read off
    a single wallclock-matched run — fp32 arms see ~3.5x fewer tokens than the
    bf16 control in the same 15 min; budget_caps.json COMPARABILITY_WARNING);
  * AvgLoss = mean train loss over the window, Final = last loss (handoff
    fitness, nats/token) -- reported so columns line up with the prior table;
  * late-train loss = mean over the last 10% of steps (more stable "Final");
  * held-out loss on a FIXED held-out slice (seed disjoint from train), in
    nats/token, plus tokenizer-invariant BPB = (nats/tok)/ln2/(bytes/tok) with
    bytes/token MEASURED on that exact held-out slice (p50k_base);
  * sustained tok/s, peak memory, total tokens seen, walltime;
  * checkpoint ROUND-TRIP loss consistency in a FRESH PROCESS (the hard
    requirement: a strict-clean load is NOT sufficient -- PILE_BPB_MEASURED
    documents an E88 ckpt that loaded 0/0 yet forward-mismatched to ~17.6
    nats/tok);
  * stability / failure taxonomy (NaN, OOM, walltime hard-stop), reusing the
    handoff schedule-free-AdamW fragility accounting.

HARD STOP: NaN -> stop+log; CUDA OOM -> stop+log; walltime exceeds cap*safety
-> stop+log. No checkpoint is published/staged; the round-trip ckpt is deleted
after use. paper/main.typ is never touched.

Usage (one arm, one seed):
  python e99_lm_controls.py --config fla-gdn --seed 0 --gpu 5 --train_minutes 15
Internal reload phase:
  python e99_lm_controls.py --phase reload --reload_ckpt /path/ckpt.pt
"""
import os, sys, json, time, math, argparse, subprocess, datetime, traceback

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)

import torch

DATA = '/home/erikg/elman/data/pile.txt'
TOKENIZER = 'p50k_base'
CHUNK = 2048
LN2 = math.log(2.0)
HELDOUT_SEED = 7777     # disjoint from train (42) and round-trip (1234)
ROUNDTRIP_SEED = 1234

# Exact production-path configs, copied verbatim from the E99 sanity driver /
# candidate_configs.json so the controls are byte-identical to the sanity shapes.
CONFIGS = {
    'typed-gdn2-lm': dict(
        level='typed-gdn2-lm', dim=3072, depth=22, n_heads=96, n_state=32,
        expansion=1.0, bf16=False, lr=9.95e-4, knob_lr_mult=1.0, batch_size=2,
        layer_kwargs=dict(
            head_type_logits=[3.9995, -1.9008, -0.9211, -2.8866, 2.4146],
            gdn_allow_neg_eigval=True, lam_max=1.585, beta_max=2.747),
        role='E99 typed Emender (PRIMARY candidate, matched-budget anchor)'),
    'fla-gdn': dict(
        level='fla-gdn', dim=2688, depth=21, n_heads=44, n_state=64,
        expansion=2.0, bf16=True, lr=8.63e-4, knob_lr_mult=1.0, batch_size=2,
        layer_kwargs=None,
        role='CONTROL: dense native GDN-2 / GatedDeltaNet'),
    'e98-cma-lm': dict(
        level='e98-cma-lm', dim=3072, depth=17, n_heads=192, n_state=16,
        expansion=1.0, bf16=False, lr=9.79e-4, knob_lr_mult=5.38, batch_size=1,
        layer_kwargs=dict(
            corner_mixture=[0.4015, 0.2821, 0.0089, 0.3075],
            lam_max=1.585, beta_max=2.747, igain_max=2.0),
        role='CONTROL: E98-CMA unified Emender (cma-capability meta-config)'),
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


def _dataset(seed):
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    return TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1,
                                  seed=seed, tokenizer_name=TOKENIZER)


def loss_on(model, chunks, bf16):
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
    return float(loss.item())


def heldout_eval(model, cfg, vocab_size, device, n_batches=8):
    """Held-out loss on a FIXED slice (disjoint seed) + measured bytes/token.

    Returns (nats_per_token, bytes_per_token, bpb, n_tokens). bytes/token is
    MEASURED by decoding the exact held-out token chunks back to UTF-8 (p50k_base)
    -- not a borrowed constant -- so BPB = (nats/tok)/ln2/(bytes/tok) is honest
    for this slice.
    """
    ds = _dataset(HELDOUT_SEED)
    bs = cfg['batch_size']
    total_nll_nats = 0.0
    total_tokens = 0
    total_bytes = 0
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            chunks, _, _ = ds.get_batch(bs, device=device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=cfg['bf16']):
                loss = model(chunks, return_loss=True)
                if isinstance(loss, tuple):
                    loss = loss[0]
            # mean CE over (bs * CHUNK) predicted tokens
            n_pred = chunks.shape[0] * (chunks.shape[1] - 1)
            total_nll_nats += float(loss.item()) * n_pred
            total_tokens += n_pred
            # measured bytes: decode the *target* tokens (positions 1..CHUNK)
            for row in chunks.tolist():
                tgt = row[1:]
                total_bytes += len(ds.enc.decode(tgt).encode('utf-8'))
    nats_per_token = total_nll_nats / total_tokens
    bytes_per_token = total_bytes / total_tokens
    bpb = (nats_per_token / LN2) / bytes_per_token
    return nats_per_token, bytes_per_token, bpb, total_tokens


def phase_reload(args):
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
    torch.manual_seed(args.seed)
    probe = _dataset(args.seed)
    vocab_size = probe.vocab_size if hasattr(probe, 'vocab_size') else 50281

    stop_reason = None
    try:
        model = build_model(cfg, vocab_size, device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            return _emit_failure(args, cfg, 'OOM_BUILD', str(e))
        raise
    n_params = sum(p.numel() for p in model.parameters())

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
        n_knob = len(knob)
    else:
        groups = model.parameters()
        n_knob = 0
    opt = schedulefree.AdamWScheduleFree(groups, lr=cfg['lr'],
                                         weight_decay=0.01, betas=(0.9, 0.95))

    train_ds = _dataset(args.seed if args.seed else 42)
    bs = cfg['batch_size']

    budget_s = args.train_minutes * 60.0
    walltime_hard_s = budget_s * args.walltime_safety  # hard-stop ceiling

    model.train(); opt.train()
    losses, tok_curve, step_dts = [], [], []
    cumulative_tokens = 0
    nan_seen = False
    t_start = time.time()
    step = 0
    try:
        while True:
            now = time.time()
            elapsed = now - t_start
            if elapsed >= budget_s:
                stop_reason = 'budget_reached'
                break
            if elapsed >= walltime_hard_s:
                stop_reason = 'HARD_STOP_walltime'
                break
            step += 1
            chunks, _, _ = train_ds.get_batch(bs, device=device)
            t0 = time.time()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=cfg['bf16']):
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
            cumulative_tokens += bs * (CHUNK + 1)
            if step > 3:
                step_dts.append(dt)
            tok_curve.append([cumulative_tokens, round(lv, 5), round(elapsed, 1)])
            if not (lv == lv):
                nan_seen = True
                stop_reason = 'HARD_STOP_NaN'
                break
            if step % 25 == 0:
                print(f'[{args.config} s{args.seed}] step {step} | loss {lv:.4f} '
                      f'| {elapsed:.0f}s/{budget_s:.0f}s | tok {cumulative_tokens} '
                      f'| {bs*(CHUNK+1)/dt:.0f} tok/s', flush=True)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return _emit_failure(args, cfg, 'HARD_STOP_OOM', str(e),
                                 partial_losses=losses)
        raise

    walltime = time.time() - t_start
    sustained_tok_s = (sum(t for t in [bs * (CHUNK + 1) / d for d in step_dts])
                       / len(step_dts)) if step_dts else 0.0

    # Fitness (handoff convention): AvgLoss over window, Final loss.
    avg_loss = sum(losses) / len(losses) if losses else None
    final_loss = losses[-1] if losses else None
    tail = max(1, len(losses) // 10)
    late_train_loss = sum(losses[-tail:]) / tail if losses else None

    # Held-out eval (nats/token) + measured BPB.
    ho_nats, ho_bytes_per_tok, ho_bpb, ho_tokens = heldout_eval(
        model, cfg, vocab_size, device)

    # Round-trip: fixed held batch -> save -> fresh process reload.
    rt_ds = _dataset(ROUNDTRIP_SEED)
    held, _, _ = rt_ds.get_batch(bs, device=device)
    l_pre = loss_on(model, held, cfg['bf16'])
    model.train(); opt.train()

    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, f'{args.config}_s{args.seed}_ctrl.pt')
    torch.save({
        'step': step, 'model_state_dict': model.state_dict(), 'cfg': cfg,
        'vocab_size': vocab_size, 'eval_batch': held.detach().cpu(),
        'l_pre': l_pre,
    }, ckpt_path)

    import gc
    del opt
    model.to('cpu'); del model
    gc.collect(); torch.cuda.empty_cache()

    rt = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--phase', 'reload',
         '--reload_ckpt', ckpt_path], env={**os.environ},
        capture_output=True, text=True)
    reload_out = None
    for line in rt.stdout.splitlines():
        if line.startswith('RELOAD_RESULT '):
            reload_out = json.loads(line[len('RELOAD_RESULT '):])
    if reload_out is None:
        print('RELOAD STDERR:\n' + rt.stderr[-2000:], flush=True)
        raise RuntimeError('reload phase produced no result')

    l_post = reload_out['l_post']
    delta = abs(l_post - l_pre)
    roundtrip_ok = (delta < 1e-2 and reload_out['n_missing'] == 0
                    and reload_out['n_unexpected'] == 0)

    def gpu_days(tokens):
        return tokens / sustained_tok_s / 86400 if sustained_tok_s else None

    result = dict(
        config=args.config, level=cfg['level'], role=cfg['role'],
        seed=args.seed, params=n_params, params_b=round(n_params / 1e9, 4),
        dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'],
        n_state=cfg['n_state'], expansion=cfg['expansion'], lr=cfg['lr'],
        knob_lr_mult=cfg['knob_lr_mult'], batch_size=bs, chunk_size=CHUNK,
        dtype=('bf16' if cfg['bf16'] else 'fp32'), n_knob_params=n_knob,
        train_minutes=args.train_minutes, walltime_s=round(walltime, 1),
        steps=step, total_tokens=cumulative_tokens,
        # handoff fitness
        avg_loss=round(avg_loss, 6) if avg_loss is not None else None,
        final_loss=round(final_loss, 6) if final_loss is not None else None,
        late_train_loss=round(late_train_loss, 6) if late_train_loss is not None else None,
        loss_first=round(losses[0], 6) if losses else None,
        # held-out + BPB extension
        heldout_nats_per_token=round(ho_nats, 6),
        heldout_bytes_per_token=round(ho_bytes_per_tok, 4),
        heldout_bpb=round(ho_bpb, 6),
        heldout_tokens=ho_tokens,
        # throughput / stability
        sustained_tok_s=round(sustained_tok_s, 1),
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
        nan_seen=nan_seen, stop_reason=stop_reason,
        # round-trip
        roundtrip_l_pre=round(l_pre, 6), roundtrip_l_post=round(l_post, 6),
        roundtrip_delta=round(delta, 8),
        roundtrip_n_missing=reload_out['n_missing'],
        roundtrip_n_unexpected=reload_out['n_unexpected'],
        roundtrip_ok=roundtrip_ok,
        gpu_days_per_1B_tok=gpu_days(1e9),
        gpu_days_per_10B_tok=gpu_days(10e9),
        loss_curve=tok_curve,   # [cum_tokens, loss, elapsed_s] for token-matched cross-walk
        timestamp=datetime.datetime.utcnow().isoformat() + 'Z',
    )
    out_json = os.path.join(args.outdir, f'{args.config}_s{args.seed}_result.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print('CONTROL_RESULT ' + json.dumps({k: v for k, v in result.items()
                                          if k != 'loss_curve'}), flush=True)

    try:
        os.remove(ckpt_path)   # do NOT publish/stage any checkpoint
    except OSError:
        pass
    return result


def _emit_failure(args, cfg, reason, detail, partial_losses=None):
    os.makedirs(args.outdir, exist_ok=True)
    result = dict(
        config=args.config, level=cfg['level'], role=cfg['role'], seed=args.seed,
        dtype=('bf16' if cfg['bf16'] else 'fp32'),
        train_minutes=args.train_minutes, stop_reason=reason, detail=detail[:500],
        partial_steps=len(partial_losses) if partial_losses else 0,
        partial_final=partial_losses[-1] if partial_losses else None,
        timestamp=datetime.datetime.utcnow().isoformat() + 'Z',
    )
    out_json = os.path.join(args.outdir, f'{args.config}_s{args.seed}_result.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print('CONTROL_FAILURE ' + json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=list(CONFIGS.keys()))
    ap.add_argument('--phase', choices=['run', 'reload'], default='run')
    ap.add_argument('--reload_ckpt', type=str, default=None)
    ap.add_argument('--gpu', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_minutes', type=float, default=15.0)
    ap.add_argument('--walltime_safety', type=float, default=1.15)
    ap.add_argument('--outdir', type=str, default=os.path.join(_THIS, 'results'))
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.phase == 'reload':
        phase_reload(args)
    else:
        try:
            phase_run(args)
        except Exception as e:
            traceback.print_exc()
            _emit_failure(args, CONFIGS[args.config], 'EXCEPTION', repr(e))
            sys.exit(1)


if __name__ == '__main__':
    main()
