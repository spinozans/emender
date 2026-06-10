#!/usr/bin/env python3
"""opt-1p3b — 1.3B matched-compute LM held-out BPB head-to-head (OPT_SYNTHESIS §4.5.2).

Three arms, each at its ~1.3B geometry, trained WALLCLOCK-MATCHED on REAL Comma-Pile
mainmix tokens (fused Triton), measured held-out BPB + token-matched loss curve:

  rstar     typed-gdn2-lm  d2688 dep21 h44 ns64 exp1.5  (1.29B, fp32 FUSED)
            house mixture 22 gdn2_recall / 11 nonlin(unified) / 11 refit(mom-off=count)
            + per-head-type LR lever head_lr_compute_mult=5 (the surviving R* lever;
            decay_init dropped per §3.3 re-run). gdn_allow_neg_eigval=1.
  cma_gdn2  fla-gdn        d2688 dep21 h44 ns64 exp2.0  (1.35B, bf16)
            = CMA-best GDN-2 geometry (hf_v03_fix_staging/gdn-1.3b). The incumbent B.
  cma_m2rnn m2rnn          d1920 dep21 h370 ns16 exp1.0 (1.31B, bf16)
            = CMA-best m2rnn geometry (hf_v03_fix_staging/m2rnn-cma-1.3b), sigmoid gate,
            linear_state=False (the raw-write power-separation foil).

FUSED-ONLY MANDATE (Erik): the typed arm runs the fused Triton path (use_triton_e97=True,
loud guard active). We assert it is on and FAIL LOUD otherwise. m2rnn uses its XMA Triton
backend when available; gdn2 uses FLA chunked.

Method mirrors experiments/e99_1p3b_controls/e99_lm_controls.py (matched-wallclock,
NaN/OOM hard-stop, token curve for token-matched cross-walk, measured-bytes BPB,
fresh-process round-trip). REAL data, real fwd/bwd, schedule-free AdamW.

Usage (one arm, one seed):
  eval "$(scripts/gpu_lease.sh 1)"
  python lm_runner.py --arm rstar --seed 0 --train_minutes 20
Reload phase (internal):
  python lm_runner.py --phase reload --reload_ckpt /path/ckpt.pt
"""
import os, sys, json, time, math, argparse, subprocess, datetime, traceback

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)

# FUSED mandate for m2rnn: the XMA Triton backend must be importable BEFORE
# m2rnn_baseline is first imported (its XMA_M2RNN_AVAILABLE flag is set at import).
os.environ.setdefault('XMA_PATH', '/home/erikg/xma')

import torch

DATA = '/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt'
TOKENIZER = 'p50k_base'
CHUNK = 2048
LN2 = math.log(2.0)
HELDOUT_SEED = 7777      # disjoint from train (42) and round-trip (1234)
ROUNDTRIP_SEED = 1234


def _house_logits(n_heads):
    """House placement: n/2 gdn2_recall + n/4 nonlin(unified) + n/4 refit(count)."""
    g = n_heads // 2
    nl = n_heads // 4
    rf = n_heads - g - nl
    # TYPE_NAMES = [gdn2_recall, e97_track, count, latch, nonlin, gdn2_nonlin_shell,
    #               e97_raw, e97_delta, refit]
    return [math.log(g), -30, -30, -30, math.log(nl), -30, -30, -30, math.log(rf)]


CONFIGS = {
    # R* geometry: the fused unified(nonlin) kernel requires n_state*expansion<=64, so
    # the typed mixture cannot use the gdn2 control's exact ns64/exp2.0 (V=128). We
    # param-match instead: d3072/dep22/h64/ns32/exp2.0 = 1.34B ~ the 1.35B gdn2 control,
    # with V=ns*exp=64 (valid). gdn2 keeps its CMA-best geometry; both sit at ~1.34B.
    'rstar': dict(
        builder='typed', level='typed-gdn2-lm', dim=3072, depth=22, n_heads=64,
        n_state=32, expansion=2.0, bf16=False, lr=8.0e-4,
        head_lr_compute_mult=5.0, head_lr_recall_mult=1.0, batch_size=1,
        layer_kwargs=dict(head_type_logits=_house_logits(64),
                          gdn_allow_neg_eigval=True, refit_has_mom=0),
        role='R* — optimized GDN+nonlin mixture (house + head_lr_compute_mult=5), fp32 FUSED'),
    'cma_gdn2': dict(
        builder='ladder', level='fla-gdn', dim=2688, depth=21, n_heads=44,
        n_state=64, expansion=2.0, bf16=True, lr=8.63e-4,
        head_lr_compute_mult=1.0, head_lr_recall_mult=1.0, batch_size=2, layer_kwargs=None,
        role='CONTROL B = CMA-best GDN-2 (gdn-1.3b geometry), bf16 FLA-fused'),
    'cma_m2rnn': dict(
        builder='m2rnn', level='m2rnn', dim=1920, depth=21, n_heads=370,
        n_state=16, expansion=1.0, bf16=True, lr=6.0e-4,
        head_lr_compute_mult=1.0, head_lr_recall_mult=1.0,
        gate_activation='sigmoid', linear_state=False, use_residual=True,
        use_conv=False, d_conv=4, batch_size=2, layer_kwargs=None,
        role='CONTROL = CMA-best m2rnn (m2rnn-cma-1.3b geometry), raw-write foil'),
}


def build_model(cfg, vocab_size, device):
    if cfg['builder'] == 'm2rnn':
        from ndm.models.m2rnn_baseline import M2RNNLM, XMA_M2RNN_AVAILABLE
        print(f"M2RNN XMA Triton backend available: {XMA_M2RNN_AVAILABLE}", flush=True)
        if not XMA_M2RNN_AVAILABLE:
            raise RuntimeError("FUSED-ASSERT: m2rnn XMA Triton backend unavailable "
                               "(set XMA_PATH=/home/erikg/xma) — eager fallback forbidden")
        m = M2RNNLM(
            vocab_size=vocab_size, dim=cfg['dim'], depth=cfg['depth'],
            n_heads=cfg['n_heads'], n_state=cfg['n_state'], expansion=cfg['expansion'],
            use_gate=True, use_residual=cfg['use_residual'], use_conv=cfg['use_conv'],
            d_conv=cfg['d_conv'], linear_state=cfg['linear_state'])
    else:
        from ndm.models.ladder_lm import LadderLM
        m = LadderLM(
            vocab_size=vocab_size, dim=cfg['dim'], depth=cfg['depth'],
            level=cfg['level'], n_heads=cfg['n_heads'], n_state=cfg['n_state'],
            expansion=cfg['expansion'], layer_kwargs=cfg['layer_kwargs'])
    m = m.to(device)
    if cfg['bf16']:
        m = m.bfloat16()
    return m


def assert_fused_typed(model, cfg):
    """FUSED-ONLY mandate: the typed arm MUST run the fused Triton e97 path. Verify the
    TypedHeadMixtureLayer inner modules carry use_triton_e97=True; fail loud otherwise."""
    if cfg['builder'] != 'typed':
        return None
    n_typed = 0
    bad = []
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == 'TypedHeadMixtureLayer':
            n_typed += 1
            ute = getattr(mod, 'use_triton_e97', None)
            if ute is not True:
                bad.append((name, ute))
    if n_typed == 0:
        raise RuntimeError("FUSED-ASSERT: no TypedHeadMixtureLayer found in rstar model")
    if bad:
        raise RuntimeError(f"FUSED-ASSERT: use_triton_e97 not True on {len(bad)} typed "
                           f"layers (eager fallback forbidden): {bad[:3]}")
    print(f"FUSED-ASSERT OK: {n_typed} TypedHeadMixtureLayer(s), use_triton_e97=True", flush=True)
    return n_typed


def _typed_head_class(name):
    """LadderLM-aware: classify a typed-mixture parameter by the child module after
    'inner' (layers.N.inner.<child>). gdn=recall-class; unified/shell/e97_raw/
    e97_delta/refit=compute-class. Mirrors train_hybrid.py:_typed_head_class adapted
    to the LadderLM module tree."""
    toks = name.split('.')
    if 'inner' in toks:
        i = toks.index('inner')
        if i + 1 < len(toks):
            child = toks[i + 1]
            if child == 'gdn':
                return 'recall'
            if child in ('unified', 'shell', 'e97_raw', 'e97_delta', 'refit'):
                return 'compute'
    return None


def build_optimizer(model, cfg):
    import schedulefree
    use_headlr = (cfg['head_lr_recall_mult'] != 1.0 or cfg['head_lr_compute_mult'] != 1.0)
    n_recall = n_compute = 0
    if use_headlr:
        recall_p, compute_p, base_p = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            c = _typed_head_class(name)
            (recall_p if c == 'recall' else compute_p if c == 'compute' else base_p).append(p)
        groups = [{'params': base_p, 'lr': cfg['lr']}]
        if recall_p:
            groups.append({'params': recall_p, 'lr': cfg['lr'] * cfg['head_lr_recall_mult']})
        if compute_p:
            groups.append({'params': compute_p, 'lr': cfg['lr'] * cfg['head_lr_compute_mult']})
        n_recall, n_compute = len(recall_p), len(compute_p)
        print(f"Head-type LR groups: recall {n_recall}@{cfg['lr']*cfg['head_lr_recall_mult']:.2e} "
              f"compute {n_compute}@{cfg['lr']*cfg['head_lr_compute_mult']:.2e} "
              f"base {len(base_p)}@{cfg['lr']:.2e}", flush=True)
        if n_compute == 0:
            raise RuntimeError("Head-LR lever requested but 0 compute-class params matched")
    else:
        groups = model.parameters()
    opt = schedulefree.AdamWScheduleFree(groups, lr=cfg['lr'], weight_decay=0.01,
                                         betas=(0.9, 0.95))
    return opt, n_recall, n_compute


def _dataset(seed):
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    return TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1, seed=seed,
                                  tokenizer_name=TOKENIZER)


def loss_on(model, chunks, bf16):
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
    return float(loss.item())


def heldout_eval(model, cfg, device, n_batches=8):
    ds = _dataset(HELDOUT_SEED)
    bs = cfg['batch_size']
    total_nll = total_tok = total_bytes = 0
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            chunks, _, _ = ds.get_batch(bs, device=device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
                loss = model(chunks, return_loss=True)
                if isinstance(loss, tuple):
                    loss = loss[0]
            n_pred = chunks.shape[0] * (chunks.shape[1] - 1)
            total_nll += float(loss.item()) * n_pred
            total_tok += n_pred
            for row in chunks.tolist():
                total_bytes += len(ds.enc.decode(row[1:]).encode('utf-8'))
    nats = total_nll / total_tok
    bpt = total_bytes / total_tok
    return nats, bpt, (nats / LN2) / bpt, total_tok


def phase_reload(args):
    ckpt = torch.load(args.reload_ckpt, map_location='cpu')
    cfg = ckpt['cfg']
    model = build_model(cfg, ckpt['vocab_size'], 'cuda')
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    l_post = loss_on(model, ckpt['eval_batch'].to('cuda'), cfg['bf16'])
    out = dict(l_post=l_post, n_missing=len(missing), n_unexpected=len(unexpected))
    print('RELOAD_RESULT ' + json.dumps(out))
    return out


def _emit_failure(args, cfg, reason, detail, partial=None):
    os.makedirs(args.outdir, exist_ok=True)
    result = dict(arm=args.arm, level=cfg['level'], role=cfg['role'], seed=args.seed,
                  dtype='bf16' if cfg['bf16'] else 'fp32', train_minutes=args.train_minutes,
                  stop_reason=reason, detail=str(detail)[:600],
                  partial_steps=len(partial) if partial else 0,
                  partial_final=partial[-1] if partial else None,
                  timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    with open(os.path.join(args.outdir, f'{args.arm}_s{args.seed}_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print('ARM_FAILURE ' + json.dumps(result), flush=True)
    return result


def phase_run(args):
    cfg = dict(CONFIGS[args.arm])
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    device = 'cuda'
    torch.manual_seed(args.seed)
    probe = _dataset(args.seed)
    vocab_size = probe.vocab_size

    try:
        model = build_model(cfg, vocab_size, device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            return _emit_failure(args, cfg, 'OOM_BUILD', e)
        raise
    n_typed = assert_fused_typed(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    opt, n_recall, n_compute = build_optimizer(model, cfg)

    train_ds = _dataset(args.seed if args.seed else 42)
    bs = cfg['batch_size']
    budget_s = args.train_minutes * 60.0
    hard_s = budget_s * args.walltime_safety

    model.train(); opt.train()
    losses, tok_curve, step_dts = [], [], []
    cum_tokens = 0
    nan_seen = False
    stop_reason = None
    t_start = time.time()
    step = 0
    try:
        while True:
            elapsed = time.time() - t_start
            if elapsed >= budget_s:
                stop_reason = 'budget_reached'; break
            if elapsed >= hard_s:
                stop_reason = 'HARD_STOP_walltime'; break
            step += 1
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
            cum_tokens += bs * (CHUNK + 1)
            if step > 3:
                step_dts.append(dt)
            tok_curve.append([cum_tokens, round(lv, 5), round(elapsed, 1)])
            if not (lv == lv):
                nan_seen = True; stop_reason = 'HARD_STOP_NaN'; break
            if step % 25 == 0:
                print(f'[{args.arm} s{args.seed}] step {step} | loss {lv:.4f} | '
                      f'{elapsed:.0f}/{budget_s:.0f}s | tok {cum_tokens} | '
                      f'{bs*(CHUNK+1)/dt:.0f} tok/s', flush=True)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return _emit_failure(args, cfg, 'HARD_STOP_OOM', e, partial=losses)
        raise

    walltime = time.time() - t_start
    sustained = (sum(bs * (CHUNK + 1) / d for d in step_dts) / len(step_dts)) if step_dts else 0.0
    avg_loss = sum(losses) / len(losses) if losses else None
    tail = max(1, len(losses) // 10)
    late = sum(losses[-tail:]) / tail if losses else None

    if nan_seen or not losses:
        return _emit_failure(args, cfg, stop_reason or 'NO_STEPS', 'nan_or_empty', partial=losses)

    ho_nats, ho_bpt, ho_bpb, ho_tok = heldout_eval(model, cfg, device)

    # round-trip in a fresh process
    rt_ds = _dataset(ROUNDTRIP_SEED)
    held, _, _ = rt_ds.get_batch(bs, device=device)
    l_pre = loss_on(model, held, cfg['bf16'])
    model.train(); opt.train()
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, f'{args.arm}_s{args.seed}_ckpt.pt')
    torch.save({'model_state_dict': model.state_dict(), 'cfg': cfg,
                'vocab_size': vocab_size, 'eval_batch': held.detach().cpu()}, ckpt_path)
    import gc
    del opt; model.to('cpu'); del model; gc.collect(); torch.cuda.empty_cache()
    rt = subprocess.run([sys.executable, os.path.abspath(__file__), '--phase', 'reload',
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
    roundtrip_ok = (delta < 1e-2 and reload_out['n_missing'] == 0 and reload_out['n_unexpected'] == 0)

    result = dict(
        arm=args.arm, level=cfg['level'], role=cfg['role'], seed=args.seed,
        params=n_params, params_b=round(n_params / 1e9, 4),
        dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'], n_state=cfg['n_state'],
        expansion=cfg['expansion'], lr=cfg['lr'],
        head_lr_compute_mult=cfg['head_lr_compute_mult'], batch_size=bs, chunk_size=CHUNK,
        dtype='bf16' if cfg['bf16'] else 'fp32', fused_typed_layers=n_typed,
        n_recall_params=n_recall, n_compute_params=n_compute,
        train_minutes=args.train_minutes, walltime_s=round(walltime, 1), steps=step,
        total_tokens=cum_tokens, avg_loss=round(avg_loss, 6), late_train_loss=round(late, 6),
        final_loss=round(losses[-1], 6), loss_first=round(losses[0], 6),
        heldout_nats_per_token=round(ho_nats, 6), heldout_bytes_per_token=round(ho_bpt, 4),
        heldout_bpb=round(ho_bpb, 6), heldout_tokens=ho_tok,
        sustained_tok_s=round(sustained, 1),
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
        nan_seen=nan_seen, stop_reason=stop_reason,
        roundtrip_l_pre=round(l_pre, 6), roundtrip_l_post=round(l_post, 6),
        roundtrip_delta=round(delta, 8), roundtrip_ok=roundtrip_ok,
        loss_curve=tok_curve, timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    with open(os.path.join(args.outdir, f'{args.arm}_s{args.seed}_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print('ARM_RESULT ' + json.dumps({k: v for k, v in result.items() if k != 'loss_curve'}), flush=True)
    try:
        os.remove(ckpt_path)
    except OSError:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(CONFIGS.keys()))
    ap.add_argument('--phase', choices=['run', 'reload'], default='run')
    ap.add_argument('--reload_ckpt', type=str, default=None)
    ap.add_argument('--gpu', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_minutes', type=float, default=20.0)
    ap.add_argument('--batch_size', type=int, default=None,
                    help='override per-arm default batch size')
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
            _emit_failure(args, CONFIGS[args.arm], 'EXCEPTION', repr(e))
            sys.exit(1)


if __name__ == '__main__':
    main()
