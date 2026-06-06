"""One-config 15-min bf16 LM screen worker (redo-e99-1-3b).

Generalizes experiments/e99_1p3b_cma/pilot.py run_one to take a PER-CONFIG
`head_type_logits` (6-type, mixture variable) + shell nonlinearity, so the mixture
is searched/screened on the REAL production 1.3B-class LM path (LadderLM
typed-gdn2-lm, real Pile, schedule-free AdamW, bf16). Records per-eval wallclock,
tok/s, tokens, GPU id, peak mem, stability, held-out BPB, and (optionally) a
fresh-process checkpoint round-trip — the same accounting as the prior E99 batch,
so the corrected batch stays directly comparable.

Usage (one GPU, one config):
  CUDA_VISIBLE_DEVICES=<g> python screen_worker.py --config_json cfg.json --out r.json
"""
import os, sys, json, time, math, argparse, datetime, subprocess, gc

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'experiments', 'e99_1p3b_cma'))

import torch
import pilot as P  # reuse heldout_bpb / loss_on / phase_reload protocol


def build_model(cfg, vocab_size, device):
    from ndm.models.ladder_lm import LadderLM
    lk = dict(head_type_logits=cfg['head_type_logits'],
              gdn_allow_neg_eigval=True, lam_max=1.585, beta_max=2.747)
    if cfg.get('shell_state_nonlin'):
        lk['shell_state_nonlin'] = cfg['shell_state_nonlin']
    if cfg.get('shell_state_chunk'):
        lk['shell_state_chunk'] = int(cfg['shell_state_chunk'])
    m = LadderLM(vocab_size=vocab_size, dim=cfg['dim'], depth=cfg['depth'],
                 level='typed-gdn2-lm', n_heads=cfg['n_heads'], n_state=cfg['n_state'],
                 expansion=cfg.get('expansion', 1.0), layer_kwargs=lk).to(device)
    if cfg.get('bf16'):
        m = m.bfloat16()
    return m


def reload_worker(ckpt_path):
    """Fresh-process reload: rebuild from saved cfg, reproduce held-batch loss."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['cfg']
    model = build_model(cfg, ckpt['vocab_size'], 'cuda')
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    chunks = ckpt['eval_batch'].to('cuda')
    print('RELOAD_RESULT ' + json.dumps(dict(
        l_post=P.loss_on(model, chunks, cfg.get('bf16', False)),
        n_missing=len(missing), n_unexpected=len(unexpected))))


def head_counts(logits, n_heads):
    from ndm.models.typed_head_mixture import allocate_types
    return allocate_types(n_heads, logits)['counts']


def param_target_status(n_params, cfg):
    target = cfg.get('param_target')
    if target is None:
        return None
    target = int(target)
    tolerance = float(cfg.get('param_tolerance', 0.02))
    rel = (int(n_params) - target) / float(target)
    return dict(
        param_target=target,
        param_tolerance=tolerance,
        param_target_rel_error=rel,
        param_target_error_pct=round(rel * 100.0, 4),
        param_target_within_tolerance=abs(rel) <= tolerance,
        driver_model_params=cfg.get('model_params'),
        driver_model_params_match=(
            int(cfg['model_params']) == int(n_params)
            if cfg.get('model_params') is not None else None
        ),
    )


def assert_param_target(n_params, cfg):
    status = param_target_status(n_params, cfg)
    if status is None:
        return None
    if not status['param_target_within_tolerance']:
        raise ValueError(
            f"{cfg.get('name', '<unnamed>')} params={int(n_params):,} "
            f"target={status['param_target']:,} "
            f"err={status['param_target_error_pct']:.3f}% exceeds "
            f"+/-{status['param_tolerance'] * 100:.1f}%")
    return status


def run_one(cfg, args, device='cuda'):
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    import schedulefree
    torch.manual_seed(0)
    probe = TokenizedStreamDataset(data_path=P.DATA, chunk_size=P.CHUNK + 1, seed=0,
                                   tokenizer_name=P.TOKENIZER)
    vocab_size = getattr(probe, 'vocab_size', 50281)
    model = build_model(cfg, vocab_size, device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    param_status = assert_param_target(n_params, cfg) or {}
    if param_status:
        print(f"PARAM_CHECK {cfg['name']} params={n_params:,} "
              f"target={param_status['param_target']:,} "
              f"err={param_status['param_target_error_pct']:+.3f}% "
              f"driver_match={param_status['driver_model_params_match']}",
              flush=True)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg['lr'],
                                         weight_decay=0.01, betas=(0.9, 0.95))
    train_ds = TokenizedStreamDataset(data_path=P.DATA, chunk_size=P.CHUNK + 1, seed=42,
                                      tokenizer_name=P.TOKENIZER)
    bs = args.batch_size
    model.train(); opt.train()
    losses, step_dts, grad_norms = [], [], []
    nan_seen = nonfinite_grad_seen = False
    stop_reason = None
    t_start = time.time()
    step = 0
    wall_cap = args.wall_minutes * 60
    while step < args.max_steps and (time.time() - t_start) < wall_cap:
        step += 1
        chunks, _, _ = train_ds.get_batch(bs, device=device)
        t0 = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg.get('bf16', False)):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
        if not torch.isfinite(loss):
            nan_seen = True; stop_reason = 'nonfinite_loss'
            losses.append(float(loss.item())); break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(gnorm)):
            nonfinite_grad_seen = True; stop_reason = 'nonfinite_grad'
            opt.zero_grad(set_to_none=True); break
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        losses.append(float(loss.item()))
        grad_norms.append(float(gnorm.item() if hasattr(gnorm, 'item') else gnorm))
        if step > 3:
            step_dts.append(dt)
        if step % 100 == 0:
            print(f"  step {step} loss {losses[-1]:.4f} "
                  f"elapsed {(time.time()-t_start)/60:.1f}m", flush=True)
    if stop_reason is None:
        stop_reason = 'step_cap' if step >= args.max_steps else 'wall_cap'
    avg_loss = sum(losses) / len(losses) if losses else float('inf')
    final_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float('inf')
    wall_min = (time.time() - t_start) / 60.0
    tok_s = (sum(bs * (P.CHUNK + 1) / dt for dt in step_dts) / len(step_dts)
             if step_dts else 0.0)

    bpb = P.heldout_bpb(model, vocab_size, device, max_batches=args.bpb_batches,
                        bs=bs, bf16=cfg.get('bf16', False))

    roundtrip = None
    if args.roundtrip and stop_reason in ('step_cap', 'wall_cap'):
        model.eval(); opt.eval()
        held = TokenizedStreamDataset(data_path=P.DATA, chunk_size=P.CHUNK + 1,
                                      seed=P.ROUNDTRIP_SEED, tokenizer_name=P.TOKENIZER)
        held_chunks, _, _ = held.get_batch(bs, device=device)
        l_pre = P.loss_on(model, held_chunks, cfg.get('bf16', False))
        os.makedirs(args.outdir, exist_ok=True)
        ckpt_path = os.path.join(args.outdir, f"_rt_{cfg['name']}.pt")
        torch.save(dict(step=step, model_state_dict=model.state_dict(), cfg=cfg,
                        vocab_size=vocab_size, eval_batch=held_chunks.detach().cpu(),
                        l_pre=l_pre), ckpt_path)
        del opt, model
        gc.collect(); torch.cuda.empty_cache()
        rt = subprocess.run([sys.executable, os.path.abspath(__file__),
                             '--phase', 'reload', '--reload_ckpt', ckpt_path],
                            capture_output=True, text=True)
        reload_out = None
        for line in rt.stdout.splitlines():
            if line.startswith('RELOAD_RESULT '):
                reload_out = json.loads(line[len('RELOAD_RESULT '):])
        try:
            os.remove(ckpt_path)
        except OSError:
            pass
        if reload_out is not None:
            delta = abs(reload_out['l_post'] - l_pre)
            roundtrip = dict(l_pre=round(l_pre, 6), l_post=round(reload_out['l_post'], 6),
                             delta=round(delta, 8), n_missing=reload_out['n_missing'],
                             n_unexpected=reload_out['n_unexpected'],
                             ok=(delta < 1e-2 and reload_out['n_missing'] == 0
                                 and reload_out['n_unexpected'] == 0))
        else:
            roundtrip = dict(error=rt.stderr[-800:])

    result = dict(
        name=cfg['name'], head_type_logits=cfg['head_type_logits'],
        shell_state_nonlin=cfg.get('shell_state_nonlin'),
        counts=head_counts(cfg['head_type_logits'], cfg['n_heads']),
        dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'], n_state=cfg['n_state'],
        lr=cfg['lr'], batch_size=bs, dtype=('bf16' if cfg.get('bf16') else 'fp32'),
        model_params=n_params, params_b=round(n_params / 1e9, 4),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        steps=step, tokens=step * bs * (P.CHUNK + 1), wall_minutes=round(wall_min, 3),
        tok_s=round(tok_s, 1), avg_loss=round(avg_loss, 5), final_loss=round(final_loss, 5),
        stop_reason=stop_reason, nan_seen=nan_seen, nonfinite_grad_seen=nonfinite_grad_seen,
        finite_losses=all(math.isfinite(x) for x in losses) if losses else False,
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
        heldout=bpb, roundtrip=roundtrip,
        timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    result.update(param_status)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['run', 'reload'], default='run')
    ap.add_argument('--reload_ckpt')
    ap.add_argument('--config_json')
    ap.add_argument('--out')
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--wall_minutes', type=float, default=15.0)
    ap.add_argument('--max_steps', type=int, default=100000)
    ap.add_argument('--bpb_batches', type=int, default=40)
    ap.add_argument('--roundtrip', type=int, default=0)
    ap.add_argument('--outdir', default=os.path.join(_THIS, 'results'))
    args = ap.parse_args()
    if args.phase == 'reload':
        reload_worker(args.reload_ckpt)
        return
    with open(args.config_json) as f:
        cfg = json.load(f)
    r = run_one(cfg, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(r, f, indent=2, default=str)
    print('SCREEN_RESULT ' + json.dumps(r, default=str), flush=True)


if __name__ == '__main__':
    main()
