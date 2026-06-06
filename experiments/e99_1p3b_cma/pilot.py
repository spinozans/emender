#!/usr/bin/env python3
"""Bounded longer pilot + round-trip + held-out BPB for promoted E99 configs.

This is a RANK-STABILITY check, NOT a full run. For each promoted config it runs
the REAL production LM path (LadderLM typed-gdn2-lm, real Pile, schedule-free
AdamW, fp32 by default; optional bf16 for the E99 follow-up) for an
EXPLICITLY-BOUNDED budget recorded before launch:
  - step cap  = --pilot_mult x short-run steps (short-run ~532 steps @ bs2/ctx2048/15min)
  - walltime cap = --pilot_wall_minutes (hard ceiling)
whichever is hit first.

Then, per config:
  - AvgLoss (mean over the pilot window) + Final (last-100 avg) -> rank vs short-run
  - ROUND-TRIP: l_pre on a fixed held batch (in-memory) -> save (production
    model_state_dict convention) -> reload in a FRESH PROCESS -> l_post; a reload
    that does not reproduce l_pre within tol is a HARD BLOCKER (the wire-e99-e98
    PILE_BPB lesson: 0-missing/0-unexpected key match is NOT sufficient).
  - HELD-OUT BPB on the canonical Pile held-out byte slice (tokenizer-invariant).

No mock data; the held batch and held-out slice are real Pile bytes. Generated
checkpoints are deleted after the round-trip (task constraint: never publish).
"""
import os, sys, json, time, argparse, subprocess, datetime, math, hashlib

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'experiments', 'e99_1p3b_sanity'))

import torch

DATA = '/home/erikg/elman/data/pile.txt'
TOKENIZER = 'p50k_base'
CHUNK = 2048
ROUNDTRIP_SEED = 1234
HELDOUT_SLICE = '/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt'
HELDOUT_SLICE_SHA256 = '3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a'

# Fixed typed-gdn2-lm architecture identity (same as the search).
LOGITS = [3.9995, -1.9008, -0.9211, -2.8866, 2.4146]
LAYER_KWARGS = dict(head_type_logits=LOGITS, gdn_allow_neg_eigval=True,
                    lam_max=1.585, beta_max=2.747)


def cfg_from_params(p, bf16=False):
    return dict(level='typed-gdn2-lm', dim=int(p['dim']), depth=int(p['depth']),
                n_heads=int(p['n_heads']), n_state=int(p['n_state']),
                expansion=1.0, bf16=bool(bf16), lr=float(p['lr']), knob_lr_mult=1.0,
                layer_kwargs=dict(LAYER_KWARGS))


def build_model(cfg, vocab_size, device):
    from ndm.models.ladder_lm import LadderLM
    m = LadderLM(vocab_size=vocab_size, dim=cfg['dim'], depth=cfg['depth'],
                 level=cfg['level'], n_heads=cfg['n_heads'], n_state=cfg['n_state'],
                 expansion=cfg['expansion'], layer_kwargs=cfg['layer_kwargs'])
    m = m.to(device)
    if cfg['bf16']:
        m = m.bfloat16()
    return m


def loss_on(model, chunks, bf16):
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
    return float(loss.item())


def _heldout_slice_info():
    if not os.path.exists(HELDOUT_SLICE):
        return None
    h = hashlib.sha256()
    with open(HELDOUT_SLICE, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    size = os.path.getsize(HELDOUT_SLICE)
    sha = h.hexdigest()
    return dict(path=HELDOUT_SLICE, bytes=size, sha256=sha,
                sha256_ok=(sha == HELDOUT_SLICE_SHA256))


def heldout_bpb(model, vocab_size, device, max_batches=40, bs=2, bf16=False):
    """Mean held-out loss (nats/token) on the slice -> BPB via tokens/byte."""
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    slice_info = _heldout_slice_info()
    if slice_info is None:
        return None
    ds = TokenizedStreamDataset(data_path=HELDOUT_SLICE, chunk_size=CHUNK + 1,
                                seed=7, tokenizer_name=TOKENIZER)
    import tiktoken
    enc = tiktoken.get_encoding(TOKENIZER)
    nbytes = os.path.getsize(HELDOUT_SLICE)
    with open(HELDOUT_SLICE, 'rb') as f:
        sample = f.read(2_000_000)
    ntok = len(enc.encode(sample.decode('utf-8', errors='ignore')))
    tokens_per_byte = ntok / len(sample)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(max_batches):
            chunks, _, _ = ds.get_batch(bs, device=device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16):
                l = model(chunks, return_loss=True)
                if isinstance(l, tuple):
                    l = l[0]
            losses.append(float(l.item()))
    mean_nats = sum(losses) / len(losses)
    bpb = mean_nats * tokens_per_byte / math.log(2)
    return dict(heldout_nats_per_token=round(mean_nats, 5),
                tokens_per_byte=round(tokens_per_byte, 5),
                heldout_bpb=round(bpb, 5), n_batches=len(losses),
                slice=slice_info)


def phase_reload(args):
    ckpt = torch.load(args.reload_ckpt, map_location='cpu')
    cfg = ckpt['cfg']
    model = build_model(cfg, ckpt['vocab_size'], 'cuda')
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    chunks = ckpt['eval_batch'].to('cuda')
    print('RELOAD_RESULT ' + json.dumps(dict(
        l_post=loss_on(model, chunks, cfg['bf16']),
        n_missing=len(missing), n_unexpected=len(unexpected))))


def run_one(p, args, device='cuda'):
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    import schedulefree
    cfg = cfg_from_params(p, bf16=args.bf16)
    torch.manual_seed(0)
    probe = TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1, seed=0,
                                   tokenizer_name=TOKENIZER)
    vocab_size = getattr(probe, 'vocab_size', 50281)
    model = build_model(cfg, vocab_size, device)
    n_params = sum(p_.numel() for p_ in model.parameters())

    KNOB = ('lam_raw', 'beta_raw', 'igain_raw', 'gamma_raw')
    knob, base = [], []
    for name, pr in model.named_parameters():
        (knob if any(name.endswith(s) for s in KNOB) else base).append(pr)
    groups = model.parameters()  # knob_lr_mult=1.0 -> single group
    opt = schedulefree.AdamWScheduleFree(groups, lr=cfg['lr'], weight_decay=0.01,
                                         betas=(0.9, 0.95))
    train_ds = TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1, seed=42,
                                      tokenizer_name=TOKENIZER)
    bs = args.batch_size
    model.train(); opt.train()
    losses = []
    step_dts = []
    grad_norms = []
    nan_seen = False
    nonfinite_grad_seen = False
    stop_reason = None
    t_start = time.time()
    step = 0
    wall_cap = args.pilot_wall_minutes * 60
    while step < args.pilot_steps and (time.time() - t_start) < wall_cap:
        step += 1
        chunks, _, _ = train_ds.get_batch(bs, device=device)
        t0 = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
        if not torch.isfinite(loss):
            nan_seen = True
            stop_reason = 'nonfinite_loss'
            losses.append(float(loss.item()))
            print(f"  [{p['dim']}/{p['depth']}/{p['n_heads']}] non-finite loss "
                  f"at step {step}: {loss.item()}", flush=True)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            nonfinite_grad_seen = True
            stop_reason = 'nonfinite_grad'
            opt.zero_grad(set_to_none=True)
            print(f"  [{p['dim']}/{p['depth']}/{p['n_heads']}] non-finite grad "
                  f"at step {step}: {grad_norm}", flush=True)
            break
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        lv = float(loss.item())
        losses.append(lv)
        grad_norms.append(float(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm))
        if step > 3:
            step_dts.append(dt)
        if step % 50 == 0:
            print(f"  [{p['dim']}/{p['depth']}/{p['n_heads']}] step {step} "
                  f"loss {lv:.4f} elapsed {(time.time()-t_start)/60:.1f}m", flush=True)
    if stop_reason is None:
        stop_reason = 'step_cap' if step >= args.pilot_steps else 'wall_cap'
    avg_loss = sum(losses) / len(losses) if losses else float('inf')
    final_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float('inf')
    wall_min = (time.time() - t_start) / 60.0
    sustained_tok_s = (
        sum(bs * (CHUNK + 1) / dt for dt in step_dts) / len(step_dts)
        if step_dts else 0.0
    )

    # held-out BPB
    bpb = heldout_bpb(model, vocab_size, device, max_batches=args.bpb_batches,
                      bs=bs, bf16=cfg['bf16'])

    # round-trip
    model.eval(); opt.eval()
    held = TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1,
                                  seed=ROUNDTRIP_SEED, tokenizer_name=TOKENIZER)
    held_chunks, _, _ = held.get_batch(bs, device=device)
    l_pre = loss_on(model, held_chunks, cfg['bf16'])
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{p['dim']}_{p['depth']}_{p['n_heads']}_{p['n_state']}"
    ckpt_path = os.path.join(args.outdir, f'pilot_{tag}.pt')
    torch.save(dict(step=step, model_state_dict=model.state_dict(), cfg=cfg,
                    vocab_size=vocab_size, eval_batch=held_chunks.detach().cpu(),
                    l_pre=l_pre), ckpt_path)
    import gc
    del opt, model
    gc.collect(); torch.cuda.empty_cache()
    rt = subprocess.run([sys.executable, os.path.abspath(__file__), '--phase', 'reload',
                         '--reload_ckpt', ckpt_path], capture_output=True, text=True)
    reload_out = None
    for line in rt.stdout.splitlines():
        if line.startswith('RELOAD_RESULT '):
            reload_out = json.loads(line[len('RELOAD_RESULT '):])
    if reload_out is None:
        print('RELOAD STDERR:\n' + rt.stderr[-1500:])
        raise RuntimeError('reload produced no result')
    delta = abs(reload_out['l_post'] - l_pre)
    roundtrip_ok = (delta < 1e-2 and reload_out['n_missing'] == 0 and reload_out['n_unexpected'] == 0)
    try:
        os.remove(ckpt_path)
    except OSError:
        pass

    return dict(
        params=p, level='typed-gdn2-lm', model_params=n_params,
        dtype=('bf16' if cfg['bf16'] else 'fp32'),
        params_b=round(n_params / 1e9, 4), pilot_steps=step, batch_size=bs,
        pilot_tokens=step * bs * (CHUNK + 1), pilot_wall_minutes=round(wall_min, 2),
        avg_loss=round(avg_loss, 5), final_loss=round(final_loss, 5),
        stop_reason=stop_reason,
        nan_seen=nan_seen, nonfinite_grad_seen=nonfinite_grad_seen,
        finite_losses=all(math.isfinite(x) for x in losses),
        finite_grad_norms=all(math.isfinite(x) for x in grad_norms),
        sustained_tok_s=round(sustained_tok_s, 1),
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
        heldout=bpb,
        roundtrip_l_pre=round(l_pre, 6), roundtrip_l_post=round(reload_out['l_post'], 6),
        roundtrip_delta=round(delta, 8), roundtrip_ok=roundtrip_ok,
        roundtrip_n_missing=reload_out['n_missing'],
        roundtrip_n_unexpected=reload_out['n_unexpected'],
        timestamp=datetime.datetime.utcnow().isoformat() + 'Z')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['run', 'reload'], default='run')
    ap.add_argument('--reload_ckpt')
    ap.add_argument('--configs_json', help='JSON list of param dicts (promoted configs)')
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--pilot_steps', type=int, default=1596)   # 3x ~532 short-run steps
    ap.add_argument('--pilot_wall_minutes', type=float, default=45.0)
    ap.add_argument('--bpb_batches', type=int, default=40)
    ap.add_argument('--bf16', action='store_true',
                    help='Run the typed pilot with production bf16 params + CUDA autocast.')
    ap.add_argument('--outdir', default=os.path.join(_THIS, 'pilot_results'))
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.phase == 'reload':
        phase_reload(args)
        return
    with open(args.configs_json) as f:
        configs = json.load(f)
    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for p in configs:
        print(f"\n=== PILOT {p} ===", flush=True)
        r = run_one(p, args)
        results.append(r)
        print('PILOT_RESULT ' + json.dumps(r), flush=True)
        with open(os.path.join(args.outdir, 'pilot_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
    print(f"\nPilot complete: {len(results)} configs -> {args.outdir}/pilot_results.json")


if __name__ == '__main__':
    main()
