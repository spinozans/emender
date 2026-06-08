"""e97delta-1p3b: one-config time-bounded fused LM screen on REAL Pile.

Trains a WITHIN-LAYER typed-gdn2-lm cell (head-type mixture over the canonical
8 types) + SwiGLU MLP at ~1.3B params, on real Pile tokens, schedule-free AdamW,
bf16, chunked-fused e97_delta kernel. Records the full loss-vs-tokens curve (so a
TOKEN-MATCHED cross-walk can be read off a single WALL-CLOCK-matched run), held-out
BPB, sustained tok/s, params, stability. Reuses pilot.heldout_bpb / loss_on /
TokenizedStreamDataset so numbers line up with the prior E99 1.3B batch.

gdn-neg == gdn2_recall heads with gdn_allow_neg_eigval=True (global on this path).
e97_delta == type idx 7, routed through the chunked-parallel fused Triton kernel.

REAL DATA / REAL TRAINING ONLY. No mocks. NaN/OOM/walltime -> stop+log.

Modes:
  --wall_seconds S    : train for S seconds of wall-clock (default)
  --token_cap N       : ALSO stop once N training tokens are consumed (for the
                        token-matched arm). Whichever bound hits first stops.
Usage (one GPU, one config):
  CUDA_VISIBLE_DEVICES=g python screen.py --config_json cfg.json --out r.json
"""
import os, sys, json, time, math, argparse, datetime, gc

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'experiments', 'e99_1p3b_cma'))

import torch
import pilot as P   # heldout_bpb, loss_on protocol (same as prior E99 batch)
from ndm.data.tokenized_dataset import TokenizedStreamDataset
from shapes import build_ladder, BASE, VOCAB_SIZE, allocate
P.TokenizedStreamDataset = TokenizedStreamDataset

CHUNK = P.CHUNK          # 2048
DATA = P.DATA            # real Pile
TOKENIZER = P.TOKENIZER  # p50k_base
KNOB_SUFFIXES = ('lam_raw', 'beta_raw', 'igain_raw', 'gamma_raw')


def _fast_loss(model, chunks, bf16):
    """Forward loss using the CHUNKED kernel path. The e97_delta chunked Triton
    kernel is gated on model.training=True (eval mode falls back to a ~90x slower
    sequential T-scan). dropout=0.0 and RMSNorm carries no batch stats in these
    configs, so train()+no_grad gives the IDENTICAL loss at chunked-kernel speed."""
    was_training = model.training
    model.train()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
    if not was_training:
        model.eval()
    return float(loss.item())


def fast_heldout_bpb(model, vocab_size, device, max_batches=40, bs=2, bf16=True):
    """pilot.heldout_bpb math, but train()+no_grad so the e97_delta chunked kernel
    is used (eval-mode would be ~90x slower). BPB = nats/token * tokens/byte / ln2,
    tokens/byte MEASURED on the exact held-out slice (p50k_base)."""
    import tiktoken
    slice_info = P._heldout_slice_info()
    if slice_info is None:
        return None
    ds = TokenizedStreamDataset(data_path=P.HELDOUT_SLICE, chunk_size=CHUNK + 1,
                                seed=7, tokenizer_name=TOKENIZER)
    enc = tiktoken.get_encoding(TOKENIZER)
    with open(P.HELDOUT_SLICE, 'rb') as f:
        sample = f.read(2_000_000)
    ntok = len(enc.encode(sample.decode('utf-8', errors='ignore')))
    tokens_per_byte = ntok / len(sample)
    losses = []
    for _ in range(max_batches):
        chunks, _, _ = ds.get_batch(bs, device=device)
        losses.append(_fast_loss(model, chunks, bf16))
    mean_nats = sum(losses) / len(losses)
    bpb = mean_nats * tokens_per_byte / math.log(2)
    return dict(heldout_nats_per_token=round(mean_nats, 5),
                tokens_per_byte=round(tokens_per_byte, 5),
                heldout_bpb=round(bpb, 5), n_batches=len(losses), slice=slice_info)


def build_model(cfg, device):
    m = build_ladder(cfg['dim'], cfg['head_type_logits'],
                     knob=dict(lam_max=cfg.get('lam_max', 1.585),
                               beta_max=cfg.get('beta_max', 2.747)))
    m = m.to(device)
    if cfg.get('bf16', True):
        m = m.bfloat16()
    return m


def make_optimizer(model, lr, knob_lr_mult):
    import schedulefree
    if abs(knob_lr_mult - 1.0) < 1e-9:
        groups = model.parameters()
    else:
        knob, base = [], []
        for name, pr in model.named_parameters():
            (knob if any(name.endswith(s) for s in KNOB_SUFFIXES) else base).append(pr)
        groups = [dict(params=base, lr=lr),
                  dict(params=knob, lr=lr * knob_lr_mult)]
    return schedulefree.AdamWScheduleFree(groups, lr=lr, weight_decay=0.01,
                                          betas=(0.9, 0.95))


def run(cfg, wall_seconds, token_cap, seed, do_roundtrip, device='cuda'):
    torch.manual_seed(seed)
    probe = P.TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1, seed=0,
                                     tokenizer_name=TOKENIZER)
    vocab_size = getattr(probe, 'vocab_size', VOCAB_SIZE)
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, cfg['lr'], cfg.get('knob_lr_mult', 1.0))
    train_ds = P.TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1,
                                        seed=42 + seed, tokenizer_name=TOKENIZER)
    bs = cfg.get('batch_size', 2)
    tok_per_step = bs * (CHUNK + 1)
    model.train(); opt.train()
    losses, step_dts, grad_norms = [], [], []
    curve = []  # (tokens, wall_s, loss)
    nan_seen = nonfinite_grad = False
    stop_reason = None

    # --- UNTIMED warmup: the chunked e97_delta kernel has B,T,H as tl.constexpr,
    # so each distinct e97_delta head count cold-JIT-compiles fwd+bwd (~minutes).
    # Run a few real steps OUTSIDE the timed budget so compilation does NOT eat
    # the training wall — every candidate then trains for EQUAL compute regardless
    # of whether its head-count shape was already cached. (These steps DO update
    # weights on real Pile data; nothing is mocked.) ---
    compile_t0 = time.time()
    n_warm = int(cfg.get('compile_warmup_steps', 3))
    for _ in range(n_warm):
        chunks, _, _ = train_ds.get_batch(bs, device=device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg.get('bf16', True)):
            loss = model(chunks, return_loss=True)
            loss = loss[0] if isinstance(loss, tuple) else loss
        if not torch.isfinite(loss):
            nan_seen = True; stop_reason = 'nonfinite_loss_warmup'; break
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); torch.cuda.synchronize()
    compile_seconds = round(time.time() - compile_t0, 2)

    t0 = time.time(); step = 0; tokens = 0
    while True:
        if (time.time() - t0) >= wall_seconds:
            stop_reason = 'wall_cap'; break
        if token_cap is not None and tokens >= token_cap:
            stop_reason = 'token_cap'; break
        step += 1
        chunks, _, _ = train_ds.get_batch(bs, device=device)
        ts = time.time()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg.get('bf16', True)):
            loss = model(chunks, return_loss=True)
            if isinstance(loss, tuple):
                loss = loss[0]
        if not torch.isfinite(loss):
            nan_seen = True; stop_reason = 'nonfinite_loss'
            losses.append(float(loss.item())); break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(gn)):
            nonfinite_grad = True; stop_reason = 'nonfinite_grad'
            opt.zero_grad(set_to_none=True); break
        opt.step(); torch.cuda.synchronize()
        dt = time.time() - ts
        lv = float(loss.item()); tokens += tok_per_step
        losses.append(lv); grad_norms.append(float(gn))
        if step > 3:
            step_dts.append(dt)
        if step % 10 == 0:
            curve.append((tokens, round(time.time() - t0, 2), round(lv, 5)))
        if step % 50 == 0:
            print(f"  step {step} tok {tokens} loss {lv:.4f} "
                  f"elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    wall_min = (time.time() - t0) / 60.0
    avg_loss = sum(losses) / len(losses) if losses else float('inf')
    final_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else float('inf')
    sustained_tok_s = (sum(tok_per_step / d for d in step_dts) / len(step_dts)
                       if step_dts else 0.0)
    bpb = fast_heldout_bpb(model, vocab_size, device, max_batches=40, bs=bs,
                           bf16=cfg.get('bf16', True))
    rt = None
    if do_roundtrip:
        rt = _roundtrip(model, cfg, vocab_size, bs, device)
    counts = allocate(cfg['head_type_logits'], BASE['n_heads'])['counts']
    out = dict(
        cfg=cfg, model_params=n_params, params_b=round(n_params / 1e9, 4),
        counts=counts, seed=seed, dtype='bf16' if cfg.get('bf16', True) else 'fp32',
        steps=step, batch_size=bs, tokens=tokens, compile_seconds=compile_seconds,
        wall_minutes=round(wall_min, 3), wall_seconds_budget=wall_seconds,
        token_cap=token_cap, avg_loss=round(avg_loss, 5),
        final_loss=round(final_loss, 5), stop_reason=stop_reason,
        nan_seen=nan_seen, nonfinite_grad=nonfinite_grad,
        sustained_tok_s=round(sustained_tok_s, 1),
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
        heldout=bpb, curve=curve, roundtrip=rt,
        timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    del opt, model; gc.collect(); torch.cuda.empty_cache()
    return out


def _roundtrip(model, cfg, vocab_size, bs, device):
    import subprocess
    held = TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1,
                                  seed=1234, tokenizer_name=TOKENIZER)
    held_chunks, _, _ = held.get_batch(bs, device=device)
    l_pre = _fast_loss(model, held_chunks, cfg.get('bf16', True))
    ck = os.path.join(_THIS, 'results', f"_rt_{os.getpid()}.pt")
    torch.save(dict(model_state_dict=model.state_dict(), cfg=cfg,
                    vocab_size=vocab_size, eval_batch=held_chunks.detach().cpu(),
                    l_pre=l_pre), ck)
    rt = subprocess.run([sys.executable, os.path.abspath(__file__),
                         '--phase', 'reload', '--reload_ckpt', ck],
                        capture_output=True, text=True)
    res = None
    for line in rt.stdout.splitlines():
        if line.startswith('RELOAD_RESULT '):
            res = json.loads(line[len('RELOAD_RESULT '):])
    try:
        os.remove(ck)
    except OSError:
        pass
    if res is None:
        return dict(ok=False, err=rt.stderr[-800:])
    delta = abs(res['l_post'] - l_pre)
    return dict(ok=bool(delta < 1e-2 and res['n_missing'] == 0 and res['n_unexpected'] == 0),
                l_pre=round(l_pre, 6), l_post=round(res['l_post'], 6),
                delta=round(delta, 8), n_missing=res['n_missing'],
                n_unexpected=res['n_unexpected'])


def reload_phase(ck):
    d = torch.load(ck, map_location='cpu')
    cfg = d['cfg']
    m = build_model(cfg, 'cuda')
    missing, unexpected = m.load_state_dict(d['model_state_dict'], strict=False)
    chunks = d['eval_batch'].to('cuda')
    print('RELOAD_RESULT ' + json.dumps(dict(
        l_post=_fast_loss(m, chunks, cfg.get('bf16', True)),
        n_missing=len(missing), n_unexpected=len(unexpected))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='train')
    ap.add_argument('--reload_ckpt')
    ap.add_argument('--config_json')
    ap.add_argument('--out')
    ap.add_argument('--wall_seconds', type=float, default=480.0)
    ap.add_argument('--token_cap', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--roundtrip', type=int, default=0)
    args = ap.parse_args()
    if args.phase == 'reload':
        reload_phase(args.reload_ckpt); return
    with open(args.config_json) as f:
        cfg = json.load(f)
    try:
        out = run(cfg, args.wall_seconds, args.token_cap, args.seed,
                  bool(args.roundtrip))
    except torch.cuda.OutOfMemoryError as e:
        out = dict(cfg=cfg, error='OOM', msg=str(e)[:400],
                   timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    h = out.get('heldout') or {}
    print(f"DONE bpb={h.get('heldout_bpb')} avg_loss={out.get('avg_loss')} "
          f"tokens={out.get('tokens')} tok/s={out.get('sustained_tok_s')} "
          f"stop={out.get('stop_reason')}", flush=True)


if __name__ == '__main__':
    main()
