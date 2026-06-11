"""emender-real-cap — convergent-loss tie: REAL Emender vs pure GDN-2, matched precision.

Held-out BPB on REAL Comma-Pile tokens, WALL-matched (train_minutes) AND token-matched
(token curve cross-walk). Both arms bf16 (matched precision -> no token starvation, the
fix vs the opt-1p3b fp32 strawman). Fused Triton, loud no-eager guard, fresh-process
round-trip reload. Adapted from experiments/opt_1p3b/lm_runner.py (same method).

  gdn2     fla-gdn, neg-eig                       = pure GDN-2 CONTROL (incumbent)
  emender  56/60 gdn2_recall + 8/4 e97_delta-tanh = REAL Emender (sparse capability sprinkle)

Small LM shape (dim=1024, depth=12, n_heads=64 -> exact 4/64 & 8/64; ns=32 exp=2.0 -> V=64):

  eval "$(scripts/gpu_lease.sh 1)"
  python experiments/emender_real_cap/lm_convergent.py --arm gdn2 --seed 0 --train_minutes 12
  python experiments/emender_real_cap/lm_convergent.py --arm emender4 --seed 0 --train_minutes 12
"""
import os, sys, json, time, math, argparse, subprocess, datetime
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)
os.environ.setdefault('XMA_PATH', '/home/erikg/xma')
import torch

DATA = '/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_1tb.txt'
TOKENIZER = 'p50k_base'
CHUNK = 2048
LN2 = math.log(2.0)
HELDOUT_SEED = 7777
ROUNDTRIP_SEED = 1234
DIM, DEPTH, NH, NS, EXP = 1024, 12, 64, 32, 2.0
MLP_RATIO = 6208 / 2304


def _logits9(counts):
    names = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin',
             'gdn2_nonlin_shell', 'e97_raw', 'e97_delta', 'refit']
    return [math.log(counts[t]) if counts.get(t, 0) > 0 else -30.0 for t in names]


CONFIGS = {
    'gdn2': dict(builder='ladder', level='fla-gdn', bf16=True, lr=8.0e-4, batch_size=2,
                 layer_kwargs=None, role='REFERENCE = pure GDN-2 incumbent (fla-gdn, neg-eig), bf16'),
    'gdn2typed': dict(builder='typed', level='typed-gdn2-lm', bf16=True, lr=8.0e-4, batch_size=2,
                      layer_kwargs=dict(head_type_logits=_logits9({'gdn2_recall': 64}),
                                        gdn_allow_neg_eigval=True, overlap_streams=True),
                      role='MATCHED-PARAM CONTROL = typed all gdn2_recall (same path as Emender, '
                           'differs only in head types), bf16'),
    'emender4': dict(builder='typed', level='typed-gdn2-lm', bf16=True, lr=8.0e-4, batch_size=2,
                     layer_kwargs=dict(head_type_logits=_logits9({'gdn2_recall': 60, 'e97_delta': 4}),
                                       gdn_allow_neg_eigval=True, e97_state_nonlin='tanh',
                                       use_chunked_e97_delta=False, overlap_streams=True),
                     role='Emender 4/64 e97_delta-tanh sprinkle, bf16 FUSED'),
    'emender8': dict(builder='typed', level='typed-gdn2-lm', bf16=True, lr=8.0e-4, batch_size=2,
                     layer_kwargs=dict(head_type_logits=_logits9({'gdn2_recall': 56, 'e97_delta': 8}),
                                       gdn_allow_neg_eigval=True, e97_state_nonlin='tanh',
                                       use_chunked_e97_delta=False, overlap_streams=True),
                     role='Emender 8/64 e97_delta-tanh sprinkle, bf16 FUSED'),
}


def build_model(cfg, vocab, device):
    from ndm.models.ladder_lm import LadderLM
    m = LadderLM(vocab_size=vocab, dim=DIM, depth=DEPTH, level=cfg['level'],
                 n_heads=NH, n_state=NS, expansion=EXP, layer_kwargs=cfg['layer_kwargs'],
                 mlp_ratio=MLP_RATIO).to(device)
    if cfg['bf16']:
        m = m.bfloat16()
    return m


def assert_fused(model, cfg):
    if cfg['builder'] != 'typed':
        return None
    mod_has_e97 = False
    n = 0
    for _, mod in model.named_modules():
        if mod.__class__.__name__ == 'TypedHeadMixtureLayer':
            n += 1
            if getattr(mod, 'use_triton_e97', None) is not True:
                raise RuntimeError("FUSED-ASSERT: use_triton_e97 != True (eager forbidden)")
            if mod.e97_delta is not None:
                mod_has_e97 = True
                if not mod._e97_delta_is_seq():
                    raise RuntimeError("FUSED-ASSERT: e97_delta present but not on seq split-edit path")
    if n == 0:
        raise RuntimeError("FUSED-ASSERT: no TypedHeadMixtureLayer found")
    print(f"FUSED-ASSERT OK: {n} typed layers, e97_delta_seq={mod_has_e97}", flush=True)
    return n


def _dataset(seed):
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    return TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1, seed=seed,
                                  tokenizer_name=TOKENIZER)


def loss_on(model, chunks, bf16):
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        loss = model(chunks, return_loss=True)
        loss = loss[0] if isinstance(loss, tuple) else loss
    return float(loss.item())


def heldout_eval(model, cfg, device, n_batches=8):
    ds = _dataset(HELDOUT_SEED); bs = cfg['batch_size']
    tot_nll = tot_tok = tot_bytes = 0
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            chunks, _, _ = ds.get_batch(bs, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
                loss = model(chunks, return_loss=True)
                loss = loss[0] if isinstance(loss, tuple) else loss
            npred = chunks.shape[0] * (chunks.shape[1] - 1)
            tot_nll += float(loss.item()) * npred; tot_tok += npred
            for row in chunks.tolist():
                tot_bytes += len(ds.enc.decode(row[1:]).encode('utf-8'))
    nats = tot_nll / tot_tok; bpt = tot_bytes / tot_tok
    return nats, bpt, (nats / LN2) / bpt, tot_tok


def phase_reload(args):
    ckpt = torch.load(args.reload_ckpt, map_location='cpu')
    cfg = ckpt['cfg']
    model = build_model(cfg, ckpt['vocab_size'], 'cuda')
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    out = dict(l_post=loss_on(model, ckpt['eval_batch'].to('cuda'), cfg['bf16']),
               n_missing=len(missing), n_unexpected=len(unexpected))
    print('RELOAD_RESULT ' + json.dumps(out))
    return out


def phase_run(args):
    cfg = dict(CONFIGS[args.arm]); device = 'cuda'
    torch.manual_seed(args.seed)
    probe = _dataset(args.seed); vocab = probe.vocab_size
    model = build_model(cfg, vocab, device)
    n_typed = assert_fused(model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    import schedulefree
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg['lr'],
                                          weight_decay=0.01, betas=(0.9, 0.95))
    ds = _dataset(args.seed if args.seed else 42); bs = cfg['batch_size']
    budget = args.train_minutes * 60.0
    model.train(); opt.train()
    losses, curve, dts = [], [], []
    cum = 0; step = 0; nan = False; stop = None
    t0 = time.time()
    while True:
        el = time.time() - t0
        if el >= budget:
            stop = 'budget_reached'; break
        step += 1
        chunks, _, _ = ds.get_batch(bs, device=device)
        ts = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['bf16']):
            loss = model(chunks, return_loss=True)
            loss = loss[0] if isinstance(loss, tuple) else loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        torch.cuda.synchronize()
        lv = float(loss.item()); losses.append(lv); cum += bs * (CHUNK + 1)
        if step > 3:
            dts.append(time.time() - ts)
        curve.append([cum, round(lv, 5), round(el, 1)])
        if lv != lv:
            nan = True; stop = 'HARD_STOP_NaN'; break
        if step % 25 == 0:
            print(f'[{args.arm} s{args.seed}] step {step} loss {lv:.4f} '
                  f'{el:.0f}/{budget:.0f}s tok {cum} {bs*(CHUNK+1)/dts[-1]:.0f} tok/s', flush=True)
    wall = time.time() - t0
    sustained = (sum(bs * (CHUNK + 1) / d for d in dts) / len(dts)) if dts else 0.0
    tail = max(1, len(losses) // 10)
    late = sum(losses[-tail:]) / tail if losses else None
    os.makedirs(args.outdir, exist_ok=True)
    if nan or not losses:
        r = dict(arm=args.arm, seed=args.seed, stop_reason=stop, nan=nan)
        json.dump(r, open(os.path.join(args.outdir, f'{args.arm}_s{args.seed}_result.json'), 'w'), indent=2)
        print('ARM_FAILURE ' + json.dumps(r), flush=True); return r
    ho_nats, ho_bpt, ho_bpb, ho_tok = heldout_eval(model, cfg, device)
    # fresh-process round-trip
    rt = _dataset(ROUNDTRIP_SEED); held, _, _ = rt.get_batch(bs, device=device)
    l_pre = loss_on(model, held, cfg['bf16']); model.train(); opt.train()
    ckpt = os.path.join(args.outdir, f'{args.arm}_s{args.seed}_ckpt.pt')
    torch.save({'model_state_dict': model.state_dict(), 'cfg': cfg,
                'vocab_size': vocab, 'eval_batch': held.detach().cpu()}, ckpt)
    import gc; del opt; model.to('cpu'); del model; gc.collect(); torch.cuda.empty_cache()
    p = subprocess.run([sys.executable, os.path.abspath(__file__), '--phase', 'reload',
                        '--reload_ckpt', ckpt], env={**os.environ}, capture_output=True, text=True)
    ro = None
    for line in p.stdout.splitlines():
        if line.startswith('RELOAD_RESULT '):
            ro = json.loads(line[len('RELOAD_RESULT '):])
    if ro is None:
        print('RELOAD STDERR:\n' + p.stderr[-1500:], flush=True)
        raise RuntimeError('reload produced no result')
    delta = abs(ro['l_post'] - l_pre)
    rt_ok = delta < 1e-2 and ro['n_missing'] == 0 and ro['n_unexpected'] == 0
    result = dict(arm=args.arm, level=cfg['level'], role=cfg['role'], seed=args.seed,
                  params=n_params, params_m=round(n_params / 1e6, 2),
                  dim=DIM, depth=DEPTH, n_heads=NH, n_state=NS, expansion=EXP, lr=cfg['lr'],
                  batch_size=bs, chunk_size=CHUNK, dtype='bf16', fused_typed_layers=n_typed,
                  train_minutes=args.train_minutes, walltime_s=round(wall, 1), steps=step,
                  total_tokens=cum, late_train_loss=round(late, 6), final_loss=round(losses[-1], 6),
                  heldout_nats_per_token=round(ho_nats, 6), heldout_bpb=round(ho_bpb, 6),
                  heldout_bytes_per_token=round(ho_bpt, 4), heldout_tokens=ho_tok,
                  sustained_tok_s=round(sustained, 1),
                  peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
                  stop_reason=stop, roundtrip_l_pre=round(l_pre, 6),
                  roundtrip_l_post=round(ro['l_post'], 6), roundtrip_delta=round(delta, 8),
                  roundtrip_ok=rt_ok, loss_curve=curve,
                  timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    json.dump(result, open(os.path.join(args.outdir, f'{args.arm}_s{args.seed}_result.json'), 'w'), indent=2)
    print('ARM_RESULT ' + json.dumps({k: v for k, v in result.items() if k != 'loss_curve'}), flush=True)
    try:
        os.remove(ckpt)
    except OSError:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(CONFIGS.keys()))
    ap.add_argument('--phase', choices=['run', 'reload'], default='run')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_minutes', type=float, default=12.0)
    ap.add_argument('--reload_ckpt', default=None)
    ap.add_argument('--outdir', default=os.path.join(_THIS, 'results_lm'))
    args = ap.parse_args()
    if args.phase == 'reload':
        phase_reload(args)
    else:
        phase_run(args)


if __name__ == '__main__':
    main()
