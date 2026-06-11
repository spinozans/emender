"""emender-real-cap: one-config bf16 LM screen + held-out BPB worker.

Trains ONE Emender mixture (gdn2_recall sea + sparse e97_delta tanh emendment
heads) on the REAL Pile (LadderLM typed-gdn2-lm), bf16 UNIFORM, fused split-edit
asserted (the layer raises if the fused kernel cannot engage during training --
no silent eager fallback), then measures held-out BPB on the fixed Pile slice.
Records tok/s, peak mem, params, the integer head counts, and the FUSED assertion.

Usage (one GPU, one config):
  CUDA_VISIBLE_DEVICES=<g> python lm_worker.py --config_json cfg.json --out r.json
"""
import os, sys, json, time, math, argparse, datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'experiments', 'e99_1p3b_cma'))
sys.path.insert(0, _THIS)

import torch
import pilot as P  # real Pile path + held-out slice constants
import emender_common as EC


def cfg_logits(cfg):
    if cfg.get('head_type_logits') is not None:
        return [float(x) for x in cfg['head_type_logits']]
    return EC.delta_logits(cfg['f_delta'])


def build_model(cfg, vocab_size, device):
    from ndm.models.ladder_lm import LadderLM
    lk = EC.layer_kwargs_from_logits(cfg_logits(cfg), cfg.get('lam_max', 1.585),
                                     cfg.get('beta_max', 2.747))
    m = LadderLM(vocab_size=vocab_size, dim=cfg['dim'], depth=cfg['depth'],
                 level='typed-gdn2-lm', n_heads=cfg['n_heads'], n_state=cfg['n_state'],
                 expansion=cfg.get('expansion', 1.0), layer_kwargs=lk).to(device)
    # bf16 UNIFORM (the only fused-compatible half precision for the emendment head)
    m = m.bfloat16()
    return m


def assert_fused(model):
    """Verify every typed mixture layer routes the e97_delta emendment heads through
    the FUSED split-edit Triton kernel on its sequential path -- no silent eager."""
    n = 0
    has_e97 = False
    for _, mod in model.named_modules():
        if mod.__class__.__name__ == 'TypedHeadMixtureLayer':
            n += 1
            if getattr(mod, 'use_triton_e97', None) is not True:
                raise RuntimeError("FUSED-ASSERT: use_triton_e97 != True (eager forbidden)")
            if mod.e97_delta is not None:
                has_e97 = True
                if not mod._e97_delta_is_seq():
                    raise RuntimeError("FUSED-ASSERT: e97_delta present but not on seq split-edit path")
    if n == 0:
        raise RuntimeError("FUSED-ASSERT: no TypedHeadMixtureLayer found")
    return n, has_e97


def heldout_bpb_chunk(model, device, chunk, max_batches, bs):
    """Held-out BPB on the fixed Pile slice at a configurable chunk length."""
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    import tiktoken
    slice_path = P.HELDOUT_SLICE
    if not os.path.exists(slice_path):
        return None
    ds = TokenizedStreamDataset(data_path=slice_path, chunk_size=chunk + 1,
                                seed=7, tokenizer_name=P.TOKENIZER)
    enc = tiktoken.get_encoding(P.TOKENIZER)
    with open(slice_path, 'rb') as f:
        sample = f.read(2_000_000)
    ntok = len(enc.encode(sample.decode('utf-8', errors='ignore')))
    tokens_per_byte = ntok / len(sample)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(max_batches):
            chunks, _, _ = ds.get_batch(bs, device=device)
            l = model(chunks, return_loss=True)
            if isinstance(l, tuple):
                l = l[0]
            losses.append(float(l.item()))
    mean_nats = sum(losses) / len(losses)
    bpb = mean_nats * tokens_per_byte / math.log(2)
    return dict(heldout_nats_per_token=round(mean_nats, 5),
                tokens_per_byte=round(tokens_per_byte, 5),
                heldout_bpb=round(bpb, 5), n_batches=len(losses),
                eval_chunk=chunk,
                slice=dict(path=slice_path, bytes=os.path.getsize(slice_path)))


def run_one(cfg, args, device='cuda'):
    from ndm.data.tokenized_dataset import TokenizedStreamDataset
    import schedulefree
    torch.manual_seed(cfg.get('seed', 0))
    chunk = cfg['chunk']
    train_ds = TokenizedStreamDataset(data_path=P.DATA, chunk_size=chunk + 1, seed=42,
                                      tokenizer_name=P.TOKENIZER)
    vocab_size = getattr(train_ds, 'vocab_size', 50281)
    logits = cfg_logits(cfg)
    model = build_model(cfg, vocab_size, device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    counts = EC.head_counts(logits, cfg['n_heads'])
    n_typed, has_e97 = assert_fused(model)  # raises if eager / kernel can't engage
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg['lr'],
                                         weight_decay=0.01, betas=(0.9, 0.95))
    bs = cfg['batch_size']
    model.train(); opt.train()
    losses, step_dts = [], []
    nan_seen = False
    t_start = time.time()
    step = 0
    wall_cap = cfg.get('wall_minutes', args.wall_minutes) * 60
    max_steps = cfg.get('steps', 10**9)
    max_tokens = cfg.get('max_tokens', None)  # token-matched budget
    fused_asserted = True  # assert_fused passed (use_triton_e97 verified on all layers)
    while step < max_steps and (time.time() - t_start) < wall_cap:
        if max_tokens is not None and step * bs * chunk >= max_tokens:
            break
        step += 1
        chunks, _, _ = train_ds.get_batch(bs, device=device)
        t0 = time.time()
        # NO autocast: the model is bf16-native end to end (uniform half precision).
        loss = model(chunks, return_loss=True)
        if isinstance(loss, tuple):
            loss = loss[0]
        if not torch.isfinite(loss):
            nan_seen = True
            losses.append(float(loss.item())); break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        losses.append(float(loss.item()))
        if step > 3:
            step_dts.append(dt)
        if step % 100 == 0:
            print(f"  step {step} loss {losses[-1]:.4f} "
                  f"elapsed {(time.time()-t_start)/60:.1f}m", flush=True)
    avg_loss = sum(losses) / len(losses) if losses else float('inf')
    final_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else float('inf')
    wall_min = (time.time() - t_start) / 60.0
    tok_s = (sum(bs * chunk / dt for dt in step_dts) / len(step_dts)
             if step_dts else 0.0)

    opt.eval()
    bpb = heldout_bpb_chunk(model, device, chunk=cfg.get('eval_chunk', chunk),
                            max_batches=args.bpb_batches, bs=bs)

    result = dict(
        name=cfg['name'], f_delta=cfg.get('f_delta'), head_type_logits=logits,
        counts=counts, dim=cfg['dim'], depth=cfg['depth'],
        n_heads=cfg['n_heads'], n_state=cfg['n_state'], lr=cfg['lr'],
        batch_size=bs, chunk=chunk, dtype=EC.DTYPE, fused_asserted=fused_asserted,
        fused_typed_layers=n_typed, has_e97_delta=has_e97,
        e97_state_nonlin=EC.E97_STATE_NONLIN,
        model_params=n_params, params_m=round(n_params / 1e6, 3),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        steps=step, tokens=step * bs * chunk, wall_minutes=round(wall_min, 3),
        tok_s=round(tok_s, 1), avg_loss=round(avg_loss, 5), final_loss=round(final_loss, 5),
        nan_seen=nan_seen,
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
        heldout=bpb, timestamp=datetime.datetime.utcnow().isoformat() + 'Z')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--wall_minutes', type=float, default=12.0)
    ap.add_argument('--bpb_batches', type=int, default=30)
    args = ap.parse_args()
    with open(args.config_json) as f:
        cfg = json.load(f)
    r = run_one(cfg, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(r, f, indent=2, default=str)
    print('LM_RESULT ' + json.dumps(r, default=str), flush=True)


if __name__ == '__main__':
    main()
