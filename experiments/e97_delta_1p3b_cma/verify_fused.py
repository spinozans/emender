"""fuse-2kernel verification (REAL data, no mocks): at the actual 1.3B dims,

  (1) chunked ENGAGEMENT — the linear-state e97_delta candidate routes EVERY
      e97_delta layer through the chunked-parallel fused Triton kernel (counts
      chunked-kernel calls per forward vs #layers-with-e97_delta-heads). The
      tanh variant is the control: it must NOT engage chunked (sequential T-scan).
  (2) NO NaN over real forward+backward steps on real Pile tokens (the log-decay
      backward fix at init-realistic decay).
  (3) THROUGHPUT vs gdn2-mlp — sustained tok/s for the chunked-linear candidate
      vs the pure gdn2-mlp baseline, both at their param-matched dims.

Run: CUDA_VISIBLE_DEVICES=<g> python verify_fused.py
"""
import os, sys, time, json, argparse
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _THIS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'experiments', 'e99_1p3b_cma'))
import torch
import shapes
import pilot as P
from ndm.data.tokenized_dataset import TokenizedStreamDataset
from shapes import VOCAB_SIZE, BASE

CHUNK = P.CHUNK          # 2048, the real training context
DATA = P.DATA            # real Pile
TOKENIZER = P.TOKENIZER  # p50k_base

# Prior decisive configs (headtohead_results.json).
DELTA = dict(dim=2112,
             head_type_logits=[-1.2426827908909686, -28.456765498541962, -29.97388828036681,
                               -29.289780094203703, -28.52456483818494, -30.0, -30.0,
                               -0.5059263781557628],
             lam_max=2.42267247635037, beta_max=2.4506232209686507)
BASEC = dict(dim=2240,
             head_type_logits=[0.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0],
             lam_max=1.585, beta_max=2.747)


def instrument_chunked():
    """Wrap the chunked kernel entry to count calls; returns a mutable counter."""
    import ndm.triton.e97_chunked_autograd as M
    import ndm.models.e88_fla_hybrid as H
    ctr = {'chunked': 0}
    orig = M.e97_delta_chunked_triton
    def wrapped(*a, **kw):
        ctr['chunked'] += 1
        return orig(*a, **kw)
    M.e97_delta_chunked_triton = wrapped
    H.e97_delta_chunked_triton = wrapped  # the name imported into the hybrid module
    return ctr


def build(cfg, state_nonlin, device):
    m = shapes.build_ladder(cfg['dim'], cfg['head_type_logits'],
                            knob=dict(lam_max=cfg['lam_max'], beta_max=cfg['beta_max']),
                            e97_state_nonlin=state_nonlin)
    return m.to(device).bfloat16()


def count_e97_delta_layers(model):
    n = 0
    for mod in model.modules():
        if mod.__class__.__name__ == 'E88FLAHybrid' \
                and getattr(mod, 'use_chunked_e97', False) and not getattr(mod, 'raw_write', True):
            n += 1
    return n


def run_arm(tag, cfg, state_nonlin, device, n_warm=3, n_timed=12, count=False):
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    ds = TokenizedStreamDataset(data_path=DATA, chunk_size=CHUNK + 1, seed=42,
                                  tokenizer_name=TOKENIZER)
    m = build(cfg, state_nonlin, device)
    n_params = sum(p.numel() for p in m.parameters())
    n_e97_layers = count_e97_delta_layers(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    m.train()
    bs = 2
    tok_per_step = bs * (CHUNK + 1)
    ctr = instrument_chunked() if count else None
    nan = False
    # warmup (untimed: constexpr-H JIT compile)
    for _ in range(n_warm):
        chunks, _, _ = ds.get_batch(bs, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = m(chunks, return_loss=True)
            loss = loss[0] if isinstance(loss, tuple) else loss
        if not torch.isfinite(loss):
            nan = True; break
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(gn)):
            nan = True; break
        opt.step(); torch.cuda.synchronize()
    chunked_calls_warm = ctr['chunked'] if ctr else None
    if ctr:
        ctr['chunked'] = 0
    # timed
    dts = []
    for i in range(n_timed):
        chunks, _, _ = ds.get_batch(bs, device=device)
        torch.cuda.synchronize(); ts = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = m(chunks, return_loss=True)
            loss = loss[0] if isinstance(loss, tuple) else loss
        if not torch.isfinite(loss):
            nan = True; break
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not torch.isfinite(torch.as_tensor(gn)):
            nan = True; break
        opt.step(); torch.cuda.synchronize()
        dts.append(time.time() - ts)
    chunked_per_fwd = (ctr['chunked'] / max(len(dts), 1)) if ctr else None
    tok_s = (tok_per_step / (sum(dts) / len(dts))) if dts else 0.0
    peak = torch.cuda.max_memory_allocated() / 1e6
    del m, opt
    torch.cuda.empty_cache()
    return dict(tag=tag, dim=cfg['dim'], params_b=round(n_params / 1e9, 4),
                n_e97_delta_layers=n_e97_layers, chunked_calls_warm=chunked_calls_warm,
                chunked_per_fwd=chunked_per_fwd, nan=nan,
                tok_s=round(tok_s, 1), peak_mem_mb=round(peak, 1))


def main():
    dev = 'cuda'
    print(f'[verify_fused] device cap {torch.cuda.get_device_name(0)}', flush=True)
    out = {}
    # Candidate: linear-state e97_delta (chunked). Instrument engagement.
    print('=== CANDIDATE: e97_delta+gdn-neg, linear state (chunked) ===', flush=True)
    out['candidate_chunked'] = run_arm('candidate_chunked', DELTA, 'identity', dev, count=True)
    print(json.dumps(out['candidate_chunked'], indent=2), flush=True)
    # Control: tanh-state e97_delta (must be sequential, 0 chunked calls).
    print('=== CONTROL: e97_delta+gdn-neg, tanh state (sequential) ===', flush=True)
    out['candidate_tanh'] = run_arm('candidate_tanh', DELTA, 'tanh', dev, count=True)
    print(json.dumps(out['candidate_tanh'], indent=2), flush=True)
    # Baseline: pure gdn2-mlp.
    print('=== BASELINE: gdn2-mlp ===', flush=True)
    out['baseline_gdn'] = run_arm('baseline_gdn', BASEC, 'tanh', dev, count=False)
    print(json.dumps(out['baseline_gdn'], indent=2), flush=True)

    c = out['candidate_chunked']; t = out['candidate_tanh']; b = out['baseline_gdn']
    ratio_chunked = c['tok_s'] / b['tok_s'] if b['tok_s'] else 0
    ratio_seq = t['tok_s'] / b['tok_s'] if b['tok_s'] else 0
    print('\n=== VERDICT ===', flush=True)
    print(f"chunked engagement: {c['chunked_per_fwd']}/{c['n_e97_delta_layers']} layers "
          f"(tanh control: {t['chunked_per_fwd']} chunked calls — should be 0)", flush=True)
    print(f"NaN: candidate={c['nan']} tanh={t['nan']} baseline={b['nan']}", flush=True)
    print(f"throughput: candidate-chunked {c['tok_s']} tok/s = {ratio_chunked:.3f}x gdn2-mlp "
          f"({b['tok_s']}); candidate-SEQ {t['tok_s']} = {ratio_seq:.3f}x", flush=True)
    out['summary'] = dict(ratio_chunked_vs_gdn=round(ratio_chunked, 3),
                          ratio_seq_vs_gdn=round(ratio_seq, 3),
                          chunked_per_fwd=c['chunked_per_fwd'],
                          n_e97_delta_layers=c['n_e97_delta_layers'])
    json.dump(out, open(os.path.join(_THIS, 'results', 'verify_fused.json'), 'w'), indent=2)
    print('WROTE results/verify_fused.json', flush=True)


if __name__ == '__main__':
    os.makedirs(os.path.join(_THIS, 'results'), exist_ok=True)
    main()
