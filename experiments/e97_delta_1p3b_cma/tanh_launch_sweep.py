"""tanh-e97-1p3b LAUNCH SWEEP (REAL models, no mocks).

The standing prior insists the fused-checkpointed tanh kernel (e88_triton,
LINEAR_STATE=False) is GDN-fast and that the 0.80x measured at the 1.3B head
shape (B=2,H=64,N=32,V=33) is a LAUNCH-bound artifact ("saturate via batch x
head x checkpoint-block parallelism"). This sweep tests that claim directly:

  For each micro-batch B in {2,4,8}, build the REAL 1.3B tanh e97_delta+gdn-neg
  model and the REAL gdn2-mlp baseline, run fwd+bwd at the real ctx T=2048, and
  measure sustained tok/s + the tanh/gdn RATIO. If the ratio climbs toward ~0.95x
  as B grows, a batch-parallel launch fix exists; if it stays flat ~0.8x, the
  sequential-scan latency is the wall (occupancy is not the bottleneck).

Also reports peak SM utilization sampled during the tanh fwd+bwd.

Run:  CUDA_VISIBLE_DEVICES=<g> python tanh_launch_sweep.py
"""
import os, sys, time, json, subprocess, threading
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _THIS); sys.path.insert(0, _ROOT)
import torch
import shapes

# The decisive configs (from headtohead_results.json delta_cfg / base_cfg).
DELTA = dict(dim=2112,
             head_type_logits=[-1.2426827908909686, -28.456765498541962, -29.97388828036681,
                               -29.289780094203703, -28.52456483818494, -30.0, -30.0,
                               -0.5059263781557628],
             lam_max=2.42267247635037, beta_max=2.4506232209686507)
BASEC = dict(dim=2240,
             head_type_logits=[0.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0],
             lam_max=1.585, beta_max=2.747)
T = 2048
VOCAB = shapes.VOCAB_SIZE


def build(cfg, state_nonlin, device):
    m = shapes.build_ladder(cfg['dim'], cfg['head_type_logits'],
                            knob=dict(lam_max=cfg['lam_max'], beta_max=cfg['beta_max']),
                            e97_state_nonlin=state_nonlin)
    return m.to(device).bfloat16()


class SMSampler(threading.Thread):
    """Sample nvidia-smi GPU util in the background; report peak/mean."""
    def __init__(self, gpu):
        super().__init__(daemon=True); self.gpu = str(gpu); self.samples = []; self.stop = False
    def run(self):
        while not self.stop:
            try:
                o = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits',
                     '-i', self.gpu], timeout=4).decode().strip().splitlines()
                self.samples.append(int(o[0]))
            except Exception:
                pass
            time.sleep(0.05)


def measure(cfg, state_nonlin, B, device, n_warm=3, n_timed=10):
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    m = build(cfg, state_nonlin, device); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in m.parameters())
    tok_per_step = B * T

    def step():
        x = torch.randint(0, VOCAB, (B, T + 1), device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = m(x, return_loss=True)
            loss = loss[0] if isinstance(loss, tuple) else loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); torch.cuda.synchronize()
        return float(loss)

    for _ in range(n_warm):
        step()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    smp = SMSampler(gpu); smp.start()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n_timed):
        step()
    dt = (time.time() - t0) / n_timed
    smp.stop = True; smp.join(timeout=1)
    util = (max(smp.samples) if smp.samples else None,
            round(sum(smp.samples) / len(smp.samples), 1) if smp.samples else None)
    peak = torch.cuda.max_memory_allocated() / 1e6
    del m, opt; torch.cuda.empty_cache()
    return dict(tok_s=round(tok_per_step / dt, 1), params_b=round(n_params / 1e9, 4),
                util_peak=util[0], util_mean=util[1], peak_mem_mb=round(peak, 1))


def main():
    dev = 'cuda'
    print(f'[sweep] {torch.cuda.get_device_name(0)}', flush=True)
    out = {}
    for B in (2, 4, 8):
        try:
            tanh = measure(DELTA, 'tanh', B, dev)
            gdn = measure(BASEC, 'tanh', B, dev)  # baseline is 100% gdn-neg; state_nonlin unused
            ratio = tanh['tok_s'] / gdn['tok_s'] if gdn['tok_s'] else 0
            out[f'B{B}'] = dict(tanh=tanh, gdn=gdn, ratio=round(ratio, 3))
            print(f"B={B}: tanh {tanh['tok_s']} tok/s (util {tanh['util_peak']}% peak/"
                  f"{tanh['util_mean']}% mean, {tanh['peak_mem_mb']}MB) | "
                  f"gdn {gdn['tok_s']} tok/s | RATIO {ratio:.3f}x", flush=True)
        except RuntimeError as e:
            out[f'B{B}'] = dict(error=str(e)[:200])
            print(f"B={B}: OOM/error {str(e)[:120]}", flush=True)
            torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(_THIS, 'results', 'tanh_launch_sweep.json'), 'w'), indent=2)
    print('WROTE results/tanh_launch_sweep.json', flush=True)
    print('\n=== RATIO vs batch (tests batch-parallel launch fix) ===', flush=True)
    for B in (2, 4, 8):
        r = out.get(f'B{B}', {})
        if 'ratio' in r:
            print(f"  B={B}: {r['ratio']:.3f}x  (tanh util {r['tanh']['util_mean']}% mean)", flush=True)


if __name__ == '__main__':
    os.makedirs(os.path.join(_THIS, 'results'), exist_ok=True)
    main()
