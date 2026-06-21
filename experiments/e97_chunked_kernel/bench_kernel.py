"""Kernel-level throughput + GPU-util benchmark for the chunked E97 split-edit
delta kernel vs (a) the OLD sequential E97 Triton T-scan and (b) native GDN-2
(FLA chunk_gated_delta_rule).

This is the decisive measurement for the within-layer latency-bound flip: the
within-layer throughput wall hinged on the E97 split-edit kernel being ~2.6x
slower than GDN-2 (13-15% util).
We measure fwd+bwd ms/iter and SM utilization (via a sampling thread over
nvidia-smi) at matched [B,T,H,N,V] shapes at BOTH the within-layer head shape and
1.3B-scale dims.

Methods compared (all bf16, all fwd+bwd unless noted):
  * gdn2              : FLA chunk_gated_delta_rule  (the throughput target)
  * e97_fused         : NEW fused chunked Triton fwd+bwd autograd kernel
                        (ndm.triton.e97_chunked_autograd) — the load-bearing fix
  * e97_pytorch       : staged PyTorch-chunked autograd (the prior, slow path)
  * e97_seq_fwd       : OLD sequential E97 Triton T-scan, FORWARD ONLY
                        (the 2.6x-slowdown source; bwd is a symmetric T-scan)

REAL kernels, REAL tensors, REAL timing. No mocks.
"""
import os, sys, json, time, threading, subprocess
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import torch

from ndm.triton.e97_chunked import e97_delta_chunked
from ndm.triton.e97_chunked_autograd import e97_delta_chunked_triton
from ndm.triton.e88_triton_forward import e88_triton_forward
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

DEV = 'cuda'
DT = torch.bfloat16


# ---------------------------------------------------------------------------
# GPU utilization sampler (nvidia-smi, real measurement)
# ---------------------------------------------------------------------------
class UtilSampler:
    def __init__(self, gpu_index=0, period=0.02):
        self.gpu_index = gpu_index
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', f'--id={self.gpu_index}',
                     '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    timeout=1.0).decode().strip()
                self.samples.append(float(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(self.period)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2.0)

    def stats(self):
        if not self.samples:
            return {'mean': None, 'max': None, 'n': 0}
        s = sorted(self.samples)
        return {'mean': round(sum(s) / len(s), 1),
                'max': round(s[-1], 1),
                'p50': round(s[len(s) // 2], 1),
                'n': len(s)}


# ---------------------------------------------------------------------------
# Input makers (matched shapes)
# ---------------------------------------------------------------------------
def mk_inputs(B, T, H, N, V, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = torch.randn(B, T, H, N, device=DEV, dtype=DT, generator=g) * 0.5
    q = torch.randn(B, T, H, N, device=DEV, dtype=DT, generator=g) * 0.5
    v = torch.randn(B, T, H, V, device=DEV, dtype=DT, generator=g) * 0.5
    decay = (torch.sigmoid(torch.randn(B, T, H, device=DEV, dtype=DT, generator=g)) * 0.3 + 0.6)
    erase = torch.sigmoid(torch.randn(B, T, H, N, device=DEV, dtype=DT, generator=g))
    wgate = torch.sigmoid(torch.randn(B, T, H, V, device=DEV, dtype=DT, generator=g))
    beta = torch.sigmoid(torch.randn(B, T, H, device=DEV, dtype=DT, generator=g))
    return dict(k=k, q=q, v=v, decay=decay, erase=erase, wgate=wgate, beta=beta)


# ---------------------------------------------------------------------------
# Per-method fwd+bwd step closures
# ---------------------------------------------------------------------------
def step_fused(inp, C=32):
    k = inp['k'].clone().requires_grad_(True)
    v = inp['v'].clone().requires_grad_(True)
    q = inp['q'].clone().requires_grad_(True)
    d = inp['decay'].clone().requires_grad_(True)
    e = inp['erase'].clone().requires_grad_(True)
    w = inp['wgate'].clone().requires_grad_(True)

    def go():
        out, _ = e97_delta_chunked_triton(k, v, q, d, e, w, chunk_size=C)
        out.sum().backward()
        for t in (k, v, q, d, e, w):
            t.grad = None
    return go


def step_pytorch(inp, C=64):
    k = inp['k'].clone().requires_grad_(True)
    v = inp['v'].clone().requires_grad_(True)
    q = inp['q'].clone().requires_grad_(True)
    d = inp['decay'].clone().requires_grad_(True)
    e = inp['erase'].clone().requires_grad_(True)
    w = inp['wgate'].clone().requires_grad_(True)

    def go():
        out, _ = e97_delta_chunked(k, v, q, d, e, w, chunk_size=C, inverse_mode='solve')
        out.sum().backward()
        for t in (k, v, q, d, e, w):
            t.grad = None
    return go


def step_seq_e97_fwd(inp):
    # OLD sequential E97 Triton kernel (the T-scan that produced the 2.6x claim).
    # FORWARD ONLY — the backward is a symmetric T-scan of comparable cost.
    B, T, H, N = inp['k'].shape
    V = inp['v'].shape[-1]
    kT = inp['k'].transpose(0, 1).contiguous()
    vT = inp['v'].transpose(0, 1).contiguous()
    qT = inp['q'].transpose(0, 1).contiguous()
    dT = inp['decay'].transpose(0, 1).contiguous()
    eT = inp['erase'].transpose(0, 1).contiguous()
    wT = inp['wgate'].transpose(0, 1).contiguous()
    S0 = torch.zeros(B, H, N, V, device=DEV, dtype=DT)

    def go():
        e88_triton_forward(S0, kT, vT, qT, dT, g=None,
                           raw_write=False, linear_state=True,
                           erase_gate=eT, value_write_gate=wT)
    return go


def step_gdn(inp):
    # Native GDN-2 recurrence kernel (FLA chunk_gated_delta_rule), fwd+bwd.
    k = inp['k'].clone().requires_grad_(True)
    v = inp['v'].clone().requires_grad_(True)
    q = inp['q'].clone().requires_grad_(True)
    glog = inp['decay'].clamp(1e-4, 1 - 1e-4).log().clone().requires_grad_(True)
    beta = inp['beta'].clone().requires_grad_(True)

    def go():
        out, _ = chunk_gated_delta_rule(q, k, v, glog, beta,
                                        use_qk_l2norm_in_kernel=True,
                                        output_final_state=False)
        out.sum().backward()
        for t in (k, v, q, glog, beta):
            t.grad = None
    return go


def bench_with_util(go, iters=60, warmup=10, sustain_s=2.5):
    for _ in range(warmup):
        go()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        go()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    # SUSTAINED loop for util: keep the GPU continuously busy for sustain_s so the
    # coarse nvidia-smi sampler reads steady state.
    n_sustain = max(200, int(sustain_s * 1e3 / max(ms, 0.05)))
    with UtilSampler(period=0.02) as sampler:
        for _ in range(n_sustain):
            go()
        torch.cuda.synchronize()
    return ms, sampler.stats()


SHAPES = {
    # within-layer head shape: typed_head_mixture n_state=32; e97 head subset.
    'within_layer': dict(B=8, T=512, H=16, N=32, V=32),
    # 1.3B-scale (matches e99_head_kernel_audit microbench geometry).
    'scale_1p3b': dict(B=4, T=2048, H=16, N=32, V=32),
    # 1.3B wider (dim ~2048): more heads.
    'scale_1p3b_wide': dict(B=4, T=2048, H=32, N=32, V=32),
}


def main():
    torch.manual_seed(0)
    results = {}
    for name, shp in SHAPES.items():
        inp = mk_inputs(**shp)
        row = {'shape': shp}
        ms_gdn, u_gdn = bench_with_util(step_gdn(inp))
        row['gdn2_fwdbwd'] = {'ms': round(ms_gdn, 3), 'util': u_gdn}
        ms_f, u_f = bench_with_util(step_fused(inp, C=32))
        row['e97_fused_fwdbwd'] = {'ms': round(ms_f, 3), 'util': u_f,
                                   'slowdown_vs_gdn': round(ms_f / ms_gdn, 2)}
        ms_pt, u_pt = bench_with_util(step_pytorch(inp, C=64))
        row['e97_pytorch_fwdbwd'] = {'ms': round(ms_pt, 3), 'util': u_pt,
                                     'slowdown_vs_gdn': round(ms_pt / ms_gdn, 2)}
        ms_seq, u_seq = bench_with_util(step_seq_e97_fwd(inp), iters=20, warmup=4)
        row['e97_sequential_fwd_only'] = {'ms': round(ms_seq, 3), 'util': u_seq}
        results[name] = row
        print(json.dumps({name: row}, indent=2))

    outp = os.path.join(os.path.dirname(__file__), 'bench_kernel.json')
    json.dump(results, open(outp, 'w'), indent=2)
    print(f"\nWROTE {outp}")


if __name__ == '__main__':
    main()
