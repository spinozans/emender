"""emender-real-cap — throughput of the REAL Emender vs pure GDN-2 at the 1.3B head shape.

The REAL Emender = a SEA of gdn2_recall heads (neg-eig -> recall+track) + a SMALL
fraction (4/64, 8/64) of NONLINEAR EMENDMENT heads. The capability-adding emendment
head is the e97_delta SPLIT-EDIT cell with a per-step bounded (tanh) state map
(phi-explore: bounded saturation on split-edit unlocks the depth capability; it is
"nearly inert on gated-delta" -> the gdn2_nonlin_shell does NOT add it). The tanh
split-edit head is SEQUENTIAL (non-chunkable, fuse-2kernel), so its per-layer latency
must be HIDDEN under the tensor-core chunkable bulk via stream overlap to keep
throughput ~0.95x GDN-2 (the HETERO_KERNEL_NOTE lever).

This bench measures, at the 1.3B head shape (dim=2240, depth=18, 64 heads, n_state=32,
bf16, REAL LadderLM, REAL token batches, fwd+bwd):
  - gdn_pure                          (64 gdn2_recall = GDN-2 ceiling)
  - emender{4,8}  (e97_delta-tanh)    overlap on/off  <- the REAL capability Emender
  - shell{4,8}    (gdn2_nonlin_shell) overlap on      <- the throughput-friendly head
                                                          (note's 0.951x; capability-inert)
asserting the fused Triton path engages with NO eager fallback. Writes throughput.json.

  eval "$(scripts/gpu_lease.sh 1)" && python experiments/emender_real_cap/throughput.py
"""
import os, sys, json, time, argparse
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)
import math
import torch
from ndm.models.ladder_lm import LadderLM
from ndm.models.typed_head_mixture import allocate_types

VOCAB = 50281
DIM = 2240                       # the hetero-note 1.3B head shape (dim=2240, ~1.3B)
BASE = dict(depth=18, n_heads=64, n_state=32, expansion=1.0, mlp_ratio=6208 / 2304)
# canonical 9-slot type order; idx0=gdn2_recall idx5=gdn2_nonlin_shell idx7=e97_delta
LOG0 = -30.0


def logits9(counts: dict) -> list:
    names = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin',
             'gdn2_nonlin_shell', 'e97_raw', 'e97_delta', 'refit']
    return [math.log(counts[t]) if counts.get(t, 0) > 0 else LOG0 for t in names]


def build(label, counts, overlap, head):
    """head: 'e97_delta' (split-edit tanh, sequential) or 'shell' (gdn2_nonlin_shell)."""
    lk = dict(head_type_logits=logits9(counts), gdn_allow_neg_eigval=True,
              overlap_streams=overlap, use_triton_e97=True,
              lam_max=1.585, beta_max=2.747)
    if head == 'e97_delta':
        # tanh split-edit = the depth-capability emendment head, kept SEQUENTIAL
        lk.update(e97_state_nonlin='tanh', use_chunked_e97_delta=False)
    else:
        lk.update(shell_state_nonlin='tanh', shell_state_chunk=64, shell_fused=True)
    m = LadderLM(vocab_size=VOCAB, dim=DIM, depth=BASE['depth'], level='typed-gdn2-lm',
                 n_heads=BASE['n_heads'], n_state=BASE['n_state'],
                 expansion=BASE['expansion'], layer_kwargs=lk, mlp_ratio=BASE['mlp_ratio'])
    return m


def assert_fused_seq(model, head):
    """Assert the nonlinear emendment heads are present, FUSED, and (for e97_delta-tanh)
    on the SEQUENTIAL split-edit path. Fail loud on any eager / missing head."""
    n_layers = 0
    for _, mod in model.named_modules():
        if mod.__class__.__name__ == 'TypedHeadMixtureLayer':
            n_layers += 1
            if getattr(mod, 'use_triton_e97', None) is not True:
                raise RuntimeError(f"FUSED-ASSERT: use_triton_e97 != True (eager forbidden)")
            if head == 'e97_delta':
                if mod.e97_delta is None:
                    raise RuntimeError("FUSED-ASSERT: no e97_delta head allocated")
                if not mod._e97_delta_is_seq():
                    raise RuntimeError("FUSED-ASSERT: e97_delta not on sequential split-edit "
                                       "path (e97_state_nonlin must be tanh)")
    if n_layers == 0:
        raise RuntimeError("FUSED-ASSERT: no TypedHeadMixtureLayer found")
    return n_layers


def tok_s(m, B=2, T=2048, n=15, w=6):
    x = torch.randint(0, VOCAB, (B, T), device='cuda')
    for _ in range(w):
        l = m(x, return_loss=True); l = l[0] if isinstance(l, tuple) else l
        l.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        l = m(x, return_loss=True); l = l[0] if isinstance(l, tuple) else l
        l.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return B * T * n / (time.time() - t0)


def run(label, counts, overlap, head, ceil=None):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = build(label, counts, overlap, head).cuda().bfloat16()
    nlay = assert_fused_seq(m, head) if head else None
    alloc = None
    for _, mod in m.named_modules():
        if hasattr(mod, 'alloc'):
            alloc = {k: v for k, v in mod.alloc['counts'].items() if v > 0}; break
    t = tok_s(m)
    peak = torch.cuda.max_memory_allocated() / 1e6
    del m; torch.cuda.empty_cache()
    ratio = (t / ceil) if ceil else 1.0
    print(f"  {label:32s} {t:>8.1f} tok/s  ratio={ratio:.3f}  peak={peak:>6.0f}MB  {alloc}",
          flush=True)
    return dict(label=label, head=head, overlap=overlap, tok_s=round(t, 1),
                ratio_vs_gdn2=round(ratio, 4), peak_mb=round(peak, 1), counts=alloc,
                fused_layers=nlay)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(_THIS, 'throughput.json'))
    args = ap.parse_args()
    assert torch.cuda.is_available(), "need a leased GPU"
    print(f"Emender throughput @ 1.3B head shape dim={DIM} {BASE}", flush=True)
    rows = []
    ceil = run('gdn_pure (GDN-2 ceiling)', {'gdn2_recall': 64}, False, None)
    base = ceil['tok_s']; ceil['ratio_vs_gdn2'] = 1.0
    rows.append(ceil)
    print("=== REAL Emender: e97_delta split-edit (tanh, sequential capability head) ===")
    for n in (4, 8):
        for ov in (False, True):
            rows.append(run(f'emender{n} e97d-tanh ov={int(ov)}',
                            {'gdn2_recall': 64 - n, 'e97_delta': n}, ov, 'e97_delta', base))
    print("=== comparison: gdn2_nonlin_shell (throughput-friendly, capability-inert) ===")
    for n in (4, 8):
        rows.append(run(f'shell{n} ov=1',
                        {'gdn2_recall': 64 - n, 'gdn2_nonlin_shell': n}, True, 'shell', base))
    with open(args.out, 'w') as f:
        json.dump(dict(dim=DIM, base=BASE, gdn2_tok_s=base, rows=rows), f, indent=2)
    print(f"\nwrote {args.out}", flush=True)
