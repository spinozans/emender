# GDN-2 and E97 Notes

Date: 2026-05-28

## Sources

- Paper: "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention",
  arXiv:2605.22791.
- Reference repo: `github.com/NVlabs/GatedDeltaNet-2`, inspected locally at
  `/home/erikg/GatedDeltaNet-2`.

The released GDN-2 code is under NVIDIA Source Code License-NC. Do not vendor
it into Emender. Treat it as a research reference or optional external import
boundary.

## Mechanism

GDN-2 keeps the linear-attention fast-weight frame but splits the memory edit
into two independent gates:

```text
S_t = (I - k_t (b_t * k_t)^T) D_t S_{t-1}
      + k_t (w_t * v_t)^T
```

`b_t` is a channel-wise erase gate on the key axis. `w_t` is a channel-wise
write gate on the value axis. This replaces the older single scalar beta that
had to mean both erase strength and write strength.

This matters for Emender because current E88 has only:

```text
r_t = S_{t-1}^T k_t
delta_t = v_t - r_t
S_t = tanh(d_t S_{t-1} + k_t delta_t^T)
```

The existing `--use_write_gate` ablation only gates the delta write with one
scalar per head. It does not add GDN-2's key-axis erase/read gate.

## E97 Candidate

E97 should be a clean Emender/NDM variant inspired by GDN-2, not a code copy:

```text
r_t = S_{t-1}^T (b_t * k_t)
delta_t = (w_t * v_t) - r_t
S_t = tanh(d_t S_{t-1} + k_t delta_t^T)
y_t = S_t^T q_t
```

Conservative first implementation:

- keep the E88 output path, residual wrapper, and scalar per-head decay.
- add `b_proj: dim -> H * n_state` for key-axis erase/read gating.
- add `w_proj: dim -> H * head_v_dim` for value-axis write gating.
- implement the PyTorch/reference path first.
- wire it as an explicit level/model type, probably `E97` and `e97`.
- do not write a Triton kernel until the reference path shows signal.

Status as of 2026-05-28: the reference path is wired as `--level E97` by
reusing E88/NDM with `use_split_edit=True`. A split-edit Triton recurrence is
also wired behind `--use_triton 1`; focused forward/backward parity tests pass
against the PyTorch reference. Full 1.27B / 2K-context CMA should still wait
for a small CLI training smoke and GPU assignment check.

Runtime check on 2026-05-28:

- reference E97 at `B=2,T=512,dim=256,depth=2,H=32,N=16`: about 1.55s/step.
- Triton E97 at the same shape: about 0.006-0.007s/step after warmup.
- Triton E97 at `B=2,T=2048,dim=256,depth=2,H=32,N=16`: about 0.016s/step.
- Triton E97 at high head count
  `B=1,T=2048,dim=1664,depth=1,H=370,N=32`: about 0.028s/step,
  peak allocation about 1.6GB.

Useful ablations:

- `E97-erase`: key-axis erase gate only, no value write gate.
- `E97-write`: value write gate only.
- `E97-linear`: split gates but no recurrent tanh.
- `E97-raw`: split gates with raw write instead of delta correction.
- `E97-channel-decay`: GDN-2-style channel-wise decay.

## GDN-2 Baseline

There are two routes:

1. Optional external wrapper around `/home/erikg/GatedDeltaNet-2`.
2. Clean local implementation of the published recurrence.

Route 1 is useful for local research, but currently needs dependency work:
the reference repo expects newer `flash-linear-attention` APIs and `flash_attn`.
Route 2 is better for the public repo because it avoids license contamination
and keeps the comparison auditable.

## Experiment Plan

First pass:

- implement `E97` in PyTorch.
- add parameter counting and `cmaes_search_v2.py` support in the research repo
  before promoting it here.
- run short 2K-context CMA/probe jobs at 1.3B scale.
- run expressivity probes, especially S5/state tracking, against E88-delta,
  E88-raw, M2RNN, GDN, and Mamba2.

Decision rule:

If E97 improves either 2K-context early loss at matched wallclock or the S5
state-tracking panel, it earns a Triton kernel. If not, keep it as a documented
negative result in the update-rule menu.
