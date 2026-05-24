# Standalone Usage — ndm

## Install

```bash
pip install ndm
```

Or from source (development install):

```bash
git clone https://github.com/ekg/ndm
pip install -e ndm/
```

## Minimal forward-pass example

```python
import torch
import ndm

print(ndm.__version__)  # "0.2.0"

from ndm import LadderLM

# Tiny CPU-feasible model (level=0 = StockElman recurrent layer)
model = LadderLM(vocab_size=256, dim=64, depth=2, level=0)
model.eval()

x = torch.randint(0, 256, (1, 16))   # [batch=1, seq_len=16]
with torch.no_grad():
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out

print(logits.shape)   # torch.Size([1, 16, 256])
```

## Production model (E88FusedLM / NDM)

`E88FusedLM` is the production NDM implementation. It has a pure-PyTorch fallback that runs on CPU, making it usable without CUDA for quick checks:

```python
import torch
from ndm.models.e88_fused import E88FusedLM, E88_FUSED_AVAILABLE

print("CUDA kernel available:", E88_FUSED_AVAILABLE)  # requires hasty_pytorch_lib

# Small CPU-feasible config: n_heads * n_state must equal dim
model = E88FusedLM(vocab_size=256, dim=64, depth=2, n_heads=4, n_state=16)
model.eval()

x = torch.randint(0, 256, (1, 8))
with torch.no_grad():
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out

print(logits.shape)   # torch.Size([1, 8, 256])
```

## Public API

All symbols exported from `ndm.__init__`:

| Symbol | Description |
|--------|-------------|
| `StockElman`, `StockElmanCell` | Level-0 base recurrent layer |
| `LadderLM` | Language model wrapper for all E-series levels |
| `create_ladder_model` | Convenience factory (size string + level) |
| `get_available_levels` | Dict of available ladder levels |
| `get_ladder_level` | Get layer class by level number |

`E88FusedLM` is accessible via `from ndm.models.e88_fused import E88FusedLM` (not yet re-exported from `ndm.__init__`).

## Known issues

### `LadderLM` CPU usage requires `mamba_ssm` to be absent

When `mamba_ssm` is installed in the same environment (it is a development dependency, not listed in `pyproject.toml`), `LadderLM` auto-selects its Triton fused-norm kernel which requires CUDA tensors. A clean `pip install ndm` without `mamba_ssm` falls back to `torch.nn.RMSNorm` and runs fine on CPU.

**Workaround for developers**: use a separate venv without `mamba_ssm`, or move tensors to CUDA.

### `E88FusedLM` CUDA kernel requires `hasty_pytorch_lib`

The fast CUDA kernel path requires the `hasty_pytorch_lib` extension (not on PyPI). Without it, the model transparently falls back to a pure-PyTorch loop. This fallback is CPU-compatible and correct, just slower for long sequences.

### `train.py` is at the repo root, not inside the package

`train.py` is a script, not a package module. It is not installed by `pip install ndm` and does not need to be. Users who want to train should clone the repo.

## Running the smoke test

```bash
python -m pytest tests/test_standalone_minimal.py -m 'not gpu'
```

In a clean install without `mamba_ssm`, all 5 tests pass including `test_ladderlm_cpu_forward`. In a developer environment with `mamba_ssm`, that test is skipped with an explanation; the other 4 tests pass.
