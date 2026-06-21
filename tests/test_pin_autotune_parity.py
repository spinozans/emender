"""Parity + no-storm tests for ndm.triton.pin_autotune.

The pinned-autotune fix replaces Triton's per-process autotune *config
benchmarking* (the launch+sync storm that deadlocks fresh inits under box
contention) with a committed registry of measured-best configs. A config
controls scheduling/occupancy (block sizes, num_warps, num_stages) — for an
IDENTICAL config the launched kernel is byte-identical, so fwd+bwd are
numerically identical. (Different configs can differ at the bf16/fp32
reduction-order noise floor; the stock autotuner's winner is itself
timing-noise-dependent across processes, so pinning also makes training
bit-reproducible across runs/ranks — a Frontier requirement.)

What we prove here:
  * CPU (always): the registry loads, covers BOTH arms, and the pin code path
    reconstructs exactly the recorded config for every (kernel, shape) entry.
  * GPU (skipped without CUDA / GatedDeltaNet-2): for the production feature
    dim, the pinned path and the native autotuner path with the SAME config
    produce byte-identical fwd+bwd; the pinned config is a legitimate member of
    the kernel's autotune sweep; and under pinning the autotuner NEVER calls
    do_bench (no storm).
"""
import os
import importlib

import pytest

import ndm.triton.pin_autotune as pin

REGISTRY = pin.load_registry()


# --------------------------------------------------------------------------- #
# CPU tests (no GPU required)
# --------------------------------------------------------------------------- #
def test_config_dict_roundtrip():
    import triton
    cfg = triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=8, num_stages=3, num_ctas=1)
    d = pin.config_to_dict(cfg)
    cfg2 = pin.config_from_dict(d)
    assert cfg2.kwargs == cfg.kwargs
    assert cfg2.num_warps == cfg.num_warps
    assert cfg2.num_stages == cfg.num_stages
    assert cfg2.num_ctas == cfg.num_ctas


def test_registry_covers_both_arms():
    # emender (E97) arm: GatedDeltaNet-2 rmsnorm bwd; gdn2-mlp arm: chunk_gdn2.
    assert REGISTRY, "pinned registry is empty — run scripts/capture_pinned_autotune.sh"
    assert "_layer_norm_bwd_kernel" in REGISTRY, "emender-arm rmsnorm bwd missing from registry"
    assert any(k.startswith("chunk_gdn2_") for k in REGISTRY), "gdn2-arm chunk kernels missing"
    # every entry must be a usable config dict
    for name, entries in REGISTRY.items():
        assert entries, f"{name} has no pinned configs"
        for keystr, d in entries.items():
            cfg = pin.config_from_dict(d)
            assert isinstance(cfg.num_warps, int) and cfg.num_warps >= 1


def test_pinned_config_matches_recorded():
    """The pin code path must return exactly the recorded config for each entry.

    This is the rigorous identity proof: pin selects the autotuner-MEASURED
    winning Config => the launched kernel is byte-identical => fwd+bwd identical.
    """
    import json
    for name, entries in REGISTRY.items():
        for keystr, d in entries.items():
            # reconstruct the key tuple from the stored keystr (list of str tokens)
            key = tuple(json.loads(keystr))
            chosen = pin._pinned_config_for(name, key)
            assert chosen is not None, f"no pin for {name} {keystr}"
            assert pin.config_to_dict(chosen) == d, f"pin mismatch for {name} {keystr}"


def test_install_is_idempotent_and_env_gated(monkeypatch):
    from triton.runtime.autotuner import Autotuner
    flag = "_ndm_pin_autotune_installed"

    monkeypatch.setenv("NDM_PIN_TRITON_AUTOTUNE", "1")
    monkeypatch.delenv("NDM_PIN_TRITON_RECORD", raising=False)
    assert pin.install() is True
    assert getattr(Autotuner, flag) == "pin"
    patched = Autotuner.run
    assert pin.install() is True  # idempotent
    assert Autotuner.run is patched

    # master switch off -> original run restored
    monkeypatch.setenv("NDM_PIN_TRITON_AUTOTUNE", "0")
    assert pin.install() is False
    assert getattr(Autotuner, flag) in (None,)
    assert Autotuner.run is pin._ORIG_RUN

    # restore pinned for any later tests in this process
    monkeypatch.setenv("NDM_PIN_TRITON_AUTOTUNE", "1")
    pin.install()


# --------------------------------------------------------------------------- #
# GPU tests (real kernels, production feature dim)
# --------------------------------------------------------------------------- #
def _load_gdn2_rmsnorm():
    # Reuse the production loader, which sets up a synthetic `_external_gdn2_lit_gpt`
    # package whose __path__ points at GatedDeltaNet-2/lit_gpt and imports submodules
    # by file (bypassing lit_gpt/__init__.py, which would pull in flash_attn). This
    # gives the SAME rmsnorm module (and its _layer_norm_* autotuned kernels) the
    # production gdn2/emender models use.
    from ndm.models import external_gdn2
    external_gdn2._load_gdn2_class()
    return importlib.import_module("_external_gdn2_lit_gpt.rmsnorm")


cuda = pytest.importorskip("torch").cuda
pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="needs CUDA")


def _run_rmsnorm_fwd_bwd(mod, D, seed=0):
    import torch
    torch.manual_seed(seed)
    x = torch.randn(8, 256, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(8, 256, D, device="cuda", dtype=torch.bfloat16)
    out = mod.rms_norm_fn(x, w, None, eps=1e-6)
    out.backward(g)
    return out.detach().clone(), x.grad.detach().clone(), w.grad.detach().clone()


@pytest.mark.parametrize("D", [1792, 2176])
def test_pin_path_byte_identical_to_native_same_config(monkeypatch, D):
    """Pin path vs native autotuner with the SAME config => byte-identical fwd+bwd."""
    import torch
    mod = _load_gdn2_rmsnorm()
    fwd_k = mod._layer_norm_fwd_1pass_kernel
    bwd_k = mod._layer_norm_bwd_kernel
    fwd_cfgs, bwd_cfgs = list(fwd_k.configs), list(bwd_k.configs)

    # 1) PINNED path: pin selects the registry (measured-best) config.
    monkeypatch.setenv("NDM_PIN_TRITON_AUTOTUNE", "1")
    monkeypatch.delenv("NDM_PIN_TRITON_RECORD", raising=False)
    pin.install()
    fwd_k.cache.clear(); bwd_k.cache.clear()
    out_pin, gx_pin, gw_pin = _run_rmsnorm_fwd_bwd(mod, D)
    win_fwd, win_bwd = fwd_k.best_config, bwd_k.best_config
    # the pinned config must be a legitimate member of the kernel's sweep
    sweep_fwd = {pin._keystr([c.num_warps, c.num_stages, c.num_ctas, sorted(c.kwargs.items())]) for c in fwd_cfgs}
    assert pin._keystr([win_fwd.num_warps, win_fwd.num_stages, win_fwd.num_ctas, sorted(win_fwd.kwargs.items())]) in sweep_fwd

    # 2) NATIVE path: restrict the autotuner to ONLY that winning config and run
    #    with pinning OFF (single-config => no benchmark, native fn.run).
    monkeypatch.setenv("NDM_PIN_TRITON_AUTOTUNE", "0")
    pin.install()  # restores original Autotuner.run
    try:
        fwd_k.configs = [win_fwd]; bwd_k.configs = [win_bwd]
        fwd_k.cache.clear(); bwd_k.cache.clear()
        out_ref, gx_ref, gw_ref = _run_rmsnorm_fwd_bwd(mod, D)
    finally:
        fwd_k.configs = fwd_cfgs; bwd_k.configs = bwd_cfgs
        monkeypatch.setenv("NDM_PIN_TRITON_AUTOTUNE", "1")
        pin.install()

    assert torch.equal(out_pin, out_ref), f"D={D}: fwd output differs for identical config"
    assert torch.equal(gx_pin, gx_ref), f"D={D}: dx differs for identical config"
    assert torch.equal(gw_pin, gw_ref), f"D={D}: dw differs for identical config"


def test_pin_never_benchmarks(monkeypatch):
    """Under pinning the autotuner must NOT call do_bench (proves: no storm)."""
    mod = _load_gdn2_rmsnorm()
    bwd_k = mod._layer_norm_bwd_kernel
    fwd_k = mod._layer_norm_fwd_1pass_kernel

    monkeypatch.setenv("NDM_PIN_TRITON_AUTOTUNE", "1")
    monkeypatch.delenv("NDM_PIN_TRITON_RECORD", raising=False)
    pin.install()
    fwd_k.cache.clear(); bwd_k.cache.clear()

    calls = {"n": 0}
    orig_bench = bwd_k._bench

    def boom(*a, **k):
        calls["n"] += 1
        return orig_bench(*a, **k)

    monkeypatch.setattr(bwd_k, "_bench", boom)
    monkeypatch.setattr(fwd_k, "_bench", boom)
    _run_rmsnorm_fwd_bwd(mod, 1792)
    assert calls["n"] == 0, "pinned mode benchmarked configs (storm not killed)"
