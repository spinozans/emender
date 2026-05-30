"""
Optional external GDN-2 baseline.

This module loads NVIDIA's GatedDeltaNet-2 layer from a local checkout instead
of vendoring the non-commercially licensed source into this repository. Set
``GDN2_PATH`` to override the default checkout location.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

import torch.nn as nn


DEFAULT_GDN2_PATH = "/home/erikg/GatedDeltaNet-2"
_GDN2_CLASS = None


def _external_root() -> Path:
    return Path(os.environ.get("GDN2_PATH", DEFAULT_GDN2_PATH)).expanduser().resolve()


def _ensure_fla_cache_helpers() -> None:
    """Install cache helper aliases expected by the external GDN-2 checkout."""
    try:
        layer_utils = importlib.import_module("fla.layers.utils")
    except ImportError:
        return

    if not hasattr(layer_utils, "get_layer_cache"):

        def get_layer_cache(layer, past_key_values):
            layer_idx = getattr(layer, "layer_idx", None)
            if (
                past_key_values is None
                or layer_idx is None
                or len(past_key_values) <= layer_idx
            ):
                return None
            return past_key_values[layer_idx]

        layer_utils.get_layer_cache = get_layer_cache

    if not hasattr(layer_utils, "update_layer_cache"):

        def update_layer_cache(layer, past_key_values, **cache_kwargs):
            layer_idx = getattr(layer, "layer_idx", None)
            if past_key_values is None or layer_idx is None:
                return None
            return past_key_values.update(layer_idx=layer_idx, **cache_kwargs)

        layer_utils.update_layer_cache = update_layer_cache


def _ensure_fla_kernel_helpers() -> None:
    """Install import shims for external GDN-2 kernel helpers missing in newer FLA."""
    try:
        ops_module = importlib.import_module("fla.ops")
    except ImportError:
        return

    if "fla.ops.backends" not in sys.modules:
        backends_module = types.ModuleType("fla.ops.backends")

        def dispatch(_backend):
            def decorator(func):
                return func

            return decorator

        backends_module.dispatch = dispatch
        sys.modules["fla.ops.backends"] = backends_module
        setattr(ops_module, "backends", backends_module)

    if "fla.ops.cp" not in sys.modules:
        cp_module = types.ModuleType("fla.ops.cp")
        cp_module.__path__ = []

        class FLACPContext:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "This installed FLA package does not provide context-parallel helpers."
                )

        cp_module.FLACPContext = FLACPContext
        sys.modules["fla.ops.cp"] = cp_module
        setattr(ops_module, "cp", cp_module)

    if "fla.ops.cp.comm" not in sys.modules:
        comm_module = types.ModuleType("fla.ops.cp.comm")

        def all_gather_into_tensor(*args, **kwargs):
            raise ImportError(
                "This installed FLA package does not provide context-parallel helpers."
            )

        comm_module.all_gather_into_tensor = all_gather_into_tensor
        sys.modules["fla.ops.cp.comm"] = comm_module
        setattr(sys.modules["fla.ops.cp"], "comm", comm_module)


def _ensure_fla_compat() -> None:
    _ensure_fla_cache_helpers()
    _ensure_fla_kernel_helpers()


def _load_gdn2_class():
    global _GDN2_CLASS
    if _GDN2_CLASS is not None:
        return _GDN2_CLASS

    root = _external_root()
    gdn2_file = root / "lit_gpt" / "gdn2.py"
    if not gdn2_file.exists():
        raise ImportError(
            f"GDN-2 checkout not found at {root}. "
            "Clone https://github.com/NVlabs/GatedDeltaNet-2 or set GDN2_PATH."
        )

    package_name = "_external_gdn2_lit_gpt"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(root / "lit_gpt")]
        sys.modules[package_name] = pkg

    module_name = f"{package_name}.gdn2"
    _ensure_fla_compat()
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, gdn2_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load GDN-2 module from {gdn2_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    ops_module = importlib.import_module(f"{package_name}.gdn2_ops.chunk_gdn2")
    original = ops_module.chunk_gla_fwd_o_gk
    if not getattr(original, "_emender_gdn2_compat", False):
        signature = inspect.signature(original)

        def compat_chunk_gla_fwd_o_gk(*args, use_exp2=None, transpose_state_layout=False, **kwargs):
            if "use_exp2" in signature.parameters:
                kwargs["use_exp2"] = use_exp2
            if "transpose_state_layout" in signature.parameters:
                kwargs["transpose_state_layout"] = transpose_state_layout
            elif "state_v_first" in signature.parameters:
                kwargs["state_v_first"] = transpose_state_layout
            return original(*args, **kwargs)

        compat_chunk_gla_fwd_o_gk._emender_gdn2_compat = True
        ops_module.chunk_gla_fwd_o_gk = compat_chunk_gla_fwd_o_gk

    _GDN2_CLASS = module.GatedDeltaNet2
    return _GDN2_CLASS


class GDN2ExternalLayer(nn.Module):
    """LadderLM-compatible wrapper around an external GatedDeltaNet-2 layer."""

    def __init__(
        self,
        dim,
        expansion=2.0,
        dropout=0.0,
        head_dim=128,
        num_heads=None,
        use_conv=None,
        d_conv=4,
        allow_neg_eigval=False,
        **kwargs,
    ):
        super().__init__()
        GatedDeltaNet2 = _load_gdn2_class()

        if num_heads is None:
            num_heads = kwargs.get("n_heads")
        if use_conv is None or use_conv is False:
            use_conv = True

        if head_dim > 256:
            raise ValueError("GDN-2 kernels require head_dim <= 256")
        if num_heads is None:
            num_heads = max(1, dim // head_dim)

        self.dim = dim
        self.expansion = expansion
        self.num_heads = num_heads
        self.actual_head_dim = head_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gdn2 = GatedDeltaNet2(
            hidden_size=dim,
            expand_v=expansion,
            head_dim=self.actual_head_dim,
            num_heads=num_heads,
            mode="chunk",
            use_short_conv=use_conv,
            conv_size=d_conv,
            allow_neg_eigval=allow_neg_eigval,
            layer_idx=0,
        )

    def set_layer_idx(self, idx):
        self.gdn2.layer_idx = idx

    def forward(self, x, h0=None, **kwargs):
        if self.training:
            output, _, _ = self.gdn2(x, use_cache=False)
            return self.dropout(output), None

        output, _, cache = self.gdn2(x, past_key_values=h0, use_cache=True)
        return self.dropout(output), cache

    def extra_repr(self):
        return (
            f"dim={self.dim}, expansion={self.expansion}, "
            f"num_heads={self.num_heads}, head_dim={self.actual_head_dim}, "
            "external=GDN2"
        )


def count_gdn2_external_params(dim, depth, vocab_size=256, expansion=2.0, num_heads=None, head_dim=128):
    """Approximate LadderLM parameter count for the external GDN-2 layer."""
    if num_heads is None:
        num_heads = max(1, dim // head_dim)
    value_head_dim = int(head_dim * expansion)
    key_dim = num_heads * head_dim
    value_dim = num_heads * value_head_dim

    per_layer = (
        dim * key_dim * 2
        + dim * value_dim
        + dim * key_dim
        + dim * value_dim
        + dim * value_head_dim
        + value_head_dim * key_dim
        + dim * value_head_dim
        + value_head_dim * value_dim
        + value_dim * dim
        + (key_dim * 2 + value_dim) * 4
        + num_heads
        + key_dim
        + 2 * num_heads * value_head_dim
        + dim
    )
    return vocab_size * dim + depth * per_layer + dim
