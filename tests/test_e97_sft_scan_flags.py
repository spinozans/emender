"""CPU contracts for the E97 SFT scan-throughput levers.

Covers the staged throughput-fix patch (workspace
e97-throughput-fix-v1): the ``--projection-chunk-size`` /
``--cuda-graph-chunk-scan`` trainer flags, the fail-closed flag
validation (which must reject mixed geometries before any distributed
initialization), and the ``validate_packed_masks`` plumbing contract
that lets internal model callers skip the redundant per-chunk
reset/valid cross-check (one host sync + small kernels per chunk call)
after the layer validated the full control masks at entry.
"""
import inspect

import pytest
import torch

import scripts.train_e97_4b_pi_sft as sft_trainer
from ndm.models.e97 import E97SplitEditLayer
from ndm.triton.e88_triton_backward import E88TritonFunction, e88_triton
from ndm.triton.e88_triton_forward import e88_triton_forward
from ndm.triton.e88_triton_optimized import e88_triton_optimized_apply
from ndm.triton.e97_sequential import e97_split_edit_triton_apply


def _base_argv(**extra):
    argv = [
        "train_e97_4b_pi_sft.py",
        "--parent-checkpoint", "/dev/null",
        "--parent-sha256", "0" * 64,
        "--source-args-json", "/dev/null",
        "--authority-root", "/dev/null",
        "--authority-sha256", "0" * 64,
        "--pack-root", "/dev/null",
        "--pack-sha256", "0" * 64,
        "--output-root", "/tmp/e97-scan-flags-test",
        "--log-jsonl", "/dev/stdout",
        "--source-commit", "0" * 40,
    ]
    argv.extend(sum(([k, v] for k, v in extra.items()), []))
    return argv


def test_validate_packed_masks_is_threaded_with_default_check():
    for fn in (e88_triton, e88_triton_forward, e88_triton_optimized_apply,
               e97_split_edit_triton_apply):
        signature = inspect.signature(fn)
        parameter = signature.parameters.get("validate_packed_masks")
        assert parameter is not None, fn
        assert parameter.default is True, fn
    # The autograd Function's forward inputs and backward returns must stay
    # arity-matched (one input per returned gradient slot).
    forward_parameters = list(inspect.signature(E88TritonFunction.forward).parameters)
    assert forward_parameters[-1] == "validate_packed_masks"
    assert len(forward_parameters) == 19  # ctx + 18 inputs


def _backward_return_lengths():
    import ast
    import textwrap
    source = textwrap.dedent(inspect.getsource(E88TritonFunction.backward))
    tree = ast.parse(source)
    lengths = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            lengths.append(len(node.value.elts))
    return lengths


def test_e88_triton_backward_returns_match_forward_inputs():
    lengths = _backward_return_lengths()
    assert lengths and all(length == 18 for length in lengths), lengths


def test_scan_graph_defaults_off_and_disabled_path_is_eager():
    layer = E97SplitEditLayer(dim=32, n_heads=2, n_state=8, use_gate=True)
    assert layer.scan_cuda_graph is False
    x = torch.zeros(1, 32, 32)
    (k, v, q, decay, g, _) = layer._compute_projections(x, torch.float32, False)
    result = layer._graphed_split_edit_scan(
        k, v, q, decay, g, torch.zeros(1, 2, 8, 8), k[:, :, 0], v[..., 0],
        None, None, False, False)
    assert result is None


def test_projection_chunk_flag_requires_alignment(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", _base_argv(
        **{"--projection-chunk-size": "33"}))
    with pytest.raises(SystemExit, match="16-token scan alignment"):
        sft_trainer.main()


def test_projection_chunk_flag_must_engage_chunked_path(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", _base_argv(
        **{"--projection-chunk-size": str(1 << 16)}))
    with pytest.raises(SystemExit, match="smaller than context-size"):
        sft_trainer.main()


def test_projection_chunk_flag_must_divide_context(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", _base_argv(
        **{"--projection-chunk-size": "3072"}))
    with pytest.raises(SystemExit, match="divide evenly"):
        sft_trainer.main()


def test_cuda_graph_flag_requires_explicit_chunk(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", _base_argv(
        **{"--cuda-graph-chunk-scan": "1"}))
    with pytest.raises(SystemExit, match="requires an explicit"):
        sft_trainer.main()


def test_cuda_graph_flag_requires_boundary_aware_packs(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", _base_argv(
        **{"--cuda-graph-chunk-scan": "1",
           "--projection-chunk-size": "2048"}))
    with pytest.raises(SystemExit, match="boundary-aware packs"):
        sft_trainer.main()
