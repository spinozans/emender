import pytest


pytest.importorskip("cma")
pytest.importorskip("tiktoken")


def test_cmaes_estimator_uses_tokenizer_vocab(monkeypatch):
    import scripts.cmaes_search_v2 as cmaes
    from scripts.calc_dim import calc_e88_params

    params = {
        "dim": 2048,
        "depth": 10,
        "n_heads": 348,
        "n_state": 32,
        "expansion": 1.0,
    }

    monkeypatch.setattr(cmaes, "PARAM_VOCAB_SIZE", 256)
    byte_count = cmaes.estimate_params_for_config(params, "e88")

    vocab_size = cmaes.resolve_vocab_size("p50k_base")
    monkeypatch.setattr(cmaes, "PARAM_VOCAB_SIZE", vocab_size)
    bpe_count = cmaes.estimate_params_for_config(params, "e88")

    assert vocab_size == 50281
    assert bpe_count - byte_count == (vocab_size - 256) * params["dim"]
    assert bpe_count == calc_e88_params(**params, vocab_size=vocab_size)


def test_gdn2_wrapper_accepts_n_heads_alias():
    pytest.importorskip("fla")

    from pathlib import Path

    if not Path("/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2.py").exists():
        pytest.skip("external GDN-2 checkout is not available")

    from ndm.models.external_gdn2 import GDN2ExternalLayer

    layer = GDN2ExternalLayer(dim=128, expansion=1, head_dim=16, n_heads=3)

    assert layer.num_heads == 3
    assert layer.gdn2.num_heads == 3


def test_cmaes_train_command_passes_compile_warmup_flags(monkeypatch, tmp_path):
    import scripts.cmaes_search_v2 as cmaes

    params = {
        "dim": 256,
        "depth": 2,
        "n_heads": 8,
        "n_state": 16,
        "batch_size": 1,
        "lr": 1e-3,
    }

    monkeypatch.setattr(cmaes, "COMPILE_WARMUP_STEPS", 2)
    monkeypatch.setattr(cmaes, "TIMER_AFTER_COMPILE_WARMUP", True)

    cmd, _ = cmaes.build_train_command(params, "e88", 0.1, str(tmp_path))

    assert "--compile_warmup_steps" in cmd
    assert cmd[cmd.index("--compile_warmup_steps") + 1] == "2"
    assert "--timer_after_compile_warmup" in cmd


def test_cmaes_skip_memory_probe_runs_requested_batch(monkeypatch, tmp_path):
    import subprocess

    import scripts.cmaes_search_v2 as cmaes

    calls = []

    def fail_probe(*args, **kwargs):
        pytest.fail("memory probe should not run when SKIP_MEMORY_PROBE is enabled")

    def fake_run(cmd, capture_output, text, timeout, env, cwd):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(cmaes, "SKIP_MEMORY_PROBE", True)
    monkeypatch.setattr(cmaes, "probe_max_batch_size", fail_probe)
    monkeypatch.setattr(cmaes.subprocess, "run", fake_run)

    batch_size, result = cmaes.find_max_batch_size(
        ["python", "train.py"],
        env={},
        cwd=str(tmp_path),
        timeout=1,
        target_bs=3,
    )

    assert batch_size == 3
    assert result.returncode == 0
    assert calls == [["python", "train.py", "--batch_size", "3"]]
