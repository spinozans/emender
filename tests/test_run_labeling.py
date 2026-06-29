import argparse
import json

import torch

import train


def _args(tmp_path, **overrides):
    values = {
        "level": "E97",
        "params": "100m",
        "output": str(tmp_path),
        "resume": None,
        "mlp_ratio": 2.2623,
        "gdn2_mlp_ratio": None,
        "e88_raw_write": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.zeros(7, 11))
        self.frozen = torch.nn.Parameter(torch.zeros(3), requires_grad=False)


def test_e97_run_label_uses_emender_and_derived_param_count():
    args = _args("/tmp/unused")

    prefix = train.build_run_label_prefix(args, 1_286_589_072)

    assert prefix == "emender_E97_1.3B"
    assert "100m" not in prefix
    assert not prefix.startswith("levelE97")


def test_setup_output_dir_records_exact_parameter_metadata(tmp_path):
    args = _args(tmp_path)
    model = TinyModel()
    metadata = train.attach_model_run_metadata(args, model)

    output_dir = train.setup_output_dir(args, model_metadata=metadata)

    assert output_dir.name.startswith("emender_E97_80_")
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["run_label_prefix"] == "emender_E97_80"
    assert manifest["model"]["params_arg"] == "100m"
    assert manifest["model"]["total_params"] == 80
    assert manifest["model"]["trainable_params"] == 77


def test_resume_from_old_named_checkpoint_creates_corrected_new_run_label(tmp_path):
    old_ckpt = (
        tmp_path
        / "runs"
        / "levelE97_100m_20260615_211750"
        / "checkpoint_step_150500_loss_3.0442.pt"
    )
    args = _args(tmp_path / "new-runs", resume=str(old_ckpt))
    metadata = {
        "model_family": "emender",
        "level": "E97",
        "params_arg": "100m",
        "derived_param_slug": "1.3B",
        "run_label_prefix": "emender_E97_1.3B",
        "total_params": 1_286_589_072,
        "trainable_params": 1_286_589_072,
    }
    args._model_run_label_prefix = metadata["run_label_prefix"]
    args._model_metadata = metadata

    output_dir = train.setup_output_dir(args, model_metadata=metadata)

    assert output_dir.name.startswith("emender_E97_1.3B_")
    assert "levelE97_100m" not in output_dir.name
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["resume"] == str(old_ckpt)
    assert manifest["model"]["total_params"] == 1_286_589_072
