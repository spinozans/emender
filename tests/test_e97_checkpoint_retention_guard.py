import importlib.util
import os
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e97_checkpoint_retention_guard.py"
SPEC = importlib.util.spec_from_file_location("e97_checkpoint_retention_guard", SCRIPT)
guard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


def _ckpt(run_dir: Path, step: int, *, size: int = 100, mtime: int) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"checkpoint_step_{step:06d}_loss_1.0000.pt"
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def test_retention_plan_keeps_required_checkpoints_and_prunes_dense_middle(tmp_path):
    root = tmp_path / "emender"
    old_run = root / "runs" / "levelE97_100m_20260621_133317"
    active_run = root / "runs" / "levelE97_100m_20260623_103742"
    old_run.mkdir(parents=True)
    active_run.mkdir(parents=True)

    for i, step in enumerate(range(62_500, 72_500, 500)):
        _ckpt(old_run, step, mtime=1_000 + i)
    resume = _ckpt(root / "runs" / "levelE97_100m_20260622_101547", 72_500, mtime=2_000)

    for i, step in enumerate(range(95_500, 105_500, 500)):
        _ckpt(active_run, step, mtime=3_000 + i)
    latest = active_run / "latest.pt"
    latest.symlink_to("checkpoint_step_105000_loss_1.0000.pt")

    plan = guard.build_plan(
        root,
        latest_active=5,
        milestone_every=10_000,
        critical_steps={500, 72_000, 72_500},
        min_age_seconds=100,
        extra_keep_paths={resume},
        now=10_000,
    )

    kept_names = {path.name for path in plan.keep}
    assert "checkpoint_step_072000_loss_1.0000.pt" in kept_names
    assert "checkpoint_step_072500_loss_1.0000.pt" in kept_names
    assert "checkpoint_step_100000_loss_1.0000.pt" in kept_names
    assert "checkpoint_step_105000_loss_1.0000.pt" in kept_names
    assert {f"checkpoint_step_{step:06d}_loss_1.0000.pt" for step in range(103_000, 105_500, 500)} <= kept_names
    assert "checkpoint_step_062500_loss_1.0000.pt" in kept_names

    deleted_names = {ckpt.name for ckpt in plan.delete}
    assert "checkpoint_step_063000_loss_1.0000.pt" in deleted_names
    assert "checkpoint_step_099500_loss_1.0000.pt" in deleted_names
    assert "checkpoint_step_105000_loss_1.0000.pt" not in deleted_names
    assert len(plan.delete) == 30


def test_retention_plan_skips_non_modal_size_and_too_new_candidates(tmp_path):
    root = tmp_path / "emender"
    run = root / "runs" / "levelE97_100m_20260623_103742"
    run.mkdir(parents=True)

    _ckpt(run, 1_000, mtime=1_000)
    odd = _ckpt(run, 1_500, size=17, mtime=1_100)
    fresh = _ckpt(run, 2_000, mtime=9_950)
    newest = _ckpt(run, 2_500, mtime=10_000)

    plan = guard.build_plan(
        root,
        latest_active=1,
        milestone_every=10_000,
        critical_steps=set(),
        min_age_seconds=100,
        extra_keep_paths=set(),
        now=10_000,
    )

    assert newest in [ckpt.path for ckpt in guard.discover_checkpoints(root)]
    assert odd not in [ckpt.path for ckpt in plan.delete]
    assert fresh not in [ckpt.path for ckpt in plan.delete]
    assert plan.skip[odd] == "non-modal-or-unknown-complete-size"
    assert plan.skip[fresh].startswith("younger-than-min-age:")
