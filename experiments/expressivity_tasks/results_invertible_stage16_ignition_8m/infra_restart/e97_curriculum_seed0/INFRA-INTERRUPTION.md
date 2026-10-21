# INFRA-INTERRUPTION
2026-10-21 ~21:00 UTC ENOSPC.
{
  "last_full_step": 252678,
  "partial_step": 252679,
  "prefix_diagnostics": {
    "loss": 7.193890571594238,
    "grad_norm": 0.09980139881372452,
    "lr": 0.000504437655035969
  },
  "completed_replay_updates": 678,
  "checkpoint": {
    "path": "experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/e97_curriculum_seed0/T16/step_252000.pt",
    "sha256": "bcf96443ac0090ef66c317ba3b92db037f4be918325e81739e040681405e9886",
    "step": 252000
  },
  "gpu": 6,
  "optimizer_state_entries": 54,
  "rng_keys_verified": [
    "numpy_rng",
    "torch_rng",
    "cuda_rng"
  ],
  "original_ceiling": 500000,
  "proof_scope": "Restart from exact checkpoint state; historical post-checkpoint updates superseded. NO at-death-restoration or exact replay claim.",
  "mode": "restart_from_checkpoint",
  "divergence_receipt": "experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/infra_resume/e97_curriculum_seed0/interrupted.json",
  "divergence_receipt_sha256": "185129e50c0ddbc29375aa75a7ebd7490670edacd36b22f7767845fdbe096f11",
  "divergence": "ValueError('replay mismatch step=252204 grad_norm: 0.09198356419801712 != 0.09198416024446487')"
}
Restart from exact checkpoint state; historical post-checkpoint updates superseded. NO at-death-restoration or exact replay claim.
