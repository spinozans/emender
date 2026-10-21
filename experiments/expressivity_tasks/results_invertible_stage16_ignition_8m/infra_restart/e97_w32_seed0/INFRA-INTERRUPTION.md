# INFRA-INTERRUPTION
2026-10-21 ~21:00 UTC ENOSPC.
{
  "last_full_step": 245475,
  "partial_step": 245476,
  "prefix_diagnostics": {
    "loss": 8.220682144165039,
    "grad_norm": 0.047095607966184616,
    "lr": 0.0005270592388024061
  },
  "completed_replay_updates": 475,
  "checkpoint": {
    "path": "experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/e97_w32_seed0/T16/step_245000.pt",
    "sha256": "2812c3ca67a66a840d38185a2050f3a918470dcd54ebf3f94b03d20d3be54fa9",
    "step": 245000
  },
  "gpu": 1,
  "optimizer_state_entries": 54,
  "rng_keys_verified": [
    "numpy_rng",
    "torch_rng",
    "cuda_rng"
  ],
  "original_ceiling": 500000,
  "proof_scope": "Restart from exact checkpoint state; historical post-checkpoint updates superseded. NO at-death-restoration or exact replay claim.",
  "mode": "restart_from_checkpoint",
  "divergence_receipt": "experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/infra_resume/e97_w32_seed0/interrupted.json",
  "divergence_receipt_sha256": "1dc9aedab9ca61b9a3a2b51bbb369e1db399e7bd0fe5c07004253de289bd7f61",
  "divergence": "ValueError('replay mismatch step=245104 grad_norm: 0.03506257385015488 != 0.03506257012486458')"
}
Restart from exact checkpoint state; historical post-checkpoint updates superseded. NO at-death-restoration or exact replay claim.
