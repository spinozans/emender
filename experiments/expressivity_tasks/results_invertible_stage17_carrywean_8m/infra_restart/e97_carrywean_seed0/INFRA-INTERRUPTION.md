# INFRA-INTERRUPTION
2026-10-21 ~21:00 UTC ENOSPC.
{
  "last_full_step": 160667,
  "partial_step": 160668,
  "prefix_diagnostics": {
    "loss": 2.065751314163208,
    "grad_norm": 0.07049132138490677,
    "lr": 0.0009414041425709177
  },
  "completed_replay_updates": 667,
  "checkpoint": {
    "path": "experiments/expressivity_tasks/results_invertible_stage17_carrywean_8m/e97_carrywean_seed0/step_160000.pt",
    "sha256": "d55ce04d3b0a816ed8148bd1358e9ebc4e4718b85b615e32604601c59be83592",
    "step": 160000
  },
  "gpu": 4,
  "optimizer_state_entries": 55,
  "rng_keys_verified": [
    "numpy_rng",
    "torch_rng",
    "cuda_rng"
  ],
  "original_ceiling": 1000000,
  "proof_scope": "Restart from exact checkpoint state; historical post-checkpoint updates superseded. NO at-death-restoration or exact replay claim.",
  "mode": "restart_from_checkpoint",
  "divergence_receipt": "experiments/expressivity_tasks/results_invertible_stage17_carrywean_8m/infra_resume/e97_carrywean_seed0/interrupted.json",
  "divergence_receipt_sha256": "7de99e2661e0a9cfecd2094dc115d1e587ec8aca856ceb64a5c2ad3195699c9f",
  "divergence": "ValueError('replay mismatch step=160022 loss: 2.1394128799438477 != 2.1394338607788086')"
}
Restart from exact checkpoint state; historical post-checkpoint updates superseded. NO at-death-restoration or exact replay claim.
