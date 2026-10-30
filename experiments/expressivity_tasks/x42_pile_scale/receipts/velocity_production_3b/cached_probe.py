from experiments.expressivity_tasks.x42_pile_scale.velocity_train import train,verify_registration
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Policy
reg='experiments/expressivity_tasks/x42_pile_scale/velocity_registration.json'
verify_registration(reg)
train(Policy(),reg,'experiments/expressivity_tasks/x42_pile_scale/runs/velocity_cached_probe_v1',seconds=120,eval_windows=2)
