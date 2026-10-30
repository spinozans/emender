# Final recovery: evaluation failed; board remains open

The depth fine-tune completed 30000/30000 updates. The unchanged checkpoint SHA256 is
`a64b5bf3c5cd4970c6e6a371f065f99eff62ccc4b4c7cadbc94387c971eca47c`.
Validation balanced accuracy: update 9000 = 0.997; update 30000 = 0.9975.
The saved best-validation weights selected for evaluation are update 28000 = 0.998.
The periodic trace ends at 29986; this does not contradict the checkpoint boundary.
See `fine_tune_completion.json` and `depth_checkpoint_rebind_amendment.json`.

One broker-leased final evaluation ran on GPU 1, with CUDA launch blocking.
Completed balanced accuracies: ID 1.0; 2x 0.634765625; 4x 0.494140625.
The 9x bin failed with an int32 addition overflow in a Triton sum reduction,
reported from the nonrecording scan launch. The exact reduction caller and
triggering sample/state remain unknown. No retry or kernel change was made.
See `final_eval_diagnosis.json`, `final_eval_launch/console.txt`, and ignition_v1's
`depth_partial.json` and `depth_eval_failure.json`. No completed `depth.json`
was fabricated. The requested closure commit message would be inaccurate.

Eighteen targeted tests passed, including exact/current and hash-bound
amendment-covered registration, rejection of uncovered amendments, durable
resume, evaluation environment restoration, smoke fixture, and search resume.
The first expanded pytest command named a nonexistent test file and collected
nothing; its failure is preserved in `test_collection_failure.txt`.

The final board has transformer/gdn2/anchors active, and velocity-loop plus
part13-resume failed. GPUs 0/5/6 are at 100%; GPU 1 is idle. The eval-priority
preemption allowed baseline queues to acquire GPUs 5/6. Subsequent pure-search
restarts launch three candidates into one available card; waiting candidates
hit the existing 60-second lease timeout and the parent terminates the active
population. Restart attempts stopped after this scheduling cause was confirmed.
Banked candidates remain unchanged. This preemption did not achieve the intended
healthy final service board. Cards 2/3/4/7 and certified arms were not operated on.
Existing unrelated workspace changes were left untouched. No checkpoint or .pt
file is committed.

The smoke fixture fix is committed with a source-only regime amendment.
The existing production registration and smoke compatibility were verified.
No resilient DiLoCo behavior was changed or tested; requirement IDs from the
resilient gap matrix do not apply. The supervisor contact tool was unavailable.

Residual risks: missing 9x metric, unresolved reduction overflow, failed eval
service, failed pure search from broker queue saturation, and an incomplete v1 report card. Recovery is not closed.
