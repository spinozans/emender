# Environment adaptations / corrections (before training)

## Correction: task-spec CE vs shipped IMM loss

The supervisor's task text claimed 'CE per the repo scripts'. This was wrong:
BOTH imm_unbound/train_rnn.py and train_deltanet.py at dbfa365a use
F.binary_cross_entropy_with_logits, final plus .3-weighted intermediate MAT
auxiliary BCE. Code governs the raw-pipeline control. Supervisor explicitly
approved retaining shipped BCE and attributed the mistaken CE instruction to
memory of the earlier DGC-script discussion bleeding into this IMM spec
(concession #21; blocking decision received before implementation/updates).
No categorical CE substitution or two-logit conversion was made.

## Lambda01 adaptation receipt

No Slurm available/required: run their bash scripts directly; #SBATCH gpu22
headers remain byte-identical comments, not a queue-binding claim. Two isolated
Python3.12.3 venvs `/tmp/x26c-{rnn,delta}-venv` are made with project Python.
A single x26c_cuda_reuse.pth points to the existing project site-packages so
CUDA torch2.9.1+cu128 and its binary dependencies can be reused without a
multi-gigabyte reinstall; local venv packages take precedence. This also exposes
other pre-existing packages, explicitly NOT a hermetic recreation of their
unspecified historical environment. No base-venv packages were upgraded.

Their launchers skip torch installation when importable; this adaptation uses
that shipped branch (CU_INDEX defaultcu126 is unused; actual CUDA12.8 retained).
Shipped pip upgrade commands were run once for pre-update smokes, and are run
AGAIN by their unchanged bash launchers at production launch. GRU upgrades
pip/setuptools/wheel/numpy/tqdm; Delta additionally upgrades unpinned
flash-linear-attention. No requirements lockfile exists in upstream root;
launcher commands, not guessed pins, govern. Installed numpy2.5.4 and Delta
FLA0.6.0 differ from X26b numpy2.4.6/FLA0.5.1 and are candidate environment
deltas. GRU does not use FLA despite inherited0.5.1 being visible.
`receipts/environment_setup_{rnn,delta}.txt` captures actual pip behavior;
`environment_pre_*.json` captures versions and full freeze. Each launch retains
its own upstream logging and `environment_runtime.json` after dependency setup
and before external interpretation, checking critical versions against smokes.
Pip freeze's editable ndm commit reflects receipt time, not an upstream dependency.

Only environment/path overrides: VENV_DIR, TMP_BASE, WORKDIR (GRU output CWD),
TRAIN_PY, DATA_DIR, SAVE_PATH. env -i removes inherited tuning knobs; HOME/USER/
PATH restored. OMP_NUM_THREADS=MKL_NUM_THREADS=2 limits CPU oversubscription on
shared lambda01 and is an explicit runtime adaptation. CUDA_VISIBLE_DEVICES
binds physical cards4/5. Each arm has its own output CWD to isolate upstream
eval_log defaults. Data reused byte-for-byte from X26b as supervisor approved;
generator T100/count overrides are carried configuration, not CLI-default data.
All14 vendored source files hash-match upstream git objects at dbfa365a.
No upstream file, collate, train loop, scheduler, loss, stopping, logging or
hyperparameter was patched. Frozen-checkpoint evaluator is a separate process.
