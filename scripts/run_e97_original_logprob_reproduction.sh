#!/usr/bin/env bash
# Reproduce the original complete assay, not a continuation or threshold change.
set -euo pipefail
umask 077
RUN=${1:?new output root required}
SOURCE_CONTROL=/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-rl-logprob-qualification-v1-control
read -r COMMIT < "$SOURCE_CONTROL/commit.txt"
[[ $COMMIT == f1c39c39777acbf1d51cfbf00773503e9af9436d ]]
test ! -e "$RUN"
unset PYTHONHASHSEED
cd "$SOURCE_CONTROL/worktree"
sha256sum --check --quiet "$SOURCE_CONTROL/source.sha256"
printf 'ORIGINAL_ASSAY_REPRODUCTION source=%s PYTHONHASHSEED=unset optimizer_updates=0\n' "$COMMIT"
timeout --kill-after=30 2600 bash scripts/run_e97_native_rl_logprob_qualification.sh "$RUN"
sha256sum --check --quiet "$SOURCE_CONTROL/source.sha256"
cmp "$RUN/recipe-private.json" /mnt/nvme2n1/erikg/e97_systematic_posttraining/native-rl-logprob-qualification-v1/recipe-private.json
printf 'ORIGINAL_ASSAY_RECIPE_IDENTICAL; original failed evidence unchanged\n'
