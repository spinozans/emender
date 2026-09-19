#!/usr/bin/env bash
# Build one repair9-family preparation (v3-identical cohort set) with optional
# dose knobs for the restored pools (grounded-authored / representation-bridge).
# Usage: build_prep.sh <output-dir> <authored-max-records> <rehearsal-max-records> [log]
# Zero caps reproduce v3 exactly (validated byte-identical on 2026-09-20).
set -euo pipefail
OUT=$1; AMAX=$2; RMAX=$3; LOG=${4:-/dev/null}
REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining
S=$P/pi-native-repair9-wide-cohort-specs
mkdir -p "$(dirname "$OUT")"
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 1; }
ARGS=(
  --translated-oh-source "$P/e97-oh-pi-native-translation-v1/candidate-authority"
  --translated-oh-sha a179c140b4e8efa77a8163574bb78d4fe34dea91adfeb0439e8339d5002be1fb
  --translated-oh-cohort openhands-translated-rehearsal
  --translated-oh-budget-targets 26000000
  --translated-oh-seed 920001
  --selected "$P/pi-native-curriculum-2000-selected-v1/candidate-authority"
  --selected-sha 8cf83db8d608f6af74bbd2cbe60206ef435fb3003ab2bb63f4eadab17eb05562
  --selection-audit "$P/pi-native-curriculum-2000-selected-v1/selection-audit.json"
  --selection-audit-sha de977726f0758206ee9defe9c3687d2485e3ba2b80eb9ef9ceafb065c5f6646b
  --overlap-audit "$P/pi-native-curriculum-2000-selected-v1/overlap-audit.json"
  --overlap-audit-sha 9c104d73b104f4d29a83ca1472b66ef2df681e020a2b9688eb4f87c6ef814a78
  --rehearsal "$P/representation-bridge-v1-data/authority"
  --rehearsal-sha a21dba6f80e58e87ea838a98018b40650decebed3cf4b0f98dacdb750520469d
  --parent-checkpoint "$P/representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt"
  --parent-sha 9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa
  --authored-source "$P/grounded-expansion-v1-data-r2/authority"
  --authored-sha 08752c53e2199533252c77821ec9696012b19dc09ea0ff1508ef6676a6cd2140
  --authored-budget-targets 1000000
  --authored-seed 882843
  --pi-native-include-families copy,binding,path-copy,process-lifecycle,repo-edit-error-recovery,repo-symbol-edit-discovery,source-check,shell-error-recovery,repo-fuzzy-path,repo-explicit-read,repo-content-search
  --loopbreak-source "$P/pi-native-extracterror-collection-v1/collect/candidate-authority"
  --loopbreak-sha 08817eb42ff835523cc62bc13208053daeb79479a4bb9a585f1f427b53659aa4
  --conversation-source /mnt/nvme1n1/erikg/sft/e97-4b-smoltalk2-admitted-v1
  --conversation-sha d64f51abc615c097910900ffa7fc88f020ae5567b89958f607008078c271c7f6
  --conversation-budget-targets 60000000
  --conversation-seed 613117
  --extra-source "$P/pi-native-longcopy-collection-v1/collect/candidate-authority"
  --extra-sha 27a7f97d64a1589f98e3ce851344aac51e4c7608b2fa86df6c92955a52e79536
  --extra-cohort longcopy-rehearsal
  --extra2-source "$P/pi-native-pointerchase-collection-v1/collect/candidate-authority"
  --extra2-sha f2361a86c3909083c93be6702f12ad8323844a2390069d368ebec92d52e0247e
  --extra2-cohort pointerchase-rehearsal
  --extra3-source "$P/pi-native-reasoning-rehearsal-collection-v1/collect/candidate-authority"
  --extra3-sha 88e535b5bec8f704163eb45bab929d7c80e4eee92f575b3748ebb758d18b5229
  --extra3-cohort reasoning-rehearsal
  --cohort-spec "$S/instruction-rehearsal.json"
  --cohort-spec "$S/compositional-rehearsal.json"
  --cohort-spec "$S/cumulative-recovery-rehearsal.json"
  --cohort-spec "$S/live-aligned-rehearsal.json"
)
if [[ "$AMAX" != 0 ]]; then ARGS+=(--authored-max-records "$AMAX"); fi
if [[ "$RMAX" != 0 ]]; then ARGS+=(--rehearsal-max-records "$RMAX" --rehearsal-subsample-seed 882844); fi
"$PY" "$REPO/scripts/prepare_e97_pi_native_repair_training.py" "${ARGS[@]}" --output "$OUT" 2>&1 | tee "$LOG"
