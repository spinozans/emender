#!/usr/bin/env bash
set -euo pipefail
REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
T=/mnt/nvme2n1/erikg/e97_systematic_posttraining
S=/mnt/nvme1n1/erikg/sft
"$PY" "$REPO/scripts/prepare_e97_pi_native_repair_training.py" \
 --translated-oh-source "$T/e97-extension-prep-v1/oh-thinking-enriched-authority" \
 --translated-oh-sha f226f3b3e1a45352dd7a9f302c741890e479827182c2a27b6919ddf4344b8fd0 \
 --translated-oh-budget-targets 58000000 --translated-oh-seed 920001 \
 --rehearsal "$T/representation-bridge-v1-data/authority" \
 --rehearsal-sha a21dba6f80e58e87ea838a98018b40650decebed3cf4b0f98dacdb750520469d \
 --authored-source "$T/grounded-expansion-v1-data-r2/authority" \
 --authored-sha 08752c53e2199533252c77821ec9696012b19dc09ea0ff1508ef6676a6cd2140 \
 --authored-seed 882843 --authored-budget-targets 1000000 \
 --selected "$T/pi-native-curriculum-2000-selected-v1/candidate-authority" \
 --selected-sha 8cf83db8d608f6af74bbd2cbe60206ef435fb3003ab2bb63f4eadab17eb05562 \
 --selection-audit "$T/pi-native-curriculum-2000-selected-v1/selection-audit.json" \
 --selection-audit-sha de977726f0758206ee9defe9c3687d2485e3ba2b80eb9ef9ceafb065c5f6646b \
 --overlap-audit "$T/pi-native-curriculum-2000-selected-v1/overlap-audit.json" \
 --overlap-audit-sha 9c104d73b104f4d29a83ca1472b66ef2df681e020a2b9688eb4f87c6ef814a78 \
 --parent-checkpoint "$T/representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt" \
 --parent-sha 9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa \
 --pi-native-include-families copy,binding,path-copy,process-lifecycle,repo-edit-error-recovery,repo-symbol-edit-discovery,source-check,shell-error-recovery,repo-fuzzy-path,repo-explicit-read,repo-content-search \
 --loopbreak-source "$T/pi-native-extracterror-collection-v1/collect/candidate-authority" \
 --loopbreak-sha 08817eb42ff835523cc62bc13208053daeb79479a4bb9a585f1f427b53659aa4 \
 --conversation-source "$S/e97-4b-smoltalk2-admitted-v1" \
 --conversation-sha d64f51abc615c097910900ffa7fc88f020ae5567b89958f607008078c271c7f6 \
 --conversation-budget-targets 700000000 --conversation-seed 613117 \
 --extra-source "$T/pi-native-longcopy-collection-v1/collect/candidate-authority" \
 --extra-sha 27a7f97d64a1589f98e3ce851344aac51e4c7608b2fa86df6c92955a52e79536 \
 --extra-cohort longcopy-rehearsal \
 --extra2-source "$T/pi-native-pointerchase-collection-v1/collect/candidate-authority" \
 --extra2-sha f2361a86c3909083c93be6702f12ad8323844a2390069d368ebec92d52e0247e \
 --extra2-cohort pointerchase-rehearsal \
 --extra3-source "$T/pi-native-reasoning-rehearsal-collection-v1/collect/candidate-authority" \
 --extra3-sha 88e535b5bec8f704163eb45bab929d7c80e4eee92f575b3748ebb758d18b5229 \
 --extra3-cohort reasoning-rehearsal \
 --extra4-source "$T/pi-native-hybrid-conversation-collection-v1/collect/candidate-authority" \
 --extra4-sha 7e5fec2a67b1e7ba45bcbcc866d86cb133bb0c454cdabdb274978eb42ebc0b1d \
 --extra4-cohort hybrid-conversation-rehearsal \
 --cohort-spec "$T/e97-extension-prep-v1/cohort-specs/instruction-rehearsal.json" \
 --cohort-spec "$T/e97-extension-prep-v1/cohort-specs/compositional-rehearsal.json" \
 --cohort-spec "$T/e97-extension-prep-v1/cohort-specs/cumulative-recovery-rehearsal.json" \
 --cohort-spec "$T/e97-extension-prep-v1/cohort-specs/live-aligned-rehearsal.json" \
 --purpose-sizing-note \
 --output "$T/e97-extension-preparation-v2"
