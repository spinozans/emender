# Repo mirror of the repair-v9r staging machinery

This directory is the version-controlled mirror of the LIVE staging root
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9r-full-arc-staging-v1/`
(that path is authoritative; it carries the run-time state such as
`proposal-audits/` receipts). It is committed so the staged freeze/audit/admit/
launch machinery and its v3-restart fixes are reviewable in git. The
`segment{N}/commands.sh` sheets and `freeze_segment_proposal.py` target the
v3 preparation (scrub-reversal remediation); the `*-v2arc-superseded` files
retain the v10-era machinery for forensics. Nothing here has been executed;
admissions and launches require operator sign-off.
