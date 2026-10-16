# Attempt 1 — stopped, no learned result claimed

E97 and E97-identity both failed on optimizer update 1: fused recurrence accepts
unaligned lengths in forward-only mode, not training. Both failures and console
tracebacks are retained. GDN2 was terminated by SIGTERM to restart all three
calibrations with the approved identical padding rule; its prior measurements
are not a completed calibration result. No headline training was launched.

Live supervisor approved constant-zero **right padding only** to multiple16 on
all arms and all splits (3->16,130->144), slicing logits to original real length
before loss/scoring. Causality: real logits depend on real prefixes only;
padding comes strictly after real tokens, masks exclude it, and no targets are
inputs. Tests permanently verify real logits/loss equality and zero padded-logit
gradients. All attempts remain available; restart uses unchanged initialization,
data seeds, optimizer, geometry, and stopping/selection rules.

`checkpoint_relocation.json` maps historical creation paths in these first-attempt
ledgers to the preserved binary paths after the restart-directory move, with
freshly verified SHA256 values. First-attempt checkpoints are not selected as
results of the restarted runs.
