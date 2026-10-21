# X15 GDN2-only attended amendment1 — pre-T64

Original registration/source commit5d51341a remains immutable, as does the
running scratch Muon@3x on card7. Original GDN2 failed-attempt worker/trainer,
command/console/terminal and FAIL marker are retained and hash-bound here.

The supervisor explicitly acknowledges an error in the task: it conflated
X11's aggregate-BEST482500 (four-perfect, **P5=333/512**) with X11's fixed-FINAL
500000 (four-perfect, **P5=318/512**). The intended champion is aggregate-BEST
482500, hash `49a0f96fe7890ca9bb8a860698641ad8407dca412a3b38a01d3ede996c011cd0`.
No checkpoint/hash selection changes in this amendment.

At the original attempt, the hash-pinned best reproduced **all16 historical
T16 counts exactly**: `[512,512,512,512,333,3,0,0,0,0,0,0,0,0,0,0]`. The additional
incorrect hardcoded P5=318 assertion then stopped fail-closed, **before any T64
forward/evaluation or T64 score was observed**. This is a protocol preflight
failure, not a model failure, and does not answer prediction(a)/(b)/(c).

Supervisor approval via live coordination: retain BEST482500 per champion
intent; correct333 expectation in a separate corrected script + own registration;
retain failed evidence; authorize exactly **one attended GDN2-only relaunch**
on physical card6. Original bound runner and running Muon closure cannot change.
No automatic retry. Registered predictions(a)/(b)/(c) are unchanged; P5 comparison
uses the correct champion baseline333, not318. All64 T64 position columns, P1–P16
T16/T64 comparison, strict stock-X11 loading, exact T16 reproduction and unchanged
state/hash checks remain. Zero training updates. No stage12 runner dependency.

Separate script: `scripts/invertible_stage15_gdn2_amendment.py`; separate output
under this amendment directory. Preregistration amendment and launchable source
must be committed/pushed before relaunch. Both registrations verified afterward.
One-hour infrastructure wall guard; failures censored, not architecture failures.

Combined watcher (supersedes original combined watcher which correctly reports
the preserved failed preflight):
`.venv/bin/python -m scripts.watch_invertible_stage15_amended --interval 900`
from `/home/erikg/emender`. Exit0 both hash-bound completions, exit1 failure,
`--once` exit2 pending; growth report and marginal signature on terminal wake.
Supervisor will arm native wake. No WorksGood service/dispatcher used.
