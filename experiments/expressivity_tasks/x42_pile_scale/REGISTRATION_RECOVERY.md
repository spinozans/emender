# Registration drift recovery

Use `/home/erikg/emender/.venv/bin/python scripts/registration_guard.py check REGISTRATION`
from the repository root. JSON reports MATCH, REGISTRATION STALE vs HEAD (with
named intervening commits), WORKING TREE DIRTY vs HEAD, or TAMPER SUSPECT per
source. External absolute source paths use their own Git repository history.
A local edit of a correctly bound HEAD is dirty; an unknown registration hash,
or unexplained local bytes on top of an already stale binding, is suspect.
Missing history must be investigated rather than automatically authorized.

The report includes an executable rebind recipe. Rebind refuses dirty or
unproven source identities, writes an exclusive amendment receipt, and changes
only source hashes. Review that receipt and commit the registration before
restarting. For the velocity suite, update dependent regime/registration hashes
and generation registration identity together, preserving CMA and completed
candidates byte-for-byte in substance. Preserve a before-state receipt.
The source-only helper intentionally does not migrate training state.

Install the advisory hook with `git config core.hooksPath .githooks`.
It checks indexed bytes, so an unstaged registration update cannot silence a
warning. It never silently authorizes an amendment or disables runtime checks.

An attended source promotion can retain a historical smoke only through a
committed, explicitly scoped resume amendment binding the smoke, new regime,
registration, and supporting committed evidence. It does not certify new-source
throughput or erase prior failures. Normal starts retain the original smoke gate.
Pure candidates wait for the shared broker without the old 60-second timeout;
otherwise baseline candidates occupying two cards could kill the pure parent.

Anchor restart preserves the hashed completed transformer row and its length
sweep. The historical c06 reference is not a completed fresh pure row; the fresh
pure length anchor must still run. No resilient DiLoCo behavior is changed.
