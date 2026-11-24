"""Admit only the first unexecuted candidate. A terminated T12 driver is never resumed or impersonated."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns as clock, time, time_ns
from .direct_control_bench import setup
from .first_paired_transition_verify import PREV, PROTO, DELTA, IMMUTABLE, audit, seal
from .serving_verify import require
from .workbench_run import dump


def binding(ref):
    out = subprocess.check_output(
        ["git", "diff", "--name-only", PREV, ref, "--", "ndm", "scripts", "*.py"], text=True
    ).splitlines()
    base = "experiments/expressivity_tasks/x47_xsa_attention/"
    require(out == [base + f for f in DELTA], "T12 frozen compute or historical CPU cases changed")
    objects = {
        name: [
            subprocess.check_output(["git", "rev-parse", f"{r}:{name}"], text=True).strip()
            for r in (PREV, ref)
        ]
        for name in IMMUTABLE
    }
    require(all(a == b for a, b in objects.values()), "different producer tree/file")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "ndm", "scripts", "*.py"], check=True)
    return dict(old=PREV, new=ref, changed_code=DELTA, git_objects=objects)


def main(root):
    start = clock()
    ref = setup(root, gate_count=196)
    require(
        time() + 130 < 1795531800
        and not (root / "driver.json").exists()
        and (root / "protocol-ref").read_text().strip() == PROTO
        and "1 passed in" in (root / "receipt-qualification.txt").read_text().splitlines()[-1],
        "T13 refused before source verification",
    )
    seal()
    dump(root / "source-check.json", binding(ref))
    dump(
        root / "driver.json",
        dict(ref=ref, pid=os.getpid(), order=["pair-A"], start_epoch_ns=time_ns(), card_s=0),
    )
    run = root / "pair-A"
    run.mkdir()
    begin = clock()
    with (run / "console").open("xb") as f:
        try:
            p = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.expressivity_tasks.x47_xsa_attention.paired_frontier_bench",
                    str(root),
                    "--cell",
                    "pair-A",
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=max(1, 115 - (begin - start) / 1e9),
            )
        except subprocess.TimeoutExpired:
            dump(run / "exit.json", dict(timeout=True, wall_ns=clock() - begin))
            raise
    dump(run / "exit.json", dict(rc=p.returncode, wall_ns=clock() - begin))
    require(p.returncode == 0, "first paired worker did not complete")
    dump(
        root / "complete.json",
        dict(ref=ref, pid=os.getpid(), end_epoch_ns=time_ns(), wall_ns=clock() - start, card_s=0),
    )
    result = audit(root)
    dump(
        root / "decision.json",
        dict(first_pair_parity=result["first_pair_parity"], unmatched=result["unmatched"]),
    )
    require(result["first_pair_parity"], "first paired execution did not match the sealed oracle")
    print(
        "one new paired worker, two complete value transitions checked; no comparative timing decision",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    main(ap.parse_args().root)
