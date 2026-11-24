"""Audit ONLY the orphaned T12 separate-A result. This cannot pass the registered four-cell gate."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from .paired_frontier_verify import ROOT, ORDER, check_cell, sig
from .serving_verify import load, require


def partial(root=ROOT):
    source = "454aea719013a269cce8faef8ba2e02e0a414a02"
    d = load(root / "driver.json")
    stop = load(root / "stop.json")
    require(
        (root / "protocol-ref").read_text().strip() == "3c9e9ca56f3664f7814142873d3fabd46a535e8f"
        and (root / "source-ref").read_text().strip()
        == (root / "gated-ref").read_text().strip()
        == d["ref"]
        == source
        and tuple(d["order"]) == ORDER
        and stop["process"] == "proc_2abb"
        and stop["wrapper_rc"] == 1
        and stop["error"] == "KeyError: 'unsynced_at_close'"
        and (root / "run.log").read_text().splitlines()[-1] == stop["error"]
        and stop["completed_cells"] == ["separate-A"]
        and stop["unstarted_cells"] == list(ORDER[1:])
        and stop["no_timing_retry"]
        and not (root / "complete.json").exists()
        and not any((root / n).exists() for n in ORDER[1:])
        and "196 passed in 16.58s" in (root / "qualification.txt").read_text().splitlines()[-1],
        "wrong T12 partial",
    )
    signature, metrics, actors, epochs = check_cell(root, "separate-A", source)
    fs = [
        "qualification.txt",
        "driver.json",
        "run.log",
        *[f"separate-A/{s}.json" for s in ("identity", "batch-0", "batch-1", "progress", "cell", "exit")],
    ]
    return dict(
        ref=source,
        phase="separate-A complete, driver audit stopped before pair-A",
        artifacts={n: sha256((root / n).read_bytes()).hexdigest() for n in fs},
        signature=sig(signature),
        cell_epochs=epochs,
        distinct_native_and_cell_pids=len(actors),
        metrics=metrics,
        selector_scored=4096,
        value_gradient_targets=4096,
        value_updates=2,
        pair_targets=0,
        card_s=0,
        cumulative_x47_upper=1429,
        remaining_s=371,
        separate_cell_checked=True,
        four_cell_equivalence=None,
        paired_recommendation=None,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(partial(ap.parse_args().root), indent=2, allow_nan=False))
