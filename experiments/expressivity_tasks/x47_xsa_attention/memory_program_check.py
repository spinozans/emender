"""CPU wiring check for ndm.memory_program. No optimizer, held-out data or W3 plan features."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch

from ndm.memory_program import MaterializedBranches, MemoryPolicy, Session, branch_reader, teacher_force
from .w3_data import HERE, OUT, Pages, STOCK
from .w3_model import ANCHOR_SHA, backbone, base_model
from .workbench_native import Live
from .workbench_witness import require
from .workbench_run import sha


def check():
    if not __debug__:
        raise RuntimeError("verification requires enabled assertions")
    frozen = HERE / "receipts/w3"
    manifest = json.loads((frozen / "train-pages.json").read_text())
    panel_sha = sha(frozen / "panels.json")
    assert manifest["stage"] == "train" and manifest["panel_sha"] == panel_sha
    selected = json.loads((frozen / "panels.json").read_text())["panels"]["train"][:2]
    shard = manifest["shards"][0]
    path = OUT / "train" / shard["file"]
    assert sha(path) == shard["sha256"]
    # Entire registered documents were hash-checked at preparation. Only 'raw', never ptr/codes/ci/on.
    with np.load(path, allow_pickle=False) as data:
        raw = data["raw"]
        assert raw.shape == (32, 2049) and raw.dtype == np.uint8
        windows = [bytes(row[:17]) for row in raw[:2]]
    net = base_model()
    torch.manual_seed(7401)
    policy = MemoryPolicy(256).eval()

    def encode(x):
        # Use the original causal 128-byte computation. Padding cannot become a memory input or a label.
        t = x.shape[1]
        h, p = backbone(net, torch.nn.functional.pad(x, (0, 128 - t)))
        return h[:, :t], p[:, :t].log()

    with torch.no_grad(), ExitStack() as stack:
        pages = Pages()
        stack.callback(pages.close)
        require(
            (pages.exe_sha, pages.desc.version) == (manifest["exe_sha"], manifest["mount"]),
            "mount/build identity",
        )
        mount = MaterializedBranches("P100", pages.desc.version, pages, scope="whole")
        archive = mount.reader()
        docs = []
        for row in selected:
            live = stack.enter_context(Live())
            context = branch_reader(
                "live",
                live.session_id,
                lambda keys, live=live: [live.branch(k) if 0 < len(k) <= 16 else None for k in keys],
                session_id=live.session_id,
                watermark=lambda live=live: live.visible,
            )
            docs.append(
                Session(
                    [archive, context],
                    advance=live.append,
                    session_id=live.session_id,
                    document=row["doc"],
                    start=row["start"],
                )
            )
        result = teacher_force(policy, encode, docs, windows, mode="greedy")
        error = mass_error = 0.0
        keys = 0
        histogram = {}
        for ds, w in zip(result.decisions, windows, strict=True):
            # Independent oracle: suffix-ID frame from raw visible byte, then its exact m20 row;
            # this is NOT MaterializedBranches' root-to-child traversal through cached pages.
            x = torch.tensor(list(w[:-1]))[None, :]
            _, logp = encode(x)
            base = logp[0].softmax(-1)
            for t, d in enumerate(ds):
                program = "/".join(e.inst.op for e in d.trace)
                histogram[program] = histogram.get(program, 0) + 1
                assert program == "SLICE/LOOKUP/EMIT" and d.trace[1].reader[:3] == (
                    "P100",
                    pages.desc.version,
                    "byte_edges",
                )
                key = w[t : t + 1]
                assert d.trace[1].keys == (key,) and d.visible == t + 1
                keys += 1
                node = int(pages.suffix(key)[-1, 0])
                want = base[t]
                if node:
                    h = torch.tensor(pages.rows([node])["h"][0, :256].astype(np.float64)).float()
                    if h.sum() > 0:
                        want = 0.5 * (want + h / h.sum())
                actual = d.log_probs.exp()
                error = max(error, float((actual - want).abs().max()))
                mass_error = max(mass_error, abs(float(actual.sum()) - 1.0))
        assert keys == result.targets == 32 and error < 2e-6 and mass_error < 2e-6
        assert all(s.visible == len(w) for s, w in zip(docs, windows, strict=True))
        return dict(
            device="cpu",
            purpose="integration parity only, no learning or quality estimate",
            torch=str(torch.__version__),
            seed=7401,
            base_sha=ANCHOR_SHA,
            panel_sha=panel_sha,
            train_shard_sha=shard["sha256"],
            sha_m20_exe=pages.exe_sha,
            sha_live_exe=sha(STOCK),
            mount=pages.desc.version,
            train_rows=[0, 1],
            targets=result.targets,
            programs=histogram,
            max_oracle_error=error,
            max_mass_error=mass_error,
            submitted_keys=keys,
            pages_fetched=mount.pages_fetched,
            provider_page_batches=mount.page_batches,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    record = json.dumps(check(), indent=2, sort_keys=True)
    print(record)
    if args.receipt:
        with args.receipt.open("x") as f:
            f.write(record + "\n")
