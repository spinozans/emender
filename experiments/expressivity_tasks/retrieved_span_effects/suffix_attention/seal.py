"""SAL-1 seal stage: complete the terminal receipt from sealed artifacts.

The probe was opened and scored exactly once (stage run3); its per-document
losses are sealed in document-losses.npz. This stage performs NO model
forward passes and NO second evaluation: it re-derives the report from the
sealed losses, re-binds the already-opened roles for the setup accounting,
recomputes the descriptive zstd frames from the extracted arrays, and writes
the terminal complete.json plus this stage's completion marker.
"""

import os

import numpy as np
import torch

from ..window import ROOT, descendant, fields, get, need, put
from .data import load
from .model import SalLM, SalPack
from .run import BATCH, bind_all, frames, summarize
from .settings import DATA, JOB, N, SEEDS, STAGE2_ARMS, sha


def lease_identity():
    gpu = os.environ.get("GPU_LEASE_HELD")
    owner = get(JOB / "owner.json")
    need(
        gpu in ("0", "1")
        and gpu == os.getenv("CUDA_VISIBLE_DEVICES")
        and torch.cuda.device_count() == 1
        and descendant(os.getpid(), owner["pid"]),
        "real leased device required",
    )
    lock = fields(ROOT / ".wg/gpu_leases" / gpu)
    need(
        (lock["pid"], lock["starttime"], lock["adopted"])
        == (str(owner["pid"]), owner["start"], "1"),
        "foreign lease",
    )
    torch.set_num_threads(1)
    return owner


def main():
    lease_identity()
    a = get(JOB / "admission.json")
    stage = a["stage"]
    arms = tuple(a.get("arms", STAGE2_ARMS))
    units = 0

    def tick(label):
        nonlocal units
        units += 1
        put(
            JOB / f"progress-{stage}.json",
            dict(units=units, pid=os.getpid(), label=label),
        )

    tick("seal stage start")
    need(
        (JOB / "frozen.json").exists()
        and get(JOB / "freeze-ack.json")["frozen_sha"] == sha(JOB / "frozen.json"),
        "missing witnessed freeze",
    )
    frozen = get(JOB / "frozen.json")
    npz = JOB / "document-losses.npz"
    need(npz.exists(), "missing sealed document losses")
    losses = {}
    with np.load(npz, allow_pickle=False) as f:
        need(
            set(f.files) == set(frozen["fits"]),
            "sealed losses do not match the frozen fits",
        )
        for k in f.files:
            losses[k] = f[k]
    quality = summarize(losses, arms)
    tick("quality re-derived from sealed losses")
    build = load("build")
    pack = SalPack(build, SalLM(SEEDS[0]), batch=BATCH)
    _, fit_batches, fit_setup = bind_all(pack, "fit", True)
    x, probe_batches, probe_setup = bind_all(pack, "probe", False)
    need(
        fit_setup["batches"] == 2048
        and fit_setup["real_targets"] == N["fit"] * 4096
        and probe_setup["real_targets"] == N["probe"] * 4096,
        "bound role accounting mismatch",
    )
    zstd_price = frames(x, build)
    tick("descriptive frames recomputed from the extracted arrays")
    fits = {k: get(JOB / (k + ".json")) for k in frozen["fits"]}
    complete = dict(
        qualified=a["qualified"],
        stage=stage,
        resumed_from="run3: fits frozen and witnessed in stage run; probe opened and scored once in stage run3; metadata sealed here",
        units=units,
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        fit_setup=fit_setup,
        probe_setup=probe_setup,
        scoring_clock_note="per-fit scoring wall times were computed in stage run3 but lost in its terminal metadata crash; they are not reported",
        archive=dict(documents=N["build"], rows=pack.rows, leaf=128),
        frozen_sha=sha(JOB / "frozen.json"),
        rows_sha=sha(DATA / "rows.json"),
        data_sha={k: sha(DATA / (k + ".npy")) for k in N},
        scores_sha=sha(npz),
        fits=fits,
        zstd=zstd_price,
        quality=quality,
    )
    put(JOB / "complete.json", complete)
    put(
        JOB / f"meter-{stage}.json",
        dict(complete_sha=sha(JOB / "complete.json"), units=units, stage=stage),
    )
    tick("terminal receipt sealed")
    print("SEALED", sha(JOB / "complete.json"), flush=True)


if __name__ == "__main__":
    main()
