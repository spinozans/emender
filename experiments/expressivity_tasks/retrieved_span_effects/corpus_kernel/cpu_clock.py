"""One-thread CPU clocks, not a surrogate for the device meter."""

from copy import deepcopy
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch

from .attention import Local, dense, indexed, prepare
from .device_meter import invented


def clock(f):
    f()
    result = []
    for _ in range(5):
        a = time.perf_counter()
        for _ in range(3):
            f()
        result.append(1000 * (time.perf_counter() - a) / 3)
    return statistics.median(result)


def run():
    if (
        not __debug__
        or os.getenv("CUDA_VISIBLE_DEVICES") != ""
        or os.getenv("GPU_LEASE_HELD")
        or any(
            os.getenv(k) != "1"
            for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        )
    ):
        raise RuntimeError("one-thread CPU only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = {}
    for nd, per, bs in ((128, 128, 64), (512, 256, 128)):
        m, a, req, y, build = invented(nd, per, bs, "cpu")
        start = time.perf_counter()
        batch = a.batch(req)
        bind_s = time.perf_counter() - start
        p = batch.plan
        with torch.no_grad():
            start = time.perf_counter()
            rows = a.rows(m)
            row_s = time.perf_counter() - start
            start = time.perf_counter()
            bank = prepare(*rows, m.omega)
            tree_s = time.perf_counter() - start
            lk, lv, x = batch.live.encode(m, need_rows=True, need_last=True)
            L = Local(lk, lv, x.new_zeros(len(lk)), batch.live.lengths)
            q = m.query(x)
            null = (m.skip(x)[:, 0] + math.log(len(rows[1])), torch.zeros_like(x))
            dc = clock(partial(dense, q, *rows[1:], local=L, null=null))
            ic = clock(
                partial(indexed, q, bank, p, local=L, null=null, work=batch.work)
            )
            del rows, bank
        res = dict(
            rows=nd * per,
            queries=bs,
            exact_pairs=sum(b - a for _, a, b in p.exact),
            possible_pairs=bs * per * nd,
            index_build_s=build,
            request_packet_and_bind_s=bind_s,
            encode_rows_s=row_s,
            build_learned_summaries_s=tree_s,
            frozen_read_ms={"gemm_exhaustive": dc, "indexed": ic},
        )
        if nd == 128:
            times = {}
            for mode in ("dense", "indexed"):
                model = deepcopy(m)
                opt = torch.optim.AdamW(model.parameters(), lr=0.0005)

                def step(pkt=batch):
                    opt.zero_grad(set_to_none=True)
                    pred = pkt.predict(model, mode=mode).probabilities
                    (-pred[range(len(y)), y].log().mean()).backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, foreach=True
                    )
                    opt.step()

                times[mode] = clock(step)
                del step
            res["full_step_ms"] = times
        out[f"{nd * per}x{bs}"] = res
        del m, a, batch
        gc.collect()
    return dict(
        scope="invented native IDs, one CPU thread, FP32, warm median of five blocks of three; no GPU",
        cells=out,
    )


if __name__ == "__main__":
    ans = run()
    with Path(sys.argv[1]).open("x") as f:
        json.dump(ans, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
