"""S4 real retained GDN2 binary-byte model, independent of training/data evaluation runners."""

from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns as clock
from types import SimpleNamespace

import torch
from ndm.memory_program import Session
from .cached_host_bench import HostDriver
from .large_host_check import stock, make, check
from .serving_bench import prepare, measure, process_usage, quantiles
from .workbench_run import dump
from .workbench_witness import require

DIR = Path("/mnt/nvme2n1/erikg/xsa_anchor_replays_v1/gdn2")
STATE_SHA = "8a214dd37172472d60fecffc6367fa29915445fd16233dc17be6360609bbbc1c"
EXTERNAL = Path("/home/erikg/GatedDeltaNet-2")
EXT_REF = "95709fc250357c2dd109361c353192f2aa5913f9"
FILES = {
    "lit_gpt/gdn2.py": "5d93765adcb4e9bf755e7d4160a01d4e2ee8438ec55759d17903223dd18b0324",
    "lit_gpt/gdn2_ops/fused_recurrent_gdn2.py": "aee06a5fe471a194680a5762cbb733118d0e52ad816732a4e1742cb7485a2db1",
}
PREFIX = LENGTH = 32


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identity():
    anchor = json.loads((DIR / "anchor.json").read_text())
    path = DIR / "final.pt"
    require(
        anchor["base"] == "gdn2"
        and anchor["seed"] == 3401
        and anchor["sha256"] == digest(path) == STATE_SHA
        and path == Path(anchor["state"]),
        "unbound host checkpoint",
    )
    require(
        Path(os.getenv("GDN2_PATH", str(EXTERNAL))).resolve() == EXTERNAL
        and subprocess.check_output(["git", "-C", str(EXTERNAL), "rev-parse", "HEAD"], text=True).strip()
        == EXT_REF
        and all(digest(EXTERNAL / n) == h for n, h in FILES.items()),
        "changed native layer/kernel",
    )
    subprocess.run(["git", "-C", str(EXTERNAL), "diff", "--quiet", "HEAD", "--", "lit_gpt"], check=True)
    return dict(
        checkpoint=STATE_SHA,
        external_ref=EXT_REF,
        files=FILES,
        fla=importlib.metadata.version("flash-linear-attention"),
    )


def adapt(h, p):
    return h[:, :256].contiguous(), p


class Full:
    def __init__(self, net, ss, lives):
        self.net, self.lives, self.ss = net, lives, ss
        self.tokens = 0

    def inputs(self, ids):
        ws = [self.lives[i].data for i in ids]
        require(all(self.ss[i].visible == len(w) for i, w in zip(ids, ws, strict=True)), "full host frontier")
        self.tokens += sum(map(len, ws))
        return adapt(*stock(self.net, ws))

    def clear(self):
        pass

    def report(self):
        return dict(encoded_tokens=self.tokens, tensor_bytes=0)


class Cached(HostDriver):
    def __init__(self, net, ss, lives):
        super().__init__(make(net, len(ss)), ss, lives, adapt)

    def report(self):
        return dict(super().report(), traffic=self.arena.backend.stats)


@torch.no_grad()
def bare(run, net, host_cls, block, ix, ws, prior, label):
    start_setup = clock()
    b = len(ix)
    ids = list(range(b))
    cell = f"{block}-b{b}-{label}"
    total = 0

    def progress(phase):
        dump(
            run / "progress.json",
            dict(
                cell=cell,
                phase=phase,
                total_emitted=prior + total,
                cell_emitted=total,
                server_pid=os.getpid(),
            ),
        )

    progress("setup")
    lives = [SimpleNamespace(data=bytearray()) for _ in ix]
    ss = [Session(advance=l.data.extend) for l in lives]
    for s, w in zip(ss, ws, strict=True):
        s.observe(w[:PREFIX])
    host = host_cls(net, ss, lives)
    ts = [[] for _ in ix]
    ns = 0
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cpu, _, _ = process_usage([])
        start = clock()
        for t in range(LENGTH):
            a = clock()
            _, p = host.inputs(ids)
            ns += clock() - a
            valid = p.isfinite().all(1) & (p.logsumexp(1).abs() < 5e-5)
            out = torch.stack([p.argmax(1), valid.long()], 1).tolist()
            for i, (c, v) in enumerate(out):
                require(v and 0 <= c < 256, "invalid bare byte")
                ss[i].observe(bytes([c]))
                ts[i].append(clock())
                total += 1
            if total % 256 == 0:
                progress("generating")
        torch.cuda.synchronize()
        stop = clock()
        cpu1, _, rss = process_usage([])
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        report = host.report()
        require(all(s.visible == PREFIX + LENGTH for s in ss), "missing bare response")
        lat = [[x - y for x, y in zip(row, [start] + row[:-1], strict=True)] for row in ts]
        first = [row[0] - start for row in ts]
        result = dict(
            cell=cell,
            arm=label,
            b=b,
            block=block,
            train_rows=ix,
            emitted=total,
            wall_ns=stop - start,
            rate=total * 1e9 / (stop - start),
            setup_prompt_ns=start - start_setup,
            host=report,
            interbyte_ns=lat,
            first_ns=first,
            interbyte=quantiles(sum(lat, [])),
            first=quantiles(first),
            host_cpu_seconds=cpu1 - cpu,
            native_cpu_seconds=0,
            native_processes=0,
            host_rss_kib=rss[0],
            sum_child_rss_kib=0,
            peak_allocated=peak,
            peak_reserved=reserved,
            backbone_submit_ns=ns,
            backbone_batches=LENGTH,
            generated_sha=[hashlib.sha256(l.data[PREFIX:]).hexdigest() for l in lives],
            event_sha=[None] * b,
            work={},
            opcodes={},
            statuses={},
            position_worker_ns=0,
        )
    finally:
        host.clear()
    result["lifetime_ns"] = clock() - start_setup
    dump(run / (cell + ".json"), result)
    progress("cell-done")
    print(f"{cell}: {total} bytes {result['rate']:.1f}/s", flush=True)
    return total


def prepare_large(run):
    # prepare verifies the entire P2 controller, panel and mount identity, no optimizer/eval.
    # The historical loader checks the allocator snapshot, not an environment spelling.
    # Apply the exact registered native-allocator option before any model allocation.
    from .workbench_run import check_gpu_owner

    check_gpu_owner()
    require(
        all(
            os.getenv(k) == "expandable_segments:True"
            for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        ),
        "wrong allocator launch",
    )
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    settings = torch.cuda.memory._snapshot().get("allocator_settings", {})
    dump(run / "allocator.json", settings)
    require(settings.get("expandable_segments") is True, "allocator did not enable expandable segments")
    c14, eager, fused, selected, windows, manifest = prepare(run)
    del c14
    torch.cuda.empty_cache()
    meta = identity()
    t0 = clock()
    from ..x45_matrix_emender.matched_arms import build_gdn2

    outer, _, _ = build_gdn2(3401)
    state = torch.load(DIR / "final.pt", map_location="cpu", weights_only=True, mmap=True)
    outer.load_state_dict(state, strict=True)
    del state
    net = outer.eval().requires_grad_(False).model
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    params = sum(p.numel() for p in net.parameters())
    require((params, len(net.blocks), net.width) == (149855670, 27, 640), "wrong scale")
    torch.cuda.synchronize()
    meta.update(
        parameters=params,
        width=net.width,
        blocks=len(net.blocks),
        model_load_s=(clock() - t0) / 1e9,
        torch=str(torch.__version__),
        precision="FP32",
        max_context=64,
    )
    idn = json.loads((run / "identity.json").read_text())
    idn["controller_training_base_sha"] = idn.pop("base_sha")
    idn.update(generation_base=meta, generated_length=LENGTH)
    dump(run / "identity.json", idn)
    return net, eager, fused, selected, windows, manifest


def main(run):
    net, eager, fused, selected, windows, manifest = prepare_large(run)
    t0 = clock()
    checks = check(net)
    torch.cuda.synchronize()
    checks["qualification_including_compile_s"] = (clock() - t0) / 1e9
    dump(run / "device-parity.json", checks)
    print("real retained 150M host parity passed", flush=True)
    total = 0
    for block in range(2):
        for b in (1, 16):
            ix = list(range(16 * block, 16 * block + b))
            ws = [windows[i] for i in ix]
            arms = [
                (m, e) for m in ("bare", "direct", "fused-greedy", "fused-sample") for e in ("full", "cache")
            ]
            for m, e in arms[::-1] if block else arms:
                label = f"{e}-{m}"
                host_cls = Cached if e == "cache" else Full
                if m == "bare":
                    total += bare(run, net, host_cls, block, ix, ws, total, label)
                else:
                    total += measure(
                        run,
                        net,
                        eager if m == "direct" else fused,
                        m,
                        block,
                        [selected[i] for i in ix],
                        ws,
                        ix,
                        manifest,
                        total,
                        label=label,
                        host_factory=host_cls,
                        length=LENGTH,
                        prefix=PREFIX,
                    )
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    main(p.parse_args().run)
