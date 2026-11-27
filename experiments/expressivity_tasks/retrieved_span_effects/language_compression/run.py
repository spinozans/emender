"""Real CUDA identity/parity/clock gate, equal-dose fits, sealed raw-text scoring."""

import hashlib
import math
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import triton
import zstandard as zstd

from ..window import ROOT, fields, descendant
from ..meter import await_ack
from .settings import DATA, JOB, HERE, REG, N, SEEDS, ARMS, get, put, need, sha
from .data import load
from .lm import LanguageLM, Pack
from .gate import order, order_sha
from .report import summarize


def sync():
    torch.cuda.synchronize()


def state_sha(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def opt(m):
    return torch.optim.AdamW(
        m.parameters(), lr=0.002, betas=(0.9, 0.98), weight_decay=0.01, fused=True
    )


def update(m, o, b, arm, backend):
    o.zero_grad(set_to_none=True)
    loss = m(b, arm, backend=backend).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
    o.step()
    return loss.detach()


def meter(tick):
    rng = np.random.default_rng(5764)
    raw = rng.integers(0, 64, (128, 129), dtype=np.uint8)
    x = rng.integers(0, 64, (16, 65), dtype=np.uint8)
    for i in range(0, 16, 2):
        x[i, 16:49] = raw[i, 7:40]
    models = [LanguageLM(5764).cuda() for _ in range(2)]
    p = Pack(raw, models[0])
    b = p.bind(x)
    backends = ("torch", "triton")
    opts = [opt(m) for m in models]
    losses = []
    logits = []
    for m, o, backend in zip(models, opts, backends, strict=True):
        with torch.no_grad():
            logits.append(m.logits(b, "indexed", backend)[0])
        losses.append(update(m, o, b, "indexed", backend))
    torch.testing.assert_close(*logits, rtol=3e-4, atol=4e-5)
    torch.testing.assert_close(*losses, rtol=5e-5, atol=4e-5)
    grad = 0.0
    for a, c in zip(models[0].parameters(), models[1].parameters(), strict=True):
        if a.grad is None:
            need(c.grad is None, "mismatched active parameters")
            continue
        torch.testing.assert_close(a.grad, c.grad, rtol=2e-3, atol=4e-5)
        grad = max(grad, (a.grad - c.grad).abs().max().item())
    trials = {k: [] for k in backends}
    mem = {k: [] for k in backends}
    cells = list(zip(models, opts, backends, strict=True))
    for m, o, k in cells:
        for _ in range(4):
            update(m, o, b, "indexed", k)
    for i in range(5):
        for m, o, k in cells if i % 2 == 0 else cells[::-1]:
            o.zero_grad(set_to_none=True)
            sync()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            start = time.perf_counter()
            for _ in range(3):
                update(m, o, b, "indexed", k)
            sync()
            trials[k].append(1000 * (time.perf_counter() - start) / 3)
            mem[k].append(torch.cuda.max_memory_allocated() - base)
        tick("invented optimizer clock block " + str(i + 1))
    ms = {k: statistics.median(v) for k, v in trials.items()}
    out = dict(
        shape=[len(p.a.index), len(b.targets), 64, 32],
        exact_pairs=b.exact_pairs,
        ms=ms,
        blocks=trials,
        extra_peak_bytes={k: max(v) for k, v in mem.items()},
        max_logit_difference=(logits[0] - logits[1]).abs().max().item(),
        max_clipped_gradient_difference=grad,
        speedup=ms["torch"] / ms["triton"],
    )
    put(JOB / "meter.json", out)
    need(
        ms["triton"] < ms["torch"], "indexed complete-step acceleration not established"
    )
    return out


def data_clock(pack, role):
    start = time.perf_counter()
    x = load(role)
    b = pack.dataset(x, shuffle=role == "fit")
    sync()
    return (
        x,
        b,
        dict(
            preparation_s=time.perf_counter() - start,
            batches=len(b),
            real_targets=sum(len(w.targets) for w in b),
            exact_pairs=sum(w.exact_pairs for w in b),
            hit_queries=sum(w.hit_queries for w in b),
            pairs=sum(len(w.targets) for w in b) * len(pack.a.index),
        ),
    )


def fits(batches, tick, ref):
    info = {}
    for s in SEEDS:
        schedule = order(s)
        for arm in ARMS:
            k = f"{s}.{arm}"
            m = LanguageLM(s).cuda().train()
            init = state_sha(m)
            o = opt(m)
            sync()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            total = torch.zeros((), device="cuda", dtype=torch.float64)
            for step, num in enumerate(schedule, 1):
                lr = (
                    0.002 * step / 32
                    if step <= 32
                    else 0.0002
                    + 0.0009 * (1 + math.cos(math.pi * (step - 32) / (1024 - 32)))
                )
                o.param_groups[0]["lr"] = lr
                b = batches[int(num)]
                y = update(m, o, b, arm, "triton")
                total += y.double() * len(b.targets)
                if step % 64 == 0:
                    sync()
                    tick(f"fit {k} {step}/1024")
            sync()
            secs = time.perf_counter() - t
            peak = torch.cuda.max_memory_allocated()
            need(step == 1024 and total.isfinite().item(), "invalid dose or fit")
            p = JOB / (k + ".pt")
            tmp = p.with_suffix(".partial")
            with tmp.open("xb") as f:
                torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, f)
            tmp.rename(p)
            meta = dict(
                seed=s,
                arm=arm,
                updates=step,
                targets=1048576,
                order_sha=order_sha(s),
                initial=init,
                weights_sha=sha(p),
                parameters=sum(p.numel() for p in m.parameters()),
                fit_nats_per_byte=total.item() / 1048576,
                fit_s=secs,
                targets_per_s=1048576 / secs,
                peak_allocated_bytes=peak,
                incremental_peak_bytes=peak - base,
            )
            put(JOB / (k + ".json"), meta)
            info[k] = sha(JOB / (k + ".json"))
            print("closed", k, round(secs, 3), "seconds", flush=True)
            del o, m
    put(
        JOB / "frozen.json",
        dict(
            qualified=ref,
            rows_sha=sha(DATA / "rows.json"),
            build_sha=sha(DATA / "build.npy"),
            fit_sha=sha(DATA / "fit.npy"),
            fits=info,
        ),
    )
    await_ack(JOB, "freeze-ack.json")


def frames(x, build):
    out = {}
    for label, d in [
        ("plain", None),
        (
            "shared_raw_archive",
            zstd.ZstdCompressionDict(
                build.tobytes(), dict_type=zstd.DICT_TYPE_RAWCONTENT
            ),
        ),
    ]:
        enc = zstd.ZstdCompressor(level=3, dict_data=d)
        dec = zstd.ZstdDecompressor(dict_data=d)
        n = 0
        for doc in x:
            raw = doc.tobytes()
            frame = enc.compress(raw)
            need(dec.decompress(frame) == raw, "zstd roundtrip failed")
            n += len(frame)
        out[label] = dict(frame_bytes=n, bpb=8 * n / x.nbytes)
    return out


@torch.no_grad()
def score(batches, tick):
    seal = get(JOB / "frozen.json")
    need(
        get(JOB / "freeze-ack.json")["frozen_sha"] == sha(JOB / "frozen.json"),
        "missing freeze witness",
    )
    losses = {}
    clocks = {}
    for s in SEEDS:
        for arm in ARMS:
            k = f"{s}.{arm}"
            p = JOB / (k + ".pt")
            j = JOB / (k + ".json")
            need(
                seal["fits"][k] == sha(j) and get(j)["weights_sha"] == sha(p),
                "unfrozen scoring weights",
            )
            m = LanguageLM(s).cuda().eval()
            m.load_state_dict(
                torch.load(p, map_location="cpu", weights_only=True), strict=True
            )
            out = []
            sync()
            start = time.perf_counter()
            for i, w in enumerate(batches, 1):
                out.append(m(w, arm, backend="triton").reshape(16, 64).double().sum(1))
                if i % 128 == 0:
                    sync()
                    tick(f"score {k} {i}")
            losses[k] = torch.cat(out).reshape(N["probe"], 64).sum(1).cpu().numpy()
            clocks[k] = time.perf_counter() - start
    with (JOB / "document-losses.npz").open("xb") as f:
        np.savez(f, **losses)
    return summarize(losses), clocks


def main():
    gpu = os.environ.get("GPU_LEASE_HELD")
    owner = get(JOB / "owner.json")
    a = get(JOB / "admission.json")
    need(
        gpu in ("0", "1")
        and gpu == os.getenv("CUDA_VISIBLE_DEVICES")
        and os.getenv("TRITON_INTERPRET") is None
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    units = 0

    def tick(s):
        nonlocal units
        units += 1
        put(JOB / "progress.json", dict(units=units, pid=os.getpid(), label=s))

    tick("verified adopted CUDA device")
    log = JOB / "device-tests.log"
    with log.open("xb") as f:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(HERE.parent / "corpus_windows/test_kernels.py"),
                str(HERE / "test_device.py"),
            ],
            cwd=ROOT,
            env=dict(os.environ, CORPUS_WINDOWS_TEST_DEVICE="cuda"),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    need(
        re.search(r"\b19 passed\b", log.read_text()), "required device fixtures missing"
    )
    tick("19 real device tests passed")
    price = meter(tick)
    torch.cuda.empty_cache()
    print("hardware parity passed; complete indexed step", price["ms"], flush=True)
    build = load("build")
    pack = Pack(build, LanguageLM(SEEDS[0]).cuda())
    _, batches, fit_setup = data_clock(pack, "fit")
    tick("fit byte routes bound")
    fits(batches, tick, a["qualified"])
    del batches
    torch.cuda.empty_cache()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("GPU_LEASE_HELD")
    subprocess.run(
        [sys.executable, "-m", __package__ + ".data", "probe"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    x, probe, probe_setup = data_clock(pack, "probe")
    tick("locked probe opened after independent freeze witness")
    zstd_price = frames(x, build)
    quality, clocks = score(probe, tick)
    tick("all fits and scores closed")
    put(
        JOB / "complete.json",
        dict(
            registered=REG,
            qualified=a["qualified"],
            units=units,
            hardware=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            triton=triton.__version__,
            real_device_tests=19,
            tests_sha256=sha(log),
            meter=price,
            fit_setup=fit_setup,
            probe_setup=probe_setup,
            scope="fresh document-disjoint Pile TRAIN probe excerpts, small fixed byte archive",
            archive=dict(
                documents=128,
                raw_bytes=build.nbytes,
                rows=len(pack.a.index),
                count_table_bytes=pack.ref.probs.nbytes,
            ),
            frozen_sha=sha(JOB / "frozen.json"),
            rows_sha=sha(DATA / "rows.json"),
            data_sha={k: sha(DATA / (k + ".npy")) for k in N},
            scores_sha=sha(JOB / "document-losses.npz"),
            fits={
                k: get(JOB / (k + ".json")) for k in get(JOB / "frozen.json")["fits"]
            },
            score_s=clocks,
            zstd=zstd_price,
            quality=quality,
        ),
    )
    await_ack(JOB)


if __name__ == "__main__":
    main()
