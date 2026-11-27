"""SAL-1 driver: equal-dose fits, witnessed all-fit freeze, one probe opening.

Every fit sees all FIT shifted bytes exactly once in complete updates. The
per-seed batch schedule and the window order are fixed by registered RNGs.
All six fits (seeds 5751/5752 x LOCAL/SKELETON/ORDER2) must be frozen and
independently acknowledged before the locked PROBE role is extracted, and the
probe is scored exactly once from the frozen weights. `main` runs only inside
a leased device stage; CPU qualification drives the functions directly.
"""

import hashlib
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import zstandard as zstd

from ..meter import await_ack
from ..window import ROOT, descendant, fields, get, put, need

from .data import load
from .ladder import CountIndex
from .model import SalLM, SalPack
from .settings import ARMS, DATA, JOB, N, SEEDS, STAGE2_ARMS, sha

# The probe is opened in a quarantined fresh subprocess (fail-closed seals,
# no training state); a rebound campaign overrides this entry to its own CLI.
PROBE_ENTRY = __package__ + ".data"

# The shared-dictionary codec baseline's raw bytes: None means "the in-memory
# archive"; a rebound campaign with an external index supplies its own
# (possibly capped/fallback) provider.
DICT_BYTES = None

# For externally indexed archives: after the quarantined probe extraction,
# an optional second quarantined subprocess binds the probe's count tables
# from the index transport before scoring. None keeps the in-RAM path.
BIND_ENTRY = None

UPDATES = 2048
BATCH = 16
TARGETS_PER_WINDOW = 64
WARMUP = 32
LR0, LR1 = 2e-3, 2e-4
CONTRASTS = (("skeleton", "local"), ("order2", "skeleton"), ("order2", "local"))


def order(seed):
    return np.random.default_rng(seed).permutation(UPDATES).astype("<i4")


def order_sha(seed):
    return hashlib.sha256(order(seed).tobytes()).hexdigest()


def lr_at(step):
    if step <= WARMUP:
        return LR0 * step / WARMUP
    return LR1 + (LR0 - LR1) * 0.5 * (
        1 + math.cos(math.pi * (step - WARMUP) / (UPDATES - WARMUP))
    )


def state_sha(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def place(batch, dev):
    if torch.device(dev).type == "cpu":
        return batch
    return {
        k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def make_optimizer(m, dev):
    cuda = torch.device(dev).type == "cuda"
    return torch.optim.AdamW(
        m.parameters(), lr=LR0, betas=(0.9, 0.98), weight_decay=0.01, fused=cuda
    )


def update(m, o, batch, arm, archive):
    o.zero_grad(set_to_none=True)
    loss = m(batch, arm, archive=archive).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
    o.step()
    return loss.detach()


def fits(batches, index, tick, ref, dev="cpu", arms=ARMS):
    """Train every (seed, arm) fit; freeze them; wait for the independent witness."""
    from ..corpus_windows.packet import WindowArchive

    need(len(batches) >= UPDATES, "registered batch count below the schedule")
    info = {}
    for s in SEEDS:
        for arm in arms:
            k = f"{s}.{arm}"
            m = SalLM(s).to(dev)
            m.train()
            init = state_sha(m)
            o = make_optimizer(m, dev)
            archive = WindowArchive(m, index) if arm == "order2" else None
            if torch.device(dev).type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            total = 0.0
            seen = 0
            for step, num in enumerate(order(s), 1):
                for g in o.param_groups:
                    g["lr"] = lr_at(step)
                b = place(batches[int(num)], dev)
                y = update(m, o, b, arm, archive)
                total += float(y) * len(b["targets"])
                seen += len(b["targets"])
                if step % 256 == 0:
                    tick(f"fit {k} {step}/{UPDATES}")
            need(step == UPDATES and math.isfinite(total), "invalid dose or fit")
            secs = time.perf_counter() - t0
            p = JOB / (k + ".pt")
            tmp = p.with_suffix(".partial")
            with tmp.open("xb") as f:
                torch.save({q: v.detach().cpu() for q, v in m.state_dict().items()}, f)
            tmp.rename(p)
            meta = dict(
                seed=s,
                arm=arm,
                updates=step,
                targets=seen,
                order_sha=order_sha(s),
                initial=init,
                weights_sha=sha(p),
                parameters=sum(q.numel() for q in m.parameters()),
                fit_nats_per_byte=total / seen,
                fit_s=secs,
                updates_per_s=step / secs,
            )
            if torch.device(dev).type == "cuda":
                meta["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                meta["incremental_peak_bytes"] = (
                    torch.cuda.max_memory_allocated() - base
                )
            put(JOB / (k + ".json"), meta)
            info[k] = sha(JOB / (k + ".json"))
            print("closed", k, round(secs, 1), "seconds", flush=True)
            del m, o, archive
    put(
        JOB / "frozen.json",
        dict(
            qualified=ref,
            rows_sha=sha(DATA / "rows.json"),
            build_sha=sha(DATA / "build.npy")
            if (DATA / "build.npy").exists()
            else get(DATA / "container.json")["container_sha256"],
            fit_sha=sha(DATA / "fit.npy"),
            fits=info,
        ),
    )
    await_ack(JOB, "freeze-ack.json")


@torch.no_grad()
def score(batches, index, tick, dev="cpu", arms=ARMS):
    """One scoring pass per frozen fit; per-document nats from window sums."""
    from ..corpus_windows.packet import WindowArchive

    seal = get(JOB / "frozen.json")
    need(
        (JOB / "freeze-ack.json").exists()
        and get(JOB / "freeze-ack.json")["frozen_sha"] == sha(JOB / "frozen.json"),
        "missing freeze witness",
    )
    windows = sum(b["targets"].numel() // TARGETS_PER_WINDOW for b in batches)
    wpd = windows // N["probe"]
    need(wpd * N["probe"] == windows, "probe batches not document-aligned")
    losses = {}
    clocks = {}
    for s in SEEDS:
        for arm in arms:
            k = f"{s}.{arm}"
            p = JOB / (k + ".pt")
            j = JOB / (k + ".json")
            need(seal["fits"][k] == sha(j), "unfrozen scoring weights")
            need(get(j)["weights_sha"] == sha(p), "changed fit weights")
            m = SalLM(s).to(dev)
            m.eval()
            m.load_state_dict(
                torch.load(p, map_location=dev, weights_only=True), strict=True
            )
            archive = WindowArchive(m, index) if arm == "order2" else None
            t0 = time.perf_counter()
            out = []
            for i, b in enumerate(batches, 1):
                out.append(
                    m(place(b, dev), arm, archive=archive)
                    .reshape(-1, TARGETS_PER_WINDOW)
                    .double()
                    .sum(1)
                )
                if i % 128 == 0:
                    tick(f"score {k} {i}")
            losses[k] = torch.cat(out).reshape(N["probe"], wpd).sum(1).cpu().numpy()
            clocks[k] = time.perf_counter() - t0
            del m, archive
    with (JOB / "document-losses.npz").open("xb") as f:
        np.savez(f, **losses)
    return losses, clocks


def frames(x, build, dict_bytes=None):
    """Actual zstd level-3 frame bytes per probe excerpt, decoded and checked.

    The shared-dictionary baseline uses `dict_bytes` when given, else the
    in-memory archive's bytes; with neither (an externally indexed archive)
    only the plain baseline is produced and the caller reports the fallback.
    """
    raw_dict = dict_bytes() if callable(dict_bytes) else dict_bytes
    if raw_dict is None and build is not None:
        raw_dict = build.tobytes()
    out = {}
    for label, d in [
        ("plain", None),
        (
            "shared_raw_archive",
            None
            if raw_dict is None
            else zstd.ZstdCompressionDict(
                raw_dict, dict_type=zstd.DICT_TYPE_RAWCONTENT
            ),
        ),
    ]:
        if label != "plain" and d is None:
            continue
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


def summarize(losses, arms=ARMS):
    """Ideal bits per original byte, paired document bootstrap, seed directions."""
    docs = N["probe"]
    need(
        set(losses) == {f"{s}.{a}" for s in SEEDS for a in arms},
        "incomplete fits",
    )
    draws = np.random.default_rng(5750).multinomial(
        docs, np.full(docs, 1 / docs), size=10000
    )
    rates, means, per_seed = {}, {}, {}
    for a in arms:
        n = np.stack([losses[f"{s}.{a}"] for s in SEEDS])
        need(
            n.shape == (len(SEEDS), docs) and np.isfinite(n).all() and (n >= 0).all(),
            "bad document losses",
        )
        b = (n / math.log(2) + 8.0) / 4097.0
        rates[a] = b
        means[a] = b.mean().item()
        per_seed[a] = b.mean(1).tolist()
    comp = {}
    pairs = [(a, c) for a, c in CONTRASTS if a in arms and c in arms]
    for a in arms:
        if a != arms[0] and (a, arms[0]) not in pairs:
            pairs.append((a, arms[0]))
    for a, c in pairs:
        d = rates[c] - rates[a]
        v = d.mean(0)
        lo, hi = np.quantile(draws @ v / docs, [0.025, 0.975])
        seeds = d.mean(1)
        comp[f"{a}:{c}"] = dict(
            gain=v.mean().item(),
            ci=[lo.item(), hi.item()],
            seeds=seeds.tolist(),
            positive=bool(lo > 0 and (seeds > 0).all()),
        )
    primary = f"{arms[1]}:{arms[0]}" if len(arms) == 2 else "skeleton:local"
    return dict(
        raw_bytes=docs * 4097,
        shifted_targets=docs * 4096,
        first_byte_bits=8 * docs,
        mean_bpb=means,
        per_seed_bpb=per_seed,
        contrasts=comp,
        primary_positive=comp[primary]["positive"],
    )


def bind_all(pack, role, shuffle):
    x = load(role)
    t0 = time.perf_counter()
    receipt = DATA / f"bound-{role}.json"
    if receipt.exists():
        # Externally indexed archive (SAL-3): the per-batch m/h tensors were
        # bound off-card from the index transport; load them only under the
        # sealed receipts. Window order mirrors dataset()'s exactly.
        r = get(receipt)
        z = np.load(DATA / f"bound-{role}.npz", allow_pickle=False)
        need(sha(DATA / f"bound-{role}.npz") == r["batches_sha"], "unbound batches")
        n = r["batches"]
        fields = ("ids", "targets", "local_succ", "m", "h")
        keys = [f"b{i}_{f}" for i in range(n) for f in fields]
        need(
            set(z.files) == set(keys)
            and r["rows_sha"] == sha(DATA / "rows.json")
            and r["targets"] == sum(len(z[f"b{i}_targets"]) for i in range(n)),
            "bound batch set mismatch",
        )
        batches = []
        for i in range(n):
            ids = torch.as_tensor(z[f"b{i}_ids"], dtype=torch.long)
            targets = torch.as_tensor(z[f"b{i}_targets"], dtype=torch.long)
            batches.append(
                dict(
                    ids=ids,
                    targets=targets,
                    spans=pack.spans,
                    lengths=pack.lengths,
                    rows=pack.rows,
                    local_rows=pack.local_rows,
                    local_succ=torch.as_tensor(z[f"b{i}_local_succ"], dtype=torch.long),
                    pair_owner=pack.pair_owner,
                    pair_row=pack.pair_row,
                    m=torch.as_tensor(z[f"b{i}_m"], dtype=torch.float32),
                    h=torch.as_tensor(z[f"b{i}_h"]),
                    core=torch.zeros((len(targets), 0), dtype=torch.long),
                    core_valid=torch.zeros(len(targets), dtype=torch.bool),
                    cover_owner=torch.zeros(0, dtype=torch.long),
                    cover_lv=torch.zeros(0, dtype=torch.long),
                    cover_idx=torch.zeros(0, dtype=torch.long),
                )
            )
    else:
        batches = pack.dataset(x, shuffle=shuffle)
    return (
        x,
        batches,
        dict(
            preparation_s=time.perf_counter() - t0,
            batches=len(batches),
            real_targets=sum(len(b["targets"]) for b in batches),
            bound_from_index=receipt.exists(),
        ),
    )


def main():
    gpu = os.environ.get("GPU_LEASE_HELD")
    owner = get(JOB / "owner.json")
    a = get(JOB / "admission.json")
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    ref = a["qualified"]
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

    tick("verified adopted CUDA device")
    build = load("build") if (DATA / "build.json").exists() else None
    cache = DATA / "counts-cache"
    counts = None
    if (cache / "meta.json").exists():
        # CPU-side prep builds the big index off the leased card; the stage
        # only loads it, and only when its binding matches the sealed archive.
        counts = CountIndex.load(
            cache, meta=dict(archive_sha=get(DATA / "build.json")["array_sha"])
        )
    if build is None:
        # Externally indexed archive (SAL-3): the row count comes from the
        # sealed container receipt, and every batch is bound off-card.
        c = get(DATA / "container.json")
        need(
            "rows" in c and c.get("bound") is True,
            "unbound external archive receipt",
        )
        pack = SalPack(
            None,
            SalLM(SEEDS[0]),
            batch=BATCH,
            bank=False,
            rows=int(c["rows"]),
        )
    else:
        pack = SalPack(
            build,
            SalLM(SEEDS[0]),
            batch=BATCH,
            bank=any(a.startswith("order2") for a in a.get("arms", STAGE2_ARMS)),
            counts=counts,
        )
    frozen_ok = (
        (JOB / "frozen.json").exists()
        and (JOB / "freeze-ack.json").exists()
        and get(JOB / "freeze-ack.json")["frozen_sha"] == sha(JOB / "frozen.json")
    )
    fit_setup = None
    if frozen_ok:
        # The witnessed all-fit freeze is the atomic resume point; the fits
        # are already closed and sealed. Do not retrain them.
        tick("resuming from the witnessed freeze")
    else:
        _, fit_batches, fit_setup = bind_all(pack, "fit", True)
        need(fit_setup["batches"] == UPDATES, "fit batch count mismatch")
        tick("fit batches bound")
        fits(fit_batches, pack.index, tick, ref, dev=dev, arms=arms)
        del fit_batches
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("GPU_LEASE_HELD", None)
    subprocess.run(
        [sys.executable, "-m", PROBE_ENTRY, "probe"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    if BIND_ENTRY is not None:
        subprocess.run(
            [sys.executable, "-m", *BIND_ENTRY],
            cwd=ROOT,
            env=env,
            check=True,
        )
    x, probe_batches, probe_setup = bind_all(pack, "probe", False)
    tick("locked probe opened after witnessed freeze")
    losses, clocks = score(probe_batches, pack.index, tick, dev=dev, arms=arms)
    quality = summarize(losses, arms)
    zstd_price = frames(x, build, dict_bytes=DICT_BYTES)
    tick("all fits and scores closed")
    put(
        JOB / "complete.json",
        dict(
            qualified=ref,
            units=units,
            device=str(dev),
            hardware=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            fit_setup=fit_setup,
            probe_setup=probe_setup,
            stage=stage,
            archive=dict(documents=N["build"], rows=pack.rows, leaf=128),
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
    await_ack(JOB, "watch-ack-run")


if __name__ == "__main__":
    main()
