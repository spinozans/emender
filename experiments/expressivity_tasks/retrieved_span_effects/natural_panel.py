"""One sequential, budgeted co-training panel. Stop closed; never reselect, resume or retry."""

import gc
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .cuda_step import Step, eval_losses
from .meter import await_ack
from .natural_data import DATA, SIZES, cached, public_gate, sha
from .natural_model import ARMS, HOSTS, SEEDS, Shape, TokenLM
from .natural_score import BANDS, cases, cell
from .token_postings import Image, fetch, locate
from .window import BASE, MOD, PY, ROOT, get, need, put


BLOCK = 64


def jobs():
    pairs = [(h, a) for a in ARMS for h in HOSTS]
    for i, s in enumerate(SEEDS):
        order = pairs[5 * i :] + pairs[: 5 * i]
        for h, a in order if i % 2 == 0 else order[::-1]:
            yield h, a, s


def preflight(job):
    public_gate()
    if job == "natural-meter":
        need(not DATA.exists(), "meter must precede new membership")
        return
    need(
        job == "panel"
        and get(DATA / "build.json")["docs"] == SIZES["build"]
        and get(DATA / "fit.json")["docs"] == SIZES["fit"]
        and not (DATA / "probe-open-attempted").exists(),
        "probe spent or missing data",
    )
    need(
        get(BASE / "natural-meter/complete.json")["admitted"]
        and get(BASE / "natural-meter/charge.json")["rc"] == 0,
        "no accepted natural step meter",
    )


def make_plan(d):
    meter = get(BASE / "natural-meter/complete.json")
    batch = meter["batch"]
    w = get(BASE / "window.json")
    secs = min(get(d / "admission.json")["cap"], w["deadline_epoch"] - time.time())
    fit = get(DATA / "fit.json")
    mean = fit["windows"] / fit["docs"]
    maxnw = SIZES["probe"] * 64
    nw = min(maxnw, math.ceil(2 * mean * SIZES["probe"]))
    batches = math.ceil(nw / batch)
    tr = []
    ev = 0
    for host, arm, _ in jobs():
        clocks = meter["clocks"][host + "/" + arm]
        tr.append(max(clocks["train"]["block_ms"]) / 1000)
        ev += batches * len(cases(arm)) * max(clocks["eval"]["block_ms"]) / 1000
    # Timing not quality sets dose, shared across every arm and seed. Allow for
    # gathering, cold graph transitions, checkpoint I/O and scoring diagnostics.
    steps = BLOCK * math.floor((secs - 900 - 1.5 * ev) / (1.15 * sum(tr) * BLOCK))
    need(steps >= 2 * BLOCK, "remaining window cannot sustain a matched panel")
    return dict(
        batch=batch,
        steps=steps,
        matched_train_s=0.6 * min(tr) * steps,
        fit_windows=fit["windows"],
        probe_window_estimate=nw,
        eval_reserve_s=1.5 * ev,
        remaining_s=secs,
        cells=[cell(*j) for j in jobs()],
        rows_sha=get(DATA / "reserved.json")["reservation_sha"],
    )


class Stream:
    def __init__(self, role, image, sh, tick):
        ds = cached(role)
        self.image = image
        self.w = torch.from_numpy(ds["windows"]).cuda()
        self.n = torch.from_numpy(ds["count"]).cuda()
        self.doc = ds["doc"]
        self.ids = []
        self.votes = []
        begin = time.perf_counter()
        for start in range(0, len(self.w), 128):
            sl = slice(start, start + 128)
            ids, votes = locate(
                image,
                self.w[sl, : sh.prefix].contiguous(),
                self.n[sl] >= sh.prefix,
                sh.hits,
            )
            self.ids.append(ids)
            self.votes.append(votes)
            if start % (128 * 128) == 0:
                torch.cuda.synchronize()
                tick(1, "locate-" + role, windows=min(start + 128, len(self.w)))
        self.ids = torch.cat(self.ids)
        self.votes = torch.cat(self.votes)
        torch.cuda.synchronize()
        self.meta = dict(
            wall_s=time.perf_counter() - begin,
            windows=len(self.w),
            targets=int(self.n.sum().item()),
            hit_windows=int((self.ids[:, 0] >= 0).sum().item()),
            ids_sha=hashlib.sha256(self.ids.cpu().numpy().tobytes()).hexdigest(),
            votes_sha=hashlib.sha256(self.votes.cpu().numpy().tobytes()).hexdigest(),
        )

    def batch(self, idx):
        pages, lengths = fetch(self.image, self.ids[idx])
        return self.w[idx], self.n[idx], pages, lengths, self.votes[idx]


def ordered(n, b, steps, seed):
    rng = np.random.default_rng(seed)
    remain = steps * b
    chunks = []
    while remain > 0:
        perm = rng.permutation(n).astype(np.int32)[:remain]
        chunks.append(perm)
        remain -= len(perm)
    idx = np.concatenate(chunks)
    return torch.from_numpy(idx.reshape(steps, b)).cuda(), hashlib.sha256(
        idx.tobytes()
    ).hexdigest()


def atomic_pt(path, obj):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def lr(j, steps):
    warm = max(1, steps // 32)
    if j < warm:
        return 0.0005 * (j + 1) / warm
    return 0.0005 * (
        0.1 + 0.9 * (1 + math.cos(math.pi * (j - warm) / (steps - warm))) / 2
    )


def fit(d, sh, plan, stream, host, arm, seed, idx, order_sha, tick):
    name = cell(host, arm, seed)
    folder = d / "fits" / name
    folder.mkdir()
    steps = plan["steps"]
    batch = plan["batch"]
    start = time.perf_counter()
    fit = Step(sh, batch, host, arm, seed)
    cumulative = 0.0
    points = {}
    weights = {}
    state = None
    for start_step in range(0, steps, BLOCK):
        begin = time.perf_counter()
        for j in range(start_step, start_step + BLOCK):
            fit.train(stream.batch(idx[j]), lr(j, steps))
        torch.cuda.synchronize()
        cumulative += time.perf_counter() - begin
        step = j + 1
        values = fit.stat.cpu().tolist()
        need(
            np.isfinite(values).all() and values[1] > 0 and values[0] >= 0,
            "nonfinite optimizer step",
        )
        state = dict(
            cell=name,
            step=step,
            train_s=cumulative,
            nll_sum=values[0],
            targets=int(values[1]),
            summed_grad_norm=values[2],
        )
        with (folder / "learn.jsonl").open("a") as f:
            f.write(__import__("json").dumps(state) + "\n")
        tick(BLOCK, "train", **state)
        if (
            cumulative >= plan["matched_train_s"] or step == steps
        ) and "time" not in points:
            points["time"] = dict(
                step=step,
                targets=int(values[1]),
                train_s=cumulative,
                band_reached=cumulative >= plan["matched_train_s"],
            )
            atomic_pt(folder / "time.pt", fit.model.state_dict())
            weights["time"] = sha(folder / "time.pt")
        if step == BLOCK or step % 1024 == 0:
            atomic_pt(
                d / "last-atomic.pt",
                dict(
                    model=fit.model.state_dict(),
                    optimizer=fit.opt.state_dict(),
                    position=state,
                    order_sha=order_sha,
                    plan=plan,
                    rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                ),
            )
            put(d / "last-atomic.json", dict(**state, sha=sha(d / "last-atomic.pt")))
    points["full"] = dict(step=steps, targets=state["targets"], train_s=cumulative)
    atomic_pt(folder / "full.pt", fit.model.state_dict())
    weights["full"] = sha(folder / "full.pt")
    put(
        folder / "fit.json",
        dict(
            cell=name,
            points=points,
            weights=weights,
            order_sha=order_sha,
            wall_s=time.perf_counter() - start,
        ),
    )
    del fit
    gc.collect()
    torch.cuda.empty_cache()


def witness_freeze(d):
    # In the independent observer, from files only, before PROBE extraction.
    frozen = get(d / "frozen.json")
    plan = get(d / "plan.json")
    need(
        set(frozen["fits"]) == set(plan["cells"]) == {cell(*j) for j in jobs()}
        and not (DATA / "probe-open-attempted").exists(),
        "incomplete/late freeze",
    )
    signatures = {}
    for host, arm, seed in jobs():
        name = cell(host, arm, seed)
        folder = d / "fits" / name
        r = get(folder / "fit.json")
        need(
            frozen["fits"][name] == sha(folder / "fit.json")
            and r["cell"] == name
            and r["points"]["full"]["step"] == plan["steps"]
            and r["points"]["time"]["step"] <= plan["steps"],
            "unfrozen fit",
        )
        need(
            all(sha(folder / (k + ".pt")) == r["weights"][k] for k in ("time", "full")),
            "weight mismatch",
        )
        sig = (r["order_sha"], r["points"]["full"]["targets"])
        need(
            signatures.setdefault(seed, sig) == sig,
            "unequal exposures/order within seed",
        )
    put(
        d / "freeze-ack.json",
        dict(
            freeze_sha=sha(d / "frozen.json"),
            fits=len(plan["cells"]),
            observer_pid=os.getpid(),
            rows_sha=plan["rows_sha"],
        ),
    )


def eval_case(d, net, case, stream, batch, sh, den, tick, name):
    condition = case.split(".")[1]
    num = np.zeros((SIZES["probe"], len(BANDS)), np.float64)
    counts = np.zeros_like(num, dtype=np.int64)
    start = time.perf_counter()
    donor = 0
    for begin in range(0, len(stream.w), batch):
        sl = slice(begin, begin + batch)
        packet = stream.batch(sl)
        win, n, pages, lens, _ = packet
        y = eval_losses(net, packet, condition)
        pos = torch.arange(sh.time, device=y.device)[None, :]
        real = pos < n[:, None]
        eligible = real & (pos >= sh.prefix - 1)
        hit = eligible & (lens >= 0).any(1)[:, None]
        # Secondary diagnostic made AFTER the prediction; never a retriever or model input.
        target8 = win[:, sh.prefix : sh.prefix + 8]
        eq = (pages.unfold(-1, 8, 1) == target8[:, None, None, :]).all(-1)
        valid = (
            torch.arange(sh.page - 7, device=y.device)[None, None, :] + 8
            <= lens[:, :, None]
        )
        long = n >= sh.prefix + 6 + 1
        novel = hit & (long & ~(eq & valid).any((1, 2)))[:, None]
        bands = torch.stack((real, eligible, hit, novel), -1)
        a = (y[:, :, None].double() * bands).sum(1).cpu().numpy()
        b = bands.sum(1).cpu().numpy()
        donor += int(
            (
                eligible & ((lens == lens[:, :1]).all(1) & (lens[:, 0] > 0))[:, None]
            ).sum()
        )
        np.add.at(num, stream.doc[sl], a)
        np.add.at(counts, stream.doc[sl], b)
        if begin % (batch * 256) == 0:
            tick(
                1,
                "score",
                cell=name,
                case=case,
                windows=min(begin + batch, len(stream.w)),
            )
    need(
        int(counts[:, 0].sum()) == stream.meta["targets"]
        and (den is None or np.array_equal(den, counts)),
        "lost or unpaired targets",
    )
    return (
        num,
        counts,
        dict(seconds=time.perf_counter() - start, donor_eligible_targets=donor),
    )


def main(d):
    public_gate()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    sh = Shape()
    plan = make_plan(d)
    put(d / "plan.json", plan)
    units = 0

    def tick(k, label, **kw):
        nonlocal units
        units += k
        put(d / "progress.json", dict(units=units, label=label, pid=os.getpid(), **kw))

    image = Image(**cached("build")).tensors("cuda")
    stream = Stream("fit", image, sh, tick)
    put(d / "fit-routes.json", stream.meta)
    (d / "fits").mkdir()
    orders = {}
    for host, arm, seed in jobs():
        if seed not in orders:
            orders = {seed: ordered(len(stream.w), plan["batch"], plan["steps"], seed)}
        fit(d, sh, plan, stream, host, arm, seed, *orders[seed], tick)
    del stream, orders
    gc.collect()
    torch.cuda.empty_cache()
    put(
        d / "frozen.json",
        dict(
            plan_sha=sha(d / "plan.json"),
            fit_routes_sha=sha(d / "fit-routes.json"),
            fits={cell(*j): sha(d / "fits" / cell(*j) / "fit.json") for j in jobs()},
        ),
    )
    tick(1, "all fits sealed")
    stop = time.monotonic() + 300
    while not (d / "freeze-ack.json").is_file():
        need(time.monotonic() < stop, "freeze witness timeout")
        time.sleep(0.2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("GPU_LEASE_HELD", None)
    subprocess.run(
        [PY, "-m", MOD.rsplit(".", 1)[0] + ".natural_data", "probe"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    stream = Stream("probe", image, sh, tick)
    put(d / "probe-routes.json", stream.meta)
    (d / "scores").mkdir()
    den = None
    scores = {}
    for host, arm, seed in jobs():
        name = cell(host, arm, seed)
        folder = d / "fits" / name
        r = get(folder / "fit.json")
        net = TokenLM(sh, host, arm).eval().cuda()
        prev = None
        score = {}
        clocks = {}
        for case in cases(arm):
            p = case.split(".")[0]
            if p != prev:
                f = folder / (p + ".pt")
                need(sha(f) == r["weights"][p], "fit changed")
                net.load_state_dict(
                    torch.load(f, weights_only=True, map_location="cuda")
                )
                prev = p
            score[case], den, clocks[case] = eval_case(
                d, net, case, stream, plan["batch"], sh, den, tick, name
            )
        path = d / "scores" / (name + ".npz")
        np.savez(path, counts=den, **score)
        scores[name] = sha(path)
        put(d / "scores" / (name + ".json"), clocks)
        del net
        tick(1, "scored", cell=name)
    put(
        d / "complete.json",
        dict(
            units=units,
            scores=scores,
            freeze_sha=sha(d / "frozen.json"),
            targets=stream.meta["targets"],
            rows_sha=plan["rows_sha"],
        ),
    )
    await_ack(d)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
