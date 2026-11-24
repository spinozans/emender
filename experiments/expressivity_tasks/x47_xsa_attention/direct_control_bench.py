"""T9 direct retrieval: four separate TRAIN-only CPU lifetimes. No learned memory program or GPU."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
from time import perf_counter_ns as clock, time, time_ns

import numpy as np
import torch
from ndm.memory_program import MaterializedBranches, Session
from ndm.memory_program.direct import DocumentCounts, DirectTrust, direct_branches, direct_features
from ndm.memory_program.frozen_packet import FrozenPacket
from .frozen_packet_bench import CAP, OPT, no_card, tensor_sha
from .inband_route_check import delta
from .memory_program_profile import training_data
from .serving_bench import process_usage
from .serving_verify import HERE, load, require
from .stream_packet_bench import binding
from .w3_data import Pages, ROW
from .workbench_run import dump
from .workbench_witness import digest_state

ORDER = ("serial-A", "grouped-A", "grouped-B", "serial-B")
SEED = 7873
LADDER = (1, 7, 17, 63, 128)
T4 = HERE / "receipts/memory-program-t4/attempt1"
PAYLOAD = 8 << 20


def setup(root, *, gate_count=114):
    no_card()
    os.umask(0o077)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(SEED)
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(
        ref == (root / "gated-ref").read_text().strip()
        and re.fullmatch(
            rf"=+ {gate_count} passed in [\d.]+s =+",
            (root / "qualification.txt").read_text().splitlines()[-1],
        ),
        "unfrozen direct control",
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "ndm/memory_program",
            str(HERE),
            "tests/test_memory_program*.py",
        ],
        check=True,
    )
    return ref


def bind(pages, man, bank_type=MaterializedBranches):
    require(
        (pages.exe_sha, pages.desc.version) == (man["exe_sha"], man["mount"]), "changed whole-archive binding"
    )
    return bank_type("P100", pages.desc.version, pages, scope="whole", cache_pages=256)


def add_hash(h, x):
    raw = repr(x).encode()
    h.update(len(raw).to_bytes(8, "little"))
    h.update(raw)


def note(h, v, a):
    add_hash(
        h,
        (
            v.document,
            v.start,
            v.visible,
            v.tail,
            v.local_key,
            v.counts,
            v.latest,
            v.age,
            a.source,
            a.version,
            a.probes,
            a.key,
            a.histogram,
        ),
    )


def qualify(root):
    """Independent local scan + a separate native worker resolving suffix IDs (never as policy input)."""
    tic = clock()
    rows, windows, man, _ = training_data(32)
    checks = []
    work = Counter()
    with closing(Pages()) as p, closing(Pages()) as other:
        bank = bind(p, man)
        bind(other, man)
        for doc, win in zip(rows[:4], windows[:4], strict=True):
            local = DocumentCounts(doc["doc"], doc["start"], byte_limit=129)
            for i, b in enumerate(win[:128], 1):
                local.observe(b, at=i - 1)
                if i not in LADDER:
                    continue
                prefix = win[:i]
                view = local.view(at=i)
                ans = direct_branches(bank, [view], grouped=True, work=work)[0]
                ids = other.suffix(prefix)[-1]
                target = None
                hist = None
                for n in (8, 6, 4, 2, 1, 0):
                    if n > i:
                        continue
                    node = other.root if not n else ids[n - 1]
                    if node:
                        row = tuple(int(v) for v in other.rows([node])[0]["h"])
                        if any(row[:256]):
                            target = prefix[i - n :]
                            hist = row
                            break
                require(
                    (target, hist) == (ans.key, ans.histogram), "independent P100 selected histogram differs"
                )
                for n in (8, 4, 2, 1, 0):
                    if n > i:
                        continue
                    k = prefix[i - n :]
                    prev = [j for j in range(n, i) if prefix[j - n : j] == k]
                    if prev:
                        break
                cnt = Counter(prefix[j] for j in prev)
                require(
                    (
                        k,
                        tuple(cnt[j] for j in range(256)),
                        prefix[prev[-1]] if k else None,
                        i - 1 - prev[-1] if k else None,
                    )
                    == (view.local_key, view.counts, view.latest, view.age),
                    "independent local followers differ",
                )
                h = hashlib.sha256()
                note(h, view, ans)
                checks.append(
                    dict(
                        document=doc["doc"],
                        start=doc["start"],
                        visible=i,
                        evidence=h.hexdigest(),
                        archive_depth=None if target is None else len(target),
                        local_depth=len(k),
                    )
                )
            local.close()
    require(p.proc.returncode == other.proc.returncode == 0, "oracle workers did not close")
    dump(
        root / "oracle.json",
        dict(
            checks=checks, probes=work, rc=[p.proc.returncode, other.proc.returncode], wall_ns=clock() - tic
        ),
    )


def fresh(
    root,
    name,
    fd,
    *,
    gate_count=114,
    order=ORDER,
    pages_factory=Pages,
    bank_type=MaterializedBranches,
    fd_name="t9-private",
):
    ref = setup(root, gate_count=gate_count)
    start_ns = time_ns()
    tic = clock()
    run = root / name
    driver = load(root / "driver.json")
    require(
        driver["pid"] == os.getppid()
        and driver["ref"] == ref
        and name in order
        and not (run / "cell.json").exists()
        and os.readlink("/proc/self/fd/" + str(fd)).startswith("/memfd:" + fd_name)
        and os.fstat(fd).st_size == 0,
        "unbound worker/diagnostic fd",
    )
    rows, ws, man, data = training_data(32)
    prod = load(T4 / "identity.json")
    meta = load(T4 / "retained-packet.json")
    b, spec = binding("retained", rows, ws, data, prod)
    require(
        spec == meta["model_spec"]
        and prod["data"] == dict(data, window129_sha=[hashlib.sha256(w).hexdigest() for w in ws])
        and (man["mount"], man["exe_sha"]) == (prod["mount"], prod["branch_exe_sha"]),
        "wrong TRAIN/source",
    )
    packet = FrozenPacket.load(
        Path("output/memory-program-t4/attempt1/retained.f32"), b, meta["sha256"], byte_limit=CAP
    )
    locals_ = [DocumentCounts(row["doc"], row["start"], byte_limit=129) for row in rows]

    def session(s):
        def advance(raw):
            for byte in raw:
                s.observe(byte, at=s.visible)

        return Session(advance=advance, document=s.document, start=s.start)

    ss = [session(s) for s in locals_]
    hosts, base = packet.encoder(ss)(torch.tensor([list(w[:-1]) for w in ws]))
    require(
        tensor_sha(torch.cat((hosts, base), -1)) == meta["payload_sha"]
        and not hosts.requires_grad
        and not base.requires_grad,
        "changed host features",
    )
    net = DirectTrust(256).train()
    params = dict(net.named_parameters())
    init = digest_state(net)
    opt = torch.optim.AdamW(net.parameters(), **OPT)
    opt.zero_grad(set_to_none=True)
    hashes = [hashlib.sha256() for s in ss]
    logp = []
    votes = []
    ce = []
    work = Counter()
    availability = Counter()
    depth = Counter()
    timers = Counter()
    snap = {}

    def progress(phase, k):
        dump(run / "progress.json", dict(pid=os.getpid(), phase=phase, targets=k))

    progress("setup", 0)
    with closing(pages_factory()) as p:
        bank = bind(p, man, bank_type)
        cpu0, kid0, _ = process_usage([p.proc.pid])
        ts = clock()
        for s, w in zip(ss, ws, strict=True):
            s.observe(w[:1])
        for t in range(128):
            t0 = clock()
            views = [d.view(at=s.visible) for d, s in zip(locals_, ss, strict=True)]
            t1 = clock()
            ans = direct_branches(bank, views, grouped=name.startswith("grouped"), work=work)
            t2 = clock()
            lp, mix = net(hosts[:, t], base[:, t], direct_features(views, ans))
            ys = torch.tensor([w[t + 1] for w in ws])
            loss = -lp.gather(1, ys[:, None]).squeeze(1)
            ce.append(loss)
            logp.append(lp.detach())
            votes.append(mix.detach())
            t3 = clock()
            for ch, v, a in zip(hashes, views, ans, strict=True):
                require(v.visible == t + 1, "advanced before scoring")
                note(ch, v, a)
                depth["archive:" + ("none" if a.key is None else str(len(a.key)))] += 1
                depth["local:" + str(len(v.local_key))] += 1
                availability["host"] += 1
                availability["archive"] += int(a.key is not None)
                availability["local"] += 1
                availability["latest"] += int(v.latest is not None)
            for s, w in zip(ss, ws, strict=True):
                s.observe(w[t + 1 : t + 2])
            if not (t + 1) % 32:
                progress("scored", (t + 1) * 32)
            timers.update(
                local_view_ns=t1 - t0,
                archive_ns=t2 - t1,
                feature_head_score_ns=t3 - t2,
                attest_advance_ns=clock() - t3,
            )
        per = torch.stack(ce, 1)
        nll = per.mean()
        fwd = clock()
        nll.backward()
        back = clock()
        for k, v in params.items():
            require(v.grad is not None and v.grad.isfinite().all().item(), "missing direct adjoint")
            snap["grad:" + k] = v.grad.detach().clone()
        norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True))
        opt.step()
        end = clock()
        progress("updated", 4096)
        cpu1, kid1, rss = process_usage([p.proc.pid])
        after = digest_state(net)
        require(norm > 0 and after != init, "no direct learning")
        for k, v in params.items():
            snap["param:" + k] = v.detach()
            st = opt.state[v]
            require(st["step"].item() == 1, "wrong optimizer step")
            for attr, z in st.items():
                snap["adam." + attr + ":" + k] = z.detach()
        snap.update(
            byte_logp=torch.stack(logp, 1), gate=torch.stack(votes, 1), nlls=per.detach(), nll=nll.detach()
        )
        mass_err = (snap["byte_logp"].exp().sum(-1) - 1).abs().max().item()
        gate_err = (snap["gate"].sum(-1) - 1).abs().max().item()
        require(
            all(z.isfinite().all().item() for z in snap.values()) and max(mass_err, gate_err) <= 2e-5,
            "invalid direct probability/gradient/state",
        )
        local = dict(
            rows=sum(len(s.rows) for s in locals_),
            edges=sum(s.edges for s in locals_),
            increments=sum(s.increments for s in locals_),
            bytes=sum(s.visible for s in locals_),
        )
        require(all(s.visible == 129 for s in locals_), "incomplete document transitions")
        pages = dict(
            batches=bank.page_batches,
            rows=bank.pages_fetched,
            request_bytes=bank.page_batches * 5 + bank.pages_fetched * 4,
            response_bytes=bank.pages_fetched * ROW.itemsize,
            cache_pages=len(bank.cache),
            cap=bank.capacity,
        )
        bank.cache.clear()
        for s in locals_:
            s.close()
        require(
            all(s.closed and not s.rows and not s.tail for s in locals_) and not bank.cache, "cache cleanup"
        )
    require(p.proc.returncode == 0, "native exit")
    elapsed = clock() - tic
    meta_snap = {k: dict(sha=tensor_sha(z), shape=list(z.shape)) for k, z in snap.items()}
    require(sum(z.numel() * 4 for z in snap.values()) < PAYLOAD, "diagnostic IPC cap")
    with os.fdopen(os.dup(fd), "wb") as transfer:
        np.savez(transfer, **{k: z.numpy() for k, z in snap.items()})
    require(os.fstat(fd).st_size <= PAYLOAD, "encoded IPC cap")
    r = resource.getrusage(resource.RUSAGE_SELF)
    dump(
        run / "cell.json",
        dict(
            name=name,
            pid=os.getpid(),
            parent=os.getppid(),
            native_pid=p.proc.pid,
            native_rc=p.proc.returncode,
            ref=ref,
            start_epoch_ns=start_ns,
            end_epoch_ns=time_ns(),
            strict=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            threads=torch.get_num_threads(),
            cuda_available=torch.cuda.is_available(),
            cuda_initialized=torch.cuda.is_initialized(),
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
            torch=str(torch.__version__),
            packet=meta["sha256"],
            features=meta["payload_sha"],
            mount=man["mount"],
            binary=p.exe_sha,
            seed=SEED,
            parameters=sum(v.numel() for v in params.values()),
            init=init,
            after=after,
            snapshots=meta_snap,
            targets=4096,
            steps=1,
            nll=nll.item(),
            preclip_norm=norm,
            gate_mean=snap["gate"].mean((0, 1)).tolist(),
            availability=availability,
            depth=depth,
            unit_mass_err=mass_err,
            gate_mass_err=gate_err,
            per_document=[h.hexdigest() for h in hashes],
            host_calls=0,
            trust_calls=128,
            work=work,
            local=local,
            pages=pages,
            setup_ns=ts - tic,
            wall_ns=end - ts,
            forward_ns=fwd - ts,
            backward_ns=back - fwd,
            clip_attest_step_ns=end - back,
            clocks=timers,
            lifecycle_ns=elapsed,
            core_s=cpu1 - cpu0 + kid1 - kid0,
            peak_main_kib=r.ru_maxrss,
            native_rss_kib=rss[1],
            ipc_bytes=os.fstat(fd).st_size,
        ),
    )
    print(name, "scored 4096", round((end - ts) / 1e9, 3), "s", flush=True)


def main(root):
    tic = clock()
    ref = setup(root)
    require(
        time() + 485 < 1795531800 and not (root / "driver.json").exists(), "no time or existing measurement"
    )
    dump(
        root / "driver.json", dict(pid=os.getpid(), ref=ref, order=ORDER, start_epoch_ns=time_ns(), card_s=0)
    )
    qualify(root)
    baseline = None
    for i, name in enumerate(ORDER):
        run = root / name
        run.mkdir()
        stamp = clock()
        with (
            os.fdopen(os.memfd_create("t9-private", os.MFD_CLOEXEC), "w+b") as fd,
            (run / "console").open("wb") as out,
        ):
            pr = subprocess.run(
                [sys.executable, "-m", __spec__.name, str(root), "--worker", name, "--fd", str(fd.fileno())],
                pass_fds=(fd.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=max(1.0, 470 - (clock() - tic) / 1e9),
            )
            dump(run / "exit.json", dict(rc=pr.returncode, process_wall_ns=clock() - stamp))
            require(pr.returncode == 0, "direct subprocess stopped: " + name)
            s = load(run / "cell.json")
            fd.seek(0)
            with np.load(fd, allow_pickle=False) as z:
                require(set(z.files) == set(s["snapshots"]), "missing cell tensors")
                xx = {k: torch.from_numpy(z[k]) for k in z.files}
            require(
                all(
                    v.dtype == torch.float32
                    and list(v.shape) == s["snapshots"][k]["shape"]
                    and tensor_sha(v) == s["snapshots"][k]["sha"]
                    for k, v in xx.items()
                ),
                "diagnostic transfer mismatch",
            )
        if baseline is None:
            baseline = xx
            control = s
        else:
            checks = {k: delta(baseline[k], v) for k, v in xx.items()}
            same = all(
                s[k] == control[k]
                for k in (
                    "per_document",
                    "init",
                    "after",
                    "availability",
                    "depth",
                    "local",
                    "preclip_norm",
                    "gate_mean",
                    "nll",
                )
            ) and all(
                s["work"].get(k, 0) == control["work"].get(k, 0)
                for k in ("selected", "unavailable", "absent", "end_only", "logical_probes")
            )
            dump(run / "vs-serial-A.json", dict(same_logical=same, tensors=checks))
            require(
                same and all(d["unequal_bits"] == 0 for d in checks.values()), "direct parity gate failed"
            )
        print(name, round(s["wall_ns"] / 1e9, 3), "s;", s["pages"]["batches"], "page RPCs", flush=True)
    no_card()
    dump(
        root / "complete.json",
        dict(
            pid=os.getpid(),
            targets=16384,
            steps=4,
            distinct_train_positions=4096,
            elapsed_ns=clock() - tic,
            end_epoch_ns=time_ns(),
            card_s=0,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--worker", choices=ORDER)
    ap.add_argument("--fd", type=int)
    a = ap.parse_args()
    fresh(a.root, a.worker, a.fd) if a.worker else main(a.root)
