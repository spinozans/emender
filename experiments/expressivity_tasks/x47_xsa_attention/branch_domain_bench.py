"""S15 CPU-only: same direct/sampled decisions on known frontiers, four vs. one immutable opens."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
import gc
from hashlib import sha256
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
from time import perf_counter_ns as clock, time, time_ns

import torch
from ndm.memory_program import Limits, PackedMemoryPolicy
from ndm.memory_program.direct_service import DirectReadout, DirectStream
from ndm.memory_program.frozen_packet import FrozenPacket
from ndm.memory_program.serving import StreamUniforms
from .arrival_service_bench import children_cpu
from .direct_control_bench import T4, add_hash, note, setup
from .direct_service_head import load_head
from .frozen_packet_bench import CAP, no_card
from .memory_program_profile import bind_program_readers, training_data
from .packed_program_profile import BindingNames, digest_rows, events
from .pinned_branch import p100_domain
from .serving_bench import CK, CK_SHA, process_usage
from .serving_verify import HERE, load, require
from .stream_packet_bench import binding
from .terminal_branch_bench import WirePages
from .workbench_run import dump, sha
from .workbench_witness import digest_state

GATES = 137
ORDER = ("fresh-A", "resident-A", "resident-B", "fresh-B")


class WaveTap:
    def begin(self):
        require(getattr(self, "_tap", None) is None, "overlapping wire tap")
        self._tap = sha256()
        self._before = self.wire["R"].copy()

    def rpc(self, op, n, payload, size):
        require(op == b"R" and self._tap is not None, "unchosen/out-of-wave frame")
        data = super().rpc(op, n, payload, size)
        self._tap.update(op + struct.pack("<I", n))
        self._tap.update(payload)
        self._tap.update(data)
        return data

    def finish(self):
        sig = self._tap.hexdigest()
        self._tap = None
        return dict(sha=sig, **(self.wire["R"] - self._before))


class WavePages(WaveTap, WirePages):
    pass


class Meter:
    def __init__(self, man):
        self.man = man
        self.pages = None
        self.opens = []

    def factory(self, *, pinned_fds):
        require(self.pages is None, "two m20 workers at once")
        self.pages = WavePages(pinned_fds=pinned_fds)
        return self.pages

    def domain(self, count):
        return p100_domain(self.man, scope="whole", max_grants=count, pages_type=self.factory)

    @contextmanager
    def schedule(self, label):
        yield [range(4)] if label.startswith("resident") else [[i] for i in range(4)]

    def extra_cell(self):
        return {}

    @contextmanager
    def open(self, count):
        start = clock()
        try:
            with self.domain(count) as pair:
                ready = clock()
                yield pair
                end = clock()
            require(pair[0].closed and pair[0].grants == count, "incomplete grant lifecycle")
            self.opens.append(
                dict(
                    open_ns=ready - start,
                    final_close_ns=clock() - end,
                    wire=self.pages.receipt(),
                    grants=count,
                )
            )
        finally:
            self.pages = None


class Digests:
    def __init__(self, ix):
        self.ix = ix
        self.hashes = [{k: sha256() for k in ("byte_logp", "factor", "argmax", "evidence")} for _ in ix]
        self.count = [0 for _ in ix]

    def score(self, j, s, lp, other):
        d = self.hashes[j]
        require(
            lp.shape == (256,)
            and len(other.reshape(-1)) in (1, 4)
            and lp.dtype == other.dtype == torch.float32
            and lp.isfinite().all()
            and other.isfinite().all()
            and lp.logsumexp(0).abs() < 5e-5
            and not lp.requires_grad
            and s.visible == 1 + self.count[j],
            "unbound or invalid prefix score",
        )
        d["byte_logp"].update(lp.contiguous().numpy().tobytes())
        d["factor"].update(other.contiguous().numpy().tobytes())
        d["argmax"].update(bytes([lp.argmax().item()]))
        self.count[j] += 1

    def report(self, words):
        require(self.count == [32] * len(self.ix), "missing frontiers")
        return [
            dict(
                row=i,
                frontiers=n,
                observed32_sha=sha256(w[:32]).hexdigest(),
                **{k: v.hexdigest() for k, v in z.items()},
            )
            for i, n, w, z in zip(self.ix, self.count, words, self.hashes, strict=True)
        ]


def prepare(*, shifted_targets=False):
    ds = training_data(32)
    rows, ws, man, data = ds
    prod = load(T4 / "identity.json")
    pm = load(T4 / "retained-packet.json")
    b, spec = binding("retained", rows, ws, data, prod)
    require(
        spec == pm["model_spec"]
        and prod["data"] == dict(data, window129_sha=[sha256(w).hexdigest() for w in ws])
        and (man["mount"], man["exe_sha"]) == (prod["mount"], prod["branch_exe_sha"])
        and sha(CK) == CK_SHA,
        "changed fixture",
    )
    packet = FrozenPacket.load(
        Path("output/memory-program-t4/attempt1/retained.f32"), b, pm["sha256"], byte_limit=CAP
    )
    head, meta = load_head(Path("output/memory-program-s14"))
    require(meta == load(HERE / "receipts/memory-program-s14/head.json"), "changed frozen direct predictor")
    ck = torch.load(CK, map_location="cpu", weights_only=True)
    policy = PackedMemoryPolicy(256).eval().requires_grad_(False)
    policy.load_state_dict(ck["model"], strict=True)
    require(digest_state(policy) == ck["policy_sha"] == prod["policy_sha"], "changed sampled predictor")
    identity = dict(
        head=meta["after"],
        policy=ck["policy_sha"],
        checkpoint=CK_SHA,
        features=pm["payload_sha"],
        packet=pm["sha256"],
        mount=man["mount"],
        binary=man["exe_sha"],
        seed=7530,
        input128_sha=[w.input_sha for w in b.rows],
        observed32_sha=[sha256(w[:32]).hexdigest() for w in ws],
        split=b.split,
    )
    return (rows, ws if shifted_targets else [w[:-1] for w in ws], packet, head, policy), man, identity


def high_water(pid):
    lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    return int(next(row.split()[1] for row in lines if row.startswith("VmHWM:")))


def prefix_wave(wno, desc, bank, inputs):
    start = clock()
    rows, ws, packet, head, policy = inputs
    ix = list(range(wno * 8, wno * 8 + 8))
    selected = [rows[i] for i in ix]
    words = [ws[i] for i in ix]
    sampled = wno % 2 == 1
    limits = Limits(pattern=16)
    hashes = Digests(ix)
    actors = []
    stat = defaultdict(Counter)
    append = Counter()
    disp = []
    opcodes = Counter()
    streams = []
    record = None
    with ExitStack() as stack:
        if sampled:
            record, ss, lives = bind_program_readers(
                stack, policy, selected, desc, bank, stat, append, limits, lambda *args: None, disp
            )
            actors = [record.worker.proc, *(l.native.proc for l in lives)]
            runner = policy.batch_runner()
            names = BindingNames(ss)
        else:
            streams = [DirectStream(d["doc"], d["start"], limits=limits, byte_limit=128) for d in selected]
            ss = [d.session for d in streams]
            for a in streams:
                stack.callback(a.close)
            idx = {s.id: j for j, s in enumerate(ss)}

            def observe(s, v, a, lp, mix):
                j = idx[s.id]
                hashes.score(j, s, lp, mix)
                note(hashes.hashes[j]["evidence"], v, a)

            direct = DirectReadout(head, bank, observe=observe)
            for s, a in zip(ss, streams, strict=True):
                direct.validate_binding(s, a)
        h, base = packet.encoder(ss, row_indices=ix)(torch.tensor([list(w) for w in words]))
        for s, w in zip(ss, words, strict=True):
            s.observe(w[:1])
        if sampled:
            rng = StreamUniforms(ss, ix, 32, 7530, "cpu")
        play = clock()
        for t in range(32):
            if sampled:
                ds = runner(ss, h[:, t], base[:, t], mode="sample", generator=rng)
                for j, (s, d, ev) in enumerate(zip(ss, ds, events([ds], names), strict=True)):
                    require(
                        s.id == d.session_id
                        and s.visible == d.visible
                        and s.pending
                        and d.mode == "sample",
                        "not an actual sampled decision",
                    )
                    hashes.score(j, s, d.log_probs, d.log_policy)
                    add_hash(hashes.hashes[j]["evidence"], ev)
                    opcodes.update(e.inst.op for e in d.trace)
            else:
                direct(ss, streams, h[:, t], base[:, t])
            if t < 31:
                for s, w in zip(ss, words, strict=True):
                    s.observe(w[t + 1 : t + 2])
        stop = clock()
        actor_peaks = [high_water(p.pid) for p in actors]
        require(all(s.visible == 32 for s in ss), "wrong final observation count")
        if sampled:
            runner.keywords["_memo"]["cells"].clear()
            logical = dict(
                opcodes=opcodes,
                dispatches=len(disp),
                dispatch_sha=digest_rows(
                    [(names.origin(o), c, tuple(map(names.key, ks))) for o, c, ks in disp]
                ),
                provider={k: {n: v for n, v in d.items() if n != "ns"} for k, d in stat.items()},
                archive={k: v for k, v in record.stats.items() if not k.endswith("_ns")},
                live=sum((l.stats for l in lives), Counter()),
            )
            for l, w in zip(lives, words, strict=True):
                require(bytes(l.data) == w[:32] and l.native.visible == 32, "stale live mirror")
                l.data.clear()
                l.native.tail = b""
        else:
            require(
                all(
                    a.counts.visible == 32 and a.counts.tail == w[24:32]
                    for a, w in zip(streams, words, strict=True)
                ),
                "stale direct counts",
            )
            logical = dict(direct=direct.work, local_increments=sum(a.counts.increments for a in streams))
        for s in ss:
            s.recent = b""
            s.cells = [None] * limits.cells
            s.advance = None
            s.readers = ()
            s.pending = True
    clean = (
        all(p.returncode == 0 for p in actors)
        and (not sampled or record.f.closed)
        and all(a.counts.closed and not a.counts.rows for a in streams)
    )
    require(clean, "unreleased wave resources")
    return dict(
        wave=wno,
        predictor="sampled" if sampled else "direct",
        frontiers=256,
        documents=hashes.report(words),
        logical=logical,
        actor_exits=[(p.pid, p.returncode) for p in actors],
        actor_peak_kib=actor_peaks,
        sessions_cleared=True,
        learner_setup_ns=play - start,
        score_ns=stop - play,
        private_lifecycle_ns=clock() - start,
        native_ns={k: v["ns"] for k, v in stat.items()},
        append_ns=append["append_ns"],
    )


@torch.no_grad()
def worker(root, label, *, gates=GATES, meter_type=Meter):
    ref = setup(root, gate_count=gates)
    start = time_ns()
    tic = clock()
    run = root / label
    require(
        load(root / "driver.json")["pid"] == os.getppid() and not (run / "cell.json").exists(),
        "unbound/reused worker",
    )
    inputs, man, identity = prepare()
    meter = meter_type(man)
    num = 0

    def group(waves):
        nonlocal num
        with meter.open(len(waves)) as (domain, desc):
            for i in waves:
                tm = clock()
                meter.pages.begin()
                with domain.borrow("P100", man["mount"], scope="whole") as bank:
                    stale = bank.reader()
                    wave = prefix_wave(i, desc, bank, inputs)
                try:
                    stale.read_many((b"",), 256)
                except ValueError as e:
                    require("expired" in str(e), "wrong stale refusal")
                else:
                    raise ValueError("expired grant served")
                gc.collect()
                wave.update(
                    wall_ns=clock() - tm,
                    grant=bank.final,
                    wire=meter.pages.finish(),
                    m20_pid=meter.pages.proc.pid,
                    m20_running_peak_kib=high_water(meter.pages.proc.pid),
                    expired_refused=True,
                )
                dump(run / f"wave{i}.json", wave)
                num += wave["frontiers"]
                dump(run / "progress.json", dict(pid=os.getpid(), frontiers=num, wave=i))

    cpu0 = process_usage([])[0]
    child0 = children_cpu()
    begin = clock()
    with meter.schedule(label) as groups:
        for group_indices in groups:
            group(group_indices)
    wall = clock() - begin
    cpu1 = process_usage([])[0]
    child1 = children_cpu()
    require(
        meter.pages is None
        and num == 1024
        and digest_state(inputs[-2]) == identity["head"]
        and digest_state(inputs[-1]) == identity["policy"],
        "unfinished or changed model",
    )
    no_card()
    dump(
        run / "cell.json",
        dict(
            name=label,
            ref=ref,
            pid=os.getpid(),
            parent=os.getppid(),
            identity=identity,
            opens=meter.opens,
            **meter.extra_cell(),
            wall_ns=wall,
            lifecycle_ns=clock() - tic,
            start_epoch_ns=start,
            end_epoch_ns=time_ns(),
            main_core_s=cpu1 - cpu0,
            children_core_s=child1 - child0,
            main_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            frontiers=num,
            steps=0,
            visible=os.environ["CUDA_VISIBLE_DEVICES"],
            cuda_initialized=torch.cuda.is_initialized(),
            cuda_available=torch.cuda.is_available(),
            strict=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            threads=torch.get_num_threads(),
            torch=str(torch.__version__),
        ),
    )
    print(label, num, "prefix predictions in", round(wall / 1e9, 3), "s incl open/close", flush=True)


def signature(w):
    return dict(
        documents=w["documents"],
        logical=w["logical"],
        wire={k: v for k, v in w["wire"].items() if k != "wall_ns"},
        grant=w["grant"],
    )


def main(root, *, gates=GATES, order=ORDER, module=__spec__.name, compare_to=None):
    tic = clock()
    ref = setup(root, gate_count=gates)
    require(time() + 365 < 1795531800 and not (root / "driver.json").exists(), "out of time or reused run")
    dump(
        root / "driver.json",
        dict(pid=os.getpid(), ref=ref, order=order, card_s=0, start_epoch_ns=time_ns()),
    )
    baseline = (
        None if compare_to is None else [signature(load(compare_to / f"wave{i}.json")) for i in range(4)]
    )
    for label in order:
        run = root / label
        run.mkdir()
        t = clock()
        with (run / "console").open("xb") as log:
            p = subprocess.run(
                [sys.executable, "-m", module, str(root), "--cell", label],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(1, 355 - (clock() - tic) / 1e9),
            )
        dump(run / "exit.json", dict(rc=p.returncode, process_wall_ns=clock() - t))
        require(p.returncode == 0, "failed " + label)
        sig = [signature(load(run / f"wave{i}.json")) for i in range(4)]
        if baseline is None:
            baseline = sig
        require(sig == baseline, "different prediction/dispatch/native page stream")
        print(label, round(load(run / "cell.json")["wall_ns"] / 1e9, 3), "s, verified work", flush=True)
    no_card()
    dump(
        root / "complete.json",
        dict(
            pid=os.getpid(),
            frontiers=4096,
            unique_frontiers=1024,
            training_targets=0,
            updates=0,
            card_s=0,
            elapsed_ns=clock() - tic,
            end_epoch_ns=time_ns(),
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--cell", choices=ORDER)
    a = ap.parse_args()
    worker(a.root, a.cell) if a.cell else main(a.root)
