"""CPU reader-lifetime diagnostic with real, already registered P100 positive locations. No LM."""

import argparse
from collections import Counter
from contextlib import ExitStack
import gc
from hashlib import sha256
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter_ns as clock, time, time_ns
from weakref import ref as weakref

import numpy as np
import torch
from ndm.memory_program.runtime import Span, Unavailable
from ndm.memory_program.spans import mean_byte_values
from .branch_domain_bench import WavePages, high_water
from .direct_control_bench import add_hash, setup
from .frozen_packet_bench import no_card
from .memory_program_mounts import ArchiveRecords
from .pinned_branch import BINARY, ROOT as NVME, p100_domain
from .scoped_records import ScopedRecords
from .serving_verify import HERE, load, require
from .w3_data import EXE as BRANCH_EXE
from .workbench_native import EXE, EXE_SHA
from .workbench_run import dump, sha

ORDER = ("ordinary-A", "retire-A", "retire-B", "ordinary-B")
GATES = 180  # frozen suite, including scoped-record cases
WAVES = 6
DOCUMENTS = 16119767
TABLE_BYTES = DOCUMENTS * 8
REG = HERE.parent / "x43_fork_resolved/receipts/p100_query_verification/registration.json"
SAMPLES = HERE / "receipts/b2/samples.jsonl"
REG_SHA = "5ea29f13388c3a3b3fc402cb54f41be51c61fdef664707a7f1d6b215e2b0bef1"
SAMPLES_SHA = "520c669473b714ad00e92b396d064663efa176d9312f7d8ff65da0d1f153aaba"
MOUNTS_SHA = "eabfebf0b7b02b3f0cfdd8d213efd7cacafcca378b792474d4ac6ab4e0573c93"


def fixture():
    p = HERE / "b1_mounts.json"
    require(
        sha(p) == MOUNTS_SHA and sha(REG) == REG_SHA and sha(SAMPLES) == SAMPLES_SHA,
        "changed positive oracle",
    )
    keys = tuple(bytes.fromhex(load(REG)["patterns"][i]["hex"]) for i in (2, 6))
    import json

    with SAMPLES.open() as f:
        row = next(
            z
            for z in map(json.loads, f)
            if z["bank"] == "P100" and z["ids"] == [2, 6] and z["mode"] == "original"
        )
    require(
        [sha256(k).hexdigest() for k in keys] == row["pattern_hashes"]
        and [len(v) for v in row["positions"]] == [2, 1],
        "wrong published positive keys/positions",
    )
    conf = load(p)["P100"]
    man = dict(mount=row["version"], exe_sha=BINARY)
    files = {str(NVME / v["path"]) for v in conf["files"]} | {str(EXE), str(BRANCH_EXE)}
    return keys, row, conf["files"][1], man, files


class Encoder:
    """Disposable value encoder. Not a target, policy, optimizer or model quality fixture."""

    def __init__(self):
        self.weight = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8)
        self.calls = 0

    def __call__(self, fragments):
        self.calls += 1
        return mean_byte_values(fragments, self.weight)


def check_value(hit, bs):
    # Independent of embedding_bag. Column j of byte b is exactly the integer 8*b+j.
    expect = np.asarray([sum(v * 8 + j for v in bs) / len(bs) for j in range(8)], dtype="<f4")
    t = hit.value
    require(
        t.device.type == "cpu"
        and t.dtype == torch.float32
        and not t.requires_grad
        and hit.mass == 1
        and np.array_equal(expect, t.numpy()),
        "wrong encoded complete span",
    )
    return t.numpy().tobytes()


def issued(readers, desc, keys, old):
    loc, contents, follow = readers
    require(
        [r.operation for r in readers] == ["occurrences", "contents", "following"]
        and all((r.source, r.version) == ("P100", desc.version) and r.watermark is None for r in readers),
        "wrong immutable readers",
    )
    h = sha256()
    coordinate = sha256()
    sizes = Counter()
    patterns = (*keys, b" the ")
    add_hash(h, ("occurrences", patterns, 8))
    a, b, over = loc.read_many(patterns, 8)
    require(
        over == Unavailable("raw interval exceeds 8") and (len(a), len(b)) == (2, 1),
        "lost positive or overcap",
    )
    add_hash(h, over.reason)
    spans = []
    absolute = []
    data = []
    with desc.text.open("rb", buffering=0) as f:
        for hits, k, registered in zip((a, b), keys, old["positions"], strict=True):
            found = []
            for hit in hits:
                require(
                    len(hit.keys) == 1
                    and isinstance(hit.keys[0], Span)
                    and hit.value is None
                    and hit.mass == 1,
                    "invalid returned address",
                )
                s = hit.keys[0]
                pos = int(desc.starts[s.document]) + s.start
                # Reconstruct both coordinates independently, not by trusting the locator's document id.
                doc = int(desc.starts.searchsorted(pos, side="right")) - 1
                lo = int(desc.starts[doc])
                hi = int(desc.starts[doc + 1]) if doc + 1 < len(desc.starts) else desc.size
                require(
                    pos in registered
                    and s == Span("P100", desc.version, doc, pos - lo, pos - lo + len(k))
                    and lo <= pos < pos + len(k) <= hi,
                    "outside historical position/document",
                )
                text = os.pread(f.fileno(), len(k), pos)
                nxt = os.pread(f.fileno(), min(16, hi - pos - len(k)), pos + len(k))
                require(text == k and len(nxt) == min(16, hi - pos - len(k)), "incomplete independent pread")
                found.append(pos)
                spans.append(s)
                data.append((text, nxt))
                add_hash(h, (s, hit.mass))
            require(sorted(found) == sorted(registered), "incomplete position list")
            absolute.append(sorted(found))
    add_hash(coordinate, absolute)
    add_hash(h, ("contents", tuple(spans), 8))
    xs = contents.read_many(tuple(spans), 8)
    add_hash(h, ("following", tuple(spans), 8))
    ys = follow.read_many(tuple(spans), 8)
    for s, (text, nxt), x, y in zip(spans, data, xs, ys, strict=True):
        require(len(x) == 1 and x[0].keys == (text,), "wrong complete contents")
        add_hash(h, (x[0].keys, check_value(x[0], text)))
        sizes.update(contents=1, bytes=len(text), values=8)
        if nxt:
            next_span = Span(s.source, s.version, s.document, s.stop, s.stop + len(nxt))
            require(len(y) == 1 and y[0].keys == (nxt[:1], nxt, next_span), "wrong visible continuation")
            add_hash(h, (y[0].keys, check_value(y[0], nxt)))
            sizes.update(following=1, bytes=len(nxt), values=8)
        else:
            require(y == (), "false continuation across document")
            add_hash(h, ())
    return dict(
        evidence=h.hexdigest(),
        positions=coordinate.hexdigest(),
        occurrence_keys=3,
        spans=len(spans),
        certified_refusals=1,
        independent_preads=2 * len(spans),
        **sizes,
    )


def sample(refs, files):
    status = Path("/proc/self/status").read_text().splitlines()
    mem = {
        k: int(next(l.split()[1] for l in status if l.startswith(k + ":")))
        for k in ("VmRSS", "VmHWM", "RssAnon", "RssFile")
    }
    fds = os.listdir("/proc/self/fd")
    pinned = 0
    for fd in fds:
        try:
            path = os.readlink("/proc/self/fd/" + fd)
        except FileNotFoundError:
            continue  # the directory enumeration fd has already closed
        pinned += path in files
    return dict(
        **mem,
        fds=len(fds),
        registered_fds=pinned,
        alive={k: [v[k]() is not None for v in refs] for k in ("descriptor", "starts", "encoder")},
        collections=[s["collections"] for s in gc.get_stats()],
        gc_counts=list(gc.get_count()),
        clock_ns=clock(),
    )


def core():
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def wave(run, number, retire, inputs):
    keys, old, receipt, man, _ = inputs
    pages = []

    def factory(**kw):
        require(not pages, "duplicated m20 open")
        p = WavePages(**kw)
        pages.append(p)
        p.begin()
        return p

    start = clock()
    with p100_domain(man, scope="whole", max_grants=1, pages_type=factory) as (domain, desc):
        require(desc.starts.nbytes == TABLE_BYTES and not desc.starts.flags.writeable, "wrong offset table")
        enc = Encoder()
        wr = dict(descriptor=weakref(desc), starts=weakref(desc.starts), encoder=weakref(enc))
        with domain.borrow("P100", man["mount"], scope="whole") as grant:
            with ExitStack() as stack:
                if retire:
                    scope = stack.enter_context(
                        ScopedRecords(desc, receipt, scope="whole", branches=grant, encode=enc)
                    )
                    pid, opened = scope.pid, scope.start_ns
                    readers = scope.readers()
                else:
                    rec = ArchiveRecords(desc, receipt, scope="whole", branches=grant, encode=enc)
                    stack.callback(rec.close)
                    readers = rec.readers()
                    pid = rec.worker.proc.pid
                    opened = rec.worker.start_ns
                require(sha(Path(f"/proc/{pid}/exe")) == EXE_SHA, "unbound running positional executable")
                dump(
                    run / "progress.json",
                    dict(
                        pid=os.getpid(),
                        phase="opened",
                        wave=number,
                        completed=number - 1,
                        position_pid=pid,
                        branch_pid=pages[0].proc.pid,
                    ),
                )
                ready = clock()
                evidence = issued(readers, desc, keys, old)
                queried = clock()
                require(enc.calls == 2, "wrong actual value encoding")
                peak = high_water(pid)
            position = (
                scope.final
                if retire
                else dict(
                    pid=pid,
                    rc=rec.worker.proc.poll(),
                    fd_closed=rec.f.closed,
                    start_ns=opened,
                    work=dict(rec.stats),
                )
            )
            require(
                position["pid"] == pid and position["rc"] == 0 and position["fd_closed"],
                "unclosed positional actor",
            )
        wirehash = pages[0].finish()
    end = clock()
    require(
        domain.closed and grant.final is not None and pages[0].proc.returncode == 0, "unclosed branch actor"
    )
    out = dict(
        number=number,
        retain_handles=number > 3,
        start_ns=start,
        open_through_check_ns=ready - start,
        query_ns=queried - ready,
        close_ns=end - queried,
        whole_ns=end - start,
        position=position,
        position_peak_kib=peak,
        evidence=evidence,
        wire=pages[0].receipt(),
        row_frames=wirehash,
        grant=grant.final,
    )
    return out, wr, (readers if number > 3 else ())


def worker(root, name):
    ref = setup(root, gate_count=GATES)
    run = root / name
    t = clock()
    epoch = time_ns()
    require(
        gc.isenabled()
        and load(root / "driver.json")["pid"] == os.getppid()
        and not (run / "cell.json").exists(),
        "unbound lifetime worker",
    )
    inputs = fixture()
    files = inputs[-1]
    refs = []
    retained = []
    waves = []
    retire = name.startswith("retire")
    before = sample(refs, files)
    cpu = core()
    start = clock()
    for n in range(1, WAVES + 1):
        x, wr, rs = wave(run, n, retire, inputs)
        refs.append(wr)
        if rs:
            retained.extend(rs)
        del rs
        snap = sample(refs, files)
        require(gc.isenabled() and snap["registered_fds"] == 0, "registered file held after close")
        x["post_close"] = snap
        waves.append(x)
        dump(run / f"wave-{n}.json", x)
        dump(run / "progress.json", dict(pid=os.getpid(), phase="released", wave=n, completed=n))
    wall = clock() - start
    cpu = core() - cpu
    require(len(retained) == 9, "wrong retained public handles")
    gc1 = gc.collect()
    held = sample(refs, files)
    refused = 0
    if retire:
        for endpoint in retained:
            try:
                endpoint.read_many((), 8)
            except ValueError as e:
                require(str(e) == "retired record reader", "wrong stale refusal")
                refused += 1
            else:
                raise ValueError("retired reader remained active")
        del endpoint
    retained.clear()
    gc2 = gc.collect()
    dropped = sample(refs, files)
    no_card()
    cell = dict(
        ref=ref,
        name=name,
        pid=os.getpid(),
        parent=os.getppid(),
        start_epoch_ns=epoch,
        end_epoch_ns=time_ns(),
        threads=torch.get_num_threads(),
        visible=os.environ["CUDA_VISIBLE_DEVICES"],
        cuda_available=torch.cuda.is_available(),
        cuda_initialized=torch.cuda.is_initialized(),
        strict=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        torch=str(torch.__version__),
        inputs=dict(
            mount=inputs[3]["mount"],
            branch_exe_sha=BINARY,
            position_exe_sha=EXE_SHA,
            registration_sha=REG_SHA,
            samples_sha=SAMPLES_SHA,
            mounts_sha=MOUNTS_SHA,
            pattern_hashes=inputs[1]["pattern_hashes"],
            table_bytes=TABLE_BYTES,
        ),
        gc_enabled=gc.isenabled(),
        gc_threshold=list(gc.get_threshold()),
        before=before,
        held_after_gc=held,
        all_dropped_after_gc=dropped,
        collected=[gc1, gc2],
        stale_refusals=refused,
        wall_ns=wall,
        core_s=cpu,
        lifecycle_ns=clock() - t,
        waves=len(waves),
        handles_at_first_gc=9,
        explicit_collections_in_primary=0,
        card_s=0,
    )
    dump(run / "cell.json", cell)
    print(
        name,
        "closed six pairs; alive tables before, after diagnostic GC with held handles, then after dropping:",
        *[sum(v["alive"]["starts"]) for v in (waves[-1]["post_close"], held, dropped)],
        flush=True,
    )


def main(root):
    t = clock()
    ref = setup(root, gate_count=GATES)
    require(
        time() + 370 < 1795531800 and not (root / "driver.json").exists(), "too late or repeated diagnostic"
    )
    dump(
        root / "driver.json", dict(ref=ref, pid=os.getpid(), order=ORDER, start_epoch_ns=time_ns(), card_s=0)
    )
    for name in ORDER:
        run = root / name
        run.mkdir()
        start = clock()
        with (run / "console").open("xb") as out:
            try:
                pr = subprocess.run(
                    [sys.executable, "-m", __spec__.name, str(root), "--cell", name],
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    timeout=max(1, 350 - (start - t) / 1e9),
                )
            except subprocess.TimeoutExpired:
                dump(run / "exit.json", dict(timeout=True, process_wall_ns=clock() - start))
                raise
        dump(run / "exit.json", dict(rc=pr.returncode, process_wall_ns=clock() - start))
        require(pr.returncode == 0, "incomplete cell: " + name)
        # Do not decide a price gate until all four prescribed lifetimes are complete.
        from .record_retirement_verify import check_cell

        check_cell(root, name, ref)
        print(name, "complete", flush=True)
    dump(
        root / "complete.json",
        dict(ref=ref, pid=os.getpid(), end_epoch_ns=time_ns(), elapsed_ns=clock() - t, waves=24, card_s=0),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--cell", choices=ORDER)
    args = ap.parse_args()
    if args.cell:
        worker(args.root, args.cell)
    else:
        main(args.root)
