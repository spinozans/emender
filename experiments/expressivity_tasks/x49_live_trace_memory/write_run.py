"""Admission for the independently registered X49-W local campaign. No approval generator or recovery lease."""

from __future__ import annotations
from functools import partial
import json
import os
import re
import subprocess
from time import monotonic

from ..x48_muon_continuation.data import ROOT as REPO, check, load, sha, write_json
from . import followup_loop as loop, panel as p, write_panel as w
from .meter import HERE, MODULE, PY
from .panels import OUT, SEEDS
from .write_data import ROWS, SPEC, WriteStream, selection
from .write_origin import origin

MOD = MODULE + ".write_run"
AUTH = OUT / "write-authorization-1.json"
GATES = 72
EXTERNAL_TESTS = ["tests/test_memory_program_direct.py", "tests/test_x49_relative.py"]


def source():
    ref = p.source()
    check(
        sha(HERE / "WRITE_READ.md") == SPEC
        and sum(j.cap for j in w.JOBS.values()) == w.TOTAL
        and len(w.JOBS) == 10,
        "changed W registration",
    )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *EXTERNAL_TESTS], cwd=REPO, check=True
    )
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", *EXTERNAL_TESTS],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        check=True,
    )
    return ref


def qualification():
    import torch
    from .endpoint import state_digest
    from .coupled import optimizers

    p.cpu()
    begin = monotonic()
    ref = source()
    qdir = OUT / ("qualify-write-" + ref)
    qdir.mkdir()
    log = qdir / "gate.log"
    prep = w.prep()
    check(load(ROWS) == selection(), "changed W selection before qualification")
    stream = WriteStream(w.W_ID)
    check(len(stream.docs) == 1024, "incorrect TRAIN dose")
    births = {}
    for s in SEEDS:
        host, head = origin(s, "L")
        opts, groups = optimizers(host, head)
        check(
            all(not opt.state and not opt.aux.state for opt in opts)
            and len(groups["host"]["muon"]) == 191
            and len(groups["head"]["muon"]) == 6
            and len(groups["head"]["adamw"]) == 9
            and not torch.count_nonzero(head.out.weight)
            and not torch.count_nonzero(head.out.bias),
            "wrong W birth/routing",
        )
        births[str(s)] = dict(
            host=state_digest(host.state_dict()),
            literal=state_digest(head.literal.state_dict()),
            linked=state_digest(head.state_dict()),
        )
        del opts, host, head
    check(not torch.cuda.is_initialized(), "qualification touched a device")
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTEST_ADDOPTS="")
    with log.open("xb") as f:
        for cmd in (
            [PY, "-m", "pytest", "-q", str(HERE), *EXTERNAL_TESTS],
            [PY, "-m", MODULE + ".native_check"],
            [PY, "-m", MODULE + ".trace_native"],
        ):
            subprocess.run(
                cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, check=True
            )
    text = log.read_text()
    passes = re.findall(r"=+ (\d+) passed in [^\n]+ =+", text)
    obj = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    check(
        passes == [str(GATES)]
        and len(obj) == 2
        and obj[0]["work"]["matched_frontiers"] == 654
        and obj[1]["counts"]["frontiers"] == 542
        and source() == ref,
        "incomplete W qualification",
    )
    q = dict(
        ref=ref,
        spec=SPEC,
        rows_sha=w.ROW_SHA,
        train_identity=w.W_ID,
        passed=GATES,
        log_sha=sha(log),
        births=births,
        short_documents=prep["short_documents"],
        short_targets=prep["short_targets"],
        evaluation_windows_opened=0,
        wall_s=monotonic() - begin,
    )
    write_json(qdir / "passed.json", q)
    return q


def authority():
    check(AUTH.is_file(), "no independent W campaign budget")
    ref = source()
    w.validate(load(AUTH), ref)
    w.prep()
    d = OUT / ("qualify-write-" + ref)
    q = load(d / "passed.json")
    check(
        q["ref"] == ref
        and q["spec"] == SPEC
        and q["rows_sha"] == w.ROW_SHA
        and q["train_identity"] == w.W_ID
        and q["passed"] == GATES
        and q["log_sha"] == sha(d / "gate.log")
        and q["evaluation_windows_opened"] == 0
        and set(q["births"]) == {str(s) for s in SEEDS},
        "W source not qualified",
    )
    return ref


def admit(job):
    return loop.admit(job, w, authority)


def guard_birth(job, ref, birth):
    z = load(OUT / ("qualify-write-" + ref) / "passed.json")["births"][str(job.seed)]
    check(
        birth
        == dict(
            host=z["host"],
            literal=z["literal"],
            head=z["literal" if job.arms == "B" else "linked"],
        ),
        "run does not start at the qualified seed-specific B origin",
    )


def guard_first(job, ref, row):
    if job.arms == "B":
        return
    d = w.ROOT / w.name("B", job.seed)
    term = load(d / "complete.json")
    check(
        term["binding"] == w.binding(ref, "B", job.seed)
        and sha(d / "steps.jsonl") == term["steps_sha256"],
        "foreign B continuation",
    )
    with (d / "steps.jsonl").open() as f:
        first = json.loads(f.readline())
    check(
        all(
            row[k] == first[k]
            for k in ("targets", "step", "preupdate_nll", "head_lr", "host_lr")
        )
        and all(row["work"][k] == v for k, v in first["work"].items()),
        "unmatched initial prediction/work on B weights",
    )


def short_device_check(job, ctx):
    """Exercise both GDN2 eval kernels on invented bytes before the fresh test is licensed to open."""
    import torch
    from .endpoint import score_window

    d = ctx["folder"]
    ref = ctx["ref"]
    f = d / "frozen.json"
    check(
        job == w.JOBS[w.FINAL]
        and sha(f) == load(d / "start.json")["freeze_sha"]
        and not (d / "source-open-attempted").exists()
        and not (d / "short-device-check.json").exists(),
        "invalid W device probe",
    )
    e = load(f)["endpoints"]["L-4901@1024"]
    path, b = w.resolve("L", 4901, 1024, ref)
    check(
        e["binding"] == b
        and sha(path) == e["sha256"]
        and path.stat().st_size == e["bytes"],
        "probe endpoint not bound",
    )
    m, c = origin(4901, "L")
    ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    check(ck["binding"] == b and ck["step"] == 1024, "foreign W probe weights")
    m.load_state_dict(ck["host"], strict=True)
    c.load_state_dict(ck["head"], strict=True)
    del ck
    m = m.eval().to("cuda")
    c.eval()
    torch.set_num_threads(4)
    lens = [1, 16, 64, 65, 4095]
    times = []
    tic = monotonic()
    synthetic = b"z\0inner:rec\x1e\xffz" * 400
    for n in lens:
        x = score_window(
            m,
            c,
            synthetic[: n + 1],
            arm="L",
            seed=4901,
            document=0,
            stratify=True,
            short_full_dose=True,
        )
        check(
            x["host_tokens"] == n
            and len(x["cells"]) == 5
            and all(cell["targets"] == n for cell in x["cells"]),
            "short-real-device counting mismatch",
        )
        times.append(x["host_prefill_s"])
    write_json(
        d / "short-device-check.json",
        dict(
            scope="invented input only before raw source open",
            checkpoint_sha=e["sha256"],
            targets_per_cell=lens,
            host_prefills=len(lens),
            prefill_s=times,
            wall_s=monotonic() - tic,
            natural_targets=0,
        ),
    )
    del m, c
    torch.cuda.empty_cache()


def inside(job):
    from .panel_train import train_admitted
    from .panel_eval import _documents, evaluate_admitted

    ctx = p.enter_admitted(job, *admit(job))
    progress = partial(p.progress, root=w.ROOT)
    if job.kind == "train":
        out = train_admitted(
            job,
            ctx,
            progress,
            w.binding,
            partial(guard_birth, job, ctx["ref"]),
            partial(guard_first, job, ctx["ref"]),
            stream_factory=partial(WriteStream, w.W_ID),
            origin=origin,
        )
    else:
        short_device_check(job, ctx)
        out = evaluate_admitted(
            job, ctx, _documents, progress, w.resolve, origin=origin
        )
    loop.complete(ctx["folder"], out)


if __name__ == "__main__":
    loop.cli(w, MOD, authority, qualification, inside, 220, targets_for=w.targets_for)
