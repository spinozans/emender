"""X49-W job identities, variable-length progress clocks and frozen no-score endpoint selection."""

from __future__ import annotations
from itertools import accumulate
from pathlib import Path
from ..x48_muon_continuation.data import check, load, sha, write_json
from ..x48_muon_continuation.train import flush_dir
from .panels import OUT, SEEDS, T, fingerprint
from .panel_spec import Job, STEPS, common_wall
from .panel_freeze import read_bound_training
from .write_data import BASE, ROWS, SPEC, selection
from .write_origin import bound_b

ROOT = OUT / "write-panel-1"
FINAL = "w-final"
ARMS = ("B", "P", "L")
PAIRS = (("L", "P"), ("L", "B"), ("P", "B"))
JOBS = {}
for seed in SEEDS:
    for arm in ARMS:
        j = Job(f"w{arm}-{seed}", "train", 1500, arm, seed)
        JOBS[j.name] = j
JOBS[FINAL] = Job(FINAL, "final_test", 5400, "BPL")
TOTAL = 18900
ROW_SHA = "38a09e84121c26af7790c4b300f4e680397c5bfa8b5b6634a309b32400cb0308"
W_ID = "689626870fabaeef1f44b76e009fa9e64e4d33944d6d8f00fed17fab05f7084c"
PREP_SHA = "2114ce75434de159e9edf2d35a45bfac9223f8fcc30ddf7eece05ce2ef06df72"
N = 377
NT = 562712


def prep():
    p = BASE / "prepared.json"
    check(
        sha(p) == PREP_SHA
        and sha(BASE / "train/identity.json") == W_ID
        and sha(ROWS) == ROW_SHA
        and sha(BASE / "reserved.json")
        == "a2d815e3ada6d77473722c10468524737f62389d8e501f50bb9ce49d8014b58b",
        "changed W metadata/stream",
    )
    r = load(p)
    check(
        r["evaluation_windows_opened"] == 0
        and r["short_documents"] == N
        and r["short_targets"] == NT
        and r["train_identity"] == W_ID
        and r["rows_sha"] == ROW_SHA
        and r["spec"] == SPEC,
        "W prep identity mismatch",
    )
    return r


def name(arm, seed):
    check(arm in ARMS and type(seed) is int and seed in SEEDS, "foreign W learner")
    return f"w{arm}-{seed}"


def binding(ref, arm, seed):
    name(arm, seed)
    return dict(
        schema="x49-read-at-writer-v1",
        ref=ref,
        arm=arm,
        seed=seed,
        origin=bound_b(seed),
        train_identity=W_ID,
        spec=SPEC,
        horizon=T,
    )


def resolve(arm, seed, i, ref):
    k = name(arm, seed)
    check(type(i) is int and 32 <= i <= STEPS and i % 32 == 0, "wrong W milestone")
    return ROOT / k / f"weights-{i:04d}.pt", binding(ref, arm, seed)


def expected(ref):
    return dict(
        scope="x49-read-at-writer-local-v1",
        qualified_ref=ref,
        spec=SPEC,
        rows_sha=ROW_SHA,
        train_identity=W_ID,
        origins={str(s): bound_b(s) for s in SEEDS},
        total_card_seconds=TOTAL,
        job_caps={k: j.cap for k, j in JOBS.items()},
        max_acquisitions=len(JOBS),
        retries=0,
        gpus=1,
        remote=False,
        final_documents=N,
        final_targets_per_cell=NT,
    )


def validate(a, ref):
    e = expected(ref)
    check(
        type(a) is dict
        and set(a) == set(e) | {"operator_note"}
        and isinstance(a["operator_note"], str)
        and a["operator_note"].strip()
        and fingerprint({k: a[k] for k in e}) == fingerprint(e),
        "no matching W campaign authority",
    )


def histories(ref):
    runs = {
        k: read_bound_training(ROOT / k, binding(ref, j.arms, j.seed), j.cap)
        for k, j in JOBS.items()
        if j.kind == "train"
    }
    for s in SEEDS:
        b, p, linked = [runs[name(a, s)]["birth"] for a in ARMS]
        check(
            b["host"] == p["host"] == linked["host"]
            and b["literal"] == b["head"] == p["literal"] == linked["literal"]
            and p["head"] == linked["head"],
            "unequal paired W births",
        )
    return runs


def clocks(runs):
    return common_wall(
        runs, pairs=PAIRS, name=name, shared_s=prep()["total_prep_wall_s"] / 9
    )


def count_cycle(counts):
    check(
        counts and all(type(x) is int and 1 <= x < T for x in counts),
        "not real short-document target counts",
    )
    cs = (0, *accumulate(counts))

    def fn(n):
        check(type(n) is int and n >= 0, "invalid seen-window count")
        q, i = divmod(n, len(counts))
        return q * cs[-1] + cs[i]

    return fn


def targets_for(job, d, start):
    if job.kind == "train":
        return lambda n: n * T
    f = d / "frozen.json"
    check(
        job == JOBS[FINAL] and sha(f) == start["freeze_sha"] and sha(ROWS) == ROW_SHA,
        "unbound witness clock",
    )
    p = load(f)
    ns = [r["length"] - 1 for r in load(ROWS)["panels"]["final_test"]]
    check(
        p["rows_sha"] == ROW_SHA
        and p["lengths_sha"] == fingerprint(ns)
        and p["documents"] == len(ns) == N
        and sum(ns) == p["targets_per_endpoint"] == NT
        and start["units"] == p["units"] == N * len(p["endpoints"]),
        "wrong variable progress clock",
    )
    return count_cycle(ns)


def freeze(job, ref):
    check(job == JOBS[FINAL], "only W short test")
    f = ROOT / FINAL / "frozen.json"
    check(not f.exists(), "W endpoints already frozen")
    runs = histories(ref)
    wall = clocks(runs)
    ends = {}

    def add(a, s, i):
        k = name(a, s)
        key = f"{a}-{s}@{i}"
        if key in ends:
            return
        p, b = resolve(a, s, i, ref)
        m = runs[k]["milestones"][i // 32 - 1]
        h = m["sha256"]
        size = m["bytes"]
        check(
            m["step"] == i
            and load(p.with_suffix(".json"))
            == dict(step=i, binding=b, bytes=size, file=p.name, sha256=h)
            and p.stat().st_size == size
            and sha(p) == h,
            "W endpoint not bound to weight bytes",
        )
        ends[key] = dict(arm=a, seed=s, step=i, bytes=size, sha256=h, binding=b)

    for s in SEEDS:
        for a in ARMS:
            add(a, s, STEPS)
        for pairs in wall.values():
            for a, v in pairs[str(s)]["selected"].items():
                if v:
                    add(a, s, v["step"])
    ns = [r["length"] - 1 for r in load(ROWS)["panels"]["final_test"]]
    check(
        sha(ROWS) == ROW_SHA
        and len(ns) == N
        and sum(ns) == NT
        and 9 <= len(ends) <= 18,
        "incorrect W budget or rows",
    )
    write_json(
        f,
        dict(
            job=FINAL,
            split="final_test",
            spec=SPEC,
            ref=ref,
            visit=0,
            horizons=[128, 1024, T],
            documents=N,
            units=N * len(ends),
            targets_per_endpoint=NT,
            rows_sha=ROW_SHA,
            lengths_sha=fingerprint(ns),
            train_identity=W_ID,
            endpoints=ends,
            wall_pairs=wall,
            training_charges={k: x["charge_sha"] for k, x in runs.items()},
        ),
    )
    flush_dir(f.parent)
    return sha(f)


def fresh_descriptor(job, p):
    from .write_run import admit
    from .panel import licensed_admitted_split

    check(
        job == JOBS[FINAL]
        and licensed_admitted_split(job, *admit(job)) == p
        and p["rows_sha"] == sha(ROWS) == ROW_SHA
        and p["job"] == FINAL
        and p["split"] == "final_test"
        and p["spec"] == SPEC
        and p["train_identity"] == W_ID,
        "unlicensed W short documents",
    )
    m = selection()
    rows = m["panels"]["final_test"]
    ns = [r["length"] - 1 for r in rows]
    d = ROOT / FINAL
    check(
        (d / "short-device-check.json").is_file()
        and m == load(ROWS)
        and p["documents"] == len(ns) == N
        and sum(ns) == p["targets_per_endpoint"] == NT
        and p["lengths_sha"] == fingerprint(ns),
        "changed short selection or no device fixture",
    )
    q = load(d / "short-device-check.json")
    check(
        q["checkpoint_sha"] == p["endpoints"]["L-4901@1024"]["sha256"]
        and q["targets_per_cell"] == [1, 16, 64, 65, 4095]
        and q["host_prefills"] == 5
        and q["natural_targets"] == 0,
        "unbound W device fixture",
    )
    return d, m["source"], rows


def finish(root):
    from .write_analysis import analyze

    check(Path(root) == ROOT, "foreign W root")
    write_json(root / "summary.json", analyze())
