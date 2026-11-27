"""SAL-2: the 512:1 archive-to-training ratio point.

Reuses the SAL-1 machinery (model, driver, gate, data extraction) verbatim by
rebinding module-level paths before importing behavior. Registered
differences from SAL-1, all in PROTOCOL-SAL2.md:

- fresh roles: BUILD 262,144 archive documents (2^30 rows) scanned from the
  100 GB-indexed region of the Pile — reference territory that no chain
  experiment has ever drawn a role from; FIT 512 + PROBE 128 extend the
  contiguous rank suffix beyond SAL-1's spent set;
- stage arms LOCAL + SKELETON only; the bank machinery is not built;
- own campaign ledger, cap and qualification receipts;
- the big static count table is built off the leased card and loaded by the
  stage only under a verified archive binding.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..language_compression.settings import DATA as PILOT_DATA, MAN, OLD
from ..natural_data import DATA as PREVIOUS
from ..window import ROOT, get, put, need
from . import data as sal_data
from . import gate as sal_gate
from . import run as sal_run
from . import settings as sal_settings

HERE = Path(__file__).resolve().parent
B2 = ROOT / "output/retrieved-span-effects/sal-2"
DATA2 = B2 / "data"
JOB2 = B2 / "device"
CACHE2 = B2 / "data/counts-cache"
SAL1_DATA = ROOT / "output/retrieved-span-effects/sal-1/data"
N2 = {"build": 262144, "fit": 512, "probe": 128}
ARMS2 = ("local", "skeleton")
CAP2 = 4000
ARCHIVE_CAP = 100_000_000_000  # never scan at or beyond the 100 GB index region
ARCHIVE_MIN_LEN = 4097
WINDOW_T = 4096
SAL1_SUPPLY = 4499
SAL2_SUPPLY = 1811  # SAL-1's supply minus SAL-1's own 2,688 spent rows


def sha(p):
    return sal_settings.sha(p)


def rebind():
    """Point the SAL-1 modules at SAL-2's registered campaign constants."""
    sal_settings.BASE = B2
    sal_settings.DATA = DATA2
    sal_settings.JOB = JOB2
    for mod in (sal_data, sal_run, sal_gate):
        mod.DATA = DATA2
        mod.JOB = JOB2
    sal_data.N = N2
    sal_gate.N = N2
    sal_gate.BASE = B2
    sal_gate.CAMPAIGN_CAP = CAP2
    sal_gate.ARMS = ARMS2
    sal_gate.qualified = qualified
    sal_data.qualified = qualified
    return sal_data, sal_run, sal_gate


def qualified():
    """SAL-2's own clean-public-pre-access qualification, mirroring SAL-1's."""
    ref = (B2 / "qualified-ref").read_text().strip()
    q = get(HERE / "receipts/sal2-qualification.json")
    src = HERE.parent.relative_to(ROOT)
    deps = [
        str(HERE.relative_to(ROOT)),
        str(src / "language_compression"),
        str(src / "corpus_windows"),
        str(src / "corpus_kernel"),
        str(src / "window.py"),
        "scripts/gpu_lease.sh",
    ]
    need(
        q["cpu_tests"] > 0
        and q["natural_memberships"] == q["natural_documents"] == 0
        and q["tests_sha256"] == sha(B2 / "sal2-qualification.log")
        and subprocess.check_output(
            ["git", "rev-parse", ref + "^"], cwd=ROOT, text=True
        ).strip()
        == q["registered"]
        and not subprocess.run(
            ["git", "diff", "--quiet", ref, "--", *deps], cwd=ROOT
        ).returncode
        and not subprocess.check_output(
            ["git", "status", "--porcelain", "--", *deps], cwd=ROOT, text=True
        ).strip(),
        "no clean public pre-access qualification",
    )
    return ref


def _chain_base():
    """SAL-1's exclusion base (62,054), derived verbatim from sealed receipts."""
    from ...x49_live_trace_memory import correction_transfer_data as pt, panels
    from ...x53_token_spelling import data as s53

    s52 = s53.p52
    s51 = s52.last
    s50 = s51.parent
    book, records, ex = pt.spent_context(pt.parent()["data"])
    short = pt.u.bound(pt.BASE / "reservation.json", s50.PT_ROWS)
    short_rows, _ = pt.checked_rows(short)
    base = ex | {r["sha256"] for r in short_rows}
    need(len(base) == 45414, "incorrect spent base")
    spent = []
    for mod, h, key in [
        (s50, s51.PARENT_ROWS, "fit"),
        (s51, s52.PREV_ROWS, "fit"),
        (s52, s53.PREV_ROWS, "fit"),
        (
            s53,
            "3aa0469581837dd6e31172a59cd4130685af0b5138be72614e770f496226b130",
            "build",
        ),
    ]:
        r = pt.u.bound(mod.RUN / "reservation.json", h)
        spent += r[key] + r["probe"]
    panel = pt.u.bound(PREVIOUS / "reservation.json", OLD)
    ex2 = base | {r["sha256"] for r in spent}
    need(
        len(ex2) == panel["excluded"] == 50790
        and panels.fingerprint(sorted(ex2)) == panel["excluded_sha"]
        and panel["manifest"] == MAN
        and panel["source"] == book["source"],
        "previous cohort or manifest mismatch",
    )
    for role, n in (("build", 2048), ("fit", 8192), ("probe", 1024)):
        rows = panel["roles"][role]
        need(len(rows) == n, "incomplete spent natural panel")
        spent += rows
    pilot = get(PILOT_DATA / "rows.json")
    need(
        sha(PILOT_DATA / "rows.json")
        == get(PILOT_DATA / "reservation.json")["rows_sha"],
        "unbound pilot rows",
    )
    blocked = ex2 | {r["sha256"] for r in spent}
    pilot_rows = [r for role in pilot["roles"] for r in pilot["roles"][role]]
    need(
        len(blocked) == pilot["excluded"] == 62054
        and panels.fingerprint(sorted(blocked)) == pilot["excluded_sha"]
        and len(pilot_rows) == 512
        and not ({r["sha256"] for r in pilot_rows} & blocked),
        "pilot exclusion chain mismatch",
    )
    return book, records, blocked, pilot, pilot_rows, panels, s51


def reserve():
    """Fresh SAL-2 roles: archive from the indexed region; fit/probe from the
    contiguous rank suffix after SAL-1; both anchored on SAL-1's receipts."""
    rebind()
    ref = qualified()
    sal_settings.cpu()
    if DATA2.exists():
        need(not any(DATA2.iterdir()), "data directory not empty from an earlier run")
    else:
        DATA2.mkdir(mode=0o700)
    book, records, blocked, pilot, pilot_rows, panels, s51 = _chain_base()
    sal1 = get(SAL1_DATA / "rows.json")
    need(
        sha(SAL1_DATA / "rows.json") == get(SAL1_DATA / "reservation.json")["rows_sha"],
        "unbound SAL-1 rows",
    )
    sal1_rows = [r for role in ("build", "fit", "probe") for r in sal1["roles"][role]]
    need(len(sal1_rows) == 2688, "incomplete SAL-1 roles")
    # Reproduce SAL-1's own selection first: same code path, same inputs.
    rows1, blocked1, supply1 = s51.next_block(
        records, blocked, pilot_rows, 2688, book["source"]["size"]
    )
    need(
        json.dumps(rows1, sort_keys=True) == json.dumps(sal1_rows, sort_keys=True)
        and len(blocked1) == sal1["excluded"] == 62566
        and panels.fingerprint(sorted(blocked1)) == sal1["excluded_sha"]
        and supply1 == sal1["supply"] == SAL1_SUPPLY
        and sal1["manifest"] == pilot["manifest"]
        and sal1["source"] == book["source"],
        "SAL-1 selection not reproduced from the sealed chain",
    )
    base_ext = blocked | {r["sha256"] for r in pilot_rows}
    rows, blocked2, supply = s51.next_block(
        records, base_ext, sal1_rows, N2["fit"] + N2["probe"], book["source"]["size"]
    )
    need(
        len(rows) == N2["fit"] + N2["probe"]
        and len(blocked2) == 65254
        and supply == SAL2_SUPPLY,
        "fresh suffix mismatch",
    )
    need(
        not ({r["sha256"] for r in rows} & {r["sha256"] for r in sal1_rows}),
        "fresh rows overlap SAL-1 roles",
    )
    arch, scan = archive_rows(book["source"])
    need(
        len(arch) == N2["build"]
        and scan["region_end"] <= ARCHIVE_CAP
        and not ({r["sha256"] for r in arch} & blocked2),
        "archive scan mismatch",
    )
    put(
        DATA2 / "rows.json",
        dict(
            qualified=ref,
            manifest=pilot["manifest"],
            sal1=sha(SAL1_DATA / "rows.json"),
            source=book["source"],
            excluded=len(blocked2),
            excluded_sha=panels.fingerprint(sorted(blocked2)),
            supply=supply,
            archive=scan,
            roles=dict(
                build=arch,
                fit=rows[: N2["fit"]],
                probe=rows[N2["fit"] :],
            ),
        ),
    )
    put(
        DATA2 / "reservation.json",
        dict(
            rows_sha=sha(DATA2 / "rows.json"),
            qualified=ref,
            sizes=N2,
            payloads=0,
        ),
    )
    print("SAL-2 roles reserved; archive scan ended at byte", scan["region_end"])


def archive_rows(source, count=None, cap=None, min_len=None):
    """First `count` distinct TRAIN documents (>= min_len) from the start of
    the indexed region, in source order, with the same hash-window rule the
    chain's select() uses. Returns (rows, scan-receipt)."""
    count = N2["build"] if count is None else count
    cap = ARCHIVE_CAP if cap is None else cap
    min_len = ARCHIVE_MIN_LEN if min_len is None else min_len
    rows, seen = [], set()
    pos = 0
    pending = b""
    with open(source["path"], "rb", buffering=0) as f:
        st = os.fstat(f.fileno())
        need(
            (st.st_ino, st.st_size, st.st_mtime_ns)
            == (source["inode"], source["size"], source["mtime_ns"]),
            "source changed",
        )
        while len(rows) < count:
            data = f.read(64 << 20)
            if not data:
                break
            data = pending + data
            parts = data.split(b"\x1e")
            pending = parts.pop()
            for raw in parts:
                n = len(raw)
                if n > 1:
                    h = hashlib.sha256(raw).hexdigest()
                    if int(h, 16) % 100 >= 2 and n >= min_len and h not in seen:
                        seen.add(h)
                        j = int(h[16:32], 16) % ((n - 1) // WINDOW_T)
                        rows.append(
                            dict(
                                doc=pos,
                                offset=pos,
                                length=n,
                                split="train",
                                sha256=h,
                                start=j * WINDOW_T,
                            )
                        )
                pos += n + 1
                if len(rows) == count:
                    break
            need(pos <= cap, "archive scan cap overrun")
    need(len(rows) == count, f"insufficient archive documents: {len(rows)}")
    return rows, dict(
        region="pile indexed region from byte 0, document order",
        region_end=pos,
        pending_bytes=len(pending),
        first_doc=rows[0]["doc"],
        last_doc=rows[-1]["doc"],
        last_length=rows[-1]["length"],
        distinct=len(rows),
        min_length=min_len,
        split_rule="sha256 % 100 in {0: calibration, 1: final_test, >=2: train}",
        window_rule="start = int(sha[16:32],16) % ((length-1)//4096) * 4096",
    )


def prepare():
    rebind()
    reserve()
    sal_data.extract("build")
    sal_data.extract("fit")
    put(
        DATA2 / "ready.json",
        dict(
            rows=sha(DATA2 / "rows.json"),
            build=sha(DATA2 / "build.json"),
            fit=sha(DATA2 / "fit.json"),
        ),
    )
    print("SAL-2 BUILD and FIT sealed; PROBE locked.")


def build_level(d, cache):
    """One worker: build depth-d arrays from the sealed archive, off the card."""
    rebind()
    sal_settings.cpu()
    qualified()
    import numpy as np

    from .ladder import _level_arrays

    meta = get(DATA2 / "build.json")
    a = np.load(DATA2 / "build.npy", mmap_mode="r", allow_pickle=False)
    need(
        a.dtype == np.uint8
        and list(a.shape) == meta["shape"]
        and sha(DATA2 / "build.npy") == meta["array_sha"],
        "unbound archive array",
    )
    arrays = _level_arrays(a, d, rowids=False)
    path = Path(cache) / f"level-{d}.npz"
    with path.open("xb") as f:
        np.savez(
            f,
            codes=arrays[0],
            counts=arrays[1],
            starts=arrays[2],
            rows=np.empty(0, np.int64),
            keys=arrays[4],
            key_counts=arrays[5],
            key_starts=arrays[6],
        )
    put(
        Path(cache) / f"level-{d}.json",
        dict(
            depth=d,
            codes=len(arrays[0]),
            level_sha=sha(path),
            archive_sha=meta["array_sha"],
        ),
    )


def index(concurrency=4):
    """Build the static count cache in parallel CPU workers, off the card."""
    rebind()
    sal_settings.cpu()
    qualified()
    import numpy as np

    from .ladder import MAX_DEPTH

    meta = get(DATA2 / "build.json")
    cache = CACHE2
    if cache.exists():
        need(not any(cache.iterdir()), "cache directory not empty")
    else:
        cache.mkdir(mode=0o700, parents=True)
    py = ROOT / ".venv/bin/python"
    mod = __package__ + ".sal2"
    running = []
    pending = list(range(1, MAX_DEPTH + 1))
    tic = time.monotonic()
    while pending or running:
        while pending and len(running) < concurrency:
            d = pending.pop(0)
            running.append(
                (
                    d,
                    subprocess.Popen(
                        [str(py), "-m", mod, "build-level", str(d), str(cache)],
                        cwd=ROOT,
                        start_new_session=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    ),
                )
            )
        d, p = running.pop(0)
        out, _ = p.communicate()
        need(p.returncode == 0, f"level-{d} worker failed: {out.decode()[-2000:]}")
        need(
            get(cache / f"level-{d}.json")["archive_sha"] == meta["array_sha"],
            f"level-{d} cache binding mismatch",
        )
        print(f"level-{d}: {get(cache / f'level-{d}.json')['codes']} codes", flush=True)
    a = np.load(DATA2 / "build.npy", mmap_mode="r", allow_pickle=False)
    root = np.bincount(np.asarray(a[:, 1:]).ravel(), minlength=256).astype(np.int64)
    put(
        cache / "meta.json",
        dict(
            rows=int(a.shape[0] * (a.shape[1] - 1)),
            depth=MAX_DEPTH,
            root=root.tolist(),
            levels=MAX_DEPTH,
            archive_sha=meta["array_sha"],
        ),
    )
    print("cache built in", round(time.monotonic() - tic, 1), "s")


def _gate_cli(argv):
    """The gate's own command-line interface, run on rebound paths."""
    action, args = argv[0], argv[1:]
    os.umask(0o077)
    if action == "spawn":
        sal_gate.spawn(args[0], int(args[1]), args[2], args[3:] or None)
    elif action == "prelease":
        cap, deadline = sal_gate.prelease(args[0], args[1], args[2])
        print(cap, deadline)
    elif action == "check":
        print(sal_gate.admission(args[0], int(args[1])))
    elif action == "watch":
        try:
            sal_gate.observe()
        except BaseException as exc:
            stage = get(JOB2 / "admission.json")["stage"]
            (JOB2 / f"watch-failure-{stage}").write_text(repr(exc))
            raise
    elif action == "monitor":
        sal_gate.monitor(args[0])
    else:
        raise ValueError("unknown action")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which == "build-level":
        build_level(int(sys.argv[2]), sys.argv[3])
    elif which == "prepare":
        prepare()
    elif which == "index":
        index()
    elif which == "probe":
        rebind()
        sal_data.extract("probe")
    elif which in ("spawn", "prelease", "check", "watch", "monitor"):
        rebind()
        _gate_cli(sys.argv[1:])
    else:
        raise ValueError("unknown action")


if __name__ == "__main__":
    main()
