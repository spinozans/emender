"""SAL-3: the count law over the real 100 GB XSA index.

The SAL-1/SAL-2 machinery (host, driver, gate, extraction) is reused with
rebound campaign constants; the archive is external — the SXI2 v10
container over the first 100 GB of the Pile, served by `xsa consult` —
so there is no BUILD role and every count table is bound off-card from
the transport and loaded under sealed receipts. Registered in
PROTOCOL-SAL3.md before any new natural membership access.
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..window import ROOT, get, put, need
from . import data as sal_data
from . import gate as sal_gate
from . import run as sal_run
from . import settings as sal_settings

HERE = Path(__file__).resolve().parent
B3 = ROOT / "output/retrieved-span-effects/sal-3"
DATA3 = B3 / "data"
JOB3 = B3 / "device"
N3 = {"fit": 512, "probe": 128}
ARMS3 = ("local", "xsacount")
CAP3 = 6000
SAL1_DATA = ROOT / "output/retrieved-span-effects/sal-1/data"
SAL2_DATA = ROOT / "output/retrieved-span-effects/sal-2/data"
CONTAINER = Path("/mnt/nvme2n1/erikg/PILE100-REHEARSAL/pile-head100.sxi2")
CONTAINER_SHA = "15419c427b5bdc9fb63548901ca6cf83d54f486da890f66faa55021f0a0dc053"
# The serving binary is a pinned private copy: the sxgc tree is live (its
# build drifted under the campaign once already), so the qualification and
# every stage run against this frozen artifact, sha-verified at each use.
XSA = ROOT / "output/retrieved-span-effects/sal-3/tools/xsa"
XSA_SHA = "8ff46bbf5fd4c1ae416940f1d6f1f613ff69ea364823cb19f22ae545da8d25fc"
XSA_SHA_FULL = None  # filled by verify_tools()
CORPUS = Path("/mnt/nvme2n1/erikg/pile.txt")
# The container's exact text, materialized by the rehearsal build and sealed
# in its journal (corpus_sha256); the X3 gate's own oracle reads the corpus
# via the file's full stat size, so it must see exactly the 100 GB text.
CORPUS100 = Path("/mnt/nvme2n1/erikg/pile-head100.txt")
CORPUS100_SHA = "de79fe96057b604a71513770111f246ab0e936d3cf56415bcc5a43bba60bbbfd"
SEPS = Path("/mnt/nvme2n1/erikg/pilemap100/seps.bin")
# Eight transport passes, d = 1..8. The deepest class is the >=8 class,
# exactly as SAL-1/2's bind_stats (its telescope subtracts only [:-1]).
DEPTHS = tuple(range(1, 9))
LAW_DEPTH = 8
WINDOW_T = 4096
SAL2_SUPPLY = 1811
# corrected from the earlier session's estimate, which double-counted the
# SAL-2 fresh set (640) in the blocked total and its supply; the chain
# arithmetic reproduces SAL-1 and SAL-2 bit-exactly through this code path
SAL3_SUPPLY = 1166
ZSTD_DICT_FALLBACK = 4 << 30


def sha(p):
    return sal_settings.sha(p)


def rebind():
    sal_settings.BASE = B3
    sal_settings.DATA = DATA3
    sal_settings.JOB = JOB3
    for mod in (sal_data, sal_run, sal_gate):
        mod.DATA = DATA3
        mod.JOB = JOB3
    sal_data.N = N3
    sal_gate.N = N3
    sal_gate.BASE = B3
    sal_gate.CAMPAIGN_CAP = CAP3
    sal_gate.ARMS = ARMS3
    sal_gate.READY_KEYS = ("rows", "fit", "container")
    sal_gate.qualified = qualified
    sal_data.qualified = qualified
    sal_run.N = N3
    sal_run.PROBE_ENTRY = __package__ + ".sal3"
    sal_run.BIND_ENTRY = (__package__ + ".sal3", "bind-probe")
    sal_run.DICT_BYTES = dict_bytes
    return sal_data, sal_run, sal_gate


def dict_bytes():
    """The zstd shared-dictionary baseline: the archive's raw bytes.

    Attempted at the full 100 GB; on any failure the registered fallback
    is the first 4 GiB, labeled in the receipt. Never affects the arms.
    """

    def provide():
        try:
            import zstandard as zstd

            with open(CORPUS, "rb") as f:
                raw = f.read(100_000_000_000)
            zstd.ZstdCompressionDict(
                raw, dict_type=zstd.DICT_TYPE_RAWCONTENT
            ).as_bytes()
            return raw
        except BaseException:
            del raw
            with open(CORPUS, "rb") as f:
                return f.read(ZSTD_DICT_FALLBACK)

    return provide


def qualified():
    ref = (B3 / "qualified-ref").read_text().strip()
    q = get(HERE / "receipts/sal3-qualification.json")
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
        and q["tests_sha256"] == sha(B3 / "sal3-qualification.log")
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


def verify_tools():
    """Fresh integrity checks of the container and the serving binary."""
    sal_settings.cpu()
    c = sha(CONTAINER)
    need(c == CONTAINER_SHA, "container sha mismatch")
    b = sha(XSA)
    global XSA_SHA_FULL
    XSA_SHA_FULL = b
    need(b.startswith(XSA_SHA), "binary sha mismatch")
    return c, b


def doc_starts():
    """Document start offsets from the pile map (the full 1.3 TB sidecar)."""
    data = open(SEPS, "rb").read(48)
    docs = struct.unpack_from("<Q", data, 16)[0]
    with open(SEPS, "rb") as f:
        f.seek(48 + 4)  # the deltas are u32 at 48+4*i for i = 1..docs
        deltas = np.frombuffer(f.read(4 * (docs - 1)), dtype="<u4")
    starts = np.empty(docs + 1, np.int64)
    starts[0] = 0
    np.cumsum(deltas, out=starts[1:docs])
    starts[docs] = 0  # the final sentinel is unused by callers
    return docs, starts


def _chain_base():
    """SAL-1's exclusion base (62,054), derived verbatim from sealed receipts."""
    from ..language_compression.settings import DATA as PILOT_DATA, MAN, OLD
    from ..natural_data import DATA as PREVIOUS
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
    """Fresh fit/probe roles extending the chain past SAL-1 and SAL-2."""
    rebind()
    ref = qualified()
    sal_settings.cpu()
    if DATA3.exists():
        need(not any(DATA3.iterdir()), "data directory not empty from an earlier run")
    else:
        DATA3.mkdir(mode=0o700)
    book, records, blocked, pilot, pilot_rows, panels, s51 = _chain_base()
    sal1 = get(SAL1_DATA / "rows.json")
    need(
        sha(SAL1_DATA / "rows.json") == get(SAL1_DATA / "reservation.json")["rows_sha"],
        "unbound SAL-1 rows",
    )
    sal1_rows = [r for role in ("build", "fit", "probe") for r in sal1["roles"][role]]
    rows1, blocked1, supply1 = s51.next_block(
        records, blocked, pilot_rows, 2688, book["source"]["size"]
    )
    need(
        json.dumps(rows1, sort_keys=True) == json.dumps(sal1_rows, sort_keys=True)
        and len(blocked1) == sal1["excluded"] == 62566
        and supply1 == sal1["supply"] == 4499,
        "SAL-1 selection not reproduced",
    )
    sal2 = get(SAL2_DATA / "rows.json")
    need(
        sha(SAL2_DATA / "rows.json") == get(SAL2_DATA / "reservation.json")["rows_sha"],
        "unbound SAL-2 rows",
    )
    sal2_fresh = [r for role in ("fit", "probe") for r in sal2["roles"][role]]
    sal2_arch = sal2["roles"]["build"]
    need(len(sal2_fresh) == 640 and len(sal2_arch) == 262144, "SAL-2 roles")
    base_ext = blocked | {r["sha256"] for r in pilot_rows}
    rows2, blocked2, supply2 = s51.next_block(
        records, base_ext, sal1_rows, 640, book["source"]["size"]
    )
    need(
        json.dumps(rows2, sort_keys=True) == json.dumps(sal2_fresh, sort_keys=True)
        and len(blocked2) == sal2["excluded"] == 65254
        and supply2 == sal2["supply"] == SAL2_SUPPLY,
        "SAL-2 selection not reproduced",
    )
    base3 = blocked2 | {r["sha256"] for r in sal2_arch}
    # AMENDMENT (see PROTOCOL-SAL3.md): the container is the corpus head;
    # the chain's fresh supply lies beyond it and the qualified consult
    # transport serves only container windows (fail-closed doc structure).
    # SAL-3's roles are drawn inside the container from documents excluded
    # from every ledger role, and every role pass carries --exclude-source
    # (the X3-verified position-self exclusion). The coarse MIN_OFFSET
    # prior-train guard is replaced by the container constraint.
    ex3 = base3 | {r["sha256"] for r in sal2_fresh}
    need(len(ex3) == 328038, "exclusion ledger mismatch")
    panels.MIN_OFFSET = 0
    records3 = [
        r for r in records if r["byte_offset"] + r["byte_length"] <= 100_000_000_000
    ]
    need(len(records3) > 0, "no in-container records")
    quotas = {"train": N3["fit"] + N3["probe"]}
    chosen, supply3 = panels.select(records3, ex3, quotas, 100_000_000_000)
    panels.verify_selection(records3, ex3, chosen, quotas, supply3)
    rows = chosen["train"]
    need(len(rows) == N3["fit"] + N3["probe"], "in-container selection mismatch")
    for row in rows:
        need(
            row["offset"] + row["length"] <= 100_000_000_000,
            "selected document outside the container",
        )
    blocked3 = ex3
    supply = supply3["train"]
    put(
        DATA3 / "rows.json",
        dict(
            qualified=ref,
            manifest=pilot["manifest"],
            sal1=sha(SAL1_DATA / "rows.json"),
            sal2=sha(SAL2_DATA / "rows.json"),
            source=book["source"],
            excluded=len(blocked3),
            excluded_sha=panels.fingerprint(sorted(blocked3)),
            supply=supply,
            roles_amendment="in-container-exclude-source",
            roles=dict(
                fit=rows[: N3["fit"]],
                probe=rows[N3["fit"] :],
            ),
        ),
    )
    put(
        DATA3 / "reservation.json",
        dict(
            rows_sha=sha(DATA3 / "rows.json"),
            qualified=ref,
            sizes=N3,
            payloads=0,
        ),
    )
    print("SAL-3 roles reserved; supply remaining:", supply)


# ---------------------------------------------------------------------------
# consult transport


def windows_file(role, d, path):
    """One whole-row window per role document, at spine budget d.

    `doc` is the container's document index, resolved from the seps
    structure (the transport validates doc/base against it, fail-closed).
    """
    r = get(DATA3 / "rows.json")
    need(r["roles"][role] and len(r["roles"][role]) == N3[role], "role rows")
    import bisect

    _, starts = doc_starts()
    with open(path, "w") as f:
        for row in r["roles"][role]:
            off, n = row["offset"], row["length"]
            i = bisect.bisect_left(starts, off)
            need(
                i < len(starts)
                and int(starts[i]) == off
                and off + n <= 100_000_000_000
                and row["start"] + 1 + WINDOW_T <= n,
                f"role document {off} not a container document",
            )
            f.write(
                json.dumps(
                    dict(
                        doc=int(i),
                        base=off,
                        start=row["start"] + 1,
                        len=WINDOW_T,  # the row's 4096 target positions
                        left=d,
                    )
                )
                + "\n"
            )
    return sha(path)


def consult_pass(role, d, tag, windows, out, jobs=96, exclude=True):
    """Run one transport pass; verify the journal and per-row exactness.

    Role passes (fit/probe binds) carry --exclude-source: the X3-verified
    position-self exclusion, required by the in-container roles amendment.
    Qualification passes run without it, matching their no-cross oracles.
    """
    out = str(out)
    windows = str(windows)
    need(not Path(out).exists(), "consult output already exists")
    args = [
        str(XSA),
        "consult",
        "--container",
        str(CONTAINER),
        "--corpus",
        str(CORPUS),
        "--windows",
        str(windows),
        "--output",
        str(out),
        "--rss-gb",
        "8",
        "-j",
        str(jobs),
    ]
    if exclude:
        args.append("--exclude-source")
    p = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    need(p.returncode == 0, f"consult {tag} d={d} failed: {p.stdout[-800:]}")
    journal = Path(out + ".journal")
    jtext = journal.read_text()
    open_line = [ln for ln in jtext.splitlines() if ln.startswith("CONSULT_OPEN")]
    need(len(open_line) == 1, "missing CONSULT_OPEN")
    need(
        ("exclude_source=1" in open_line[0]) == exclude,
        "position-self exclusion state mismatch in the journal",
    )
    done = [ln for ln in jtext.splitlines() if "CONSULT_DONE" in ln]
    need(len(done) == 1, "missing CONSULT_DONE")
    facts = dict(kv.split("=", 1) for kv in done[0].split()[1:] if "=" in kv)
    return facts


def read_rows(path):
    """The v3 row stream, exactly as the X3 gate parses it."""
    f = open(path, "rb").read()
    need(f[:8] == b"XSACONS1", "bad row magic")
    ver = struct.unpack_from("<I", f, 8)[0]
    need(ver == 3, f"row file version {ver}")
    rows = []
    p = 16
    n = len(f)
    while p < n:
        nsym, ctxlen, flags = struct.unpack_from("<IHH", f, p)
        doc, off, width, eof = struct.unpack_from("<QQQQ", f, p + 8)
        q = p + 40
        hist = np.zeros(256, np.int64)
        for _ in range(nsym):
            hist[f[q]] = struct.unpack_from("<Q", f, q + 1)[0]
            q += 9
        docid, base, depth = struct.unpack_from("<QQQ", f, q)
        rows.append(
            dict(
                flags=flags,
                ctxlen=ctxlen,
                off=off,
                width=width,
                eof=eof,
                hist=hist,
                docid=docid,
                base=base,
                depth=depth,
            )
        )
        p = q + 24
    need(p == n, "torn row file")
    return rows


def pass_arrays(path, d, expect):
    """Per-position (c_d, hist_d) from a consult row stream.

    The law's truncation convention: only rows whose spine is the full d
    bytes contribute at depth d; shorter spines (positions closer than d to
    the document start) have an empty depth-d class. Every used row must be
    boundary-exact (flag bit 4 clear); a conflated answer fails closed.
    """
    buf = np.frombuffer(open(path, "rb").read(), np.uint8)
    need(buf[:8].tobytes() == b"XSACONS1", "bad row magic")
    need(struct.unpack_from("<I", buf, 8)[0] == 3, "row version")
    c = np.zeros(expect, np.int64)
    h = np.zeros((expect, 256), np.int64)
    p = 16
    n = len(buf)
    for i in range(expect):
        need(p + 40 <= n, "torn row file")
        nsym = (
            int(buf[p + 3]) << 24
            | int(buf[p + 2]) << 16
            | int(buf[p + 1]) << 8
            | int(buf[p])
        )
        ctxlen = int(buf[p + 4]) | int(buf[p + 5]) << 8
        flags = int(buf[p + 6]) | int(buf[p + 7]) << 8
        need(not (flags & 16), f"boundary-inexact row at depth {d}")
        need(ctxlen <= d, "ctxlen overrun")
        q = p + 40
        if ctxlen == d:
            width, eof = struct.unpack_from("<QQ", buf, p + 24)
            if nsym:
                pairs = buf[q : q + 9 * nsym].reshape(nsym, 9)
                hist = np.zeros(256, np.int64)
                counts = np.frombuffer(pairs[:, 1:].tobytes(), "<u8").astype(np.int64)
                np.add.at(hist, pairs[:, 0].astype(np.int64), counts)
                h[i] = hist
            c[i] = width - eof
        p = q + 9 * nsym + 24
    need(p == n, "trailing row bytes")
    return c, h


def depth0_pass(out, jobs=96, present=None, head_mb=256):
    """The archive's successor marginal via the single-byte contexts.

    Rare byte values can be absent from the head scan; they are located by
    a deep chunked scan of the container's text. `present` is the
    authoritative set of byte values that occur as contexts (from the
    qualification's full depth-1 pair table): every one of them must be
    located. Without `present`, the scan runs to the end of the text and
    byte values never located are recorded as assumed-zero.
    """
    docs, starts = doc_starts()
    import bisect

    text_len = 100_000_000_000
    with open(CORPUS, "rb", buffering=0) as f:
        head = f.read(head_mb << 20)
    need(len(head) > WINDOW_T, "corpus head")

    def accept(p, b, found):
        if b in found:
            return
        i = bisect.bisect_right(starts, p) - 1
        # a document-start position has an empty spine under the
        # transport's cut law; only in-document positions may host a
        # single-byte context
        if p > starts[i]:
            found[b] = (i, p - starts[i])

    found = {}
    head_arr = np.frombuffer(head, np.uint8)
    for b in range(256):
        idx = np.flatnonzero(head_arr[:-1] == b)
        if len(idx):
            for p in idx[::-1]:
                accept(int(p) + 1, b, found)
                if b in found:
                    break
    missing = [
        b for b in range(256) if b not in found and (present is None or b in present)
    ]
    if missing:
        mm = np.memmap(CORPUS, dtype=np.uint8, mode="r")
        step = 1 << 30
        want = np.asarray(missing, np.uint8)
        for lo in range(len(head) - 1, text_len, step):
            hi = min(lo + step + 1, text_len)
            chunk = mm[lo:hi]
            hits = np.flatnonzero(np.isin(chunk[:-1], want))
            for j in hits[::-1]:
                accept(lo + int(j) + 1, int(chunk[j]), found)
                if all(b in found for b in missing):
                    break
            missing = [b for b in missing if b not in found]
            want = np.asarray(missing, np.uint8)
            if not missing:
                break
        del mm
    if present is not None:
        need(
            all(b in found for b in present),
            f"authoritative byte values not located: "
            f"{sorted(b for b in present if b not in found)}",
        )
    assumed_zero = sorted(b for b in range(256) if b not in found)
    path = out + ".windows.jsonl"
    with open(path, "w") as f:
        for b, (i, s) in sorted(found.items()):
            f.write(json.dumps(dict(doc=int(i), start=int(s), len=1, left=1)) + "\n")
    facts = consult_pass("depth0", 1, "depth0", path, out, jobs=jobs, exclude=False)
    rows = read_rows(out)
    need(len(rows) == len(found), "depth0 rows")
    h0 = np.zeros(256, np.int64)
    for r in rows:
        need(not (r["flags"] & 16) and r["ctxlen"] == 1, "depth0 row flags")
        h0 += r["hist"]
    n = int(h0.sum())
    return dict(
        rows=int(n),
        h0=h0.tolist(),
        facts=facts,
        windows_sha=sha(path),
        contexts=len(found),
        assumed_zero=assumed_zero,
    )


def telescope(classes, hists, n_rows, h0, expect):
    """Exact-depth classes (m, h) from the eight >=-depth consult passes.

    Depths 1..7 telescope c_d - c_{d+1}; the deepest class is the >=8 class
    (bind_stats' convention). Depth 0 telescopes the global constants
    against the >=1 class.
    """
    m = np.zeros((expect, LAW_DEPTH + 1), np.int64)
    h = np.zeros((expect, LAW_DEPTH + 1, 256), np.int64)
    # Depth 0 telescopes like every other depth: the exact-0 class is the
    # full row count minus the >=1 class (bind_stats' m_ex[:, :, 0]).
    m[:, 0] = n_rows - classes[1]
    h[:, 0] = h0[None, :] - hists[1]
    for d in range(1, LAW_DEPTH):
        cd, hd = classes[d], hists[d]
        cn, hn = classes[d + 1], hists[d + 1]
        m[:, d] = cd - cn
        h[:, d] = hd - hn
    m[:, LAW_DEPTH] = classes[LAW_DEPTH]
    h[:, LAW_DEPTH] = hists[LAW_DEPTH]
    need(
        bool((m >= 0).all()) and bool((h.sum(-1) == m).all()),
        "negative exact-depth classes",
    )
    return m, h


def build_batches(role, docs, m, h):
    """The driver's exact window order: 16-window batches over 64-target
    windows; fit permuted with the registered seed, probe in source order."""
    need(
        docs.ndim == 2
        and docs.dtype == np.uint8
        and docs.shape == (N3[role], WINDOW_T + 1),
        "role array geometry",
    )
    wins = np.lib.stride_tricks.sliding_window_view(docs, 65, axis=1)[
        :, ::64, :
    ].reshape(-1, 65)
    order = (
        np.arange(len(wins))
        if role == "probe"
        else np.random.default_rng(5750).permutation(len(wins))
    )
    need(len(wins) % 16 == 0, "window count")
    zs = {}
    local_rows = (np.arange(16)[:, None] * 64 + np.arange(63)).ravel()
    for bi, lo in enumerate(range(0, len(order), 16)):
        sel = order[lo : lo + 16]
        ids = wins[sel, :-1].astype(np.int64).ravel()
        targets = wins[sel, 1:].astype(np.int64).ravel()
        # window-list index -> the role's whole-row target index
        base = (sel // 64) * WINDOW_T + (sel % 64) * 64
        idx = np.repeat(base, 64) + np.tile(np.arange(64), 16)
        zs[f"b{bi}_ids"] = ids
        zs[f"b{bi}_targets"] = targets
        zs[f"b{bi}_local_succ"] = ids[local_rows + 1]
        zs[f"b{bi}_m"] = m[idx].astype(np.float32)
        zs[f"b{bi}_h"] = h[idx]
    return zs


def encode_pass(path, d, counts, hists, dist):
    """Write one synthetic v3 pass row stream (tests and oracles only).

    One row per position in order, exactly like the transport: the spine is
    min(d, dist[j]) bytes; a truncated spine reports its own (shorter)
    class, which the parser then drops at depth d per the law. Pairs are
    emitted count-desc, byte-asc like the real tool.
    """
    n = len(counts)
    with open(path, "wb") as f:
        f.write(b"XSACONS1" + struct.pack("<II", 3, 0))
        for j in range(n):
            L = int(min(d, dist[j]))
            hist = hists[j, L]
            nz = np.nonzero(hist)[0]
            order_idx = sorted(map(int, nz), key=lambda b: (-hist[b], b))
            f.write(struct.pack("<IHH", len(order_idx), L, 0))
            f.write(struct.pack("<QQQQ", j, j, int(counts[j, L]), 0))
            for b in order_idx:
                f.write(struct.pack("<BQ", b, int(hist[b])))
            f.write(struct.pack("<QQQ", j, 0, int(dist[j])))
        f.flush()


def bind_role(role, jobs=96):
    """Nine transport passes, telescoped exact-depth classes, batch cache."""
    rebind()
    sal_settings.cpu()
    qualified()
    if role == "probe":
        need(
            (DATA3 / "probe-opened").exists()
            and get(JOB3 / "freeze-ack.json")["frozen_sha"]
            == sha(JOB3 / "frozen.json"),
            "probe binding before witnessed freeze",
        )
    verify_tools()
    expect = N3[role] * WINDOW_T
    work = DATA3 / f"passes-{role}"
    work.mkdir(mode=0o700, exist_ok=True)
    passes = []
    classes = {}
    hists = {}
    for d in DEPTHS:
        w = work / f"w{d}.jsonl"
        wsha = windows_file(role, d, w)
        out = work / f"rows{d}.bin"
        need(not out.exists(), f"pass {d} already exists")
        facts = consult_pass(role, d, f"{role}", w, out, jobs=jobs)
        need(
            int(facts["positions"]) == expect,
            f"pass {d} positions {facts['positions']} != {expect}",
        )
        c, h = pass_arrays(out, d, expect)
        classes[d], hists[d] = c, h
        passes.append(dict(depth=d, rows_sha256=facts["rows_sha256"], windows_sha=wsha))
        print(
            f"{role} depth {d}: {facts['positions_per_s']} positions/s",
            flush=True,
        )
    # depth 0 constants
    c0 = get(DATA3 / "container.json")
    h0 = np.asarray(c0["h0"], np.int64)
    n_rows = int(c0["rows"])
    m, h = telescope(classes, hists, n_rows, h0, expect)
    docs = np.load(DATA3 / f"{role}.npy", allow_pickle=False)
    zs = build_batches(role, docs, m, h)
    zpath = DATA3 / f"bound-{role}.npz"
    need(not zpath.exists(), "bound batches already exist")
    np.savez(zpath, **zs)
    put(
        DATA3 / f"bound-{role}.json",
        dict(
            batches=len(zs),
            batches_sha=sha(zpath),
            rows_sha=sha(DATA3 / "rows.json"),
            targets=expect,
            passes=passes,
            container_sha256=CONTAINER_SHA,
            binary_sha256=XSA_SHA_FULL,
        ),
    )
    print(f"{role} bound: {len(zs)} batches sealed")


XSA_GATES = Path("/home/erikg/sxgc/gates")

SHALLOW_DEPTHS = (1, 2, 3)


def shallow_oracle(contexts, text, starts, with_table=False):
    """Exact no-cross brute force for depth 1..3 contexts over `text`.

    `contexts` is a set of context tuples (bytes of length 1..3); returns
    {ctx: dict(width=, eof=, hist=256 ints)}. Raw (context, successor)
    pair counts come from one chunked pass; every no-cross correction
    (crossing occurrences and END branches) is a gather over the KNOWN
    document starts, so boundary work is O(docs), not O(text).

    The law: an occurrence of the L-byte context at position p is a
    crossing match iff a document start lies in (p, p+L-1]; an END branch
    iff p+L is a document start or the end of the text; else its content
    successor is text[p+L].
    """
    text = np.asarray(text, np.uint8)
    starts = np.asarray(starts, np.int64)
    text_len = len(text)

    def end(off):
        # a gather around every document start; positions outside the text
        # (the first document has no preceding byte) carry -1, which can
        # never equal a byte value, so fabricated bytes cannot match
        idx = starts + off
        ok = (idx >= 0) & (idx < text_len)
        out = np.full(len(starts), -1, np.int64)
        out[ok] = text[idx[ok]]
        return out

    wants = {}
    for L in SHALLOW_DEPTHS:
        cs = sorted({bytes(c) for c in contexts if len(c) == L})
        # the rolling chunk keys treat the FIRST context byte as most
        # significant; read the same way here
        wants[L] = np.asarray([int.from_bytes(c, "big") for c in cs], np.int64)
    raw_tables = {}

    step = 1 << 28
    if len(wants[1]):
        raw_tables[1] = np.zeros(1 << 16, np.int64)
    if len(wants[2]):
        raw_tables[2] = np.zeros(1 << 24, np.int64)
    if len(wants[3]):
        lut = np.zeros(256, bool)
        lut[(wants[3] >> 16).astype(np.uint8)] = True
        raw_tables[3] = np.zeros((len(wants[3]), 256), np.int64)
    for lo in range(0, text_len, step):
        chunk = text[lo : min(lo + step + 3, text_len)]
        # each chunk owns exactly [lo, lo+step) pair positions: the +3
        # overlap exists only so the key builders see whole contexts
        if len(wants[1]):
            keep = min(step, text_len - 1 - lo)
            k = chunk[: keep + 1]
            k = k[:-1].astype(np.int64) * 256 + k[1:]
            raw_tables[1] += np.bincount(k, minlength=1 << 16)
        if len(wants[2]):
            keep = min(step, text_len - 2 - lo)
            k = chunk[: keep + 2]
            k = k[:-2].astype(np.int64) * 65536 + k[1:-1].astype(np.int64) * 256 + k[2:]
            raw_tables[2] += np.bincount(k, minlength=1 << 24)
        if len(wants[3]):
            keep = min(step, text_len - 3 - lo)
            k = chunk[: keep + 3]
            s = k[3:]
            k = (
                k[:-3].astype(np.int64) * 65536
                + k[1:-2].astype(np.int64) * 256
                + k[2:-1]
            )
            m = lut[(k >> 16).astype(np.int64)]
            if m.any():
                cand = k[m]
                pos = np.clip(np.searchsorted(wants[3], cand), 0, len(wants[3]) - 1)
                hit = wants[3][pos] == cand
                if hit.any():
                    pair = pos[hit] * 256 + s[m][hit]
                    np.add.at(raw_tables[3], (pair >> 8, pair & 255), 1)

    out = {}
    for L in SHALLOW_DEPTHS:
        for ctx_b in (bytes(c) for c in contexts if len(c) == L):
            key = ctx_b
            kint = int.from_bytes(key, "big")
            if L == 1:
                raw = raw_tables[1][kint * 256 : kint * 256 + 256].copy()
            elif L == 2:
                raw = raw_tables[2][kint * 256 : kint * 256 + 256].copy()
            else:
                row = int(np.searchsorted(wants[3], kint))
                raw = (
                    raw_tables[3][row].copy()
                    if row < len(wants[3]) and wants[3][row] == kint
                    else np.zeros(256, np.int64)
                )
            if L == 1:
                pre = end(-1)
                s0 = end(0)
                m = pre == key[0]
                eof = int(m.sum()) + int(text[-1] == key[0])
                hist = raw - np.bincount(np.clip(s0[m], 0, 255), minlength=256).astype(
                    np.int64
                )
            elif L == 2:
                a, b = key
                pre2, pre1, s0, s1 = end(-2), end(-1), end(0), end(1)
                cross = (pre1 == a) & (s0 == b)
                eofm = (pre2 == a) & (pre1 == b)
                eof = int(eofm.sum()) + int(text[-2:].tobytes() == key)
                hist = (
                    raw
                    - np.bincount(np.clip(s1[cross], 0, 255), minlength=256)
                    - np.bincount(np.clip(s0[eofm], 0, 255), minlength=256)
                ).astype(np.int64)
            else:
                a, b, c = key
                pre3, pre2, pre1 = end(-3), end(-2), end(-1)
                s0, s1, s2 = end(0), end(1), end(2)
                cross1 = (pre1 == a) & (s0 == b) & (s1 == c)
                cross2 = (pre2 == a) & (pre1 == b) & (s0 == c)
                eofm = (pre3 == a) & (pre2 == b) & (pre1 == c)
                eof = int(eofm.sum()) + int(text[-3:].tobytes() == key)
                hist = (
                    raw
                    - np.bincount(np.clip(s2[cross1], 0, 255), minlength=256)
                    - np.bincount(np.clip(s1[cross2], 0, 255), minlength=256)
                    - np.bincount(np.clip(s0[eofm], 0, 255), minlength=256)
                ).astype(np.int64)
            out[key] = dict(width=int(hist.sum()) + eof, eof=eof, hist=hist)
    if with_table:
        return out, raw_tables[1]
    return out


def _oracle_query_lines(cands):
    lines = []
    for pos, d in cands:
        lines.append(f"{pos} {d}")
    return lines


def _parse_oracle(path):
    out = {}
    for ln in open(path):
        kv = dict(x.split("=", 1) for x in ln.split() if "=" in x)
        pairs = {}
        if kv.get("pairs"):
            for item in kv["pairs"].split(","):
                b, c = item.split(":")
                pairs[int(b)] = int(c)
        out[(int(kv["off"]), int(kv["ctxlen"]))] = dict(
            width=int(kv["width"]), eof=int(kv["eof"]), pairs=pairs
        )
    return out


def qualify(jobs=96):
    """Transport qualification on the real container, invented positions.

    1. The project's X3 end-to-end gate (no-cross law, boundary-exactness,
       --exclude-source semantics, docid parity, >=50 brute-forced rows).
    2. A depth sweep d=1..8 over 64 deterministic archive-document positions
       vs the no-cross brute-force oracle, per (width, eof, pairs).
    3. Determinism: a repeated pass must reproduce the same rows sha.
    4. Depth 0: the 256 single-byte contexts vs the oracle, then the
       archive's successor marginal and row count.
    """
    rebind()
    sal_settings.cpu()
    qualified()
    csha, bsha = verify_tools()
    need(sha(CORPUS100) == CORPUS100_SHA, "container text file sha mismatch")
    work = B3 / "transport-qualification"
    resumed = work.exists()
    if not resumed:
        work.mkdir(mode=0o700, parents=True)

    def journal_facts(rows_path, expect_positions):
        """Re-verify a finished consult pass from its journal and rows."""
        j = Path(str(rows_path) + ".journal")
        need(j.is_file(), f"missing journal for {rows_path}")
        done = [
            ln for ln in j.read_text().splitlines() if ln.startswith("CONSULT_DONE")
        ]
        need(len(done) == 1, f"missing CONSULT_DONE in {j}")
        facts = dict(kv.split("=", 1) for kv in done[0].split()[1:] if "=" in kv)
        need(
            int(facts["positions"]) == expect_positions,
            f"journal positions mismatch for {rows_path}",
        )
        return facts

    x3log = work / "x3.log"
    if resumed and x3log.is_file() and "X3_RESULT PASS" in x3log.read_text():
        x3 = x3log.read_text()
    else:
        with x3log.open("wb") as f:
            p = subprocess.run(
                [
                    "bash",
                    str(XSA_GATES / "no_cross_x3_consult.sh"),
                    str(CONTAINER),
                    str(CORPUS100),
                    str(SEPS),
                    str(work / "x3"),
                    str(jobs),
                ],
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=14400,
            )
        x3 = x3log.read_text()
        need(p.returncode == 0 and "X3_RESULT PASS" in x3, "X3 gate failed")
    x3_result = [ln for ln in x3.splitlines() if ln.startswith("X3_RESULT")][0]
    x3_spot = [ln for ln in x3.splitlines() if ln.startswith("X3_ORACLE_SPOT_CHECK")][0]

    docs, starts = doc_starts()
    cands = []
    for i in range(len(starts) - 1):
        s, e = int(starts[i]), int(starts[i + 1])
        if e <= 100_000_000_000 and e - s >= 2048:
            cands.append((i, s + 1024))
        if len(cands) == 64:
            break
    need(len(cands) == 64, "insufficient qualification documents")
    qfile = work / "sweep-queries.txt"
    with qfile.open("w") as f:
        for i, pos in cands:
            for d in DEPTHS:
                if d >= 4:  # shallower depths use the numpy no-cross oracle:
                    # the C oracle's short-context path is a full scan
                    f.write(f"{pos} {d}\n")

    rates = {}
    sweep_shas = {}
    for d in DEPTHS:
        w = work / f"sweep-w{d}.jsonl"
        out = work / f"sweep-rows{d}.bin"
        if out.is_file():
            facts = journal_facts(out, 64)
        else:
            with w.open("w") as f:
                for i, pos in cands:
                    f.write(
                        json.dumps(dict(doc=int(i), start=1024, len=1, left=d)) + "\n"
                    )
            facts = consult_pass("qual", d, "sweep", w, out, jobs=jobs, exclude=False)
            need(int(facts["positions"]) == 64, "sweep positions")
        rates[d] = facts["positions_per_s"]
        sweep_shas[d] = facts["rows_sha256"]

    oracle = work / "nocross_oracle"
    if not oracle.is_file():
        subprocess.run(
            ["gcc", "-O2", "-o", str(oracle), str(XSA_GATES / "nocross_oracle.c")],
            check=True,
        )
    oout = work / "sweep-oracle.out"
    if not oout.is_file():
        with oout.open("wb") as f:
            subprocess.run(
                [
                    str(oracle),
                    str(SEPS),
                    str(CORPUS),
                    "100000000000",
                    str(qfile),
                ],
                stdout=f,
                check=True,
                timeout=3600,
            )
    want = _parse_oracle(oout)
    need(len(want) == 5 * 64, "oracle rows")
    # the shallow no-cross oracle: one chunked numpy pass over the
    # container's text for every depth-1..3 context of the sweep
    text100 = np.memmap(CORPUS, dtype=np.uint8, mode="r")[:100_000_000_000]
    _, starts = doc_starts()
    shallow_ctxs = set()
    for i, pos in cands:
        for d in (1, 2, 3):
            shallow_ctxs.add(tuple(bytes(text100[pos - d : pos])))
    shallow, table1 = shallow_oracle(shallow_ctxs, text100, starts, with_table=True)
    present = {
        int(c) for c in range(256) if int(table1[c * 256 : (c + 1) * 256].sum()) > 0
    }
    checked = 0
    for d in DEPTHS:
        rows = read_rows(work / f"sweep-rows{d}.bin")
        need(len(rows) == 64, "sweep rows")
        for j, (i, pos) in enumerate(cands):
            r = rows[j]
            if d >= 4:
                wq = want[(pos, d)]
                w_pairs = wq["pairs"]
            else:
                sw = shallow[bytes(text100[pos - d : pos])]
                wq = dict(width=sw["width"], eof=sw["eof"])
                w_pairs = {int(b): int(c) for b, c in enumerate(sw["hist"]) if c}
            got_pairs = {int(b): int(c) for b, c in enumerate(r["hist"]) if c}
            need(
                r["ctxlen"] == d
                and not (r["flags"] & 16)
                and r["width"] == wq["width"]
                and r["eof"] == wq["eof"]
                and got_pairs == w_pairs,
                f"depth-sweep mismatch at {pos} d={d}",
            )
            checked += 1

    # determinism: repeat one pass, byte-identical rows
    w = work / "det-w.jsonl"
    det = work / "det-rows.bin"
    if det.is_file():
        facts = journal_facts(det, 64)
    else:
        if not w.is_file():
            with w.open("w") as f:
                for i, pos in cands:
                    f.write(
                        json.dumps(dict(doc=int(i), start=1024, len=1, left=4)) + "\n"
                    )
        facts = consult_pass("qual", 4, "det", w, det, jobs=jobs, exclude=False)
    need(facts["rows_sha256"] == sweep_shas[4], "determinism mismatch")

    # depth 0: the 256 single-byte contexts vs the oracle, then the marginal
    d0out = str(work / "depth0.bin")
    need(not Path(d0out).is_file(), "depth0 pass already exists; remove to redo")
    depth0_pass(d0out, jobs=jobs, present=present)
    d0rows = read_rows(d0out)
    need(len(d0rows) == len(present), "depth0 rows")
    shallow0 = shallow_oracle(
        {tuple(bytes(text100[r["off"] - 1 : r["off"]])) for r in d0rows},
        text100,
        starts,
    )
    for r in d0rows:
        sw = shallow0[bytes(text100[r["off"] - 1 : r["off"]])]
        got = {int(b): int(c) for b, c in enumerate(r["hist"]) if c}
        wq_pairs = {int(b): int(c) for b, c in enumerate(sw["hist"]) if c}
        need(
            r["ctxlen"] == 1
            and not (r["flags"] & 16)
            and r["width"] == sw["width"]
            and r["eof"] == sw["eof"]
            and got == wq_pairs,
            f"depth0 mismatch at {r['off']}",
        )
    h0 = np.zeros(256, np.int64)
    for r in d0rows:
        h0 += r["hist"]
    n_rows = int(h0.sum())
    put(
        B3 / "transport-qualification.json",
        dict(
            container_sha256=csha,
            binary_sha256=bsha,
            container_text_sha256=CORPUS100_SHA,
            resumed_from_artifacts=resumed,
            sweep_rows_shas=sweep_shas,
            note="first attempt failed in the X3 script's own oracle arm: it stat-reads the full 1.3 TB corpus file; the gate now runs against the container's exact materialized text (sha-matched to the rehearsal journal)",
            x3_result=x3_result,
            x3_spot=x3_spot,
            sweep_positions=len(cands),
            sweep_depth_checks=checked,
            sweep_rates=rates,
            determinism_sha=sweep_shas[4],
            depth0_h0=h0.tolist(),
            depth0_rows=n_rows,
            oracle_sha256=sha(XSA_GATES / "nocross_oracle.c"),
        ),
    )
    print("transport qualification PASS; law rows:", n_rows)


def prepare():
    rebind()
    reserve()
    sal_data.extract("fit")
    csha, bsha = verify_tools()
    cpath = DATA3 / "container.json"
    if cpath.is_file():
        c0 = get(cpath)
        need(
            c0["container_sha256"] == csha
            and c0["rows"] > 0
            and len(c0["h0"]) == 256
            and (DATA3 / "depth0.bin").is_file(),
            "sealed depth0 receipt mismatch",
        )
        d0 = dict(rows=c0["rows"], h0=c0["h0"], facts=c0["depth0"])
    else:
        d0 = depth0_pass(str(DATA3 / "depth0.bin"))
    n = d0["rows"]
    st = os.stat(CORPUS)
    put(
        DATA3 / "container.json",
        dict(
            path=str(CONTAINER),
            container_sha256=csha,
            binary=str(XSA),
            binary_sha256=bsha,
            corpus=dict(
                path=str(CORPUS),
                inode=st.st_ino,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
            ),
            n=100000000001,
            runs=31381597694,
            rows=n,
            h0=d0["h0"],
            depth0=d0["facts"],
            roles_sha=sha(DATA3 / "rows.json"),
            roles_amendment="in-container-exclude-source",
            bound=False,
        ),
    )
    bind_role("fit")
    c = get(DATA3 / "container.json")
    c["bound"] = True
    put(DATA3 / "container.json", c)
    put(
        DATA3 / "ready.json",
        dict(
            rows=sha(DATA3 / "rows.json"),
            fit=sha(DATA3 / "fit.json"),
            container=sha(DATA3 / "container.json"),
        ),
    )
    print("SAL-3 FIT bound; PROBE locked.")


def _gate_cli(argv):
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
            stage = get(JOB3 / "admission.json")["stage"]
            (JOB3 / f"watch-failure-{stage}").write_text(repr(exc))
            raise
    elif action == "monitor":
        sal_gate.monitor(args[0])
    else:
        raise ValueError("unknown action")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which == "prepare":
        prepare()
    elif which == "bind-probe":
        bind_role("probe")
    elif which == "probe":
        rebind()
        sal_data.extract("probe")
    elif which == "verify-tools":
        print(verify_tools())
    elif which == "qualify":
        qualify()
    elif which in ("spawn", "prelease", "check", "watch", "monitor"):
        rebind()
        _gate_cli(sys.argv[1:])
    else:
        raise ValueError("unknown action")


if __name__ == "__main__":
    main()
