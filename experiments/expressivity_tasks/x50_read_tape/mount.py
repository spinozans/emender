"""Pin the existing P100 branch transport; check invented queries with an independent binary."""

from contextlib import contextmanager
import re
import subprocess

import numpy as np

from .data import check, sha
from .kernel import Kernel, WEIGHTS

NATIVE = "12af5d707155e9301f1c092e7b75bad224b0b36dff2e6befd4789d33e8a9ec40"
STOCK_SHA = "fbc436aac93190368bbed637a5f69cca0fa5848fc81a311e60c4ebe1b4d3bf50"
DESCRIPTOR = "eabfebf0b7b02b3f0cfdd8d213efd7cacafcca378b792474d4ac6ab4e0573c93"


@contextmanager
def mount():
    from experiments.expressivity_tasks.x47_xsa_attention import w3_data as w

    check(
        sha(w.EXE) == NATIVE and sha(w.HERE / "b1_mounts.json") == DESCRIPTOR,
        "changed native P100 mount",
    )
    page = w.Pages()
    try:
        yield page
    finally:
        page.close()


def native_check(pages):
    from experiments.expressivity_tasks.x47_xsa_attention.w3_data import STOCK

    check(sha(STOCK) == STOCK_SHA, "foreign stock binary")
    hist = {}

    def stock(key):
        if key not in hist:
            raw = subprocess.run(
                [STOCK, "branch-of", pages.desc.index, "hex:" + key.hex()],
                text=True,
                capture_output=True,
                check=True,
                timeout=60,
            ).stdout
            counts = np.zeros(257, np.float64)
            if raw.startswith("present "):
                for b, n in re.findall(
                    r"^child (END|0x[0-9a-f]{2}) count=(\d+) node=", raw, re.M
                ):
                    counts[256 if b == "END" else int(b, 16)] = int(n)
                check(
                    counts.sum() == int(re.match(r"present count=(\d+)", raw)[1]),
                    "incomplete stock branch",
                )
            else:
                check(raw.startswith("absent depth="), "unknown stock status")
            hist[key] = counts[:256]
        return hist[key]

    bank = Kernel(pages)
    words = (b" the", b"\0\x1e\xff", b"ababababab")
    n = 0
    for word in words:
        x = np.frombuffer(word, np.uint8)[None, :]
        scan = bank.scan(x)
        for i in range(len(word)):
            q = word[: i + 1]
            a = int(scan.pdf[0, i].argmax())
            followed = bank.follow(scan.at(i), np.array([a], np.uint8))
            for key, p in [(q, scan.pdf[0, i]), (q + bytes([a]), followed.pdf[0])]:
                v = stock(b"").copy()
                for d in range(1, min(8, len(key)) + 1):
                    v += WEIGHTS[d] * stock(key[-d:])
                np.testing.assert_array_equal(p, (v / v.sum()).astype(np.float32))
                n += 1
    return dict(
        prefix_and_selected_reads=n,
        independent_stock_literals=len(hist),
        native=NATIVE,
        stock=STOCK_SHA,
        descriptor=DESCRIPTOR,
        input_documents=0,
    )
