"""Read only the pinned v2 TRAIN byte stream, never an eval panel or arbitrary corpus offset."""

from __future__ import annotations

import json
from .panels import OUT, SEEDS, T, V2_QUOTAS, V2_SPEC, check, digest, fingerprint, order

V2_IDENTITY = "f6ea7725ada25ce23b304e0af1a64588669bf13b3ce3a1c11a8743da43dc961b"


def bound(path, expected):
    raw = path.read_bytes()
    check(digest(raw) == expected, "changed " + path.name)
    return raw


class TrainStream:
    """Small immutable stream in RAM. Step is 1-based; no wrap and no corpus file handle."""

    def __init__(self, root=OUT):
        base = root / "v2"
        folder = base / "train"
        ident = json.loads(bound(folder / "identity.json", V2_IDENTITY))
        check(
            ident["schema"] == "x49-train-v2"
            and ident["evaluation_windows_read"] == 0
            and ident["targets"] == V2_QUOTAS["train"] * T
            and ident["rows"] == V2_QUOTAS["train"],
            "not the certified TRAIN dose",
        )
        reserve = json.loads(bound(base / "reservation.json", ident["reservation_sha"]))
        check(
            reserve["schema"] == "x49-v2" and reserve["amendment_sha"] == V2_SPEC,
            "foreign reservation",
        )
        rows = reserve["panels"]["train"]
        self.data = bound(folder / "windows.u8", ident["raw_sha"])
        hashes = json.loads(bound(folder / "hashes.json", ident["shifts_sha"]))
        check(
            len(self.data) == len(rows) * (T + 1)
            and len(rows) == len(hashes) == ident["rows"],
            "truncated TRAIN stream",
        )
        for i, (row, h) in enumerate(zip(rows, hashes, strict=True)):
            w = self._row(i)
            check(
                row["split"] == "train"
                and h
                == dict(window=digest(w), inputs=digest(w[:-1]), targets=digest(w[1:])),
                "wrong document shift or split",
            )
        self.docs = tuple(r["doc"] for r in rows)
        self.orders = {seed: tuple(order(rows, seed)) for seed in SEEDS}
        check(
            len(set(self.docs)) == len(rows)
            and {str(s): fingerprint(list(v)) for s, v in self.orders.items()}
            == ident["order_sha"],
            "wrong seed-specific order",
        )

    def _row(self, i):
        a = i * (T + 1)
        return self.data[a : a + T + 1]

    def sample(self, seed, step):
        check(
            type(seed) is int
            and seed in self.orders
            and type(step) is int
            and 1 <= step <= len(self.docs),
            "invalid TRAIN seed or step",
        )
        i = self.orders[seed][step - 1]
        return self.docs[i], self._row(i)
