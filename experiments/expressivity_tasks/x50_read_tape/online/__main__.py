"""Try a P100 work tape on a supplied short prefix. No fit, dataset access, or implicit fallback."""

import argparse
import json

import torch

from .. import data, study
from ..kernel import Kernel
from ..mount import mount
from ..program import ARMS, T
from .stream import ReadSession


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, default="ONE")
    ap.add_argument("--seed", type=int, choices=data.SEEDS, default=5001)
    ap.add_argument(
        "--birth",
        action="store_true",
        help="use the unfitted initializer, not the fitted reader",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="UTF-8 prefix")
    group.add_argument("--hex", help="arbitrary bytes in hex")
    args = ap.parse_args()
    raw = bytes.fromhex(args.hex) if args.hex is not None else args.text.encode("utf-8")
    if not 1 <= len(raw) <= T:
        ap.error(
            f"expected 1..{T} bytes in one window; no silent truncation or padding"
        )
    study.setup()
    m = study.model_at_birth(args.seed)
    if not args.birth:
        key = f"{args.seed}-{args.arm}"
        path = data.RUN / (key + ".pt")
        if not path.exists():
            ap.error(
                "local FIT state unavailable; use --birth only to try the explicitly unfitted reader"
            )
        seal = data.frozen()
        data.check(seal["states"][key] == data.sha(path), "changed matching FIT state")
        m.load_state_dict(
            torch.load(path, weights_only=True, map_location="cpu"), strict=True
        )
    m.eval()
    with mount() as pages:
        session = ReadSession(m, Kernel(pages), args.arm)
        for b in raw:
            result = session.observe(b)
        probs = result.logits.softmax(-1)
        pred = probs.argmax().item()
        print(
            json.dumps(
                dict(
                    arm=args.arm,
                    fit="birth" if args.birth else "frozen",
                    observed_bytes=len(raw),
                    work_rows=session.rows,
                    hypothetical_extension=result.extension,
                    predicted_byte=pred,
                    predicted_probability=probs[pred].item(),
                )
            )
        )


if __name__ == "__main__":
    main()
