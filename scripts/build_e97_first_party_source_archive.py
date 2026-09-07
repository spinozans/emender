#!/usr/bin/env python3
"""Build or verify the checked-in deterministic first-party source archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ndm.e97_first_party_source_archive import build_source_archive, verify_source_archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        digest = verify_source_archive(args.manifest, args.archive, checkout_root=args.checkout_root)
    else:
        digest = build_source_archive(args.manifest, args.archive, checkout_root=args.checkout_root)
    print(json.dumps({"source_archive_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
