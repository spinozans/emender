#!/usr/bin/env python3
"""Fail-closed W3 root-volume capacity gate; called immediately before each arm."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

# Conservatively interpret the supervisor's 200G floor as 200 GiB.
MIN_ROOT_FREE_BYTES = 200 * (1 << 30)


def capacity_receipt(arm, free_bytes):
    return dict(schema='emender-w3-root-capacity-gate-v1', arm=arm, volume='/',
                minimum_free_bytes=MIN_ROOT_FREE_BYTES, free_bytes=free_bytes,
                threshold_units='200 GiB (conservative 200G interpretation)',
                status='passed' if free_bytes >= MIN_ROOT_FREE_BYTES else 'blocked',
                automatic_retry=False, checked_at=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('A', 'B'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = capacity_receipt(args.arm, shutil.disk_usage('/').free)
    with args.output.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print('W3_ROOT_CAPACITY_GATE', json.dumps(report, sort_keys=True), flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
