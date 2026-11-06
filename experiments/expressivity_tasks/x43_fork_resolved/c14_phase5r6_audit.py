"""Arithmetic receipt, not a physical DRAM counter or architecture certificate."""
import json
from pathlib import Path


def audit(tokens=1024, peak=257, additions_per_position=1, initial_live=0):
    # R5 fixture: no array insertions/deaths, stream write enabled at t<257.
    live_visits = sum(min(initial_live + additions_per_position * t, peak) for t in range(tokens))
    slot_bytes = 4 * (64 + 128)
    inputs = tokens * (64 + 260) * 4
    # One incoming and outgoing FP32 two-parity carry per position.
    carries = tokens * 2 * (2 * 128) * 4
    total = live_visits * slot_bytes + inputs + carries
    max_head_ms = 86400 * tokens * 1000 / (2_000_000_000 * 88)
    return dict(tokens=tokens, peak=peak, additions_per_position=additions_per_position,
                initial_live=initial_live, live_slot_visits=live_visits,
                slot_payload_bytes=live_visits * slot_bytes,
                query_and_260_features_bytes=inputs, carry_read_write_bytes=carries,
                one_pass_payload_bytes=total,
                required_payload_bandwidth_GBs=total / (max_head_ms / 1000) / 1e9,
                necessary_head_ms_for_24h=max_head_ms,
                interpretation='Conditional one-pass fast-tier payload model; NOT compulsory physical DRAM traffic. '
                'Excludes metadata, dirty-state persistence, outputs, initialization, dense and backward work. '
                'Cache residency can eliminate repeated physical DRAM transfers.')


if __name__ == '__main__':
    import sys
    output = Path(sys.argv[1])
    assert not output.exists()
    output.write_text(json.dumps(dict(fixture=audit(), two_allocations_per_position=audit(peak=2048, additions_per_position=2),
        persistent_full_store=audit(peak=2048, initial_live=2048),
        actual_dram_bytes=None, actual_to_floor_ratio=None,
        actual_traffic_status='FAILED: fresh ncu ERR_NVGPUCTRPERM; RmProfilingAdminOnly=1',
        architecture_verdict='UNDETERMINED: counter gap missing and fast-tier payload is not a compulsory DRAM floor'), indent=2) + '\n')
