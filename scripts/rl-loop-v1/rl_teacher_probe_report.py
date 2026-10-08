#!/usr/bin/env python3
"""Recompute F7 metrics from raw API attempts, sealed outcomes and cohort walls.

Never trust the initial A/B controller's misleading completed_per_hour label:
its numerator was ALL finalized outcomes, not closed/accepted corrections.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

MODELS = ("glm-5.3-flash-background", "glm-5.3-background", "deepseek-4.1-flash-background")


def distribution(values):
    values = sorted(values)
    def percentile(p):
        if not values:
            return None
        position = (len(values)-1)*p
        lo = int(position)
        hi = min(lo+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(position-lo)
    return {"n": len(values), "min": min(values) if values else None,
            "p50": percentile(.5), "p90": percentile(.9), "p95": percentile(.95),
            "max": max(values) if values else None}


def measure(path):
    cohort = json.loads(path.read_text())
    rows = cohort["rows"]
    assert len(rows) == cohort["n"] and cohort["wall_s"] > 0
    calls = []
    calls_per_correction = []
    successful_calls_per_correction = []
    complete_episode_calls = []
    retry_outcomes = {"receipt_after_api_error": 0, "no_receipt_after_api_error": 0}
    for row in rows:
        api = path.parent / f'case-{row["index"]:04d}' / "api.jsonl"
        attempt_calls = [json.loads(line) for line in api.read_text().splitlines()] if api.exists() else []
        calls.extend(attempt_calls)  # includes attempts in crashed/timeout workers
        calls_per_correction.append(len(attempt_calls))
        if row.get("status") == "finished" and row.get("close_verified", False):
            complete_episode_calls.append(len(attempt_calls))
        successful_calls_per_correction.append(sum(not c.get("error") for c in attempt_calls))
        if any(c.get("error") for c in attempt_calls):
            retry_outcomes["receipt_after_api_error" if row["receipt"] else "no_receipt_after_api_error"] += 1
    receipts = sum(bool(r["receipt"]) for r in rows)
    targets = sum(r["targets"] for r in rows)
    finished = sum(r.get("status") == "finished" and r.get("close_verified", False) for r in rows)
    assert all(not r["receipt"] or (r.get("grade_passed") and r.get("close_verified")
               and r.get("status") == "finished" and r["targets"] > 0) for r in rows)
    errors = [c for c in calls if c.get("error")]
    tokens = {key: sum((c.get("usage") or {}).get(key, 0) for c in calls)
              for key in ("prompt_tokens", "completion_tokens")}
    per_hour = 3600/cohort["wall_s"]
    return {"model": cohort["model"], "concurrency": cohort["concurrency"], "n": len(rows),
        "wall_s": cohort["wall_s"], "receipts": receipts,
        "grade_passes": sum(bool(r["grade_passed"]) for r in rows), "finished_closed": finished,
        "receipt_rate": receipts/len(rows), "targets": targets,
        "canonical_targets_per_receipt": targets/receipts if receipts else None,
        "api_tokens": tokens,
        "api_completion_tokens_per_receipt": tokens["completion_tokens"]/receipts if receipts else None,
        "api_prompt_tokens_per_receipt": tokens["prompt_tokens"]/receipts if receipts else None,
        "api_attempts": len(calls), "api_errors": len(errors),
        "mean_call_latency_s": sum(c["latency_s"] for c in calls)/len(calls) if calls else None,
        "mean_calls_per_correction": len(calls)/len(rows),
        "calls_per_correction": distribution(calls_per_correction),
        "calls_per_complete_correction": distribution(complete_episode_calls),
        "mean_calls_per_complete_correction": sum(complete_episode_calls)/len(complete_episode_calls) if complete_episode_calls else None,
        "complete_correction_api_attempts": sum(complete_episode_calls),
        "successful_calls_per_correction": distribution(successful_calls_per_correction),
        "sealed_complete_grade_failures": sum(r.get("status") == "finished" and
            r.get("close_verified", False) and not r["grade_passed"] for r in rows),
        "incomplete_or_queue_missing": len(rows)-finished,
        "nonqueue_api_errors": sum("HTTP Error 429" not in c["error"] for c in errors),
        "queue_wait_429_s": distribution([c["latency_s"] for c in errors if "HTTP Error 429" in c["error"]]),
        "terminal_drain_after_last_admission_s": cohort.get("terminal_drain_after_last_admission_s"),
        "api_error_rate": len(errors)/len(calls) if calls else None,
        "http_429": sum("HTTP Error 429" in c["error"] for c in errors),
        "queue_timeout_240s_errors": sum(235 <= c["latency_s"] <= 245 for c in errors),
        "transport_retry_attempts": sum(c.get("attempt", 0) > 0 for c in calls),
        "format_retry_count": sum(r.get("generator_state", {}).get("invalid_frames", 0) for r in rows),
        "worker_errors": sum(bool(r.get("error")) for r in rows),
        "retry_outcomes": retry_outcomes,
        "api_latency_s": distribution([c["latency_s"] for c in calls]),
        "api_success_latency_s": distribution([c["latency_s"] for c in calls if not c.get("error")]),
        "api_error_latency_s": distribution([c["latency_s"] for c in errors]),
        "post_import_episode_latency_s": distribution([r["elapsed_s"] for r in rows]),
        "finalized_outcomes_per_hour": len(rows)*per_hour,
        "completed_closed_per_hour": finished*per_hour, "receipts_per_hour": receipts*per_hour,
        "accepted_targets_per_hour": targets*per_hour,
        "raw_cohort_path": str(path), "capacity_definition": "burst+drain, finite sample, NOT sustained production limit"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / "plan.json").read_text())
    ab = [measure(args.root / "ab-4" / m / "cohort.json") for m in MODELS]
    ladder = [measure(args.root / f"ladder-{w}" / m / "cohort.json")
              for w in (4, 8, 12) for m in MODELS[1:]]
    assert all(r["n"] == 30 for r in ab)
    assert all(r["n"] == 2*r["concurrency"] for r in ladder)
    for row in ladder:
        # Policy ceiling with F6's noncollect gap held; teacher concurrency is global.
        policy_ceiling = 7*3600/(11.69467115+.2126593) + 3600/(301.6058817+11.69467115+.2126593)
        row["conditional_B_tasks_per_hour"] = min(policy_ceiling, row["finalized_outcomes_per_hour"]/.65)
        row["conditional_B_targets_per_hour_F6_yield_held"] = row["conditional_B_tasks_per_hour"]*189.75
        row["fresh_only_yield_B_targets_per_hour"] = row["conditional_B_tasks_per_hour"]*(62.7+.65*row["targets"]/row["n"])
    report = {"schema": "emender-f7-measurements-v1", "ab": ab, "ladder": ladder,
        "deepseek_receipt_delta_pp": 100*(ab[2]["receipt_rate"]-ab[0]["receipt_rate"]),
        "preregistered_sample_noninferiority_met": ab[2]["receipt_rate"] >= ab[0]["receipt_rate"]-.05,
        "archived_mix": plan["archived_mix"],
        "endpoint_limit_lanes": 12,
        "endpoint_limit_source": "operator statement 2026-10-08; 429s are 240s queue waits, NOT degradation",
        "projection_label": "fresh-correction capacity measured; mixed-capacity projected, unmeasured",
        "projection_assumptions": ["7 continuously collecting GPUs + 1 learner retaining F6 noncollect gaps",
            "0.65 corrections per finalized task attempt; fixed scalar policy/10s admission",
            "fresh-only burst capacity assumed transferable to majority-spliced workload (UNMEASURED)",
            "F6 accepted yield 189.75/task held only in the named conditional column",
            "fresh-only alternative uses F6 onpolicy targets 62.7/task + measured teacher targets/correction"],
        "F6_single_collect_tasks_h": 40*3600/4366.001572,
        "F6_single_collect_targets_h": 7590*3600/4366.001572,
        "F6_single_loop_tasks_h": 40*3600/16430.23684,
        "F6_single_loop_targets_h": 7590*3600/16430.23684,
        "F6_same_grid_tasks_h_projected": 7*40*3600/4366.001572 + 40*3600/16430.23684,
        "F6_same_grid_targets_h_projected": 7*7590*3600/4366.001572 + 7590*3600/16430.23684}
    # All three sealed-complete outcomes must exist for a cognitive paired comparison.
    cohorts = [json.loads((args.root / "ab-4" / m / "cohort.json").read_text())["rows"] for m in MODELS]
    complete = set.intersection(*[{r["index"] for r in rows if r.get("status") == "finished"
        and r.get("close_verified", False)} for rows in cohorts])
    report["paired_sealed_complete"] = {"n": len(complete), "indices": sorted(complete),
        "grade_passes": {m: sum(bool(r["grade_passed"]) for r in rows if r["index"] in complete)
                         for m, rows in zip(MODELS, cohorts)}}
    report["pairwise_sealed_complete"] = []
    for baseline_index in (0, 1):
        paired = {r["index"] for r in cohorts[baseline_index] if r.get("status") == "finished"
                  and r.get("close_verified", False)} & {r["index"] for r in cohorts[2]
                  if r.get("status") == "finished" and r.get("close_verified", False)}
        passes = [sum(bool(r["grade_passed"]) for r in cohorts[i] if r["index"] in paired)
                  for i in (baseline_index, 2)]
        report["pairwise_sealed_complete"].append({"baseline": MODELS[baseline_index], "n": len(paired),
            "baseline_passes": passes[0], "deepseek_passes": passes[1], "indices": sorted(paired),
            "deepseek_delta_pp": 100*(passes[1]-passes[0])/len(paired) if paired else None})
    ceilings = []
    grid = report["F6_same_grid_tasks_h_projected"]
    for model in MODELS:
        samples = [r for r in (ladder if model != MODELS[0] else ab) if r["model"] == model]
        calls = sum(r["api_attempts"] for r in samples)
        count = sum(r["n"] for r in samples)
        latency_sum = sum(r["mean_call_latency_s"]*r["api_attempts"] for r in samples)
        call_ceiling = 12*3600/(latency_sum/calls)
        complete_n = sum(r["finished_closed"] for r in samples)
        complete_calls = sum(r["complete_correction_api_attempts"] for r in samples)
        episode_ceiling = call_ceiling/(complete_calls/complete_n)
        attempt_ceiling = episode_ceiling/.65
        campaign = []
        for label, attempts in (("current B arm (F6-normalized projected demand)", grid),
                                ("2x campaign", 2*grid), ("12-lane episode proxy ceiling", attempt_ceiling)):
            demand = .65*attempts
            campaign.append({"campaign": label, "attempts_h": attempts,
                "minimum_policy_gpus_at_300_attempts_h": math.ceil(attempts/300),
                "correction_demand_h": demand, "ideal_continuous_backlog_growth_h": max(0, demand-episode_ceiling)})
        ceilings.append({"model": model, "latency_mean_s": latency_sum/calls,
            "mean_calls_per_correction_outcome": calls/count,
            "mean_calls_per_complete_correction": complete_calls/complete_n,
            "call_equivalent_ceiling_h": call_ceiling,
            "episode_proxy_ceiling_h": episode_ceiling, "attempt_proxy_ceiling_h": attempt_ceiling,
            "campaigns": campaign,
            "qualification": "operator formula using client wall latencies INCLUDING queue/retries; not queue-free service occupancy or a sustained measurement"})
    report["operator_formula_ceilings"] = ceilings
    pool_path = args.root / "pool-timing" / MODELS[2] / "summary.json"
    pool = json.loads(pool_path.read_text())
    report["actual_pool_timing_replay"] = {k: v for k, v in pool.items() if k != "rows"}
    report["actual_pool_timing_replay"]["eligible_receipts"] = sum(r["eligible_receipt"] for r in pool["rows"])
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
    # Raw evidence stays private. Manifest hashes contain no sealed prompts/answers.
    files = [p for p in args.root.rglob("*") if p.is_file() and
             (p.name in ("plan.json", "ladder-design.json", "pilot-isolated.py", "cohort.json", "result.json",
                         "api.jsonl", "correction-api.jsonl", "grade.json", "candidate-receipt.json",
                         "episode-private.json", "summary.json")
              or p.name.startswith(("gym-terminal-projection", "spec-", "result-")))]
    manifest = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
    args.manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"ab_receipts": [r["receipts"] for r in ab],
        "deepseek_delta_pp": report["deepseek_receipt_delta_pp"],
        "endpoint_limit_lanes": report["endpoint_limit_lanes"], "evidence_files": len(manifest)}))


if __name__ == "__main__":
    main()
