from __future__ import annotations

import json
from pathlib import Path

from scripts import build_e97_systematic_representation_stage as stage


def test_frozen_recipe_is_exactly_bound():
    recipe = json.loads(Path("configs/pi/e97-systematic-representation-stage-v1.json").read_text())
    assert recipe["schema"] == stage.SCHEMA
    assert recipe["status"] == "frozen"
    assert recipe["seed"] == stage.SEED
    assert recipe["assistant_target_tokens"] == stage.TARGETS
    assert recipe["source_manifest_sha256s"] == {
        name: digest for name, (_, digest) in stage.SOURCES.items()
    }
    assert recipe["matched_validation_sha256"] == stage.MATCHED_VALIDATION_SHA256
    assert recipe["matched_pack_validation_sha256"] == stage.MATCHED_PACK_VALIDATION_SHA256
    assert recipe["matched_trajectory_identity_sha256"] == stage.MATCHED_TRAJECTORY_SET_SHA256


def test_closest_prefix_keeps_complete_integral_units():
    assert stage.closest_prefix((("a", 4), ("b", 4), ("c", 4)), 6) == ["a", "b"]
    assert stage.closest_prefix((("a", 4), ("b", 7)), 5) == ["a", "b"]


def test_second_pass_finds_exact_single_or_pair():
    weights = {"a": 3, "b": 5, "c": 7, "d": 11}
    selected, total = stage.choose_second_pass(weights, 8)
    assert total == 8
    assert sum(weights[item] for item in selected) == 8
    assert len(selected) == len(set(selected))


def test_second_pass_is_deterministic_and_bounded_to_one_repeat():
    weights = {f"t-{index}": index + 2 for index in range(20)}
    first = stage.choose_second_pass(weights, 67)
    second = stage.choose_second_pass(weights, 67)
    assert first == second
    identities, total = first
    assert len(identities) == len(set(identities))
    assert total == sum(weights[item] for item in identities)
