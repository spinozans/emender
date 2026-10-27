from experiments.expressivity_tasks.x39_memory_usage.part4_match import match_receipt


def test_match_wire_requires_semantic_keys_and_preserves_novel_writes():
    receipt = match_receipt()
    identity, unwired, wired = receipt['rows']
    assert identity['repeat_allocation'] == unwired['repeat_allocation'] == 1.
    assert wired['novel_allocation'] == 1.
    assert wired['repeat_allocation'] == 0.
