from experiments.expressivity_tasks.x43_fork_resolved.c14_gap_receipts import sharing_counts


def test_internal_repetition_is_not_cross_head_redundancy():
    total,per_head,union=sharing_counts([[b'a',b'a'],[b'b',b'b']])
    assert total/union == 2
    assert per_head/union == 1


def test_shared_keys_count_once_across_heads():
    total,per_head,union=sharing_counts([[b'a',b'a'],[b'a']])
    assert (total,per_head,union)==(3,2,1)


def test_empty_head_does_not_manufacture_sharing():
    assert sharing_counts([[],[b'a']]) == (1,1,1)
