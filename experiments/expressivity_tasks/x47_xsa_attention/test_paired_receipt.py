from collections import Counter
from .paired_frontier_verify import accounted_eager_debt


def test_zero_omitted_by_counter_addition_requires_complete_positive_byte_accounting():
    total = dict(
        sum(
            (Counter(observed_bytes=129, native_bytes=129, unsynced_at_close=0) for _ in range(16)), Counter()
        )
    )
    assert "unsynced_at_close" not in total and accounted_eager_debt(total)
    assert accounted_eager_debt(dict(total, unsynced_at_close=0))
    assert not any(
        map(
            accounted_eager_debt,
            [
                {},
                {"observed_bytes": 129},
                {"native_bytes": 129},
                dict(total, native_bytes=2063),
                dict(total, native_bytes=2065),
                dict(total, unsynced_at_close=1),
                dict(total, unsynced_at_close=-1),
                dict(total, unsynced_at_close=False),
            ],
        )
    )
