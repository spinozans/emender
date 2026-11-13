"""Tests for the colinear chaining layer (chaining.py).

The central test is a synthetic case where the true correspondence is known by
construction: a query shares a long run with a known corpus offset, and the
anchor set contains distractor anchors on other diagonals. The chain must pick
the colinear set and the correspondence must name the corpus offset that the
shared run implies -- that is the whole claim of "seeds -> chain ->
correspondence", and it is checkable exactly.
"""

import random
import unittest

from chaining import (
    ABUTTING,
    GAP,
    MATCHED,
    Anchor,
    Chain,
    best_correspondence,
    chain_anchors,
    cluster_regions,
    correspondence_at,
    correspondences,
    fork_outcome,
)


def seeded_anchors(query_pos, corpus_pos, length, seed_id):
    """Anchors for one exact run; the way `xsa seeds` reports a seed's positions."""
    return [Anchor(query_pos, corpus_pos, length, seed_id)]


class SyntheticCorrespondenceTests(unittest.TestCase):
    def test_chain_recovers_the_known_corpus_offset_of_a_shared_run(self):
        # A 40-byte run shared with corpus offset 1000, plus three distractors
        # that each share the query's middle with unrelated corpus offsets.
        anchors = [
            Anchor(0, 1000, 40, seed_id=0),
            Anchor(8, 500_000, 24, seed_id=1),
            Anchor(12, 700_000, 20, seed_id=2),
            Anchor(20, 900_000, 12, seed_id=3),
        ]
        chains = chain_anchors(anchors)
        self.assertEqual(chains[0].n_anchors, 1)
        self.assertEqual((chains[0].corpus_start, chains[0].corpus_end), (1000, 1040))
        # The prediction position is the end of the query: the corpus position
        # that answers for it is the end of the shared run.
        corr = best_correspondence(chains, 40)
        self.assertIsNotNone(corr)
        self.assertEqual(corr.corpus_pos, 1040)
        self.assertEqual(corr.kind, ABUTTING)
        self.assertTrue(corr.usable)
        # And it is the same answer as the exact offset-arithmetic of the run.
        self.assertEqual(corr.corpus_pos, 1000 + 40)

    def test_two_colinear_anchors_beat_one_longer_anchor_on_another_diagonal(self):
        # 20 + 20 matched bytes on one diagonal (gap 10) versus a 30-byte anchor
        # on an unrelated diagonal. Score is covered bytes minus gaps, and the
        # 30-byte anchor is longer than either single anchor, so this also shows
        # chaining wins by *adding evidence*, not by picking the longest match.
        anchors = [
            Anchor(0, 5000, 20, seed_id=0),
            Anchor(30, 5030, 20, seed_id=1),
            Anchor(10, 800_000, 30, seed_id=2),
        ]
        chains = chain_anchors(anchors)
        self.assertEqual(chains[0].n_anchors, 2)
        self.assertEqual(chains[0].covered_query_bytes, 40)
        self.assertEqual([a.corpus_pos for a in chains[0].anchors], [5000, 5030])
        corr = best_correspondence(chains, 50)
        self.assertEqual(corr.corpus_pos, 5050)
        self.assertEqual(corr.kind, ABUTTING)

    def test_crossing_anchors_are_not_one_chain(self):
        # Same query span, corpus order reversed: not monotone, so no chain.
        anchors = [Anchor(0, 2000, 16, seed_id=0), Anchor(10, 1500, 16, seed_id=1)]
        chains = chain_anchors(anchors, min_score=1.0)
        self.assertEqual(len(chains), 2)
        for chain in chains:
            self.assertEqual(chain.n_anchors, 1)

    def test_inconsistent_diagonal_is_rejected_by_the_gap_penalty(self):
        # Colinear in order but the diagonal drifts by 40 bytes over a 4-byte
        # gap: the drift costs more than the second anchor earns, so it does not
        # join the chain.
        anchors = [Anchor(0, 3000, 20, seed_id=0), Anchor(24, 3044, 4, seed_id=1)]
        chains = chain_anchors(anchors, min_score=1.0)
        self.assertEqual(chains[0].n_anchors, 1)

    def test_inside_the_anchor_the_correspondence_is_matched_not_extrapolated(self):
        chain = Chain(anchors=(Anchor(10, 4000, 20, seed_id=0),), score=20.0)
        middle = correspondence_at(chain, 15)
        self.assertEqual((middle.kind, middle.corpus_pos, middle.extrapolation_bytes),
                         (MATCHED, 4005, 0))
        edge = correspondence_at(chain, 30)
        self.assertEqual((edge.kind, edge.corpus_pos), (ABUTTING, 4020))
        before = correspondence_at(chain, 9, max_extrapolation=8)
        self.assertIsNone(before)

    def test_gap_answer_needs_explicit_extrapolation_budget(self):
        chain = Chain(anchors=(Anchor(0, 4000, 20, seed_id=0),), score=20.0)
        self.assertIsNone(correspondence_at(chain, 25))
        gap = correspondence_at(chain, 25, max_extrapolation=8)
        self.assertEqual((gap.kind, gap.corpus_pos, gap.extrapolation_bytes), (GAP, 4025, 5))
        self.assertFalse(gap.usable)
        self.assertIsNone(correspondence_at(chain, 40, max_extrapolation=8))


class ChainShapeTests(unittest.TestCase):
    def test_score_is_covered_query_bytes_and_ignores_input_order(self):
        # Three mutually abutting 16-byte anchors: no gap, so the score is
        # exactly the covered query bytes.
        anchors = [Anchor(0, 100, 16, 0), Anchor(16, 116, 16, 1), Anchor(32, 132, 16, 2),
                   Anchor(5, 705, 12, 3)]
        chains = chain_anchors(anchors)
        self.assertEqual(chains[0].score, 48.0)
        self.assertEqual(chains[0].covered_query_bytes, 48)
        shuffled = list(anchors)
        random.Random(11).shuffle(shuffled)
        self.assertEqual([c.anchors for c in chain_anchors(shuffled)],
                         [c.anchors for c in chains])

    def test_an_interior_gap_costs_half_a_matched_byte_per_skipped_byte(self):
        # Two 16-byte anchors 24 bytes apart on both axes: 32 covered bytes, and
        # the gap costs 0.25 * (24 + 24) = 12, so the chain scores 20.
        chains = chain_anchors([Anchor(0, 100, 16, 0), Anchor(40, 140, 16, 1)])
        self.assertEqual(chains[0].score, 20.0)

    def test_overlapping_anchors_do_not_inflate_the_score(self):
        # A 20-byte seed and a 16-byte seed that starts 8 bytes into it and lies
        # on the same diagonal: 20 covered bytes, not 36.
        anchors = [Anchor(0, 100, 20, 0), Anchor(8, 108, 16, 1)]
        chains = chain_anchors(anchors, min_score=1.0)
        self.assertEqual(chains[0].covered_query_bytes, 20)
        self.assertEqual(chains[0].score, 20.0)

    def test_each_anchor_is_used_at_most_once(self):
        anchors = [Anchor(0, 100, 16, 0), Anchor(40, 140, 16, 1),
                   Anchor(41, 141, 16, 2), Anchor(80, 180, 16, 3)]
        chains = chain_anchors(anchors)
        used = [a for c in chains for a in c.anchors]
        self.assertEqual(len(used), len(set(used)))

    def test_anchors_below_the_score_floor_are_dropped(self):
        anchors = [Anchor(0, 100, 6, 0)]
        self.assertEqual(chain_anchors(anchors, min_score=8.0), [])
        self.assertEqual(len(chain_anchors(anchors, min_score=1.0)), 1)

    def test_correspondences_enumerate_matched_bytes_and_one_abutting_point(self):
        chain = Chain(anchors=(Anchor(0, 700, 4, 0), Anchor(8, 708, 4, 1)), score=8.0)
        got = correspondences(chain)
        matched = [c for c in got if c.kind == MATCHED]
        self.assertEqual([c.query_pos for c in matched], [0, 1, 2, 3, 8, 9, 10, 11])
        self.assertEqual([c.corpus_pos for c in matched], [700, 701, 702, 703,
                                                           708, 709, 710, 711])
        self.assertEqual([c.query_pos for c in got if c.kind == ABUTTING], [4, 12])

    def test_region_clustering_groups_nearby_chains(self):
        a = Chain(anchors=(Anchor(0, 1000, 16, 0),), score=16.0)
        b = Chain(anchors=(Anchor(40, 1500, 16, 1),), score=16.0)
        c = Chain(anchors=(Anchor(80, 900_000, 16, 2),), score=16.0)
        clusters = cluster_regions([a, b, c], gap=2000)
        self.assertEqual([len(x) for x in clusters], [2, 1])


class ForkTests(unittest.TestCase):
    def test_rank_and_mass_of_the_actual_next_byte(self):
        hist = {ord('e'): 50, ord(' '): 40, ord('a'): 6, ord('z'): 4}
        got = fork_outcome(hist, ord('a'))
        self.assertTrue(got["present"])
        self.assertEqual(got["rank"], 3)
        self.assertAlmostEqual(got["mass"], 6 / 100)
        self.assertAlmostEqual(got["top1_mass"], 0.5)
        self.assertEqual(got["distinct"], 4)
        self.assertEqual(got["total"], 100)

    def test_absent_byte_is_an_outcome_not_a_missing_cell(self):
        got = fork_outcome({ord('e'): 7}, ord('q'))
        self.assertFalse(got["present"])
        self.assertIsNone(got["rank"])
        self.assertEqual(got["mass"], 0.0)

    def test_empty_histogram_has_no_fork(self):
        self.assertEqual(fork_outcome({}, ord('e'))["total"], 0)


if __name__ == '__main__':
    unittest.main()
