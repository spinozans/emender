"""Base-cut context construction and source-occurrence exclusion for the consult path.

The three rules under test are the 2026-11-10 hardening of the registered consult
path (RESULT_CONSULT_BOUNDARY_HARDENING.md):

1. a context is cut at the DOCUMENT BASE, `corpus[max(base, off-left) .. off]`,
   never by scanning for 0x1E, which remains only a belt-and-braces extra cut;
2. `off == base` yields the empty context, whose histogram is empty by rule;
3. the query position's own occurrence is removed from `width` and the branch
   counts BEFORE any support or width test, and `width` then means "occurrences
   other than the one at this position".

Every function exercised here is pure: identical inputs give identical outputs.
"""
import random
import unittest

from experiments.expressivity_tasks.x43_fork_resolved.c14_branch_serve import (
    END_BRANCH, SPINE_CAP, SPINE_POLICY, SPINE_POLICY_REGISTERED, SPINE_POLICY_SHORT,
    consult_context, exclude_occurrence, features, is_source_occurrence, select,
    spine_left)
from experiments.expressivity_tasks.x43_fork_resolved.test_c14_branch_serve import oracle


def placed(corpus, base, off, left=SPINE_CAP):
    """`consult_context` over a corpus-relative position, as the callers see it."""
    return consult_context(corpus[base:], base, off, left)


class BaseCutTests(unittest.TestCase):
    def test_context_never_crosses_the_document_base(self):
        corpus = b'AAAA' + b'BCDEFGH'
        base = 4
        # Four bytes into the document with a 1024-byte budget: the budget
        # reaches into the previous document, the base stops it.
        self.assertEqual(placed(corpus, base, base+4), b'BCDE')
        self.assertEqual(placed(corpus, base, base+4, left=2), b'DE')

    def test_document_shorter_than_left_contributes_only_its_bytes(self):
        corpus = b'0123456789' + b'ab'
        base = 10
        self.assertEqual(placed(corpus, base, base+2, left=1024), b'ab')
        self.assertEqual(placed(corpus, base, base+1, left=1024), b'a')

    def test_position_at_the_document_base_is_empty(self):
        corpus = b'0123456789' + b'ab'
        base = 10
        self.assertEqual(placed(corpus, base, base), b'')
        self.assertEqual(placed(corpus, base, base, left=1), b'')
        # The existing rule: an empty context is an empty histogram, never the
        # global corpus histogram, so every feature is zero.
        rows, rates, _ = select([placed(corpus, base, base)], oracle(corpus))
        self.assertEqual(rows[0]['context_bytes'], 0)
        self.assertEqual(rows[0]['context_count'], 0)
        self.assertEqual(rows[0]['branches'], [])
        self.assertFalse(rows[0]['coverage'])
        self.assertEqual(features(rows[0]), [0.]*260)
        self.assertEqual(rates['16'], dict(eligible=0, supported=0))

    def test_internal_separator_is_content_and_only_shortens(self):
        """An internal 0x1E is a content byte; the extra cut may only shorten.

        FreeLaw and PhilPapers carry such bytes (4,733 + 2,006 occurrences in the
        first 100 GB). Here the *base* cut is what keeps the context inside the
        document; the 0x1E cut is extra conservatism, not the boundary.
        """
        ambiguous = b'COURT\x1eFILING'
        previous = b'PRIOR-DOCUMENT'
        base = len(previous)
        corpus = previous + ambiguous
        # Base cut alone would return the whole document prefix ...
        self.assertEqual(consult_context(ambiguous, base, base+len(ambiguous), left=SPINE_CAP,
            ), b'FILING')
        # ... so confirm the pre-separator bytes are only lost to the extra cut by
        # asking for the same position in a document without the separator.
        plain = b'COURTFILING'
        self.assertEqual(consult_context(plain, base, base+len(plain)), plain)
        # The base cut still bounds below: nothing from the previous document.
        self.assertNotIn(b'PRIOR', placed(corpus, base, base+len(ambiguous), left=SPINE_CAP))

    def test_positions_inside_a_known_ambiguous_document_class(self):
        """A chat-log/legal-filing shape: many internal separators in one document.

        Positions near the document start are where a byte-scan boundary would
        have admitted the previous document; positions after an internal 0x1E are
        where the belt-and-braces cut loses content. Both stay inside the document.
        """
        body = (b'ubuntu 12.04 \x1e<user> hello\x1e<user> again\n')*8
        document = b'HEADER LINE\n' + body
        previous = b'NOT-MY-DOCUMENT'*4
        corpus = previous + document
        base = len(previous)
        for depth in (1, 2, 18, 19, 20, len(document)//2, len(document)-1, len(document)):
            context = placed(corpus, base, base+depth)
            window = document[max(0, depth-SPINE_CAP):depth]
            self.assertLessEqual(len(context), min(depth, SPINE_CAP))
            # The extra 0x1E cut can only shorten: the context is a suffix.
            self.assertTrue(window.endswith(context))
            self.assertTrue(document[:depth].endswith(context))
            self.assertNotIn(b'NOT-MY-DOCUMENT', context)
        # Off the base with a full left budget there is no previous-document byte,
        # and the belt-and-braces cut leaves exactly the post-separator content.
        late = placed(corpus, base, base+len(document))
        self.assertEqual(late, document[len(document)-SPINE_CAP:].rsplit(b'\x1e', 1)[-1])

    def test_geometry_is_rejected(self):
        for call in (lambda: consult_context(b'abc', 10, 9),
                     lambda: consult_context(b'abc', -1, 0),
                     lambda: consult_context(b'abc', 0, 4),
                     lambda: consult_context(b'abc', 0, 2, 0),
                     lambda: consult_context('abc', 0, 2)):
            with self.assertRaises((ValueError, TypeError)):
                call()

    def test_context_is_deterministic(self):
        document = b'xyz' + b'\x1e' + b'abcabcabc'
        first = consult_context(document, 500, 500+len(document), 8)
        self.assertEqual(first, consult_context(document, 500, 500+len(document), 8))
        self.assertEqual({consult_context(document, 500, 500+len(document), 8)
                          for _ in range(50)}, {first})

    def test_registered_spine_cap_is_the_default(self):
        self.assertEqual(SPINE_CAP, 1024)
        self.assertEqual(SPINE_POLICY, SPINE_POLICY_REGISTERED)
        self.assertEqual(spine_left(), 1024)
        # The 2026-11-10 amendment exists as a flag only: it refuses, so no
        # registered board can silently consume an unimplemented 8..16-byte spine.
        with self.assertRaises(NotImplementedError):
            spine_left(SPINE_POLICY_SHORT)
        with self.assertRaises(ValueError):
            spine_left('unregistered')


class ExclusionArithmeticTests(unittest.TestCase):
    def test_source_occurrence_as_the_only_one_leaves_width_zero(self):
        reply = dict(width=1, eof=0, branches=[(ord('Y'), 1)])
        excluded = exclude_occurrence(reply, ord('Y'))
        self.assertEqual(excluded, dict(width=0, eof=0, branches=[]))
        # The raw reply is pure input: exclusion never mutates it.
        self.assertEqual(reply, dict(width=1, eof=0, branches=[(ord('Y'), 1)]))

    def test_source_occurrence_is_removed_from_a_shared_histogram(self):
        reply = dict(width=3, eof=0, branches=[(ord('R'), 2), (ord('Q'), 1)])
        self.assertEqual(exclude_occurrence(reply, ord('R')),
            dict(width=2, eof=0, branches=[(ord('R'), 1), (ord('Q'), 1)]))
        # Emitted order is preserved and terminal 0x1E is an ordinary branch.
        terminal = dict(width=2, eof=0, branches=[(30, 2)])
        self.assertEqual(exclude_occurrence(terminal, 30), dict(width=1, eof=0, branches=[(30, 1)]))

    def test_end_branch_occurrence(self):
        reply = dict(width=2, eof=1, branches=[(ord('A'), 1)])
        self.assertEqual(exclude_occurrence(reply, END_BRANCH),
            dict(width=1, eof=0, branches=[(ord('A'), 1)]))
        with self.assertRaisesRegex(ValueError, 'absent'):
            exclude_occurrence(dict(width=1, eof=0, branches=[(ord('A'), 1)]), END_BRANCH)

    def test_absent_witness_is_an_error(self):
        reply = dict(width=2, eof=0, branches=[(ord('A'), 2)])
        for bad in (lambda: exclude_occurrence(reply, ord('B')),
                    lambda: exclude_occurrence(dict(width=0, eof=0, branches=[]), ord('A')),
                    lambda: exclude_occurrence(reply, ord('A'), 3),
                    lambda: exclude_occurrence(reply, ord('A'), 0),
                    lambda: exclude_occurrence(reply, 256),
                    lambda: exclude_occurrence(reply, 'A')):
            with self.assertRaises((ValueError, TypeError)):
                bad()

    def test_exclusion_is_deterministic(self):
        reply = dict(width=4, eof=1, branches=[(7, 3), (9, 1)])
        once = exclude_occurrence(reply, 7)
        self.assertEqual(exclude_occurrence(reply, 7), once)
        self.assertEqual({repr(exclude_occurrence(reply, 7)) for _ in range(25)}, {repr(once)})

    def test_source_occurrence_filter_rule(self):
        hit = dict(doc_id=17, offset_in_doc=4096, global_offset=1_000_000)
        self.assertTrue(is_source_occurrence(hit, 17, 4096))
        self.assertFalse(is_source_occurrence(hit, 17, 4095))
        self.assertFalse(is_source_occurrence(hit, 18, 4096))
        # An unattributable hit refuses: a silently unapplied exclusion leaks.
        with self.assertRaises(KeyError):
            is_source_occurrence(dict(global_offset=1_000_000), 17, 4096)


class ExcludedSelectTests(unittest.TestCase):
    def test_exclusion_is_applied_before_the_support_test(self):
        # The context occurs once in the corpus: at the query position itself.
        corpus = b'ZZZZabcdY'
        context = b'ZZZZabcd'
        raw, _, _ = select([context], oracle(corpus))
        self.assertEqual(raw[0]['context_bytes'], 8)
        self.assertEqual(raw[0]['context_count'], 1)
        self.assertEqual(raw[0]['branches'], [(ord('Y'), 1)])
        self.assertNotIn('exclude_source', raw[0])

        excluded, rates, _ = select([context], oracle(corpus), excludes=[ord('Y')])
        self.assertEqual(excluded[0]['context_bytes'], 0)
        self.assertEqual(excluded[0]['context_count'], 0)
        self.assertEqual(excluded[0]['branches'], [])
        self.assertFalse(excluded[0]['coverage'])
        self.assertEqual(features(excluded[0]), [0.]*260)
        self.assertEqual(rates['16'], dict(eligible=0, supported=0))
        self.assertTrue(excluded[0]['exclude_source'])

    def test_exclusion_keeps_support_from_other_positions(self):
        corpus = b'abcdQabcdR'
        raw, _, _ = select([b'abcd'], oracle(corpus))
        self.assertEqual(raw[0]['context_count'], 2)
        self.assertEqual(raw[0]['branches'], [(ord('Q'), 1), (ord('R'), 1)])
        excluded, rates, _ = select([b'abcd'], oracle(corpus), excludes=[ord('R')])
        self.assertEqual(excluded[0]['context_bytes'], 4)
        self.assertEqual(excluded[0]['context_count'], 1)
        self.assertEqual(excluded[0]['branches'], [(ord('Q'), 1)])
        self.assertEqual(excluded[0]['support_size'], 1)
        self.assertEqual(excluded[0]['distinct_branches'], 1)
        self.assertEqual(rates['16'], dict(eligible=0, supported=0))
        self.assertTrue(excluded[0]['exclude_source'])

    def test_asserted_absent_positions_keep_the_raw_counts(self):
        corpus = b'abcdQabcdR'
        raw, _, _ = select([b'abcd'], oracle(corpus))
        outside, _, _ = select([b'abcd'], oracle(corpus), excludes=[None])
        self.assertEqual(outside[0]['context_count'], 2)
        self.assertEqual(outside[0]['branches'], raw[0]['branches'])
        self.assertTrue(outside[0]['exclude_source'])
        with self.assertRaisesRegex(ValueError, 'cardinality'):
            select([b'abcd'], oracle(corpus), excludes=[])

    def test_exclusion_subtracts_exactly_the_source_occurrence(self):
        """Brute-force check: the excluded histogram is the raw one minus one position."""
        rng = random.Random(20261110)
        checked = 0
        for _ in range(12):
            block = bytes(rng.choice(b'abcde') for _ in range(120))
            corpus = block + block[:80] + b'Q' + block[:96]
            for depth in (3, 5, 8, 13, 40):
                p = len(corpus) - 1 - depth
                context = corpus[p-depth:p]
                raw, _, _ = select([context], oracle(corpus))
                excluded, _, _ = select([context], oracle(corpus), excludes=[corpus[p]])
                # Only comparable at equal depth: exclusion may legitimately
                # shorten the maximal censored context, which is the point.
                if not raw[0]['context_bytes'] or raw[0]['context_bytes'] != excluded[0]['context_bytes']:
                    continue
                expected = dict(width=raw[0]['context_count'], eof=raw[0]['eof'],
                                branches=raw[0]['branches'])
                self.assertEqual(excluded[0]['branches'],
                    exclude_occurrence(expected, corpus[p])['branches'])
                self.assertEqual(excluded[0]['context_count'], raw[0]['context_count']-1)
                self.assertEqual(excluded[0]['support_size'],
                    raw[0]['support_size'] - (corpus[p] != 30))
                checked += 1
        # Every position keeps a maximal context after exclusion: if one stopped
        # being comparable, either the corpora changed or exclusion shortened a
        # depth it should not have, and this exact count says so.
        self.assertEqual(checked, 60)

    def test_spans_cut_at_the_document_base(self):
        document = b'ab'*8
        base, depth = 10_000, 10
        spans_row, _, _ = select([document], oracle(document), spans=[(base, base+depth)], left=6)
        cut_row, _, _ = select([consult_context(document, base, base+depth, 6)], oracle(document))
        self.assertEqual(spans_row, cut_row)
        self.assertEqual(spans_row[0]['context_bytes'], 6)
        self.assertTrue(spans_row[0]['censored'])
        with self.assertRaisesRegex(ValueError, 'cardinality'):
            select([document], oracle(document), spans=[])
        with self.assertRaises(ValueError):
            select([document], oracle(document), spans=[(base, base+depth)], excludes=[None, None])


if __name__ == '__main__':
    unittest.main()
