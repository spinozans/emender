"""Exact outgoing token counts and followable loci without enumerating occurrences.

This small static oracle uses a suffix automaton, not the byte XSA or its page format.
Each state groups strings with the same occurrence end positions. A locus also needs
its length: several nested suffixes can have identical fanout and share a state.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

_END = -1  # structural separator, never accepted as an input token


def _token(t: int) -> int:
    if type(t) is not int or t < 0:
        raise ValueError("expected a nonnegative integer token ID")
    return t


@dataclass(slots=True)
class _State:
    longest: int
    link: int
    occurrences: int
    edges: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Locus:
    _mount: object = field(repr=False)
    _state: int
    length: int  # tokens, not bytes


@dataclass(frozen=True, slots=True)
class Edge:
    token: int
    count: int
    child: Locus  # local handle, not a read of the child's fanout


@dataclass(frozen=True, slots=True)
class Fanout:
    edges: tuple[Edge, ...]
    ends: int


@dataclass(frozen=True, slots=True)
class SpellingStep:
    # Unnormalized masses. A completed token is not a structural document END.
    continuing_bytes: dict[int, int]
    completed: tuple[Edge, ...]
    # Spelling finished earlier. The suffix has NOT been verified at its child.
    needs_successor: tuple[tuple[Edge, bytes], ...]
    document_ends: int


def spelling_step(
    answer: Fanout, spellings: Mapping[int, bytes], observed: bytes
) -> SpellingStep:
    """Filter an existing answer by the bytes since the lookup. No new index read.

    Return completed branches separately, including when one spelling is a prefix
    of another. Do not normalize these away, commit a segmentation, or speculate
    on successor nodes. Requires exact, context-independent token byte spellings.
    """
    if type(observed) is not bytes:
        raise ValueError("expected a byte prefix")
    next_byte: dict[int, int] = {}
    complete = []
    pending = []
    for edge in answer.edges:
        piece = spellings[edge.token]
        if type(piece) is not bytes or not piece:
            raise ValueError("a content token needs a nonempty exact byte spelling")
        if piece.startswith(observed):
            if len(piece) == len(observed):
                complete.append(edge)
            else:
                b = piece[len(observed)]
                next_byte[b] = next_byte.get(b, 0) + edge.count
        elif observed.startswith(piece):
            pending.append((edge, observed[len(piece) :]))
    return SpellingStep(
        next_byte, tuple(complete), tuple(pending), 0 if observed else answer.ends
    )


class TokenIndex:
    """Immutable token stream with a structural separator after every document."""

    def __init__(self, documents: Iterable[Iterable[int]]):
        self._mount = object()
        self.root = Locus(self._mount, 0, 0)
        states = [_State(0, -1, 0)]
        last = 0

        def append(t):
            # Standard online suffix-automaton construction. Only new prefix states
            # initially count an endpoint, never the clones used to split classes.
            nonlocal last
            p = last
            c = len(states)
            states.append(_State(states[p].longest + 1, 0, 1))
            while p >= 0 and t not in states[p].edges:
                states[p].edges[t] = c
                p = states[p].link
            if p >= 0:
                q = states[p].edges[t]
                if states[q].longest == states[p].longest + 1:
                    states[c].link = q
                else:
                    clone = len(states)
                    states.append(
                        _State(
                            states[p].longest + 1,
                            states[q].link,
                            0,
                            states[q].edges.copy(),
                        )
                    )
                    while p >= 0 and states[p].edges.get(t) == q:
                        states[p].edges[t] = clone
                        p = states[p].link
                    states[q].link = states[c].link = clone
            last = c

        for doc in documents:
            for t in doc:
                append(_token(t))
            append(_END)
        # Occurrences of a longer string also end occurrences of each true suffix.
        for v in sorted(
            range(1, len(states)), key=lambda i: states[i].longest, reverse=True
        ):
            states[states[v].link].occurrences += states[v].occurrences
        self._states = states

    def _check(self, h):
        if not isinstance(h, Locus) or h._mount is not self._mount:
            raise ValueError("foreign token-index locus")

    def follow(self, h: Locus, t: int) -> Locus | None:
        """Right extension only. A missing exact branch is not an observation or a backoff."""
        self._check(h)
        q = self._states[h._state].edges.get(_token(t))
        return None if q is None else Locus(self._mount, q, h.length + 1)

    def suffix(self, h: Locus) -> Locus:
        """Drop exactly one leading token, even within a shared end-position class."""
        self._check(h)
        if not h.length:
            return self.root
        s = h._state
        link = self._states[s].link
        n = h.length - 1
        return Locus(self._mount, link if n == self._states[link].longest else s, n)

    def advance(self, h: Locus, t: int) -> Locus:
        """Consume a real token. Return the longest indexed suffix of this history."""
        self._check(h)
        _token(t)
        s = h._state
        n = h.length
        while t not in self._states[s].edges:
            if not s:
                return self.root
            s = self._states[s].link
            n = self._states[s].longest
        return Locus(self._mount, self._states[s].edges[t], n + 1)

    def seek(self, prefix: Iterable[int]) -> Locus:
        """Get a reset document's longest indexed token suffix."""
        h = self.root
        for t in prefix:
            h = self.advance(h, t)
        return h

    def branch(self, h: Locus) -> Fanout:
        """All occurring next token IDs and multiplicities. Structural END stays separate."""
        self._check(h)
        edges = self._states[h._state].edges
        end = edges.get(_END)
        return Fanout(
            tuple(
                Edge(
                    t, self._states[q].occurrences, Locus(self._mount, q, h.length + 1)
                )
                for t, q in sorted(edges.items())
                if t != _END
            ),
            0 if end is None else self._states[end].occurrences,
        )
