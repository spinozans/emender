"""Trainable token-native integration; frozen serving reuses one current-weight bank.

Pointer mode returns a successor-token law directly (small vocab oracle). Vector mode
reads learned successor embeddings, so archive storage does NOT scale as rows × vocab.
"""

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import Answer, Local, exhaustive, gaussian_features, indexed, prepare
from .index import tokens
from .plan import Layout, Plan


@dataclass(frozen=True)
class Prediction:
    probabilities: torch.Tensor
    attention: Answer
    plan: Plan


class CorpusTokenLM(nn.Module):
    def __init__(
        self,
        vocab,
        dim,
        rank,
        *,
        tokenizer,
        leaf=64,
        feature_seed=5601,
        values="vector",
    ):
        super().__init__()
        if (
            not tokenizer
            or min(vocab, dim, rank, leaf) < 1
            or values not in ("pointer", "vector")
        ):
            raise ValueError("invalid token head")
        self.vocab = vocab
        self.dim = dim
        self.tokenizer = tokenizer
        self.leaf = leaf
        self.kind = values
        self.width = vocab if values == "pointer" else dim
        self.embed = nn.Embedding(vocab, dim)
        self.context = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.lm = nn.Linear(dim, vocab)
        self.skip = nn.Linear(dim, 1)
        if values == "vector":
            self.value = nn.Linear(dim, dim, bias=False)
            self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer("omega", gaussian_features(dim, rank, feature_seed))

    def _ids(self, seq):
        if seq and (min(seq) < 0 or max(seq) >= self.vocab):
            raise ValueError("token outside head vocabulary")
        return torch.tensor(seq, dtype=torch.long, device=self.embed.weight.device)

    def _states(self, ids):
        x = self.embed(ids)
        length = torch.arange(1, 1 + len(ids), dtype=x.dtype, device=x.device)[:, None]
        return (x + self.context(x.cumsum(0) / length)).tanh()

    def _rows(self, ids, states):
        v = (
            F.one_hot(ids[1:], self.vocab).to(states.dtype)
            if self.kind == "pointer"
            else self.value(self.embed(ids[1:]))
        )
        return self.key(states[:-1]), v

    def _archive(self, index, enabled=True):
        if self.tokenizer != index.tokenizer:
            raise ValueError("foreign tokenizer")
        k = self.embed.weight.new_empty((0, self.dim))
        v = k.new_empty((0, self.width))
        if enabled and len(index):
            ks = []
            vs = []
            for doc in index.documents:
                if len(doc) > 1:
                    ids = self._ids(doc)
                    key, value = self._rows(ids, self._states(ids))
                    ks.append(key)
                    vs.append(value)
            order = torch.tensor(index.order, device=k.device)
            k = torch.cat(ks)[order]
            v = torch.cat(vs)[order]
        return Layout(index.version, len(k), self.leaf), k, v, k.new_zeros(len(k))

    def _predict(
        self,
        index,
        requests,
        rows,
        bank,
        *,
        min_match,
        local,
        corpus,
        allow_skip,
        mode,
        backend,
    ):
        layout, K, V, b = rows
        live = []
        ls = []
        lv = []
        lengths = []
        ranges = []
        for source, seq in requests:
            index.check_live(source)
            seq = tokens(seq)
            if not seq:
                raise ValueError("need at least one observed token")
            ids = self._ids(seq)
            state = self._states(ids)
            live.append(state[-1])
            ranges.append(
                [index.frontier(seq).hit_range(min_match)]
                if corpus and mode == "indexed"
                else []
            )
            if local:
                a, c = self._rows(ids, state)
                ls.append(a)
                lv.append(c)
                lengths.append(len(a))
        if not live:
            raise ValueError("empty query batch")
        if (
            not allow_skip
            and not (corpus and len(K))
            and (not local or not all(lengths))
        ):
            raise ValueError("no supported next-token law; enable the base LM row")
        x = torch.stack(live)
        q = self.query(x)
        L = (
            Local(
                torch.cat(ls), torch.cat(lv), q.new_zeros(sum(lengths)), tuple(lengths)
            )
            if local
            else None
        )
        plan = Plan.build(layout, ranges)
        # Size offset is an explicit prior for the no-read row, not a post-softmax gate.
        null = None
        if allow_skip:
            null = (
                self.skip(x)[:, 0] + math.log(max(len(K) if corpus else 0, 1)),
                self.lm(x).softmax(-1)
                if self.kind == "pointer"
                else x.new_zeros((len(x), self.width)),
            )
        if mode == "exhaustive":
            ans = exhaustive(q, K, V, b, local=L, null=null, corpus=corpus)
        elif mode == "indexed":
            ans = indexed(
                q,
                prepare(*rows, self.omega) if bank is None else bank,
                plan,
                local=L,
                null=null,
                corpus=corpus,
                backend=backend,
            )
        else:
            raise ValueError("unknown read mode")
        prob = (
            ans.read
            if self.kind == "pointer"
            else self.lm(x + self.out(ans.read)).softmax(-1)
        )
        return Prediction(prob, ans, plan)

    def forward(
        self,
        index,
        requests,
        *,
        min_match=1,
        local=True,
        corpus=True,
        allow_skip=True,
        mode="indexed",
        backend="torch",
    ):
        """One update uses current encodings, not a stale sidecar. Inputs contain no labels."""
        return self._predict(
            index,
            requests,
            self._archive(index, corpus),
            None,
            min_match=min_match,
            local=local,
            corpus=corpus,
            allow_skip=allow_skip,
            mode=mode,
            backend=backend,
        )

    def freeze(self, index):
        return FrozenTokenLM(self, index)


class TokenPointer(CorpusTokenLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, values="pointer", **kwargs)


class FrozenTokenLM:
    @torch.no_grad()
    def __init__(self, model, index):
        self._model = deepcopy(model).eval().requires_grad_(False)
        self._index = index
        self._rows = self._model._archive(index)
        self._bank = prepare(*self._rows, self._model.omega)

    @torch.no_grad()
    def predict(
        self,
        requests,
        *,
        min_match=1,
        local=True,
        corpus=True,
        allow_skip=True,
        mode="indexed",
        backend="torch",
    ):
        return self._model._predict(
            self._index,
            requests,
            self._rows,
            self._bank,
            min_match=min_match,
            local=local,
            corpus=corpus,
            allow_skip=allow_skip,
            mode=mode,
            backend=backend,
        )
