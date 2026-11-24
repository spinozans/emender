"""Online bounded LZ factors and a committed occurrence address space."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    source: int | None
    kind: int  # exact expanded-content intern, document-local


class Factorizer:
    def __init__(self):
        self.raw = bytearray()
        self.rows = []
        self.types = {}
        self.start = 0
        self.matches = []
        self.ended = False

    def _commit(self, end):
        content = bytes(self.raw[self.start:end])
        kind = self.types.setdefault(content, len(self.types))
        self.rows.append(Span(self.start, end, min(self.matches, default=None), kind))
        self.start = end
        self.matches = []

    def observe(self, byte):
        if self.ended:
            raise ValueError('confirmed end')
        n = len(self.raw)
        self.raw.append(byte)
        before = len(self.rows)
        if self.start < n:
            extended = [p for p in self.matches if self.raw[p+n-self.start] == byte]
            if extended:
                self.matches = extended
            else:
                self._commit(n)
        if self.start == n:
            self.matches = [p for p in range(max(0,n-256), n) if self.raw[p] == byte]
        if not self.matches or n+1-self.start == 4:
            self._commit(n+1)
        return self.rows[before:]

    def end(self):
        if self.ended:
            raise ValueError('already ended')
        self.ended = True
        if self.start < len(self.raw):
            self._commit(len(self.raw))
            return self.rows[-1:]
        return []


def offline(raw):
    start = 0
    while start < len(raw):
        choices = [(k, -p) for p in range(max(0,start-256),start)
                   for k in range(1,min(4,len(raw)-start)+1)
                   if raw[p:p+k] == raw[start:start+k]]
        k, p = max(choices, default=(1, 1))
        yield start, start+k, None if p == 1 else -p
        start += k


class Index:
    def __init__(self, raw_session, phrase_session):
        self.parse = Factorizer()
        self.raw, self.ids = raw_session, phrase_session

    def _publish(self, rows):
        for row in rows:
            self.ids.rpc('append '+row.kind.to_bytes(4,'big').hex())
        return rows

    def observe(self, y):
        if self.parse.ended:
            raise ValueError("confirmed end")
        self.raw.rpc(f'append {y:02x}')
        rows = self._publish(self.parse.observe(y))
        if len(self.parse.raw) == 128:
            for session in (self.raw,self.ids):
                session.rpc('seal')
        return rows

    def end(self):
        rows = self._publish(self.parse.end())
        for session in (self.raw,self.ids):
            session.rpc('end')
        return rows

    def sources(self):
        f = self.parse
        if f.start != len(f.raw) or f.ended:
            return 0, []
        rows = f.rows
        n = len(rows)
        for depth in range(min(2,n),0,-1):
            suffix = b''.join(row.kind.to_bytes(4,'big') for row in rows[-depth:])
            ans = self.ids.rpc(f'locate {suffix.hex()} 0 {4*n} 16384')
            legal = [p//4+depth for p in ans['positions'] if p%4 == 0 and p//4+depth < n]
            if legal:
                return depth, legal[-4:][::-1]
        return 0, []


def scan_sources(f):
    if f.start != len(f.raw) or f.ended:
        return 0, []
    kinds = [s.kind for s in f.rows]
    for d in range(min(2,len(kinds)),0,-1):
        hits = [i for i in range(len(kinds)-1,d-1,-1) if kinds[i-d:i] == kinds[-d:]]
        if hits:
            return d, hits[:4]
    return 0, []


def targets(contents, future):
    """Offline only. KEEP targets at reached offsets; span exhaustion is forced."""
    lengths, keeps = [], []
    for content in contents:
        pairs = list(zip(content, future[:4]))
        length = next((i for i,(x,y) in enumerate(pairs) if x != y), len(pairs))
        lengths.append(length)
        keeps.append([int(i < length) for i in range(1,min(length+1,len(pairs)))])
    longest = max(lengths, default=0)
    return dict(start=None if not future or not contents else int(longest>0),
                best=[i for i,k in enumerate(lengths) if k == longest and longest],
                lengths=lengths, keep=keeps)
