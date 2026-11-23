"""Paired-horizon historical reach (measures available candidates, not model choices)."""
from collections import Counter
import json
import sys
from pathlib import Path
from .k15_bookmark_joint import EXE, Stream, flagship_identity, digest, BookmarkSession
from .k15_historical_port import historical_hits
from .k15_recovery_panel import panel


def scan(text, horizon, trace):
    c=Counter()
    for start in range(0,1024,horizon):
        raw=text[start:start+horizon]; s=BookmarkSession(EXE)
        try:
            for stop in range(16,horizon,16):
                assert s.rpc('append '+raw[stop-16:stop].hex())['visible']==stop
                informative=alternate=False
                for depth in (3,12):
                    p=raw[stop-depth:stop]; hits=historical_hits(s.rpc,p,stop)
                    want=[i for i in range(stop-depth) if raw[i:i+depth]==p]
                    assert hits==want
                    replies=[s.rpc(f'byte {i+depth}')["byte"] for i in hits]
                    assert replies==[raw[i+depth] for i in want]
                    informative |= any(b!=raw[stop-1] for b in replies)
                    alternate |= len(set(replies))>1
                    if trace is not None: trace.append(dict(start=start,stop=stop,depth=depth,hits=hits,bytes=replies))
                c.update(sites=1,informative=int(informative),alternate=int(alternate))
        finally: s.close()
    return c


def main(out):
    ident=flagship_identity(); assert ident['manifest_sha256']=='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'
    assert digest(EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    stream=Stream(ident['source'],ident['manifest']); used=set(); out.mkdir(exist_ok=False)
    with (out/'reach.jsonl').open('x') as f:
        for split in ('train','calibration'):
            for n,(doc,text) in enumerate(panel(stream,split)):
                assert doc not in used; used.add(doc)
                for h in (128,1024):
                    trace=[] if not n else None
                    counts=scan(text,h,trace)
                    f.write(json.dumps(dict(split=split,doc=doc,h=h,counts=counts,trace=trace))+'\n')

if __name__=='__main__': main(Path(sys.argv[1]))
