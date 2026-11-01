"""Direct-SA oracle for the exact same xsa query records."""
from bisect import bisect_left, bisect_right
import json
import numpy as np
from direct_sa import suffix_array
from index_pipeline import OUT

corpus=(OUT/'corpus/corpus.bin').read_bytes()
sa=suffix_array(np.frombuffer(corpus,dtype=np.uint8))
patterns=(OUT/'patterns.bin').read_bytes().split(b'\x1e')[:-1]
answers=[json.loads(line) for line in (OUT/'query.stdout').read_text().splitlines()]
checks=[]
for i,(pat,answer) in enumerate(zip(patterns,answers)):
    key=lambda p:corpus[int(p):int(p)+len(pat)]
    lo=bisect_left(sa,pat,key=key); hi=bisect_right(sa,pat,key=key)
    expected=sorted(map(int,sa[lo:hi])); actual=sorted(answer['offsets'])
    checks.append(dict(pattern_id=i,expected_offsets=expected,actual_offsets=actual,match=expected==actual))
(OUT/'direct_sa_queries.json').write_text(json.dumps(dict(status='failed' if any(not c['match'] for c in checks) else 'passed',checks=checks),indent=2)+'\n')
print('direct-SA query matches',sum(c['match'] for c in checks),'/',len(checks))
