import gzip,re,json,collections
from pathlib import Path
root=Path('experiments/expressivity_tasks/x43_fork_resolved')
p=root/'receipts/scalability_phase5/t1024_compiler_input.ptx.gz'
loc=None; counts=collections.Counter(); adds=collections.Counter(); samples=collections.defaultdict(list)
for line in gzip.open(p,'rt'):
 m=re.search(r'\.loc\s+\d+\s+(\d+)',line)
 if m: loc=int(m[1])
 if loc and re.match(r'\s*(?:@!?%p\d+\s+)?[a-z]+\.',line):
  counts[loc]+=1
  if re.search(r'add\..*f32',line):
   adds[loc]+=1
   if len(samples[loc])<8:samples[loc].append(line.strip())
src=(root/'c14_phase5_chunked_kernel.py').read_text().splitlines()
rows=[dict(line=l,source=src[l-1] if l<=len(src) else '',instructions=n,float_adds=adds[l],add_samples=samples[l]) for l,n in counts.most_common(25)]
out=root/'receipts/scalability_phase5_round2/static_profile.json'
out.write_text(json.dumps(dict(input=str(p),scope='Static compiler-input instruction attribution, not hardware counters or physical register liveness',rows=rows),indent=2)+'\n')
for row in rows[:12]:print(row)
