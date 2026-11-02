import gzip, json, math, re
from pathlib import Path
p=Path(__file__).resolve().parent; o=p/'receipts/c14'
s=json.loads((o/'coverage.json').read_text()); timings=s['timings']
branch=next(t for t in timings if t['name']=='branches')
feature_path=o/'features.json'
features=json.loads(feature_path.read_bytes() if feature_path.exists() else gzip.decompress((o/'features.json.gz').read_bytes()))
maxctx=json.loads((o/'max_context.json').read_text()); present={r['sample'] for r in features['rows']}
for i,k in enumerate(maxctx):
    if i not in present:
        assert k==0
        features['rows'].append(dict(sample=i,context_bytes=0,counts=[0]*255,support_size=0,entropy=0.,coverage=False))
features['rows'].sort(key=lambda r:r['sample'])
if feature_path.exists(): feature_path.write_text(json.dumps(features))
n=len(features['rows'])
invalid=[r['sample'] for r in features['rows'] if not r.get('valid',True)]
secs=branch['seconds']; kernel=float(re.search(r'query_ms=([\d.]+)',(o/'branches.stderr').read_text())[1])/1000
rate=(n-len(invalid))/secs; krate=(n-len(invalid))/kernel
budget=dict(measured_positions=n,branch_patterns=branch['patterns'],rejected_features=invalid,successful_features=n-len(invalid),host_wall_seconds=secs,host_features_per_second=rate,host_kernel_seconds=kernel,host_kernel_features_per_second=krate,forty_million_branch_wall_seconds=40000000/rate,forty_million_branch_kernel_seconds=40000000/krate,required_rate=40000000/2700,all_measurement_wall_seconds=sum(t['seconds'] for t in timings),scope='Host throughput with default Rayon pool despite CLI -j 1. Not single-core throughput. Branch extraction only: maximal-context searches, decoding, projection and training transfer are extra costs. Linear extrapolation is not an executed 40M run.')
(o/'budget.json').write_text(json.dumps(budget,indent=2))
rates=s['support_rates']; q=s['max_context_quantiles']; exact=s['verbatim_256_rate']; agreement=s['branch_agreement_supported']
wilson={}
for name,rate_,num in [(f'support_{k}',v,10000) for k,v in rates.items()]+[('verbatim_256',exact,1000)]:
    z=1.96; den=1+z*z/num; center=(rate_+z*z/(2*num))/den; half=z*math.sqrt(rate_*(1-rate_)/num+z*z/(4*num*num))/den
    wilson[name]=[center-half,center+half]
(o/'intervals.json').write_text(json.dumps(wilson,indent=2))
result=f'''# C14 phase 1: Pile stream against Wikipedia

**Query-health gate failed:** two impossible branch counts were quarantined.
Coverage below is a provisional measurement against this binary, not an
end-to-end correctness certificate. C14 training remains blocked.

10,000 deterministic eligible Pile training-token boundaries, seed 1401. No
training, GPU allocation, or held-out scoring. Wikipedia MANIFEST SHA-256:
`{s['provenance']['wiki_manifest_sha256']}`. Rebuilt xsa at
`{s['provenance']['sxgc_head']}`; binary SHA in provenance.json.

| Left context bytes | Wikipedia support rate |
|---|---:|
'''+''.join(f'| {k} | {v:.4%} |\n' for k,v in rates.items())+f'''
Maximal context quantiles (bytes): `{json.dumps(q,sort_keys=True)}`.
At the 1024-byte cap: {s['right_censored_1024']} / 10,000 (right censored).
Observed next-token first-byte agreement at the maximal supported context:
{agreement:.4%}, conditional on nonempty context support. This is a measured
first-byte statistic, not a full-token accuracy or NLL prediction.

256-byte spans (128 bytes before / after each of the first 1000 sampled
boundaries): {exact:.4%} exact Wikipedia support. Bounded locate is deterministic
suffix-rank sampling of at most one hit per span, not unbiased occurrence or
document sampling. `validation.json` verifies every returned offset against
immutable corpus.bin bytes. The rate measures exact span overlap, not Wikipedia
source attribution. Wikipedia markup and Pile preprocessing differ.

This world has shared short English contexts but sparse long verbatim overlap;
near-disjoint and substantially overlapping are length dependent descriptions.
It is not legitimate to call these corpora disjoint, infer a document component
fraction, or treat C12's no-whole-task-copy property as established here.
`intervals.json` gives conditional binomial Wilson 95% intervals. Excluding short
documents/near-edge positions changes the population; no all-token estimate is
claimed. Rejected draws: {s['provenance']['rejected']}.

## Machinery and budget

AGC FFI cannot query SXI2. The batch Python bridge emits token-boundary-aligned
summary geometry. See C14_FEATURE_BRIDGE.md for the exact upstream handoff and
byte-30 omission. Features never consume Pile target/future bytes. Measurement
queries for observed next bytes and verbatim spans are separate diagnostics.

Branch enumeration: {n} positions, {branch['patterns']} searches;
{secs:.3f}s process wall, {kernel:.3f}s journaled query kernel;
{rate:.2f} features/s whole process, {krate:.2f}/s kernel on this host.
40M-position linear extrapolation: {budget['forty_million_branch_wall_seconds']/3600:.2f}h
whole-process branch work, {budget['forty_million_branch_kernel_seconds']/3600:.2f}h
kernel; context search/decoding/projection/transfer add cost. The training budget
needs 14,814.815 features/s. Neither CLI opening nor a cached persistent service
is free. Full measured workflow wall is {budget['all_measurement_wall_seconds']:.2f}s
for this sample, including coverage and locate diagnostics (not all needed for
production features). No 40M run was performed; cache effects and repetition
could change scaling. 260 float32 scalars need 41.6 GB/40M positions, while full
50,257-token vectors need 8.041 TB.

A performance trap: `-j 1` skips Rayon global-pool initialization in the current
CLI, so the actual default pool can use the whole 256-logical-CPU host. This is
host throughput, not single-core throughput. Materialized run columns are about
266 GB. CPU model/topology and affinity are receipted. No single-core benchmark
or leased GPU benchmark is claimed; the present machinery does not establish
a one-core online budget. The persistent batched API remains an upstream handoff.

## Residual risks and readiness

No complete BPE candidate distribution, EOS continuation model, document-seam
filter, held-out coverage measurement, isolated one-core benchmark, or 40M
feature cache is delivered. The 255-byte outgoing total omits separator byte 30;
context support can also include terminal occurrences. 1024 is a cap, not a
proof that no longer suffix matches. Training is blocked on freezing a complete
causal cache geometry and receipting its generation/transfer costs, not on any
unmeasured promise of speed. This phase registers an experiment, not an outcome.
'''
strata=json.loads((o/'content_strata.json').read_text())
result += '\n## Confirmed query failure: training blocked\n\nTwo extension queries returned UINT64_MAX (samples 2769 and 9057, appended byte 194). Their outgoing totals exceed the parent context counts. The receipt audit caught this; raw outputs and initial failure are preserved. The two feature rows are explicitly invalid, with no entropy or probability distribution. Aggregate coarse support queries and observed-next queries had no out-of-range counts, but this does not prove all remaining index answers correct. The source subtracts r-l without a guard; the underlying interval/rank cause remains unisolated. No upstream code was modified. The bridge now fails closed on impossible counts. Repair and independent upstream regression evidence are mandatory before any C14 training.\n'
result += '\n## Content versus formatting overlap\n\n'
result += 'All five supported centered 256-byte spans are spaces only: 0/1000 non-whitespace centered-span matches. This is not evidence of a 0.5% Wikipedia content component.\n\n'
result += '| Context bytes | Non-whitespace supported / non-whitespace eligible | Whitespace-only supported |\n|---|---:|---:|\n'
for k in ['16','64','256','1024']:
    row=strata[k]
    result += f"| {k} | {row['nonwhitespace_supported']} / {row['nonwhitespace_denominator']} | {row['whitespace_supported']} |\n"
result += '\nSupplemental bounded locate covers all 59 supported 256-byte contexts and all nine supported 1024-byte contexts, including two non-whitespace 1024-byte contexts. One is four-byte-alphabet encoding noise; the other is a basketball statistics table. Thus rare substantive long overlap exists, but formatting/repetition dominates many long matches. These selected hits are not an unbiased document-component estimate. Raw patterns, references, offsets and direct-byte checks are retained.\n'
(p/'RESULT_C14_PHASE1.md').write_text(result)
pr=f'''# C14 preregistration — Pile training, Wikipedia consultation

Registered after phase-1 coverage.json and validation.json, before training or
held-out inference. Train the attested pile24 GPT-2 BPE stream, consult only the
immutable Wikipedia index. This is no longer a guaranteed disjoint/no-copy task.
The coverage receipt shows 16/64/256/1024-byte support rates
{', '.join(f'{k}: {v:.4%}' for k,v in rates.items())}; 256-byte span overlap
{exact:.4%}; maximal-context next-byte agreement {agreement:.4%}.
These training-interior diagnostics calibrate predictions, not held-out outcomes.
All five centered-span matches are whitespace only; non-whitespace supported
contexts are 3264/9605, 118/9882, 23/9962 and 2/9990 at those lengths.
Rare substantive long overlap exists (including one 1024-byte statistics table),
but no measurable Wikipedia document-component fraction follows. Expect any
aggregate gain to be driven mainly by short-context lexical consultation, not
broad long-span copying; score non-whitespace strata separately.

P1 (directional, evaluated separately for each head): Wikipedia-conditioned
inputs reduce raw held-out full-vocabulary token NLL relative to the matched
dense control. No numeric gain extrapolated from C12: long-context support is
sparse and overlap/preprocessing differ. Predict the larger gain on supported,
next-byte-agreeing strata than zero-target strata; do not drop any targets.
P2: all-input feature dropout (independent Bernoulli keep=.5, no rescaling,
including length metadata) preserves device-off NLL within 0.02 nats/token of
its matched dense control. P3: calibration plus consultation improves final-test
NLL relative to both consultation alone and dense plus calibration. These are
falsifiable predictions, not accepted improvements. Report all signs and losses.

Use velocity 150M trainable matched-head candidates (Emender and transformer),
GPT-2 vocabulary 50,257, 2700 active training seconds per arm. Within each head,
run dense zero-input control and Wikipedia consultation with identical data order,
initialization seed 3401, batch/context, optimizer and trainable parameter count
(tolerance 0.1%, including input projection). Retain a matched zero-input projection
in the control. Keep dense cross-entropy over every token, no fork-only writes,
no target masks, hard routing, or self-containing Pile index. Use only preceding
text at actual training reset boundaries; no cross-document left context. Register
exact model configs, counts, data/cache hashes and effective steps/tokens before
launch. Report both active time and end-to-end feature cost and the off-mode loss.
No new hyperparameter search or held-out selection is authorized by this ledger.

Use a causal maximal-supported-Wikipedia context selector capped at 1024 bytes;
this deliberately replaces C12's transductive selector. Token-boundary summary
geometry is the intended bridge, not exact 50,257-token support. The phase-1
255-byte geometry is provisional: before training, freeze how separator/EOS,
empty contexts, reset boundaries and normalization are handled, check a complete
training/held-out sidecar, and receipt throughput/storage/transfer. The current
CLI lacks an efficient persistent Python API and produced two impossible UINT64_MAX
extension counts. Two feature rows were rejected; no distribution is fabricated.
Require upstream repair and independent count/branch regression evidence before training. Fail closed until this prerequisite
is met; no large training was run in phase 1.

Stacking (second half): freeze trained weights; split the held-out evaluation
stream into disjoint calibration and final-test halves before inference. Fit the
C13-style confidence/context/support gate only on calibration data; freeze it
before final test. Compare dense neural, consultation neural, dense+calibration,
and consultation+calibration, with enabled and all-input-zero consultation rows.
Exact-byte distributions are not full BPE distributions: forced next byte cannot
force a BPE token. Permit only validated normalized full-token continuation for
forced-token walks, otherwise report that C13 component unavailable and use a
separately frozen first-byte-feature calibration. Do not silently substitute a
byte probability for token probability. Report total token NLL and comparable
byte-normalized NLL, denominators, context-length/support/zero-target strata,
feature dependence, steps/tokens, extraction cost and failures. No causal
operator-only interpretation or universal unplugged robustness from one seed.
'''
(p/'PREREGISTRATION_C14.md').write_text(pr)
