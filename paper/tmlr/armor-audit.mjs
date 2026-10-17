// P21: live claims and scopes follow the P20 reader-first transfer.
import assert from 'node:assert/strict';
const norm = s => s.replace(/\s+/g, ' ').trim();
export const liveScopes = [
  'Each arm has one training seed.',
  'training logs; held-out tokenizer-invariant bpb and downstream evaluation of the released pair are pending',
  'with a small difference whose sign varies by averaging window',
  'GDN2 is ahead of E97 at every reported length, and both curves decline toward the random baseline',
  "GDN2's attempted length-512 curriculum stage was OOM-infeasible even at sub-batch 1, after one completed optimizer step",
  'reaches 0.7918, GDN 0.3552, and raw-write M²RNN-CMA 0.2157',
  'these are selected checkpoints, not equal-step comparisons',
  'It does not identify which shortcut, if any, the trained models used',
  'The cross-rate comparison is observational with a protocol confound',
  'This scope is the primary matched cells, not every metric or post-seven-driver comparison',
  'We call this marginal-like loss behavior, not an identified internal marginal-solution mechanism or successful deep recurrence',
  'These findings establish neither architecture superiority in either direction nor a universal expressivity separation',
  'fixed-weight raw-write resource class, over abstract real-valued gate/key/state signatures',
  'Exact non-equality holds on that real-valued witness for each finite $k$',
  'The complexity separation TC#super[0] $!=$ NC#super[1] remains open, and no result here depends on it',
  '*G1: the cost of generating token-dependent actions.*',
  '*G2: state-dependent parameter schedules.*',
  '*G3: realizing group actions in the constrained affine channel.*',
  '*O1: realizing the group tracker.*',
  'supply total search effort and within-search improvement for these two configurations (Open item U3.1)',
  'the stated hold theorem assumes $lambda >= 1$, while production decay lies below one',
  'error, latency, and closed-loop stability across operating regimes',
  'Its final recorded rank-0 training-loss average is 2.2798 nats/token over the last 100 logging windows',
  'These are measurements from one production segment',
  'Its 100,000-update stage-local floor is calibrated against the successful one-step replay, not a guarantee of later learning',
  'All six learned solutions stopped by the registered stagnation rule at the initial four-driver stage, so no later stage was entered',
  'but only through position 2 in seed 2 (4/512 at position 3)',
  'E97 best seeds 1/2 each have one terminal hit, unlike their final checkpoints',
  'This is a finding about learned solutions under this training regime and protocol, not an architectural capacity limit or impossibility result, nor an identified internal mechanism'
];
export const captionScopes = [
  'Mean and sample SD over three paired seeds',
  'with validation-only checkpoint selection frozen before test',
  'S5 capacity/long-horizon accuracy on the released pair',
  'GDN2 completed one optimizer step at length 512',
  'OOM-infeasible even at sub-batch 1',
  'these counts are not final scores or independent test results'
];
export function verifyArmor(source, captions) {
  const s = norm(source), c = norm(captions);
  for (const retired of [/negative transfer/i, /does\s+(?:\*not\*|NOT|not)\s+replicate/i,
    /reverses the predicted ordering/i, /S_5\$ prediction fails/i, /not a strawman/i, /Non-claim/]) {
    assert(!retired.test(s + ' ' + c), `Retired armor returned: ${retired}`);
  }
  for (const phrase of liveScopes) assert(s.includes(phrase), `Live measured/formal boundary missing: ${phrase}`);
  for (const phrase of captionScopes) assert(c.includes(phrase), `Caption boundary missing: ${phrase}`);
  // The P20 transfer replaces the claim table and historical appendices with prose.
  for (const retired of ['<tab_claims>', '<sec:appendix_cohort>', '<sec:appendix_bpb>']) {
    assert(!s.includes(retired), `Retired structure returned: ${retired}`);
  }
  return {liveScopes: liveScopes.length, captionScopes: captionScopes.length, currentPairS5: 'finding', lineage: 'P20 reader-first transfer'};
}
