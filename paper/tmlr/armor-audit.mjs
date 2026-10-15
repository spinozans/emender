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
  'The common rule therefore yields different realized training durations',
  'Intermediate GDN2 trajectory probes had higher accuracy than its terminal checkpoint',
  'These single-seed trajectories motivate repeats that compare fixed training budgets and checkpoint-selection rules',
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
  'These are measurements from one production segment'
];
export const captionScopes = [
  'with unequal realized training: E97 stopped after #train-e97.steps_trained steps and GDN2 after #train-gdn2.steps_trained',
  'plateau, above the declared #percent(wall-gdn2.random_baseline_acc) random baseline',
  'S5 capacity/long-horizon accuracy on the released pair',
  'GDN2 completed one optimizer step at length 512',
  'OOM-infeasible even at sub-batch 1'
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
