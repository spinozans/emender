// P18: live claim scopes, measured boundaries, and current-pair finding voice.
import assert from 'node:assert/strict';
const norm = s => s.replace(/\s+/g, ' ').trim();
export function verifyArmor(source, captions) {
  const s = norm(source), c = norm(captions);
  for (const retired of [/negative transfer/i, /does\s+(?:\*not\*|NOT|not)\s+replicate/i,
    /reverses the predicted ordering/i, /S_5\$ prediction fails/i,
    /not a strawman/i, /Non-claim/]) {
    assert(!retired.test(s + ' ' + c), `Retired armor returned: ${retired}`);
  }
  const table = source.slice(source.indexOf('The table below'), source.indexOf(') <tab_claims>'));
  assert(table.includes('columns: (1.45fr, 2.05fr, 3.3fr)'), 'Claim table has exactly three columns');
  assert(table.includes('[*Claim*], [*Evidence*], [*Scope*]'), 'Claim / Evidence / Scope header');
  for (const phrase of [
    'One E97 run, single seed; a viability demonstration at this scale',
    'Training-log loss at exactly matched tokens; held-out bpb for the 150 B pair is pending',
    'at 8 M both nonlinear updates plateau below ceiling at length',
    'different successful-seed subsets',
    'Constrained affine-channel realizability remains open (G3)',
    'TC#super[0] versus NC#super[1] is unresolved',
    'unequal realized step counts',
    "GDN2's long-length modular-quadratic plateau remains above random"
  ]) assert(norm(table).includes(phrase), `Claim-row boundary missing: ${phrase}`);
  for (const phrase of [
    '99.96% versus 76.09% at trained length, and 76.07% versus 53.36%',
    'matched 150.79-billion-token cutoff the smoothed loss gap between the two arms is 0.010 nats/token',
    'with the sign of the gap depending on the averaging window',
    'both models\' accuracy declines toward random with length, and GDN2 leads E97 at every tested length',
    "GDN2's attempted length-512 curriculum stage was OOM-infeasible even at sub-batch 1, after one completed optimizer step",
    'At $S_5$ training length the Emender reaches 0.79, GDN 0.36',
    'The current modular-quadratic comparison shares a cap and plateau rule but realizes 2,500 versus 2,200 steps',
    'These are terminal stop weights; best-validation checkpoints are a separate comparison',
    'the reported mean is conditional on that subset',
    'This synthetic model excludes the actual dense right-multiplication $H W$ and feature-network costs',
    'No strict separation of TC#super[0] and NC#super[1] is known or assumed here',
    'G1 — no per-token circuit-size or depth lower bound',
    'G2 — state-dependent gates are unmodeled',
    'G3 — constrained affine/reflection realizability is open',
    'Open item O1 — recurrent $tanh$-tracker realization unavailable',
    'not uniformly bounded away from zero',
    'the hold theorem assumes $lambda >= 1$, whereas production decay is below one',
    'Feedback-surrogate use additionally requires stability and safety validation'
  ]) assert(s.includes(phrase), `Live measured/formal boundary missing: ${phrase}`);
  for (const phrase of [
    'with unequal realized training: E97 stopped after #train-e97.steps_trained steps and GDN2 after #train-gdn2.steps_trained',
    'plateau, above the declared #percent(wall-gdn2.random_baseline_acc) random baseline',
    'S5 capacity/long-horizon accuracy on the released pair',
    'GDN2 completed one optimizer step at length 512',
    'OOM-infeasible even at sub-batch 1'
  ]) assert(c.includes(phrase), `Caption boundary missing: ${phrase}`);
  return {claimColumns: 3, measuredAndFormalBoundaries: 'preserved', currentPairS5: 'finding, not replication expectation'};
}
