// Regression checks for the three P15 author rulings, not full-stack equivalence.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const root = new URL('../../', import.meta.url);
const read = p => fs.readFileSync(new URL(p, root), 'utf8');
const source = read('paper/main.typ');

test('Algorithm 1 keeps exact norms and states production regularization separately', () => {
  assert.ok(source.includes('k_h &<- "silu"(k_h) / norm("silu"(k_h))_2'));
  assert.ok(source.includes('q_h &<- "silu"(q_h) / norm("silu"(q_h))_2'));
  assert.ok(source.includes('1. k <- silu(k) / norm2(silu(k))\n2. q <- silu(q) / norm2(silu(q))'));
  assert.equal(source.match(/production regularizes with `norm\+1e-6`/g).length, 1);
  assert.ok(source.includes('idealized for nonzero addresses'));
  assert.ok(source.includes('`elman/models/e88_fused.py:239-242`'));
});

test('Algorithms 1 and 2 define abstract cores with a resolved G2 limitation reference', () => {
  assert.ok(source.includes('Algorithms 1 and 2 present the abstract core; the released baselines add projection, normalization, gate-domain, and residual machinery whose full-stack binding is carried by limitation G2 (#ref(<sec:limitations>)).'));
  assert.ok(source.includes('= Limitations <sec:limitations>'));
  assert.ok(source.includes('*G2 — state-dependent gates are unmodeled.*'));
});

test('O1 consistently names tracker realization and leaves saturation Theorem F1 intact', () => {
  // P16b imports the reader-first realization target and future-work synopsis.
  assert.equal(source.match(/\bO1\b/g).length, 14);
  assert.ok(source.includes('*O1: realizing the group tracker.*'));
  assert.ok(source.includes('recurrent $tanh$ tracker (O1)'));
  assert.ok(source.includes('Open item O1 — recurrent $tanh$-tracker realization unavailable.'));
  for (const path of ['paper/main.typ', 'paper/OUTLINE.md',
    'paper/results/RESULTS_INVENTORY.md', 'paper/figures/validate_assembly.js']) {
    assert.doesNotMatch(read(path), /\bF1\b/, path);
  }
  assert.ok(read('paper/data/math-notation-spec.md').includes('Theorem F1, *saturation insensitivity*'));
});
