// CPU-only assembly regression: section counters, reference targets, honest
// theorem scopes, limitations, inventory, and bibliography ledger coverage.
// Run from the worktree root: node paper/figures/validate_assembly.js
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');
const source = fs.readFileSync('paper/main.typ', 'utf8');
const normalized = source.replace(/\s+/g, ' ');
const labels = ['intro', 'background', 'arch', 'systems', 'lm', 'expressivity',
  'applications', 'formal', 'related', 'limitations', 'conclusion', 'predictions',
  'future_work', 'appendix', 'appendix_cohort', 'appendix_bpb'];
const probe = 'paper/figures/.assembly-query.typ';
assert.ok(!fs.existsSync(probe), 'do not overwrite a pre-existing probe');
fs.writeFileSync(probe, `#include "../main.typ"
#context [#metadata((
  sections: query(heading.where(level: 1)).filter(h => h.has("label")).map(h => (
    label: str(h.label), counter: counter(heading).at(h.location()),
    page: h.location().page(),
  )), pages: counter(page).final().first(),
)) <assembly-check>]
`);
let layout;
try {
  layout = JSON.parse(execFileSync('typst', ['query', '--root', '.', probe,
    '<assembly-check>', '--field', 'value', '--one'], { encoding: 'utf8' }));
} finally { fs.unlinkSync(probe); }
assert.deepEqual(layout.sections.map(h => h.label), labels.map(label => `sec:${label}`));
assert.deepEqual(layout.sections.slice(0, 13).map(h => h.counter),
  Array.from({ length: 13 }, (_, i) => [i + 1]));
assert.deepEqual(layout.sections.slice(13).map(h => h.counter), [[1], [2], [3]]);
assert.ok(!/§\d/.test(source), 'no stale literal manuscript numbers');
assert.ok(source.includes('#show: arkheion-appendices'));
for (const match of source.matchAll(/#ref\(<([^>]+)>\)/g)) {
  assert.ok(source.includes(`<${match[1]}>`), `resolved source label ${match[1]}`);
}
for (const phrase of [
  'Open item F1 — recurrent $tanh$-tracker realization unavailable.',
  'No strict separation of TC#super[0] and NC#super[1] is known or assumed here.',
  'No minimal parameter-count corollary is claimed.',
  'not uniformly bounded away from zero',
  'input $(lambda,k,v)$',
  'The formal E97 fold fixes $lambda$',
  'G1 — no per-token circuit-size or depth lower bound.',
  'G2 — state-dependent gates are unmodeled.',
  'G3 — constrained affine/reflection realizability is open.',
  'U3.1 — carried-in winner first-generation fitness unavailable.',
  'Single-seed 1.3 B evidence and protocol boundaries',
  'Untested application predictions',
  'TMLR',
  '150,793,420,800',
  'five optimizer steps',
  'not an unconditional four-seed mean',
]) assert.ok(normalized.includes(phrase), phrase);
for (const old of ['margin bounded away from zero', 'strictly inside NC#super[1]',
  'linear recurrence provably cannot maintain', 'explicit configuration of it realizes',
  'strictly more expressive than raw-write', "Section 7's theorems"]) {
  assert.ok(!normalized.includes(old), `overstatement absent: ${old}`);
}
const inventory = JSON.parse(fs.readFileSync('paper/data/lean-trust-gate-theorems.json', 'utf8'));
assert.equal(inventory.trust_gate_files.length, 17);
assert.equal(inventory.theorem_count, inventory.theorems.length);
const closure = execFileSync('bash', ['formal/lean/scripts/local_import_closure.sh'],
  { encoding: 'utf8' }).trim().split('\n').map(path => path.split('/formal/lean/')[1]);
assert.deepEqual(inventory.trust_gate_files, closure, 'inventory covers recursive closure');
for (const file of ['ElmanProofs/Activations/Lipschitz.lean',
  'ElmanProofs/Expressivity/E1HDefinition.lean']) {
  assert.ok(inventory.theorems.some(d => d.file === file), file);
}
for (const name of [
  'CompositionalDepthDuality.e97_compositional_depth_grows_in_time',
  'CompositionalDepthDuality.gdn_state_channel_depth_bounded_in_time',
  'S5GDNBridge.gdn_s5_realization_requires_per_token_action',
  'NDMRealizesS5.emender_realizes_s5_tracker',
  'MultiStepSeparation.emender_m2rnn_k_step_separation',
]) assert.ok(inventory.theorems.some(d => d.name === name), name);
// The table's wrapped hashes must preserve the exact released authorities.
const identity = JSON.parse(execFileSync('typst', ['query', 'paper/main.typ',
  '<tab_pair_identities>', '--field', 'body'], { encoding: 'utf8' }));
const hashes = [];
function collectRaw(node) {
  if (!node || typeof node !== 'object') return;
  if (node.func === 'raw' && node.text) hashes.push(node.text.replace(/\s/g, ''));
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) value.forEach(collectRaw);
    else collectRaw(value);
  }
}
collectRaw(identity);
for (const hash of [
  '0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2',
  'ffae61cfeeeff820f469f7d66dd31c60e48c455872ff9102e57b15ad75bad59b',
]) assert.ok(hashes.includes(hash), `rendered checkpoint SHA-256: ${hash}`);
const bib = fs.readFileSync('paper/refs.bib', 'utf8');
const keys = [...bib.matchAll(/@\w+\{([^,]+),/g)].map(m => m[1]);
const ledger = fs.readFileSync('paper/review/P3-reference-audit.md', 'utf8');
for (const key of keys) assert.ok(ledger.includes(`| \`${key}\` |`), key);
assert.equal((ledger.match(/^\| `[^`]+` \|/gm) || []).length, keys.length);
assert.ok(bib.includes('The Road Less Scheduled'));
assert.ok(bib.includes('Danieli, Federico'));
console.log(`PASS: 13 numbered body sections, 3 lettered appendices, ${layout.pages} pages; references, G1–G3/F1/U3.1, theorem scopes, 17-file inventory, and all ${keys.length} bibliography entries checked.`);
console.log(JSON.stringify(layout));
