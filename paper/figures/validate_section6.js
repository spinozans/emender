// CPU-only regression checks for main.typ's result-derived Section 6/CMA tables.
// Run from the worktree root: node paper/figures/validate_section6.js
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');
const resultRoot = 'paper/results/expressivity_1p3b/';
const json = path => JSON.parse(fs.readFileSync(path, 'utf8'));
const source = fs.readFileSync('paper/main.typ', 'utf8');
function text(node) {
  if (typeof node !== 'object' || node === null) return '';
  if (node.func === 'text') return node.text;
  return Object.values(node).map(value => Array.isArray(value)
    ? value.map(text).join('') : text(value)).join('');
}
function findTable(node) {
  if (!node || typeof node !== 'object') return undefined;
  if (node.func === 'table') return node;
  for (const value of Object.values(node)) {
    const found = Array.isArray(value) ? value.map(findTable).find(Boolean) : findTable(value);
    if (found) return found;
  }
}
function tableRows(label, columns) {
  const query = JSON.parse(execFileSync('typst',
    ['query', 'paper/main.typ', `<${label}>`, '--field', 'body'], { encoding: 'utf8' }));
  const cells = findTable(query).children.filter(child => child.func === 'cell').map(child => text(child.body));
  assert.equal(cells.length % columns, 0, label);
  return Array.from({ length: cells.length / columns }, (_, i) => cells.slice(i * columns, (i + 1) * columns));
}
const wallE97 = json(resultRoot + 'iterated-map/ft_e97_modular_quadratic_sweep.json');
const wallGdn2 = json(resultRoot + 'iterated-map/ft_gdn2_modular_quadratic_sweep.json');
const trainE97 = json(resultRoot + 'iterated-map/ft_e97_modular_quadratic.json');
const trainGdn2 = json(resultRoot + 'iterated-map/ft_gdn2_modular_quadratic.json');
const budget = json(resultRoot + 'iterated-map/budget.json');
const percent = value => (100 * value).toFixed(2) + '%';
assert.deepEqual(tableRows('tab_depth_wall', 4), Object.entries(wallE97.acc_vs_T).map(([length, result]) =>
  [length, `${Number(length) / trainE97.recipe.train_len}×`, percent(result.acc), percent(wallGdn2.acc_vs_T[length].acc)]));
assert.deepEqual(trainE97.recipe, trainGdn2.recipe, 'common recipe, cap, and stop rule');
assert.equal(budget.cap_steps, trainE97.recipe.steps_cap);
assert.notEqual(trainE97.steps_trained, trainGdn2.steps_trained, 'realized steps are not matched');
assert.ok(wallGdn2.acc_vs_T['1024'].acc > wallGdn2.random_baseline_acc, '8x is not random baseline');
assert.ok(trainGdn2.traj_curve.some(point => point.acc_vs_T['128'] > wallGdn2.acc_vs_T['128'].acc),
  'terminal accuracy is not a hard ceiling');
const control = json(resultRoot + 'iterated-map/phase1_8m_control_table.json').logistic_contractive_control;
assert.deepEqual(tableRows('tab_contractive_control', 6), Object.entries(control.arms).map(([arm, result]) =>
  [arm, (result.params / 1000000).toFixed(2), ...[128, 256, 512, 1024].map(length => result.pooled_acc_vs_T[length].mean_acc.toFixed(3))]));
const s5E97 = json(resultRoot + 'capacity/cap_e97_s5_sweep.json');
const s5Gdn2 = json(resultRoot + 'capacity/cap_gdn2_s5_sweep.json');
assert.deepEqual(tableRows('tab_s5_boundary', 3), Object.entries(s5E97.acc_vs_T).map(([length, result]) =>
  [length + (length === '512' ? '*' : ''), result.acc_all.toFixed(3), s5Gdn2.acc_vs_T[length].acc_all.toFixed(3)]));
const normalize = value => value.replace(/\*\*/g, '').replace(/\s+/g, ' ').trim();
const initialSummary = normalize(fs.readFileSync(resultRoot + 'SUMMARY.md', 'utf8'));
const capacitySummary = normalize(fs.readFileSync(resultRoot + 'capacity/SUMMARY.md', 'utf8'));
for (const negative of [
  'The 8M S5 separation does NOT replicate at 1.3B on this pair.',
  'in the capacity-exceeded regime on S5 there is no wall for either model — both decay in parallel toward the random baseline, and GDN2-MLP is ahead of E97 at every length.',
]) {
  assert.ok(normalize(source).includes(negative), 'negative retained verbatim');
  assert.ok(initialSummary.includes(negative) || capacitySummary.includes(negative), 'negative source exists');
}
// Parse RFC-style CSV fields independently of Typst, including escaped quotes.
function csv(path) {
  const data = fs.readFileSync(path, 'utf8');
  let rows = [], row = [], field = '', quoted = false;
  for (let i = 0; i < data.length; i++) {
    const character = data[i];
    if (character === '"') {
      if (quoted && data[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (character === ',' && !quoted) { row.push(field); field = ''; }
    else if (character === '\n' && !quoted) {
      row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = '';
    } else field += character;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const header = rows.shift();
  return rows.map(values => Object.fromEntries(header.map((name, i) => [name, values[i]])));
}
const evals = csv('paper/results/cmaes_burst_v2/cmaes_eval_summary.csv');
const gens = csv('paper/results/cmaes_burst_v2/cmaes_generation_summary.csv');
const sweeps = [...new Set(evals.map(row => row.sweep_id))];
const cmaRows = tableRows('tab_cma_sweeps', 7);
assert.equal(cmaRows.length, sweeps.length + 2, 'snapshot sweeps plus two carried-in winners');
for (const [i, sweep] of sweeps.entries()) {
  const rows = evals.filter(row => row.sweep_id === sweep);
  const fitness = rows.filter(row => row.cma_loss !== '').map(row => Number(row.cma_loss));
  const initial = rows.filter(row => row.generation === '0' && row.cma_loss !== '').map(row => Number(row.cma_loss));
  const status = rows[0].run_status === 'complete_usable' ? 'complete' : rows[0].run_status.startsWith('partial') ? 'partial' : 'failed';
  assert.deepEqual(cmaRows[i], [sweep.replaceAll('_', ' '), status, String(rows.length),
    String(gens.filter(row => row.sweep_id === sweep).length),
    initial.length ? Math.min(...initial).toFixed(4) : '—', Math.min(...fitness).toFixed(4),
    initial.length ? (Math.min(...initial) - Math.min(...fitness)).toFixed(4) : '—']);
}
assert.ok(cmaRows.slice(-2).every(row => row[1] === 'U3.1' && row[4] === '—' && row[6] === '—'));
assert.ok(source.includes('within-sweep gain not computable from retained artifacts'));
assert.ok(source.includes('throughput-budget smokes reported here\nare CUDA-only'));
assert.ok(!source.includes('TODO-EXPRESSIVITY-1P3B'));
console.log(`PASS: all ${Object.keys(wallE97.acc_vs_T).length} modular-quadratic, ${Object.keys(control.arms).length} control, ${Object.keys(s5E97.acc_vs_T).length} S5, and ${sweeps.length} CMA rendered rows match committed results; negatives, common-cap caveats, CUDA scope, and U3.1 checked.`);
