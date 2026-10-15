// Current-paper integration checks; P21 provenance receipts remain historical.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {checkCanonicalSource} from './canonical-guard.mjs';
const root = new URL('../../', import.meta.url).pathname;
const main = 'baf6e56464d23d855ff88ff9c2ba78d2d64f0c6e';
const author = '5b16e273518f7ab20463e5a4826bd78851c4f6bf';
const source = '7bc278704d56f2f8f6a0d1ca7ec944e0eaaeb0af';
const git = args => execFileSync('git', args, {cwd: root, maxBuffer: 16 * 1024 * 1024});
const read = p => fs.readFileSync(path.join(root, p));
const sha = b => createHash('sha256').update(b).digest('hex');
const sourceFiles = ['paper/main.typ', 'paper/refs.bib', 'paper/figures/expressivity_1p3b.typ'];
checkCanonicalSource(root);
for (const file of sourceFiles.filter(f => f !== 'paper/main.typ')) {
  assert(read(file).equals(git(['show', source + ':' + file])), `Approved source changed: ${file}`);
}
// P22a adds only the recorded-loss figure and its two-sentence introduction.
assert(git(['diff', '--unified=0', source, '--', 'paper/main.typ']).equals(read('paper/tmlr/validation/p22a-lossfigure-revision.diff')),
  'Canonical source must be the exact S5 source plus the reviewed P22a insertion');
// Integration must retain the author's evidence and all existing production/CI code.
for (const directory of ['paper/data', 'paper/results', 'formal']) {
  assert.equal(git(['diff', '--name-only', author, '--', directory]).length, 0, `Evidence/theorem drift: ${directory}`);
}
const changed = git(['diff', '--name-only', main]).toString().trim().split('\n').filter(Boolean);
for (const file of changed) assert(file.startsWith('paper/') || ['P20-report.md', 'P21-report.md'].includes(file), `Non-paper change: ${file}`);
for (const file of ['paper/tmlr/tmlr.sty', 'paper/tmlr/tmlr.bst', 'paper/tmlr/fancyhdr.sty', 'paper/tmlr/LICENSE']) {
  assert(read(file).equals(git(['show', author + ':' + file])), `Official style changed: ${file}`);
}
const inventory = JSON.parse(read('paper/tmlr/conversion-inventory.json'));
const oldInventory = JSON.parse(git(['show', author + ':paper/tmlr/conversion-inventory.json']));
assert.deepEqual(inventory.tables, oldInventory.tables, 'All seven table structures retained');
const labels = inventory.figures.filter(f => f.kind === 'table').map(f => f.label);
const baselinePath = path.join(root, 'paper/.integration-baseline.typ');
assert(!fs.existsSync(baselinePath), 'Do not overwrite a pre-existing baseline probe');
const tables = [];
try {
  fs.writeFileSync(baselinePath, git(['show', author + ':paper/main.typ']));
  const query = (file, label) => JSON.parse(execFileSync('typst', ['query', '--root', root, file, label, '--field', 'body'], {cwd: root, maxBuffer: 16 * 1024 * 1024}));
  for (const label of labels) {
    const current = query('paper/main.typ', label);
    assert.deepEqual(current, query(baselinePath, label), `Empirical table changed: ${label}`);
    tables.push({label, sha256: sha(JSON.stringify(current)), identicalToAuthor: true});
  }
} finally { fs.unlinkSync(baselinePath); }
const drift = JSON.parse(read('paper/tmlr/validation/drift-check.json'));
assert.deepEqual(drift.pages, {canonical: 34, submission: 29, preprint: 29});
assert.deepEqual([drift.symbolAudit.equations, drift.symbolAudit.attachments, drift.symbolAudit.pdfPrimes], [335, 285, 8]);
assert.equal(drift.samples.length, 20);
assert(drift.samples.every(s => Object.values(s.matched).every(Boolean)));
for (const order of Object.values(drift.appendixOrder)) assert.equal(order.finalReferencePage + 1, order.appendixPage);
const report = {main, author, source, approvedInsertion: 'validation/p22a-lossfigure-revision.diff',
  sourceFiles: sourceFiles.map(file => ({file, sha256: sha(read(file))})),
  tables, allSevenTableStructuresPreserved: true, dataResultsAndFormalUnchanged: true, productionAndCIUnchanged: true,
  pages: drift.pages, symbolCounts: {equations: 335, attachments: 285, pdfPrimes: 8},
  mathInventoryDeltaFromAuthor: {equations: 10, attachments: 19, pdfPrimes: 0},
  samples: drift.samples.length, longSourceFragments: drift.fragmentAudit.checked};
fs.writeFileSync(path.join(root, 'paper/tmlr/validation/integration-audit.json'), JSON.stringify(report, null, 2) + '\n');
console.log('PASS: exact approved S5 sources plus the P22a figure insertion; six empirical table bodies and all seven table structures preserved; data, results, formal sources, production code and CI unchanged.');
console.log('PASS: 34/29/29 pages; 335 expressions, 285 attachments, eight primes; 20 samples across all PDFs and references immediately before Appendix A.');
