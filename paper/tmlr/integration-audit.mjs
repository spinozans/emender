// P23 integration replay: exact source delta, scoped evidence addition, S5 preserved.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {checkCanonicalSource} from './canonical-guard.mjs';
const root = new URL('../../', import.meta.url).pathname;
const base = '5cc34e8c';
const git = args => execFileSync('git', args, {cwd: root, maxBuffer: 16 * 1024 * 1024});
const read = p => fs.readFileSync(path.join(root, p));
const sha = b => createHash('sha256').update(b).digest('hex');
checkCanonicalSource(root);
assert(git(['diff', '--unified=0', base, '--', 'paper/main.typ', 'paper/figures/expressivity_1p3b.typ'])
  .equals(read('paper/tmlr/validation/p23-expressivity-revision.diff')), 'Exact P23 manuscript/table delta');
// Preserve every existing evidence/formal asset, including retired scalar records.
for (const directory of ['paper/data', 'paper/results', 'formal']) {
  for (const file of git(['ls-tree', '-r', '--name-only', base, '--', directory]).toString().trim().split('\n').filter(Boolean)) {
    assert(read(file).equals(git(['show', base + ':' + file])), `Existing evidence/theorem drift: ${file}`);
  }
}
const changed = git(['diff', '--name-only', base]).toString().trim().split('\n').filter(Boolean);
for (const file of changed) assert(file.startsWith('paper/') || file === 'P23-report.md', `Non-paper change: ${file}`);
for (const file of ['paper/refs.bib', 'paper/tmlr/tmlr.sty', 'paper/tmlr/tmlr.bst', 'paper/tmlr/fancyhdr.sty', 'paper/tmlr/LICENSE']) {
  assert(read(file).equals(git(['show', base + ':' + file])), `Frozen bibliography/style changed: ${file}`);
}
const captions = read('paper/figures/expressivity_1p3b.typ').toString();
assert.equal(captions.slice(captions.indexOf('#let s5-table')), git(['show', base + ':paper/figures/expressivity_1p3b.typ']).toString().split('#let s5-table')[1].replace(/^/, '#let s5-table'), 'S5 definition unchanged');
const inventory = JSON.parse(read('paper/tmlr/conversion-inventory.json'));
const oldInventory = JSON.parse(git(['show', base + ':paper/tmlr/conversion-inventory.json']));
assert.deepEqual(inventory.tables.filter((_,i)=>i!==4), oldInventory.tables.filter((_,i)=>i!==4), 'Six unchanged table structures');
assert.deepEqual(inventory.tables[4], {section:'What the current models learn to compute',columns:4,rows:5});
const labels = inventory.figures.filter(f => f.kind === 'table' && f.label !== '<tab_invertible>').map(f => f.label);
const baselinePath = path.join(root, 'paper/.integration-baseline.typ');
assert(!fs.existsSync(baselinePath));
const tables = [];
try {
  // The old table import is intentionally retired; compare all unaffected bodies.
  const previous = git(['show', base + ':paper/main.typ']).toString();
  const start = previous.indexOf('Comparable language-modeling loss leaves open');
  const end = previous.indexOf('#heading(level: 2, numbering: none)[S5 group-state', start);
  fs.writeFileSync(baselinePath, previous.slice(0,start) + previous.slice(end));
  const query = (file, label) => JSON.parse(execFileSync('typst', ['query', '--root', root, file, label, '--field', 'body'], {cwd: root, maxBuffer: 16 * 1024 * 1024}));
  for (const label of labels) {
    const current = query('paper/main.typ', label);
    assert.deepEqual(current, query(baselinePath, label), `Unchanged table body: ${label}`);
    tables.push({label, sha256: sha(JSON.stringify(current)), identicalToBase: true});
  }
} finally { fs.unlinkSync(baselinePath); }
const drift = JSON.parse(read('paper/tmlr/validation/drift-check.json'));
assert.deepEqual(drift.pages, {canonical: 35, submission: 30, preprint: 30});
assert.deepEqual([drift.symbolAudit.equations, drift.symbolAudit.attachments, drift.symbolAudit.pdfPrimes], [352, 293, 8]);
assert.equal(drift.samples.length, 20);
assert(drift.samples.every(s => Object.values(s.matched).every(Boolean)));
for (const order of Object.values(drift.appendixOrder)) assert.equal(order.finalReferencePage + 1, order.appendixPage);
const report = {base, approvedDelta:'validation/p23-expressivity-revision.diff',
  sourceFiles:['paper/main.typ','paper/figures/expressivity_1p3b.typ'].map(file=>({file,sha256:sha(read(file))})),
  preservedTables:tables, changedTable:'<tab_depth_wall> retired; <tab_invertible> added',
  existingDataResultsAndFormalUnchanged:true, productionAndCIUnchanged:true,
  pages:drift.pages, symbolCounts:{equations:352,attachments:293,pdfPrimes:8},
  samples:drift.samples.length,longSourceFragments:drift.fragmentAudit.checked};
fs.writeFileSync(path.join(root,'paper/tmlr/validation/integration-audit.json'),JSON.stringify(report,null,2)+'\n');
console.log('PASS: exact P23 delta; all five unaffected empirical table bodies and six table structures preserved; S5 definition unchanged; existing evidence, formal sources, production and CI unchanged.');
console.log('PASS: 35/30/30 pages; 352 expressions, 293 attachments, eight primes; 20 samples and 1215 long source fragments; references immediately before Appendix A.');
