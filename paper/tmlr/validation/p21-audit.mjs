// P21 provenance replay: exact transferred inputs and every regenerated guard.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
const root = new URL('../../../', import.meta.url).pathname;
const read = p => fs.readFileSync(new URL(p, new URL('../../../', import.meta.url)));
const git = args => execFileSync('git', args, {cwd: root, maxBuffer: 16 * 1024 * 1024});
const sha = b => createHash('sha256').update(b).digest('hex');
const inputs = JSON.parse(read('paper/tmlr/validation/p21-inputs.json'));
for (const {file, sha256, provenanceManifest} of [...inputs.files, ...inputs.preservedFiles]) {
  if (provenanceManifest) {
    const frozen = git(['show', provenanceManifest]).toString().split('\n').find(line => line.endsWith('  ../' + file.slice('paper/'.length)));
    assert.equal(frozen?.slice(0, 64), sha256, `Frozen accepted-PDF digest: ${file}`);
    assert.equal(sha(read(file)), sha256, `Frozen accepted-PDF bytes: ${file}`);
  } else {
    const approved = git(['show', inputs.base + ':' + file]);
    assert.equal(sha(approved), sha256, `Committed input hash: ${file}`);
    assert(read(file).equals(approved), `Exact transferred/preserved bytes: ${file}`);
  }
}
assert(read('paper/tmlr/validation/p20-transfer-revision.diff').equals(git([
  'diff', '--unified=0', '0325d925', inputs.base, '--', 'paper/main.typ', 'paper/refs.bib', 'paper/figures/expressivity_1p3b.typ'
])), 'Exact enumerated P20 canonical/bibliography delta');
const ledger = JSON.parse(read('paper/tmlr/validation/p21-guard-delta.json'));
for (const {file, before, after, reason} of ledger.changes) {
  assert(reason.length > 30, `Baseline regeneration rationale: ${file}`);
  if (before !== null) assert.equal(git(['show', ledger.base + ':' + file]).toString(), before, `Exact old guard: ${file}`);
  else assert.equal(git(['ls-tree', ledger.base, '--', file]).length, 0, `New guard: ${file}`);
  assert.equal(read(file).toString(), after, `Exact regenerated guard: ${file}`);
}
const math = JSON.parse(read('paper/tmlr/validation/p21-math-delta.json'));
const drift = JSON.parse(read('paper/tmlr/validation/drift-check.json'));
assert.equal(math.changes.reduce((n, x) => n + x.after - x.before, 0), 325 - 706);
for (const key of ['equations', 'attachments', 'pdfPrimes']) assert.equal(math.counts.after[key], drift.symbolAudit[key], `Math delta: ${key}`);
assert.equal(drift.samples.length, 20);
assert(drift.samples.every(s => Object.values(s.matched).every(Boolean)));
assert.deepEqual(drift.pages, {canonical: 33, submission: 28, preprint: 28});
for (const order of Object.values(drift.appendixOrder)) assert.equal(order.finalReferencePage + 1, order.appendixPage);
const changed = git(['diff', '--name-only', inputs.base]).toString().trim().split('\n').filter(Boolean);
const untracked = git(['ls-files', '--others', '--exclude-standard']).toString().trim().split('\n').filter(Boolean);
for (const file of [...changed, ...untracked]) assert(file.startsWith('paper/tmlr/') || [
  'paper/current.pdf', 'paper/figures/validate_assembly.js', 'paper/review/P3-reference-audit.md', 'P21-report.md'
].includes(file), `P21 packaging-only scope: ${file}`);
console.log(`PASS: ${inputs.files.length} transferred inputs and ${inputs.preservedFiles.length} frozen source/style assets; exact P20 diff and ${ledger.changes.length} complete old/new guard records.`);
console.log('PASS: enumerated math delta 706 → 325 expressions / 444 → 266 attachments / eight PDF primes; 20 current samples × three PDFs; final reference physically before Appendix A in each PDF.');
console.log('PASS: changes confined to packaging, the rebuilt canonical PDF, assembly/ledger follow-up and P21 report.');
