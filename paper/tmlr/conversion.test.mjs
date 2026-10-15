// P21 exact structure pins follow the approved P20 manuscript, including Frontier.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {checkCanonicalSource} from './canonical-guard.mjs';
const read = p => fs.readFileSync(new URL(p, import.meta.url));
const inventory = JSON.parse(read('conversion-inventory.json'));
const tex = read('main.tex').toString();
test('convert: P20 section order, captions, tables, algorithms and amsthm counts', () => {
  assert.equal(inventory.headings.length, 63);
  assert.deepEqual(inventory.headings.filter(h => h.level === 1).map(h => h.title), [
    'Introduction', 'Background', 'Computing with a nonlinear memory',
    'Making sequential updates practical', 'Language modeling at 1.3 billion parameters',
    'What the current models learn to compute', 'What the update rules establish',
    'Repeated nonlinear maps in the wild', 'Related Work', 'Open questions', 'Conclusion',
    'Next tests for the current pair', 'Future Work', 'Development history and source register'
  ]);
  assert.deepEqual(inventory.tables.map(t => [t.columns, t.rows]), [[2,11],[3,14],[4,6],[4,5],[4,8],[3,7],[5,6]]);
  assert.equal((tex.match(/\\caption\{/g) || []).length, 9);
  assert.equal((tex.match(/\\begin\{tabularx\}/g) || []).length, 7);
  assert.equal((tex.match(/\\begin\{verbatim\}/g) || []).length, 5);
  assert.deepEqual(inventory.theoremEnvironments.map(t => t.environment), [
    'theorem','theorem','theorem','theorembullet','theorembullet','theorembullet',
    'theorembullet','theorem','theorembullet','theorem','theorembullet','theorembullet','theorembullet'
  ]);
});
test('convert: exact current figure/table labels and byte-identical Frontier PNG', () => {
  assert.deepEqual(inventory.figures.map(f => f.label), ['<fig_taxonomy>','<fig_arch>',
    '<tab_pair_identities>','<tab_matched_loss>','<tab_frontier_anchors>','<fig_frontier_segment>',
    '<tab_depth_wall>','<tab_s5_boundary>','<tab_s5>']);
  assert(read('assets/frontier_segment_5134243.png').equals(read('../figures/frontier_segment_5134243.png')));
  assert.deepEqual(fs.readdirSync(new URL('assets/', import.meta.url)).sort(), [
    'figure-1.pdf', 'figure-2.pdf', 'frontier_segment_5134243.png'
  ]);
  for (const fragment of ['513,013,841,920', '2.2798', '6.320', '72 s', 'CommaPile mainmix']) assert(tex.includes(fragment), fragment);
});
test('convert: all 50 current citations, official fonts and exact copied bibliography', () => {
  assert.equal(new Set(inventory.citations).size, 50);
  assert(read('refs.bib').equals(read('../refs.bib')));
  assert(tex.includes('\\newunicodechar{Ž}{\\v{Z}}'));
  assert(tex.includes('\\newunicodechar{²}{\\textsuperscript{2}}'));
  assert(!/\\(?:fontsize|small|footnotesize|scriptsize)\b/.test(tex));
  assert.equal(inventory.anonymizedURLs.length, 3);
});
test('canonical-guard: exact approved P20 source and bibliography', () => {
  checkCanonicalSource(new URL('../../', import.meta.url).pathname);
});
for (const [file, message] of [['main.typ', /Unapproved canonical manuscript drift/],
  ['refs.bib', /Unapproved canonical bibliography change/]]) {
  test(`canonical-guard: rejects a one-byte change in ${file}`, () => {
    const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'p21-guard-'));
    try {
      fs.mkdirSync(path.join(temporary, 'paper/tmlr'), {recursive: true});
      fs.writeFileSync(path.join(temporary, 'paper/tmlr/canonical-baseline.json'), read('canonical-baseline.json'));
      for (const input of ['main.typ', 'refs.bib']) fs.writeFileSync(path.join(temporary, 'paper', input), read('../' + input));
      fs.appendFileSync(path.join(temporary, 'paper', file), ' ');
      assert.throws(() => checkCanonicalSource(temporary), message);
    } finally { fs.rmSync(temporary, {recursive: true, force: true}); }
  });
}

test('convert: notation definitions receive two-thirds of the table width', () => {
  const notation = tex.slice(tex.indexOf('\\subsection*{Notation}'), tex.indexOf('\\subsection*{Linear-state and nonlinear-state recurrence}'));
  assert(notation.includes('hsize=0.66667\\hsize'));
  assert(notation.includes('hsize=1.33333\\hsize'));
  assert(notation.indexOf('hsize=0.66667') < notation.indexOf('hsize=1.33333'));
  assert(notation.includes('Meaning and source-name correspondence'));
});
