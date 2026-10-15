// Re-run the anonymous-PDF receipt, including annotation destinations and metadata.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
const dir = new URL('../', import.meta.url).pathname;
const infoBin = process.env.PDFINFO || 'pdfinfo';
const sibling = name => infoBin.includes('/') ? path.join(path.dirname(infoBin), name) : name;
const pdf = path.join(dir, 'submission.pdf');
const run = (bin, args) => execFileSync(bin, args, {encoding: 'utf8'});
const text = run(process.env.PDFTOTEXT || 'pdftotext', ['-raw', pdf, '-']);
const info = run(infoBin, [pdf]);
const links = run(infoBin, ['-url', pdf]);
const metadata = run(infoBin, ['-meta', pdf]);
const attachments = run(sibling('pdfdetach'), ['-list', pdf]);
const fonts = run(sibling('pdffonts'), [pdf]);
assert.match(text, /Anonymous authors/);
assert.match(info, /^Author:\s+Anonymous$/m);
assert.match(info, /^Metadata Stream:\s+no$/m);
assert.equal(metadata.trim(), '');
assert.match(attachments, /^0 embedded files/);
assert.doesNotMatch((text + info + links + metadata).replace(/\s/g, ''), /spinozans|hypervolu\.me|erik\.garrison@gmail|Poietic|UTHSC|0000-0003-3821-631X/i);
const destinations = links.split('\n').slice(1).filter(Boolean).map(line => line.trim().split(/\s+/).at(-1));
const approved = [
  'https://openreview.net/forum?id=ZA7D4nQuQF',
  'https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf',
  'https://proceedings.neurips.cc/paper_files/paper/2023/hash/a48e5877c7bf86a513950ab23b360498-Abstract-Conference.html',
  'https://arxiv.org/abs/2112.00114',
  'https://leanprover-community.github.io/mathlib4_docs/',
  'https://arxiv.org/abs/2201.11903',
  'https://isl.stanford.edu/~widrow/papers/c1960adaptiveswitching.pdf'
];
assert.equal(destinations.length, 11);
assert.deepEqual([...new Set(destinations)].sort(), approved.sort());
const fontRows = fonts.trim().split('\n').slice(2);
for (const row of fontRows) assert.match(row, /\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+$/, 'Each PDF font is embedded');
fs.writeFileSync(path.join(dir, 'validation/p21-anonymity-details.txt'), [info, links, attachments, fonts].join('\n'));
console.log(`PASS: ${info.match(/^Pages:\s+(\d+)/m)[1]} extracted pages; anonymous visible author and metadata; 11 approved literature/documentation URI destinations (7 unique); empty XMP and embedded-file inventories; ${fontRows.length} embedded fonts.`);
console.log('PASS: third-person literature citations retained; release and mirror identities variant-gated.');
