// The accepted snapshot remains frozen; P20 transfers the manuscript and adds
// four bibliography entries. Exact approved source hashes retain both boundaries.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {createHash} from 'node:crypto';
export function checkCanonicalSource(root) {
  const baseline=JSON.parse(fs.readFileSync(path.join(root,'paper/tmlr/canonical-baseline.json'),'utf8'));
  const source=fs.readFileSync(path.join(root,'paper/main.typ'));
  assert.equal(createHash('sha256').update(source).digest('hex'),baseline.mainTypSha256,'Unapproved canonical manuscript drift');
  const bibliography=fs.readFileSync(path.join(root,'paper/refs.bib'));
  assert.equal(createHash('sha256').update(bibliography).digest('hex'),baseline.refsBibSha256,'Unapproved canonical bibliography change');
  execFileSync('git',['diff','--exit-code',baseline.acceptedSnapshot,'--','paper/build.sh'],{cwd:root});
  const transferred=execFileSync('git',['show',baseline.p21BaseCommit+':paper/refs.bib'],{cwd:root});
  assert(bibliography.equals(transferred),'Bibliography must match approved P20 transfer');
}
