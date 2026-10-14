// P13 moves the source baseline by operator directive; the accepted snapshot
// remains frozen. Only the approved mathlib4 bibliography amendment may differ.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {createHash} from 'node:crypto';
export function checkCanonicalSource(root) {
  const baseline=JSON.parse(fs.readFileSync(path.join(root,'paper/tmlr/canonical-baseline.json'),'utf8'));
  const source=fs.readFileSync(path.join(root,'paper/main.typ'));
  assert.equal(createHash('sha256').update(source).digest('hex'),baseline.mainTypSha256,'Unapproved canonical manuscript drift');
  execFileSync('git',['diff','--exit-code',baseline.acceptedSnapshot,'--','paper/build.sh'],{cwd:root});
  const original=execFileSync('git',['show','eb2b9636:paper/refs.bib'],{cwd:root,encoding:'utf8'});
  const before='  note         = {Lean 4 project/library documentation; Lean 4.26.0 toolchain; accessed 2026-10-14},';
  const after='  year         = {2026},\n  note         = {Lean 4 project/library documentation; Lean 4.26.0 toolchain; accessed 2026-10-14; citation year is the access year, not a publication year},';
  assert.equal(original.split(before).length,2,'Access-year amendment must match exactly once');
  assert.equal(fs.readFileSync(path.join(root,'paper/refs.bib'),'utf8'),original.replace(before,after),'Unapproved canonical bibliography change');
}
