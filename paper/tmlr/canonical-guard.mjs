// Only the supervisor-approved mathlib4 access-year amendment may differ.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
export function checkCanonicalSource(root) {
  execFileSync('git',['diff','--exit-code','eb2b9636','--','paper/main.typ','paper/build.sh'],{cwd:root});
  const original=execFileSync('git',['show','eb2b9636:paper/refs.bib'],{cwd:root,encoding:'utf8'});
  const before='  note         = {Lean 4 project/library documentation; Lean 4.26.0 toolchain; accessed 2026-10-14},';
  const after='  year         = {2026},\n  note         = {Lean 4 project/library documentation; Lean 4.26.0 toolchain; accessed 2026-10-14; citation year is the access year, not a publication year},';
  assert.equal(original.split(before).length,2,'Access-year amendment must match exactly once');
  assert.equal(fs.readFileSync(path.join(root,'paper/refs.bib'),'utf8'),original.replace(before,after),'Unapproved canonical bibliography change');
}
