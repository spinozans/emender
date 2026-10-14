// Source-named regression for recursive enumeration and fail-closed gates.
// Run: node formal/lean/scripts/test_local_import_closure.js
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'local-import-closure-'));
try {
  const scripts = path.join(tmp, 'scripts');
  fs.mkdirSync(scripts);
  for (const name of ['local_import_closure.sh', 'check_paper_core.sh',
    'check_trusted_no_placeholders.sh']) {
    fs.copyFileSync(path.join(__dirname, name), path.join(scripts, name));
    fs.chmodSync(path.join(scripts, name), 0o755);
  }
  fs.mkdirSync(path.join(tmp, 'ElmanProofs'));
  fs.mkdirSync(path.join(tmp, '.lake'));
  for (const name of ['Root', 'Direct', 'Recursive']) {
    fs.writeFileSync(path.join(tmp, 'ElmanProofs', name + '.lean'), '-- fixture\n');
  }
  fs.writeFileSync(path.join(tmp, '.lake', 'External.lean'), 'axiom external : True\n');
  fs.writeFileSync(path.join(tmp, 'lake'), `#!/usr/bin/env bash
set -euo pipefail
case "$(basename "$4")" in
  Root.lean) printf '%s\\n' "$PWD/ElmanProofs/Direct.lean" "$PWD/.lake/External.lean" ;;
  Direct.lean) printf '%s\\n' "$PWD/ElmanProofs/Recursive.lean" ;;
  Recursive.lean) [[ ! -f "$PWD/fail" ]] ;;
  *) exit 2 ;;
esac
`);
  fs.chmodSync(path.join(tmp, 'lake'), 0o755);
  const run = script => spawnSync('bash', [path.join(scripts, script),
    'ElmanProofs/Root.lean'], { cwd: tmp, encoding: 'utf8',
    env: { ...process.env, PATH: tmp + ':' + process.env.PATH } });
  let result = run('local_import_closure.sh');
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(result.stdout.trim().split('\n').map(p => path.basename(p)),
    ['Direct.lean', 'Recursive.lean', 'Root.lean']);
  result = run('check_paper_core.sh');
  assert.equal(result.status, 0, result.stderr);
  for (const token of ['sorry', 'admit', 'axiom hidden : True',
    'opaque hidden : True', 'native_decide']) {
    fs.writeFileSync(path.join(tmp, 'ElmanProofs/Recursive.lean'), token + '\n');
    assert.notEqual(run('check_paper_core.sh').status, 0, token);
  }
  fs.writeFileSync(path.join(tmp, 'ElmanProofs/Recursive.lean'), '-- fixture\n');
  fs.writeFileSync(path.join(tmp, 'fail'), '');
  assert.notEqual(run('local_import_closure.sh').status, 0, 'compiler failure');
  assert.notEqual(run('check_paper_core.sh').status, 0, 'gate fails closed');
  console.log('PASS: recursive imports, external exclusion, five forbidden tokens, compiler-error fail-closed');
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
