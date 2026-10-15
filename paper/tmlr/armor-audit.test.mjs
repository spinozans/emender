import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {verifyArmor, liveScopes, captionScopes} from './armor-audit.mjs';
const read = p => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const source = read('../main.typ'), captions = read('../figures/expressivity_1p3b.typ');
const remove = (text, phrase) => text.replaceAll(new RegExp(phrase.split(/\s+/).map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s+'), 'g'), '');
test('P21 current source keeps P20 live scopes and findings', () => {
  assert.equal(verifyArmor(source, captions).liveScopes, 21);
});
test('P21 rejects retired expectation armor', () => {
  for (const armor of ['The negative transfer is:', 'The 8M S5 separation does NOT replicate',
    'reverses the predicted ordering', 'Non-claim', 'not a strawman']) {
    assert.throws(() => verifyArmor(source + '\n' + armor, captions), /Retired armor/);
  }
});
for (const phrase of liveScopes) test(`P21 rejects loss of live scope: ${phrase}`, () => {
  const mutated = remove(source, phrase);
  assert.notEqual(mutated, source);
  assert.throws(() => verifyArmor(mutated, captions), /boundary missing/);
});
for (const phrase of captionScopes) test(`P21 rejects caption loss: ${phrase}`, () => {
  const mutated = remove(captions, phrase);
  assert.notEqual(mutated, captions);
  assert.throws(() => verifyArmor(source, mutated), /Caption boundary missing/);
});
test('P21 rejects return of retired claim-table and predecessor appendices', () => {
  for (const label of ['<tab_claims>', '<sec:appendix_cohort>', '<sec:appendix_bpb>']) {
    assert.throws(() => verifyArmor(source + label, captions), /Retired structure/);
  }
});
