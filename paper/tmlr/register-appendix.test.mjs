import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {headingLevel} from './heading-level.mjs';
import {verifyRegisterAppendix} from './register-appendix.mjs';
const read = p => fs.readFileSync(new URL(p, import.meta.url), 'utf8');
const tex = read('main.tex');
const pdf = read('validation/submission-text.txt');
const aux = read('submission.aux');

test('convert: preserve markup depth and explicit heading level', () => {
  for (const depth of [1, 2, 3]) assert.equal(headingLevel({depth}), depth);
  for (const level of [1, 2, 3]) assert.equal(headingLevel({level}), level);
  assert.equal(headingLevel({level: 3, depth: 2}), 3);
  assert.equal(headingLevel({}), 1);
});
test('convert: P20 register has one level-1 heading and zero subsections', () => {
  const {headings} = JSON.parse(read('conversion-inventory.json'));
  const register = headings.slice(headings.findIndex(h => h.title === 'Development history and source register'));
  assert.equal(register.length, 1);
  assert.equal(register[0].level, 1);
});
test('register-appendix: built source categories, scopes, label and references resolve', () => {
  const result = verifyRegisterAppendix(pdf, tex, aux);
  assert.equal(result.subsections, 0);
  assert.equal(result.evidence.length, 4);
  assert.equal(result.resolvedReferences, 0);
});
test('register-appendix: rejects stale labels and wrong destination pages', () => {
  assert.throws(() => verifyRegisterAppendix(pdf, tex, aux.replace('{sec:appendix}{{A}', '{sec:appendix}{{E}')), /resolve to A/);
  assert.throws(() => verifyRegisterAppendix(pdf, tex, aux.replace(/(\{sec:appendix\}\{\{A\}\{)\d+/, '$11')), /heading page/);
  assert.throws(() => verifyRegisterAppendix(pdf, tex, aux + '\nsec:appendix_cohort'), /Historical appendix/);
});
test('register-appendix: rejects old separate appendices and subsection hierarchy', () => {
  assert.throws(() => verifyRegisterAppendix(pdf, tex + '\n\\section{Systems}', aux), /top-level/);
  assert.throws(() => verifyRegisterAppendix(pdf, tex + '\n\\subsection{Cohort}', aux), /zero subsections/);
  assert.throws(() => verifyRegisterAppendix(pdf.replace('A Development history', 'B Development history'), tex, aux), /One Appendix/);
});
test('register-appendix: rejects loss or relocation of source evidence and scopes', () => {
  assert.throws(() => verifyRegisterAppendix(pdf.replace('paper/main.typ', '') + ' paper/main.typ', tex, aux), /missing evidence/);
  assert.throws(() => verifyRegisterAppendix(pdf.replace('actual E97 and GDN2 arms', ''), tex, aux), /missing evidence/);
  assert.throws(() => verifyRegisterAppendix(pdf.replace('current E97/GDN2 pair', ''), tex, aux), /scope missing/);
  assert.throws(() => verifyRegisterAppendix(pdf + '\nFormal development.', tex, aux), /Unique register/);
});
test('register-appendix: rejects stale rendered appendix references', () => {
  for (const pointer of ['Section A', 'Section A.4', 'Section F']) {
    assert.throws(() => verifyRegisterAppendix(pointer + '\n' + pdf, tex, aux), /reference.*target/);
  }
});
