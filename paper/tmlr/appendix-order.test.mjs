import test from 'node:test';
import assert from 'node:assert/strict';
import {verifyAppendixOrder} from './appendix-order.mjs';
const title = 'A Development history and source register';
const tex = '\\bibliography{refs}\n\\clearpage\\appendix';

test('appendix-order: physical reference pages and final entry precede consolidated appendix', () => {
  assert.deepEqual(verifyAppendixOrder(`Body\fReferences\fLast DOI\f${title}`, tex, 'Last DOI'),
    {referencesPage: 2, finalReferencePage: 3, appendixPage: 4});
});
test('appendix-order: rejects old appendix-before-references PDF and TeX orders', () => {
  assert.throws(() => verifyAppendixOrder(`${title}\fReferences`), /physical page/);
  assert.throws(() => verifyAppendixOrder(`References\f${title}`,
    '\\appendix\\bibliography{refs}'), /TeX references/);
});
test('appendix-order: rejects missing headings, same-page appendix, and bibliography spill', () => {
  assert.throws(() => verifyAppendixOrder(title), /One References heading/);
  assert.throws(() => verifyAppendixOrder('References'), /One Appendix A heading/);
  assert.throws(() => verifyAppendixOrder(`References\n${title}`), /physical page/);
  assert.throws(() => verifyAppendixOrder(`References\f${title}\nLast DOI`, tex, 'Last DOI'), /Final bibliography/);
  assert.throws(() => verifyAppendixOrder(`References\f${title}`, tex, 'Last DOI'), /Final bibliography/);
});
test('appendix-order: rejects multiple appendices or bibliographies', () => {
  const text = `References\f${title}`;
  assert.throws(() => verifyAppendixOrder(text, tex + '\\appendix'), /One consolidated/);
  assert.throws(() => verifyAppendixOrder(text, tex + '\\bibliography{refs}'), /One bibliography/);
});
