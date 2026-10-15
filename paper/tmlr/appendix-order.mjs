// Verify physical PDF pages, not merely the generated TeX command order.
import assert from 'node:assert/strict';
export function verifyAppendixOrder(text, tex = null, finalReference = null) {
  const rawPages = text.split('\f');
  const pages = rawPages.map(p => p.replace(/\s+/g, ' ').trim());
  const references = rawPages.findIndex(p => /^References\s*$/m.test(p));
  const appendix = pages.findIndex(p => p.includes('Development history and source register'));
  assert(references >= 0, 'References heading missing from PDF');
  assert(appendix > references, 'Appendix must begin on a physical page after references');
  let finalReferencePage = null;
  if (finalReference) {
    finalReferencePage = pages.findIndex(p => p.includes(finalReference));
    assert(finalReferencePage >= references && finalReferencePage < appendix,
      'Final bibliography entry must precede the appendix in the PDF');
    finalReferencePage++;
  }
  if (tex !== null) {
    assert.equal((tex.match(/\\appendix\b/g) || []).length, 1, 'One consolidated appendix');
    assert.equal((tex.match(/\\bibliography\{refs\}/g) || []).length, 1, 'One bibliography');
    assert(tex.indexOf('\\bibliography{refs}') < tex.indexOf('\\appendix'),
      'TeX references must precede appendix');
  }
  return {referencesPage: references + 1, finalReferencePage, appendixPage: appendix + 1};
}
