// Verify physical PDF pages, not merely the generated TeX command order.
import assert from 'node:assert/strict';
export function verifyAppendixOrder(text, tex = null, finalReference = null) {
  const rawPages = text.split('\f');
  const pages = rawPages.map(p => p.replace(/\s+/g, ' ').trim());
  const references = rawPages.findIndex(p => /^References\s*$/m.test(p));
  const appendix = rawPages.findIndex(p => /^(?:APPENDIX )?A Development history and source register\s*$/m.test(p));
  assert.equal(rawPages.filter(p => /^References\s*$/m.test(p)).length, 1, 'One References heading');
  assert.equal(rawPages.filter(p => /^(?:APPENDIX )?A Development history and source register\s*$/m.test(p)).length, 1, 'One Appendix A heading');
  assert(references >= 0, 'References heading missing from PDF');
  assert(appendix > references, 'Appendix must begin on a physical page after references');
  let finalReferencePage = null;
  if (finalReference) {
    finalReferencePage = pages.findIndex(p => p.includes(finalReference));
    assert(finalReferencePage >= references && finalReferencePage === appendix - 1,
      'Final bibliography entry must end on the physical page immediately before the appendix');
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
