// P17 C5: check the built PDF's hierarchy and all register evidence references.
import assert from 'node:assert/strict';
const titles = [
  'Predecessor-cohort systems measurements',
  'Downstream capability panels: a null, on the predecessor cohort',
  'Lineage of the E63 → E88 → E97 experimental program',
  'The predecessor 1.3 B cohort (E88 era)',
  'Held-out bits-per-byte measurement (predecessor cohort)'
];
const norm = s => s.normalize('NFKC').replace(/\s+/g, ' ').trim();
const evidence = [
  ['22,200', '7,492', '15.7%', '8,248'],
  ['2,049-item', '2,048-item', 'Table 10', '0.345', '0.373', '0.358', '0.353'],
  ['E63', 'E88', 'E97', 'step 8,400'],
  ['0.966', '0.961', '0.973'],
  ['Common Pile', '0.981', '0.963', '0.973']
];
const pointers = [
  'E88 is the developmental predecessor (Section A).',
  'Predecessor-cohort occupancy, MFU, and the linear-scan kernel comparison are recorded in Section A.',
  'Predecessor-cohort downstream capability panels are recorded in Section A.'
];
export function verifyRegisterAppendix(text, tex, aux) {
  const appendixTex = tex.slice(tex.indexOf('\\appendix'));
  assert.equal((tex.match(/\\appendix\b/g) || []).length, 1, 'One register appendix activation');
  assert.equal((appendixTex.match(/\\section\{/g) || []).length, 1, 'Exactly one top-level register appendix');
  assert.equal((appendixTex.match(/\\subsection\{/g) || []).length, 5, 'Exactly five register subsections');
  const pdf = norm(text);
  const register = 'A Development history and source register';
  assert.equal(pdf.split(register).length - 1, 1, 'One Appendix A heading in built PDF');
  const start = pdf.indexOf(register);
  const positions = titles.map((title, i) => {
    const heading = `A.${i + 1} ${title}`;
    assert.equal(pdf.split(heading).length - 1, 1, `Built PDF missing unique ${heading}`);
    return pdf.indexOf(heading);
  });
  assert.deepEqual([...positions].sort((a, b) => a - b), positions, 'Register subsection order');
  assert(positions[0] > start, 'Subsections must follow Appendix A');
  const subsections = titles.map((title, i) => {
    const body = pdf.slice(positions[i], positions[i + 1] ?? pdf.length);
    for (const sample of evidence[i]) assert(body.includes(sample), `A.${i + 1} missing evidence: ${sample}`);
    return {number: `A.${i + 1}`, title, evidence: evidence[i]};
  });
  for (const pointer of pointers) assert(pdf.slice(0, start).includes(pointer), `Unresolved main-text pointer: ${pointer}`);
  const labels = {'sec:appendix': 'A', 'sec:appendix_cohort': 'A.4', 'sec:appendix_bpb': 'A.5'};
  const targets = Object.entries(labels).map(([label, number]) => {
    const entry = aux.split('\n').find(line => line.startsWith(`\\newlabel{${label}}`));
    assert(entry, `Missing built label: ${label}`);
    const [, actualNumber, page] = entry.match(/^\\newlabel\{[^}]+\}\{\{([^}]+)\}\{(\d+)\}/) || [];
    assert.equal(actualNumber, number, `Built label ${label} must resolve to ${number}`);
    const heading = number === 'A' ? register : `${number} ${titles[Number(number.slice(2)) - 1]}`;
    assert(norm(text.split('\f')[Number(page) - 1] || '').includes(heading), `Built label ${label} must target the PDF heading page`);
    return {label, number, page: Number(page)};
  });
  const expected = Object.fromEntries(Object.entries(labels).map(([label, number]) => {
    const count = tex.split(`Section~\\ref{${label}}`).length - 1;
    assert(count > 0, `Missing register label reference: ${label}`);
    return [number, count];
  }));
  const actual = {};
  for (const [, number] of pdf.matchAll(/\bSection (A(?:\.\d+)*)\b/g)) actual[number] = (actual[number] || 0) + 1;
  assert.deepEqual(actual, expected, 'Every rendered register reference must match the TeX target and count');
  assert(!/\bSection [B-F]\b/.test(pdf), 'Stale separate-appendix reference');
  return {appendix: 'A', subsections, targets, resolvedReferences: actual, mainTextPointers: pointers};
}
