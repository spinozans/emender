// P21: the transferred register is one Appendix A with four source categories.
import assert from 'node:assert/strict';
const norm = s => s.normalize('NFKC').replace(/\s+/g, ' ').trim();
const categories = [
  ['Architecture development.', ['paper/main.typ', 'f17b3be20b395cef914cd7034e8691c997164230', 'E63-to-E97']],
  ['Exploratory task and search work.', ['experiments/expressivity_tasks/', 'experiments/grok_symmetric_width/', 'paper/ndmpapernotes.md', 'paper/results/cmaes_burst_v2/']],
  ['The present comparison.', ['two immutable model releases', 'actual E97 and GDN2 arms', 'paper/results/expressivity_1p3b/', 'result JSONs directly']],
  ['Formal development.', ['formal/lean/', 'specific statements used here']]
];
export function verifyRegisterAppendix(text, tex, aux) {
  const appendixTex = tex.slice(tex.indexOf('\\appendix'));
  assert.equal((tex.match(/\\appendix\b/g) || []).length, 1, 'One register appendix activation');
  assert.equal((appendixTex.match(/\\section\{/g) || []).length, 1, 'Exactly one top-level register appendix');
  assert.equal((appendixTex.match(/\\sub(?:sub)?section\{/g) || []).length, 0, 'Compact source register has zero subsections');
  const pdf = norm(text), register = 'A Development history and source register';
  assert.equal(pdf.split(register).length - 1, 1, 'One Appendix A heading in built PDF');
  const start = pdf.indexOf(register), body = pdf.slice(start);
  for (const scope of ['The empirical study in this paper is the current E97/GDN2 pair',
    'The present comparison is defined by its own model identities, data, and protocols']) {
    assert(body.includes(scope), `Register scope missing: ${scope}`);
  }
  const positions = categories.map(([title]) => {
    assert.equal(body.split(title).length - 1, 1, `Unique register category: ${title}`);
    return body.indexOf(title);
  });
  assert.deepEqual([...positions].sort((a, b) => a - b), positions, 'Register category order');
  const evidence = categories.map(([title, samples], i) => {
    const span = body.slice(positions[i], positions[i + 1] ?? body.length);
    for (const sample of samples) assert(span.replace(/\s/g, '').includes(sample.replace(/\s/g, '')), `Register category ${title} missing evidence: ${sample}`);
    return {title, samples};
  });
  const entry = aux.split('\n').find(line => line.startsWith('\\newlabel{sec:appendix}'));
  const [, number, page] = entry?.match(/^\\newlabel\{[^}]+\}\{\{([^}]+)\}\{(\d+)\}/) || [];
  assert.equal(number, 'A', 'Built register label must resolve to A');
  assert(norm(text.split('\f')[Number(page) - 1] || '').includes(register), 'Built register label must target the PDF heading page');
  assert(!/sec:appendix_(?:cohort|bpb)/.test(tex + aux), 'Historical appendix labels are retired');
  const expected = (tex.match(/Section~\\ref\{sec:appendix\}/g) || []).length;
  const actual = [...pdf.matchAll(/\bSection (A(?:\.\d+)*|[B-F])\b/g)].map(m => m[1]);
  assert.deepEqual(actual, Array(expected).fill('A'), 'Every rendered register reference must match its TeX target');
  return {appendix: 'A', subsections: 0, evidence, target: {label: 'sec:appendix', number, page: Number(page)}, resolvedReferences: expected};
}
