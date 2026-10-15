import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {headingLevel} from './heading-level.mjs';
import {verifyRegisterAppendix} from './register-appendix.mjs';

test('convert: preserve markup depth and explicit heading level', () => {
  for (const depth of [1, 2, 3]) assert.equal(headingLevel({depth}), depth);
  for (const level of [1, 2, 3]) assert.equal(headingLevel({level}), level);
  assert.equal(headingLevel({level: 3, depth: 2}), 3);
  assert.equal(headingLevel({}), 1);
});

test('convert: actual register inventory has one level-1 and five level-2 headings', () => {
  const {headings} = JSON.parse(fs.readFileSync(new URL('conversion-inventory.json', import.meta.url)));
  const register = headings.slice(headings.findIndex(h => h.title === 'Development history and source register'));
  assert.equal(register.filter(h => h.level === 1).length, 1);
  assert.equal(register.filter(h => h.level === 2).length, 5);
  assert(register.every(h => h.section === 'Development history and source register'));
  assert(register.filter(h => h.level === 3).length > 0);
});

const tex = String.raw`\appendix
\section{Development history and source register}\label{sec:appendix}
\subsection{Systems}\subsection{Panels}\subsection{Lineage}
\subsection{Cohort}\label{sec:appendix_cohort}
\subsection{Bpb}\label{sec:appendix_bpb}
Section~\ref{sec:appendix} Section~\ref{sec:appendix} Section~\ref{sec:appendix}
Section~\ref{sec:appendix_cohort} Section~\ref{sec:appendix_bpb}`;
const pdf = `E88 is the developmental predecessor (Section A).
Predecessor-cohort occupancy, MFU, and the linear-scan kernel comparison are recorded in Section A.
Predecessor-cohort downstream capability panels are recorded in Section A.
Cohort Section A.4, method Section A.5.
\fA Development history and source register
A.1 Predecessor-cohort systems measurements
22,200 7,492 15.7% 8,248
A.2 Downstream capability panels: a null, on the predecessor cohort
2,049-item 2,048-item Table 10 0.345 0.373 0.358 0.353
A.3 Lineage of the E63 → E88 → E97 experimental program
E63 E88 E97 step 8,400
A.4 The predecessor 1.3 B cohort (E88 era)
0.966 0.961 0.973
A.5 Held-out bits-per-byte measurement (predecessor cohort)
Common Pile 0.981 0.963 0.973`;

const aux = String.raw`\newlabel{sec:appendix}{{A}{2}{Register}{appendix.A}{}}
\newlabel{sec:appendix_cohort}{{A.4}{2}{Cohort}{subsection.A.4}{}}
\newlabel{sec:appendix_bpb}{{A.5}{2}{Bpb}{subsection.A.5}{}}`;

test('register-appendix: rejects stale labels and wrong PDF destination pages', () => {
  assert.throws(() => verifyRegisterAppendix(pdf, tex, aux.replace('{{A.4}', '{{E}')), /must resolve/);
  assert.throws(() => verifyRegisterAppendix(pdf, tex, aux.replace('{{A.5}{2}', '{{A.5}{1}')), /heading page/);
});

test('register-appendix: built PDF evidence spans and all reference counts resolve', () => {
  const result = verifyRegisterAppendix(pdf, tex, aux);
  assert.equal(result.subsections.length, 5);
  assert.deepEqual(result.resolvedReferences, {A: 3, 'A.4': 1, 'A.5': 1});
});
test('register-appendix: rejects old separate appendices and missing/duplicate subsections', () => {
  assert.throws(() => verifyRegisterAppendix(pdf, tex.replace('\\subsection{Systems}', '\\section{Systems}'), aux), /top-level/);
  assert.throws(() => verifyRegisterAppendix(pdf.replace('A.1 Predecessor', 'B Predecessor'), tex, aux), /missing unique/);
  assert.throws(() => verifyRegisterAppendix(pdf + '\nA.1 Predecessor-cohort systems measurements', tex, aux), /missing unique/);
  assert.throws(() => verifyRegisterAppendix(pdf, tex.replace('\\subsection{Systems}', ''), aux), /five register/);
});
test('register-appendix: rejects evidence outside its target subsection', () => {
  assert.throws(() => verifyRegisterAppendix(pdf.replace('7,492', '') + ' 7,492', tex, aux), /A.1 missing evidence/);
  assert.throws(() => verifyRegisterAppendix(pdf.replace('0.963', ''), tex, aux), /A.5 missing evidence/);
});
test('register-appendix: rejects stale, missing, or misnumbered rendered pointers', () => {
  assert.throws(() => verifyRegisterAppendix(pdf.replace('method Section A.5', 'method Section F'), tex, aux), /reference.*count/);
  assert.throws(() => verifyRegisterAppendix(pdf.replace('Cohort Section A.4', 'Cohort Section A.3'), tex, aux), /reference.*count/);
  assert.throws(() => verifyRegisterAppendix(pdf.replace('recorded in Section A.', 'recorded elsewhere.'), tex, aux), /main-text pointer/);
});
