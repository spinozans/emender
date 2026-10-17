// Source-named regression checks for the transferred notation and protocols.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {verifySymbols} from './verify-symbols.mjs';
const root=new URL('../../',import.meta.url);
const read=p=>fs.readFileSync(new URL(p,root),'utf8');
const source=read('paper/main.typ');
test('main.typ and current manuscript variants retain the operator tone',()=>{
  for(const p of ['paper/main.typ','paper/figures/expressivity_1p3b.typ',
    'paper/tmlr/main.tex','paper/tmlr/submission.tex','paper/tmlr/preprint.tex',
    'paper/tmlr/validation/canonical-text.txt','paper/tmlr/validation/submission-text.txt',
    'paper/tmlr/validation/preprint-text.txt'])assert.doesNotMatch(read(p),/honest|carefully|rigorously/i,p);
  assert.ok(source.includes('The formal results describe the state computations tested above'));
  assert.doesNotMatch(source, /this is not|must not be described|do not call it|do not describe|is not a|not merely/i);
});
test('main.typ has one notation table and five time-ordered numbered algorithms',()=>{
  assert.equal(source.match(/\[Notation\]/g).length,1);
  assert.deepEqual([...source.matchAll(/\*Algorithm (\d+) —/g)].map(m=>Number(m[1])),[1,2,3,4,5]);
  const blocks=[...source.matchAll(/```text\n([\s\S]*?)\n```/g)].map(m=>m[1]);
  assert.equal(blocks.length,5);
  for(const block of blocks){assert.match(block,/^Input:/);assert.match(block,/\n1\. /);assert.match(block,/\nReturn:/);}
  assert.equal(read('paper/tmlr/main.tex').match(/\\begin\{verbatim\}/g).length,5);
});
test('E97 uses returned state for output and Hadamard split gates consistently',()=>{
  for(const fragment of ['r_h &= S_h^T (e_h dot.o k_h)',
    'delta_h &= w_h dot.o v_h - r_h',"S'_h &= tanh(d_h S_h + k_h delta_h^T)",
    '"silu"(g_h) dot.o (S\'_h)^T q_h','lambda S - k(e dot.o k)^T S',
    '(I - k(e dot.o k)^T) D S','S_T=A S_0+C','values $(k,e,w,v)$'])assert.ok(source.includes(fragment),fragment);
  assert.doesNotMatch(source,/e_h dot k_h|w_h dot v_h|e dot k|w dot v|lambda H|H_T=A H_0/);
  assert.ok(read('formal/lean/ElmanProofs/Architectures/SplitGatedDelta.lean').includes('hadamard b k'));
  assert.ok(read('formal/lean/ElmanProofs/Architectures/SplitGatedDelta.lean').includes('splitGatedLinearCore 1 (D * H)'));
});
test('invertible-map pseudocode retains validation-only selection and calibrated plateau',()=>{
  for(const fragment of ['validate every 250 updates','at least 10,000 updates',
    'accuracy gain over 2,000 updates < 0.001','within those 2,000 updates, < 0.001',
    'Cap: 60,000 updates','Freeze selection; evaluate test at 128/512/1024 drivers',
    'by lower validation cross-entropy, then earlier update'])assert.ok(source.includes(fragment),fragment);
});
test('verify-symbols counts primes in imported vector drawings without converting them',()=>{
  const prime={func:'equation',body:{func:'attach',base:{func:'symbol',text:'S'},tr:{func:'primes',count:1}}};
  const ast={func:'figure',kind:'image',body:prime,caption:{func:'text',text:'drawing'}};
  const report=verifySymbols(ast,'',{canonical:'S′',submission:'S′'});
  assert.equal(report.equations,0);assert.equal(report.vectorPrimes,1);
  assert.throws(()=>verifySymbols(ast,'',{submission:'S'}),/prime lost/);
});
