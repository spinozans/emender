// Source-named regression checks for the P13 notation/protocol revision.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {verifySymbols} from './verify-symbols.mjs';
const root=new URL('../../',import.meta.url);
const read=p=>fs.readFileSync(new URL(p,root),'utf8');
const source=read('paper/main.typ');
test('main.typ and all current manuscript variants exclude self-vouching words',()=>{
  for(const p of ['paper/main.typ','paper/figures/expressivity_1p3b.typ',
    'paper/tmlr/main.tex','paper/tmlr/submission.tex','paper/tmlr/preprint.tex',
    'paper/tmlr/validation/canonical-text.txt','paper/tmlr/validation/submission-text.txt',
    'paper/tmlr/validation/preprint-text.txt'])assert.doesNotMatch(read(p),/honest|carefully|rigorously/i,p);
  for(const label of ['EXHIBIT','ANALOGY','OPEN'])assert.ok(source.includes(label));
});
test('main.typ has one notation table and time-ordered numbered pseudocode',()=>{
  assert.equal(source.match(/\[Notation\]/g).length,1);
  assert.deepEqual([...source.matchAll(/\*Algorithm (\d+) —/g)].map(m=>Number(m[1])),[1,2,3,4,5,6,7,8]);
  const blocks=[...source.matchAll(/```text\n([\s\S]*?)\n```/g)].map(m=>m[1]);
  assert.equal(blocks.length,8);
  for(const block of blocks){assert.match(block,/^Input:/);assert.match(block,/\n1\. /);assert.match(block,/\nReturn:/);}
  assert.equal(read('paper/tmlr/main.tex').match(/\\begin\{verbatim\}/g).length,8);
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
test('depthwall pseudocode preserves implemented indexing and terminal-not-best selection',()=>{
  for(const fragment of ['step j = 0,...,cap-1','j >= min(1000,cap)',
    'loss <= 1e-4 OR both 200/300-step gains are < 1%',
    'Retain stop weights','full-vocabulary argmax','halve B'])assert.ok(source.includes(fragment),fragment);
  const driver=read('scripts/expressivity_1p3b_depthwall.py');
  for(const fragment of ['if hist[-1][0] < min_step:','if v3 <= 1e-4:',
    'return rel300 < 0.01 and rel200 < 0.01','for step in range(args.steps):',
    'preds = logits.argmax(dim=-1)','B //= 2'])assert.ok(driver.includes(fragment),fragment);
});
test('verify-symbols counts primes in imported vector drawings without converting them',()=>{
  const prime={func:'equation',body:{func:'attach',base:{func:'symbol',text:'S'},tr:{func:'primes',count:1}}};
  const ast={func:'figure',kind:'image',body:prime,caption:{func:'text',text:'drawing'}};
  const report=verifySymbols(ast,'',{canonical:'S′',submission:'S′'});
  assert.equal(report.equations,0);assert.equal(report.vectorPrimes,1);
  assert.throws(()=>verifySymbols(ast,'',{submission:'S'}),/prime lost/);
});
