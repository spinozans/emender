import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {verifySymbols} from './verify-symbols.mjs';
const symbol=text=>({func:'symbol',text});
const equation=body=>({func:'equation',body});
const base=symbol('S'),b=symbol('i'),t=symbol('2'),tr={func:'primes',count:1};
const cases=[
  [{base},'{S}'],
  [{base,b},'{S}_{i}'],
  [{base,t},'{S}^{2}'],
  [{base,b,t},'{S}_{i}^{2}'],
  [{base,tr},"{S}'"],
  [{base,b,tr},"{S}'_{i}"],
  [{base,tr:{func:'primes',count:2}},"{S}''"]
];
for(const [fields,expected] of cases)test(`attachment ${Object.keys(fields).join('/')} => ${expected}`,()=>{
  const ast=equation({func:'attach',...fields});
  assert.equal(verifySymbols(ast,'$'+expected+'$').attachments,1);
  assert.throws(()=>verifySymbols(ast,'$S$'),/lost\/changed/);
});
test('new attachment features fail closed',()=>{
  assert.throws(()=>verifySymbols(equation({func:'attach',base,bl:b}),'${S}$'),/Unaudited attachment fields/);
});
test('PDF prime loss is independent of TeX preservation',()=>{
  const ast=equation({func:'attach',base,tr});
  verifySymbols(ast,"${S}'$",{canonical:'𝑆′',submission:'S′',preprint:'S′'});
  assert.throws(()=>verifySymbols(ast,"${S}'$",{submission:'S'}),/prime lost/);
});
const root=new URL('../../',import.meta.url).pathname;
const ast=JSON.parse(execFileSync('typst',['query','--root',root,new URL('serialize.typ',import.meta.url).pathname,'<manuscript>','--field','value'],{maxBuffer:16*1024*1024}))[0];
const tex=fs.readFileSync(new URL('main.tex',import.meta.url),'utf8');
test('canonical complete math and all rendered attachments match',()=>{
  const report=verifySymbols(ast,tex);
  assert.equal(report.primes,6);assert.equal(report.vectorPrimes,2);
  // P20 transfer deltas are enumerated in p21-math-delta.json;
  // each current complete expression is checked by the independent oracle.
  assert.equal(report.pdfPrimes,8);assert.equal(report.attachments,266);assert.equal(report.equations,325);
});
test('regression: dropping any converted canonical prime fails',()=>{
  const positions=[...tex.matchAll(/\{S \}'/g)].map(m=>m.index+m[0].length-1);
  assert.equal(positions.length,6);
  for(const i of positions)assert.throws(()=>verifySymbols(ast,tex.slice(0,i)+tex.slice(i+1)),/lost\/changed/);
});
test('P10 generated TeX is rejected (actual original defect)',()=>{
  const old=execFileSync('git',['show','13115501:paper/tmlr/main.tex'],{cwd:root,encoding:'utf8'});
  assert.throws(()=>verifySymbols(ast,old),/lost\/changed/);
});
