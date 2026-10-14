// Fail-closed PDF drift/identity checks. Poppler binaries can be supplied by env.
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {checkCanonicalSource} from './canonical-guard.mjs';
import {verifySymbols} from './verify-symbols.mjs';
const dir=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(dir,'../..');
const pdftotext=process.env.PDFTOTEXT||'pdftotext',pdfinfo=process.env.PDFINFO||'pdfinfo';
const files={canonical:path.resolve(dir,'../Garrison_2026_Emender-eb2b9636.pdf'),submission:path.join(dir,'submission.pdf'),preprint:path.join(dir,'preprint.pdf')};
// NFKC folds math italic letters and ligatures, not numeric values. Remove
// running furniture and typography only; retain decimal points and percentages.
function clean(s){return s.replaceAll('Under review as submission to TMLR','').replace(/(?:^|\n)\d+\s*(?=\n|\f|$)/g,'\n').normalize('NFKC')}
function norm(s){return clean(s).replace(/[^\p{L}\p{N}.%≥≤λ]/gu,'').toLowerCase()}
const text={},info={};
for(const [variant,file] of Object.entries(files)){
 text[variant]=execFileSync(pdftotext,['-raw',file,'-'],{encoding:'utf8',maxBuffer:8*1024*1024});
 info[variant]=execFileSync(pdfinfo,[file],{encoding:'utf8'});
}
const samples=[
 ['Abstract','modular-quadratic terminal accuracy is 99.96% versus 76.09% at trained length, and 76.07% versus 53.36% at eight times that length.'],
 ['Introduction','at the matched 150.79-billion-token cutoff the smoothed loss gap between the two arms is 0.010 nats/token'],
 ['Background','At fixed precision and width, both state regimes are finite-state recognizers.'],
 ['Architecture','1,286,589,072 parameters in total'],
 ['Systems','All throughput measurements and throughput-budget smokes reported here are CUDA-only; none establishes correctness, performance, or parity on ROCm.'],
 ['Language modeling','exact common logged steps through step 2,300,925 (150,793,420,800 tokens): 92,037 aligned records, zero E97-only and zero GDN2-only records through the cutoff, no interpolation.'],
 ['Expressivity / contractive control','gdn-8m has 7.94 M parameters versus E97’s 7.38 M (7.6% more, favoring GDN).'],
 ['Expressivity / pre-registered budget','The throughput-derived common cap of 18269 steps was fixed before accuracy was seen (budget.json, 1989656c).'],
 ['Expressivity / terminal sweep','At the terminal sweep E97 reaches 99.96% at trained length, retaining 90.16% at four times and 76.07% at eight times that length.'],
 ['Applications','A learned surrogate would additionally need stability and safety validation; a state-tracking result is not a control guarantee.'],
 ['Formal / trust inventory','857 named declarations, 723 user-written, 17 files; extraction source commit e58ae937'],
 ['Formal / recognizer floor','Any finite-state recognizer with a state decoder that reproduces the tracker on every input word needs at least 120 states'],
 ['Formal / latching boundary','the hold theorem assumes λ ≥ 1, whereas production decay is below one'],
 ['Related work','with 87.5% of its blocks linear-state, it sits closer to the linear-state class than to the pure-nonlinear-recurrent one.'],
 ['Limitations','The current modular-quadratic comparison shares a cap and plateau rule but realizes 2,500 versus 2,200 steps'],
 ['Conclusion / negative transfer','The 8M S5 separation does NOT replicate at 1.3B on this pair.'],
 ['Testable predictions','Falsified if per-token throughput degrades faster than the parameter count grows, beyond the kernel-arithmetic baseline.'],
 ['Future work','the single-arm E97 513 B-token Frontier continuation that exists cannot answer a paired question'],
 ['Appendix B / predecessor cohort','E88 0.973 BPB, GDN 0.973 BPB, and M²RNN-CMA 0.979 BPB.'],
 ['Appendix C / Common Pile','Our three models on the same Common-Pile slice measure Emender 0.981, GDN 0.963, M²RNN-CMA 0.973']
];
const results=samples.map(([section,sentence],i)=>{
 const matched={};for(const variant of Object.keys(files)){
   matched[variant]=norm(text[variant]).includes(norm(sentence));
   assert(matched[variant],`Sample ${i+1} absent from ${variant}: ${sentence}`);
 }
 return {sample:i+1,section,sentence,exactNumbers:sentence.match(/\d[\d,]*(?:\.\d+)?%?/g)||[],matched};
});
for(const variant of ['submission','preprint'])assert(!/\b0pt\b/.test(text[variant]),'Unexpected theorem-head text');
assert(/Anonymous authors/.test(text.submission));
assert(/Author:\s+Anonymous\s*\n/.test(info.submission));
assert(!/spinozans|hypervolu\.me|erik\.garrison@gmail|Poietic|UTHSC|0000-0003-3821-631X/.test(text.submission+info.submission));
assert(/Erik Garrison/.test(text.preprint)&&/Author:\s+Erik Garrison/.test(info.preprint));
assert(/huggingface\.co\/spinozans/.test(text.preprint.replace(/\s/g,'')));
assert(text.submission.includes('the artifact repository (anonymized)'));
assert(fs.readFileSync(path.join(dir,'refs.bib')).equals(fs.readFileSync(path.join(dir,'../refs.bib'))));
for(const file of ['tmlr.sty','tmlr.bst','fancyhdr.sty','LICENSE']){
 // Vendored SHA-256 digests are frozen in the provenance manifest.
 assert(fs.existsSync(path.join(dir,file)));
}
checkCanonicalSource(root);
// Additional all-prose audit: every >=30-character canonical text/raw leaf.
// Float insertions can split a leaf in PDF reading order. Permit one insertion,
// but require the two exact halves in their original order and record it.
const ast=JSON.parse(execFileSync('typst',['query','--root',root,path.join(dir,'serialize.typ'),'<manuscript>','--field','value'],{maxBuffer:16*1024*1024}))[0];
const proseNorm=s=>clean(s).replace(/[^\p{L}\p{N}]/gu,'').toLowerCase();
const proseTexts=Object.fromEntries(Object.entries(text).map(([k,v])=>[k,proseNorm(v)]));
const fragmentAudit={checked:0,layoutSplitFragments:[],approvedIdentityExclusions:[]};
function inspect(x){
 if(!x||typeof x!=='object')return;
 if(['text','raw'].includes(x.func)&&typeof x.text==='string'&&x.text.length>=30){
   if(/huggingface\.co\/spinozans|hypervolu\.me|Poietic PBC \/ UTHSC/.test(x.text)){fragmentAudit.approvedIdentityExclusions.push(x.text)}
   else{
     fragmentAudit.checked++;
     const expected=proseNorm(x.text);
     for(const [variant,pdf] of Object.entries(proseTexts)){
       if(pdf.includes(expected))continue;
       let split=null;
       for(let i=5;i<expected.length-5;i++){
         const left=pdf.indexOf(expected.slice(0,i));
         if(left>=0&&pdf.indexOf(expected.slice(i),left+i)>=0){split=i;break}
       }
       assert(split!==null,`Unmatched source fragment in ${variant}: ${x.text}`);
       fragmentAudit.layoutSplitFragments.push({variant,fragment:x.text,split});
     }
   }
 }
 Object.values(x).forEach(v=>Array.isArray(v)?v.forEach(inspect):inspect(v));
}
inspect(ast);
const pages=Object.fromEntries(Object.entries(info).map(([k,v])=>[k,Number(v.match(/^Pages:\s+(\d+)/m)[1])]));
const generatedTex=fs.readFileSync(path.join(dir,'main.tex'),'utf8');
assert(!/\\(?:fontsize|tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)\b|\\captionsetup\{[^}]*font\s*=/.test(generatedTex),'Explicit font-size override');
const symbolAudit=verifySymbols(ast,generatedTex,text);
const report={symbolAudit,approvedBibliographyAmendment:'mathlib4: explicit 2026 access year (not publication year)',acceptedCommit:'eb2b9636',styleCommit:fs.readFileSync(path.join(dir,'STYLE-COMMIT'),'utf8').trim(),pages,samples:results,fragmentAudit,identityChecks:'passed (third-person literature self-citation retained)',canonicalManuscriptUnchanged:true,bibliographyByteIdentical:true};
fs.mkdirSync(path.join(dir,'validation'),{recursive:true});
fs.writeFileSync(path.join(dir,'validation/drift-check.json'),JSON.stringify(report,null,2)+'\n');
for(const [k,v] of Object.entries(text))fs.writeFileSync(path.join(dir,'validation',k+'-text.txt'),v);
console.log(`PASS: ${results.length} samples × 3 PDFs; exact numeric strings; ${fragmentAudit.checked} long source fragments; identity and canonical-source guards.`);
console.log(`PASS: ${symbolAudit.equations} complete math expressions; ${symbolAudit.attachments} attachments; ${symbolAudit.primes} S primes in each PDF.`);
console.log(JSON.stringify(pages));
