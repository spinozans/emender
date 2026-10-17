// Fail-closed PDF drift/identity checks. Poppler binaries can be supplied by env.
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {checkCanonicalSource} from './canonical-guard.mjs';
import {verifySymbols} from './verify-symbols.mjs';
import {verifyAppendixOrder} from './appendix-order.mjs';
import {verifyRegisterAppendix} from './register-appendix.mjs';
import {verifyArmor} from './armor-audit.mjs';
const dir=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(dir,'../..');
const pdftotext=process.env.PDFTOTEXT||'pdftotext',pdfinfo=process.env.PDFINFO||'pdfinfo';
const files={canonical:path.resolve(dir,'../current.pdf'),submission:path.join(dir,'submission.pdf'),preprint:path.join(dir,'preprint.pdf')};
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
  [
    "Abstract / invertible map",
    "across three seeds gives 0.268% versus 0.227% full-state accuracy, with no robust trained-length separation."
  ],
  [
    "Introduction / matched geometry",
    "Its GDN2 control is within 0.01% of its parameter count."
  ],
  [
    "Background / fixed precision",
    "At fixed precision and width, both state regimes are finite-state recognizers."
  ],
  [
    "Architecture / stack",
    "1,286,589,072 parameters in total"
  ],
  [
    "Systems / actual geometry",
    "The ratio is 0.976: comparable throughput, with GDN2 slightly faster. E97 uses 28,942 MB peak memory per GPU against 35,715 MB for GDN2."
  ],
  [
    "Language modeling / matched tokens",
    "exact common logged steps through step 2,300,925 (150,793,420,800 tokens): 92,037 aligned records, zero E97-only and zero GDN2-only records through the cutoff, no interpolation."
  ],
  [
    "Language modeling / interpretation",
    "The nonlinear recurrence trains competitively under this substantial shared workload, with a small difference whose sign varies by averaging window."
  ],
  [
    "Expressivity / paired uncertainty",
    "The paired E97 minus GDN2 difference is 0.041 ± 0.201 percentage points (mean ± sample SD over three seeds), giving no robust separation."
  ],
  [
    "Expressivity / rate confound",
    "The cross-rate comparison is observational with a protocol confound, not an identified learning-rate mechanism."
  ],
  [
    "Applications / stability",
    "A learned surrogate comparison could vary how often saturation determines subsequent decisions, measuring trajectory error, latency, and closed-loop stability across operating regimes."
  ],
  [
    "Formal / inventory",
    "857 named declarations, 723 user-written, 17 files; extraction source commit e58ae937"
  ],
  [
    "Formal / recognizer floor",
    "Any finite-state recognizer with a state decoder that reproduces the tracker on every input word needs at least 120 states"
  ],
  [
    "Open questions / persistence",
    "the stated hold theorem assumes λ ≥ 1, while production decay lies below one."
  ],
  [
    "Related work / state path",
    "with 87.5% of its blocks linear-state, it sits closer to the linear-state class than to the pure-nonlinear-recurrent one."
  ],
  [
    "Open questions / sampling",
    "The current pair supplies one language-model training seed; the matched-rate invertible-map study uses three fine-tuning seeds per arm."
  ],
  [
    "Expressivity / S5 finding",
    "GDN2 is ahead of E97 at every reported length, and both curves decline toward the random baseline."
  ],
  [
    "Next tests / regime",
    "This would test whether the observed regime difference survives the protocol confound."
  ],
  [
    "Future work / scale",
    "A matched 3–7 B E97/GDN2 study would test training viability, memory cost, and computational behavior across scale."
  ],
  [
    "Frontier / endpoint",
    "Its final recorded rank-0 training-loss average is 2.2798 nats/token over the last 100 logging windows."
  ],
  [
    "Frontier / communication",
    "completed 73 hierarchical K40 merges, with median merge duration 6.320 s and median inter-merge interval 72 s."
  ]
];
const results=samples.map(([section,sentence],i)=>{
 const matched={};for(const variant of Object.keys(files)){
   matched[variant]=norm(text[variant]).includes(norm(sentence));
   assert(matched[variant],`Sample ${i+1} absent from ${variant}: ${sentence}`);
 }
 return {sample:i+1,section,sentence,exactNumbers:sentence.match(/\d[\d,]*(?:\.\d+)?%?/g)||[],matched};
});
for(const variant of ['submission','preprint']){
 assert(!/\b0pt\b/.test(text[variant]),'Unexpected theorem-head text');
 assert(text[variant].includes('Žiga Avsec'),'Bibliography caron glyph lost');
 assert(/M2rnn:/.test(text[variant].normalize('NFKC').replace(/\s/g,'')),'Bibliography superscript glyph lost');
}
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
const armorAudit=verifyArmor(fs.readFileSync(path.join(root,'paper/main.typ'),'utf8'),fs.readFileSync(path.join(root,'paper/figures/expressivity_1p3b.typ'),'utf8'));
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
const approvedSymbolCounts=JSON.parse(fs.readFileSync(path.join(dir,'canonical-baseline.json'),'utf8')).symbolCounts;
assert.deepEqual({equations:symbolAudit.equations,attachments:symbolAudit.attachments,pdfPrimes:symbolAudit.pdfPrimes},approvedSymbolCounts,'Unapproved symbol-inventory drift');
const appendixOrder=Object.fromEntries(Object.entries(text).map(([variant,pdf])=>[
 variant,verifyAppendixOrder(pdf,variant==='canonical'?null:generatedTex,
   variant==='canonical'?'arXiv:2505.17852':'arXiv:2406.06484')
]));
fs.writeFileSync(path.join(dir,'validation/p16b-appendix-order.json'),JSON.stringify(appendixOrder,null,2)+'\n');
const registerAppendix=Object.fromEntries(['submission','preprint'].map(variant=>[
 variant,verifyRegisterAppendix(text[variant],generatedTex,fs.readFileSync(path.join(dir,variant+'.aux'),'utf8'))
]));
fs.writeFileSync(path.join(dir,'validation/p17c-register-appendix.json'),JSON.stringify(registerAppendix,null,2)+'\n');
const report={armorAudit,registerAppendix,appendixOrder,symbolAudit,approvedBibliographyAmendment:'mathlib4 access year plus four P20 transferred entries (exact SHA-256 guarded)',acceptedCommit:'eb2b9636',canonicalBaseline:JSON.parse(fs.readFileSync(path.join(dir,'canonical-baseline.json'),'utf8')),styleCommit:fs.readFileSync(path.join(dir,'STYLE-COMMIT'),'utf8').trim(),pages,samples:results,fragmentAudit,identityChecks:'passed (third-person literature self-citation retained)',canonicalManuscriptMatchesApprovedBaseline:true,bibliographyByteIdentical:true};
fs.mkdirSync(path.join(dir,'validation'),{recursive:true});
fs.writeFileSync(path.join(dir,'validation/drift-check.json'),JSON.stringify(report,null,2)+'\n');
for(const [k,v] of Object.entries(text))fs.writeFileSync(path.join(dir,'validation',k+'-text.txt'),v);
console.log(`PASS: ${results.length} samples × 3 PDFs; exact numeric strings; ${fragmentAudit.checked} long source fragments; identity and canonical-source guards.`);
console.log(`PASS: ${symbolAudit.equations} complete math expressions; ${symbolAudit.attachments} attachments; ${symbolAudit.pdfPrimes} S primes in each PDF (${symbolAudit.vectorPrimes} in drawings).`);
console.log('PASS: one compact Appendix A; scoped source categories, built-label destination page, and every register reference in each TMLR PDF.');
console.log(JSON.stringify(pages));
