// P16b reproducible source-preservation and reader-first disposition audit.
// Run from the repository root; the two historical inputs are immutable commits.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
const dir='paper/tmlr/validation/';
const git=p=>execFileSync('git',['show',p],{encoding:'utf8',maxBuffer:4*1024*1024});
const before=git('3f3f2134:paper/main.typ');
const source=fs.readFileSync('paper/main.typ','utf8');
const reader=git('fba15ce5:paper/main.reader-first.typ');
const paragraphs=reader.split('\n\n');
const imports=JSON.parse(fs.readFileSync(dir+'p16b-reader-imports.json'));
const edits=JSON.parse(fs.readFileSync(dir+'p16b-edit-ledger.json'));
const normalize=s=>s.replace(/\s+/g,' ').trim();
const normSource=normalize(source);
const displays=s=>[...s.matchAll(/^\$\n[\s\S]*?^\$$/gm)].map(m=>m[0]);
const figures=s=>[...s.matchAll(/#figure\([\s\S]*?\) <[^>]+>/g)].map(m=>m[0]);
const pseudocode=s=>[...s.matchAll(/```text\n[\s\S]*?```/g)].map(m=>m[0]);
const standaloneTables=s=>[
 s.slice(s.indexOf('#table(\n  columns: (auto, auto),'),s.indexOf('#heading(level: 2, numbering: none)[Linear-state')),
 s.slice(s.indexOf('#align(center)[#table(\n  columns: (auto, auto, auto, auto),'),s.indexOf('[Per-architecture CMA-ES protocol]')).replace(/#heading\(level: [23],/g,'#heading(level: 2,')
];
for(const [label,extract,count] of [['display equations',displays,12],['figure/table blocks',figures,17],['pseudocode blocks',pseudocode,8],['standalone tables',standaloneTables,2]]) {
 assert.equal(extract(before).length,count,label);
 assert.deepEqual(extract(source),extract(before),`${label} byte-unchanged`);
}
const inlineCode=s=>[...s.matchAll(/(?<!`)`([^`\n]+)`(?!`)/g)].map(m=>m[1]);
const theoremNames=inlineCode(before).filter(t=>/^[A-Za-z][\w.]*$/.test(t)&&t!=='model.safetensors');
for(const name of theoremNames) assert(source.includes('`'+name+'`'),`exact source theorem/name retained: ${name}`);
const predictions=s=>s.slice(s.indexOf('= Testable Predictions'),s.indexOf('= Future Work'));
assert.equal(predictions(source),predictions(before),'Testable Predictions unchanged');
for(const file of ['paper/refs.bib','paper/figures/expressivity_1p3b.typ','paper/data/lean-trust-gate-theorems.json']) {
 assert.equal(fs.readFileSync(file,'utf8'),git('3f3f2134:'+file),file+' frozen');
}
assert(!source.includes('GDN2-MLP'),'no manuscript naming regression');
assert.equal((source.match(/^= Development history and source register <sec:appendix>$/gm)||[]).length,1);
for(const name of ['Lineage of the E63','The predecessor 1.3 B cohort','Held-out bits-per-byte measurement'])assert(source.includes('== '+name));
assert(source.indexOf('#bibliography(')<source.indexOf('#show: arkheion-appendices'));
const samples=s=>s.slice(s.indexOf('const samples=['),s.indexOf('const results=samples.map'));
assert.equal(samples(fs.readFileSync('paper/tmlr/verify.mjs','utf8')),samples(git('3f3f2134:paper/tmlr/verify.mjs')),'All 20 hardcoded guard samples unchanged');
// P13's sanctioned roles: x_t is token/layer input; S_(t-1) is previous carried state.
assert(source.includes('Token/layer input, generic input injection'));
assert(source.includes('Carried state, head state, time-indexed state'));
assert(source.includes('Conditioning on these parameters\nmakes the update affine in $S_(t-1)$'));
assert(source.includes("independently\nof that layer's previous carried state"));
const reasons={
 22:'Excluded: Transformer citation is absent from the frozen canonical bibliography; canonical arithmetic/uniformity-qualified Transformer results retained.',
 25:'Canonical notation table byte-frozen: retain H comparator/count overload, Lyapunov lambda, and scalar/vector delta disambiguations omitted by reader-first.',
 29:'Shared scan/finite-state sentences retained; preserve exact arithmetic, uniformity, precision and open class-separation conditions in the longer canonical remainder.',
 30:'Shared nonlinear-state lineage retained; preserve fixed-width/fixed-precision Turing-completeness boundary and open TC0/NC1 distinction.',
 43:'Figure/table/caption byte-frozen. Canonical caption keeps production non-inclusion and historical-tag scope.',
 45:'Ruling 2 genre skeleton keeps Architecture heading; content revoiced.',
 51:'Preserve silu(v), all-one gates, exact E97/E88 theorem binding, and full-stack G2 condition absent from alternate.',
 55:'Figure/caption byte-frozen; preserve Lean-certified predecessor annotation.',
 59:'Shared delta read/correct sentences already present; retain raw-write one-step fixed-weight boundary and math unchanged.',
 71:'Ruling 2 fixes Systems position/title; content revoiced.',
 76:'Preserve CUDA-only heading identifying the accepted platform scope.',
 77:'Retain hardcoded verify.mjs sample: CUDA-only evidence explicitly excludes ROCm correctness/performance/parity.',
 81:'Shared fused-kernel sentences already present; retain accepted depth-12 50–60 ms measurement omitted by alternate.',
 84:'Canonical heading keeps source portability distinct from CUDA-only validation.',
 89:'Ruling 2 fixes Language-Modeling Results heading; narrative opening adopted.',
 93:'Table/caption byte-frozen; Ruling 5 uses GDN2, not alternate GDN2-MLP label.',
 96:'Pseudocode byte-frozen; Algorithm 4 numbering retains the canonical raw-write Algorithm 3.',
 99:'Table/caption byte-frozen; Ruling 5 uses GDN2.',
 102:'Canonical heading retains full search breadth/budget provenance, beyond the two-winner summary.',
 103:'Shared per-candidate settings retained in canonical search protocol; preserve population 8, exact knobs, tokenizer/vocab, mandatory fused path and fitness/tie rules.',
 104:'Shared winner numbers/records retained; preserve H notation and provisional 64-evaluation versus final 104-evaluation winner distinction and raw-write-arm exclusion.',
 105:'Shared unavailable-generation-zero statement retained in U3.1; preserve unavailable spread and prohibition on imputation from earlier sweeps.',
 106:'Ruling 2 keeps Expressivity Results heading; current-pair opening adopted.',
 111:'Pseudocode byte-frozen; canonical Algorithm 6 number retained.',
 115:'Shared temporal-composition explanation retained; preserve exact Lean names/commits, fixed-schedule/decoder hypotheses and full-stack/minimal-depth non-claims.',
 118:'Negative finding retained in canonical exact artifact quotations, including phase-0 salvage; the negative IS the result (Ruling 1 exception).',
 123:'Ruling 2 places Formal Results before systems/experiments; retain theorem-set identity.',
 124:'Alternate says tested above; Ruling 2 puts experiments below. Canonical synopsis also retains matched-signature separation, synthetic cost and witness/trajectory distinctions omitted by alternate.',
 125:'Retain Theorem set H heading and identity for exact source bindings and packaging environments.',
 129:'Retain Theorem set A heading/identity for exact source bindings.',
 135:'Retain Theorem set B heading and lookup/trajectory distinction.',
 138:'Retain Theorem set G heading and abstract E97/E88/GDN2 relation.',
 140:'Bullet text is identical; canonical list setup is earlier because abstract E97/E88 statements also survive.',
 143:'Retain Theorem set F identity; slot-bound synopsis adopted.',
 150:'Trust/verification paragraph adopted at canonical formal opening; retain all theorem-set headings and evidence.',
 152:'Section title identical; Ruling 2 changes banner/order only.',
 154:'Pseudocode byte-frozen; canonical Algorithm 7 number retained.',
 156:'Retain canonical feedback heading and six-example ordering; reader-first feedback prose imported.',
 158:'Retain six canonical EXHIBIT/ANALOGY/OPEN examples and their scope, not alternate grouping with new unsourced applications.',
 159:'Retain canonical heading; quantization-in-loop scope remains explicit.',
 161:'Retain canonical heading; ordinary affine LMS versus nonlinear engineering variants remains explicit.',
 163:'Retain canonical heading; fixed-map versus untied-unrolling scope remains explicit.',
 165:'Excluded new cited subsection: scratchpad/chain-of-thought citations absent from frozen bibliography.',
 166:'Conversation/perception motivation imported in introduction; new cited subsection excluded because bibliography/evidence is frozen.',
 167:'Retain canonical reservoir heading; E97/echo-state-training and property distinction is load-bearing.',
 169:'Excluded new cited subsection: Enformer citation and genomic-regulatory predictions absent from canonical evidence/bibliography.',
 170:'Terabase pangenomics motivation already retained with existing citations; Enformer/regulatory-prediction claims excluded as unsupported by canonical evidence.',
 171:'Retain canonical ecology heading and autonomous/contractive/periodic/chaotic regime qualifications.',
 175:'Excluded new cited subsection: scratchpad/chain-of-thought citations absent from frozen bibliography.',
 176:'Excluded: new scratchpad-training/CoT results have no canonical bibliography entries; carry-memory motivation adopted without those claims.',
 181:'Shared S5 model/group distinction already present; retain cited model scope versus billion-parameter recurrent LMs.',
 187:'Keep canonical negative empirical record (no published >=500 M classical LSTM/GRU on Pile-class corpus): finding-negation, plus existing math mention.',
 188:'Ruling 2 requires Limitations heading; every accepted limitation retained.',
 189:'Canonical subsection headings retain exact single-seed/protocol and evaluation provenance bindings; research-target prose adopted.',
 192:'Canonical formal-scope heading preserves G1–G3/O1 labels and exact non-claims.',
 194:'Preserve exact G3 representation/production-range alternatives, shallow-token possibility, and arbitrary-matrix versus scalar-decay distinction.',
 196:'Preserve G1 non-lower-bound disclosure and dependency on resolving G3 before parameter-generator resource bounds.',
 197:'Shared production persistence question retained; preserve exact finite-k versus robustness and S5-alphabet scope with canonical equations.',
 198:'Canonical scale/comparator/backend headings preserve distinct evidence gaps, including unmatched single-arm continuation.',
 205:'Reproducibility content retained; heading remains inline to preserve canonical heading structure.',
 207:'Testable Predictions must survive as-is under original contract; retain title and falsifiers.',
 208:'Excluded replacement of Testable Predictions: explicit as-is constraint; seed/budget/ablation/evaluation research targets imported elsewhere.',
 217:'Canonical bibliography byte-frozen, refs.bib rather than alternate refs.reader-first.bib; moved before appendix for Ruling 6.'
};
const rows=paragraphs.map((p,i)=>{
 const recorded=imports.filter(x=>x.readerParagraph===i);
 const exact=normSource.includes(normalize(p));
 const status=exact?'as-is':recorded.length?'scoped adoption':i<10?'packaging preserved':'scoped deviation';
 const reason=exact?'Actual reader-first paragraph already present or adopted verbatim.':recorded.length?[...new Set(recorded.map(x=>x.reason))].join(' '):reasons[i]??(i<10?'Canonical metadata, imports, template and build instructions retained.':null);
 assert(reason,`Missing disposition for reader-first paragraph ${i}`);
 return {readerParagraph:i,readerLine:reader.slice(0,paragraphs.slice(0,i).join('\n\n').length+(i?2:0)).split('\n').length,status,anchor:p.replace(/\n/g,' ').slice(0,110),reason};
});
fs.writeFileSync(dir+'p16b-reader-voice-audit.json',JSON.stringify(rows,null,2)+'\n');
fs.writeFileSync(dir+'p16b-reader-voice-audit.md','# Ruling 7 — complete reader-first paragraph inventory\n\nSource: `fba15ce5:paper/main.reader-first.typ`. Zero-based paragraph IDs refer to blank-line-separated source paragraphs; line numbers and exact anchors make every disposition replayable. As-is is whitespace-insensitive; punctuation/words are unchanged. Scoped-adoption exceptions retain the complete canonical scope or frozen content, as specified. Canonical-only evidence remains where reader-first omitted it.\n\n| ID / source line | Disposition | Reader-first anchor | Scope reason / adoption |\n|---|---|---|---|\n'+rows.map(r=>`| R${r.readerParagraph} / ${r.readerLine} | ${r.status} | ${r.anchor.replaceAll('|','\\|')} | ${r.reason.replaceAll('|','\\|')} |`).join('\n')+'\n');
// Enumerate every inline-math addition/removal per edit, including relocated mentions.
const math=s=>[...s.matchAll(/\$([^$]*)\$/g)].map(m=>m[1]);
function difference(a,b){const left=[...a];return b.filter(v=>{const i=left.indexOf(v);if(i<0)return true;left.splice(i,1);return false;});}
const delta=[];
for(const [i,e] of edits.entries()) {
 const old=math(e.old),cur=math(e.new),added=difference(old,cur),removed=difference(cur,old);
 if(!added.length&&!removed.length)continue;
 const candidates=imports.filter(x=>e.new.includes(x.text)||x.reason===e.reason||normalize(e.new).includes(normalize(x.text).slice(0,75))).map(x=>'R'+x.readerParagraph);
 if(e.new.includes('abstract: ['))candidates.push('R3');
 if(e.reason.startsWith('Ruling 4'))candidates.push('R54 + canonical predecessor count');
 const first=normalize(e.new).slice(0,90);
 let rawStart=source.indexOf(e.new);
 if(rawStart<0)rawStart=source.indexOf(e.new.split('\n').slice(0,3).join('\n'));
 const line=rawStart>=0?source.slice(0,rawStart).split('\n').length:null;
 const occurrences=added.map(expression=>({expression,finalLines:[...source.matchAll(/\$([^$]*)\$/g)].filter(m=>m[1]===expression).map(m=>source.slice(0,m.index).split('\n').length)}));
 delta.push({edit:i+1,added,removed,finalParagraphStartLine:line,finalAnchor:first,readerPassages:[...new Set(candidates)],reason:e.reason,occurrences});
}
const mathMultisetDelta=[];for(const expr of new Set([...math(before),...math(source)])){const change=math(source).filter(x=>x===expr).length-math(before).filter(x=>x===expr).length;if(change)mathMultisetDelta.push({expression:expr,change});}
fs.writeFileSync(dir+'p16b-math-delta.json',JSON.stringify({approvedCounts:{expressions:718,attachments:449},displaysByteIdentical:12,notationConsistency:'Unchanged P13 table sanctions x_t as layer input and S_t as time-indexed carried state. Criterion now uses S_(t-1) for the previous state, not an input; no new overload. All other math formulas are unchanged.',mathMultisetDelta,edits:delta},null,2)+'\n');
const summary={base:'3f3f2134',readerFirst:'fba15ce5',displaysByteIdentical:12,figureTableBlocksByteIdentical:17,standaloneTablesByteIdentical:2,pseudocodeBlocksByteIdentical:8,exactTheoremAndSourceNamesPreserved:[...new Set(theoremNames)].length,predictionsByteIdentical:true,bibliographyAndImportedFiguresByteIdentical:true,oneRegisterAppendix:true,readerParagraphs:rows.length,dispositions:rows.reduce((a,r)=>(a[r.status]=(a[r.status]||0)+1,a),{}),mathMultisetDelta,sourceSha256:createHash('sha256').update(source).digest('hex')};
fs.writeFileSync(dir+'p16b-preservation.json',JSON.stringify(summary,null,2)+'\n');
console.log('PASS: '+JSON.stringify(summary));
