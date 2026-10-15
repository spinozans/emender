// Replay the enumerated P18 source diff; preserve measurements and inventory guards.
// Run from repository root. No guards or baselines are written by this audit.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {verifyArmor} from '../armor-audit.mjs';
const dir='paper/tmlr/validation/';
const base='aa28773f';
const git=p=>execFileSync('git',['show',base+':'+p],{encoding:'utf8',maxBuffer:8*1024*1024});
const edits=JSON.parse(fs.readFileSync(dir+'p18-edit-ledger.json'));
const files=['paper/main.typ','paper/figures/expressivity_1p3b.typ'];
const source=fs.readFileSync(files[0],'utf8'), captions=fs.readFileSync(files[1],'utf8');
const normalize=s=>s.replace(/\s+/g,' ').trim();
const math=s=>[...s.matchAll(/\$([^$]*)\$/g)].map(m=>m[1]);
function difference(a,b){const copy=[...a];return b.filter(v=>{const i=copy.indexOf(v);if(i<0)return true;copy.splice(i,1);return false;});}
const mathDelta=[];
for(const file of files){
 const before=git(file);let replay=before;
 for(const e of edits.filter(e=>(e.file??files[0])===file)){
  assert.equal(replay.split(e.old).length,2,e.id+' exact unique input');
  assert.equal(before.slice(0,before.indexOf(e.old)).split('\n').length,e.baseLine,e.id+' base citation');
  replay=replay.replace(e.old,e.replacement);
  const added=difference(math(e.old),math(e.replacement)),removed=difference(math(e.replacement),math(e.old));
  if(added.length||removed.length)mathDelta.push({id:e.id,file,baseLine:e.baseLine,added,removed,claim:e.claim,reason:e.reason});
 }
 assert.equal(replay.replace(/[ \t]+\n/g,'\n'),fs.readFileSync(file,'utf8'),'Exact enumerated source replay: '+file);
}
const old=git(files[0]);
for(const [kind,re,count] of [
 ['display equations',/^\$\n[\s\S]*?^\$$/gm,12],
 ['pseudocode blocks',/```text\n[\s\S]*?```/g,8]
]) {
 const extract=s=>[...s.matchAll(re)].map(m=>m[0]);
 assert.equal(extract(old).length,count,kind);assert.deepEqual(extract(source),extract(old),kind+' byte-identical');
}
const standalone=s=>[
 s.slice(s.indexOf('#table(\n  columns: (auto, auto),'),s.indexOf('#heading(level: 2, numbering: none)[Linear-state')),
 s.slice(s.indexOf('#align(center)[#table(\n  columns: (auto, auto, auto, auto),'),s.indexOf('[Per-architecture CMA-ES protocol]'))
];assert.deepEqual(standalone(source),standalone(old),'Both standalone notation/cohort tables unchanged');
const blocks=s=>new Map([...s.matchAll(/#figure\([\s\S]*?\) <([^>]+)>/g)].map(m=>[m[1],m[0]]));
const beforeBlocks=blocks(old),afterBlocks=blocks(source);
assert.deepEqual([...afterBlocks.keys()],[...beforeBlocks.keys()],'Every figure/table label and order retained');
assert.equal(beforeBlocks.size,17);
for(const [label,block] of beforeBlocks){
 if(['tab_claims','fig_taxonomy','tab_ablation','tab_bpb_landscape'].includes(label))continue;
 assert.equal(afterBlocks.get(label),block,'Scientific figure/table source unchanged: '+label);
}
for(const file of ['paper/refs.bib','paper/data/lean-trust-gate-theorems.json','paper/results/expressivity_1p3b/iterated-map/ft_e97_modular_quadratic_sweep.json','paper/results/expressivity_1p3b/iterated-map/ft_gdn2_modular_quadratic_sweep.json','paper/results/expressivity_1p3b/capacity/cap_e97_s5_sweep.json','paper/results/expressivity_1p3b/capacity/cap_gdn2_s5_sweep.json']){
 assert.equal(fs.readFileSync(file,'utf8'),git(file),'Frozen measurement/bibliography: '+file);
}
// All result trees, formal proof source, math-spec and drawing assets stay frozen.
execFileSync('git',['diff','--exit-code',base,'--','paper/results','formal','paper/data','paper/tmlr/assets','paper/refs.bib','paper/build.sh']);
const names=s=>[...s.matchAll(/`([A-Z][A-Za-z0-9]*(?:\.[A-Za-z0-9_]+)+)`/g)].map(m=>m[1]);
for(const name of new Set(names(old)))assert(source.includes('`'+name+'`'),'Qualified theorem/source identifier retained: '+name);
const armor=verifyArmor(source,captions);
const baseline=JSON.parse(fs.readFileSync('paper/tmlr/canonical-baseline.json'));
assert.equal(createHash('sha256').update(source).digest('hex'),baseline.mainTypSha256);
assert.deepEqual(baseline.symbolCounts,{equations:706,attachments:444,pdfPrimes:8});
const net=[];for(const expression of new Set([...math(old),...math(source)])){
 const change=math(source).filter(x=>x===expression).length-math(old).filter(x=>x===expression).length;
 if(change)net.push({expression,change});
}
fs.writeFileSync(dir+'p18-math-delta.json',JSON.stringify({base,previousCounts:{equations:718,attachments:449,pdfPrimes:8},approvedCounts:baseline.symbolCounts,displayEquationsByteIdentical:12,pseudocodeBlocksByteIdentical:8,netInlineMentionDelta:net,edits:mathDelta},null,2)+'\n');
// Complete line-level sweep: inherited P16b candidates plus affirmative boundaries.
// Changed ranges are represented in full by the edit table, not duplicated here.
const inherited=new Set(JSON.parse(fs.readFileSync(dir+'p16b-negation-audit.json')).map(r=>r.line));
const candidate=/\b(?:not|no|neither|nor|only|rather than|universal|outside|remain\w*|pending|unproved|unproven|hypothes\w*|conditional\w*|confound\w*|single.seed|scope|boundary|ceiling|cannot|fail\w*|negative|license|replicat\w*|predicted|claim\w*|guarantee\w*|impossib\w*|precision|assum\w*|budget\w*|protocol\w*|exclud\w*|under|limited|constrain\w*|countercase|prospective|open)\b/i;
function liveClaim(section,heading,context){
 const t=normalize(context);
 if(/Unified notation|Symbol|Conventions/.test(heading)||/not the erase gate|not every application's|not length|not the value vector/.test(t))return 'Notation: consistent state/input, time/threshold, and gate/perturbation roles';
 if(/U3\.1|generation.zero|first.generation|winner|sweep|fitness|search/.test(t)&&!/S_5.*search/.test(t))return 'CMA-hardened comparison: retained search records, candidate budget and unavailable winner-series fields';
 if(/O1|generator.*lookup|preloaded|recurrent.*realiz/.test(t))return 'Formal lookup bridge: supplied group state/external composition; recurrent realization O1 stays open';
 if(/TC#super|NC#super|complexity|circuit|Barrington|SSM/.test(t))return 'Cited circuit containments under arithmetic/uniformity assumptions; no assumed TC0/NC1 separation';
 if(/latch|saturat|hold theorem|write.budget|counter.delta|lambda >=|lambda >/.test(t)&&!/^Repeated nonlinear/.test(section))return 'Conditional slot-level saturation/hold/release, with production-range persistence open';
 if(/synthetic|state.pass|FLOP|dense|precomputed.gate|resource scope/.test(t))return 'Formal cost is synthetic/precomputed-gate accounting; actual empirical FLOPs require measurement';
 if(/ROCm|CUDA|occupancy|MFU|backend/.test(t))return 'Systems throughput/occupancy for the named CUDA geometry; E97 occupancy and paired ROCm tests pending';
 if(/seed|checkpoint|plateau|OOM|curricul|cap |stopping|terminal|update count/.test(t)&&/Expressivity|Limitations|Conclusion|Front matter/.test(section))return 'Task-dependent probe accuracy under recorded seed, cap, unequal steps, checkpoint and hardware conditions';
 if(/state.count|120 states|fewer.*states|capacity.*floor|recognizer|headroom/.test(t))return 'Architecture-agnostic finite-state floor/headroom; constrained realizability and optimization remain separate';
 if(/fixed.weight|non.equality|witness|finite.\$k|raw.write.*resource|row.column.cell|one.step|k.step|T\(d\)/.test(t))return 'Matched-signature abstract update-family separation; exact finite-k witness, with robust/task-specific tracking open';
 if(/split.gat|all.one|production.*gate|decay placement|core.*identit/.test(t))return 'Adjacent split-gated core identities on the abstract domain; production ranges/whole stack retain their own conditions';
 if(/G1|G2|G3|state.independent|schedule|routing|parameter.generat|shallower|depth/.test(t)&&/Formal|Limitations|Applications|Repeated nonlinear|Future/.test(section))return 'Executed nonlinear composition versus fixed-schedule affine fold; routing/representation-depth/production realizability scoped separately';
 if(/loss|bpb|bits.per.byte|perplex|held.out|token|equivalen|normalization/.test(t)&&!/Expressivity/.test(section))return 'Named-cohort descriptive loss measurements: training versus held-out normalization, aligned tokens, slice/seed uncertainty';
 if(/Applications|Repeated nonlinear/.test(section))return 'Prospective '+heading.replace(/^.*\]\s*/,'')+': structural surrogate test in the specified dynamics, error/encoding and stability regime';
 if(/Language.Modeling/.test(section))return 'Current-pair training-log loss null at aligned tokens after per-family selection; pending held-out evaluation';
 if(/Expressivity/.test(section))return heading.includes('8 M')||/8 M|S_3|three seeds/.test(t)?'8 M/from-scratch probe learning under named defaults, seed set, task and trained/evaluated lengths':'Named probe learning result and mechanism hypothesis under its recorded scale, task and optimization protocol';
 if(/Systems/.test(section))return 'Width-axis multi-programming: serial bounded per-head loop with the named geometry and measured resource protocol';
 if(/Architecture/.test(section))return 'E97 per-head update/stack and structural comparison at the specified key, gate, normalization and geometry domains';
 if(/Background/.test(section))return 'State-channel taxonomy by fixed-input affine/nonlinear update, with cited architecture and arithmetic assumptions';
 if(/Related Work/.test(section))return 'Positioning against the cited architecture, training setting and model scale';
 if(/Testable Predictions/.test(section))return 'Named future prediction, its specified budget/scale and explicit falsification condition';
 if(/Future Work/.test(section))return 'Open research target: '+heading;
 if(/Limitations/.test(section))return 'Scope of the named current-pair/structural claim; follow-up target '+heading;
 if(/Development history/.test(section))return 'Predecessor-only provenance and measured cohort result: '+heading;
 if(/Introduction|Conclusion|Front matter/.test(section))return 'Training viability and task-dependent learned state computation in the named architecture, scale and protocol';
 return 'Technical meaning of '+heading;
}
const kept=[];
for(const file of files){
 const lines=git(file).split('\n');let section=file===files[0]?'Front matter':'Imported current-pair captions',heading=section;
 lines.forEach((line,i)=>{
  if(/^= /.test(line)){section=line.slice(2);heading=section;}
  if(/^== |^#heading/.test(line))heading=line;
  if(!line.trim()||line.trim().startsWith('//'))return;
  if(!candidate.test(line)&&!(file===files[0]&&inherited.has(i+1)))return;
  if(/^(?:#let|let |else |if |return |\.\.|columns:|align:|stroke:|inset:|table\.header)/.test(line.trim()))return;
  if(edits.some(e=>(e.file??files[0])===file&&i+1>=e.baseLine&&i+1<=e.baseEndLine))return;
  const current=fs.readFileSync(file,'utf8');
  assert(current.includes(line.trim()),'Unchanged sweep candidate still exists: '+file+':'+(i+1));
  const context=lines.slice(Math.max(0,i-2),i+3).join('\n');
  kept.push({id:'K'+(kept.length+1),file,baseLine:i+1,section,heading,context,element:line.trim(),liveClaim:liveClaim(section,heading,context),decision:'keep',replacement:line.trim()});
 });
}
fs.writeFileSync(dir+'p18-kept-inventory.json',JSON.stringify(kept,null,2)+'\n');
const summary={base,edits:edits.length,retiredStandaloneElements:edits.filter(e=>e.decision==='retire').length,keptSweepLines:kept.length,displayEquationsByteIdentical:12,pseudocodeBlocksByteIdentical:8,standaloneTablesByteIdentical:2,figureTableLabelsPreserved:17,experimentalFigureTableBlocksByteIdentical:13,qualifiedTheoremSourceIdentifiersPreserved:new Set(names(old)).size,resultsAndFormalSourcesFrozen:true,sourceSha256:baseline.mainTypSha256,armor,inlineMentionDelta:net};
fs.writeFileSync(dir+'p18-preservation.json',JSON.stringify(summary,null,2)+'\n');
console.log('PASS: '+JSON.stringify(summary));
