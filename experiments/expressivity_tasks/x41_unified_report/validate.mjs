import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const dir='experiments/expressivity_tasks/x41_unified_report/';
const base='experiments/expressivity_tasks/';
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const get=(x,p)=>p.split('/').slice(1).reduce((v,k)=>v[k.replace(/~1/g,'/').replace(/~0/g,'~')],x);
const read=p=>JSON.parse(fs.readFileSync(base+p));
const ledger=JSON.parse(fs.readFileSync(dir+'claims.json'));
const paper=fs.readFileSync(dir+'PAPER.md','utf8');
const claimsMarkdown=fs.readFileSync(dir+'CLAIMS.md','utf8');
const reconciliation=fs.readFileSync('paper/RECONCILIATION.md','utf8');
assert.deepEqual([...paper.matchAll(/^## (\d+)\. (.*)$/gm)].map(m=>m[2]),[
 'The mechanism: emendation as a write program',
 'Motivation measured at full scale: the 1.3B null',
 'The certified boundary',
 'The cost account: ordered state and sequentiality',
 'LM pilots: lane activity and architectural design science',
 'Open: scale, replication and the pure model',
]);
for(const c of ledger.claims)assert(claimsMarkdown.includes(`id="${c.id.toLowerCase()}"`),`Missing claim anchor ${c.id}`);
for(const disposition of ['stands (with pointer)','MIXED (X26c)','refuted-sign-reversed (X27)','subsumed-by (the Emender results)','still-valid-methodology'])assert(reconciliation.includes(disposition));
const provenance=JSON.parse(fs.readFileSync(dir+'receipts/provenance.json'));
for(const s of provenance.files)assert.equal(hash(fs.readFileSync(s.snapshot)),s.sha256,s.snapshot);
let pointers=0,sources=0;
for(const c of ledger.claims){assert(paper.includes(`CLAIMS.md#${c.id.toLowerCase()}`),`Unused ${c.id}`);for(const s of c.sources){const raw=fs.readFileSync(s.path,'utf8');assert.equal(hash(raw),s.sha256,s.path+' changed');assert(raw.split('\n')[s.line-1].includes(s.needle),s.path+':'+s.line);if(s.pointer){assert.deepEqual(get(JSON.parse(raw),s.pointer),s.value);pointers++}sources++}}
for(const ref of paper.matchAll(/CLAIMS.md#(c\d+)/g))assert(ledger.claims.some(c=>c.id.toLowerCase()===ref[1]));
const derivations={};
derivations.boundary={};
for(const arm of ['tape_B0','tape_B0x4','gdn2_native','e97_native']){
 const r=read('results_x28e_memory_head/'+arm+'_seed0/final_eval.json');
 derivations.boundary[arm]=Object.fromEntries(['bin_id','bin_2x','bin_4x','bin_9x'].map(k=>[k,r[k].report.balanced_accuracy]));
}
assert.equal(derivations.boundary.tape_B0.bin_2x,.98046875);
assert.equal(derivations.boundary.tape_B0x4.bin_9x,.947265625);
derivations.pressure={};
for(const arm of ['tape_B0','tape_B0x4']){
 const rows=fs.readFileSync(base+'x33_pressure_probe/data/'+arm+'_per_example.jsonl','utf8').trim().split('\n').map(JSON.parse);
 const accepts=rows.filter(r=>r.label===1), fired=accepts.filter(r=>r.n_rcap>0), wrong=accepts.filter(r=>!r.correct), tp=fired.filter(r=>!r.correct).length;
 const sweep=accepts.filter(r=>r.pool.startsWith('sweep_k'));
 const onset=Math.min(...sweep.filter(r=>r.n_rcap>0).map(r=>r.k));
 const collapse=Math.min(...[...new Set(sweep.map(r=>r.k))].filter(k=>sweep.filter(r=>r.k===k).every(r=>!r.correct)));
 derivations.pressure[arm]={all_words:rows.length,accept_words:accepts.length,fired:fired.length,incorrect:wrong.length,true_positive:tp,ppv:tp/fired.length,sensitivity:tp/wrong.length,false_alarms:fired.length-tp,onset,first_accept_collapse_in_all_sweeps:collapse,saturation_positions:[...new Set(accepts.map(r=>r.first_sat).filter(k=>k>=0))]};
}
assert.equal(derivations.pressure.tape_B0.ppv,1);assert.equal(derivations.pressure.tape_B0.sensitivity,1);assert.equal(derivations.pressure.tape_B0.all_words,6400);
const large=fs.readFileSync(base+'x33_pressure_probe/data/tape_B0x4_per_example.jsonl','utf8').trim().split('\n').map(JSON.parse).filter(r=>r.label===1&&r.pool.startsWith('sweep_k')&&r.k>=2816);
const largeCollapse=Math.min(...large.filter(r=>!r.correct).map(r=>r.k));
derivations.pressure.b0x4_own_window_lead=largeCollapse-derivations.pressure.tape_B0x4.onset;assert.equal(derivations.pressure.b0x4_own_window_lead,96);
derivations.usage=read('x39_memory_usage/receipts/part9_adjudication.json').rows.map(r=>({arm:r.arm,perplexity:r.perplexity,ratio:r.ratio,slopes:r.slopes,mean_slope:r.slopes.reduce((a,b)=>a+b,0)/r.slopes.length}));
const fam=read('x39_memory_usage/receipts/part6_family_comparability.json');const control=fam.rows[0].b64_control_perplexity;
const x37=read('x37_head_differentiation/receipts/summary.json');
derivations.family=Object.entries(x37.arms).map(([arm,r])=>({arm,tokens:r.training.tokens,batch:r.training.fingerprint.batch,identical_stories:r.primary.identical_stories,perplexity:r.validation.perplexity,ratio:r.validation.perplexity/control}));
assert(derivations.family.every(r=>r.tokens===100007936&&r.batch===64));
assert.equal(x37.arms.d.primary.identical_stories,0);
assert(x37.arms.d.validation.perplexity<x37.arms.c.validation.perplexity);
derivations.x34={};
for(const arm of ['tape_lm','base_lm']){const r=read('x34_lm_pilot/receipts/pilot_'+arm+'.json'),last=r.curve.at(-1);derivations.x34[arm]={tokens:r.tokens,loss:last.loss,nll:last.validation.nll,perplexity:last.validation.perplexity}}
assert(derivations.x34.tape_lm.loss<derivations.x34.base_lm.loss);assert(derivations.x34.tape_lm.perplexity>derivations.x34.base_lm.perplexity);
const pre=read('x39_memory_usage/receipts/preflight.json');
derivations.x34_occupancy=pre.control.rows.map(r=>({T:r.context,peak:r.peak_live_per_head}));
assert(derivations.x34_occupancy.every(r=>r.peak.every(n=>n===r.T/2)));
const pair=read('x41_unified_report/receipts/matched_pair.json');
const comparison=pair.comparison;
assert.equal(comparison.cutoff.tokens,150793420800);
assert.equal(comparison.alignment.aligned_record_count,92037);
assert.equal(comparison.alignment.missing_or_nonoverlapping.e97_only_count,0);
assert.equal(comparison.alignment.missing_or_nonoverlapping.gdn2_only_count,0);
assert.equal(pair.token_semantics.tokens_per_step,pair.e97_token_semantics.tokens_per_step);
assert.equal(comparison.cutoff.tokens,comparison.cutoff.step*pair.token_semantics.tokens_per_step);
const windows=comparison.intervals;
assert(windows.last_100_optimizer_steps.delta_gdn2_minus_e97>0);
assert(windows.last_1000_optimizer_steps.delta_gdn2_minus_e97<0);
derivations.inherited_pair={cutoff:comparison.cutoff,alignment:comparison.alignment,
 raw:comparison.cutoff_raw,smoothed:comparison.cutoff_smoothed,
 intervals:Object.fromEntries(Object.entries(windows).map(([k,r])=>[k,{delta:r.delta_gdn2_minus_e97,e97:r.e97.mean_loss_observed_records,gdn2:r.gdn2.mean_loss_observed_records,observations:r.e97.observation_count}]))};
const bpb=JSON.parse(fs.readFileSync('experiments/lb_compare_20260613/bpb_results.json'));
assert.equal(bpb.length,5);
const bpbs=bpb.map(r=>Number(r.heldout_bpb_nonavg));
derivations.inherited_band={families:bpb.length,min:Math.min(...bpbs),max:Math.max(...bpbs),width:Math.max(...bpbs)-Math.min(...bpbs),tokens:bpb.map(r=>Number(r.heldout_tokens))};
assert.equal(derivations.inherited_band.width.toFixed(4),'0.0887');
assert(derivations.inherited_band.tokens.every(n=>n===131072));
assert(Number(bpb.find(r=>r.label==='emender-mlp').heldout_bpb_nonavg)<Number(bpb.find(r=>r.label==='gdn2-mlp').heldout_bpb_nonavg));
derivations.inherited_cma=Object.fromEntries(['e1','e88','fla-gdn','mamba2'].map(n=>{const r=read('x41_unified_report/receipts/cma_'+n+'.json');return[n,{best_loss:r.best_loss,target_params:r.target_params,total_evals:r.total_evals}]}));
for(const n of Object.keys(derivations.inherited_cma))assert.equal(derivations.inherited_cma[n].target_params,480000000);
const report={status:'passed',claims:ledger.claims.length,source_pointers:sources,json_pointers:pointers,derivations,scope:'Read-only receipt audit; no model reruns, Lean rebuild, GPU, Python, Slurm or resilient DiLoCo behavior. Semantic entailment requires manual review.'};
if(process.argv.includes('--write'))fs.writeFileSync(dir+'verification.json',JSON.stringify(report,null,2)+'\n');
else assert.deepEqual(JSON.parse(fs.readFileSync(dir+'verification.json')),report,'Verification derivations differ');
console.log(JSON.stringify({status:report.status,claims:report.claims,source_pointers:sources,json_pointers:pointers,pressure_lead:derivations.pressure.b0x4_own_window_lead,usage:derivations.usage.map(r=>({arm:r.arm,mean_slope:r.mean_slope,ratio:r.ratio}))},null,2));
