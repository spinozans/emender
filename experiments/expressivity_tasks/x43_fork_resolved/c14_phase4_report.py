"""Produce a scale scenario from terminal native timings, never a full-model claim."""
import json
from pathlib import Path
from .c14_documents import sha
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/scalability_phase4'

def main():
    assert json.loads((OUT/'terminal.json').read_text())['exit_code']==0
    assert json.loads((OUT/'boundary_terminal.json').read_text())['exit_code']==0
    rows=json.loads((OUT/'profile.json').read_text())['rows'];boundary=json.loads((OUT/'boundary.json').read_text())['rows']
    assert len(rows)==6 and len(boundary)==4 and all(r['status']=='timed_repeat_exact_not_certified' for r in rows+boundary)
    terminal=next(r for r in rows if r['tokens']==1024 and r['batch']==1)
    seconds=terminal['median_ms']/1000;sec_per_byte=seconds*22*2*2/1024
    need=2e9/86400;rate=1/sec_per_byte;days=2e9*sec_per_byte/86400
    cost=dict(status='not_qualified_named_kernel_internal_blocker',measured_native_head_seconds=seconds,assumptions=dict(layers=22,heads=2,native_passes=2,context=1024,targets=2000000000,wall_budget_seconds=86400,controlled_peak_live=257),native_seconds_per_byte=sec_per_byte,native_scenario_targets_per_second=rate,native_scenario_days_for_2B=days,required_targets_per_second=need,required_scenario_speedup=need/rate,fp32_parameter_gradient_adam_minimum_bytes=1300000000*16,full_model_measured=False,omissions=['backward/replay','dense/FFN/projection','decode and feature assembly','telemetry','optimizer','checkpoint/setup/eval','full-scale memory and natural occupancy'],scope='Native-only controlled forward + recomputation scenario, not a certified training bound. No1.3B training steps measured.',sources={str(OUT/n):sha(OUT/n) for n in ('profile.json','boundary.json','registration.json','boundary_registration.json')})
    (OUT/'tractability.json').write_text(json.dumps(cost,indent=2)+'\n')
    lines=['# X40/X43 scalability phase4 — 1.3B tractability verdict','',
      '**The registered1.3B/24h class is not qualified. The named blocker requires kernel-internals work; no wrapper-only fix is promoted.**','',
      'Physical card4 via detached broker wrappers. No model training, checkpoint, frozen arm, union service, card1, sxgc or mounted corpus changes. Every row below is native-forward only; repeated fingerprints and controlled telemetry pass, without long-context reference/gradient certification.','',
      '| B | T | cap | BS | median ms | registers | compiler spills |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows+boundary,key=lambda r:(r['tokens'],r['batch'])):
        bs=1<<(min(r['cap'],2*r['tokens'])-1).bit_length();lines.append(f"|{r['batch']}|{r['tokens']}|{r['cap']}|{bs}|{r['median_ms']:.6f}|{r['registers']}|{r['spills']}|")
    lines+=['','Residual spills already exist at C14 T128/BS256 (862); “return” is an amplification of an existing spill wall. The lower-context sweep brackets the transition. Counts are compiler metadata, not hardware spill transactions. Nsight hardware counters remain unavailable per phase2 ERR_NVGPUCTRPERM.','',
      f"Terminal B1/T1024 native head median is {seconds:.9f}s. Math: {seconds:.9f} *22 layers *2 heads *2 native passes /1024 = {sec_per_byte:.9f}s/byte; {rate:.6f} targets/s;2e9 targets => {days:.3f} native-only days. Required2e9/86400 = {need:.6f} targets/s, a {need/rate:.1f}x scenario gap. This is a controlled peak257 workload and a raw-byte budget-count scenario; the historical terminal corpus was GPT2 tokens. No token/byte equivalence is claimed.",
      '', 'The single-sequence grid launches one CTA on142 SMs. At8 warps/CTA, analytic global warp occupancy is at most8/(142*48)=0.1174%, versus1.878% at B16. Measured B16/B1 rows compare throughput at fixed shape; batching changes the B1 protocol and is not a free solution.',
      '', 'The reachable wrapper already scans nextpow2(min(cap,2T)); reducing cap would violate unbounded capacity. The kernel hard-codes8-warp Gluon layouts; increasing num_warps alone cannot change those layouts legally. At BS2048 the full value vector has2048*128/256=1024 FP32 elements/thread before keys and metadata, versus at most255 registers/thread. Global staging between tokens still reloads a full BSx128 value tensor within each token. A spill-free long-context path needs tiled value reads/updates with an exact reduction tree and bounded intermediate lifetime; distributing a sequence across CTAs also needs deterministic inter-CTA state/update synchronization. These are internal design changes, not certified wrapper knobs.',
      '', 'X31 bill: implement and retain the new kernel separately; rerun the109-oracle/105-C14-operator regression gate; exact nonzero read/state/trace and replay/gradient fingerprints including tie/POP/CORRECT/MERGE/DECAY cases; add B1/T512/1024/2048 with cap2T+1, high-live fixtures, masked tails and document resets; rerun >=2M real masked targets with every loss/gradient/optimizer/RNG/cursor/checkpoint-resume fingerprint; then measure150M and1.3B-width full training throughput and peak memory. Existing20.8GB FP32 parameter/gradient/Adam arithmetic excludes activations/store/replay. Engineering time and GPU-hour cost for that rewrite are unmeasured, so no invented numeric fix price is reported.',
      '', 'Attempt1 is retained as failure: B16 fixture had B1 logits, rejected before valid measurement. Own process group stopped, wrapper exit143. Corrected fixture was pushed before rerun. Lower-context supplement was independently registered before measurement. No kernel internals were edited or certified here. Fresh document/branch tests:9 passed. No resilient DiLoCo behavior changes; gap-matrix IDs inapplicable.',
      '', 'Residual risks: full1.3B training and natural-scale occupancy/memory remain unmeasured; repeated native fingerprints do not certify long-context gradients; the controlled schedule may differ from training. Kernel rewrite cost is a named validation bill, not a measured duration estimate.']
    (ROOT/'RESULT_UNBOUNDED_SCALABILITY_PHASE4.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(cost,indent=2))

if __name__=='__main__':main()
