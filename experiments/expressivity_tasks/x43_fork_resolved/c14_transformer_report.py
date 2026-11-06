"""Measured head-asymmetry verdict, retaining signs and registered strata."""
import json
from pathlib import Path
from .c14_documents import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/c14_transformer_eval'


def report():
    table=json.loads((OUT/'table.json').read_text())
    emender=json.loads((ROOT/'receipts/c14_paired_eval/table.json').read_text())
    training=json.loads((ROOT/'receipts/c14_transformer/training.json').read_text())
    dense=json.loads((ROOT/'receipts/c14_dense/training.json').read_text())['models'][1]
    def value(arm,key,half='final_test',t=table):return t[f'{arm}/{half}/{key}']['nll']
    keys=['total','window_reset_1_16','window_interior_17_128','wikipedia_support','empty_support','supported_target','nonempty_zero_target','zero_target','agreement','argmax_agreement','fork']
    gains={k:value('dense',k)-value('v2_on',k) for k in keys}
    emender_gains={k:value('dense',k,t=emender)-value('v2_on',k,t=emender) for k in keys}
    emender_gain=emender_gains['total']
    p1=gains['total']>0 and all(gains[k]>gains['zero_target'] for k in ('wikipedia_support','supported_target','agreement'))
    off_delta=value('v2_off','total')-value('dense','total')
    verdict=dict(primary='final_test',P1='PASS' if p1 else 'FAIL',P2='PASS' if abs(off_delta)<=.02 else 'FAIL',gains_dense_minus_on=gains,
        off_minus_dense=off_delta,emender_total_gain=emender_gain,
        transformer_exposure_ratio_to_dense=training['actual_bytes_targets']/dense['targets'],
        emender_gains_dense_minus_on=emender_gains,transformer_minus_emender_gains={k:gains[k]-emender_gains[k] for k in keys},
        support_minus_empty_gain=gains['wikipedia_support']-gains['empty_support'],
        supported_minus_nonempty_zero_gain=gains['supported_target']-gains['nonempty_zero_target'],transformer_minus_emender_gain=gains['total']-emender_gain,
        transformer_consulted_minus_emender_consulted=value('v2_on','total')-value('v2_on','total',t=emender),
        transformer_supported_minus_emender_supported=value('v2_on','supported_target')-value('v2_on','supported_target',t=emender),
        emender_supported_advantage_holds=value('v2_on','supported_target',t=emender)<value('v2_on','supported_target'),
        training={k:training[k] for k in ('steps','actual_bytes_targets','active_seconds','wall_seconds')},
        historical_dense_final_nll=dense['heldout']['final_test']['nll'],
        dense_replay_minus_historical=value('dense','total')-dense['heldout']['final_test']['nll'],
        gain_against_historical_dense=dense['heldout']['final_test']['nll']-value('v2_on','total'),
        table_sha256=sha(OUT/'table.json'),emender_table_sha256=sha(ROOT/'receipts/c14_paired_eval/table.json'))
    (OUT/'verdicts.json').write_text(json.dumps(verdict,indent=2)+'\n')
    lines=['# C14 transformer consultation: the head-asymmetry verdict','',
        'Primary results use the frozen document-disjoint final-test half. NLL is nats/byte. Positive gain means dense minus device-on; every real target remains in the total.', '',
        f"Transformer P1 **{verdict['P1']}**, P2 **{verdict['P2']}**. Total gain {gains['total']:+.9f}; device-off minus dense {off_delta:+.9f} (P2 limit ±.02).",'',
        f"The Emender gain is {emender_gain:+.9f}; transformer minus Emender gain is {gains['total']-emender_gain:+.9f}. Consulted transformer minus consulted Emender total NLL is {verdict['transformer_consulted_minus_emender_consulted']:+.9f}.",'',
        f"Emender supported-target advantage against the consulted transformer: **{'HOLDS' if verdict['emender_supported_advantage_holds'] else 'DOES NOT HOLD'}**. Transformer minus Emender supported-target NLL: {verdict['transformer_supported_minus_emender_supported']:+.9f}.", '',
        f"Historical dense transformer: {dense['heldout']['final_test']['nll']:.9f}; paired batch-128 replay minus historical batch-16 NLL: {verdict['dense_replay_minus_historical']:+.9f}. Gain against the historical dense reference: {verdict['gain_against_historical_dense']:+.9f}. No prediction byte parity with the historical eval is claimed.", '',
        '| Final-test stratum | Targets | Transformer dense | Transformer on | Transformer off | Gain | Emender v2 on | Emender gain | Transformer − Emender gain |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key in keys:
        n=table[f'dense/final_test/{key}']['targets']
        lines.append(f"| {key} | {n:,} | {value('dense',key):.9f} | {value('v2_on',key):.9f} | {value('v2_off',key):.9f} | {gains[key]:+.9f} | {value('v2_on',key,t=emender):.9f} | {emender_gains[key]:+.9f} | {gains[key]-emender_gains[key]:+.9f} |")
    lines+=['','At the registered budget, the transformer gains less from the array. Its consulted total and supported-target NLL are lower than the Emender’s. Consultation hurts zero-target positions in both heads.', '', '## Training exposure and cost','',
        '| Arm | Steps | Targets | Active seconds |','| --- | ---: | ---: | ---: |',
        f"| Transformer dense | {dense['steps']:,} | {dense['targets']:,} | {dense['active_seconds']:.6f} |",
        f"| Transformer consultation | {training['steps']:,} | {training['actual_bytes_targets']:,} | {training['active_seconds']:.6f} |",'',
        'This is a single-seed fixed-time comparison with unequal exposure; it does not isolate architecture at matched training targets. The registered rule is 2700 active seconds, with identical source-order stream from window zero, seed 3401, batch 16/context 128, BF16, AdamW 3e-4/decay .01/clip 1 and independent window resets. Consultation uses keep=.5 joint dropout across all 260 input features per position, without rescaling. Exposure differences are measured consequences of the fixed active budget.', '',
        f"Training wall time: {training['wall_seconds']:.3f}s; excluded checkpoint save/hash stalls: {sum(x['seconds'] for x in training['excluded_stalls']):.3f}s. Full extraction is prepaid and separately receipted in union_service.jsonl; this is not an end-to-end extraction-inclusive speed claim.", '',
        f"Paired inference loop elapsed: {json.loads((OUT/'resume.json').read_text())['elapsed_seconds']:.3f}s across all three modes, separately from training active time.", '',
        'The transformer uses its registered attention head and the shared byte-exact v2 feature decoder. Phase 1–3 Emender store-kernel evidence is hash-bound, unchanged; those kernels do not execute in the transformer head. No resilient DiLoCo behavior was changed or run; its requirement IDs are not applicable.', '',
        '## Calibration and combined totals','',
        '| Half | Dense | On | Off | Targets |','| --- | ---: | ---: | ---: | ---: |']
    for half in ('calibration','all'):
        lines.append(f"| {half} | {value('dense','total',half):.9f} | {value('v2_on','total',half):.9f} | {value('v2_off','total',half):.9f} | {table[f'dense/{half}/total']['targets']:,} |")
    lines+=['','All strata, sums and denominators for calibration/final-test/all are retained in receipts/c14_transformer_eval/table.json. P3 calibration stacking is outside this head-asymmetry arm; no P3 result is claimed. Extraction parity uses retained oracle attestations rather than a fresh native-query rerun.']
    (ROOT/'RESULT_C14_TRANSFORMER.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verdict))

if __name__=='__main__':report()
