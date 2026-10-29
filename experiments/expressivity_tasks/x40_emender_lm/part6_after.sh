#!/usr/bin/env bash
# Run sequentially only after the surface driver exits.
set -uo pipefail
shopt -s nullglob
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root=experiments/expressivity_tasks/x40_emender_lm
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/erikg/emender/.venv/bin/python -c 'import itertools,json,pathlib;root=pathlib.Path("experiments/expressivity_tasks/x40_emender_lm/receipts/part6/surface");arms=[json.loads((root/f"{s}_{a}_B{b}_T{t}.json").read_text()) for s,a,b,t in itertools.product(("22M","150M","1.3B"),("pure","transformer"),(64,128),(512,2048))];assert all(r["status"]=="passed" and len(r.get("seconds",[]))>=20 for r in arms),"complete the 24-arm surface first"' || exit 1

qualify_if_gate() {
  local label=$1
  if /home/erikg/emender/.venv/bin/python -c 'import json,sys;r=json.load(open(sys.argv[1]));sys.exit(not(r["status"]=="passed" and r["tokens_per_second"]>=100000))' "$root/receipts/part6/${label}.json"; then
    "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_batch_curve --receipt "${label}.json" > "$root/receipts/part6/${label}_matched_curve.log" 2>&1
    local curve_status=$?
    echo "100k candidate $label matched curve exit=$curve_status"
    if (( curve_status == 0 )); then
      /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part6_decomposition --pilot "$root/receipts/part6/pilot_B128" --output "$root/receipts/part6/backward_decomposition.json"
      /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part6_decomposition --pilot "$root/receipts/part6/$label" --output "$root/receipts/part6/qualified_backward_decomposition.json"
      /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part6_summary
      exit 0
    fi
  fi
}
"$root/part6_baselines.sh" > "$root/receipts/part6/baselines.log" 2>&1
"$root/part6_memory_variants.sh" > "$root/receipts/part6/memory_variants.log" 2>&1 || exit 1
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1 X40_FUSED_INIT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for suite in x31_tape_cell_fused x36_tape_perf; do
  "$root/run_leased.sh" -m pytest -q -p experiments.expressivity_tasks.x40_emender_lm.oracle_adapter "experiments/expressivity_tasks/$suite/tests" --junitxml="$root/receipts/part6/${suite}.xml" > "$root/receipts/part6/${suite}_tests.log" 2>&1
  suite_status=$?
  echo "$suite tests exit=$suite_status"
  if (( suite_status != 0 )); then exit 1; fi
done
/home/erikg/emender/.venv/bin/python -m pytest -q "$root/tests" > "$root/receipts/part6/local_tests.log" 2>&1
local_status=$?
echo "local tests exit=$local_status"
if (( local_status != 0 )); then exit 1; fi
"$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_curve > "$root/receipts/part6/curve.log" 2>&1
curve_status=$?
echo "curve exit=$curve_status"
if (( curve_status != 0 )); then exit 1; fi
for chunk in 4096 16384; do
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_loss_gate --chunk "$chunk" > "$root/receipts/part6/loss_chunk${chunk}_gate.log" 2>&1
  echo "loss chunk $chunk gate exit=$?"
done
for batch in 64 128; do
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_stream_probe --batch "$batch" > "$root/receipts/part6/head_stream_gate_B${batch}.log" 2>&1
  echo "head stream B$batch gate exit=$?"
done
"$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_profile --batch 128 --bounded-loss --label ../part6/pilot_B128 --gate "/home/erikg/emender/$root/receipts/part6/production_T512_expandable.json" > "$root/receipts/part6/pilot_B128.log" 2>&1
echo "pilot B128 exit=$?"
if /home/erikg/emender/.venv/bin/python -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"]!="passed")' "$root/receipts/part6/head_stream_gate_B64_T512.json"; then
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_stream_probe --action curve --batch 64 --output "$root/receipts/part6/head_stream_curve" --bounded-loss --steps 64 > "$root/receipts/part6/head_stream_curve.log" 2>&1
  echo "head stream curve exit=$?"
fi
for batch in 64 128; do
  if /home/erikg/emender/.venv/bin/python -c 'import json,sys;sys.exit(any(json.load(open(p))["status"]!="passed" for p in sys.argv[1:]))' "$root/receipts/part6/head_stream_gate_B${batch}_T512.json" "$root/receipts/part6/head_stream_curve/summary.json"; then
    "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_stream_probe --action pilot --batch "$batch" --bounded-loss --label "../part6/pilot_head_stream_B${batch}" --gate "/home/erikg/emender/$root/receipts/part6/production_T512_expandable.json" > "$root/receipts/part6/pilot_head_stream_B${batch}.log" 2>&1
    echo "pilot head streams B$batch exit=$?"
  fi
done
"$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pool_fusion > "$root/receipts/part6/fused_pool_gate.log" 2>&1
fusion_status=$?
echo "integer pool fusion gate exit=$fusion_status"
if (( fusion_status == 0 )); then
  fixtures_status=0
  for suite in x31_tape_cell_fused x36_tape_perf; do
    "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pool_fusion --action tests --test-label "$suite" -q -p experiments.expressivity_tasks.x40_emender_lm.oracle_adapter "experiments/expressivity_tasks/$suite/tests" --junitxml="$root/receipts/part6/fused_pool_${suite}.xml" > "$root/receipts/part6/fused_pool_${suite}_tests.log" 2>&1
    suite_status=$?;echo "fused pool $suite fixtures exit=$suite_status"
    if (( suite_status != 0 )); then fixtures_status=1; fi
  done
  if (( fixtures_status == 0 )); then
    "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pool_fusion --action curve --output "$root/receipts/part6/fused_pool_curve" --bounded-loss --steps 64 > "$root/receipts/part6/fused_pool_curve.log" 2>&1
    fusion_curve=$?;echo "fused pool curve exit=$fusion_curve"
    if (( fusion_curve == 0 )); then
      for fusion in off on; do
        options=();if [[ "$fusion" == on ]]; then options+=(--fused-pool);fi
        label="pilot_B128_fusion_${fusion}_clear_registers0_chunk1024"
        "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --profile --batch 128 --no-checkpoint --label "$label" "${options[@]}" > "$root/receipts/part6/${label}.log" 2>&1
        echo "$label isolated fusion exit=$?";qualify_if_gate "$label"
      done
      for physical_batch in 128 256 512; do
        for mode in packed checkpoint; do
          options=(--packed --no-checkpoint);if [[ "$mode" == checkpoint ]]; then options=();fi
          for chunk in 1024 4096 16384; do
            if (( chunk != 1024 )) && ! /home/erikg/emender/.venv/bin/python -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"]!="passed")' "$root/receipts/part6/loss_chunk${chunk}_parity.json"; then continue;fi
            label="pilot_B${physical_batch}_fused_pool_${mode}_registers0_chunk${chunk}"
            "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --profile-if-gate --fused-pool --batch "$physical_batch" --loss-chunk "$chunk" --label "$label" "${options[@]}" > "$root/receipts/part6/${label}.log" 2>&1
            echo "$label exit=$?";qualify_if_gate "$label"
          done
        done
      done
    fi
  fi
fi
for registers in 128 64; do
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_register_probe --registers "$registers" > "$root/receipts/part6/registers${registers}_gate.log" 2>&1
  gate_status=$?
  echo "registers $registers gate exit=$gate_status"
  if (( gate_status == 0 )); then
    fixtures_status=0
    for suite in x31_tape_cell_fused x36_tape_perf; do
      "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_register_probe --registers "$registers" --action tests --test-label "$suite" -q -p experiments.expressivity_tasks.x40_emender_lm.oracle_adapter "experiments/expressivity_tasks/$suite/tests" --junitxml="$root/receipts/part6/registers${registers}_${suite}.xml" > "$root/receipts/part6/registers${registers}_${suite}_tests.log" 2>&1
      suite_status=$?
      echo "registers $registers $suite fixtures exit=$suite_status"
      if (( suite_status != 0 )); then fixtures_status=1; fi
    done
    if (( fixtures_status != 0 )); then continue; fi
    "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_register_probe --registers "$registers" --action curve --output "$root/receipts/part6/registers${registers}_curve" --bounded-loss --steps 64 > "$root/receipts/part6/registers${registers}_curve.log" 2>&1
    curve_status=$?
    echo "registers $registers curve exit=$curve_status"
    if (( curve_status == 0 )); then
      "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_register_probe --registers "$registers" --action pilot --batch 128 --bounded-loss --label "../part6/pilot_registers${registers}" --gate "/home/erikg/emender/$root/receipts/part6/production_T512_expandable.json" > "$root/receipts/part6/pilot_registers${registers}.log" 2>&1
      echo "registers $registers pilot exit=$?"
      # Physical-batch occupancy remains worth testing even if cross-stream
      # gradient accumulation fails its stricter byte gate.
      for physical_batch in 128 256 512; do
        for chunk in 1024 4096 16384; do
          if (( chunk != 1024 )) && ! /home/erikg/emender/.venv/bin/python -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"]!="passed")' "$root/receipts/part6/loss_chunk${chunk}_parity.json"; then continue; fi
          label="pilot_B${physical_batch}_default_stream_registers${registers}_chunk${chunk}"
          "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --profile-if-gate --batch "$physical_batch" --registers "$registers" --loss-chunk "$chunk" --label "$label" > "$root/receipts/part6/${label}.log" 2>&1
          echo "$label exit=$?"
          qualify_if_gate "$label"
        done
      done
      for physical_batch in 128 256 512; do
      "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_register_probe --registers "$registers" --head-streams --batch "$physical_batch" > "$root/receipts/part6/registers${registers}_head_streams_B${physical_batch}_gate.log" 2>&1
      combined_status=$?
      echo "registers $registers head streams B$physical_batch gate exit=$combined_status"
      if (( combined_status == 0 )); then
        if (( physical_batch <= 256 )); then
          for chunk in 1024 4096 16384; do
            if (( chunk == 1024 )) || /home/erikg/emender/.venv/bin/python -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"]!="passed")' "$root/receipts/part6/loss_chunk${chunk}_parity.json"; then
              "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --profile-if-gate --batch "$physical_batch" --registers "$registers" --head-streams --no-checkpoint --loss-chunk "$chunk" --label "pilot_B${physical_batch}_clear_gate_registers${registers}_chunk${chunk}" > "$root/receipts/part6/pilot_B${physical_batch}_clear_gate_registers${registers}_chunk${chunk}.log" 2>&1
              echo "clear gate B$physical_batch registers $registers chunk $chunk pilot exit=$?"
              qualify_if_gate "pilot_B${physical_batch}_clear_gate_registers${registers}_chunk${chunk}"
            fi
          done
        fi
        "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --profile-if-gate --batch "$physical_batch" --registers "$registers" --head-streams --label "pilot_B${physical_batch}_checkpoint_registers${registers}" > "$root/receipts/part6/pilot_B${physical_batch}_checkpoint_registers${registers}.log" 2>&1
        echo "checkpoint B$physical_batch registers $registers pilot exit=$?"
        qualify_if_gate "pilot_B${physical_batch}_checkpoint_registers${registers}"
        for chunk in 4096 16384; do
          if /home/erikg/emender/.venv/bin/python -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"]!="passed")' "$root/receipts/part6/loss_chunk${chunk}_parity.json"; then
            "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --profile-if-gate --batch "$physical_batch" --registers "$registers" --head-streams --loss-chunk "$chunk" --label "pilot_B${physical_batch}_checkpoint_registers${registers}_chunk${chunk}" > "$root/receipts/part6/pilot_B${physical_batch}_checkpoint_registers${registers}_chunk${chunk}.log" 2>&1
            echo "checkpoint B$physical_batch registers $registers chunk $chunk pilot exit=$?"
            qualify_if_gate "pilot_B${physical_batch}_checkpoint_registers${registers}_chunk${chunk}"
          fi
        done
      fi
      done
      "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_register_probe --registers "$registers" --head-streams --batch 64 --deferred-pack > "$root/receipts/part6/registers${registers}_head_streams_B64_deferred_pack_gate.log" 2>&1
      deferred_gate=$?
      echo "registers $registers deferred pack B64 gate exit=$deferred_gate"
      if (( deferred_gate == 0 )); then
        "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_register_probe --registers "$registers" --head-streams --batch 64 --deferred-pack --action curve --output "$root/receipts/part6/registers${registers}_deferred_curve" --bounded-loss --steps 64 > "$root/receipts/part6/registers${registers}_deferred_curve.log" 2>&1
        deferred_curve=$?
        echo "registers $registers deferred curve exit=$deferred_curve"
        if (( deferred_curve == 0 )); then
          for physical_batch in 128 256 512; do
            "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_register_probe --registers "$registers" --head-streams --batch "$physical_batch" --deferred-pack > "$root/receipts/part6/registers${registers}_head_streams_B${physical_batch}_deferred_pack_gate.log" 2>&1
            if (( $? == 0 )); then
              "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --profile-if-gate --batch "$physical_batch" --registers "$registers" --head-streams --no-checkpoint --deferred-pack --label "pilot_B${physical_batch}_deferred_registers${registers}" > "$root/receipts/part6/pilot_B${physical_batch}_deferred_registers${registers}.log" 2>&1
              echo "deferred B$physical_batch registers $registers pilot exit=$?"
              qualify_if_gate "pilot_B${physical_batch}_deferred_registers${registers}"
              for chunk in 4096 16384; do
                if /home/erikg/emender/.venv/bin/python -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"]!="passed")' "$root/receipts/part6/loss_chunk${chunk}_parity.json"; then
                  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --profile-if-gate --batch "$physical_batch" --registers "$registers" --head-streams --no-checkpoint --deferred-pack --loss-chunk "$chunk" --label "pilot_B${physical_batch}_deferred_registers${registers}_chunk${chunk}" > "$root/receipts/part6/pilot_B${physical_batch}_deferred_registers${registers}_chunk${chunk}.log" 2>&1
                  echo "deferred B$physical_batch registers $registers chunk $chunk pilot exit=$?"
                  qualify_if_gate "pilot_B${physical_batch}_deferred_registers${registers}_chunk${chunk}"
                fi
              done
            fi
          done
        fi
      fi
    fi
  fi
done
candidate_list=$(/home/erikg/emender/.venv/bin/python -c 'import json,pathlib;root=pathlib.Path("experiments/expressivity_tasks/x40_emender_lm/receipts/part6");r=[(json.loads(p.read_text()),p.name) for p in root.glob("pilot_B*_registers*.json")];v=[(d["tokens_per_second"],n) for d,n in r if d["status"]=="passed" and not (root/("matched_"+n.removesuffix(".json"))).exists()];print("\n".join(n for _,n in sorted(v,reverse=True)))')
while IFS= read -r receipt; do
  [[ -n "$receipt" ]] || continue
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_batch_curve --receipt "$receipt" > "$root/receipts/part6/${receipt%.json}_matched_curve.log" 2>&1
  curve_status=$?
  echo "matched batch curve $receipt exit=$curve_status"
  if (( curve_status == 0 )); then break; fi
done <<< "$candidate_list"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part6_decomposition --pilot "$root/receipts/part6/pilot_B128" --output "$root/receipts/part6/backward_decomposition.json"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part6_summary
