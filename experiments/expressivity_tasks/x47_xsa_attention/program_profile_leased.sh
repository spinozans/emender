#!/usr/bin/env bash
# Bounded detached training-path profiles. Each protocol has a separate CPU gate and cumulative cap.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
extra=(); kind=${2:-p1}; case "$kind" in
  p1) cap=200; gate=41; module=memory_program_profile;;
  p2) cap=320; gate=47; module=packed_program_profile;;
  s1) cap=420; gate=55; module=serving_bench;;
  s2) cap=260; gate=59; module=serving_bench; extra=(--cohort);;
  s3) cap=200; gate=65; module=cached_host_bench;;
  s4) cap=360; gate=67; module=large_host_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s5) cap=260; gate=69; module=host_graph_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s6) cap=120; gate=72; module=resident_host_check; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s7) cap=240; gate=74; module=wide_host_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s8) cap=150; gate=76; module=controller_placement; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  t1) cap=220; gate=80; module=host_vjp_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s9) cap=170; gate=81; module=long_prefill_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s10) cap=175; gate=82; module=segment_prefill_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s11) cap=150; gate=84; module=width_cover_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s12) cap=150; gate=88; module=arrival_service_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  t2) cap=155; gate=92; module=sparse_clock_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s13) cap=115; gate=94; module=open_cohort_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  t3) cap=145; gate=97; module=frozen_packet_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  t4) cap=160; gate=99; module=stream_packet_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  t6) cap=115; gate=103; module=segment_packet_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s14) cap=115; gate=129; module=direct_arrival_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s17) cap=130; gate=170; module=demand_tape_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s19) cap=150; gate=208; module=onclock_mount_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  s20) cap=160; gate=232; module=demand_position_bench; export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;;
  *) exit 2;;
esac
root=output/memory-program-$kind; dir=$root/attempt${1:?attempt required}; D=experiments/expressivity_tasks/x47_xsa_attention
[[ $1 =~ ^[1-9][0-9]*$ && -d $dir && ! -e $dir/initial && -z ${GPU_LEASE_HELD:-} && $(hostname) == lambda01 ]]
for i in {1..50}; do [[ $(ps -o ppid= -p $$ | tr -d ' ') == 1 ]] && break; sleep .1; done
grep -q '^PPid:[[:space:]]*1$' /proc/$$/status; grep -E '^(Pid|PPid):' /proc/$$/status > "$dir/detachment"
git diff --quiet HEAD -- ndm/memory_program "$D" tests/test_memory_program*.py; git rev-parse HEAD > "$dir/ref"
cmp "$dir/ref" "$root/gated-ref"
# Match the final pytest summary, not substrings of successfully executed test names.
tail -n 1 "$root/qualification.txt" | grep -Eq "^=+ ${gate} passed in [0-9.]+s =+$" || { echo "invalid CPU gate summary" >&2; exit 1; }
used=0
for old in "$root"/attempt*/initial; do [[ -f $old ]] || continue; parent=${old%/initial}; [[ -f $parent/released ]]; if [[ $kind != p1 ]] && grep -qx 'rc=0' "$parent/released"; then echo "$kind already completed"; exit 1; fi; created=$(awk -F= '$1=="created"{print $2}' "$old"); released=$(awk -F= '$1=="released_epoch"{print $2}' "$parent/released"); [[ $released -ge $created ]]; used=$((used+released-created+1)); done
limit=$((cap-used-12)); [[ $limit -gt 15 ]]
if [[ $kind == s19 ]]; then
  # This admission spends at most 150 of the 371 seconds remaining after sealed T13.
  cmp "$root/source-ref" "$root/gated-ref"
  [[ $(git rev-parse HEAD^) == a50d250e4dec26af56b8ec694a2a16226a7087d3 && $(<"$root/protocol-ref") == $(git rev-parse HEAD^) ]]
  printf '%s  %s\n' e05ec80992e93b4ed919030fc0f2493585423726d2b801fed2e1f575b1982c4a "$D/receipts/memory-program-t13/verified.json" | sha256sum --check -
fi
if [[ $kind == s20 ]]; then
  cmp "$root/source-ref" "$root/gated-ref"
  [[ $(git rev-parse HEAD^) == accec6181aad44ed70edce4c674c4c3d3976736c && $(<"$root/protocol-ref") == $(git rev-parse HEAD^) ]]
  printf '%s  %s\n' 369064a20f2326618ee90bc52548922e6fbde5bac73428e708705f936dae1b39 "$D/receipts/memory-program-s19/verified.json" | sha256sum --check -
fi
if [[ $kind == s14 || $kind == s17 || $kind == s19 || $kind == s20 ]]; then
  [[ $1 == 1 && $used == 0 ]]
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x47_xsa_attention.direct_service_head check output/memory-program-s14
fi
if [[ $kind =~ ^(t1|t2|t3|t4|t6|s9|s10|s11|s12|s13|s14|s17|s19|s20)$ ]]; then [[ $(($(date +%s)+limit+12)) -le 1795531800 ]]; fi
export OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=2 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait); eval "$lease"; file=${GPU_LEASE_DIR:-.wg/gpu_leases}/$GPU_LEASE_HELD
finish(){ rc=$?; set +e; cp "$file" "$dir/last"; gpu_lease_release; if grep -qx "pid=$$" "$file" 2>/dev/null; then rc=1; fi; printf 'released_epoch=%s\nrc=%s\n' "$(date +%s)" "$rc" > "$dir/released.tmp"; mv "$dir/released.tmp" "$dir/released"; exit "$rc"; }; trap finish EXIT
[[ $GPU_LEASE_HELD =~ ^[01]$ && $CUDA_VISIBLE_DEVICES == "$GPU_LEASE_HELD" ]]; cp "$file" "$dir/initial"
grep -qx "pid=$$" "$dir/initial"; grep -qx "starttime=$(awk '{sub(/.*\) /, ""); print $20}' /proc/$$/stat)" "$dir/initial"
timeout -k 3 "$limit" /home/erikg/emender/.venv/bin/python -m "experiments.expressivity_tasks.x47_xsa_attention.$module" "$dir" "${extra[@]}"
if [[ $kind != p1 ]]; then
  for i in {1..25}; do [[ -s $dir/watch-ack ]] && break; sleep .1; done
  [[ -s $dir/watch-ack ]]
fi
cp "$file" "$dir/after"; [[ $(awk -F= '$1=="heartbeat"{print $2}' "$dir/after") -gt $(awk -F= '$1=="heartbeat"{print $2}' "$dir/initial") ]]
