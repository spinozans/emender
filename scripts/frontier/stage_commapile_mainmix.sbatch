#!/bin/bash
#SBATCH -A bif148
#SBATCH -J stage-commapile-mainmix
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o /lustre/orion/bif148/proj-shared/commapile/stage_commapile_mainmix_%j.log
#SBATCH -e /lustre/orion/bif148/proj-shared/commapile/stage_commapile_mainmix_%j.log

set -euo pipefail

SRC=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst
OUT=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
TMP=${OUT}.part.${SLURM_JOB_ID:-manual}
LOG=/lustre/orion/bif148/proj-shared/commapile/stage_commapile_mainmix_${SLURM_JOB_ID:-manual}.command.log

exec > >(tee -a "$LOG") 2>&1

echo "stage_start=$(date -Is)"
echo "host=$(hostname -f)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "pwd=$(pwd)"
echo "user=$(id)"
echo "src=$SRC"
echo "out=$OUT"
echo "tmp=$TMP"
echo "zstd=$(command -v zstd)"
zstd --version

echo "== source =="
ls -lh "$SRC"
stat -c 'source_path=%n source_size_bytes=%s source_mtime=%y' "$SRC"

if [[ -e "$OUT" ]]; then
  echo "ERROR: output already exists; refusing to overwrite without explicit completeness check: $OUT" >&2
  ls -lh "$OUT" >&2
  stat -c 'existing_output_path=%n existing_output_size_bytes=%s existing_output_mtime=%y' "$OUT" >&2
  exit 10
fi

if [[ -e "$TMP" ]]; then
  echo "ERROR: temp output already exists; refusing to overwrite: $TMP" >&2
  ls -lh "$TMP" >&2
  exit 11
fi

echo "== filesystem before =="
df -h /lustre/orion/bif148/proj-shared/commapile
lfs quota -h -g bif148 /lustre/orion || true
lfs quota -h -u erikgarrison /lustre/orion || true
du -sh /lustre/orion/bif148/proj-shared/commapile || true

echo "== decompress =="
decompress_start_epoch=$(date +%s)
decompress_start_iso=$(date -Is)
echo "decompress_start=$decompress_start_iso"
/usr/bin/time -v zstd -T0 -d --no-progress "$SRC" -o "$TMP"
sync
decompress_end_epoch=$(date +%s)
decompress_end_iso=$(date -Is)
echo "decompress_end=$decompress_end_iso"
echo "decompress_elapsed_seconds=$((decompress_end_epoch - decompress_start_epoch))"

echo "== output validation before rename =="
ls -lh "$TMP"
stat -c 'tmp_path=%n tmp_size_bytes=%s tmp_mtime=%y' "$TMP"
/usr/bin/time -v wc -c -l "$TMP"
echo "head_sample:"
head -n 2 "$TMP" | sed -e 's/[[:cntrl:]]/?/g' | cut -c 1-240
echo "tail_sample:"
tail -n 2 "$TMP" | sed -e 's/[[:cntrl:]]/?/g' | cut -c 1-240

mv -n "$TMP" "$OUT"

echo "== final output =="
ls -lh "$OUT"
stat -c 'output_path=%n output_size_bytes=%s output_mtime=%y' "$OUT"

echo "== filesystem after =="
df -h /lustre/orion/bif148/proj-shared/commapile
lfs quota -h -g bif148 /lustre/orion || true
lfs quota -h -u erikgarrison /lustre/orion || true
du -sh /lustre/orion/bif148/proj-shared/commapile || true

echo "stage_end=$(date -Is)"
