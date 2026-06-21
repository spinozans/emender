# Frontier Commapile Mainmix Staging Log - 2026-06-21

Task: `frontier-stage-commapile-mainmix`

## Decision

Stage the decompressed commapile mainmix text under the existing project-shared
dataset directory:

`/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`

Reasoning:

- The compressed source already lives in
  `/lustre/orion/bif148/proj-shared/commapile` and is group-readable by
  `bif148`, which is the appropriate scope for downstream Frontier training.
- The operation is a sustained 235 GB read plus roughly 1 TB write, so it was
  submitted as a one-node SLURM batch job instead of being run directly on
  `login04.frontier.olcf.ornl.gov`.
- No existing decompressed target was found before submission. The job writes
  to a `.part.<jobid>` path and uses `mv -n` only after validation, so an
  existing final output is not overwritten.

## Preflight

Commands run from:

`/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-17`

Host and tools:

```text
hostname -f
login04.frontier.olcf.ornl.gov

date -Is
2026-06-21T08:39:48-04:00

command -v sbatch
/usr/bin/sbatch

command -v zstd
/usr/bin/zstd
```

Source file:

```text
ls -lh /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst
-rw-r--r-- 1 erikgarrison bif148 235G Jun 21 02:21 /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst

stat -c 'source_size_bytes=%s source_mtime=%y' /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst
source_size_bytes=251655400225 source_mtime=2026-06-21 02:21:57.000000000 -0400
```

Existing-output check:

```text
find /lustre/orion/bif148/proj-shared/commapile /lustre/orion/bif148/scratch/erikgarrison -maxdepth 4 \( -name 'commapile_mainmix_v0.1_1tb.txt' -o -name 'commapile_mainmix_v0.1_1tb.txt.zst' -o -name '*commapile*mainmix*' \) -printf '%p\t%s bytes\t%TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null | sort
/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst 251655400225 bytes 2026-06-21 02:21:57.0000000000
```

Filesystem and quota before staging:

```text
df -h /lustre/orion/bif148/proj-shared /lustre/orion/bif148/scratch/erikgarrison /lustre/orion/bif148/scratch
Filesystem                                      Size  Used Avail Use% Mounted on
10761@kfi,10824@kfi:10825@kfi,10760@kfi:/orion  604P  308P  296P  52% /lustre/orion

lfs quota -h -g bif148 /lustre/orion
Disk quotas for grp bif148 (gid 31692):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
  /lustre/orion  230.3G      0k      0k       -   11461       0       0       -
gid 31692 is using default block quota setting
gid 31692 is using default file quota setting

lfs quota -h -u erikgarrison /lustre/orion
Disk quotas for usr erikgarrison (uid 19032):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
  /lustre/orion  234.5G      0k      0k       -   64567       0       0       -
uid 19032 is using default block quota setting
uid 19032 is using default file quota setting

du -sh /lustre/orion/bif148/proj-shared/commapile /lustre/orion/bif148/scratch/erikgarrison
230G /lustre/orion/bif148/proj-shared/commapile
4.8G /lustre/orion/bif148/scratch/erikgarrison
```

## Batch Command

Script:

`scripts/frontier/stage_commapile_mainmix.sbatch`

Submit commands:

```bash
sbatch scripts/frontier/stage_commapile_mainmix.sbatch
```

Notes:

- First submission was rejected by SLURM because the requested 4 hour walltime
  exceeded the account/queue limit of 120 minutes.
- A second submission, job `4880566`, was canceled before it started because it
  had inherited the default 1 CPU request.
- Final successful submission used `#SBATCH -q debug`, `--ntasks=1`,
  `--cpus-per-task=16`, and `-t 02:00:00`.

Job log paths:

- `/lustre/orion/bif148/proj-shared/commapile/stage_commapile_mainmix_4880568.log`
- `/lustre/orion/bif148/proj-shared/commapile/stage_commapile_mainmix_4880568.command.log`

## Batch Result

SLURM accounting:

```text
sacct -j 4880568 --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocNodes,AllocCPUS,MaxRSS -P
JobID|JobName|State|ExitCode|Elapsed|AllocNodes|AllocCPUS|MaxRSS
4880568|stage-commapile-mainmix|COMPLETED|0:0|00:32:21|1|112|
4880568.batch|batch|COMPLETED|0:0|00:32:21|1|56|183164472K
4880568.extern|extern|COMPLETED|0:0|00:32:21|1|112|
```

Job host and command context:

```text
stage_start=2026-06-21T08:45:08-04:00
host=frontier08071.frontier.olcf.ornl.gov
job_id=4880568
pwd=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-17
src=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst
out=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
tmp=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.part.4880568
zstd=/usr/bin/zstd
*** Zstandard CLI (64-bit) v1.5.7, by Yann Collet ***
```

Decompression command and timing:

```text
decompress_start=2026-06-21T08:45:10-04:00
/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst: 1000000725401 bytes
Command being timed: "zstd -T0 -d --no-progress /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst -o /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.part.4880568"
Elapsed (wall clock) time (h:mm:ss or m:ss): 24:30.18
Exit status: 0
decompress_end=2026-06-21T09:09:40-04:00
decompress_elapsed_seconds=1470
```

Validation before atomic rename:

```text
tmp_path=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.part.4880568 tmp_size_bytes=1000000725401 tmp_mtime=2026-06-21 02:21:57.000000000 -0400
12308526802 1000000725401 /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.part.4880568
Command being timed: "wc -c -l /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.part.4880568"
Elapsed (wall clock) time (h:mm:ss or m:ss): 7:42.98
Exit status: 0
```

The `wc` output is `lines bytes path`, so the decompressed file has
`12,308,526,802` lines and `1,000,000,725,401` bytes.

Sanity samples from the job log:

```text
head_sample:
var arrayize, isObject, ref;

tail_sample:
[3] Review in the context of real-world code fragments:
https://mail.python.org/pipermail/python-dev/2005-September/056803.html
```

Final output:

```text
ls -lh /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst
-rw-r--r-- 1 erikgarrison bif148 932G Jun 21 02:21 /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
-rw-r--r-- 1 erikgarrison bif148 235G Jun 21 02:21 /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst

stat -c 'path=%n size_bytes=%s mtime=%y mode=%A owner=%U group=%G' /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
path=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt size_bytes=1000000725401 mtime=2026-06-21 02:21:57.000000000 -0400 mode=-rw-r--r-- owner=erikgarrison group=bif148
```

Filesystem and quota after staging:

```text
df -h /lustre/orion/bif148/proj-shared/commapile
Filesystem                                      Size  Used Avail Use% Mounted on
10761@kfi,10824@kfi:10825@kfi,10760@kfi:/orion  604P  308P  296P  52% /lustre/orion

lfs quota -h -g bif148 /lustre/orion
Disk quotas for grp bif148 (gid 31692):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
  /lustre/orion  1.115T      0k      0k       -   11464       0       0       -
gid 31692 is using default block quota setting
gid 31692 is using default file quota setting

lfs quota -h -u erikgarrison /lustre/orion
Disk quotas for usr erikgarrison (uid 19032):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
  /lustre/orion  1.122T      0k      0k       -  105498       0       0       -
uid 19032 is using default block quota setting
uid 19032 is using default file quota setting

du -sh /lustre/orion/bif148/proj-shared/commapile
1.2T /lustre/orion/bif148/proj-shared/commapile

stage_end=2026-06-21T09:17:25-04:00
```

Checksum note: no external checksum sidecar was present for the source. The
decompression command completed with `zstd` exit status 0 and reported the exact
decoded byte count; the completed decoded file was then independently scanned
with `wc -c -l` before the atomic rename.

## Validation Summary

- Source file exists and compressed size recorded:
  `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst`,
  `251,655,400,225` bytes.
- Candidate output filesystem/path checked before decompression:
  `/lustre/orion` had `296P` available by `df`; user/group `lfs quota` reported
  default block/file quota settings with no explicit project quota limit.
- Decompression method documented:
  one-node SLURM batch job `4880568`, host
  `frontier08071.frontier.olcf.ornl.gov`, not a sustained login-node job.
- Output file size and line/byte sanity checks recorded:
  `1,000,000,725,401` bytes, `12,308,526,802` lines, head/tail samples above.
- Final dataset path:
  `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`.
- Existing decompressed target was not overwritten:
  no target existed before submission; job wrote `.part.4880568` and renamed
  only after successful validation.

## Downstream Recommendation

Do not make every Frontier training worker scan one monolithic 1 TB text file
directly if avoidable. The decompressed text is useful as a canonical source,
but the next preprocessing task should create deterministic shards or tokenized
binary/memmap shards for e97/gdn2 training. Recommended next step:

- Build fixed-size shards suitable for rank-strided reads using the existing
  training `--data_rank/--data_world_size` plan referenced in
  `docs/SCALE_PLAN.md`.
- Prefer tokenized binary shards if the tokenizer is fixed for the run
  (`p50k_base` appears in the existing recipes); otherwise create compressed
  text shards with a small manifest recording byte ranges, line counts, and
  source hash/mtime.
- Keep the monolithic decompressed text only as the canonical source until the
  sharded/tokenized artifact is validated, then decide whether the 1 TB text
  copy is still worth retaining.
