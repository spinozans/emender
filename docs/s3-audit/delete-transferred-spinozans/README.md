# Delete Transferred Spinozans Artifact Audit

Task: `delete-transferred-spinozans`
Date: 2026-06-22 UTC
AWS caller: `arn:aws:iam::027441888822:user/erik`

## Decision

Deleted exactly two S3 objects from `s3://spinozans/`:

| S3 URI | Role | Size | Last modified | ETag |
|---|---:|---:|---|---|
| `s3://spinozans/commapile/commapile_mainmix_v0.1_1tb.txt.zst` | transferred compressed artifact | `251,655,400,225` bytes | `2026-06-21T06:21:57+00:00` | `"b39cd8e93bf3d919846a2b6004e71235-3750"` |
| `s3://spinozans/commapile/commapile_mainmix_v0.1_1tb.txt.manifest.json` | directly associated metadata sidecar | `13,863` bytes | `2026-06-21T06:21:50+00:00` | `"56b040930711b70f9febd36f7a1201f3"` |

No other S3 keys were deleted. `s3://spinozans/index.html` remained present after
the delete.

## Identity Evidence

`before-spinozans-listing.json` shows the bucket contained only three keys before
deletion:

- `commapile/commapile_mainmix_v0.1_1tb.txt.manifest.json`
- `commapile/commapile_mainmix_v0.1_1tb.txt.zst`
- `index.html`

The same listing shows exactly one object larger than 200 GB:
`commapile/commapile_mainmix_v0.1_1tb.txt.zst`, size
`251,655,400,225` bytes.

`head-artifact.json` records the large object's object metadata:

- `ContentLength`: `251655400225`
- `ContentType`: `application/zstd`
- server-side encryption: `AES256`
- no custom metadata

`head-manifest.json` records the sidecar object's object metadata:

- `ContentLength`: `13863`
- `ContentType`: `application/json`
- server-side encryption: `AES256`
- no custom metadata

`commapile_mainmix_v0.1_1tb.txt.manifest.json` was copied from S3 before deletion.
It describes the generated `commapile_mainmix_v0.1_1tb.txt` dataset with
`actual_file_size` and `actual_bytes_sum` both equal to `1,000,000,725,401`,
`sources` count implied by the recorded 31-source mixture, and `sha256`
`44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9`.

The manifest sidecar relationship is by exact shared basename:

- artifact basename: `commapile_mainmix_v0.1_1tb.txt`
- compressed artifact suffix: `.zst`
- metadata sidecar suffix: `.manifest.json`

## Transfer Evidence

Committed project state already records this dataset as the releasable
commapile training data and identifies the intended S3 staging artifact:

- `docs/HANDOFF.md:639` records `commapile_mainmix_v0.1_1tb.txt`,
  its `.zst` artifact at 251.7 GB, its `.manifest.json`, and the intended
  upload target `s3://garrisonlab/commapile/`.
- `docs/FRONTIER_COMMAPILE_MAINMIX_STAGE_20260621.md:222` through
  `docs/FRONTIER_COMMAPILE_MAINMIX_STAGE_20260621.md:234` records the transferred
  Frontier copy at
  `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt.zst`,
  exact compressed size `251,655,400,225` bytes, decompression through SLURM job
  `4880568`, decoded size `1,000,000,725,401` bytes, line count
  `12,308,526,802`, and final decompressed dataset path
  `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`.
- `docs/FRONTIER_COMMAPILE_MAINMIX_STAGE_20260621.md:215` through
  `docs/FRONTIER_COMMAPILE_MAINMIX_STAGE_20260621.md:218` records that `zstd`
  decompression completed successfully and the completed decoded file was
  independently scanned before atomic rename.

The live Spinozans S3 object size and basename matched the documented transferred
artifact exactly, and there was no second plausible 250 GB object in the bucket.

## Delete Evidence

The delete commands were constrained to these exact keys:

```text
aws s3api delete-object --bucket spinozans --key commapile/commapile_mainmix_v0.1_1tb.txt.zst
aws s3api delete-object --bucket spinozans --key commapile/commapile_mainmix_v0.1_1tb.txt.manifest.json
```

`bucket-versioning.json` is empty, which is the AWS CLI representation for no
enabled bucket versioning status returned by `get-bucket-versioning`.

## Post-Delete Evidence

`after-spinozans-listing.json` shows only `index.html` remains in the bucket.

The deleted keys were checked with `head-object` after deletion:

- `after-head-artifact.exit` is `254`, and `after-head-artifact.stderr` records
  `An error occurred (404) when calling the HeadObject operation: Not Found`.
- `after-head-manifest.exit` is `254`, and `after-head-manifest.stderr` records
  `An error occurred (404) when calling the HeadObject operation: Not Found`.

`after-head-index.json` confirms the unrelated sibling object
`s3://spinozans/index.html` remained present after deletion.
