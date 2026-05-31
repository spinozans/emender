# V3 Push-Current Checkpoints to Hugging Face — Blocker Report

Date: 2026-05-31 UTC
Task: `v3-push-current`

Per the task's **If blocked** clause ("STOP and report the exact blocker — do
NOT guess a destination"), this report records what was located, what was
verified live, and the precise reason a fresh ("current"/V3) public checkpoint
push cannot proceed under this task's authorization without inventing a path or
fabricating an approval.

No Hugging Face write operation was performed by this task. Only unauthenticated
(`token=False`) hub readbacks were run.

## 1. Existing release recipe identified and cited (not invented)

Two existing mechanisms target the canonical `poietic-pbc` repos
`emender-e88-1.3b`, `gdn-1.3b`, `m2rnn-cma-1.3b`:

| Recipe | Path | Destination | Gate |
| --- | --- | --- | --- |
| Public publish | `scripts/publish_v02_public_hf.py` | public repos, tag `v0.2` | **hard human-approval gate** |
| Private staging | `scripts/hf_private_staging_upload.py` | private repos, branch `staging` | refuses if repo is not private |

Supporting docs: `docs/HUGGINGFACE_RELEASE.md`,
`docs/RELEASE_V02_PUBLIC_HF_PUBLISH_20260529.md`,
`docs/RELEASE_V01_PRIVATE_HF_STAGING_UPLOAD_20260527.md`,
`docs/RACER_HF_FRESHNESS_AUDIT_20260529.md`.

The established refresh pipeline is: freshness audit → `validate-v0-X-racer`
(select by smoothed BPB + convert + local & Docker CPU/GPU smoke, **no upload**)
→ **approval-gated** public publish. The v0.2 publish followed exactly this.

## 2. Current checkpoints identified (corresponding to the reported V3 racer)

Training is **still in progress** (logs append at 2026-05-31T13:48Z). E88/NDM
config param count is 1,273,191,856 (~1.27B; repo slug uses `1.3b`).

| Model | Canonical repo | Local `latest.pt` step (2026-05-31) | V0.2 published step | V3 paper AS_OF endpoint step |
| --- | --- | ---: | ---: | ---: |
| E88 / NDM | `poietic-pbc/emender-e88-1.3b` | 1,521,000 | 1,395,000 | 1,405,450 |
| GDN | `poietic-pbc/gdn-1.3b` | 1,998,000 | 1,845,000 | 1,847,050 |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | 1,464,000 | 1,332,000 | 1,343,050 |

`latest.pt` is a noisy single-step save (e.g. E88 step 1,521,000 raw
loss 2.6943 — a high outlier); the established practice **selects by
smoothed BPB**, not by `latest.pt`. So even checkpoint *selection* requires the
`validate-v0-X-racer` analysis, which has not been run for V3.

## 3. Live verification of the canonical artifacts (hub API, unauthenticated)

The V3 paper's "checkpoints-on-HF" reproducibility claim is **currently backed
live** at `v0.2` (`token=False` `HfApi.repo_info` / `list_repo_refs`):

| Repo | private | tags | `v0.2` sha | `model.safetensors` size | required files |
| --- | --- | --- | --- | ---: | --- |
| `poietic-pbc/emender-e88-1.3b` | False | v0.1, v0.2 | `be77d1ee5b744ffa653e9b03abd5254d2ef8a41c` | 2,713,728,024 | all present |
| `poietic-pbc/gdn-1.3b` | False | v0.1, v0.2 | `7395b6b6588726a3bca963aa7e6150e0971e71d6` | 2,975,047,820 | all present |
| `poietic-pbc/m2rnn-cma-1.3b` | False | v0.1, v0.2 | `2e5f8f3be8a7c8ac42802485afb40d023874ea06` | 2,807,297,000 | all present |

Live tree URLs:

- <https://huggingface.co/poietic-pbc/emender-e88-1.3b/tree/v0.2>
- <https://huggingface.co/poietic-pbc/gdn-1.3b/tree/v0.2>
- <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/tree/v0.2>

## 4. Precise blocker (why a fresh push cannot proceed here)

**A. Mandatory human approval for a public HF write is absent from this task
context.** The only recipe that writes to the canonical *public* repos,
`scripts/publish_v02_public_hf.py`, hard-fails without it:

```python
def ensure_approval(args):
    if not args.approved_public_v02_publication:
        raise SystemExit("--approved-public-v02-publication is required for any public HF write")
    ...
    required = ["authorizes public hugging face v0.2 publication",
                "poietic-pbc/emender-e88-1.3b", "poietic-pbc/gdn-1.3b",
                "poietic-pbc/m2rnn-cma-1.3b"]
```

The established process treats "explicit human approval present in task context"
as a non-negotiable gate (RELEASE_V02 publish report; freshness-audit validation
list). This `v3-push-current` task carries **no** such approval — its only
context is the dependency log "Spawned assignment inline / Task marked as done",
and there are no task messages. Supplying the approval flag/note myself would be
**fabricating a human authorization**, which is forbidden.

**B. No existing recipe uploads a fresh "current"/V3 set to the canonical
location without inventing a path.**
- `publish_v02_public_hf.py` is locked to revision `v0.2` (which already exists)
  and to the pre-validated candidate set at
  `/tmp/release-v02-local-hf-candidates-agent-500/` (E88 step 1,395,000 etc),
  cross-checked against a manifest. Pointing it at current step-~1.5M
  checkpoints, or to a new `v0.3` tag, is exactly the "invent a new upload
  path / repo name" the task forbids.
- `hf_private_staging_upload.py` refuses to update a repo whose readback
  `private` is not true — but the canonical repos are now **public**
  (`private=False`), so it errors. Its hardcoded source checkpoints
  (E88 step 1,281,000 etc) also no longer exist on disk.

**C. The mandatory pre-upload validation has not been run for V3.** No
validated/converted current-checkpoint candidate set exists on disk (only the
v0.2 set). The `validate-v0-X-racer`-style step (select-by-smoothed-BPB,
convert to safetensors, local + Docker CPU/GPU smoke) is a hard prerequisite of
every prior publish and has not been done. Training is still in progress; the
paper AS_OF explicitly says "Do not cite these results without re-running."

## 5. Graph growth (prerequisites created)

To unblock a real V3 push along the established pipeline, this task created:

- `v3-validate-racer-candidates` — select current checkpoints by smoothed BPB,
  convert, local + Docker smoke, **no upload** (mirrors `validate-v0-2-racer`).
- `v3-publish-public-hf` (`--after` validate) — approval-gated public publish;
  requires an explicit human approval note in its task context.

## 6. Validation checklist result

- [x] Existing HF release recipe identified and cited (not invented).
- [ ] Current checkpoints uploaded — **NOT done** (blocker A/B/C above).
- [x] Canonical artifacts verified live via hub API (`v0.1`+`v0.2` resolve).
- [x] HF URLs + current/published/paper checkpoint steps recorded.
- [x] Precise blocker reported; **nothing guessed**, no destination invented,
      no approval fabricated, no `v0.1`/`v0.2` tag touched.
