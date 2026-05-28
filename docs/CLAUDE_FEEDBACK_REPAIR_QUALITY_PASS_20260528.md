# Claude Feedback Repair Quality Pass

Date: 2026-05-28 UTC
Task: `quality-pass-20260528-claude-review-repair`
Reviewer role: Evaluator

## Verdict

Overall grade after corrective action: 0.95
Confidence: 0.86
Rubric underspecified: no

This quality pass reviewed the Claude feedback repair pipeline before
execution. At initial inspection the repair tasks were not yet visible in the
ready/open task list; during the pass the canonical two-task pipeline appeared:
`claude-review-focused-repair-edit` followed by
`claude-review-repair-synthesis`. This pass tightened both task descriptions
and added `.evaluate-quality-pass-20260528-claude-review-repair` as an explicit
dependency of the focused edit task, so the repair waits for this quality pass
and its evaluation gate.

A duplicate fallback task, `claude-feedback-paper`, was created during the
concurrent graph update and then abandoned once the canonical pipeline was
identified and tightened. The execution path left in the graph is the canonical
edit-plus-synthesis pipeline.

After correction, the repair task is scoped to the four requested manuscript
issues only:

- transformer/TC0 sentence;
- realizability-vs-trainability seam;
- SiLU/update notation consistency, including Figure 1A/B/F checks;
- first-use 1.3B-class parenthetical.

The tightened tasks explicitly preserve the current HF `1.3b` links, GitHub
`poietic-pbc/emender`, Figure 2 values, and release state. They require the
paper build, Lean trust gate, PDF upload to hypervolu.me, and non-destructive
GitHub push if paper source changes. They also forbid HF visibility changes,
repo renames, generated artifact commits, tokens, checkpoints, safetensors,
caches, and unrelated staging.

## Reviewed Surfaces

- `wg show quality-pass-20260528-claude-review-repair`
- `wg context quality-pass-20260528-claude-review-repair`
- `wg list`, `wg ready`, and `.wg/graph.jsonl` searches for existing Claude
  repair tasks
- `editorial_notes.md`
- `docs/EDITORIAL_NOTES_STOCKTAKE_20260528.md`
- `docs/EDITORIAL_NOTES_PAPER_SYNTHESIS_20260528.md`
- `paper/main.typ`
- root `README.md` and `paper/README.md` release/link surfaces

## Corrective Actions

Reviewed and tightened tasks:

- `claude-review-focused-repair-edit`
- `claude-review-repair-synthesis`

Sequencing correction:

- Added `.evaluate-quality-pass-20260528-claude-review-repair` as an explicit
  dependency of `claude-review-focused-repair-edit`.
- Kept `claude-review-repair-synthesis` after
  `claude-review-focused-repair-edit`.
- Abandoned duplicate fallback task `claude-feedback-paper` and its auto
  evaluation tasks after the canonical pipeline was identified.

Description corrections:

- Tightened the focused-edit task to require exact primary-source assumptions
  for the transformer/TC0 sentence, explicit realizability-vs-trainability
  separation, Figure 1A/B/F notation checks, and the first-use
  1.3B-class parenthetical.
- Tightened the synthesis task to require full diff mapping to the four repair
  points, exact build/Lean commands, PDF upload plus HTTP 200, GitHub push or
  logged branch/blocker, and explicit artifact/release-state guardrails.
- Kept the work serialized because the same manuscript files are involved.

## Dimension Scores

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Task existence and sequencing | 0.94 | The canonical edit-plus-synthesis pipeline now waits on the quality-pass evaluation gate before execution, then runs serially through edit and synthesis. The transient duplicate task was abandoned. |
| Four-issue coverage | 0.97 | The focused-edit and synthesis tasks name all four requested issues and give local instructions for each, including the citation-sensitive transformer/TC0 sentence, the Lean-vs-empirical seam, Figure 1A/B/F notation checks, and the first-use 1.3B-class parenthetical. |
| Scope control | 0.96 | The focused edit is limited to `paper/main.typ` and `paper/refs.bib` only if citation work is needed, forbids whole-paper rewriting, and requires the diff to stay within the four concrete repairs. The synthesis task allows only tiny consistency fixes tied to the same four points. |
| Release invariant preservation | 0.98 | Both tasks explicitly preserve the three current HF `poietic-pbc/*-1.3b` links, `v0.1` revision usage where present, GitHub `poietic-pbc/emender`, Figure 2 values E88 0.979 / GDN 0.975 / M2RNN-CMA 0.984, and current release state. |
| Validation gates | 0.97 | The pipeline requires `bash paper/build.sh`, the Lean trust gate, visual PDF spot-check, hypervolu.me PDF upload plus HTTP 200, and a non-destructive GitHub push or exact logged blocker/branch if `main` is not fast-forward safe. |
| Artifact and credential safety | 0.98 | Both tasks forbid HF visibility calls, repo renames/transfers, force-pushes, destructive git commands, generated PDFs, checkpoints, safetensors, HF caches, Docker layers, tokens, and unrelated file staging. |
| Evaluation transparency | 0.93 | This report records the initial gap, correction, score, confidence, dimension scores, residual risks, and validation commands used to make the review reproducible. |

## Checklist Result

- Pass after correction: tasks are scoped to the concrete feedback and do not
  relitigate the whole paper. The focused edit and synthesis tasks allow only
  the four named manuscript repairs.
- Pass after correction: tasks preserve current HF `1.3b` links, GitHub
  `poietic-pbc/emender`, Figure 2 values, and release state.
- Pass after correction: tasks require build, Lean gate, PDF upload to
  hypervolu.me, and GitHub push if paper changes.
- Pass after correction: tasks do not change HF repo visibility, rename repos,
  or commit generated artifacts, tokens, checkpoints, safetensors, or caches.

## Residual Risks

The transformer/TC0 sentence is the main content risk. The current paper cites
Merrill-Petty-Sabharwal, while the stocktake notes that stronger transformer
TC0 language may require an additional primary source and exact model
assumptions. The repair task therefore requires a bounded, assumption-qualified
sentence and primary-source citation, and it forbids claims that frontier
deployed transformers provably fail S5 or that Emender exceeds TC0/NC1.

Figure 1 currently has source panels A and B in `paper/main.typ`; no panel F
was found in the source search. The repair tasks still require checking Figure
1A/B/F because the feedback names F explicitly. If no F panel exists in the
source/PDF at repair time, the worker must log that fact rather than silently
ignoring the requested check.

The current checkout is on `main` with a local synthesis commit whose tree
matches `poietic/main` but whose commit hash differs. The repair task accounts
for this by requiring a non-destructive push and, if pushing `main` is not
fast-forward safe, a task branch push plus exact logged blocker.

## Validation Commands

- `wg msg read quality-pass-20260528-claude-review-repair --agent $WG_AGENT_ID`
- `wg quickstart`
- `wg show quality-pass-20260528-claude-review-repair`
- `wg context quality-pass-20260528-claude-review-repair`
- `wg list`
- `wg ready`
- `rg -n "claude|repair|transformer|realizability|trainability|silu|1\\.3B|1\\.3b|Figure 1" .wg docs README.md paper editorial_notes.md`
- `rg -n "TC0|transformer|realizability|trainability|silu|delta_h|Figure 1|1\\.3B|1\\.3b|poietic-pbc/emender|0\\.979|0\\.975|0\\.984|v0\\.1" paper/main.typ README.md paper/README.md docs/EDITORIAL_NOTES_STOCKTAKE_20260528.md docs/EDITORIAL_NOTES_PAPER_SYNTHESIS_20260528.md`
- `wg show claude-review-focused-repair-edit`
- `wg show claude-review-repair-synthesis`
- `git status -sb`
- `git remote -v`
