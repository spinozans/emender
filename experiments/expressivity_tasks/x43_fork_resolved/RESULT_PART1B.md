# X43 part 1b: correct-path build passed; index construction failed

`cd ~/sxgc/xsa && cargo build --release` passed in 4m 48s with 26 warnings.
The exact original TinyShakespeare file was then passed to `xsa build --text`
with one thread, text mode, and `--verify-text-sample 1000`. Index construction
exited 1 at `endpoints`; no `.sxi` was published. The index leg stopped.

The endpoint log ends verbatim:

```text
CYCLIC_SEAM_REFUSED class_size=40000 limit=1000 policy=max(1000,raw_r/1000); no O(n) fallback
FATAL: cyclic seam repair exceeds compressed-work policy; no O(n) fallback
```

The corpus is unchanged: 1,115,394 bytes, SHA-256
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
It has 40,000 newline bytes, ends in newline, and has no 0x1e record separators.
No separator was appended and no corpus transform or source patch was attempted.
The audit was requested but never reached. Partial scratch artifacts are local,
ignored, and are not a validated index.

Both original checkpoints and per-target arrays are retained and match every
corresponding SHA-256 in the original artifact manifest. Retraining was therefore
unnecessary. No GPU was accessed. The prior model measurements remain:

| Arm | Held-out NLL, nats/byte | Accuracy | Forced/fork split |
|---|---:|---:|---|
| EmenderLM | 2.201701 | 35.199% | blocked |
| Transformer | 1.829293 | 52.884% | blocked |

Fork positions, chi/n, aligned labels, forced/fork NLL and accuracy, NLL share at
forks, and both baseline decompositions are unavailable. No values are inferred
from the partial BWT or claimed as results. The trivial baselines were not run
because their requested class split depends on validated labels.

Evidence: `receipts/part1b/index_status.json` records exact commands, exits,
binary hash, source commit, and stop reason. `xsa_build.log`, `index_build.log`,
and the copied pipeline journal and stage logs retain the verbatim output.
`corpus_binding.json`, `retained_artifacts.json`, and `artifact_manifest.json`
bind the corpus, retained model evidence, and new receipts. The xsa source diff
was empty; no operator source was modified. All work stayed single-threaded in
the attended session, without worker services or delegated agents. xsa's Cargo
build invokes its own native compilation jobs internally.

Residual risks: the requested scientific decomposition is still blocked by an
xsa construction failure on the exact model corpus; orientation and chi-to-target
alignment remain unvalidated. The earlier short-run, unequal-token-budget,
precision, context-reset, and single-seed limitations still apply. No architecture
ranking or oracle ceiling is established. The requested `contact_supervisor`
tool is unavailable in this session.
