# Quality Pass: Figure 2 Legend-Value Cleanup Preview - 2026-05-30

Task: `quality-pass-figure-2`

## Verdict

Pass after graph tightening. The follow-up batch remains narrow and sequential:

1. `figure-2-remove-inline` edits only the Figure 2 renderer/generated image.
2. `figure-2-legend-reports` edits only the Figure 2 caption/legend wording.
3. `build-paper-preview-2` builds and uploads a non-stable preview PDF only.

No stable Hypervolume PDF overwrite, arXiv update, public source push,
Hugging Face update, or formal E97 branch merge is requested by the batch.

## Window Verification

The endpoint averaging window was verified before caption text is changed.

Checked sources:

- `paper/results/figure_2/plot_normalized.py`
- `paper/results/figure_2/AS_OF.md`
- `paper/results/figure_2/E88_NDM.csv`
- `paper/results/figure_2/FLA_GDN.csv`
- `paper/results/figure_2/M2RNN_CMA.csv`
- `scripts/estimate_tokenizer_bytes_per_token.json`

The normalized Figure 2 renderer uses:

- `SMOOTH_COLUMN = "trail_100k"`
- `SMOOTH_LABEL = "100K-step trailing average"`
- `WINDOW_STEPS = 100_000`

`AS_OF.md` independently records that Figure 2 and the paper labels use
`trail_100k`, not `trail_50k`.

Tail CSV values recomputed with
`bits/byte = nats/token * log2(e) / mean_bytes_per_token` from the pinned
tokenizer JSON:

| Model | Tail step | 10K BPB | 50K BPB | 100K BPB | Rounded label value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Emender/E88 | 1,405,450 | 0.976525 | 0.976995 | 0.976819 | 0.977 |
| GDN | 1,847,050 | 0.962779 | 0.965280 | 0.969636 | 0.970 |
| M2RNN-CMA | 1,343,050 | 0.981920 | 0.982038 | 0.983472 | 0.983 |

Caption/legend work should therefore state the 100K-step trailing endpoint
average unless the plotting code is intentionally changed and revalidated.

## Graph Tightening

I made two WG metadata changes:

- Added `.evaluate-figure-2-remove-inline` as a dependency of
  `figure-2-legend-reports`, so caption wording waits for the evaluated plot
  cleanup and the exact recorded endpoint values.
- Added `.evaluate-figure-2-legend-reports` as a dependency of
  `build-paper-preview-2` and set that task to `context_scope = graph`, so the
  uploaded preview waits for evaluated plot and caption changes.

## Checklist

- [x] Downstream tasks are narrow: plot cleanup, caption/legend value wording,
      build/upload preview.
- [x] Averaging window explicitly verified before caption text is changed:
      current renderer and source data use `trail_100k`.
- [x] Inline numeric label removal is covered by `figure-2-remove-inline`.
- [x] Dashed/lighter/distinct leader styling is covered by
      `figure-2-remove-inline`.
- [x] Non-stable preview upload only is covered by `build-paper-preview-2`.

## Validation Commands

```bash
wg show quality-pass-figure-2
wg show figure-2-remove-inline
wg show figure-2-legend-reports
wg show build-paper-preview-2
wg viz quality-pass-figure-2
rg -n "^(SMOOTH_COLUMN|SMOOTH_LABEL|WINDOW_STEPS) =|trail_100k|Figure 2 and the paper labels use" \
  paper/results/figure_2/plot_normalized.py paper/results/figure_2/AS_OF.md
python3 - <<'PY'
import csv, json, math
from pathlib import Path
root = Path("paper/results/figure_2")
canon = json.load(open("scripts/estimate_tokenizer_bytes_per_token.json"))
factor = math.log2(math.e) / float(canon["mean_bytes_per_token"])
assert abs(factor - float(canon["bits_per_byte_per_nat_per_token"])) < 1e-12
for name, filename in [
    ("Emender/E88", "E88_NDM.csv"),
    ("GDN", "FLA_GDN.csv"),
    ("M2RNN-CMA", "M2RNN_CMA.csv"),
]:
    rows = list(csv.DictReader(open(root / filename)))
    tail = rows[-1]
    print(
        name,
        int(tail["step"]),
        float(tail["trail_10k"]) * factor,
        float(tail["trail_50k"]) * factor,
        float(tail["trail_100k"]) * factor,
    )
PY
```
