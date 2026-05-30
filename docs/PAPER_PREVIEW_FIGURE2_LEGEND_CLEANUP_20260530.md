# Figure 2 Legend Cleanup Paper Preview - 2026-05-30

Task: `build-paper-preview-2`

## Preview Artifact

- Source commit at build time: `dfbb2339`
- Local build output:
  `paper/Garrison_2026_Emender-dfbb2339.pdf`
- Local preview copy:
  `paper/Garrison_2026_Emender-figure2-legend-cleanup-preview-20260530T174010Z.pdf`
- Hypervolume preview URL:
  `http://hypervolu.me/~erik/ndm/Garrison_2026_Emender-figure2-legend-cleanup-preview-20260530T174010Z.pdf`
- SHA-256:
  `46101e030d0d2a4f5710db47e0010da65745a02899eb431de0e0d39a17b2f46f`
- HTTP readback:
  `200 OK`, `Content-Type: application/pdf`, `Content-Length: 849206`

## Validation

- `python paper/results/figure_2/plot_normalized.py` regenerated
  `paper/results/figure_2/figure_2_draft.png`; the tracked PNG was already
  current and no source diff was produced.
- `bash paper/build.sh` completed successfully and wrote
  `paper/Garrison_2026_Emender-dfbb2339.pdf`.
- Page 13 of the generated PDF was rendered with PyMuPDF for visual checking.
  Figure 2 has model-name-only right-edge labels: `M2RNN-CMA`, `Emender`, and
  `GDN`. No inline numeric BPB labels are present on the plot.
- The leader/guide segments from trajectory endpoints to right-edge labels are
  light gray dashed guides, visually distinct from the solid colored model
  trajectories.
- The Figure 2 caption reports endpoint values as final 100K-step trailing
  averages: GDN `0.970` BPB, E88 `0.977` BPB, and M2RNN-CMA `0.983` BPB.
  The caption records the verified window and source snapshot as a
  `2026-05-29T18:04:51Z` active-log snapshot.
- The preview PDF was copied to Hypervolume under the timestamped non-stable
  filename above.
- The stable Hypervolume file was checked before and after upload and was not
  overwritten:
  - `~/www/ndm/Garrison_2026_Emender.pdf`
  - size `912666`
  - mtime `2026-05-29 20:48:27.970376620 +0000`
  - SHA-256 `223e4cf1fe1b4622f0732c55969b2049f265799de6ab66a469f0047b3e288928`

No stable public PDF overwrite, arXiv update, public source push, Hugging Face
update, or formal E97 branch merge was performed.
