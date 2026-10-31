# Donor checkpoint search

Before launching the fresh donor, an exhaustive `find` under
`experiments/expressivity_tasks/x42_pile_scale` found no `.pt`, `.pth`, or
`.safetensors` files. `donor_search.txt` preserves that empty result.
The registered search and anchor training functions do not save model weights.

Broader searches of `/tmp`, `/home/erikg` (excluding cache/git trees), and
`/mnt/nvme{1,2,3}n1/erikg` found unrelated historical models and test fixtures,
with no locatable checkpoint attributable to the X42 best transformer search
candidate or anchor on this slice. Mount searches were bounded in depth/output;
this is not a claim to have inventoried every file on every disk.

The authorized fallback trained a fresh transformer for the full 2700-second
protocol using the best completed transformer recipe at launch:
`runs/velocity_transformer_v1/g2_c03/attempt1/result.json`, prior score
3.6856608937184014. `donor_provenance.json` binds its exact policy and the
training/validation corpus SHA-256 hashes. The fresh donor's result and training
log are in `donor/`; its local `donor_weights.pt` is ignored by Git, and its
SHA-256 is retained in `mapping.json`.
