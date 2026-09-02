# E97 4B document-aware mixed 64K preflight

**Status:** 4B systems and close fused-CUDA resume qualification complete; behavioral qualification pending

**Date:** 2026-09-02

## Scope and authority boundary

This is a one-node, eight-rank, fixed-world DiLoCo training stage. It follows
the ADR-003 execution model in `RESILIENT_DILOCO_COMPUTE_POOL.md`. It does not
claim elastic membership, native data-plane, or async-v2.1 conformance. The
applicable safety intent is gap-matrix **R07** and **NDP15**: checkpoints are
published synchronously with atomic rename at K-aligned boundaries. The
elastic receipt chain, background apply, communicator shrink, and recovery
clauses remain unclaimed.

## Stopped legacy continuation

The pure masked-SFT continuation was allowed to reach its next already-defined
K8/save boundary and then intentionally terminated after atomic publication:

- update: `512`
- total input tokens: `7,407,654`
- assistant targets: `2,345,311`
- checkpoint:
  `/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/runs/e97-4b-broad-dual-prompt-64k-u2048-1770df97/checkpoints/checkpoint_agent_sft_u000512_loss_1.4668.pt`
- bytes: `24,276,128,891`
- SHA-256: `8742115cb5b98b5a9d3200752f951f4eb7db697a7fd7518ac3e1d1ccdd49d7dc`
- mmap reload: passed

The process exit was the guard's post-publication termination, not evidence of
a failed checkpoint. This unselected u512 checkpoint is retained as evidence;
the new objective branches with fresh Schedule-Free state from the
behaviorally selected complete-64K u256 parent.

## Packed schema

Boundary-aware pack schema:
`emender-e97-sft-boundary-aware-packs-v2`.

Each example provides token-aligned `tokens`, `valid_mask`, and
`reset_before`, plus prediction-aligned `loss_mask`. The invariant is:

1. every independent record resets every E97 recurrent layer before its first
   token;
2. an authentic trajectory remains one record and receives no internal reset;
3. the next-token target entering a new record is masked;
4. invalid padding contributes no loss, emits no recurrent readout, and is an
   exact identity transition on recurrent state;
5. backward propagation is cut at every reset.

The shared historical `e88_*` Triton files remain the implementation core, but
E97 enters through `E97SplitEditLayer` and `e97_split_edit_triton_apply`, which
require the split erase and value-write gates.

## Filtered CommaPile causal authority

- root:
  `/mnt/nvme1n1/erikg/sft/e97-4b-commapile-document-causal-5m-v1`
- manifest SHA-256:
  `980f35dcbd9527835afcedf38002cf602827eaa03d46e82c033dfd0e10ffc2bc`
- upstream revision: `5afc546db324e7f39f297ba757c9a60547151e7c`
- records: `3,474`
- tokens: `5,192,765`
- causal targets: `5,189,291`
- admitted sources: `30`
- verified Git LFS shards touched: `30`
- `github_archive`: forbidden
- whole records over 65,537 tokens: excluded and counted, never truncated
- holdout repository identity exclusions observed: three `humanize` matches in
  `peS2o` and one `prettytable` match in `stackv2_edu`
- repository content overlap: screened with normalized five-line shingles
  derived from the frozen real-repository holdout
- frozen holdout manifest SHA-256:
  `939bcd66768884a1e5ec44bcf11fdf5d09602ebdabe86d4caefffd4ace8dbb60`

The unlabeled combined 1 TB file is not used by this stage because its source
identity cannot be recovered directly and therefore cannot prove GitHub
exclusion.

## Mixed 25M authority

- root: `/mnt/nvme1n1/erikg/sft/e97-4b-mixed-document-25m-64k-v1`
- authority manifest SHA-256:
  `e9343fd8ad719597d63cd1dc23bd0c8ef05693e207b33f6573a430e56b414f4f`
- records: `44,101`
- input tokens: `80,350,707`
- supervised targets: `25,011,787`

Actual target shares after whole-record overshoot:

| Source | Targets | Fraction |
|---|---:|---:|
| filtered CommaPile causal | 5,007,119 | 20.019% |
| authentic OpenHands actions | 8,753,830 | 34.999% |
| broad SmolTalk2 instruction/reasoning | 6,250,346 | 24.990% |
| Pi-v2 retention | 5,000,492 | 19.993% |

Boundary-aware packs:

- root: `packs-65536-boundary-v2` below the mixed authority
- pack manifest SHA-256:
  `20cee68bb1fd3d4101e662c7ea09bd318a020d11e4bce6ce55935ca7ab5126f2`
- train: 1,480 packs / 43,653 records / 79,550,405 tokens /
  24,772,143 targets
- validation: 16 packs / 448 records / 800,302 tokens / 239,644 targets
- oversized exclusions: zero
- independent pack validation: passed for all 1,496 packs and 44,101 records

## Completed tests

- boundary pack materialization and independent validator;
- packed full-model forward equality against separate per-document forwards;
- packed full-model gradient equality against summed per-document gradients;
- cross-document loss rejection;
- padding-target rejection and padding-token invariance;
- fused Triton reset/valid forward and backward parity against the PyTorch
  recurrence;
- zero initial-state gradient after a leading reset;
- zero projection gradient on padding;
- no information or gradient flow from one document into the next;
- full E97 Triton model masked forward/backward smoke;
- grouped activation-checkpoint replay with packed controls.

## Eight-GPU qualification attempts

The first K8 attempt,
`e97-4b-mixed-document-64k-boundary-q8-0631411b`, failed closed during the
first backward pass. Rank 1 could not obtain one 480 MiB block with 416.69 MiB
physically free while PyTorch reported 2.49 GiB reserved but unallocated. No
update or checkpoint was claimed. The launcher had set the newer
`PYTORCH_ALLOC_CONF` alias, but this installed CUDA build's diagnostic named
`PYTORCH_CUDA_ALLOC_CONF`; the retry sets both aliases to
`expandable_segments:True` explicitly.

The corrected run,
`e97-4b-mixed-document-64k-boundary-q8-expandable-0631411b`, completed eight
updates and one K8 merge:

- global input tokens: `3,461,923`;
- supervised targets: `1,088,370`;
- mean logged loss: `1.2540144362`;
- finite step gradient norms: `0.7734375` through `3.140625`;
- peak allocated HBM: `43.8864 GiB`;
- peak reserved HBM: `44.6621 GiB`;
- optimizer state: `16,183,888,320` bytes of pinned-CPU Schedule-Free state;
- checkpoint:
  `checkpoint_agent_sft_u000008_loss_1.2540.pt`;
- checkpoint bytes: `24,276,128,891`;
- checkpoint SHA-256:
  `5e118d950d01f298e5cfa9297e98251d82c45d4976201c55565d931102b85337`;
- independent mmap reload: passed;
- reload parameter count: `4,045,972,080`;
- checkpoint identity records `boundary_aware_packs=true` and the immutable
  authority, pack, parent, and source digests above.

## Resume and uninterrupted control

The u8 checkpoint resumed through u16 with exact sampler and token clocks:

- total input tokens: `6,879,876`;
- total targets: `1,998,088`;
- resumed checkpoint SHA-256:
  `cc59fa1e77819a66d5d960d89b00a06c181c485705b67f344e9374a4435f1215`;
- uninterrupted-control checkpoint SHA-256:
  `41486937de714386d504aa0e2481903605bc92d3a6078f92b346d44c018eefd5`;
- both mmap reloads: passed.

The independently repeated first eight updates produced bit-identical model
weights. One of 8,091,944,160 optimizer-state elements differed by
`2.2737367544323206e-13`, establishing a fused-CUDA nondeterminism floor. A
resume from the uninterrupted control's own u8 checkpoint therefore used an
identical saved state and reproduced the same close, but not bit-exact, u16
result:

- differing model elements: `3,987,667 / 4,239,051,120`;
- maximum model absolute difference: `9.5367431640625e-06`;
- differing optimizer elements: `2,960,509,616 / 8,091,944,160`;
- maximum optimizer absolute difference: `1.9073486328125e-05`;
- close-parity checkpoint SHA-256:
  `ab74c70c1556ef47fffff1af0c8f63927760fc0532994027bea69ce630787d06`.

This is an exact identity/clock/state restore with close fused-CUDA trajectory
parity. It is explicitly not a bit-exact continuation claim. The CPU reference
optimizer restore remains bit-exact in its unit qualification.

## Remaining gate

Build and validate the scaled no-replacement authority, qualify the new epoch
permutation sampler, and select checkpoints behaviorally. Systems fit and low
loss alone are not promotion evidence.
