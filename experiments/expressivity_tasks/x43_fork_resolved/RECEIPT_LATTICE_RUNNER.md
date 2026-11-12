# 150M lattice runner — built and gated (2026-11-11)

The lattice now has its own measurement harness,
`x42_pile_scale/lattice.py`, deliberately separate from the search harness in
`byte_velocity.py`. It fires arms 1-4 of PREREGISTRATION_150M_LATTICE.md on the
registered flagship stream. Arms 5-7 (the two consultation arms and the append-only
control, which twins arm 6) remain gated on the live consult path and **fail closed** if
requested.

## What it binds (and why the search harness could not be reused)

`byte_velocity.train()` is hardwired to `reg['stream']` — pile.txt plus
`/home/erikg/c14_dense_bytes/pile_offsets.jsonl`, the **in-region** C14 document set — and
to the three search arms, the `train`/`valid` split vocabulary and the *search*
registration hash. Reusing it for the lattice would have measured the wrong corpus while
writing a receipt that looked correct. The lattice module therefore owns its own:

| binding | source |
|---|---|
| stream | `receipts/flagship_stream/documents.jsonl.gz` (243,401 documents, all beyond byte 100,000,000,000) |
| stream identity | `receipts/flagship_stream/registration.json` (source size/inode/mtime verified) |
| registration | `PREREGISTRATION_150M_LATTICE.md` sha256 |
| arms | 4 dense arms; policies taken from the harvested nominations with their receipt hashes |
| budget | 2700 active seconds, seed as a parameter |
| eval | first 24 full 2048-byte windows of a registered split (default `calibration`) |

## Gates (all pass; `lattice gate --gpu`)

| gate | result |
|---|---|
| **stream_binding** | the C14 in-region manifest is **rejected structurally**: "5+ documents start below the 100000000000 boundary". A hash check alone is insufficient — the C14 manifest hashes cleanly against pile.txt, so a wrong stream is *valid* and would train happily |
| **splits** | train 238,461 · calibration 2,478 · final_test 2,462; zero document ids and zero content hashes shared across splits |
| **arms** | 4 genuinely distinct configurations; unbounded has a non-binding cap, bounded does not; gated arms are not buildable |
| **boundary** | 20,000 windows checked: every window inside one document, aligned starts, all beyond the boundary |
| **repro** | two independent 90 s runs → identical loss for all 11 compared steps and identical eval `3.2148846558162143` |

## Arms smoke-tested end-to-end (120 s each, card 5)

| arm | steps | bytes | eval (7 windows, calibration) |
|---|---:|---:|---|
| lattice_transformer | 44 | 4.99 M | 2.614960 |
| lattice_gdn2 | 38 | 4.32 M | 2.266812 |
| lattice_emender_bounded | 11 (90 s) | — | 3.214885 |
| lattice_emender_unbounded | 5 | 0.57 M | 4.280013 |

These are **smoke numbers on a 7-window eval after 2 minutes of training** — they are not
results and must never be cited as such. Their purpose is to prove every arm builds,
trains, evaluates and receipts on the registered stream.

## Two engineering findings worth keeping

1. **`CUDA_VISIBLE_DEVICES` is a lease, not a label.** The first version recorded a card
   number while `model.cuda()` would have run on physical device 0. The harness now
   requires `CUDA_VISIBLE_DEVICES` to be set and to equal the recorded card.
2. **"Unbounded" is a provably non-binding finite cap, not the native ceiling.** The
   Triton kernel rejects `budget=262144` ("invalid argument"). Since the store resets
   every window and at most one slot is appended per token, occupancy ≤ CONTEXT, so a cap
   of `CONTEXT+1 = 2049` provably never binds — the same `finite_cap_never_bound`
   convention the C14-B unbudgeted control used at `budget=257`.

## Status

Wave 1 (arms 1-4) is runnable at the registered 2700 s budget on any leased card.
Not yet done: a peak-occupancy column to receipt `finite_cap_never_bound` empirically
(currently analytic), and the cross-harness parity test against the search path.
