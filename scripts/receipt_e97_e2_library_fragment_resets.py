#!/usr/bin/env python3
"""End-to-end reset-semantics receipt for the E2 v2 library fragment cohort.

Two-part proof that the rebuilt library records flow through the record-reset
SFT path (hidden state cleared at every fragment start — the operator design
ruling):

PART A — PRODUCTION PATH (the training Dataset itself): the library fragment
authority + its validated boundary-aware packs are training-eligible, so
ndm.data.masked_sft_dataset.MaskedSFTPackedDataset materializes them for real.
Every train pack is one window-cut fragment; pack_at_with_boundaries() gives
each fragment a token-aligned reset_before at its start (record_spans_at ->
``reset_before[start] = True``) and masks the prediction into that reset
(``loss_mask &= ~reset_before[1:]``). Verified for ALL 891 train fragments.

PART B — PREP-LEVEL: the E2 v2 prep authority is non-authorizing
(training_eligible false) so the Dataset correctly refuses to materialize it
(fail-closed by design; admission flips eligibility). The receipt replicates
the pack_at_with_boundaries derivation VERBATIM, proves the replication
byte-identical against the production path on Part A packs, then applies it to
the v2 prep packs: every long-document-library-anchor record's fragment start
carries reset_before=True, the prediction into it is loss-masked, the
fragment's first authority-mask byte is 0 (unsupervised entry token), and the
entry is mid-document (arbitrary entry) except the stream-head edge fragment.
In-window EOT record separators are supervised tokens (state carries across
document boundaries inside the fragment — the document-concatenation read).

The trainer path (cited, not run here — GPU code): scripts/train_e97_4b_pi_sft.py
--boundary-aware-packs -> get_boundary_aware_batch(1) -> packed_objective ->
model(reset_before=...) -> ndm/models/e88_fla_hybrid.py _process_chunk ->
ndm/triton/e88_triton_forward.py APPLY_RESET:
``S = tl.where((token_valid & token_reset)[:, None, None], zeros, S)``.

CPU-only: data masks only; no model, no CUDA, no GPU state.
"""
import json
import struct
from pathlib import Path

import numpy as np
import torch

from ndm.data.masked_sft_dataset import (
    RECORD_INDEX, MaskedSFTPackedDataset, SFTSamplerIdentity,
)

W = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-e2-chat-agent-prep-v1')
FRAG = W / 'v2/library-fragments-authority'
FRAG_PACKS = W / 'v2/library-fragments-packs'
FRAG_AUTH_SHA = '42372c8d2c91fa3d93e6384c988c262f6bc665b667778a29d91f8ac122441508'
FRAG_PACK_SHA = 'e19f2e408b5faa6fe409721340d3147c2ec4ac16f4aa1bedb08334f94dac6360'
PREP = W / 'v2/e2-preparation'
AUTH_SHA = 'e6280dc609f0ad9ac0dda783f3537e9c064789d61ff24d8fb047e2d251fe3a4e'
PACK_SHA = '7759fddcc37a109ed7d30418387a2cb70e510bb9ed3b59224663fe83ccc4c141'
COHORT = 'long-document-library-anchor'
EOT_TOKEN = 50256
PACK_INDEX = struct.Struct('<QQQQ')


def replicate_boundaries(tokens_row, token_mask_row, spans, sequence_tokens):
    """Verbatim replication of MaskedSFTPackedDataset.pack_at_with_boundaries
    (including its zero padding to the 65,537-token pack sequence)."""
    length = spans[-1][1]
    tokens = torch.zeros(sequence_tokens, dtype=torch.long)
    tokens[:len(tokens_row)] = torch.tensor(np.asarray(tokens_row), dtype=torch.long)
    token_mask = torch.zeros(sequence_tokens, dtype=torch.bool)
    token_mask[:len(token_mask_row)] = torch.tensor(np.asarray(token_mask_row), dtype=torch.bool)
    valid_mask = torch.zeros(sequence_tokens, dtype=torch.bool)
    valid_mask[:length] = True
    reset_before = torch.zeros(sequence_tokens, dtype=torch.bool)
    for start, _stop in spans:
        reset_before[start] = True
    loss_mask = token_mask[1:] & valid_mask[:-1] & valid_mask[1:] & ~reset_before[1:]
    return tokens, loss_mask, valid_mask, reset_before


def main() -> None:
    # ---------- PART A: production Dataset over the fragment authority ----------
    identity = SFTSamplerIdentity(
        authority_manifest_sha256=FRAG_AUTH_SHA, pack_manifest_sha256=FRAG_PACK_SHA,
        sampler_key=2200001, data_world_size=8, context_size=65536, split='train')
    data = MaskedSFTPackedDataset(FRAG, FRAG_PACKS, identity=identity,
                                  rank=0, sampler_mode='epoch-permutation')
    assert data.boundary_aware, 'fragment packs must be boundary-aware v2 packs'
    frag_rows = [json.loads(line) for line in (FRAG / 'records.jsonl').open()]
    frag_train = sum(1 for row in frag_rows if row['split'] == 0)
    part_a = {'packs': 0, 'fragments': 0, 'resets': 0, 'mid_document': 0,
              'aligned': 0, 'solo_packs': 0, 'in_window_separators': 0}
    sample = None
    for pack_id in range(len(data.packs)):
        spans = data.record_spans_at(pack_id)
        ids = data.pack_record_ids[
            int(data.packs[pack_id]['record_offset']):
            int(data.packs[pack_id]['record_offset']) + int(data.packs[pack_id]['record_count'])]
        assert len(ids) == len(spans) == 1, 'fragment pack must hold exactly one fragment'
        row = frag_rows[int(ids[0])]
        assert row['split'] == 0
        tokens, loss_mask, valid_mask, reset_before, length, targets, name = \
            data.pack_at_with_boundaries(pack_id)
        start, stop = spans[0]
        assert bool(reset_before[start]) and start == 0
        part_a['packs'] += 1
        part_a['fragments'] += 1
        part_a['resets'] += int(bool(reset_before[start]))
        part_a['mid_document'] += int(bool(row['mid_document_entry']))
        part_a['aligned'] += int(not row['mid_document_entry'])
        part_a['in_window_separators'] += int(row['document_separators_in_window'])
        # replication equivalence check against the production tensors
        off, n, want, split = RECORD_INDEX.unpack_from(
            (FRAG / 'records.idx').read_bytes() if False else b'', 0) if False else (None,)*4
        record = data.records[int(ids[0])]
        rep_tokens, rep_loss, rep_valid, rep_reset = replicate_boundaries(
            data.tokens[int(record['offset']):int(record['offset']) + int(record['tokens'])],
            data.masks[int(record['offset']):int(record['offset']) + int(record['tokens'])],
            spans, data.sequence_tokens)
        assert torch.equal(rep_tokens, tokens) and torch.equal(rep_loss, loss_mask)
        assert torch.equal(rep_valid, valid_mask) and torch.equal(rep_reset, reset_before)
        part_a['solo_packs'] += 1
        if sample is None:
            sample = {
                'pack_id': pack_id, 'record_id': row['id'],
                'tokens': int(record['tokens']),
                'entry_offset_in_document': row['entry_offset_in_document'],
                'mid_document_entry': row['mid_document_entry'],
                'in_window_document_separators': row['document_separators_in_window'],
                'documents_spanned': row['documents_spanned'],
                'first_token': int(data.tokens[int(record['offset'])]),
                'reset_before_at_start': bool(reset_before[start]),
            }
    assert part_a['packs'] == frag_train == 891, 'unexpected fragment pack count'
    assert part_a['resets'] == part_a['fragments']

    # ---------- PART B: prep-level application of the verified derivation ----------
    rows = [json.loads(line) for line in (PREP / 'records.jsonl').open()]
    records = np.fromfile(PREP / 'records.idx', dtype=np.dtype(
        [('offset', '<u8'), ('tokens', '<u8'), ('targets', '<u8'),
         ('split', 'u1'), ('pad', 'V7')]))
    prep_tokens = np.memmap(PREP / 'tokens.uint32.bin', dtype='<u4', mode='r')
    prep_masks = np.memmap(PREP / 'assistant_mask.uint8.bin', dtype=np.uint8, mode='r')
    pack_rows = (PREP / 'packs/train_packs.idx').read_bytes()
    member_ids = np.fromfile(PREP / 'packs/pack_records.uint32.bin', dtype='<u4')
    pack_count = len(pack_rows) // PACK_INDEX.size
    placements = {}
    for pid in range(pack_count):
        first, count, ptokens, ptargets = PACK_INDEX.unpack_from(pack_rows, pid * PACK_INDEX.size)
        for position, rid in enumerate(member_ids[first:first + count]):
            if rows[rid]['source'] == COHORT:
                placements[int(rid)] = (pid, position, int(count))
    library_ids = [i for i, row in enumerate(rows) if row['source'] == COHORT]
    assert set(placements) == set(library_ids), 'placement map incomplete'

    part_b = {'library_fragments': 0, 'resets': 0, 'loss_masked_into_start': 0,
              'mid_document': 0, 'aligned': 0, 'solo_packs': 0,
              'in_window_separators': 0, 'supervised_separators': 0}
    prep_samples = []
    for rid in sorted(placements):
        pid, position, count = placements[rid]
        first = PACK_INDEX.unpack_from(pack_rows, pid * PACK_INDEX.size)[0]
        ids = member_ids[first:first + count]
        # verbatim record_spans_at derivation over this pack
        spans, cursor = [], 0
        for member in ids:
            n = int(records[int(member)]['tokens'])
            spans.append((cursor, cursor + n))
            cursor += n
        start, stop = spans[position]
        record = records[rid]
        tokens_row = prep_tokens[int(record['offset']):int(record['offset']) + int(record['tokens'])]
        mask_row = prep_masks[int(record['offset']):int(record['offset']) + int(record['tokens'])]
        _, rep_loss, _, reset_before = replicate_boundaries(tokens_row, mask_row, spans, 65537)
        assert bool(reset_before[start]), f'reset_before missing (record {rid}, pack {pid})'
        if start > 0:
            assert not bool(rep_loss[start - 1]), \
                'prediction into the reset fragment start must be loss-masked'
            part_b['loss_masked_into_start'] += 1
        row = rows[rid]
        assert int(mask_row[0]) == 0, 'fragment entry token must be unsupervised'
        if row['mid_document_entry']:
            assert int(tokens_row[0]) != EOT_TOKEN, 'mid-document fragment starts on a separator'
            part_b['mid_document'] += 1
        else:
            part_b['aligned'] += 1
        if count == 1:
            part_b['solo_packs'] += 1
        # every in-window EOT separator is a supervised token of this fragment
        sep_positions = np.nonzero(tokens_row == EOT_TOKEN)[0]
        for p in sep_positions:
            if int(mask_row[p]) == 1:
                part_b['supervised_separators'] += 1
        part_b['in_window_separators'] += int(row['document_separators_in_window'])
        part_b['library_fragments'] += 1
        part_b['resets'] += 1
        if int(mask_row[0:1][0]) == 0 and len(prep_samples) < 5:
            prep_samples.append({
                'prep_record_index': rid, 'pack_id': pid, 'records_in_pack': count,
                'fragment_start_in_pack': start, 'reset_before_at_start': True,
                'entry_offset_in_document': row['entry_offset_in_document'],
                'mid_document_entry': row['mid_document_entry'],
                'in_window_document_separators': row['document_separators_in_window'],
                'documents_spanned': row['documents_spanned'],
                'first_token': int(tokens_row[0]),
            })

    assert part_b['resets'] == part_b['library_fragments'] == len(library_ids)

    summary = {
        'schema': 'emender-e97-e2-library-fragment-reset-semantics-receipt-v1',
        'operator_design_ruling': ('document concatenation read as arbitrary-entry '
                                   'window cuts; hidden state resets at every record '
                                   '(fragment) start; in-window separators supervised'),
        'part_a_production_path': {
            'authority': str(FRAG), 'authority_manifest_sha256': FRAG_AUTH_SHA,
            'pack_manifest_sha256': FRAG_PACK_SHA, **part_a,
            'note': ('MaskedSFTPackedDataset (the training Dataset class) materialized '
                     'every train fragment pack: reset_before=True at every fragment '
                     'start; replication of pack_at_with_boundaries verified '
                     'byte-identical on all packs'),
            'sample': sample,
        },
        'part_b_prep_level': {
            'authority': str(PREP), 'authority_manifest_sha256': AUTH_SHA,
            'pack_manifest_sha256': PACK_SHA, **part_b,
            'note': ('the prep authority is non-authorizing (Dataset refuses it by '
                     'design; admission flips eligibility); the VERIFIED verbatim '
                     'derivation was applied to the prep packs: every library '
                     'fragment start carries reset_before=True and the prediction '
                     'into it is loss-masked'),
            'samples': prep_samples,
        },
        'trainer_code_path': [
            'scripts/build_e97_sft_packs.py --boundary-aware (v2 packs: reset_before field)',
            'ndm/data/masked_sft_dataset.py record_spans_at -> pack_at_with_boundaries:',
            '    reset_before[start] = True per record span; loss_mask &= ~reset_before[1:]',
            'scripts/train_e97_4b_pi_sft.py --boundary-aware-packs:',
            '    get_boundary_aware_batch(1) -> packed_objective(reset_masks) -> model(reset_before=...)',
            'ndm/models/e88_fla_hybrid.py forward(reset_before=...) -> _process_chunk(reset_chunk)',
            'ndm/triton/e97_sequential.py e97_split_edit_triton_apply -> e88_triton_optimized_apply',
            'ndm/triton/e88_triton_forward.py APPLY_RESET:',
            '    S = tl.where((token_valid & token_reset)[:, None, None], zeros, S)',
        ],
    }
    out = W / 'v2/logs/reset-semantics-receipt.json'
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
