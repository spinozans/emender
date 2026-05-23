"""Tokenized streaming dataset — same interface as DocumentStreamDataset but
tokens come from tiktoken BPE instead of raw bytes.

Design:
- mmap the raw byte corpus
- per __getitem__, pick a random position, read enough bytes that tokenizing
  will produce > chunk_size tokens, tokenize, drop the first token (likely
  a mid-token split at the chosen start position), return the next chunk_size.
- Lazy / on-the-fly; no pre-tokenization needed.
"""

import mmap
import numpy as np
import torch
from torch.utils.data import Dataset
import tiktoken


class TokenizedStreamDataset(Dataset):
    """Mmap the corpus, tokenize chunks with tiktoken, return token tensors."""

    # Bytes-per-token average for sizing mmap reads. tiktoken GPT-2 averages
    # ~4 B/tok on English, ~3 B/tok on commapile's mixed code+text. Use a
    # generous 6x safety factor so we never under-read.
    BYTES_PER_TOKEN_SAFETY = 6

    def __init__(
        self,
        data_path: str,
        chunk_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        tokenizer_name: str = 'gpt2',
    ):
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.enc.n_vocab

        self.data_file = open(data_path, 'rb')
        self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_size = len(self.mmap)

        self.rng = np.random.RandomState(seed + rank)

        # Stats
        self.chunks_served = 0
        self.bytes_processed = 0
        self.tokens_served = 0

    def __len__(self):
        return 1_000_000_000  # effectively infinite

    def __getitem__(self, idx):
        """Returns (chunk_tensor, is_final_chunk_in_doc, actual_chunk_length).

        Matches DocumentStreamDataset's signature so train.py's code path works
        unchanged. is_final_chunk_in_doc is always False (we don't track docs
        in the tokenized path).
        """
        need_bytes = self.chunk_size * self.BYTES_PER_TOKEN_SAFETY
        max_start = max(1, self.file_size - need_bytes - 1)

        while True:
            pos = self.rng.randint(0, max_start)
            raw = bytes(self.mmap[pos:pos + need_bytes])
            # Decode bytes to str with error replacement so tiktoken doesn't
            # choke on mid-sequence UTF-8. The start byte may be mid-codepoint
            # or mid-token; we drop the first token below to realign.
            try:
                s = raw.decode('utf-8', errors='replace')
                tokens = self.enc.encode(s, disallowed_special=())
            except Exception:
                continue
            # Drop the first token (likely mis-aligned) and take chunk_size.
            if len(tokens) < self.chunk_size + 2:
                # Not enough tokens produced — try another position.
                continue
            tokens = tokens[1:self.chunk_size + 1]
            if len(tokens) < self.chunk_size:
                continue
            break

        chunk = torch.tensor(tokens, dtype=torch.long)
        self.chunks_served += 1
        self.tokens_served += self.chunk_size
        self.bytes_processed += need_bytes
        return chunk, False, self.chunk_size

    def get_batch(self, batch_size: int, device=None):
        """Match DocumentStreamDataset.get_batch signature."""
        if not hasattr(self, '_pinned_chunks') or self._pinned_chunks.shape[0] != batch_size:
            self._pinned_chunks = torch.empty(batch_size, self.chunk_size, dtype=torch.long, pin_memory=True)
            self._pinned_doc_ends = torch.empty(batch_size, dtype=torch.bool, pin_memory=True)
            self._pinned_lengths = torch.empty(batch_size, dtype=torch.long, pin_memory=True)

        for i in range(batch_size):
            chunk, is_doc_end, actual_length = self[0]
            self._pinned_chunks[i] = chunk
            self._pinned_doc_ends[i] = is_doc_end
            self._pinned_lengths[i] = actual_length

        if device is not None:
            chunks = self._pinned_chunks.to(device, non_blocking=True)
            is_doc_end = self._pinned_doc_ends.to(device, non_blocking=True)
            actual_lengths = self._pinned_lengths.to(device, non_blocking=True)
            return chunks, is_doc_end, actual_lengths
        return self._pinned_chunks, self._pinned_doc_ends, self._pinned_lengths

    def get_stats(self):
        return {
            'chunks_served': self.chunks_served,
            'tokens_served': self.tokens_served,
            'bytes_processed': self.bytes_processed,
        }
