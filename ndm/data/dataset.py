"""
Document-aware dataset for language modeling with streaming tokenization.

Data format: Raw bytes with 0x1e (ASCII record separator) as document delimiter.

Modes:
1. DocumentStreamDataset - Single stream for byte-level, advances on each __getitem__
2. BatchedStreamDataset - batch_size independent streams for byte-level TBPTT
3. TokenizedStreamDataset - Streaming tokenization (tiktoken, etc.) on-the-fly
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mmap
from typing import Optional, List


class DocumentStreamDataset(Dataset):
    """
    Document-aware streaming dataset for training.

    Key features:
    - Respects document boundaries (0x1e delimiter)
    - Byte-level tokenization (vocab size 256)
    - Memory-mapped file access for efficiency
    - Random starting position per rank
    """

    def __init__(self, data_path: str, chunk_size: int, rank: int = 0,
                 world_size: int = 1, seed: int = 42):
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size

        # Open the data file with memory mapping
        self.data_file = open(data_path, 'rb')
        self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_size = len(self.mmap)

        # Random starting position for this rank
        rng = np.random.RandomState(seed + rank)
        self.position = rng.randint(0, max(1, self.file_size - 1000))

        # Track statistics
        self.chunks_served = 0
        self.docs_completed = 0
        self.bytes_processed = 0
        self.wraps = 0

        # Scan forward to next document boundary to start clean
        self._scan_to_next_document()

        # Buffer for accumulating bytes
        self.token_buffer = []

    def __len__(self):
        return 1_000_000_000  # Effectively infinite

    def _scan_to_next_document(self):
        """Scan forward to the start of the next document."""
        while self.position < self.file_size and self.mmap[self.position] != 0x1e:
            self.position += 1

        # Skip the delimiter itself
        if self.position < self.file_size:
            self.position += 1
        else:
            self.position = 0
            self.wraps += 1

    def __getitem__(self, idx):
        """
        Returns: (chunk_tensor, is_final_chunk_in_doc, actual_chunk_length)

        Reads raw bytes sequentially, including 0x1e delimiters as regular byte tokens.
        No special handling of document boundaries — all bytes are data.
        """
        while len(self.token_buffer) < self.chunk_size:
            # Check if we need to wrap
            if self.position >= self.file_size:
                self.position = 0
                self.wraps += 1

            # Read in larger chunks for efficiency
            bytes_needed = self.chunk_size - len(self.token_buffer)
            end_pos = min(self.position + bytes_needed, self.file_size)
            raw_bytes = self.mmap[self.position:end_pos]

            self.token_buffer.extend(raw_bytes)
            self.bytes_processed += len(raw_bytes)
            self.position = end_pos

        # Full chunk
        chunk = torch.tensor(self.token_buffer[:self.chunk_size], dtype=torch.long)
        self.token_buffer = self.token_buffer[self.chunk_size:]
        self.chunks_served += 1

        return chunk, False, self.chunk_size

    def get_stats(self):
        """Return current dataset statistics."""
        return {
            'chunks_served': self.chunks_served,
            'docs_completed': self.docs_completed,
            'bytes_processed': self.bytes_processed,
            'wraps': self.wraps,
            'position': self.position
        }

    def get_batch(self, batch_size: int, device=None):
        """
        Get a batch of chunks using pinned memory for efficient GPU transfer.

        Args:
            batch_size: Number of chunks to get
            device: Target device (GPU) for transfer

        Returns:
            chunks: [batch_size, chunk_size] tensor
            is_doc_end: [batch_size] boolean tensor
            actual_lengths: [batch_size] tensor of actual lengths
        """
        # Lazily initialize pinned memory buffers
        if not hasattr(self, '_pinned_chunks') or self._pinned_chunks.shape[0] != batch_size:
            self._pinned_chunks = torch.empty(batch_size, self.chunk_size, dtype=torch.long, pin_memory=True)
            self._pinned_doc_ends = torch.empty(batch_size, dtype=torch.bool, pin_memory=True)
            self._pinned_lengths = torch.empty(batch_size, dtype=torch.long, pin_memory=True)

        # Fill buffers directly
        for i in range(batch_size):
            chunk, is_doc_end, actual_length = self[0]
            self._pinned_chunks[i] = chunk
            self._pinned_doc_ends[i] = is_doc_end
            self._pinned_lengths[i] = actual_length

        if device is not None:
            # Use non_blocking=True with pinned memory for async transfer
            chunks = self._pinned_chunks.to(device, non_blocking=True)
            is_doc_end = self._pinned_doc_ends.to(device, non_blocking=True)
            actual_lengths = self._pinned_lengths.to(device, non_blocking=True)
        else:
            chunks = self._pinned_chunks.clone()
            is_doc_end = self._pinned_doc_ends.clone()
            actual_lengths = self._pinned_lengths.clone()

        return chunks, is_doc_end, actual_lengths

    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'data_file'):
            self.data_file.close()


class BatchedStreamDataset:
    """
    Multi-stream dataset for TBPTT training.

    Each batch element has its own independent data stream that persists
    across calls. This is REQUIRED for TBPTT to work correctly - hidden
    states must match up with the data streams they were computed from.

    Usage:
        dataset = BatchedStreamDataset(path, batch_size=16, chunk_size=512)
        for step in range(num_steps):
            chunks, is_doc_end = dataset.get_batch()
            # chunks[i] continues from where chunks[i] left off last step
    """

    def __init__(self, data_path: str, batch_size: int, chunk_size: int,
                 rank: int = 0, world_size: int = 1, seed: int = 42):
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        # Open the data file with memory mapping
        self.data_file = open(data_path, 'rb')
        self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_size = len(self.mmap)

        # Pre-allocate pinned memory buffer for faster GPU transfers
        self._pinned_chunks = torch.empty(batch_size, chunk_size, dtype=torch.long, pin_memory=True)
        self._pinned_doc_ends = torch.empty(batch_size, dtype=torch.bool, pin_memory=True)

        # Each batch element has its own stream state
        # Spread starting positions evenly across file
        rng = np.random.RandomState(seed + rank * 1000)
        total_streams = batch_size * world_size
        stream_offset = rank * batch_size

        self.positions = []
        self.buffers = []
        for i in range(batch_size):
            # Evenly spaced + small random offset
            base_pos = (stream_offset + i) * self.file_size // total_streams
            jitter = rng.randint(0, max(1, self.file_size // (total_streams * 10)))
            pos = (base_pos + jitter) % self.file_size
            self.positions.append(pos)
            self.buffers.append([])

        # Scan each stream to next document boundary
        for i in range(batch_size):
            self._scan_to_next_document(i)

    def _scan_to_next_document(self, stream_idx: int):
        """Scan stream to the start of the next document."""
        while self.positions[stream_idx] < self.file_size:
            if self.mmap[self.positions[stream_idx]] == 0x1e:
                self.positions[stream_idx] = (self.positions[stream_idx] + 1) % self.file_size
                return
            self.positions[stream_idx] += 1
        self.positions[stream_idx] = 0

    def _get_chunk(self, stream_idx: int):
        """Get next chunk from a specific stream. All bytes including 0x1e are data."""
        buf = self.buffers[stream_idx]

        while len(buf) < self.chunk_size:
            pos = self.positions[stream_idx]
            if pos >= self.file_size:
                self.positions[stream_idx] = 0
                pos = 0

            byte_val = self.mmap[pos]
            self.positions[stream_idx] = pos + 1
            buf.append(byte_val)

        # Full chunk
        chunk = torch.tensor(buf[:self.chunk_size], dtype=torch.long)
        self.buffers[stream_idx] = buf[self.chunk_size:]
        return chunk, False, self.chunk_size

    def get_batch(self, device=None):
        """
        Get one batch where each element continues its own stream.

        Returns:
            chunks: [batch_size, chunk_size] tensor
            is_doc_end: [batch_size] boolean tensor (True if chunk ends at doc boundary)
        """
        # Fill pinned memory buffers directly (avoids tensor creation/stacking overhead)
        for i in range(self.batch_size):
            chunk, is_doc_end, _ = self._get_chunk(i)
            self._pinned_chunks[i] = chunk
            self._pinned_doc_ends[i] = is_doc_end

        if device is not None:
            # Use non_blocking=True with pinned memory for async transfer
            chunks = self._pinned_chunks.to(device, non_blocking=True)
            is_doc_end = self._pinned_doc_ends.to(device, non_blocking=True)
        else:
            chunks = self._pinned_chunks.clone()
            is_doc_end = self._pinned_doc_ends.clone()

        return chunks, is_doc_end

    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'data_file'):
            self.data_file.close()


def create_dataloader(data_path: str, batch_size: int, chunk_size: int,
                      device=None, num_workers: int = 0, seed: int = 42):
    """
    Create a dataloader for training.

    Args:
        data_path: Path to training data file
        batch_size: Batch size
        chunk_size: Sequence length
        device: Target device (optional)
        num_workers: Number of data loading workers
        seed: Random seed

    Returns:
        DataLoader that yields (chunks, is_doc_end, actual_lengths)
    """
    dataset = DocumentStreamDataset(
        data_path=data_path,
        chunk_size=chunk_size,
        rank=0,
        world_size=1,
        seed=seed,
    )

    def collate_fn(batch):
        """Collate function that handles doc boundary info."""
        chunks = torch.stack([b[0] for b in batch])
        is_doc_end = torch.tensor([b[1] for b in batch], dtype=torch.bool)
        actual_lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)

        if device is not None:
            chunks = chunks.to(device)
            is_doc_end = is_doc_end.to(device)
            actual_lengths = actual_lengths.to(device)

        return chunks, is_doc_end, actual_lengths

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return dataloader


class TokenizedStreamDataset:
    """
    Streaming tokenization dataset for TBPTT training with subword tokenizers.

    Reads raw text, tokenizes on-the-fly with configurable tokenizer (tiktoken, etc.).
    Each batch element has its own independent data stream that persists across calls.

    Like gruboros: buffer raw bytes, periodically flush to text and tokenize,
    accumulate tokens in separate buffer, return fixed-size chunks.

    Args:
        data_path: Path to training data file
        tokenizer: Tokenizer object with encode(text) -> List[int] method
        batch_size: Number of independent streams
        chunk_size: Sequence length per chunk
        rank: DDP rank (for data sharding)
        world_size: Total number of DDP processes
        seed: Random seed for reproducibility
        text_buffer_size: Bytes to accumulate before tokenizing (default 4096)
        doc_delimiter: Document delimiter byte (default 0x1e, use None for no delimiter)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        batch_size: int,
        chunk_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        text_buffer_size: int = 4096,
        doc_delimiter: int = 0x1e,  # ASCII record separator
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.text_buffer_size = text_buffer_size
        self.doc_delimiter = doc_delimiter

        # Open the data file with memory mapping
        self.data_file = open(data_path, 'rb')
        self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_size = len(self.mmap)

        # Each batch element has its own stream state
        rng = np.random.RandomState(seed + rank * 1000)
        total_streams = batch_size * world_size
        stream_offset = rank * batch_size

        self.positions = []      # File positions per stream
        self.byte_buffers = []   # Raw byte buffers per stream
        self.token_buffers = []  # Token buffers per stream

        for i in range(batch_size):
            # Evenly spaced starting positions + jitter
            base_pos = (stream_offset + i) * self.file_size // total_streams
            jitter = rng.randint(0, max(1, self.file_size // (total_streams * 10)))
            pos = (base_pos + jitter) % self.file_size
            self.positions.append(pos)
            self.byte_buffers.append([])
            self.token_buffers.append([])

        # Scan each stream to next document boundary (only if delimiter is set)
        if self.doc_delimiter is not None:
            for i in range(batch_size):
                self._scan_to_next_document(i)

    def _scan_to_next_document(self, stream_idx: int):
        """Scan stream to the start of the next document."""
        if self.doc_delimiter is None:
            return
        while self.positions[stream_idx] < self.file_size:
            if self.mmap[self.positions[stream_idx]] == self.doc_delimiter:
                self.positions[stream_idx] = (self.positions[stream_idx] + 1) % self.file_size
                return
            self.positions[stream_idx] += 1
        self.positions[stream_idx] = 0

    def _flush_bytes_to_tokens(self, stream_idx: int):
        """Convert accumulated bytes to tokens."""
        if not self.byte_buffers[stream_idx]:
            return

        try:
            text = bytes(self.byte_buffers[stream_idx]).decode('utf-8', errors='ignore')
            if text:
                tokens = self.tokenizer.encode(text)
                self.token_buffers[stream_idx].extend(tokens)
        except Exception:
            pass  # Skip malformed text

        self.byte_buffers[stream_idx] = []

    def _get_chunk(self, stream_idx: int):
        """Get next chunk from a specific stream."""
        doc_ended = False

        # Fill token buffer until we have enough — skip delimiters, concatenate docs
        while len(self.token_buffers[stream_idx]) < self.chunk_size:
            # Accumulate bytes
            bytes_read = 0
            while bytes_read < self.text_buffer_size:
                pos = self.positions[stream_idx]
                if pos >= self.file_size:
                    self.positions[stream_idx] = 0
                    pos = 0

                byte_val = self.mmap[pos]
                self.positions[stream_idx] = pos + 1

                self.byte_buffers[stream_idx].append(byte_val)
                bytes_read += 1

            # Flush accumulated bytes to tokens
            self._flush_bytes_to_tokens(stream_idx)

        # Full chunk (always chunk_size real tokens, no padding)
        buf = self.token_buffers[stream_idx]
        chunk = torch.tensor(buf[:self.chunk_size], dtype=torch.long)
        self.token_buffers[stream_idx] = buf[self.chunk_size:]
        return chunk, False, self.chunk_size

    def get_batch(self, device=None):
        """
        Get one batch where each element continues its own stream.

        Returns:
            chunks: [batch_size, chunk_size] tensor of token IDs
            is_doc_end: [batch_size] boolean tensor
            actual_lengths: [batch_size] tensor of actual (non-padded) lengths
        """
        chunks = []
        is_doc_ends = []
        actual_lengths = []

        for i in range(self.batch_size):
            chunk, is_doc_end, actual_len = self._get_chunk(i)
            chunks.append(chunk)
            is_doc_ends.append(is_doc_end)
            actual_lengths.append(actual_len)

        chunks = torch.stack(chunks)
        is_doc_end = torch.tensor(is_doc_ends, dtype=torch.bool)
        actual_lengths = torch.tensor(actual_lengths, dtype=torch.long)

        if device is not None:
            chunks = chunks.to(device)
            is_doc_end = is_doc_end.to(device)
            actual_lengths = actual_lengths.to(device)

        return chunks, is_doc_end, actual_lengths

    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'data_file'):
            self.data_file.close()


class FastTokenizedDataset:
    """
    Fast pre-tokenized dataset for maximum throughput.

    Pre-tokenizes the entire file once and caches as .npy for instant loading.
    Uses memory-mapped numpy arrays for zero-copy data access.

    ~100x faster than TokenizedStreamDataset.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        batch_size: int,
        chunk_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer

        # Cache path based on tokenizer name
        tokenizer_name = getattr(tokenizer, 'name', 'default')
        if hasattr(tokenizer, 'encoding'):
            tokenizer_name = str(tokenizer.encoding.name)
        cache_path = data_path + f'.{tokenizer_name}.tokens.npy'

        # Load or create tokenized cache
        if not self._load_cache(cache_path):
            self._create_cache(data_path, cache_path)
            self._load_cache(cache_path)

        # Setup stream positions
        rng = np.random.RandomState(seed + rank * 1000)
        total_streams = batch_size * world_size
        stream_offset = rank * batch_size

        self.positions = []
        for i in range(batch_size):
            base_pos = (stream_offset + i) * self.num_tokens // total_streams
            jitter = rng.randint(0, max(1, self.num_tokens // (total_streams * 10)))
            pos = (base_pos + jitter) % self.num_tokens
            self.positions.append(pos)

        # Pre-allocate output tensors for speed
        self._chunk_buffer = torch.zeros(batch_size, chunk_size, dtype=torch.long)

    def _load_cache(self, cache_path: str) -> bool:
        """Load pre-tokenized cache if exists."""
        try:
            import os
            if os.path.exists(cache_path):
                # Memory-map the token array for zero-copy access
                self.tokens = np.load(cache_path, mmap_mode='r')
                self.num_tokens = len(self.tokens)
                print(f"Loaded {self.num_tokens:,} tokens from cache: {cache_path}")
                return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
        return False

    def _create_cache(self, data_path: str, cache_path: str):
        """Pre-tokenize entire file and save to cache."""
        print(f"Pre-tokenizing {data_path}...")

        # Read entire file
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Replace document delimiters with newlines for cleaner tokenization
        text = text.replace('\x1e', '\n')

        # Tokenize in chunks to avoid memory issues
        chunk_size = 10_000_000  # 10MB chunks
        all_tokens = []

        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i+chunk_size]
            tokens = self.tokenizer.encode(chunk_text)
            all_tokens.extend(tokens)
            if (i // chunk_size) % 10 == 0:
                print(f"  Tokenized {i:,}/{len(text):,} chars...")

        # Save as numpy array
        tokens_array = np.array(all_tokens, dtype=np.int32)
        np.save(cache_path, tokens_array)
        print(f"Saved {len(tokens_array):,} tokens to {cache_path}")

    def get_batch(self, device=None):
        """
        Get one batch - each element continues its own stream.

        Returns:
            chunks: [batch_size, chunk_size] tensor of token IDs
            is_doc_end: [batch_size] boolean tensor (always False - no doc tracking)
            actual_lengths: [batch_size] tensor (always chunk_size)
        """
        # Fast path: copy directly from mmap'd array
        for i in range(self.batch_size):
            pos = self.positions[i]
            end_pos = pos + self.chunk_size

            if end_pos <= self.num_tokens:
                # Fast case: no wrap needed
                self._chunk_buffer[i] = torch.from_numpy(
                    self.tokens[pos:end_pos].astype(np.int64)
                )
            else:
                # Wrap around
                first_part = self.num_tokens - pos
                self._chunk_buffer[i, :first_part] = torch.from_numpy(
                    self.tokens[pos:].astype(np.int64)
                )
                self._chunk_buffer[i, first_part:] = torch.from_numpy(
                    self.tokens[:self.chunk_size - first_part].astype(np.int64)
                )

            # Advance position
            self.positions[i] = end_pos % self.num_tokens

        chunks = self._chunk_buffer
        is_doc_end = torch.zeros(self.batch_size, dtype=torch.bool)
        actual_lengths = torch.full((self.batch_size,), self.chunk_size, dtype=torch.long)

        if device is not None:
            chunks = chunks.to(device, non_blocking=True)
            is_doc_end = is_doc_end.to(device, non_blocking=True)
            actual_lengths = actual_lengths.to(device, non_blocking=True)

        return chunks, is_doc_end, actual_lengths
