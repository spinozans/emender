"""Configurable tokenization with support for bytes, TikToken, and SentencePiece."""
import torch
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """Return end-of-sequence token ID."""
        pass


class ByteTokenizer(BaseTokenizer):
    """Simple byte-level tokenizer (current approach)."""

    def __init__(self):
        self._vocab_size = 256
        self._eos_token_id = 0  # Use null byte as EOS

    def encode(self, text: str) -> List[int]:
        return list(text.encode('utf-8'))

    def decode(self, tokens: List[int]) -> str:
        # Handle both lists and tensors
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return bytes(tokens).decode('utf-8', errors='replace')

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id


class TikTokenTokenizer(BaseTokenizer):
    """TikToken BPE tokenizer."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Args:
            encoding_name: One of:
                - "cl100k_base" (GPT-4, GPT-3.5-turbo) - 100K vocab
                - "p50k_base" (GPT-3, Codex) - 50K vocab
                - "r50k_base" (GPT-3, older) - 50K vocab
                - "o200k_base" (GPT-4o) - 200K vocab
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError("Install tiktoken: pip install tiktoken")

        self.encoding = tiktoken.get_encoding(encoding_name)
        self._eos_token_id = self.encoding.eot_token  # End of text token
        self._encoding_name = encoding_name

    def encode(self, text: str) -> List[int]:
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, tokens: List[int]) -> str:
        # Handle both lists and tensors
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.encoding.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def __repr__(self):
        return f"TikTokenTokenizer(encoding={self._encoding_name}, vocab_size={self.vocab_size})"


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece BPE tokenizer."""

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to .model file from SentencePiece training
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("Install sentencepiece: pip install sentencepiece")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self._eos_token_id = self.sp.eos_id()
        self._model_path = model_path

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: List[int]) -> str:
        # Handle both lists and tensors
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.sp.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def __repr__(self):
        return f"SentencePieceTokenizer(model={self._model_path}, vocab_size={self.vocab_size})"


class HuggingFaceTokenizer(BaseTokenizer):
    """Wrapper for any HuggingFace tokenizer (for maximum flexibility)."""

    def __init__(self, tokenizer_name: str, use_fast: bool = True):
        """
        Args:
            tokenizer_name: HF model name or path
            use_fast: Use fast (Rust) tokenizer if available
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=use_fast
        )
        self._eos_token_id = self.tokenizer.eos_token_id or 0
        self._tokenizer_name = tokenizer_name

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        # Handle both lists and tensors
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def __repr__(self):
        return f"HuggingFaceTokenizer(name={self._tokenizer_name}, vocab_size={self.vocab_size})"


def get_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
    """Factory function to create tokenizers."""
    tokenizer_map = {
        'byte': ByteTokenizer,
        'tiktoken': TikTokenTokenizer,
        'sentencepiece': SentencePieceTokenizer,
        'huggingface': HuggingFaceTokenizer,
    }

    if tokenizer_type not in tokenizer_map:
        raise ValueError(f"Unknown tokenizer: {tokenizer_type}. "
                        f"Choose from: {list(tokenizer_map.keys())}")

    return tokenizer_map[tokenizer_type](**kwargs)
