from . import tool
from typing import Iterable
import re

try:
    import regex as regex_re
except ImportError:
    regex_re = None


class Tokenizer:
    def __init__(self, vocab: tuple[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # Sort longest-first so overlapping specials (e.g. "<|eot|>" vs
        # "<|eot|><|eot|>") match the longer alternative first.
        sorted_specials = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        special_pattern = "|".join(re.escape(tok) for tok in sorted_specials)
        if special_pattern:
            special_pattern = f"({special_pattern})"

        self.special_pattern = special_pattern
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [tok.encode('utf-8') for tok in special_tokens] if special_tokens else []

        self.merge = merges
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        # the original format for vocab is (ID, Token), reverse it
        self.vocab_id_token = {k: v for k, v in vocab.items()}
        self.vocab_token_id = {v: k for k, v in vocab.items()}

        # get vocab token set
        vocab_tokens = list(self.vocab_id_token.keys())
        if self.special_tokens:
            for special_token in self.special_tokens:
                if special_token not in vocab_tokens:
                    self.vocab_id_token[special_token] = len(self.vocab_id_token)
                    self.vocab_token_id[len(self.vocab_token_id)] = special_token

    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        self = cls.__new__(cls)
        vocab = tool.load_bpe(vocab_filepath)
        merges = tool.load_bpe(merges_filepath)
        self._init(vocab, merges, special_tokens)
        return self
    
    def _pre_tokenize(self, text: str) -> list[bytes]:
        token_list = []
        # Pre-tokenize the input text into a list of bytes
        segments = re.split(self.special_pattern, text) if self.special_pattern else [text]
        for segment in segments:
            if segment in self.special_tokens:
                token_list.append(segment)
            else:
                # Tokenize the segment and count the tokens
                token_list.extend(regex_re.findall(self.PAT, segment))
        return [token.encode('utf-8') for token in token_list]

    def _merge(self, pre_token: bytes) -> list[int]:
        # Merge the pre-tokenized token into a list of token IDs
        # This is a placeholder for the actual merging logic
        merged_bytes_list = [bytes([i]) for i in pre_token]
        while True:
            best_rank = None
            best_rank_id = None
            for i in range(len(merged_bytes_list) - 1):
                rank = self.merge_rank.get((merged_bytes_list[i], merged_bytes_list[i + 1]), None)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_rank_id = i
            if best_rank_id is not None:
                merged_bytes_list[best_rank_id] = merged_bytes_list[best_rank_id] + merged_bytes_list[best_rank_id + 1]
                del merged_bytes_list[best_rank_id + 1]
            else:
                break
        return [self.vocab_token_id[token] for token in merged_bytes_list]


    def encode(self, text: str) -> list[int]:
        # Encode an input text into a sequence of token IDs
        if not text:
            return []
        
        pre_token_list = self._pre_tokenize(text)
        ids = []
        for pre_token in pre_token_list:
            # Process each pre-tokenized token  
            if pre_token in self.special_tokens_bytes:
                ids.append(self.vocab_token_id[pre_token])
            else:
                ids.extend(self._merge(pre_token))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        # Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is 
        # required for memory-efficient tokenization of large files that we cannot directly load into  memory
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        # Decode a sequence of token IDs into text
        return b"".join(self.vocab_id_token[i] for i in ids).decode("utf-8", errors="replace")