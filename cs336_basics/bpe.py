from multiprocessing import Pool
import os
import re
import threading
from typing import BinaryIO
from collections import Counter, defaultdict

try:
    import regex as regex_re
except ImportError:
    regex_re = None

import heapq

class _PairHeapEntry:
    """A snapshot of a pair's count, used as an element in a max-heap.

    Heap ordering rules (so heapq's min-heap behaves as a max-heap with BPE tie-break):
      1. Higher count wins.
      2. On equal counts, lexicographically LARGER pair wins (BPE rule).

    Entries are immutable snapshots — when a pair's count changes, push a NEW
    entry instead of mutating an existing one. Stale entries are filtered out
    at pop time by comparing `count` against the authoritative pair_to_count dict.
    """
    __slots__ = ('count', 'pair')

    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair

    def __lt__(self, other: '_PairHeapEntry') -> bool:
        # heapq is a min-heap, so "smaller" means "comes out first".
        # We want the max count first, and on ties the larger pair first.
        if self.count != other.count:
            return self.count > other.count   # invert: higher count → "smaller" in heap
        return self.pair > other.pair         # invert: larger pair → "smaller" in heap

    def __repr__(self) -> str:
        return f"_PairHeapEntry(count={self.count}, pair={self.pair})"
    
def find_chunk_boundaries(
    file_path: str | os.PathLike,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    with open(file_path, 'rb') as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

def bpe_vocab_init(vocab_size: int, special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    # Assign IDs for special tokens above the range of bytes
    for i, tok in enumerate(special_tokens, start=256):
        vocab[i] = tok.encode('utf-8')

    assert len(vocab) <= vocab_size, "Initial vocabulary size is greater than the desired vocabulary size"
    return vocab

def bpe_update_vocab(vocab, pair_to_merge):
    max_id = max(vocab.keys())
    vocab[max_id + 1] = pair_to_merge  # assign next available integer
    return vocab

def _init_worker(special_tokens: list[str], pat: str, file_path: str | os.PathLike):
    """Initialize worker process with shared data."""
    global _worker_special_tokens, _worker_pat, _worker_file
    _worker_special_tokens = special_tokens
    _worker_pat = pat
    _worker_file = open(file_path, 'rb')

def _tokenize_chunk_and_count(chunk_boundary: tuple[int, int]) -> Counter[bytes, int]:
    """Tokenize a chunk of the file and count token pairs."""
    start, end = chunk_boundary
    _worker_file.seek(start)
    chunk_data = _worker_file.read(end - start).decode('utf-8', errors='ignore')
    # Normalize newlines for OS-independent tokenization behavior.
    chunk_data = chunk_data.replace('\r\n', '\n').replace('\r', '\n')
    
    # split the chunk data into segments by the special tokens using regex
    pattern = "|".join(re.escape(tok) for tok in _worker_special_tokens)
    if pattern:
        pattern = f"({pattern})"
    segments = re.split(pattern, chunk_data) if pattern else [chunk_data]

    # Tokenize a batch of text segments, and count the tokens in each segment
    token_counts = Counter()
    if regex_re is None:
        raise RuntimeError(
            "The 'regex' package is required for PAT with Unicode properties (\\p{L}, \\p{N}). "
            "Install it with: uv add regex"
        )

    for segment in segments:
        if segment and segment not in _worker_special_tokens:
            # Tokenize the segment and count the tokens
            token_counts.update(regex_re.findall(_worker_pat, segment))
    
    return Counter({s.encode('utf-8'): c for s, c in token_counts.items()})

def bpe_pre_tokenize(file_path: str | os.PathLike, special_tokens: list[str], num_workers: int) -> Counter[bytes, int]:
    assert num_workers > 0, "Number of workers must be positive"
    assert len(special_tokens) > 0, "Must provide at least one special token"
    
    boundaries = find_chunk_boundaries(file_path, num_workers, special_tokens[0].encode('utf-8'))
    chunk_boundaries = list(zip(boundaries[:-1], boundaries[1:]))

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if num_workers == 1:
        # Single-threaded approach
        _init_worker(special_tokens, PAT, file_path)
        return _tokenize_chunk_and_count((0, boundaries[-1]))

    # Manually manage the pool lifecycle. On Windows, BOTH `pool.terminate()`
    # (used by `with Pool(...)` on exit) and `pool.close()+pool.join()` block
    # ~150 seconds inside Pool's internal thread cleanup. Since all useful
    # work is finished by the time we exit the for-loop, we hand cleanup off
    # to a daemon thread so the OS reaps the workers in the background while
    # this function returns immediately.
    pool = Pool(processes=num_workers, initializer=_init_worker, initargs=(special_tokens, PAT, file_path))
    try:
        master_counter = Counter()

        # Process chunks lazily and aggregate results as they complete
        results_iterator = pool.imap_unordered(_tokenize_chunk_and_count, chunk_boundaries)

        print("Starting token counting across all chunks...")
        for i, chunk_counter in enumerate(results_iterator):
            master_counter.update(chunk_counter)
            print(f"Processed chunk {i + 1}/{len(chunk_boundaries)}...")
    except BaseException:
        pool.terminate()
        pool.join()
        raise

    # Fire-and-forget cleanup: workers are idle, daemon thread waits for them.
    def _cleanup(p):
        p.close()
        p.join()
    threading.Thread(target=_cleanup, args=(pool,), daemon=True).start()

    return master_counter

# Convert pre-tokens to list-of-bytes, and keep track of counts
def bpe_pre_token_bytes_seqs_with_counts(pre_tokens: Counter[bytes, int]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], dict[tuple[bytes, ...], int]], list[_PairHeapEntry]]:
    # Use defaultdict so bpe_merge_v2 can do incremental updates without re-wrapping
    bytes_pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    # inverted mapping: from byte pairs to their byte sequences with counts
    # one seq can have duplicate pairs
    bytes_pairs_to_seq: dict[tuple[bytes, bytes], dict[tuple[bytes, ...], int]] = defaultdict(dict)
    
    for byte_list, count in pre_tokens.items():
        byte_seqs = [bytes([x]) for x in byte_list]
        seq_key = tuple(byte_seqs)
        pair_set = set() # avoid counting the same pair multiple times in the same sequence
        for i in range(1, len(byte_seqs)):
            pair_key = (byte_seqs[i - 1], byte_seqs[i])
            bytes_pair_counts[pair_key] += count
            pair_set.add(pair_key)
        
        # seq_key is unique -> (pair, seq_key) is unique -> so we can safely assign the count
        for pair in pair_set:
            bytes_pairs_to_seq[pair][seq_key] = count

    # create a max-heap of the most frequent pairs
    max_pair_heap = [_PairHeapEntry(c, p) for p, c in bytes_pair_counts.items()]
    heapq.heapify(max_pair_heap)

    return bytes_pair_counts, bytes_pairs_to_seq, max_pair_heap

def bpe_find_max_freq(bytes_pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[tuple[bytes, bytes] | None, int | None]:
    if not bytes_pair_counts:
        print("No pairs found.")
        return None, None
    max_count, max_pair = max((c, p) for p, c in bytes_pair_counts.items())
    return max_pair, max_count

def bpe_find_max_freq_from_heap(counts: dict[tuple[bytes, bytes], int], max_pair_heap: list[_PairHeapEntry]) -> tuple[tuple[bytes, bytes] | None, int | None]:
    # Filter out stale entries
    while max_pair_heap:
        max_pair_entry = heapq.heappop(max_pair_heap)
        if max_pair_entry.count == counts.get(max_pair_entry.pair, 0):
            return max_pair_entry.pair, max_pair_entry.count
    return None, None

def bpe_merge(bytes_pairs_to_seq: dict[tuple[bytes, bytes], dict[tuple[bytes, ...], int]], max_pair: tuple[bytes, bytes], counts: dict[tuple[bytes, bytes], int], max_pair_heap: list[_PairHeapEntry]) -> tuple[dict[tuple[bytes, bytes], dict[tuple[bytes, ...], int]], dict[tuple[bytes, bytes], int], list[_PairHeapEntry]]:
    # Merge the max pair in the bytes_seqs, update both bytes_pairs_to_seq and counts
    max_first, max_second = max_pair
    max_pair_seqs = list(bytes_pairs_to_seq[max_pair].items())
    
    for byte_seq_tuple, seq_count in max_pair_seqs:
        seq_len = len(byte_seq_tuple)
        # Tuples support indexing — no need to copy to a list.
        i = 0
        updated_byte_seq = []
        append = updated_byte_seq.append  # local alias avoids attribute lookup per call
        while i < seq_len - 1:
            if byte_seq_tuple[i] == max_first and byte_seq_tuple[i + 1] == max_second:
                append(byte_seq_tuple[i] + byte_seq_tuple[i + 1])
                i += 2
            else:
                append(byte_seq_tuple[i])
                i += 1
        if i < seq_len:
            append(byte_seq_tuple[i])

        updated_seq_len = len(updated_byte_seq)
        updated_byte_seq_tuple = tuple(updated_byte_seq)

        # reducing the count for impacted pairs from the original sequence
        # delete the pairs to seq mapping for the original sequence
        for i in range(1, seq_len):
            pair = (byte_seq_tuple[i - 1], byte_seq_tuple[i])
            counts[pair] -= seq_count
            if counts[pair] <= 0:
                del counts[pair]
            else:
                heapq.heappush(max_pair_heap, _PairHeapEntry(counts[pair], pair))
            if byte_seq_tuple in bytes_pairs_to_seq[pair]:
                del bytes_pairs_to_seq[pair][byte_seq_tuple]

        # add the count for the new sequence
        # add the new pairs to the bytes_pairs_to_seq mapping
        for i in range(1, updated_seq_len):
            pair = (updated_byte_seq[i - 1], updated_byte_seq[i])
            counts[pair] += seq_count
            heapq.heappush(max_pair_heap, _PairHeapEntry(counts[pair], pair))
            bytes_pairs_to_seq[pair][updated_byte_seq_tuple] = seq_count

    # Return counts as-is (defaultdict is a dict subclass) so callers can keep using it incrementally.
    return bytes_pairs_to_seq, counts, max_pair_heap

def bpe_train(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = bpe_vocab_init(vocab_size, special_tokens)
    merges = []
    import time as _t
    _start = _t.perf_counter()
    pre_tokens = bpe_pre_tokenize(input_path, special_tokens, num_workers)
    _t1 = _t.perf_counter()
    print(f"[TIMING] bpe_pre_tokenize: {_t1 - _start:.2f}s")
    bytes_pair_counts, bytes_pairs_to_seq, max_pair_heap = bpe_pre_token_bytes_seqs_with_counts(pre_tokens)
    _t2 = _t.perf_counter()
    print(f"[TIMING] bpe_pre_token_bytes_seqs_with_counts: {_t2 - _t1:.2f}s")
    for i in range(vocab_size - len(vocab)):
        max_pair, max_count = bpe_find_max_freq_from_heap(bytes_pair_counts, max_pair_heap)
        if max_pair is None:
            break
        
        if i % 100 == 0:
            print(f"Merge {i + 1}: {max_pair} (count: {max_count})")
        
        bpe_update_vocab(vocab, max_pair[0] + max_pair[1])
        merges.append(max_pair)
        bpe_merge(bytes_pairs_to_seq, max_pair, bytes_pair_counts, max_pair_heap)
    _t3 = _t.perf_counter()
    print(f"[TIMING] merge loop ({len(merges)} merges): {_t3 - _t2:.2f}s")
    return vocab, merges