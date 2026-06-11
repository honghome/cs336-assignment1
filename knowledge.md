# Unicode and Encoding Summary

## Core Ideas

- Unicode defines characters and assigns each a code point (an integer ID), for example U+0041 for A.
- A code point is not the same thing as bytes on disk or network.
- Encoding is the rule that converts code points to bytes and back.
- Byte-level tokenization uses a fixed 256-symbol base vocabulary (byte values 0 to 255), which avoids out-of-vocabulary issues at the byte level.

## Why Encoding Is Needed (Not Just Storage)

- Systems store and transmit bytes, not abstract code points.
- You need a shared, unambiguous format so different systems decode the same bytes the same way.
- Encoding defines boundaries, validity rules, and error handling for byte streams.
- Different encodings trade off size, compatibility, and simplicity.

## UTF-8, UTF-16, UTF-32 (High-Level)

- UTF-8: 1 to 4 bytes per character, dominant on the web, very efficient for ASCII-heavy text.
- UTF-16: 2 or 4 bytes per character (surrogate pairs for some characters), common in some platform APIs.
- UTF-32: always 4 bytes per character, simplest conceptually, usually largest storage cost.

## Key Numeric Facts

- U+0041 is A, decimal 65.
- U+0073 is s, decimal 115.
- U+10FFFF is decimal 1114111 (max Unicode code point value).
- Unicode 17.0 (Sep 2025) reports 159801 assigned characters across 172 scripts.

## UTF-8 Prefix Rule (How Length Is Known)

- First byte pattern indicates total bytes:
  - 0xxxxxxx: 1 byte
  - 110xxxxx: 2 bytes
  - 1110xxxx: 3 bytes
  - 11110xxx: 4 bytes
- Continuation bytes are always 10xxxxxx.
- This is why decoding is unambiguous even though a short bit pattern could be a prefix of longer raw bits.

## Main Q and A From This Session

1. Q: What does __repr__() mean?
	A: It is the official developer-facing string representation of an object, typically unambiguous and useful for debugging. repr(obj) calls obj.__repr__().

2. Q: How can 4 bytes represent so many Unicode characters?
	A: 4 bytes give 2^32 possible values, far more than the number of assigned Unicode characters; a code point is just an integer ID.

3. Q: How many bytes is U+0073?
	A: A code point itself is an integer; bytes depend on encoding. U+0073 uses 1 byte in UTF-8, 2 in UTF-16, 4 in UTF-32.

4. Q: What is the integer value of 0x10FFFF?
	A: 1114111.

5. Q: Why not use code points directly and skip encoding?
	A: You still need a byte format for files/networks and cross-system agreement. Choosing that byte format is exactly what encoding is.

6. Q: Is encoding only for storage savings?
	A: No. It also ensures interoperability, clear byte boundaries, and valid decoding behavior.

7. Q: What does compact for ASCII-heavy text mean?
	A: In UTF-8, ASCII characters use 1 byte each, so English-like text is often smaller than UTF-16 or UTF-32.

8. Q: Why is U+007F one byte, and where are leading zeros?
	A: U+007F is value 127, binary 01111111, which fits in one byte. Leading zeros are implied by numeric notation, not stored as extra bytes.

9. Q: If 01111111 can be a prefix of longer bits, how is decoding safe?
	A: UTF-8 is parsed byte-by-byte with prefix rules. 0xxxxxxx is a complete single-byte character; continuation bytes must start with 10.

10. Q: Are first bits used to encode byte length?
	 A: Yes. UTF-8 leading bits are metadata for total length; remaining bits carry payload.

11. Q: Examples for each UTF-8 byte length?
	 A:
	 - 1-byte: A -> 41
	 - 2-byte: é -> C3 A9
	 - 3-byte: 你 -> E4 BD A0
	 - 4-byte: 🙂 -> F0 9F 99 82

12. Q: Any reason to use UTF-16 or UTF-32 rather than UTF-8?
	 A: Yes. UTF-16 can be useful for compatibility with platform/runtime APIs and can be smaller for some text mixes; UTF-32 can simplify fixed-width indexing and certain low-level algorithms, but costs more memory/bandwidth.

## Practical Takeaway For Tokenization

- Byte-level tokenization is robust because all text can always be represented as bytes 0 to 255.
- UTF-8 is usually preferred in modern pipelines due to ecosystem compatibility and good size efficiency on typical web text.

## When UTF-16 or UTF-32 Can Make Sense

- Use UTF-16 when existing systems/APIs already operate in UTF-16 and minimizing conversions is important.
- UTF-16 may be size-competitive for some corpora with many non-ASCII characters.
- Use UTF-32 when fixed-width code units simplify implementation (for example, direct code-point indexing).
- UTF-32 is typically the easiest to reason about internally but usually the worst for storage and bandwidth.
- For most NLP and web-scale text pipelines, UTF-8 remains the practical default.

## Python's bytes Type

- `bytes` is an immutable sequence of byte values (integers 0 to 255).
- Type: `type(b"hello")` -> `<class 'bytes'>`.
- Immutable: once created, individual elements cannot be changed.
- Indexing returns `int`: `b"ABC"[0]` -> `65` (not `b"A"`).
- Can be created three ways:
  - Byte literal: `b"hello"`
  - From integers: `bytes([65, 66, 67])`
  - From string encoding: `"hello".encode("utf-8")`
- Can be decoded back to `str`: `b"hello".decode("utf-8")` -> `"hello"`.
- Key relationship: encoding converts `str` -> `bytes`; decoding converts `bytes` -> `str`.

## GPT-Style Pre-Tokenization Regex (PAT)

- Pattern:
	- `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
- Purpose:
	- Split raw text into coarse chunks before BPE merges are learned.
	- Keep common contraction endings separate.
	- Attach a leading space to the following token when possible, which lets the tokenizer learn frequent tokens like ` the` or ` and`.

- Main alternatives, left to right:
	- `'(?:[sdmt]|ll|ve|re)`:
		matches common English contractions such as `'s`, `'d`, `'m`, `'t`, `'ll`, `'ve`, `'re`.
	- ` ?\p{L}+`:
		matches one optional leading space plus one or more Unicode letters.
	- ` ?\p{N}+`:
		matches one optional leading space plus one or more Unicode number characters.
	- ` ?[^\s\p{L}\p{N}]+`:
		matches one optional leading space plus one or more punctuation/symbol characters.
	- `\s+(?!\S)`:
		matches trailing whitespace, especially whitespace right before end of text.
	- `\s+`:
		matches any other remaining whitespace.

---

# BPE Tokenizer (Concepts)

How byte-level BPE is defined and *why* it is defined that way. The performance
sections below are about how to implement it fast; this section is about what the
algorithm actually does and the design choices behind it.

## 1. What BPE Produces

Training a BPE tokenizer produces two artifacts:

- **`vocab: dict[int, bytes]`** — a mapping from integer token ID to the raw bytes
  that token represents. The vocab always starts with:
  - **256 single-byte tokens** for byte values `0x00..0xFF` (ensures *every* byte
    sequence is encodable — no out-of-vocabulary at the byte level).
  - **Special tokens** like `<|endoftext|>` (added explicitly; never produced by
    merges).
  - **Merged tokens** — one new token per merge.
- **`merges: list[tuple[bytes, bytes]]`** — an *ordered* list of byte-pair merges,
  in the order they were learned. The order *is* the priority.

The relationship is exact:

```
len(vocab) = 256 + |special_tokens| + len(merges)
```

So for OpenWebText with `vocab_size=32000` and one special token,
`len(merges) = 32000 - 256 - 1 = 31743`.

## 2. Training (Conceptual)

1. **Pre-tokenize** the corpus into "pre-tokens" (e.g. with the GPT regex). Each
   pre-token is converted into a sequence of single-byte tokens.
2. **Count adjacent byte pairs** across all pre-tokens (weighted by pre-token
   frequency).
3. **Pick the most frequent pair**, append it to `merges`, and add it as a new
   token in `vocab`.
4. **Apply that merge everywhere** — every adjacent occurrence of the pair becomes
   the new token. Update pair counts incrementally.
5. **Repeat** until `len(vocab) == vocab_size`.

The merges list is therefore "most frequent first," roughly. Each merge created
during training reflects a pattern that was statistically common in the corpus.

## 3. Encoding (Inference): Priority-Ordered Merges

Given a new pre-token, BPE encoding is **not** "find the longest matching vocab
token" and **not** "minimize total token count." It is:

> Repeatedly find the **lowest-indexed merge** (highest priority) that applies to
> any adjacent pair in the current symbol sequence, and apply it. Stop when no
> merge applies.

Pseudocode:

```python
def encode_pretoken(pre_token_bytes, merge_rank):
    symbols = [bytes([b]) for b in pre_token_bytes]
    while True:
        best_rank, best_idx = None, -1
        for i in range(len(symbols) - 1):
            r = merge_rank.get((symbols[i], symbols[i+1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_idx = r, i
        if best_rank is None:
            break
        symbols = symbols[:best_idx] + [symbols[best_idx] + symbols[best_idx+1]] + symbols[best_idx+2:]
    return symbols
```

**Worked example.** Vocab includes `b'th'`, `b'the'`; merges = `[(b't',b'h'),
(b' ',b'c'), (b' ',b'a'), (b'th',b'e'), (b' a',b't')]`. Pre-token `'the'` →
`[b't', b'h', b'e']`:

- Scan merges in order: #0 `(b't',b'h')` is present → apply → `[b'th', b'e']`.
- Scan again: #0–#2 not present, #3 `(b'th',b'e')` is present → apply →
  `[b'the']`. Done. → token ID `9`.

## 4. Why Priority-Ordered (and Not "Longest Match")

Several segmentations might give valid token sequences. BPE specifically uses
priority-ordered merging because of one critical invariant:

> **Encoding must reproduce the same segmentation that training would have
> produced on this exact pre-token.**

The model's embeddings were trained on *that* segmentation. If you encode
differently at inference time, you feed the model token IDs it never saw together
during training, and quality drops.

Counter-example showing the rules disagree. Suppose merges =
`[(b'b',b'c'), (b'a',b'b')]`, vocab contains `b'ab'` and `b'bc'`, input is `b'abc'`:

| Strategy | Result | Reasoning |
|---|---|---|
| Longest-match left-to-right | `[b'ab', b'c']` | Greedy from left |
| Shortest-tokenization DP | tied at 2 tokens | Either segmentation |
| **BPE priority-ordered** ✓ | `[b'a', b'bc']` | Merge #0 `(b,c)` fires first |

Same length, different token IDs. Only the BPE-priority output matches training.

## 5. Why Merge at All

The compression benefit *is* real, and it is the practical reason BPE works:

- **Shorter sequences** — attention is O(n²) in sequence length; halving tokens
  per sentence is ~4× compute savings.
- **More text per context window** — a 2048-token window covers more characters.
- **Better learning signal** — the model sees `b'the'` as one unit rather than
  having to learn that `[t][h][e]` always co-occurs.

Minimum token count is a near-side-effect, not the formal objective. Because
merges were learned by frequency, common patterns become single tokens and the
resulting segmentations are usually close to minimal — but the algorithm
optimizes "reproducibility," not "compactness."

## 6. Domain Matters: Tokenizer ↔ Inference Data

Because the tokenizer encodes the *training distribution* of byte patterns,
mismatched domains cause two distinct problems:

1. **Compression drops.** A web-trained tokenizer fed Chinese text or chemistry
   formulas falls back to per-byte tokens (3–4 tokens per character), inflating
   sequence length 2–3×. Measure with **bytes-per-token**.
2. **Unfamiliar token IDs.** Even within "English," a sports article might
   trigger merges the model rarely saw, so its embeddings for those tokens are
   weak.

Common practice:

- **Broad mixed corpus** for the tokenizer (GPT-2: WebText; Llama: + code +
  multilingual).
- **Domain-specific tokenizers** for specialized models (CodeLlama, BioBERT).
- **Tokenizer extension**: add merges/tokens for a new domain, then continue
  pre-training the model with the extended embedding table (common in
  English → Chinese adaptation).

Rule of thumb: **train the tokenizer on data as similar as possible to what the
model will encode**. The expected per-corpus contrast on this assignment:

- **TinyStories tokenizer** — heavy on simple narrative vocabulary
  (`b' little'`, `b' said'`). Great on TinyStories, weak on news/code.
- **OpenWebText tokenizer** — broader (URLs, news terms, code fragments). Better
  general coverage, but "wastes" vocab capacity on TinyStories.

## 7. Special Tokens Are Never Produced by Merges

Special tokens like `<|endoftext|>` are added to `vocab` directly and are never
created by, or considered for, BPE merging. The training pipeline:

- **Splits documents on special-token boundaries** before pre-tokenization, so
  merge counts never cross a `<|endoftext|>`.
- **Tokenizes specials as a single unit** at encoding time, bypassing the
  byte-level merge loop. They must be matched in the input text *before* the
  pre-token regex runs.

This is what the assignment hint refers to: "The `<|endoftext|>` token is
handled as a special case before the BPE merges are applied."

---

# BPE Training: Key Performance Improvements

Lessons learned while optimizing a Python BPE tokenizer from **14.4 s → 0.77 s** on the
small fixture (`corpus.en`, vocab 500) and from **~10 min → target ~30 s** on TinyStories
(2 GB, vocab 10,000). Listed roughly in order of biggest impact for a typical
all-Python BPE implementation.

## 1. Pre-Tokenization: Parallelize with `multiprocessing.Pool`

- Pre-tokenization is **embarrassingly parallel** — split the corpus on a special-token
  boundary (e.g. `<|endoftext|>`), tokenize each chunk independently, merge `Counter`s
  at the end.
- Use `Pool.imap_unordered` for lazy result aggregation (lower memory than `map`).
- On Windows, the `if __name__ == '__main__':` guard is mandatory because workers are
  spawned (not forked).
- Use a `Pool` `initializer` to load the file handle / regex once per worker instead of
  re-importing on every task.
- **Skip the pool entirely for `num_workers == 1`** — pool startup costs ~3 s, which
  dominates on small fixtures.

## 2. Algorithmic: Incremental Pair-Count Updates (don't rebuild every iteration)

Naïve BPE merge rebuilds the entire `pair → count` dictionary from scratch every
iteration. With 9700 iterations on TinyStories, this is the **#1 bottleneck**.

Instead, during each merge: subtract the counts of the **old** pairs that disappeared
from the changed sequences, then add counts for the **new** pairs created by the merge.

| Approach           | TinyStories merge time |
|--------------------|-----------------------:|
| Full rebuild       | ~460 s                 |
| Incremental update | ~80–140 s              |

Implementation hint: use `defaultdict(int)` for the counts so missing keys auto-init to
0 when a merge creates a brand-new pair. Add **zero-cleanup** — if a count drops to ≤ 0,
delete the key, so the dict doesn't grow forever.

## 3. Algorithmic: Inverted Index `pair → {seq: count}`

Even with incremental updates, the merge loop still scans **all** ~700 K unique
sequences each iteration to find the ones touched by the max pair. Most are irrelevant.

Maintain an inverted index alongside the pair counts:

```python
pair_to_count: dict[(bytes, bytes), int]                              # for max selection
pair_to_seqs:  dict[(bytes, bytes), dict[tuple[bytes, ...], int]]     # nested inner dict
```

The inner dict stores `seq → count` so iterating `pair_to_seqs[max_pair].items()` gives
both the affected sequence **and** its count in one lookup — no second dict access.

Then `bpe_merge` only iterates `pair_to_seqs[max_pair]` — typically ~1 % of all
sequences. **Measured: 6× faster merge** on `corpus.en` (0.77 s → 0.13 s).

Invariants to maintain on every change:

- When a pair's count drops to ≤ 0: `del pair_to_count[pair]` (and ideally drop empty
  inner dicts in `pair_to_seqs` too).
- When a new merge creates a new pair: add it to both `pair_to_count` and
  `pair_to_seqs` in lockstep.
- The seq's count is **invariant** across merges — copy the same `seq_count` to the
  new seq entry. Do not `+=` on the inner dict for the same seq.

Watch out for: iterating `pair_to_seqs[max_pair]` while the inner loops mutate it.
Snapshot with `list(...)` first if the merge body removes entries from the same dict
you're iterating.

Memory cost: ~100–200 MB on TinyStories (storing pointers, not copies). Worth it.


## 4. Use `defaultdict` Instead of `dict.get()` + Assignment

```python
# Slow: 2 hash lookups + Python function call
d[key] = d.get(key, 0) + 1

# Fast: 1 hash lookup, all C-level
d[key] += 1            # with d = defaultdict(int)
```

In the BPE profile this dropped 5.6 M `dict.get()` calls (≈ 1.04 s) to 34 K (≈ 6 ms) —
a **170×** reduction at that call site.

## 5. Push Hot Loops Into C: Generator + `max()`

A hand-written "find the max pair" loop in Python runs ~8–12 bytecode dispatches per
iteration. Replacing it with a generator expression lets the comparison happen in C:

```python
# ~2–4× faster than a hand-written loop
max_count, max_pair = max((c, p) for p, c in pair_to_count.items())
```

Bonus: tuple comparison `(count, pair)` is lexicographic, so the BPE tie-breaking rule
("on equal counts, prefer the lexicographically larger pair") is free — no `elif`
branch needed.

Avoid `max(..., key=lambda)` — the lambda becomes a Python call per item and cancels
most of the win.

## 6. Cache `len()` Outside Hot Loops

`len()` is a C function but each call still pays Python-name-resolution + frame setup
overhead. In a tight inner loop running ~10 M times, this adds up to seconds.

```python
# Bad: len called every iteration
while i < len(seq) - 1: ...

# Good: hoist invariant
seq_len = len(seq)
while i < seq_len - 1: ...
```

Profiled impact: 13.5 M `len()` calls (1.28 s) → 10.2 M (0.95 s).

## 7. Don't Convert Tuples to Lists Just to Index

Tuples support `t[i]` indexing and iteration with the same `O(1)` complexity as lists.
The only thing they lack is **mutation**. A common reflex is `list(some_tuple)` —
which is a pure waste if you only read.

```python
# Bad: O(n) allocation + copy per call
seq = list(byte_seq_tuple)
while i < seq_len - 1:
    if seq[i] == ...

# Good: tuples are already indexable
while i < seq_len - 1:
    if byte_seq_tuple[i] == ...
```

At ~700 K sequences × 9700 iterations, that's roughly 7 billion fewer Python-level
copy operations.

## 8. Early-Exit `not in` Check Before Doing Real Work

For pairs of single bytes, `max_first not in byte_seq_tuple` short-circuits in C using
`PyObject_RichCompareBool`. Interned single-byte `bytes` objects compare by identity
first, making this very fast.

```python
if max_first not in byte_seq_tuple or max_second not in byte_seq_tuple:
    # Most sequences don't contain the pair — skip the merge work entirely
    merged[byte_seq_tuple] += seq_count
    continue
```

Once you have an inverted index (Tip #3), this check becomes unnecessary because you
only iterate sequences known to contain the pair.

## 9. Local Aliases for Method Lookups in Tight Loops

```python
# Bad: attribute lookup every iteration
for x in items:
    result.append(x)

# Good: bind once
append = result.append
for x in items:
    append(x)
```

Small per-call savings, but in a 10 M-iteration loop it adds up.

## 10. Pre-allocate Single-Byte `bytes` Objects

`bytes([x])` allocates a new object each call. Build a 256-entry lookup table once:

```python
_BYTE_TABLE = [bytes([i]) for i in range(256)]
byte_seqs = [_BYTE_TABLE[x] for x in byte_list]
```

Bonus: subsequent `==` and `+` operations short-circuit via `is` on the cached objects.

## 11. Max-Heap with Lazy Deletion (Replace Full-Scan `max()`)

After the inverted index (#3) the bottleneck shifts to `bpe_find_max_freq` — every
iteration scans **all** pairs to find the maximum. With ~30 K active pairs over ~10 K
merges that's ~300 M comparisons.

A `heapq` max-heap reduces it to amortized O(log N) per lookup. The trick is that
`heapq` is a min-heap and BPE needs ties broken by lexicographically-larger pair, so
wrap entries in a class with a custom `__lt__`:

```python
class _PairHeapEntry:
    __slots__ = ('count', 'pair')   # ~30% faster attribute access; smaller objects

    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        # Invert: higher count → "smaller" in heap (pops first).
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair   # tie: larger pair pops first
```

### Lazy deletion pattern

Updating an arbitrary entry in a heap is O(N) — impractical. Instead:

- Push a **new** `_PairHeapEntry(new_count, pair)` every time a pair's count changes.
- Keep `pair_to_count` as the authoritative source of truth.
- At pop time, discard entries whose `count` doesn't match `pair_to_count[pair]`.

```python
def find_max_from_heap(counts, heap):
    while heap:
        top = heapq.heappop(heap)
        if top.count == counts.get(top.pair):   # fresh entry?
            return top.pair, top.count
        # stale — keep popping
    return None, None
```

### Critical: don't push when count drops to 0

When a pair disappears (count reaches 0), do **not** push to the heap. The entry would
be stale immediately and waste pop work later:

```python
counts[pair] -= seq_count
if counts[pair] <= 0:
    del counts[pair]
    # no push — pair is gone; future pops will detect via counts.get() == None
else:
    heapq.heappush(heap, _PairHeapEntry(counts[pair], pair))
```

### Build with `heapify`, not N pushes

Initial seeding is O(N) with `heapify`, vs O(N log N) with N successive `heappush`:

```python
heap = [_PairHeapEntry(c, p) for p, c in counts.items()]
heapq.heapify(heap)   # O(N) — uses sift-down on lower half of array
```

Profiled impact on TinyStories: `bpe_find_max_freq` dropped from **~113 s → ~2 s**
(60× speedup at that call site).

## 12. Encode Once Per Unique Token, Not Per Occurrence

A subtle but enormous waste in a tokenization hot loop:

```python
# Bad: encodes the SAME word to bytes millions of times
for m in regex_re.finditer(pat, segment):
    counts[m.group(0).encode('utf-8')] += 1
```

For TinyStories: ~10 K unique tokens but ~500 M total occurrences. That's
**~500 M calls to `.encode()`** producing ~500 M `bytes` objects identical to ones
just created. At ~200 ns per call, that's ~100 s of wasted work.

The fix: count as strings first, encode each **unique** string once at the end.

```python
# Good: ~10 K encode calls instead of ~500 M
str_counts = Counter()
for segment in segments:
    for tok in regex_re.findall(pat, segment):
        str_counts[tok] += 1
return Counter({s.encode('utf-8'): c for s, c in str_counts.items()})
```

This pattern generalizes whenever the value space is much smaller than the call count:
**don't repeatedly transform values you've already transformed**. Cache, batch, or
defer to the end.

## 13. `findall` vs `finditer` in Tight Loops

`finditer` returns `Match` objects; `findall` returns plain strings. When you only
need the matched substring (no positions, no groups), `findall` is **noticeably
faster** in a hot loop:

```python
# finditer path: per iteration
#   1. C: scan to next match
#   2. Python: alloc Match object (~100 bytes, GC-tracked)
#   3. Python: m.group(0) — method dispatch + slice
#   4. Python: Match becomes garbage
for m in regex_re.finditer(pat, segment):
    token = m.group(0)

# findall path: stays in C, builds list of strings directly
for token in regex_re.findall(pat, segment):
    ...
```

Even better, hand the list straight to `Counter.update()`, which uses a C helper
(`_count_elements`) that bumps dict slots without round-tripping through Python:

```python
str_counts.update(regex_re.findall(pat, segment))   # all C-level
```

### When NOT to switch to `findall`

- You need match positions (`.span()`) or groups → `finditer` is required.
- Memory pressure: `findall` materializes the whole list at once. For a 30 MB chunk
  with millions of tokens this is ~500 MB. Fine for batch tokenization, problematic
  for streaming.
- You want to short-circuit (`break` early) → `finditer` is lazy.

Typical speedup for the BPE tokenizer hot loop: **15–25 %** on top of the
encode-once-per-unique-token fix.



- Use `cProfile` for deterministic, function-level breakdowns. On Windows put the
  profiling call inside `if __name__ == '__main__':` so child processes don't recurse.
- For notebook profiling, use `%%prun -s cumulative -l 25`.
- For multiprocessing visibility (`cProfile` doesn't see child processes), use `py-spy
  record --subprocesses` to get a flamegraph that includes workers.
- Sort by **`cumtime`** first (where is time going overall?) then by **`tottime`**
  (which function itself is slow?).
- Look for high `ncalls` on built-ins like `len`, `dict.get` — those are micro-opt
  candidates. Look for high `tottime` on your own functions — those need algorithmic
  fixes.

## What NOT to Do

- **Don't** assume more workers always helps. Pool startup is ~3 s. For small inputs
  single-threaded wins.
- **Don't** rewrite hot loops in Python `for ... else`, list comprehensions over
  generators, etc., unless they replace a measurable bottleneck. Comprehensions are
  faster than `for + append`, but only by ~20 %.
- **Don't** use `Counter.most_common(1)` for BPE max selection — it doesn't apply the
  lexicographic tie-break.
- **Don't** keep a per-iteration `dict(counts)` cast on the return value. `defaultdict`
  is a `dict` subclass; callers can use it transparently and you save an O(N) copy
  every iteration.

## Cheat Sheet: From Slowest to Fastest Implementation

Measured on `tests/fixtures/corpus.en` (vocab 500, single-threaded, cProfile time):

| Stage                                                    | corpus.en time  | `bpe_merge` time |
|----------------------------------------------------------|----------------:|-----------------:|
| Naïve: rebuild pair counts from scratch each iter        | ~14.4 s         | dominant         |
| + `defaultdict`, cache `len`, tuple indexing, `not in`   | ~2.5 s          | 2.10 s           |
| + Incremental pair-count updates (in-place)              | ~1.02 s         | 0.77 s           |
| + Inverted index `pair → {seq: count}`                   | ~0.41 s         | 0.13 s           |
| + Max-heap with lazy deletion (Tip #11)                  | **~0.28 s**     | **0.10 s**       |

After the inverted index the bottleneck shifts from `bpe_merge` to `bpe_find_max_freq`
(~47 % of remaining time). The max-heap (Tip #11) addresses exactly that — on
TinyStories it dropped `bpe_find_max_freq` from ~113 s to ~2 s.

For the multiprocess pre-tokenization stage on TinyStories, the dominant remaining
cost is the per-token `.encode()` and `m.group(0)` overhead inside workers — see
Tips #12 and #13 for ~3× speedup of `bpe_pre_tokenize`.

## TinyStories End-to-End Status (current best on Windows, 4 workers)

`vocab_size=10000`, `<|endoftext|>` special token, 2 GB train file:

| Stage                                | Time     | Notes |
|--------------------------------------|---------:|-------|
| `bpe_pre_tokenize`                   | ~77 s    | After Tips #12/#13 (find_iter→`findall`, encode at end). Was 145 s before. |
| `bpe_pre_token_bytes_seqs_with_counts` | ~0.5 s   | Trivial. |
| Merge loop (9743 merges)             | ~33 s    | Dominated by heap maintenance (`__lt__`, stale `heappop`s). |
| **Total wall clock**                 | **~111 s** | Hint target was < 120 s — done. |

**Hot remaining costs** (from cProfile):

```
9743         bpe_find_max_freq_from_heap   19 s cumtime
2,097,884    heapq.heappop                 17 s cumtime   ← 215 stale pops/merge
9743         bpe_merge                     13 s cumtime
45,652,688   _PairHeapEntry.__lt__          8 s tottime   ← Python-level comparator
```

### Future improvement candidates (not applied — all under 2 min already)

1. **Tuple keys instead of `_PairHeapEntry`.** `_PairHeapEntry.__lt__` is invoked 45 M
   times via `heapq` callbacks; each is a Python method dispatch. Switching to tuple
   keys lets `heapq` use the C-level tuple comparator:
   ```python
   # max-heap key: (-count, INV(pair))
   # heappush(heap, (-count, INV(pair)))
   # On pop, compare with counts.get(pair) for staleness.
   ```
   The trick is encoding the BPE tie-break (larger pair wins). One option: precompute
   `INV(pair) = (~pair[0], ~pair[1])`, but bytes don't support `~`. Cleaner: store
   `(-count, neg_pair_key, pair)` where `neg_pair_key` is a sortable inversion (e.g.
   subtract from a sentinel). Estimate: trims ~6-8 s.

2. **Periodic heap compaction.** Heap grows to ~2 M entries while `counts` holds only
   ~30 K. Every ~500 merges, if `len(heap) > 4 * len(counts)`, rebuild:
   ```python
   heap = [_PairHeapEntry(c, p) for p, c in counts.items()]
   heapq.heapify(heap)
   ```
   Cuts the 2.1 M `heappop` calls (currently popping ~99 % stale) by ~70 %. Estimate:
   trims ~10 s.

3. **More workers + smaller chunks for `bpe_pre_tokenize`.** Currently 4 chunks, 4
   workers, ~500 MB each. With `num_workers=8` and `num_chunks = num_workers * 16`,
   memory pressure drops and load-balances better. Estimate: 77 s → ~40-50 s on a
   4-core / 8-thread machine.

4. **Pre-compile regex in `_init_worker`.** `regex_re.finditer(pat_str, segment)`
   does a cache lookup per segment. Compiling `_worker_token_re = regex_re.compile(pat)`
   in `_init_worker` and using `_worker_token_re.findall(segment)` saves the lookup.
   Estimate: 5-10 % of pre-tokenize time.

5. **Skip newline normalization on Linux-formatted files.** `chunk_data.replace(...)`
   does two full-string passes (~60 MB each on a 30 MB chunk × 4 workers). Guard with
   `if '\r' in chunk_data:` to skip when the file has only `\n`. Estimate: ~5 s on
   TinyStories (which is `\n`-only).

6. **`bpe_merge` micro-opts.** 13 s for 9743 merges = ~1.3 ms/call. The append loop
   that scans `byte_seq_tuple` for matches is Python-level. Could be faster with
   `bytes.find` on a packed representation, but readability suffers.

Apply 1+2 together first if pushing for sub-60 s. Apply 3+5 if reducing memory
footprint. Apply 4 only if it shows up in profile.

## Pool Cleanup on Windows (Side Quest)

`with multiprocessing.Pool(...)` calls `terminate()` on exit, which on Windows blocks
~150 s in `SemLock.acquire` waiting for child-process state cleanup. `close() + join()`
also blocks (just on the helper threads instead). Workaround: hand cleanup off to a
**daemon thread** so the main thread returns immediately:

```python
pool = Pool(...)
try:
    for r in pool.imap_unordered(work, items):
        ...
except BaseException:
    pool.terminate(); pool.join(); raise

def _cleanup(p): p.close(); p.join()
threading.Thread(target=_cleanup, args=(pool,), daemon=True).start()
return result
```

Daemon threads are killed when the interpreter exits, so leaked workers eventually
die. Acceptable for batch jobs (call `bpe_train` 1-3 times). Not acceptable for
long-running services.

This shaved ~150 s of misleading "function call" time off our profile, even though
the actual work was ~30 s the whole time — profile was attributing pool teardown to
`bpe_pre_tokenize` because that's the call site.




- Important regex pieces:
	- `\p{L}` means any Unicode letter.
	- `\p{N}` means any Unicode number.
	- `[^...]` means any character not in that set.
	- `(?:...)` is a non-capturing group.
	- `(?!...)` is a negative lookahead.

- Example idea:
	- `hello world` is often split more like `hello` and ` world` than `hello`, ` `, `world`.
	- `can't` is often split as `can` and `'t`.

- Python detail:
	- `\p{L}` and `\p{N}` are not supported by the built-in `re` module.
	- This pattern usually requires the third-party `regex` package instead of `re`.

## Byte-Level BPE and UTF-8 Validity

- `text.encode("utf-8")` returns bytes encoded with UTF-8, but a Python `bytes` object does not carry a built-in encoding label.
- `raw = text.encode("utf-8")` gives the byte stream; interpretation depends on how you later decode it (for example, UTF-8).
- Splitting bytes as `[bytes([x]) for x in raw]` creates one-byte tokens.
- One-byte tokens are not guaranteed to be valid standalone UTF-8 characters.
	- ASCII bytes can be valid single-byte UTF-8 characters.
	- Many non-ASCII characters require multiple bytes, so individual bytes are only partial pieces.
- Byte-level BPE learns merges over adjacent byte pieces and can create tokens that are not valid UTF-8 by themselves.
- This is expected behavior in byte-level tokenization.
- Correctness rule:
	- Individual tokens do not need to decode as UTF-8.
	- The concatenation of all token bytes must reproduce the original byte stream.
	- Decode after concatenation, not per token.

