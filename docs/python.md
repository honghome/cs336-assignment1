# Python Performance Optimization Notes

Lessons learned from profiling and optimizing the BPE tokenizer implementation.

## Table of Contents
- [Why `defaultdict` is faster than `dict.get()`](#why-defaultdict-is-faster-than-dictget)
- [Why caching `len()` saves time](#why-caching-len-saves-time)
- [Using `in` / `not in` for early-exit optimization](#using-in--not-in-for-early-exit-optimization)
- [Tuples are indexable — don't copy them to lists](#tuples-are-indexable--dont-copy-them-to-lists)
- [Using `max()` with a generator to push the loop into C](#using-max-with-a-generator-to-push-the-loop-into-c)
- [The bigger lesson: Python function calls are expensive](#the-bigger-lesson-python-function-calls-are-expensive)
- [Rules of thumb for hot loops](#rules-of-thumb-for-hot-loops)
- [Profiling Methodology](#profiling-methodology)
- [PyTorch: Advanced Indexing on Tensors](#pytorch-advanced-indexing-on-tensors)
- [PyTorch: Lazy Batch Reads from `np.memmap` and `int32` -> `long`](#pytorch-lazy-batch-reads-from-npmemmap-and-int32---long)
- [PyTorch: Broadcasting and the `[:, None]` / `[None, :]` Outer-Product Idiom](#pytorch-broadcasting-and-the---none----none--outer-product-idiom)
- [PyTorch: Element-wise vs Reduction Ops — What `dim` Actually Controls](#pytorch-element-wise-vs-reduction-ops--what-dim-actually-controls)

---

## Why `defaultdict` is faster than `dict.get()`

### Before (with `dict.get()`)

```python
updated_counts[pair_key] = updated_counts.get(pair_key, 0) + seq_count
```

This does **3 operations** per call:
1. `dict.get(pair_key, 0)` — Python function call + hash lookup
2. `+ seq_count` — addition
3. `updated_counts[pair_key] = ...` — Python `__setitem__` call + hash lookup **again**

### After (with `defaultdict`)

```python
from collections import defaultdict

updated_counts = defaultdict(int)
updated_counts[pair_key] += seq_count
```

This does **1 conceptual operation**:
1. `__getitem__` triggers `__missing__` (calls `int()` → 0) if key absent, then `+=` updates in-place.
2. Hashing happens **once**, not twice.

### Measured impact in BPE training

```
Before: 5,653,423 dict.get() calls = 1.043s
After:    34,752 dict.get() calls = 0.006s   (170x reduction!)
```

**Why it matters:** Every `.get()` is a Python-level method call (vs C-level `__getitem__`). Plus the same key gets hashed twice (once for get, once for set).

---

## Why caching `len()` saves time

### Before

```python
if len(byte_seq) > 1:
    while i < len(byte_seq) - 1:   # called every iteration!
        ...
    if i == len(byte_seq) - 1:
        ...
```

Every iteration of the `while` loop calls `len(byte_seq)`. For a sequence of length 10, that's ~10 `len()` calls per sequence merge.

### After

```python
seq_len = len(byte_seq)   # called ONCE
if seq_len > 1:
    while i < seq_len - 1:   # use cached value
        ...
    if i == seq_len - 1:
        ...
```

### What `len()` actually does

Even though `len()` is a built-in C function, every call:
1. Looks up `len` in builtins (Python name resolution)
2. Calls `__len__` on the object
3. Returns the result

For tuples, `__len__` is fast (just reading a field), but the **function call overhead** dominates when you do it millions of times.

### Measured impact

```
Before: 13,521,423 len() calls = 1.28s
After:  10,171,461 len() calls = 0.95s   (saved ~330ms)
```

---

## Tuples are indexable — don't copy them to lists

A common reflex when you need positional access is to convert a tuple to a list:

```python
byte_seq = list(byte_seq_tuple)   # ❌ unnecessary O(n) copy
while i < seq_len - 1:
    if byte_seq[i] == max_first and byte_seq[i + 1] == max_second:
        ...
```

But **tuples already support indexing and iteration** with the same `O(1)` access as lists. The only differences are:

| Operation | Tuple | List |
|-----------|-------|------|
| `t[i]` | ✅ O(1) | ✅ O(1) |
| `for x in t` | ✅ | ✅ |
| `len(t)` | ✅ O(1) | ✅ O(1) |
| `x in t` | ✅ O(n) | ✅ O(n) |
| `t.append(x)` | ❌ immutable | ✅ |
| `t[i] = x` | ❌ immutable | ✅ |

So if you're only **reading**, the `list(...)` call is pure waste. For a sequence of length 10, it's:
- 1 list allocation
- 10 element copies (with refcount bumps)
- 1 list header allocation

### After

```python
# byte_seq_tuple is already indexable
while i < seq_len - 1:
    if byte_seq_tuple[i] == max_first and byte_seq_tuple[i + 1] == max_second:
        ...
```

### When this matters

A single avoided copy is tiny. But in BPE training:
- ~700K unique sequences
- 9743 merge iterations
- = ~7 billion fewer Python-level copy operations

**Rule:** Only copy a tuple to a list when you actually need to mutate it.

---

## Using `max()` with a generator to push the loop into C

A classic Python optimization: replace a hand-written "find the max" loop with `max()` over a generator. The win isn't from a magical algorithm — it's from **moving the per-item work from the Python bytecode interpreter into CPython's C eval loop**.

### The hand-written version (slow)

```python
def find_max(counts):
    max_pair, max_count = None, None
    for pair, count in counts.items():
        if max_count is None or count > max_count:
            max_count = count
            max_pair = pair
        elif max_count == count and max_pair < pair:
            max_pair = pair
    return max_pair, max_count
```

Per iteration, the CPython interpreter dispatches **~8–12 bytecode ops**:
`FOR_ITER`, `UNPACK_SEQUENCE`, `LOAD_FAST`, `IS_OP`, `COMPARE_OP`, `POP_JUMP_IF_*`, `STORE_FAST`...

Each op goes through the giant `switch` in CPython's `ceval.c`. That dispatch overhead dominates.

### Tempting but only modest gain: `max(key=lambda)`

```python
max_pair = max(counts, key=lambda p: (counts[p], p))
```

- The C `max()` loop iterates in C ✅
- But `key()` is **called as a Python function once per item** ❌
- A Python frame entry+exit costs roughly the same as the bytecode ops you saved

Net result: **~1.3–1.7× speedup**, not 5×.

### The actually-fast version: generator + tuple comparison

```python
def find_max(counts):
    if not counts:
        return None, None
    max_count, max_pair = max((c, p) for p, c in counts.items())
    return max_pair, max_count
```

Why this wins:

1. **No `key=` callback** — `max()` compares the yielded `(count, pair)` tuples directly via `PyObject_RichCompareBool` in C.
2. **Tuple comparison handles tie-breaking for free** — Python compares tuples lexicographically. `(count, pair)` means "larger count wins; on ties, larger pair wins" — exactly the BPE rule.
3. **The branch + assignment-of-current-best happens in C**, not Python bytecode.
4. The only Python-level work per item is the generator's `for p, c in ...` unpack.

Expected speedup: **2–4×** over the hand-written loop on a dict with ~100K items.

### Summary comparison

| Version | Per-item Python overhead | Relative speed |
|---------|--------------------------|----------------|
| Manual `for` loop | 8–12 bytecode ops | 1× (baseline) |
| `max(…, key=lambda)` | 1 Python call + few bytecodes | ~1.3–1.7× |
| `max((c, p) for …)` | 1 generator tick (unpack) | **~2–4×** |
| Incremental heap | `O(log n)` instead of `O(n)` | ~10–50× asymptotic |

### Why the generator wins over the lambda

The key insight: **Python function calls are expensive (~50–100 ns)**. The `key=lambda kv: ...` trick saves bytecode dispatch but pays for it with a function call. The generator expression avoids both — it produces values via the optimized iterator protocol (`tp_iternext`), and `max()` compares them with no Python callback.

### When to reach for this pattern

- You have a `for` loop that just tracks one running value (max/min/sum/count).
- The body has no side effects.
- The dataset is large enough that per-iteration overhead matters (≳ 10K items in a hot path).

### When NOT to bother

- The loop runs once or a few times.
- The body does substantial work that dominates per-iteration overhead.
- The asymptotic answer is a different data structure (heap, sorted container) — those win at scale, generators don't.

---

## The bigger lesson: Python function calls are expensive

In Python:
- Every function call costs **~50-100 nanoseconds** (vs ~1ns for C)
- Even "free" things like `len()`, `.get()`, attribute access add up at scale
- **At 10+ million calls, microseconds become seconds**

### A concrete example

```python
# Slow (3 operations × 5.6M = 16.8M operations)
counts[key] = counts.get(key, 0) + 1

# Fast (1 operation × 5.6M = 5.6M operations)
counts[key] += 1  # with defaultdict
```

That's a **3x reduction in Python-level operations**, which translates directly to time savings.

**Bottom line:** When you do something 10 million times, the difference between a 100ns operation and a 50ns operation is **0.5 seconds**. Micro-optimizations matter only in hot loops, but they matter A LOT there.

---

## Rules of thumb for hot loops

### 1. Hoist invariants out of loops

```python
# Bad: len() called every iteration
for i in range(len(items)):
    process(items[i], len(items))

# Good: len() called once
n = len(items)
for i in range(n):
    process(items[i], n)
```

### 2. Use specialized data structures

| Use Case | Slow | Fast |
|----------|------|------|
| Counting | `d[k] = d.get(k, 0) + 1` | `Counter()` or `defaultdict(int)` |
| Grouping | `d.setdefault(k, []).append(v)` | `defaultdict(list)` |
| Membership | `if x in list` | `if x in set` |
| FIFO queue | `list.pop(0)` (O(n)) | `collections.deque` (O(1)) |

### 3. Bind method lookups outside loops

```python
# Bad: list.append looked up every iteration
result = []
for item in items:
    result.append(transform(item))

# Good: bind once (saves attribute lookup)
result = []
append = result.append
for item in items:
    append(transform(item))
```

### 4. Minimize object creation in tight loops

```python
# Bad: creates a new tuple every iteration
for i in range(n - 1):
    counts[(seq[i], seq[i+1])] += 1

# Better: use local variables to avoid repeated indexing
prev = seq[0]
for curr in seq[1:]:
    counts[(prev, curr)] += 1
    prev = curr
```

### 5. Prefer comprehensions over explicit loops

```python
# Slower
result = []
for x in items:
    if x > 0:
        result.append(x * 2)

# Faster (no .append lookup, optimized bytecode)
result = [x * 2 for x in items if x > 0]
```

### 6. Use local variable aliases for global/builtin functions

```python
# Bad: 'len' resolved in builtins every call
for item in items:
    if len(item) > 5: ...

# Good: local lookup is faster than global/builtin
_len = len
for item in items:
    if _len(item) > 5: ...
```

---

## Profiling Methodology

### cProfile (built-in, deterministic)

```python
import cProfile, pstats

def main():
    your_function()

if __name__ == '__main__':  # required on Windows for multiprocessing!
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("profile.prof")

    p = pstats.Stats("profile.prof")
    p.sort_stats("cumulative").print_stats(25)  # top 25 by cumulative time
    p.sort_stats("time").print_stats(20)        # top 20 by self time
```

### Reading the output

| Column | Meaning |
|--------|---------|
| `ncalls` | Number of times the function was called |
| `tottime` | Time spent **inside** the function (excl. sub-calls) |
| `percall` | `tottime / ncalls` |
| `cumtime` | Time spent in this function **and** all sub-calls |
| `percall` | `cumtime / ncalls` |

### What to look for

1. **High `ncalls`** → candidate for caching/hoisting
2. **High `tottime`** → the function itself is slow (optimize internals)
3. **High `cumtime` but low `tottime`** → look at what it calls
4. **Built-ins like `len`, `dict.get` near the top** → micro-optimize hot paths

### py-spy (sampling profiler, works without code changes)

```bash
pip install py-spy
py-spy record -o profile.svg -- python your_script.py
# Open profile.svg in browser to see flamegraph
```

Better for:
- Multiprocessing code (sees child processes)
- Long-running processes you can't modify
- Visualizing where time is spent (flamegraph)

---

## Quick reference: Common Python slow patterns

| Pattern | Why it's slow | Fix |
|---------|---------------|-----|
| `d.get(k, 0) + v; d[k] = ...` | Double hash lookup | `defaultdict(int)` + `+=` |
| `len(x)` in loop condition | Function call per iteration | Cache before loop |
| `list.pop(0)` | O(n) shift | `collections.deque.popleft()` |
| `x in some_list` (large list) | O(n) scan | Convert to `set` |
| String concat in loop | O(n²) | `''.join(parts)` |
| Repeated attribute access | Lookup overhead | Bind to local: `f = obj.method` |
| Creating tuples for keys | Object allocation | Use simpler key (int, str) if possible |

---

## PyTorch: Advanced Indexing on Tensors

Indexing a tensor with another **integer tensor** does a **gather** — and the
output shape follows a precise, broadcastable rule. This is the foundation of
embedding lookups, batched scatter/gather operations, and most "select these
rows" patterns.

### The shape rule

Given a tensor `T` of shape `(D0, D1, D2, ...)` and an integer index tensor
`idx` of shape `S`:

```
T[idx].shape == idx.shape + T.shape[1:]
```

In words: the **leading axis of `T` is "consumed"** by the indexing and
**replaced by `idx`'s shape**. The remaining axes of `T` are appended
unchanged at the end.

### Concrete examples (embedding-table style)

```python
import torch
V, d = 100, 8
E = torch.randn(V, d)               # (vocab=100, d_model=8)

E[torch.tensor(5)].shape            # (8,)            — scalar id → one vector
E[torch.tensor([1, 2, 3])].shape    # (3, 8)          — 1-D id batch
E[torch.randint(0, V, (4, 7))].shape       # (4, 7, 8)       — 2-D batch
E[torch.randint(0, V, (2, 3, 5))].shape    # (2, 3, 5, 8)    — any rank works
```

The trailing `d_model=8` is preserved every time because indexing only
consumes axis 0 of `E`.

### Why it "just works" for batch dimensions

`torch.Tensor.__getitem__` with a `LongTensor` is implemented as a fused
gather kernel:

- For each scalar `id` at position `(i, j, ...)` of `idx`, copy `T[id]` (a
  `(D1, D2, ...)`-shaped slice) into output position `(i, j, ..., :, :, ...)`.
- No reshape, no loop, no `.view()` needed in your Python code.
- The same kernel powers `torch.nn.functional.embedding`.

This is why an embedding layer's `forward` can be a single line:

```python
def forward(self, token_ids):
    return self.weights[token_ids]   # works for any rank of token_ids
```

### Equivalent ways to express the same gather

```python
E[token_ids]                                          # idiomatic
torch.nn.functional.embedding(token_ids, E)           # explicit API
E.index_select(0, token_ids.view(-1)).view(*token_ids.shape, d)  # manual
```

All three dispatch to the same kernel and produce identical results.
Prefer the first.

### Dtype gotchas

| Index dtype | What happens |
|---|---|
| `torch.long` (int64) | ✓ gather (recommended, default for `torch.tensor([1,2,3])`) |
| `torch.int` / `torch.int32` | ✓ gather (works, but `.long()` is safer for portability) |
| `torch.float*` | ✗ `IndexError: tensors used as indices must be long, int, byte or bool` |
| `torch.bool` | ⚠️ **boolean masking**, not gather — completely different semantics |

```python
E[torch.tensor([1, 2, 3])]                  # ✓ gather → (3, 8)
E[torch.tensor([1.0, 2.0])]                 # ✗ IndexError
E[torch.tensor([True, False, True, ...])]   # ⚠ mask → selects matching rows
```

If your IDs come from NumPy (`np.int32` by default), cast before indexing:

```python
ids = torch.from_numpy(np_ids).long()       # safe
```

### How gradients flow through indexing

`__getitem__` is differentiable. If `E.requires_grad=True`:

- Only rows actually looked up receive gradient.
- A row looked up multiple times in one batch gets the gradients **summed**.
- Rows never looked up have gradient zero — which is why embedding tables get
  **sparse updates** (only ~`B·T` unique IDs per step out of `V` total).

This sparseness is why `torch.optim.SparseAdam` and `nn.Embedding(sparse=True)`
exist — for very large vocabularies, sparse gradient updates can be a big win.

### Quick reference

| Want | Code |
|---|---|
| Look up rows by IDs | `E[ids]` — shape `ids.shape + E.shape[1:]` |
| Look up with bounds check | `torch.nn.functional.embedding(ids, E)` (same speed) |
| Cast NumPy → index-safe tensor | `torch.from_numpy(arr).long()` |
| Force `int64` from a Python list | `torch.tensor([1, 2, 3])` (default is `int64`) |
| Boolean **mask** (not gather) | `E[mask]` where `mask.dtype == torch.bool` |

---

## PyTorch: Lazy Batch Reads from `np.memmap` and `int32` -> `long`

Large token datasets should be stored on disk and read lazily. A common pattern
is:

```python
dataset = np.memmap("training/TinyStoriesV2-GPT4-train_tokens.bin", dtype=np.int32, mode="r")
```

This does **not** load the whole token file into RAM. It creates a memory-mapped
view of the file. The operating system only reads the pages touched by your
code.

### The trap: converting the whole memmap to Torch

This looks convenient, but is bad for huge datasets:

```python
data = torch.as_tensor(dataset, dtype=torch.long)
```

If `dataset` has 540M tokens, this tries to convert all 540M tokens into a
Torch tensor before one training step can run. It defeats the purpose of the
memmap and can be slow or memory-heavy. It may also warn that the NumPy array is
not writable, because read-only memmaps produce non-writable arrays.

### The better pattern: sample small slices first

Read only one mini-batch worth of token IDs, then convert that small batch to
Torch:

```python
def get_batch(dataset, batch_size: int, context_length: int, device: str):
    max_start = len(dataset) - context_length
    starts = torch.randint(0, max_start, (batch_size,)).numpy()

    x_np = np.stack([dataset[start : start + context_length] for start in starts])
    y_np = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])

    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    return x, y
```

For `batch_size=32` and `context_length=256`, this reads about:

```text
32 * (256 + 1) = 8,224 token IDs
```

instead of converting hundreds of millions of token IDs. Each training
iteration samples one mini-batch, trains on it, discards those tensors, then the
next iteration samples another mini-batch.

### Why store as `int32` but train with `torch.long`?

Token IDs only need enough range to represent the vocabulary. If
`vocab_size=10000`, `np.int32` is plenty for disk storage:

```text
int32: 4 bytes per token
int64: 8 bytes per token
```

For 540,796,778 tokens, that is roughly:

```text
int32 storage: 2.16 GB
int64 storage: 4.33 GB
```

So the best split is:

```python
# Disk / memmap: compact storage.
dataset = np.memmap(path, dtype=np.int32, mode="r")

# Current batch: PyTorch-friendly index dtype.
x = torch.as_tensor(x_np, dtype=torch.long, device=device)
y = torch.as_tensor(y_np, dtype=torch.long, device=device)
```

`torch.long` is PyTorch's standard index dtype for embedding lookup and
cross-entropy targets. Keeping the file as `int32` saves disk and memory; casting
only the current batch to `long` gives PyTorch the dtype it expects without
materializing the full dataset.

### Mental model

```text
Training iteration 1: read one batch from memmap -> convert batch to long -> train
Training iteration 2: read another batch       -> convert batch to long -> train
Training iteration 3: read another batch       -> convert batch to long -> train
```

The full dataset stays on disk. Only the sampled slices are converted into Torch
tensors.

---

## PyTorch: Broadcasting and the `[:, None]` / `[None, :]` Outer-Product Idiom

A single line like

```python
angles = positions[:, None] * freqs[None, :]   # shape: (max_seq_len, d_k/2)
```

is doing a **broadcasting outer product** — building a full 2-D table from
two 1-D vectors in one vectorized op, no Python loop. This pattern shows
up constantly (distance matrices, positional encodings, attention masks,
pairwise scoring).

### Inputs

```python
positions = torch.arange(max_seq_len)   # shape: (max_seq_len,)   e.g. (1024,)
freqs     = 1.0 / (theta ** (2*k/d_k))  # shape: (d_k/2,)         e.g. (32,)
```

We want a 2-D table where entry `(i, k) = positions[i] * freqs[k]`.

### What `[:, None]` and `[None, :]` do — insert a length-1 axis

`None` (equivalent to `np.newaxis`) **inserts a length-1 dimension** at
that position. No data is copied — it's just a new view of the same
storage with a `1` stuck into the shape.

```python
positions             # shape: (1024,)        vector
positions[:, None]    # shape: (1024, 1)      column vector
positions[None, :]    # shape: (1, 1024)      row vector

freqs                 # shape: (32,)          vector
freqs[None, :]        # shape: (1, 32)        row vector
freqs[:, None]        # shape: (32, 1)        column vector
```

### What the multiplication does — broadcasting

PyTorch's broadcasting rule: **dimensions of size 1 are virtually
stretched to match the other operand.**

```
positions[:, None]:    (1024,  1)
freqs[None, :]:        (   1, 32)
                       ──────────
result shape:          (1024, 32)
```

The `1`s expand virtually:

- `positions[:, None]` is conceptually copied **across columns** → every column gets all 1024 positions.
- `freqs[None, :]` is conceptually copied **down rows** → every row gets all 32 frequencies.

Then element-wise multiply gives:

```
angles[i, k] = positions[i] * freqs[k] = i * freqs[k]
```

### Concrete tiny example

`max_seq_len = 4`, two pairs (`d_k/2 = 2`):

```python
positions = tensor([0, 1, 2, 3])     # shape (4,)
freqs     = tensor([1.0, 0.1])       # shape (2,)
```

After reshaping:

```
positions[:, None] =          freqs[None, :] =
  [[0],                         [[1.0, 0.1]]   ← shape (1, 2)
   [1],
   [2],
   [3]]
  ← shape (4, 1)
```

Broadcasting and multiplying:

```
            freqs →  1.0    0.1
positions ↓     ┌──────────────┐
   i=0          │  0*1.0  0*0.1 │   =  [[ 0.0,  0.0],
   i=1          │  1*1.0  1*0.1 │       [ 1.0,  0.1],
   i=2          │  2*1.0  2*0.1 │       [ 2.0,  0.2],
   i=3          │  3*1.0  3*0.1 │       [ 3.0,  0.3]]
                └──────────────┘
                                       shape: (4, 2)
```

### Without broadcasting (what we're avoiding)

```python
angles = torch.empty(max_seq_len, d_k // 2)
for i in range(max_seq_len):
    for k in range(d_k // 2):
        angles[i, k] = positions[i] * freqs[k]   # slow Python loop
```

The broadcasting line does the same thing in **one vectorized GPU/CPU
kernel**.

### Equivalent ways to express the same outer product

```python
positions[:, None] * freqs[None, :]            # idiomatic broadcasting
torch.outer(positions, freqs)                  # explicit outer-product API
torch.einsum("i,k->ik", positions, freqs)      # einsum form
positions.unsqueeze(1) * freqs.unsqueeze(0)    # .unsqueeze() equivalent of None
```

All produce identical `(max_seq_len, d_k/2)` tensors. The first is the
most common; `torch.outer` is clearest when the intent is literally an
outer product; einsum scales naturally to more dimensions.

### Broadcasting rule, in general

When NumPy/PyTorch combine two arrays:

1. **Align shapes from the right.** Shorter shape is padded with `1`s on the left.
2. **For each dimension**, the two sizes must be **equal**, or one must be `1`.
3. **Dimensions of size 1 are stretched** to match the other operand.

```
  A.shape: (    4, 1)
  B.shape: (1, 1, 3)
  result:  (1, 4, 3)
```

No data copy happens; the stretch is virtual.

### The general pattern

`a[:, None] * b[None, :]` is the **outer product** of two 1-D vectors →
a 2-D matrix where entry `(i, j) = a[i] * b[j]`. Uses everywhere:

| Use case | Code |
|---|---|
| Pairwise distance matrix | `(x[:, None] - x[None, :]).pow(2).sum(-1)` |
| Positional encoding angles | `positions[:, None] * freqs[None, :]` |
| Causal mask | `torch.arange(T)[:, None] >= torch.arange(T)[None, :]` |
| Cartesian product of grids | `a[:, None] * b[None, :]` |
| Any "for every pair `(i, j)`, compute `f(a[i], b[j])`" | `f(a[:, None], b[None, :])` |

### TL;DR

```python
angles = positions[:, None] * freqs[None, :]
```

= **column vector of positions** × **row vector of frequencies**, via
broadcasting, giving a `(max_seq_len, d_k/2)` matrix where row `i`,
column `k` is `positions[i] * freqs[k]`. One vectorized op replaces a
double Python loop.

---

## PyTorch: Element-wise vs Reduction Ops — What `dim` Actually Controls

A common confusion when writing tensor code (especially things like
`softmax`): **which operations care about `dim` and which don't?**

The rule is simple:

- **Element-wise ops** (`.exp()`, `.sin()`, `.abs()`, `+`, `*`, `-`, `/`, ...)
  apply $f(x)$ to **every single number** in the tensor, independently.
  Shape unchanged. They have **no `dim` parameter** because there's
  nothing to choose — every cell is treated the same way.
- **Reduction ops** (`.sum`, `.mean`, `.max`, `.amax`, `.min`, `.amin`,
  `.var`, `.std`, `.norm`, `.argmax`, ...) collapse one (or more) axes
  into a single value per slice. **They take a `dim` argument** that
  picks which axis to collapse.

Mixing these up is what causes the classic "why doesn't softmax along
the wrong axis work" mystery.

### Element-wise: shape in = shape out

```python
x = torch.tensor([
    [ 0.0, -1.0, -2.0],
    [-3.0,  0.0, -1.0],
])
# shape: (2, 3)

x.exp()
# = tensor([
#     [exp(0.0),  exp(-1.0), exp(-2.0)],   ← every entry transformed independently
#     [exp(-3.0), exp( 0.0), exp(-1.0)],
# ])
# shape: (2, 3)   ← unchanged
```

Same story for `.sin()`, `.cos()`, `.sqrt()`, `.log()`, unary `-`, and
any arithmetic op between tensors of compatible shape.

### Reduction: one axis collapses (or vanishes with `keepdim=False`)

```python
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
# shape: (2, 3)

x.sum(dim=1)                    # → tensor([ 6., 15.])           shape (2,)
x.sum(dim=1, keepdim=True)      # → tensor([[ 6.], [15.]])       shape (2, 1)
x.sum(dim=0)                    # → tensor([5., 7., 9.])         shape (3,)
x.sum(dim=0, keepdim=True)      # → tensor([[5., 7., 9.]])       shape (1, 3)
```

`keepdim=True` is the magic that keeps the result broadcastable back
against the original tensor — essential for normalization patterns like
softmax and RMSNorm.

### Anatomy of softmax — a clean illustration

```python
def softmax(x, dim):
    x_max   = x.amax(dim=dim, keepdim=True)      # ← reduction along dim
    x_shift = x - x_max                          # element-wise (broadcasts)
    exp_x   = x_shift.exp()                      # element-wise (no dim)
    s       = exp_x.sum(dim=dim, keepdim=True)   # ← reduction along dim
    return exp_x / s                             # element-wise (broadcasts)
```

| Op | Cares about `dim`? | Why |
|---|---|---|
| `.amax(dim=...)` | ✅ yes | Reduces that axis to one value per slice |
| `-` (subtraction) | ❌ no | Element-wise (with broadcasting) |
| `.exp()` | ❌ no | Element-wise; applies $e^x$ to every cell |
| `.sum(dim=...)` | ✅ yes | Reduces that axis |
| `/` (division) | ❌ no | Element-wise (with broadcasting) |

So softmax's "axis-awareness" comes **entirely** from `amax` and `sum`.
The `exp` in the middle just sits there, doing the same thing to every
number. If you swap `dim=-1` for `dim=0`, only the reduction ops behave
differently — `.exp()` is identical.

### `.max()` vs `.amax()` — a footgun

Both reduce along `dim`, but they return different things:

```python
x.max(dim=1)        # → torch.return_types.max(values=..., indices=...)
x.max(dim=1).values # → just the values (a Tensor)
x.amax(dim=1)       # → just the values, directly (a Tensor)

x.min(dim=1)        # same: NamedTuple
x.amin(dim=1)       # values only

x.argmax(dim=1)     # indices only
```

Writing `x_max = x.max(dim=dim, keepdim=True)` then doing
`x - x_max` will fail because `x_max` is a NamedTuple, not a Tensor.
Use `.amax(...)` or `.max(...).values`.

### Common reduction ops cheat-sheet

| Op | What it computes | Returns |
|---|---|---|
| `.sum(dim=...)` | sum along axis | Tensor |
| `.mean(dim=...)` | mean along axis | Tensor |
| `.amax(dim=...)` | max along axis | Tensor (values only) |
| `.amin(dim=...)` | min along axis | Tensor (values only) |
| `.max(dim=...)` | max + argmax | NamedTuple `(values, indices)` |
| `.min(dim=...)` | min + argmin | NamedTuple `(values, indices)` |
| `.argmax(dim=...)` | index of max | LongTensor |
| `.argmin(dim=...)` | index of min | LongTensor |
| `.var(dim=...)` | variance | Tensor |
| `.std(dim=...)` | std-dev | Tensor |
| `.norm(dim=...)` | L2 norm | Tensor |
| `.prod(dim=...)` | product along axis | Tensor |
| `.any(dim=...)` / `.all(dim=...)` | boolean reduction | BoolTensor |
| `.logsumexp(dim=...)` | numerically stable $\log\sum e^x$ | Tensor |

### Common element-wise ops cheat-sheet

None of these take a `dim` argument; they all preserve shape.

| Category | Examples |
|---|---|
| Unary math | `.exp`, `.log`, `.sqrt`, `.rsqrt`, `.abs`, `.neg`, `.sign`, `.reciprocal` |
| Trig | `.sin`, `.cos`, `.tan`, `.atan2` |
| Activations | `torch.sigmoid`, `torch.tanh`, `torch.relu`, `torch.silu`, `torch.gelu` |
| Rounding | `.floor`, `.ceil`, `.round`, `.trunc` |
| Clamping | `.clamp(min=..., max=...)` |
| Binary | `+`, `-`, `*`, `/`, `**`, `torch.maximum`, `torch.minimum`, `torch.where` |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` (return BoolTensor) |

### Three patterns you'll write constantly

**1. Normalize so each slice sums to 1 (softmax-style):**

```python
out = x / x.sum(dim=d, keepdim=True)
```

**2. Normalize so each slice has unit RMS (RMSNorm-style):**

```python
rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
out = x / (rms + eps)
```

**3. Center then exp safely (numerically stable softmax):**

```python
x_shift = x - x.amax(dim=d, keepdim=True)
out = x_shift.exp() / x_shift.exp().sum(dim=d, keepdim=True)
```

All three follow the same shape: **reduce with `keepdim=True`** →
**element-wise operate** with broadcasting.

### TL;DR

> Element-wise ops (`.exp`, `+`, `*`, ...) apply to every cell regardless
> of shape and have no `dim` argument. Reduction ops (`.sum`, `.amax`,
> `.mean`, ...) collapse the specified axis and **do** take `dim`.
> Softmax / RMSNorm / layernorm get their axis-awareness from the
> reduction step; the surrounding `exp`/`sqrt`/divide just ride along
> with broadcasting.

## PyTorch optimizer internals (p, p.grad, p.data, state)

When writing a custom optimizer, the same four objects appear repeatedly:

- `p`: one trainable parameter tensor (usually `torch.nn.Parameter`).
- `p.grad`: gradient for `p`, written by `loss.backward()`.
- `p.data`: raw parameter values to update in-place.
- `self.state[p]`: persistent optimizer memory for this parameter.

### Minimal mental model

One training iteration is:

1. `optimizer.zero_grad()`
2. forward pass -> scalar `loss`
3. `loss.backward()` -> fills `p.grad`
4. `optimizer.step()` -> reads `p.grad`, updates `p.data`

So an optimizer is simply the code that consumes gradients and mutates
parameters.

### Why `state` exists

Stateful optimizers (Adam/AdamW) keep extra running statistics per parameter.
Typical keys are:

- `state["t"]`: step counter
- `state["m"]`: first moment (EMA of gradients)
- `state["v"]`: second moment (EMA of squared gradients)

That state is stored in `self.state[p]` so every parameter has its own history.

### AdamW-style step flow in code

```python
for group in self.param_groups:
    lr = group["lr"]
    beta1, beta2 = group["betas"]
    eps = group["eps"]
    wd = group["weight_decay"]

    for p in group["params"]:
        if p.grad is None:
            continue

        grad = p.grad
        state = self.state[p]
        if len(state) == 0:
            state["t"] = 0
            state["m"] = torch.zeros_like(p)
            state["v"] = torch.zeros_like(p)

        m = state["m"]
        v = state["v"]
        state["t"] += 1
        t = state["t"]

        # Decoupled weight decay.
        p.data.mul_(1.0 - lr * wd)

        # Moment updates.
        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # Bias-corrected Adam update.
        step_size = lr * ((1.0 - beta2 ** t) ** 0.5) / (1.0 - beta1 ** t)
        denom = v.sqrt().add_(eps)
        p.data.addcdiv_(m, denom, value=-step_size)
```

### Tiny scalar example

Assume one scalar parameter:

- initial `p=1.0`, `grad=0.5`
- `lr=0.1`, `wd=0.01`, `beta1=0.9`, `beta2=0.999`
- initial `m=0`, `v=0`, `t=0`

After one step:

- `t=1`
- decay: `p = p * (1 - lr*wd) = 1.0 * 0.999 = 0.999`
- `m = 0.9*0 + 0.1*0.5 = 0.05`
- `v = 0.999*0 + 0.001*0.25 = 0.00025`
- then Adam update subtracts a scaled version of `m/(sqrt(v)+eps)`

This is why AdamW is called *stateful*: each parameter carries running memory
(`m`, `v`, `t`) across steps.

### One parameter group vs multiple groups

In `for group in self.param_groups`, a *group* is just a set of parameters
sharing the same optimizer hyperparameters (`lr`, `weight_decay`, `betas`,
`eps`, ...).

Use this rule:

- Start with **one group**.
- Split into **multiple groups** only when some parameters should use
  different hyperparameters.

Typical reasons to split groups:

- Different weight decay policy (most common).
- Different learning rates (e.g., smaller LR for embeddings, larger LR for
  a newly added head).
- Different optimizer constants (`betas`, `eps`) for specific tensors.

Transformer default in many codebases:

- decay group: matrix-like weights (attention/MLP projection weights)
- no-decay group: norm weights and biases (`weight_decay=0.0`)

Minimal pattern:

```python
decay_params = []
no_decay_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.endswith("bias") or "norm" in name.lower():
        no_decay_params.append(p)
    else:
        decay_params.append(p)

optimizer = adamw_cls(
    [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
)
```

If you cannot clearly justify a different setting, keep one group.

### How to locate specific parameters inside `step`

If you need special logic for only some parameters, use one of these patterns.

Pattern A (preferred): encode intent in groups before training

```python
optimizer = adamw_cls(
    [
        {"params": decay_params, "weight_decay": 0.01, "tag": "decay"},
        {"params": no_decay_params, "weight_decay": 0.0, "tag": "no_decay"},
    ],
    lr=3e-4,
)

for group in optimizer.param_groups:
    tag = group.get("tag", "default")
    for p in group["params"]:
        # p-specific update, with behavior selected by group tag
        if tag == "no_decay":
            ...
        else:
            ...
```

Pattern B (fallback): keep a param -> name map and branch by name

```python
id_to_name = {id(p): n for n, p in model.named_parameters()}

for group in optimizer.param_groups:
    for p in group["params"]:
        name = id_to_name[id(p)]
        if name.endswith("bias") or "norm" in name.lower():
            ...
```

Notes:

- Pattern A is cleaner and usually enough.
- Pattern B is useful when the rule truly depends on module name.
- You can also use `p.ndim == 1` as a quick heuristic for norm/bias-like
  parameters, but names are more explicit.

## Gradient clipping (global L2 norm)

Gradient clipping is applied after `loss.backward()` and before
`optimizer.step()`.

### Conceptual math object vs code tensor

- In math, $g$ usually means one concatenated global gradient vector.
- In code, each `g` in a loop like `for g in grads:` is a single
    `torch.Tensor` (same shape as one parameter's gradient).

Both views are consistent: the global $g$ is what you get by flattening and
concatenating all per-parameter gradient tensors.

### Global norm and clipping rule

Compute the global L2 norm:

$$
\|g\|_2 = \sqrt{\sum_i g_i^2}.
$$

Given max norm $M$ and small stability constant $\epsilon$:

- if $\|g\|_2 \le M$, keep gradients unchanged
- if $\|g\|_2 > M$, scale all gradients by

$$
    ext{scale} = \frac{M}{\|g\|_2 + \epsilon},
\quad g \leftarrow g \cdot \text{scale}.
$$

This keeps direction and shrinks only magnitude.

### Numeric example (properly formatted)

$$
\|g\|_2 = 20,\quad M = 1,\quad \epsilon \approx 0
$$

$$
    ext{scale} = \frac{M}{\|g\|_2 + \epsilon}
\approx \frac{1}{20} = 0.05
$$

$$
g \leftarrow 0.05\,g
$$

So the new norm is approximately $1$.

## Function implementation walkthrough (current code)

This section explains what each training utility does line by line in practical
terms.

### AdamW `step`

Core flow:

1. Read group hyperparameters (`lr`, `weight_decay`, `betas`, `eps`).
2. For each parameter `p`, skip if `p.grad is None`.
3. Initialize per-parameter state on first use:
    `state["t"]`, `state["m"]`, `state["v"]`.
4. Apply decoupled weight decay directly on parameter values.
5. Update moment estimates:

$$
m \leftarrow \beta_1 m + (1-\beta_1)g,
\quad
v \leftarrow \beta_2 v + (1-\beta_2)g^2.
$$

6. Compute bias-corrected step size and perform elementwise update with
    `addcdiv_`.

Interpretation: each parameter has its own running memory (`m`, `v`, `t`),
while groups share hyperparameters.

### Cosine LR schedule with warmup

The schedule has three phases:

1. Warmup (`it < warmup_iters`): linear ramp from `0` to `max_learning_rate`.
2. Cosine decay (`warmup_iters <= it <= cosine_cycle_iters`): smooth decay to
    `min_learning_rate`.
3. Tail (`it > cosine_cycle_iters`): constant `min_learning_rate`.

Useful boundary checks:

- at `it = warmup_iters`, LR equals `max_learning_rate`
- at `it = cosine_cycle_iters`, LR equals `min_learning_rate`

### Global gradient clipping

Implementation meaning of `grads`:

- `grads` is a list of gradient tensors (`torch.Tensor`) for parameters that
  currently have gradients.
- each tensor keeps the original parameter shape.

Then the function:

1. Computes total global norm from all tensors.
2. Computes clip coefficient:

$$
	ext{clip\_coef} = \frac{M}{\|g\|_2 + 10^{-6}}.
$$

3. If `clip_coef < 1.0`, scales every gradient tensor in place.

This preserves gradient direction and only reduces magnitude when needed.
