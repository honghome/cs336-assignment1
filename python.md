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
