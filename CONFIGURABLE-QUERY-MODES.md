# Configurable Query Modes — Implementation

## What was implemented

### 1. Query mode enum (`src/semantic/semantic.h`)

```c
typedef enum {
    FCE_QUERY_AUTO   = 0,  /* use fast path, fall back to brute if no inv index */
    FCE_QUERY_BRUTE  = 1,  /* always brute-force scan all docs */
    FCE_QUERY_FAST   = 2,  /* inverted index + rerank, fall back to brute if < top_k */
    FCE_QUERY_TFIDF  = 3,  /* TF-IDF candidate retrieval + RI rerank */
} fce_query_mode_t;
```

Added `fce_query_mode_t query_mode` field to `fce_sem_config_t`. Default: `FCE_QUERY_AUTO`.

### 2. Library API (no env var reading)

The library does NOT read env vars for query mode. The caller sets `cfg.query_mode` on
the config struct before calling search functions. This keeps the library generic.

- `fce_sem_search_query()` and `fce_sem_search_query_tfidf()` now take `const fce_sem_config_t *cfg`
- Pass `NULL` for backward-compatible AUTO behavior
- Redirect logic: BRUTE → direct brute-force; TFIDF → calls tfidf function; FAST → forces inverted index

### 3. Finalize-time skip (`FCE_BRUTE_ONLY`)

Two mechanisms to skip inverted index build during finalize:

- **Compile-time**: `cc -DFCE_BRUTE_ONLY ...` — unconditionally skips inverted index
- **Runtime**: `FCE_SEM_SKIP_INV_INDEX=1` env var — skips inverted index for benchmarking without recompilation

Savings (193K docs, 657K tokens):
| Metric | Normal | Brute-only | Δ |
|--------|--------|------------|---|
| Finalize time | 64,654ms | 62,476ms | -3.4% |
| Post-build RSS | 1.0 GB | 0.8 GB | **-0.2 GB** |
| Peak RSS | 2.0 GB | 1.9 GB | -0.1 GB |

Note: The inverted index is smaller than originally estimated (~67 MB persistent,
not 1-1.8 GB). `inv_doc_ids` is 64.8 MB for 16.9M unique doc-token pairs.

### 4. Benchmark tool (`bench_mem_query.c`)

New `--brute-only` flag:
```
./bench_mem_query <dir> [chunk_size] [--brute-only]
```

Sets `FCE_SEM_SKIP_INV_INDEX=1` before finalize and `query_mode=FCE_QUERY_BRUTE` for all queries.
Shows finalize time, memory, and query benchmarks for comparison.

### 5. Makefile

New `make bench` target builds `bench_mem_query`.

## Testing

- 53/53 tests pass (normal build)
- 53/53 tests pass (ASAN build, zero warnings)
- 53/53 tests pass with `FCE_SEM_SKIP_INV_INDEX=1` env var (inverted index skipped)
- Brute-only mode: all three search paths produce identical 10/10 overlap (all brute-force)
