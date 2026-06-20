# Configurable Query Modes

The search path is selectable per query, trading recall for latency. This page
documents the available modes, how to select one, and the memory/speed
trade-offs.

## Query mode enum (`src/semantic/semantic.h`)

```c
typedef enum {
    FCE_QUERY_AUTO   = 0,  /* use fast path, fall back to brute if no inv index */
    FCE_QUERY_BRUTE  = 1,  /* always brute-force scan all docs */
    FCE_QUERY_FAST   = 2,  /* inverted index + rerank; does NOT fall back to brute-force */
    FCE_QUERY_TFIDF  = 3,  /* TF-IDF candidate retrieval + RI rerank */
} fce_query_mode_t;
```

`fce_sem_config_t` carries a `fce_query_mode_t query_mode` field. `FCE_QUERY_AUTO`
is the zero value, but the generic dispatcher treats a `NULL` cfg as
`FCE_QUERY_BRUTE` (the reference path) — see below.

## Selecting the mode (no env var reading)

The library does NOT read env vars for query mode. The caller either picks a named
wrapper or sets `cfg.query_mode` on the config struct before calling the generic
search function. This keeps the library generic.

- The generic `fce_sem_search_query(corpus, query, top_k, results, count, cfg)`
  dispatches on `cfg.query_mode`. Pass `cfg == NULL` to use the **brute-force
  reference** (`FCE_QUERY_BRUTE`).
- Named presets (no cfg) cover the common cases:
  - `fce_sem_search_query_bruteforce()` — brute (same as the `NULL`-cfg default).
  - `fce_sem_search_query_fast()` — `FCE_QUERY_FAST`.
  - `fce_sem_search_query_tfidf()` — `FCE_QUERY_TFIDF`.
- Redirect logic inside the dispatcher: BRUTE → direct brute-force; TFIDF → calls the tfidf implementation; FAST/AUTO → inverted index (AUTO falls back to brute).

## Finalize-time skip (`FCE_BRUTE_ONLY`)

Two mechanisms to skip inverted index build during finalize:

- **Compile-time**: `cc -DFCE_BRUTE_ONLY ...` — unconditionally skips inverted index
- **Runtime**: `FCE_SEM_SKIP_INV_INDEX=1` env var — skips inverted index for benchmarking without recompilation

Savings (measured on the 193K-doc, 657K-token Linux-source corpus; absolute
timings vary by machine and corpus):
| Metric | Normal | Brute-only | Δ |
|--------|--------|------------|---|
| Finalize time | 64,654ms | 62,476ms | -3.4% |
| Post-build RSS | 1.0 GB | 0.8 GB | **-0.2 GB** |
| Peak RSS | 2.0 GB | 1.9 GB | -0.1 GB |

Note: The inverted index is smaller than originally estimated (~67 MB persistent,
not 1-1.8 GB). `inv_doc_ids` is 64.8 MB for 16.9M unique doc-token pairs.

## Benchmark tool (`bench_mem_query.c`)

The tool accepts the following flags:
```
./bench_mem_query <dir> [chunk_size] [--brute-only] [--sparse[=N]]
```

- `--brute-only`: sets `FCE_SEM_SKIP_INV_INDEX=1` before finalize and `query_mode=FCE_QUERY_BRUTE` for all queries.
- `--sparse[=N]`: enables sparse vector storage with top-N non-zero entries per vector (default 32). Saves ~60-70% memory on enriched/doc vectors.

  ⚠️ **Sparse mode changes ranking, not just precision.** The query magnitude is
  computed over all dimensions, but the document magnitude is computed only over
  the retained top-N dimensions. This produces a non-monotone cosine that can
  reorder top-k results compared to dense mode. Use dense mode for faithful
  rank order; sparse mode is a memory/speed trade-off only.

Shows finalize time, memory, and query benchmarks for comparison.

## Building the benchmark

The `make bench` target builds `bench_mem_query`.

## Behavior notes

- The test suite passes in the normal build, the ASan/UBSan build, and with
  `FCE_SEM_SKIP_INV_INDEX=1` (inverted index skipped).
- In brute-only mode all three search paths reduce to a brute-force scan and
  therefore return identical results.
