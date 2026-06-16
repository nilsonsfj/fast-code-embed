# 🚀 fast-code-embed

**Version 0.0.6** — Algorithmic code embeddings. No GPU. No API keys. No nonsense.

✨ fast-code-embed is a standalone C library that scores code function pairs by
semantic similarity — using a 30 MB lookup table, TF-IDF, and Random Indexing.
Zero runtime inference. Works everywhere.

Born as an extraction and rewrite of the embedding logic from
[codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp), then
optimized further: faster SIMD kernels, lower memory footprint, improved
scoring defaults, and a cleaner API.

[![Tests](https://github.com/nilsonsfj/fast-code-embed/actions/workflows/test.yml/badge.svg)](https://github.com/nilsonsfj/fast-code-embed/actions/workflows/test.yml)
[![Maven Central](https://img.shields.io/maven-central/v/io.github.nilsonsfj/fast-code-embed.svg)](https://central.sonatype.com/artifact/io.github.nilsonsfj/fast-code-embed)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## ✨ What you get

- ⚡ **Sub-millisecond scoring** — Lookup-based. No transformers, no GPU, no latency spikes.
- 📦 **30 MB total footprint** — Pretrained nomic-embed-code token vectors embedded as a binary blob.
- 🧠 **Meaningful signals** — Combines TF-IDF, Random Indexing (enriched with co-occurrence context), and module proximity into a single [0, 1] score.
- 🔌 **Drop-in C API** — Link `build/libfast_code_embed.a`, include a header, score functions.
- ☕ **Java JNI binding** — Works from JVM environments via `FastCodeEmbed`.
- ✅ **Sanitizer-clean** — ASan, UBSan, and MSan all pass.

## ⚡ Quick Start

```bash
make            # build build/libfast_code_embed.a
make test       # run the test suite (64/64 pass)
make bench      # build bench_mem_query
```

```c
#include "semantic/semantic.h"

fce_sem_ensure_ready();

/* 1. Build corpus from all function token lists */
fce_sem_corpus_t *corp = fce_sem_corpus_new();
const char *tok_a[] = {"handle", "request", "parse"};
const char *tok_b[] = {"validate", "user", "auth"};
fce_sem_corpus_add_doc(corp, tok_a, 3);
fce_sem_corpus_add_doc(corp, tok_b, 3);
fce_sem_corpus_finalize(corp);

/* 2. Build func descriptors — TF-IDF weights + RI vector */
int idx_a[] = {0, 1, 2};
float w_a[] = { fce_sem_corpus_idf(corp, "handle"),
                fce_sem_corpus_idf(corp, "request"),
                fce_sem_corpus_idf(corp, "parse") };
int idx_b[] = {0, 1, 2};
float w_b[] = { fce_sem_corpus_idf(corp, "validate"),
                fce_sem_corpus_idf(corp, "user"),
                fce_sem_corpus_idf(corp, "auth") };

fce_sem_func_t func_a = {0}, func_b = {0};
func_a.file_path     = "src/handler.c";
func_a.tfidf_indices = idx_a;
func_a.tfidf_weights = w_a;
func_a.tfidf_len     = 3;
const fce_sem_vec_t *rv = fce_sem_corpus_ri_vec(corp, "handle");
if (rv) func_a.ri_vec = *rv;

func_b.file_path     = "src/auth.c";
func_b.tfidf_indices = idx_b;
func_b.tfidf_weights = w_b;
func_b.tfidf_len     = 3;
rv = fce_sem_corpus_ri_vec(corp, "validate");
if (rv) func_b.ri_vec = *rv;

/* 3. Score */
float score = fce_sem_simple_score(&func_a, &func_b);

fce_sem_corpus_free(corp);
```

## ☕ Java JNI Binding

A first-class Java binding is available under `java/`. Full details in
[java/README.md](java/README.md). In short:

**Maven:**
```xml
<dependency>
    <groupId>io.github.nilsonsfj</groupId>
    <artifactId>fast-code-embed</artifactId>
    <version>0.0.5</version>
</dependency>
```

**From source:**

```bash
# Build the C library, then the Java binding
make
cd java && ./build.sh
```

```java
import io.github.nilsonsfj.fastcodeembed.*;

FastCodeEmbed.init();

try (Corpus corp = new Corpus()) {
    // Add all function token lists to compute IDF + RI vectors
    corp.addDocsBatch(new String[][]{
        {"handle", "request", "parse"},
        {"validate", "user", "auth"}
    });
    corp.complete();

    // Build func descriptors
    FuncDescriptor a = new FuncDescriptor("src/handler.c");
    a.setTfidf(new int[]{0, 1, 2}, new float[]{
        corp.getIdf("handle"), corp.getIdf("request"), corp.getIdf("parse")
    });
    a.setRiVec(corp.getRiVec("handle"));

    FuncDescriptor b = new FuncDescriptor("src/auth.c");
    b.setTfidf(new int[]{0, 1, 2}, new float[]{
        corp.getIdf("validate"), corp.getIdf("user"), corp.getIdf("auth")
    });
    b.setRiVec(corp.getRiVec("validate"));

    // Pairwise score
    float score = FastCodeEmbed.simpleScore(a, b);

    // Rank a corpus (use simpleRankBatch + extractFlat for large corpora)
    FuncDescriptor[] all = {a, b};
    SearchResult[] results = FastCodeEmbed.simpleRank(a, all, 10);
}
```

All classes are in `io.github.nilsonsfj.fastcodeembed`:
`FastCodeEmbed`, `Corpus`, `FuncDescriptor`, `SearchResult`, `NativeLibrary`.

## How it works

Given a corpus of code metadata (function names, paths, signatures, docstrings),
the library builds two complementary representations:

| Signal | Weight | What it captures |
|--------|--------|-----------------|
| TF-IDF cosine | 0.20 | Token-level overlap, downweighted by frequency |
| Random Indexing cosine | 0.25 | Semantic proximity via pretrained + enriched vectors |
| Module Proximity | ×1.10 | Same-directory boost (files near each other are related) |

For advanced use, you can also wire in your own call graph, type system, or AST
via the `api_vec`, `type_vec`, `decorator_vec`, and `struct_profile[25]` fields
in `fce_sem_func_t`. Search path is configurable via `fce_query_mode_t`
(AUTO, BRUTE, FAST, TFIDF) — see [CONFIGURABLE-QUERY-MODES.md](docs/CONFIGURABLE-QUERY-MODES.md).

Sparse vector storage is available via `fce_sem_corpus_set_sparse()`; it saves
memory but is a rank-changing trade-off, not a lossless optimization. See the
docs for details.

`fce_sem_ensure_ready()` eagerly builds the pretrained token lookup map. Under
extreme memory pressure the map may be only partially populated; any token that
fails to insert falls back to a deterministic sparse random vector, with a log
warning, and embedding quality may degrade slightly.

## Platform Support

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| Linux | x86_64 | Fully supported | AVX2 runtime dispatch |
| Linux | aarch64 | Supported | ARMv8.2+ recommended (DotProd) |
| macOS | arm64 (Apple Silicon) | Fully supported | ARMv8.2+ with DotProd |
| macOS | x86_64 | Supported | AVX2 runtime dispatch |
| Windows | x86_64 | Partial | Single-threaded recommended |

### ARM builds

Pre-built binaries target **ARMv8.2-a** (DotProd extension). This covers all
Apple Silicon Macs, AWS Graviton 2+, Raspberry Pi 4+, and most modern aarch64
Linux systems.

If your target is ARMv8.0 without DotProd (e.g. older Raspberry Pi 3, some
embedded SoCs), build from source instead:

```bash
make clean && make -j4 lib CC=aarch64-linux-gnu-gcc AR=aarch64-linux-gnu-ar \
  CFLAGS="-O2 -std=c11 -DNDEBUG -fPIC"
```

This produces a library that uses the portable integer multiply-accumulate
fallback instead of the DotProd instruction. Functionality is identical; the
int8 dot product hot path (used in brute-force search) is 2–4x slower on
ARMv8.0 vs ARMv8.2+.

## Regenerating the Embedding Table

The included `code_vectors.bin` (30 MB) was extracted from nomic-embed-code 7B
via offline inference. To reproduce:

```bash
pip install torch transformers sentence-transformers
python3 scripts/extract_nomic_vectors.py
```

Expect ~2–3 h on GPU, ~6–10 h on Apple Silicon CPU.

To build the 256-dim reduced-dimension mode (`-DFCE_SEM_DIM_256`), also generate
the PCA projection matrix:

```bash
python3 scripts/compute_pca_matrix.py src/embed/code_vectors.bin > src/embed/pca_projection.h
```

## Architecture

```
src/
├── version.h/c        Semantic version
├── embed/             Pretrained code vectors (nomic-embed-code, 30 MB)
├── semantic/          Tokenization, TF-IDF, RI, corpus, scoring
├── foundation/        Hash table, threading, logging, platform detection
├── pipeline/          Parallel-for dispatcher (pthreads)
└── xxhash/            Vendored xxHash (header-only)

java/                  JNI binding — see java/README.md
```

## License

MIT — see [LICENSE](LICENSE). Embedding vectors are Apache 2.0.
