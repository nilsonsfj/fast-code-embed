# 🚀 fast-code-embed

**Version 0.0.1** — Algorithmic code embeddings. No GPU. No API keys. No nonsense.

✨ fast-code-embed is a standalone C library that scores code function pairs by
semantic similarity — using a 30 MB lookup table, TF-IDF, and Random Indexing.
Zero runtime inference. Works everywhere.

Born as an extraction and rewrite of the embedding logic from
[codebase-memory-mcp](https://github.com/anomalyco/codebase-memory-mcp), then
optimized further: faster SIMD kernels, lower memory footprint, improved
scoring defaults, and a cleaner API.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## ✨ What you get

- ⚡ **Sub-millisecond scoring** — Lookup-based. No transformers, no GPU, no latency spikes.
- 📦 **30 MB total footprint** — Pretrained nomic-embed-code token vectors embedded as a binary blob.
- 🧠 **Meaningful signals** — Combines TF-IDF, Random Indexing (enriched with co-occurrence context), and module proximity into a single [0, 1] score.
- 🔌 **Drop-in C API** — Link `libfast_code_embed.a`, include a header, score functions.
- ☕ **Java JNI binding** — Works from JVM environments via `FastCodeEmbed`.
- ✅ **Sanitizer-clean** — ASan, UBSan, and MSan all pass.

## ⚡ Quick Start

```bash
make            # build lib/libfast_code_embed.a
make test       # run the test suite
```

```c
#include "semantic/semantic.h"

fce_sem_ensure_ready();

fce_sem_corpus_t *corp = fce_sem_corpus_new();
char *tokens[] = {"handle", "request", "parse"};
fce_sem_corpus_add_doc(corp, tokens, 3);
fce_sem_corpus_finalize(corp);

/* build func descriptors... */

float score = fce_sem_simple_score(&func_a, &func_b);

fce_sem_corpus_free(corp);
```

## ☕ Java JNI Binding

A first-class Java binding is available under `java/`. Full details in
[java/README.md](java/README.md). In short:

```bash
# Build the C library, then the Java binding
make
cd java && ./build.sh
```

```java
import com.github.nilsonsfj.fastcodeembed.*;

FastCodeEmbed.init();

try (Corpus corp = new Corpus()) {
    corp.addDocsBatch(new String[][]{
        {"handle", "request", "parse"},
        {"validate", "user", "check"}
    });
    corp.complete();

    FuncDescriptor a = corp.buildFunc("src/handler.c", new String[]{"handle", "request"});
    FuncDescriptor b = corp.buildFunc("src/auth.c", new String[]{"validate", "user"});

    float score = FastCodeEmbed.simpleScore(a, b);
    SearchResult[] results = FastCodeEmbed.simpleRank(a, new FuncDescriptor[]{a, b}, 10);
}
```

All classes are in `com.github.nilsonsfj.fastcodeembed`:
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
in `fce_sem_func_t`.

## Regenerating the Embedding Table

The included `code_vectors.bin` (30 MB) was extracted from nomic-embed-code 7B
via offline inference. To reproduce:

```bash
pip install torch transformers sentence-transformers
python3 scripts/extract_nomic_vectors.py
```

Expect ~2–3 h on GPU, ~6–10 h on Apple Silicon CPU.

## Architecture

```
src/
├── version.h/c        Semantic version (0.0.1)
├── embed/             Pretrained code vectors (nomic-embed-code, 30 MB)
├── semantic/          Tokenization, TF-IDF, RI, corpus, scoring
├── foundation/         Hash table, threading, logging, platform detection
├── pipeline/          Parallel-for dispatcher (pthreads)
└── xxhash/            Vendored xxHash (header-only)

java/                  JNI binding — see java/README.md
```

## License

MIT — see [LICENSE](LICENSE). Embedding vectors are Apache 2.0.

Nilson Santos Figueiredo Junior 2026. All rights reserved.