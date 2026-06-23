# 🚀 fast-code-embed

**Version 0.0.15** — Algorithmic code embeddings. No GPU. No API keys. No nonsense.

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

**Status:** Early `0.0.x` release. The API and on-disk corpus cache format may
change between minor versions; pin a specific version in production and check the
[changelog](CHANGELOG.md) before upgrading.

## ✨ What you get

- ⚡ **Sub-millisecond scoring** — Lookup-based. No transformers, no GPU, no latency spikes.
- 📦 **30 MB total footprint** — Pretrained nomic-embed-code token vectors embedded as a binary blob.
- 🧠 **Meaningful signals** — Combines TF-IDF, pretrained nomic-embed-code vectors, and module proximity into a single [0, 1] score. Optional co-occurrence (Random Indexing) enrichment can be toggled on per corpus.
- 🔌 **Drop-in C API** — Link `build/libfast_code_embed.a`, include a header, score functions.
- ☕ **Java JNI binding** — Works from JVM environments via `FastCodeEmbed`.
- ✅ **Sanitizer-clean** — ASan and UBSan pass (run in CI on every push).

## ⚡ Quick Start

```bash
make            # build build/libfast_code_embed.a
make test       # run the test suite
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

## 📦 Installing (C)

To install the static library, public headers, and a pkg-config file:

```bash
make
sudo make install            # PREFIX defaults to /usr/local
# or stage into a package root:
make install DESTDIR=/tmp/pkg PREFIX=/usr
```

Then build against it with pkg-config:

```bash
cc app.c $(pkg-config --cflags --libs fast-code-embed) -o app
```

`make uninstall` removes the installed files. Headers land under
`$PREFIX/include/fast-code-embed`, so include the API as
`#include "semantic/semantic.h"`.

## ☕ Java JNI Binding

A first-class Java binding is available under `java/`. Full details in
[java/README.md](java/README.md). In short:

**Maven:**
```xml
<dependency>
    <groupId>io.github.nilsonsfj</groupId>
    <artifactId>fast-code-embed</artifactId>
    <version>0.0.15</version>
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
| Vector cosine | 0.25 | Semantic proximity via pretrained nomic-embed-code vectors (with optional co-occurrence enrichment) |
| Module Proximity | ×1.10 | Same-directory boost (files near each other are related) |

### RI co-occurrence enrichment (optional, off by default)

At `finalize` time the library can blend each token vector with its
co-occurring neighbors via two Reflective Random Indexing (RRI) passes. **This
is disabled by default**: the pretrained nomic vectors are used directly, with
IDF weighting suppressing ubiquitous tokens and mean-centering removing the
anisotropic common direction. Skipping enrichment finalizes ~3–4× faster and,
because the pretrained vectors are already high quality, matches or beats the
enriched ranking on most queries (on Linux-kernel reference queries, no-RI won
3/5; enrichment helped queries whose terms are highly distributed/polysemous,
e.g. *memory allocation pages*).

Enable enrichment when you want it:

```c
fce_sem_corpus_set_ri_enrichment(corpus, true);   /* before finalize */
```

```java
corpus.setRiEnrichment(true);   // before complete()
corpus.complete();
// or in one call:
corpus.complete(true);
```

The environment variable `FCE_SEM_SKIP_RI` overrides the per-corpus setting
globally: `=1` forces enrichment off, `=0` forces it on.

For advanced use, you can also wire in your own call graph, type system, or AST
via the `api_vec`, `type_vec`, `decorator_vec`, and `struct_profile[25]` fields
in `fce_sem_func_t`. Search path is configurable via `fce_query_mode_t`
(AUTO, BRUTE, FAST, TFIDF) — see [CONFIGURABLE-QUERY-MODES.md](docs/CONFIGURABLE-QUERY-MODES.md).

Sparse vector storage is available via `fce_sem_corpus_set_sparse()`; it saves
memory but is a rank-changing trade-off, not a lossless optimization. See the
docs for details.

### Embedding dimension: 256 or 768 (runtime selectable)

The embedding dimension can be chosen **at runtime** — no recompile, so the
prebuilt JNI/Maven artifact can serve either:

```c
fce_sem_set_dim(256);   /* before building/loading/querying a corpus */
```

```java
FastCodeEmbed.setDim(256);   // before any Corpus work
```

768 (full nomic-embed-code) is the default. 256 stores each int8 vector ~3×
smaller — a large resident-memory saving on big corpora (e.g. ~1.2 GB → ~0.8 GB
total RSS indexing the Linux kernel) — using a baked PCA projection that
preserves ~84% of the variance (vs naive truncation). The main trade-off is
slightly lower ranking quality; finalize is only modestly slower than 768 (the
projection is precomputed once over the pretrained table, then applied per token
as a cheap linear combine). The environment variable `FCE_SEM_DIM` overrides the
choice for the bundled benchmark tool.

A cache file records its own dimension; `fce_sem_corpus_load` / `Corpus.load`
adopt it automatically. The legacy compile-time `-DFCE_SEM_DIM_256` build still
exists (it locks the binary to 256 and shrinks the in-memory vector struct), but
runtime selection is now preferred.

For a quality/performance comparison against a general-purpose static embedding
(potion-base-8M) on Linux kernel source, see
[COMPARISON-VS-POTION-BASE-8M.md](docs/COMPARISON-VS-POTION-BASE-8M.md).

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
| Windows | x86_64 | ⚠️ Alpha / best-effort | mingw-w64 compile+link only; never executed in CI; no MSVC, no prebuilt binaries |

Pre-built binaries (GitHub release assets and the Maven Central JAR) are
provided for **Linux x86_64**, **Linux aarch64**, and **macOS arm64**. The
other supported platforms build cleanly from source (`make` / `cd java &&
./build.sh`) but are not shipped as prebuilt artifacts.

> [!WARNING]
> **Windows support is alpha and best-effort.** It is developed and maintained
> primarily for macOS/Linux; the Windows code paths exist but receive no runtime
> testing. What is actually guaranteed today is narrow:
>
> - A mingw-w64 cross-compile job (`make windows-cross`) compiles and links the
>   `_WIN32` code paths for a Windows target on every push. **This proves the
>   code builds — nothing more.** The resulting binaries are never *executed* in
>   CI (no Windows runner, no Wine), so runtime behavior on Windows is
>   unverified.
> - It is a **GCC-based (mingw) build, not MSVC.** A clean mingw build does not
>   guarantee an MSVC build, and MSVC is not tested.
> - Some Windows paths are known-degraded — e.g. thread-local scratch buffers
>   are leaked at thread exit (no TLS destructor), so Windows is best treated as
>   read-only, single-threaded use.
> - No Windows binaries are shipped in any release.
>
> Treat any Windows use as experimental and verify it yourself for your
> workload. Issues and PRs improving Windows support are welcome, but it is not
> a tier-1 target.

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

The 256-dim PCA projection matrix (`src/embed/pca_projection.h`) is checked in
and compiled into every build, powering runtime 256-dim mode (see
[Embedding dimension](#embedding-dimension-256-or-768-runtime-selectable)). To
regenerate it after re-extracting the vectors:

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

MIT — see [LICENSE](LICENSE). The redistributed nomic-embed-code vectors are
Apache 2.0 and vendored xxHash is BSD-2-Clause; see [NOTICE](NOTICE) for
third-party attributions. To report a security issue, see [SECURITY.md](SECURITY.md).
