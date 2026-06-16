# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.8] — 2026-06-16

## [0.0.6] — 2026-06-16

### Changed
- aarch64 cross-compile now targets ARMv8.2-a (DotProd) for faster int8 dot products
- Added ARM build documentation for ARMv8.0 targets without DotProd

## [0.0.5] — 2026-06-16

### Added
- Version validation: CI checks that all version strings in the codebase match the git tag before releasing

## [0.0.4] — 2026-06-16

### Changed
- Strip all internal review/task ID prefixes from C and Java source comments
- Rewrite `semantic.h` header to remove project-specific language from original codebase
- Move `bench_c.c` and `index_dir.c` to `tools/` directory
- Fix linker flag references in tool build comments (`-lstatic_nomic` → `-lfast_code_embed`)
- Fix `java/README.md` Java version claim to reflect CI reality
- Document PCA projection script for 256-dim builds

### Added
- `CONTRIBUTING.md` with build, test, and PR guidelines

## [0.0.3] — 2026-06-15

### Fixed
- Maven JAR now bundles native libraries for Linux x86\_64, Linux aarch64, and macOS arm64
  (0.0.2 JAR on Maven Central was published without natives and would fail at runtime)
- `NativeLibrary` loads from `/native/{os}-{arch}/{lib}` in the JAR, with `amd64→x86_64` normalization
- `scripts/extract_nomic_vectors.py`: corrected `--output-dir` default (`src/nomic` → `src/embed`)
  and hardcoded `incbin_path` (`vendored/nomic/...` → derived from `--output-dir`), so
  `make extract` now produces output that actually builds
- README: corrected upstream link (`anomalyco` → `DeusData`), removed contradictory
  "All rights reserved" footer, fixed artifact path (`lib/` → `build/`)
- CHANGELOG: corrected `/proc/self/status` field name (`VmkPeMax` → `VmHWM`)

## [0.0.2] — 2026-06-15

### Added

**C library:**
- Configurable query modes: `fce_sem_config_t` with `query_mode` (AUTO, BRUTE, FAST, TFIDF)
- TF-IDF candidate retrieval path (`fce_sem_search_query_tfidf`)
- `doc_map_out` parameter on `add_docs_batch` for tracking source file paths
- Inverted index skip via `FCE_BRUTE_ONLY` env or `FCE_SEM_SKIP_INV_INDEX`
- `fce_sem_search_candidate_count` API
- `fce_sem_get_peak_rss_bytes` / `fce_sem_get_current_rss_bytes`
- Memory measurement on Linux via `/proc/self/status` (VmHWM)
- `make bench` target (`bench_mem_query` tool)

**Java JNI binding:**
- `addDocsBatch(docs, paths)` / `addDocsTokenized(names, paths)` — batch-add with file paths
- `addFiles(paths, chunkSize, maxTokens)` — read, chunk at `}` boundaries, tokenize in C
- `FlatCorpus` / `Corpus.extractFlat()` for pre-extracted flat arrays
- `getDocPath` / `getDocPaths` / `clearDocPaths`
- `tokenizeBatch` (batch tokenization via single JNI call)
- `searchQuery` / `searchQueryTfidf` / `searchQueryBruteforce`
- `searchCandidateCount`
- `getPeakRssBytes` / `getCurrentRssBytes`

**Code quality:**
- Static analysis fixes (C-1: jresult hoisting, C-2: sweep-before-create with age gate)
- Thread safety fixes (H-2: broadcast on shared condvar, H-3: parallel brute-force nworkers)
- Security hardening (H-1: token leak via goto, H-4: .note.GNU-stack for NX)

### Changed
- Search path now configurable via `fce_query_mode_t` in `fce_sem_config_t`
- Inverted index ~67 MB (skippable)
- Test suite: 64/64 C, 21/21 Java

## [0.0.1] — 2026-05-27

### Added

**C library:**
- TF-IDF scoring on tokenized metadata (sparse cosine)
- Random Indexing with co-occurrence enrichment (768d vectors)
- Module proximity multiplier based on file paths
- Full combined scoring API (`fce_sem_combined_score`)
- Simple scoring API (`fce_sem_simple_score`) — TF-IDF + RI only, [0.0, 1.0]
- Corpus lifecycle: `new`, `add_doc`, `add_docs_batch`, `finalize`, `free`
- IDF lookup and enriched RI vector access per token
- Ranking and search with configurable thresholds
- Graph diffusion (neighbor blending)
- Pretrained nomic-embed-code token vectors (30 MB blob, zero-copy via `.incbin`)
- xxHash for deterministic token hashing
- Parallel-for dispatch via pthreads
- Tokenizer: camelCase, snake_case, dot-separated, PascalCase, abbreviations
- Version API: `fce_version()`, `FCE_VERSION_MAJOR/MINOR/PATCH`

**Java JNI binding:**
- `FastCodeEmbed` — init, tokenize, proximity, simpleScore, simpleRank, simpleSearch
- `Corpus` — addDoc, addDocsBatch, complete, getIdf, getRiVec, buildFunc, close (AutoCloseable)
- `FuncDescriptor` — setTfidf, setRiVec, getFilePath
- `SearchResult` — getIndex, getScore
- `NativeLibrary` — loads .dylib/.so/.dll from system path or JAR resources
- Standalone test runner (no framework dependencies)

**Project:**
- Makefile with `make`, `make test`, `make clean`, `make extract`
- `scripts/extract_nomic_vectors.py` for regenerating nomic vectors offline
- README with usage examples for both C and Java
- 31 C tests, 14 Java tests
