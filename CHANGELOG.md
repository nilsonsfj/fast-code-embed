# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] — 2026-06-15

### Added

**C library:**
- Configurable query modes: `fce_sem_config_t` with `query_mode` (AUTO, BRUTE, FAST, TFIDF)
- TF-IDF candidate retrieval path (`fce_sem_search_query_tfidf`)
- `doc_map_out` parameter on `add_docs_batch` for tracking source file paths
- Inverted index skip via `FCE_BRUTE_ONLY` env or `FCE_SEM_SKIP_INV_INDEX`
- `fce_sem_search_candidate_count` API
- `fce_sem_get_peak_rss_bytes` / `fce_sem_get_current_rss_bytes`
- Memory measurement on Linux via `/proc/self/status` (VmkPeMax)
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
