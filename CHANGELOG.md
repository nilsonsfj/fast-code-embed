# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
