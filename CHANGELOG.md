# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Corpus cache save/load: `fce_sem_corpus_save` writes a finalized corpus to a
  single file and `fce_sem_corpus_load` maps it back via zero-copy `mmap`,
  rebuilding only the small vocabulary hash table. Reloading is dramatically
  faster than rebuilding (measured ~166× vs. `finalize` on the Linux-source
  corpus: ~0.24 s load vs. ~39 s build) and shares pages across processes.
  Optional per-document labels (e.g. file paths) round-trip in the same file and
  are exposed zero-copy via `fce_sem_corpus_doc_label`. Exposed in Java as
  `Corpus.save(String)` / `Corpus.load(String)`. The file is a same-build cache
  (tied to host byte order and `FCE_SEM_DIM`); mismatched or corrupt files are
  rejected at load. `bench_mem_query` gained a `--save-load[=path]` flag that
  times save/load and verifies query parity against the in-memory corpus.
  The loader range-validates every value it later uses as an index, offset,
  string, or score input (token/label offsets, `doc_freq`, inverted-index
  sortedness, sparse dimension indices, inverse magnitudes, and header flags),
  so a truncated or crafted cache is rejected rather than mapped; saves are
  atomic (written to a sibling temp file then renamed over the destination) so a
  concurrent reader never observes a partially written file
- `make install` / `make uninstall` targets that install the static library,
  public headers, and a generated `fast-code-embed.pc` pkg-config file (honors
  `PREFIX` and `DESTDIR`; portable across Linux and macOS), so C consumers can
  build with `pkg-config --cflags --libs fast-code-embed` instead of hand-wiring
  include and link paths
- `NOTICE` file documenting third-party redistribution: the Apache 2.0
  nomic-embed-code vectors and the BSD-2-Clause vendored xxHash
- `SECURITY.md` describing the private vulnerability-reporting policy

### Changed
- Renamed the internal "Priority / Edge Case / Concurrency Fixes" test-output
  banners to neutral functional groupings (Robustness & Memory Limits, Edge
  Cases, Concurrency, Tokenization, Corpus Lifecycle), and dropped the internal
  `h*/m*` priority prefixes from test function names

### Fixed
- JNI handle table: a slot could be reused (`alloc_handle`) after `take_handle`
  cleared its pointer but before in-flight users had drained, letting a
  concurrent `close` free a corpus another thread was still querying
  (use-after-free). Slots are now reused only when both unowned and fully
  drained
- JNI native allocation failures (`malloc`/`calloc`/`strdup`) in the document
  add, tokenize, and ranking paths now raise `OutOfMemoryError` instead of
  silently no-op'ing or returning a bare sentinel; result-array builders check
  `ExceptionCheck` after `SetObjectArrayElement`
- `Corpus.addDocsTokenized(names, paths)` no longer misassigns file-path labels
  when a name tokenizes to zero documents (it routes through the doc-map-aware
  batch path), so saved caches carry correct labels; `Corpus.load` closes the
  partially built corpus if label materialization throws
- Removed the last leftover internal review reference from the
  `.note.GNU-stack` comment in `code_vectors_blob.S`, completing the review-ID
  cleanup
- Filled in the previously empty `[0.0.9]` and `[0.0.10]` changelog sections;
  the release workflow uses these as GitHub release notes, so tagged releases no
  longer ship with a blank description

## [0.0.10] — 2026-06-17

### Changed
- Maintenance release: version and packaging metadata only; no functional source
  changes since 0.0.9

## [0.0.9] — 2026-06-17

### Changed
- Maintenance release: version and packaging metadata only; no functional source
  changes since 0.0.8

## [0.0.8] — 2026-06-16

### Changed
- Removed the remaining development-time review/task ID prefixes (e.g. `C-1:`,
  `H-2:`, `Review NNNN §X`) from all C, JNI, and test source comments,
  completing the cleanup begun in 0.0.4
- Rewrote `docs/OVERVIEW-AND-ARCHITECTURE.md` and
  `docs/CONFIGURABLE-QUERY-MODES.md` from internal-review/worklog style into
  user-facing reference documentation

### Fixed
- Default build no longer fails under Clang on Linux: `hash_table.c` and
  `compat_thread.c` now request the feature-test macros (`_DEFAULT_SOURCE` /
  `_POSIX_C_SOURCE`) needed to declare `arc4random`, `getentropy`, and
  `posix_memalign` under strict `-std=c11`. Previously these produced
  implicit-declaration warnings under GCC and a hard error under Clang
- Sanitizer build now compiles with `-fsanitize=address,undefined` and treats
  UBSan errors as fatal, so the CI sanitizer job exercises both ASan and UBSan.
  The README no longer claims MSan coverage, which was never wired into the
  build or CI
- The tree now builds warning-clean under `-Wall -Wextra -Wpedantic` on both
  GCC and Clang: log macros no longer rely on the GNU zero-variadic extension,
  the `STDC FENV_ACCESS` pragmas are guarded to the compilers that honor them,
  and stray warnings in the test and benchmark drivers were removed
- `java/build.sh` now selects the uninitialized-variable warning flag by
  compiler family (GCC vs Clang) and honors `$CC`, instead of assuming GCC on
  Linux
- The Tests workflow now actually builds with each matrix compiler
  (`CC=${{ matrix.compiler }}`), so the Clang job exercises Clang
- The Release workflow now gates artifact builds and Maven publishing on a
  version-validation job, so a tag whose version strings do not match cannot
  publish (the standalone validate-release workflow was folded in)
- `scripts/bump_version.sh` is now portable to GNU sed (Linux): in-place edits
  use a GNU/BSD shim and the changelog insertion uses awk instead of BSD `sed`

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
- Static analysis fixes (jresult hoisting, sweep-before-create with age gate)
- Thread safety fixes (broadcast on shared condvar, parallel brute-force worker count)
- Security hardening (fixed token leak on the error path, `.note.GNU-stack` for NX)

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
