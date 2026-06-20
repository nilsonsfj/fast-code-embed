/* * semantic.h — Algorithmic code embeddings: TF-IDF, Random Indexing,
 * API/Type/Decorator signatures, combined scoring, graph diffusion.
 *
 * Combines multiple signals into a unified [0, 1] similarity score without
 * external models or dependencies. Zero runtime inference. All signals are
 * computed from tokenized metadata (function names, paths, signatures).
 *
 * Signals:
 * 1. TF-IDF on metadata tokens (vocabulary overlap)
 * 2. Random Indexing with co-occurrence (within-codebase synonym bridging)
 * 3. API Signature vectors (same callees → related)
 * 4. Type Signature vectors (same param/return types → related)
 * 5. Module Proximity (same directory → boost)
 * 6. Decorator Pattern vectors (same annotations → related)
 * 7. AST Structural Profile (control flow shape, expression types)
 * 8. Graph Diffusion (transitive closure via neighbor blending)
 *
 * Note: signal 7 includes data flow and Halstead complexity features.
 * The rest are computed from the tokenized metadata. */
#ifndef FCE_SEMANTIC_H
#define FCE_SEMANTIC_H

#include <stdbool.h>
#include <stdint.h>
#include "version.h"

/* 64-bit-only guard. On 32-bit builds with the
 * full 5 M-vocab / 1 M-doc caps, size_t products (e.g. entry_count * 768)
 * overflow and under-allocate → heap overflow. Enforce 64-bit at compile time. */
_Static_assert(sizeof(void *) >= 8,
               "fast-code-embed requires a 64-bit target (size_t overflow on 32-bit)");

/* ── Configuration ───────────────────────────────────────────────── */

/* Random Indexing dimension. 256 is sufficient for <500K functions. */
/* 768 = nomic-embed-code embedding dimension. Matches FCE_PRETRAINED_DIM.
 * FCE_SEM_DIM_256: compile with -DFCE_SEM_DIM_256 to use 256 dims instead.
 * Saves ~640 MB memory (enriched_vecs_q + doc_vectors_q) at the cost of
 * lower embedding quality. All SIMD kernels adapt automatically. */
#ifdef FCE_SEM_DIM_256
enum { FCE_SEM_DIM = 256 };
#else
enum { FCE_SEM_DIM = 768 };
#endif

/* Random Indexing: non-zero entries per sparse random vector. */
enum { FCE_SEM_SPARSE_NNZE = 8 };

/* Co-occurrence window half-width. */
enum { FCE_SEM_WINDOW = 5 };

/* Default score threshold for SEMANTICALLY_RELATED edge emission.
 * 0.75 balances recall with precision: validated ~95% precision on
 * Linux kernel (0.80 = 100% but only 90 edges, 0.70 = 2047 edges
 * but ~80% precision). */
#define FCE_SEM_EDGE_THRESHOLD 0.75

/* Maximum SEMANTICALLY_RELATED edges per node. */
enum { FCE_SEM_MAX_EDGES = 10 };

/* AST structural profile: 25 float features per function (control flow,
 * nesting, expression types, literals, data flow, Halstead). */
enum { FCE_SEM_AST_PROFILE_DIMS = 25 };

/* Query path selection: runtime mode for dispatching search calls.
 * FCE_QUERY_AUTO is the default — preserves current behavior with no regression. */
typedef enum {
    FCE_QUERY_AUTO = 0,  /* use fast path, fall back to brute if no inv index */
    FCE_QUERY_BRUTE = 1, /* always brute-force scan all docs */
    FCE_QUERY_FAST = 2,  /* inverted index + rerank; does NOT fall back to brute-force */
    FCE_QUERY_TFIDF = 3, /* TF-IDF candidate retrieval + RI rerank */
} fce_query_mode_t;

/* Sparse vector storage: keep top-K non-zero elements per vector.
 * Enriched vectors are naturally sparse (built from 8-entry RI vectors).
 * Top-32 gives 12x compression (768→64 bytes/vector) with minimal quality loss. */
enum { FCE_SPARSE_NNZ_DEFAULT = 32 };

/* sentinel value for sparse index padding.
 * 0xFFFF marks empty slots; must be > any valid dimension index. */
_Static_assert(FCE_SEM_DIM < 0xFFFF,
               "FCE_SEM_DIM must be < 0xFFFF so the sparse-index sentinel is unambiguous");

/* Applied weights sum to ~0.85; proximity is a multiplier on top. */
typedef struct {
    float w_tfidf;
    float w_ri;
    float w_api;
    float w_type;
    float w_decorator;
    float w_struct_profile;
    float threshold;
    int max_edges;
    fce_query_mode_t query_mode;
    bool sparse_vectors; /* use sparse storage for enriched/doc vectors.
                          *
                          * WARNING — sparse mode changes RANKING, not just
                          * precision. The query magnitude is computed over
                          * all 768 dims but the document magnitude over only
                          * the top-K retained dims, producing a non-monotone
                          * cosine that can reorder results vs dense mode.
                          * Callers that assume sparse mode is a lossy-but-
                          * rank-preserving optimization will get subtly wrong
                          * top-k ordering. Use dense mode for faithful rank-
                          * order; sparse mode is a memory/speed trade-off
                          * only. Making cosine monotone would require per-
                          * document query magnitude (different docs have
                          * different sparse indices), defeating the fast
                          * pre-quantized query path. */
    int sparse_nnz;      /* non-zero entries per vector (default 32) */
} fce_sem_config_t;

/* Get default config (can be overridden via env vars). */
fce_sem_config_t fce_sem_get_config(void);

/* Check if semantic embeddings are enabled (FCE_SEMANTIC_ENABLED=1). */
bool fce_sem_is_enabled(void);

/* ── Token extraction ────────────────────────────────────────────── */

/* Maximum tokens per function from metadata (name + qn + path + sig + docstring + params). */
enum { FCE_SEM_MAX_TOKENS = 512 };

/* Split a name into tokens: camelCase, snake_case, dot.separated.
 * Writes up to max_out tokens into out. Returns token count.
 * Tokens are lowercased. Caller must free each token.
 * I1: shares the no-concurrent-shutdown requirement
 * with fce_sem_random_index — reads g_abbrev_ht without the reader-count
 * bracket. */
int fce_sem_tokenize(const char *name, char **out, int max_out);

/* Batch tokenize: tokenize count names in one call.
 * all_tokens_out is a flat array: all_tokens_out[f * max_out + t] = token string.
 * token_counts_out[f] = number of tokens for name f.
 * Caller must free each individual token string. */
void fce_sem_tokenize_batch(const char **names, int count,
                            char **all_tokens_out, int *token_counts_out,
                            int max_out);

/* ── Dense vectors ───────────────────────────────────────────────── */

/* A fixed-size dense vector for cosine similarity. */
typedef struct {
    float v[FCE_SEM_DIM];
} fce_sem_vec_t;

/* Compute cosine similarity between two dense vectors. */
float fce_sem_cosine(const fce_sem_vec_t *a, const fce_sem_vec_t *b);

/* Generate a deterministic sparse random vector for a token.
 * Uses xxHash(token) as seed. Output has SEM_SPARSE_NNZE non-zeros. */
void fce_sem_random_index(const char *token, fce_sem_vec_t *out);

/* Eagerly initialize the pretrained token lookup map.
 * Call this BEFORE dispatching parallel work that invokes fce_sem_random_index,
 * so the lazy init races are avoided entirely on the hot path. */
void fce_sem_ensure_ready(void);

/* Free global resources (pretrained token map).
 * Safe to call even if ensure_ready was never called.
 * After shutdown, ensure_ready can be called again to re-initialize.
 * env-var caches (FCE_BRUTE_WORKERS, FCE_STACK_SIZE)
 * are never reset by shutdown — changes to these env vars after a
 * shutdown/re-init cycle are silently ignored. This is consistent with
 * the env-var-read-once semantics but inconsistent with the re-init
 * contract documented here. Acceptable because env vars are a build-time
 * tuning mechanism, not a runtime knob.
 * T1: MUST NOT be called concurrently with any fce_sem_* operation.
 * Concurrent readers after shutdown will UAF — callers are responsible
 * for quiescing all worker threads before invoking this function. */
void fce_sem_shutdown(void);

/* Normalize a vector to unit length in-place. */
void fce_sem_normalize(fce_sem_vec_t *v);

/* Add src to dst: dst[i] += scale * src[i]. */
void fce_sem_vec_add_scaled(fce_sem_vec_t *dst, const fce_sem_vec_t *src, float scale);

/* ── Per-function semantic data ──────────────────────────────────── */

/* All computed signals for one function.
 *
 * WARNING FOR DIRECT C CALLERS:
 * The tfidf_indices array MUST be sorted in strict ascending order (no
 * duplicates). The two-pointer merge in the sparse TF-IDF cosine scorer
 * relies on this invariant; unsorted or duplicate indices silently produce
 * incorrect (too-low) similarity scores. The JNI layer validates this
 * automatically on marshaled Java arrays; direct C consumers MUST enforce
 * this themselves before passing fce_sem_func_t to any ranking, search,
 * or scoring function. See the debug-only FCE_ASSERT_TFIDF_SORTED macro
 * for a runtime check usable during development. */
typedef struct {
    int64_t node_id;
    const char *file_path;
    const char *file_ext;

    /* Sparse TF-IDF: stored as parallel arrays of (token_index, weight).
     *
     * CONTRACT (applies to ALL consumers — JNI, C API, and tests):
     *   tfidf_indices MUST be sorted in strict ascending order (no duplicates).
     *
     * The sparse cosine merge (fce_sparse_tfidf_cosine) uses a two-pointer
     * scan that assumes ascending order. Violations cause:
     *   - Unsorted indices: missed matches → artificially low scores.
     *   - Duplicate indices: desynchronized pointers → incorrect dot product.
     *
     * The JNI marshaling layer (marshal_func in fast_code_embed_jni.c) checks
     * this at runtime. Direct C callers receive NO automatic check in release
     * builds — enable FCE_ASSERT_TFIDF_SORTED in debug builds to catch
     * violations early. */
    int *tfidf_indices;
    float *tfidf_weights;
    int tfidf_len;

    /* Dense vectors for RI, API, Type, Decorator. */
    fce_sem_vec_t ri_vec;
    fce_sem_vec_t api_vec;
    fce_sem_vec_t type_vec;
    fce_sem_vec_t deco_vec;

    /* AST profile as float vector (decoded from "sp" property). */
    float struct_profile[FCE_SEM_AST_PROFILE_DIMS];
} fce_sem_func_t;

/* ── Corpus-level data ───────────────────────────────────────────── */

/* Opaque corpus handle for IDF and Random Indexing state. */
typedef struct fce_sem_corpus fce_sem_corpus_t;

/* Create a new corpus from function data. */
fce_sem_corpus_t *fce_sem_corpus_new(void);

/* Register a function's tokens in the corpus (for IDF counting).
 * THREAD-SAFETY: not thread-safe. Concurrent calls from multiple threads to
 * any of fce_sem_corpus_add_doc / fce_sem_corpus_add_docs_batch /
 * fce_sem_corpus_finalize / fce_sem_corpus_free on the same corpus will
 * corrupt the corpus (hash-table resize races, realloc races, torn writes
 * to doc_count). Externalise synchronization if needed. The corpus and the
 * underlying fce_ht_* are designed for single-threaded batch building, then
 * concurrent read-only queries (search/rank). */
void fce_sem_corpus_add_doc(fce_sem_corpus_t *corpus, const char **tokens, int count);

/* Batch-build the corpus from pre-tokenized documents (PARALLEL variant).
 * `all_tokens` layout: all_tokens[f * max_tokens_per_doc + t] = token pointer.
 * `token_counts[f]` = number of tokens in document f.
 * This replaces a loop of fce_sem_corpus_add_doc() calls.
 * `doc_map_out` (optional): if non-NULL, receives a malloc'd array of
 * doc_count ints where doc_map_out[d] is the index of doc d in the valid
 * docs list, or -1 if doc d was rejected. Caller must free().
 * THREAD-SAFETY: not thread-safe — see fce_sem_corpus_add_doc. */
void fce_sem_corpus_add_docs_batch(fce_sem_corpus_t *corpus, char **all_tokens,
                                   const int *token_counts, int doc_count,
                                   int max_tokens_per_doc, int *doc_map_out);

/* Finalize: compute IDF, build enriched token vectors via co-occurrence.
 * Returns 0 on success, -1 on failure (OOM). On failure the corpus is NOT
 * marked finalized and internal buffers allocated before the failure are
 * left attached — the caller MUST free the corpus (fce_sem_corpus_free) and
 * must NOT call finalize again on the same handle (retry leaks memory).
 * the previous wording "caller may free and retry" was
 * ambiguous; clarified that retry is only safe after a full free+new. */
int fce_sem_corpus_finalize(fce_sem_corpus_t *corpus);

/* Configure sparse vector storage. Must be called before finalize.
 * When enabled, enriched and doc vectors are stored as top-K non-zero
 * entries per vector (sorted index+value pairs), saving ~60-70% memory.
 * nnz=0 disables sparse mode (dense, default).
 *
 * RANK DISTORTION RISK: Sparsification introduces asymmetric magnitude
 * normalization because document vectors are truncated but queries remain
 * dense. This can cause non-monotonic rank swapping for diffuse vectors.
 * Use dense mode (nnz=0) for exact mathematical parity, and sparse mode
 * only for resource-constrained setups. */
void fce_sem_corpus_set_sparse(fce_sem_corpus_t *corpus, int nnz);

/* Get IDF weight for a token. Returns 0.0 for unknown tokens. */
float fce_sem_corpus_idf(const fce_sem_corpus_t *corpus, const char *token);

/* Get the enriched Random Indexing vector for a token (after co-occurrence).
 * WARNING: BORROWED POINTER — when the corpus stores int8 enriched vectors
 * (the normal path), this dequantizes into a _Thread_local scratch buffer.
 * The returned pointer is only valid until the next call to this function
 * from the SAME thread.
 * - DO NOT cache the pointer across calls
 * - DO NOT hold two ri_vec pointers simultaneously (the second call
 * overwrites the first)
 * - DO NOT pass the pointer to a function that may itself call ri_vec
 * Each function has its own _Thread_local scratch, so calling
 * fce_sem_corpus_token_at() does NOT invalidate a pointer from this function.
 * If you need a stable copy, memcpy the data (fce_sem_dup_ri_vec does
 * not exist — the JNI consumer copies immediately via SetFloatArrayRegion). */
const fce_sem_vec_t *fce_sem_corpus_ri_vec(const fce_sem_corpus_t *corpus, const char *token);

/* Get the total document count. */
int fce_sem_corpus_doc_count(const fce_sem_corpus_t *corpus);

/* Get the total token count (vocabulary size). */
int fce_sem_corpus_token_count(const fce_sem_corpus_t *corpus);

/* Get token name and enriched vector by index (for serialization).
 * Returns NULL if index is out of range.
 * WARNING: same _Thread_local lifetime rule as fce_sem_corpus_ri_vec, but this
 * function uses its own scratch buffer — calling ri_vec() does NOT invalidate
 * a pointer returned by this function, and vice versa. */
const char *fce_sem_corpus_token_at(const fce_sem_corpus_t *corpus, int index,
                                    const fce_sem_vec_t **out_vec, float *out_idf);

/* Persist a finalized corpus to a local cache file so it can be reloaded
 * without re-running the tokenize + finalize pipeline. doc_labels is optional
 * per-document metadata (e.g. file paths) of length doc_label_count; pass
 * NULL/0 to omit. Returns 0 on success, -1 on error (corpus not finalized,
 * unsupported representation, or I/O failure).
 *
 * This is a SAME-BUILD cache, not a portable interchange format: the file is
 * tied to the host byte order and to FCE_SEM_DIM (the 256- vs 768-dim build).
 * fce_sem_corpus_load rejects any file that does not match the running binary.
 *
 * doc_label_count must be either 0 or exactly corpus->doc_count (one label per
 * document); any other positive count returns -1, matching the invariant the
 * loader enforces. Labels are stored verbatim; when a cache will be read back
 * through the Java binding the bytes must be valid modified UTF-8 (Java strings
 * already are), since the JNI layer hands them to NewStringUTF unchanged.
 *
 * The write is atomic: data goes to a sibling temporary file that is renamed
 * over `path` on success, so a concurrent reader never observes a torn file. */
int fce_sem_corpus_save(const fce_sem_corpus_t *corpus, const char *path,
                        const char *const *doc_labels, int doc_label_count);

/* Load a corpus written by fce_sem_corpus_save, via zero-copy mmap. Returns a
 * finalized, queryable corpus, or NULL on error (missing/corrupt/truncated
 * file, or magic/version/dim/endianness mismatch). Every value the file
 * supplies that is later used as an index, offset, string, or score input is
 * range-validated at load; structurally invalid files are rejected, not mapped.
 *
 * The returned corpus references the memory mapping until fce_sem_corpus_free,
 * which unmaps it. Any per-document labels stored in the file are exposed
 * (zero-copy) via fce_sem_corpus_doc_label below.
 *
 * CAVEAT: because the corpus points directly into the mapping, the underlying
 * file must not be truncated or overwritten while the corpus is live — on POSIX
 * a shrinking file can fault (SIGBUS) on later access to no-longer-backed pages.
 * Treat loaded cache files as immutable for the corpus's lifetime; replace them
 * via atomic rename (as fce_sem_corpus_save does) rather than in-place edits. */
fce_sem_corpus_t *fce_sem_corpus_load(const char *path);

/* Number of per-document labels carried by a loaded corpus (0 if the corpus
 * was built in memory or saved without labels). */
int fce_sem_corpus_doc_label_count(const fce_sem_corpus_t *corpus);

/* Borrowed pointer to document `index`'s label (e.g. its file path), or NULL
 * if out of range or unavailable. The pointer is valid until the corpus is
 * freed; copy it if you need to outlive the corpus. */
const char *fce_sem_corpus_doc_label(const fce_sem_corpus_t *corpus, int index);

/* Free corpus. */
void fce_sem_corpus_free(fce_sem_corpus_t *corpus);

/* Batch tokenize and add to corpus in one call.
 * Tokenizes each name, registers tokens in the corpus for IDF counting.
 * This avoids the overhead of returning String[][] to Java. */
void fce_sem_corpus_add_docs_tokenized(fce_sem_corpus_t *corpus,
                                       const char **names, int count);

/* Read source files, chunk by } boundaries, tokenize, and add to corpus.
 * All work happens in C — no intermediate Java objects.
 * file_doc_counts[i] receives the number of chunks produced by file i.
 * Pass NULL if not needed.
 * max_tokens_per_chunk caps tokens per chunk (0 = FCE_SEM_MAX_TOKENS).
 * Returns total documents added, or -1 on error.
 *
 * DoS BOUNDS:
 * - per-file: 64 MB default, configurable via FCE_MAX_FILE_SIZE env var
 * (subject to errno=ERANGE rejection, so LONG_MAX is never accepted)
 * - per-doc: 512 tokens (FCE_SEM_MAX_TOKENS) hard limit
 * - per-batch: 5000 documents buffered at a time, then flushed to the corpus
 * - corpus total vocab: 5 M tokens (FCE_SEM_MAX_ENTRY_COUNT)
 * - corpus total doc count: 1 M documents (FCE_SEM_MAX_DOC_COUNT)
 * The library is offline-only and trusts the caller.
 *
 * file_doc_counts: caller-allocated array of at least path_count ints.
 * On return, file_doc_counts[fi] is the number of documents contributed
 * by paths[fi]. Passing a shorter array is undefined behavior. */
int fce_sem_corpus_add_files(fce_sem_corpus_t *corpus,
                             const char **paths, int path_count,
                             int chunk_size, int *file_doc_counts,
                             int max_tokens_per_chunk);

/* ── Combined scoring ────────────────────────────────────────────── */

/* Compute combined similarity score between two functions. */
float fce_sem_combined_score(const fce_sem_func_t *a, const fce_sem_func_t *b,
                             const fce_sem_config_t *cfg);

/* Module proximity multiplier based on file paths.
 * path comparison is byte-wise over '/' only.
 * Multi-byte UTF-8 path segments and Windows '\' separators are not handled;
 * cross-platform indexers feeding backslash paths get a flat proximity of 1.0.
 * For the stated corpora (Unix-style repo paths) this is fine. */
float fce_sem_proximity(const char *path_a, const char *path_b);

/* ── Ranking / Search ────────────────────────────────────────────── */

typedef struct {
    uint32_t index;
    float score;
} fce_sem_ranked_t;

/* Rank corpus against query and return top-k results.
 * Results are sorted descending by score (ties broken by index ascending for
 * deterministic output). Output written to results_out (caller-allocated).
 * count_out receives the number of results (<= top_k).
 * CONTRACT: corpus-side tfidf_indices MUST be sorted
 * ascending per-item even in release builds. Unsorted input silently
 * produces wrong scores via the two-pointer merge in fce_sparse_tfidf_cosine.
 * The JNI layer enforces this; direct C callers must ensure it themselves. */
void fce_sem_rank(fce_sem_func_t *query, fce_sem_func_t *corpus,
                  uint32_t corpus_size, uint32_t top_k,
                  const fce_sem_config_t *cfg,
                  fce_sem_ranked_t *results_out, uint32_t *count_out);

/* Search with minimum score threshold. */
void fce_sem_search(fce_sem_func_t *query, fce_sem_func_t *corpus,
                    uint32_t corpus_size, uint32_t top_k, float min_score,
                    const fce_sem_config_t *cfg,
                    fce_sem_ranked_t *results_out, uint32_t *count_out);

/* ── Simple API ──────────────────────────────────────────────────── */

/* Score two functions using only TF-IDF + Random Indexing.
 * Returns [0.0, 1.0] — no user-provided vectors needed.
 * 0.0 = unrelated, 1.0 = identical. */
float fce_sem_simple_score(fce_sem_func_t *a, fce_sem_func_t *b);

/* Rank corpus by simple score (exhaustive RI-cosine scan over all docs).
 * No inverted-index fast path — every corpus entry is scored. For large
 * corpora with a batch API, use simple_rank_flat instead. */
void fce_sem_simple_rank(fce_sem_func_t *query, fce_sem_func_t *corpus,
                         uint32_t corpus_size, uint32_t top_k,
                         fce_sem_ranked_t *results_out, uint32_t *count_out);

/* Search with min_score threshold using simple scoring. */
void fce_sem_simple_search(fce_sem_func_t *query, fce_sem_func_t *corpus,
                           uint32_t corpus_size, uint32_t top_k,
                           float min_score,
                           fce_sem_ranked_t *results_out, uint32_t *count_out);

/* ── Batch ranking (flat arrays, zero struct marshaling) ────────── */

/* * Rank corpus against a query using flat pre-extracted arrays.
 * All scoring happens in C — no per-item JNI overhead.
 *
 * IMPORTANT: all_tfidf_indices and q_tfidf_indices MUST be sorted ascending
 * per-document (row). The sparse TF-IDF cosine merge relies on this invariant.
 *
 * SCORING: this function intentionally uses RI-only
 * scoring (cosine of the random-index vectors). The TF-IDF weights and
 * indices in the flat layout are accepted as input for layout symmetry
 * with the structured API, but they are NOT consumed by the scorer.
 * Rationale: the flat path is intended for the Corpus.extractFlat +
 * simpleRankBatch flow, where query tokens are passed in their *original*
 * order and the corpus-side indices are positional (0..N) within a row —
 * not global corpus vocabulary IDs. Computing a sparse TF-IDF cosine
 * merge with positional indices would silently produce wrong (under-)
 * scores. To get a TF-IDF-aware score, use fce_sem_simple_search with
 * fce_sem_func_t structs that carry real vocabulary IDs (see Corpus
 * .buildFunc in the Javadoc for the equivalent Java wrapper).
 *
 * No module-proximity signal is available either, since no query file
 * path is passed in (proximity is always 1.0). For proximity-weighted
 * results, use fce_sem_simple_search with fce_sem_func_t structs.
 *
 * Corpus layout (flat, row-major):
 * all_tfidf_weights[f * max_tokens + t] — IDF weight for token t in func f
 * all_tfidf_indices[f * max_tokens + t] — vocabulary index for token t
 * tfidf_lens[f] — number of tokens in func f
 * all_ri_vecs[f * FCE_SEM_DIM + d] — RI vector dimension d for func f
 * file_paths[f] — file path string for func f
 *
 * Query layout:
 * q_tfidf_indices[0..q_tfidf_len-1] — vocabulary indices
 * q_tfidf_weights[0..q_tfidf_len-1] — corresponding IDF weights
 * q_ri_vec[0..FCE_SEM_DIM-1] — summed RI vector
 *
 * Results are sorted by score descending. */
void fce_sem_simple_rank_flat(
    /* corpus (flat) */
    const float *all_tfidf_weights,
    const int *all_tfidf_indices,
    const int *tfidf_lens,
    const float *all_ri_vecs,
    const char **file_paths,
    uint32_t corpus_size,
    int max_tokens,
    /* query */
    const int *q_tfidf_indices,
    const float *q_tfidf_weights,
    int q_tfidf_len,
    const float *q_ri_vec,
    /* output */
    uint32_t top_k,
    fce_sem_ranked_t *results_out,
    uint32_t *count_out);

/* ── Search query ──────────────────────────────────────────────── */

/* High-level configurable search: tokenize a query string, build a query
 * vector from enriched token vectors, return top-k ranked results.
 *
 * This is the generic entry point; the strategy is chosen by cfg.query_mode:
 *   FCE_QUERY_BRUTE — scan ALL document vectors (the reference path).
 *   FCE_QUERY_AUTO  — fast inverted-index path, fall back to brute.
 *   FCE_QUERY_FAST  — inverted-index path; does NOT fall back to brute.
 *   FCE_QUERY_TFIDF — TF-IDF sparse-cosine candidates, then RI rerank.
 *
 * Pass cfg == NULL to use the brute-force reference (FCE_QUERY_BRUTE). The
 * suffixed wrappers below are presets over this function for the common cases.
 *
 * NOTE: the inverted-index strategies (AUTO/FAST/TFIDF) only find documents
 * sharing at least one literal token with the query; documents related solely
 * through Random-Indexing synonym bridging will NOT appear. Use the default
 * brute-force path for exhaustive search when this matters. */
void fce_sem_search_query(const fce_sem_corpus_t *corpus,
                          const char *query,
                          uint32_t top_k,
                          fce_sem_ranked_t *results_out,
                          uint32_t *count_out,
                          const fce_sem_config_t *cfg);

/* Brute-force reference search: scans ALL document vectors with cosine
 * similarity. Slower but guaranteed to find the global top-k. Equivalent to
 * fce_sem_search_query(..., NULL). */
void fce_sem_search_query_bruteforce(const fce_sem_corpus_t *corpus,
                                     const char *query,
                                     uint32_t top_k,
                                     fce_sem_ranked_t *results_out,
                                     uint32_t *count_out);

/* Fast search: inverted-index candidate retrieval + RI rerank. Does NOT fall
 * back to brute-force. Preset of fce_sem_search_query with FCE_QUERY_FAST. */
void fce_sem_search_query_fast(const fce_sem_corpus_t *corpus,
                               const char *query,
                               uint32_t top_k,
                               fce_sem_ranked_t *results_out,
                               uint32_t *count_out);

/* TF-IDF hybrid search: TF-IDF sparse cosine for candidate retrieval, then
 * reranks with RI cosine. Better candidate quality than the keyword approach
 * but slightly slower candidate scoring. Preset of fce_sem_search_query with
 * FCE_QUERY_TFIDF. */
void fce_sem_search_query_tfidf(const fce_sem_corpus_t *corpus,
                                const char *query,
                                uint32_t top_k,
                                fce_sem_ranked_t *results_out,
                                uint32_t *count_out);

/* Return the number of candidates the inverted index would retrieve
 * for a given query. Useful for understanding search selectivity. */
int fce_sem_search_candidate_count(const fce_sem_corpus_t *corpus,
                                   const char *query);

/* ── Graph diffusion ─────────────────────────────────────────────── */

/* Apply one iteration of graph diffusion to a combined embedding.
 * Blends with mean of top-k neighbor embeddings (α=0.3).
 *
 * CONTRACT: `combined` MUST NOT alias any entry in `neighbors[]`. The
 * internal `mean` accumulator reads each `neighbors[n].v` once before any
 * write to `combined`, so today an aliased call is accidentally safe, but
 * the function depends on this contract for correctness — future refactors
 * that read `combined` after the mean loop would silently produce
 * self-blended (corrupted) results. An assert enforces the contract in
 * debug builds. */
void fce_sem_diffuse(fce_sem_vec_t *combined, const fce_sem_vec_t *neighbors, int neighbor_count,
                     float alpha);

#endif /* FCE_SEMANTIC_H */
