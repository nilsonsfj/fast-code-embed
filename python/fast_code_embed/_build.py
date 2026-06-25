"""cffi build script (API / out-of-line mode).

Compiles a small CPython extension `_fast_code_embed` that links against the
already-built static library `build/libfast_code_embed.a`. The 30 MB pretrained
blob is embedded inside that archive, so the resulting wheel is self-contained —
no model download at install or run time.

Run `make lib` at the repo root first (cibuildwheel does this in CI), then this
file is invoked by the build backend via the `cffi_modules` entry in
pyproject.toml.
"""

import os
from cffi import FFI

# Repo root is two levels up from this file (python/fast_code_embed/_build.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

ffibuilder = FFI()

# ── Declarations (subset of src/semantic/semantic.h actually used) ──────────
# Only what the high-level wrapper calls; kept deliberately small so the ABI
# surface we depend on is explicit. Signatures copied verbatim from the header.
ffibuilder.cdef(
    r"""
    /* libc free() — tokenize() returns malloc'd strings the caller must free.
       Resolves from the same shared libc the archive links against. */
    void free(void *ptr);

    /* Opaque corpus handle. */
    typedef struct fce_sem_corpus fce_sem_corpus_t;

    /* Ranked search result (semantic.h: { uint32_t index; float score; }). */
    typedef struct { uint32_t index; float score; } fce_sem_ranked_t;

    /* Query strategy selector. */
    typedef enum {
        FCE_QUERY_AUTO  = 0,
        FCE_QUERY_BRUTE = 1,
        FCE_QUERY_FAST  = 2,
        FCE_QUERY_TFIDF = 3
    } fce_query_mode_t;

    /* Scoring config (only query_mode is set by this binding; the rest come
       straight from fce_sem_get_config()). */
    typedef struct {
        float w_tfidf; float w_ri; float w_api; float w_type;
        float w_decorator; float w_struct_profile; float threshold;
        int max_edges;
        fce_query_mode_t query_mode;
        bool sparse_vectors;
        int sparse_nnz;
    } fce_sem_config_t;

    fce_sem_config_t fce_sem_get_config(void);

    /* Global process-state config. */
    void fce_sem_set_dim(int dim);
    int  fce_sem_active_dim(void);
    void fce_sem_set_idf_weighting(bool enabled);
    bool fce_sem_idf_weighting(void);
    void fce_sem_set_abbrev_expansion(bool enabled);
    bool fce_sem_abbrev_expansion(void);

    /* Tokenization. */
    int fce_sem_tokenize(const char *name, char **out, int max_out);

    /* Corpus lifecycle / build. */
    fce_sem_corpus_t *fce_sem_corpus_new(void);
    void fce_sem_corpus_add_doc(fce_sem_corpus_t *corpus,
                                const char **tokens, int count);
    void fce_sem_corpus_set_ri_enrichment(fce_sem_corpus_t *corpus, bool enabled);
    int  fce_sem_corpus_finalize(fce_sem_corpus_t *corpus);
    void fce_sem_corpus_free(fce_sem_corpus_t *corpus);
    int  fce_sem_corpus_doc_count(const fce_sem_corpus_t *corpus);

    int fce_sem_corpus_add_files(fce_sem_corpus_t *corpus,
                                 const char **paths, int path_count,
                                 int chunk_size, int *file_doc_counts,
                                 int max_tokens_per_chunk);

    /* Persistence. */
    int fce_sem_corpus_save(const fce_sem_corpus_t *corpus, const char *path,
                            const char *const *doc_labels, int doc_label_count);
    fce_sem_corpus_t *fce_sem_corpus_load(const char *path);
    int fce_sem_corpus_doc_label_count(const fce_sem_corpus_t *corpus);
    const char *fce_sem_corpus_doc_label(const fce_sem_corpus_t *corpus, int index);

    /* Search. */
    void fce_sem_search_query(const fce_sem_corpus_t *corpus, const char *query,
                              uint32_t top_k, fce_sem_ranked_t *results_out,
                              uint32_t *count_out, const fce_sem_config_t *cfg);
    void fce_sem_search_query_fast(const fce_sem_corpus_t *corpus,
                                   const char *query, uint32_t top_k,
                                   fce_sem_ranked_t *results_out,
                                   uint32_t *count_out);
    void fce_sem_search_query_tfidf(const fce_sem_corpus_t *corpus,
                                    const char *query, uint32_t top_k,
                                    fce_sem_ranked_t *results_out,
                                    uint32_t *count_out);
    void fce_sem_search_query_bruteforce(const fce_sem_corpus_t *corpus,
                                         const char *query, uint32_t top_k,
                                         fce_sem_ranked_t *results_out,
                                         uint32_t *count_out);
    int fce_sem_search_candidate_count(const fce_sem_corpus_t *corpus,
                                       const char *query);
    """
)

# Build against CPython's stable ABI (PEP 384). cffi's out-of-line API mode
# supports this; it makes the generated `.abi3.so` name accurate and lets a
# single wheel per platform serve every Python >= 3.9 (matching requires-python),
# instead of one wheel per interpreter minor version. Py_LIMITED_API pins the
# floor the C glue is allowed to use.
ffibuilder.set_source(
    "fast_code_embed._fast_code_embed",
    '#include "semantic/semantic.h"',
    include_dirs=[os.path.join(_ROOT, "src")],
    # Link the prebuilt static archive directly (it carries the embedded blob).
    extra_objects=[os.path.join(_ROOT, "build", "libfast_code_embed.a")],
    libraries=["pthread", "m"],
    py_limited_api=True,
    define_macros=[("Py_LIMITED_API", "0x03090000")],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
