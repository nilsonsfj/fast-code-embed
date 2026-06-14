/* * bench_c.c — Benchmark fast-code-embed using its own source code.
 *
 * Measures: init, tokenization, corpus build (single + batch),
 * IDF lookup, RI vector access, simple scoring, ranking.
 *
 * Build: cc -O2 -std=c11 -Isrc bench_c.c -Lbuild -lstatic_nomic -lpthread -lm -o bench_c
 * Run: ./bench_c */
#include "semantic/semantic.h"
#include "foundation/platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Corpus data: real function names from this project's source ── */

typedef struct {
    const char *path;
    const char *full_name; /* unsplit name, used for tokenize benchmark */
    const char *tokens[12];
    int ntokens;
} bench_func_t;

static bench_func_t corpus[] = {
    /* semantic/semantic.c */
    {"src/semantic/semantic.c", "fce_sem_get_config", {"fce", "sem", "get", "config"}, 4},
    {"src/semantic/semantic.c", "fce_sem_is_enabled", {"fce", "sem", "is", "enabled"}, 4},
    {"src/semantic/semantic.c", "fce_sem_tokenize", {"fce", "sem", "tokenize"}, 3},
    {"src/semantic/semantic.c", "fce_sem_cosine", {"fce", "sem", "cosine"}, 3},
    {"src/semantic/semantic.c", "fce_sem_random_index", {"fce", "sem", "random", "index"}, 4},
    {"src/semantic/semantic.c", "fce_sem_ensure_ready", {"fce", "sem", "ensure", "ready"}, 4},
    {"src/semantic/semantic.c", "fce_sem_normalize", {"fce", "sem", "normalize"}, 3},
    {"src/semantic/semantic.c", "fce_sem_vec_add_scaled", {"fce", "sem", "vec", "add", "scaled"}, 5},
    {"src/semantic/semantic.c", "fce_sem_corpus_new", {"fce", "sem", "corpus", "new"}, 4},
    {"src/semantic/semantic.c", "fce_sem_corpus_add_doc", {"fce", "sem", "corpus", "add", "doc"}, 5},
    {"src/semantic/semantic.c", "fce_sem_corpus_add_docs_batch", {"fce", "sem", "corpus", "add", "docs", "batch"}, 6},
    {"src/semantic/semantic.c", "fce_sem_corpus_finalize", {"fce", "sem", "corpus", "finalize"}, 4},
    {"src/semantic/semantic.c", "fce_sem_corpus_idf", {"fce", "sem", "corpus", "idf"}, 4},
    {"src/semantic/semantic.c", "fce_sem_corpus_ri_vec", {"fce", "sem", "corpus", "ri", "vec"}, 5},
    {"src/semantic/semantic.c", "fce_sem_corpus_doc_count", {"fce", "sem", "corpus", "doc", "count"}, 5},
    {"src/semantic/semantic.c", "fce_sem_corpus_token_count", {"fce", "sem", "corpus", "token", "count"}, 5},
    {"src/semantic/semantic.c", "fce_sem_corpus_token_at", {"fce", "sem", "corpus", "token", "at"}, 5},
    {"src/semantic/semantic.c", "fce_sem_corpus_free", {"fce", "sem", "corpus", "free"}, 4},
    {"src/semantic/semantic.c", "fce_sem_combined_score", {"fce", "sem", "combined", "score"}, 4},
    {"src/semantic/semantic.c", "fce_sem_proximity", {"fce", "sem", "proximity"}, 3},
    {"src/semantic/semantic.c", "fce_sem_rank", {"fce", "sem", "rank"}, 3},
    {"src/semantic/semantic.c", "fce_sem_search", {"fce", "sem", "search"}, 3},
    {"src/semantic/semantic.c", "fce_sem_simple_score", {"fce", "sem", "simple", "score"}, 4},
    {"src/semantic/semantic.c", "fce_sem_simple_rank", {"fce", "sem", "simple", "rank"}, 4},
    {"src/semantic/semantic.c", "fce_sem_simple_search", {"fce", "sem", "simple", "search"}, 4},
    {"src/semantic/semantic.c", "fce_sem_diffuse", {"fce", "sem", "diffuse"}, 3},
    {"src/semantic/semantic.c", "fce_sparse_tfidf_cosine", {"sparse", "tfidf", "cosine"}, 3},
    {"src/semantic/semantic.c", "fce_small_cosine", {"small", "cosine"}, 2},
    {"src/semantic/semantic.c", "fce_ranked_cmp_desc", {"ranked", "cmp"}, 2},
    {"src/semantic/semantic.c", "fce_flush_token", {"flush", "token"}, 2},

    /* foundation/hash_table.c */
    {"src/foundation/hash_table.c", "fce_ht_create", {"fce", "ht", "create"}, 3},
    {"src/foundation/hash_table.c", "fce_ht_free", {"fce", "ht", "free"}, 3},
    {"src/foundation/hash_table.c", "fce_ht_set", {"fce", "ht", "set"}, 3},
    {"src/foundation/hash_table.c", "fce_ht_get", {"fce", "ht", "get"}, 3},
    {"src/foundation/hash_table.c", "fce_ht_has", {"fce", "ht", "has"}, 3},
    {"src/foundation/hash_table.c", "fce_ht_delete", {"fce", "ht", "delete"}, 3},
    {"src/foundation/hash_table.c", "fce_ht_count", {"fce", "ht", "count"}, 3},
    {"src/foundation/hash_table.c", "fce_ht_foreach", {"fce", "ht", "foreach"}, 3},
    {"src/foundation/hash_table.c", "fce_ht_clear", {"fce", "ht", "clear"}, 3},
    {"src/foundation/hash_table.c", "ht_resize", {"ht", "resize"}, 2},

    /* foundation/platform.c */
    {"src/foundation/platform.c", "fce_mmap_read", {"fce", "mmap", "read"}, 3},
    {"src/foundation/platform.c", "fce_munmap", {"fce", "munmap"}, 2},
    {"src/foundation/platform.c", "fce_now_ns", {"fce", "now", "ns"}, 3},
    {"src/foundation/platform.c", "fce_now_ms", {"fce", "now", "ms"}, 3},
    {"src/foundation/platform.c", "fce_nprocs", {"fce", "nprocs"}, 2},
    {"src/foundation/platform.c", "fce_file_exists", {"fce", "file", "exists"}, 3},
    {"src/foundation/platform.c", "fce_is_dir", {"fce", "is", "dir"}, 3},
    {"src/foundation/platform.c", "fce_file_size", {"fce", "file", "size"}, 3},
    {"src/foundation/platform.c", "fce_normalize_path_sep", {"fce", "normalize", "path", "sep"}, 4},
    {"src/foundation/platform.c", "fce_safe_getenv", {"fce", "safe", "getenv"}, 3},
    {"src/foundation/platform.c", "fce_get_home_dir", {"fce", "get", "home", "dir"}, 4},
    {"src/foundation/platform.c", "fce_app_config_dir", {"fce", "app", "config", "dir"}, 4},
    {"src/foundation/platform.c", "fce_app_local_dir", {"fce", "app", "local", "dir"}, 4},
    {"src/foundation/platform.c", "fce_resolve_cache_dir", {"fce", "resolve", "cache", "dir"}, 4},

    /* foundation/system_info.c */
    {"src/foundation/system_info.c", "fce_system_info", {"fce", "system", "info"}, 3},
    {"src/foundation/system_info.c", "fce_default_worker_count", {"fce", "default", "worker", "count"}, 4},

    /* foundation/log.c */
    {"src/foundation/log.c", "fce_log_set_sink", {"fce", "log", "set", "sink"}, 4},
    {"src/foundation/log.c", "fce_log_set_level", {"fce", "log", "set", "level"}, 4},
    {"src/foundation/log.c", "fce_log_get_level", {"fce", "log", "get", "level"}, 4},
    {"src/foundation/log.c", "fce_log", {"fce", "log"}, 2},

    /* foundation/profile.c */
    {"src/foundation/profile.c", "fce_profile_init", {"fce", "profile", "init"}, 3},
    {"src/foundation/profile.c", "fce_profile_enable", {"fce", "profile", "enable"}, 3},
    {"src/foundation/profile.c", "fce_profile_now", {"fce", "profile", "now"}, 3},
    {"src/foundation/profile.c", "fce_profile_log_elapsed", {"fce", "profile", "log", "elapsed"}, 4},

    /* pipeline/worker_pool.c */
    {"src/pipeline/worker_pool.c", "fce_parallel_for", {"fce", "parallel", "for"}, 3},
    {"src/pipeline/worker_pool.c", "pthread_worker", {"pthread", "worker"}, 2},
    {"src/pipeline/worker_pool.c", "run_serial", {"run", "serial"}, 2},
    {"src/pipeline/worker_pool.c", "run_pthreads", {"run", "pthreads"}, 2},

    /* version.c */
    {"src/version.c", "fce_version", {"fce", "version"}, 2},
};

#define CORPUS_SIZE (sizeof(corpus) / sizeof(corpus[0]))

static double ms_since(struct timespec start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start.tv_sec) * 1000.0 +
           (now.tv_nsec - start.tv_nsec) / 1e6;
}

int main(void) {
    struct timespec t0;
    int iterations = 100;

    printf("fast-code-embed C benchmark\n");
    printf("================================\n");
    printf("Corpus: %d functions from this project's source\n\n", (int)CORPUS_SIZE);

    /* ── 1. Init ──────────────────────────────────────────────── */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fce_sem_ensure_ready();
    double init_ms = ms_since(t0);
    printf(" init (ensure_ready) %8.2f ms\n", init_ms);

    /* ── 2. Tokenize (full names, exercises camel/snake splits + abbrev expansion) ── */
    char *tok_buf[64];
    int ntok;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iterations; i++) {
        for (size_t c = 0; c < CORPUS_SIZE; c++) {
            ntok = fce_sem_tokenize(corpus[c].full_name, tok_buf, 64);
            for (int t = 0; t < ntok; t++) free(tok_buf[t]);
        }
    }
    double tokenize_ms = ms_since(t0);
    printf(" tokenize %d func names × %d %8.2f ms (%.1f µs/call)\n",
           (int)CORPUS_SIZE, iterations, tokenize_ms,
           tokenize_ms * 1000.0 / (CORPUS_SIZE * iterations));

    /* ── 2b. Tokenize batch API ──────────────────────────────── */
    const char *batch_names[69];
    for (size_t c = 0; c < CORPUS_SIZE; c++) batch_names[c] = corpus[c].full_name;
    char *tok_buf_batch[69 * 64];
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iterations; i++) {
        int batch_counts[69];
        fce_sem_tokenize_batch(batch_names, (int)CORPUS_SIZE, tok_buf_batch, batch_counts, 64);
        for (size_t c = 0; c < CORPUS_SIZE; c++) {
            for (int t = 0; t < batch_counts[c]; t++) free(tok_buf_batch[c * 64 + t]);
        }
    }
    double batch_tok_ms = ms_since(t0);
    printf(" tokenize_batch %d × %d %8.2f ms (%.1f µs/call)\n",
           (int)CORPUS_SIZE, iterations, batch_tok_ms,
           batch_tok_ms * 1000.0 / (CORPUS_SIZE * iterations));

    /* ── 3. Corpus build (single doc) ────────────────────────── */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fce_sem_corpus_t *corp_single = fce_sem_corpus_new();
    for (size_t c = 0; c < CORPUS_SIZE; c++) {
        fce_sem_corpus_add_doc(corp_single, corpus[c].tokens, corpus[c].ntokens);
    }
    fce_sem_corpus_finalize(corp_single);
    double single_ms = ms_since(t0);
    printf(" corpus build (single doc) %8.2f ms\n", single_ms);

    /* ── 4. Corpus build (batch) ─────────────────────────────── */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fce_sem_corpus_t *corp_batch = fce_sem_corpus_new();

    int max_tok = 0;
    for (size_t c = 0; c < CORPUS_SIZE; c++) {
        if (corpus[c].ntokens > max_tok) max_tok = corpus[c].ntokens;
    }

    char **all_tokens = (char **)malloc(CORPUS_SIZE * max_tok * sizeof(char *));
    int *token_counts = (int *)malloc(CORPUS_SIZE * sizeof(int));
    for (size_t c = 0; c < CORPUS_SIZE; c++) {
        token_counts[c] = corpus[c].ntokens;
        for (int t = 0; t < corpus[c].ntokens; t++) {
            all_tokens[c * max_tok + t] = (char *)corpus[c].tokens[t];
        }
    }
    fce_sem_corpus_add_docs_batch(corp_batch, all_tokens, token_counts, CORPUS_SIZE, max_tok, NULL);
    fce_sem_corpus_finalize(corp_batch);
    double batch_ms = ms_since(t0);
    printf(" corpus build (batch) %8.2f ms (%.1fx faster)\n",
           batch_ms, single_ms / batch_ms);

    free(all_tokens);
    free(token_counts);

    /* ── 5. IDF lookup ───────────────────────────────────────── */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iterations; i++) {
        for (size_t c = 0; c < CORPUS_SIZE; c++) {
            fce_sem_corpus_idf(corp_batch, corpus[c].tokens[0]);
        }
    }
    double idf_ms = ms_since(t0);
    printf(" IDF lookup %d × %d %8.2f ms (%.1f µs/call)\n",
           (int)CORPUS_SIZE, iterations, idf_ms,
           idf_ms * 1000.0 / (CORPUS_SIZE * iterations));

    /* ── 6. RI vector access ─────────────────────────────────── */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iterations; i++) {
        for (size_t c = 0; c < CORPUS_SIZE; c++) {
            fce_sem_corpus_ri_vec(corp_batch, corpus[c].tokens[0]);
        }
    }
    double ri_ms = ms_since(t0);
    printf(" RI vec access %d × %d %8.2f ms (%.1f µs/call)\n",
           (int)CORPUS_SIZE, iterations, ri_ms,
           ri_ms * 1000.0 / (CORPUS_SIZE * iterations));

    /* ── 7. Simple scoring (all pairs) ───────────────────────── */
    int npairs = (int)CORPUS_SIZE * (int)CORPUS_SIZE;
    fce_sem_func_t *funcs = (fce_sem_func_t *)calloc(CORPUS_SIZE, sizeof(fce_sem_func_t));
    for (size_t c = 0; c < CORPUS_SIZE; c++) {
        funcs[c].file_path = corpus[c].path;
        funcs[c].tfidf_len = corpus[c].ntokens;
        int *idx = (int *)malloc(corpus[c].ntokens * sizeof(int));
        float *w = (float *)malloc(corpus[c].ntokens * sizeof(float));
        for (int t = 0; t < corpus[c].ntokens; t++) {
            idx[t] = t;
            w[t] = fce_sem_corpus_idf(corp_batch, corpus[c].tokens[t]);
        }
        funcs[c].tfidf_indices = idx;
        funcs[c].tfidf_weights = w;
        const fce_sem_vec_t *rv = fce_sem_corpus_ri_vec(corp_batch, corpus[c].tokens[0]);
        if (rv) funcs[c].ri_vec = *rv;
    }

    clock_gettime(CLOCK_MONOTONIC, &t0);
    float score_sum = 0;
    for (int i = 0; i < iterations; i++) {
        for (size_t a = 0; a < CORPUS_SIZE; a++) {
            for (size_t b = a + 1; b < CORPUS_SIZE; b++) {
                score_sum += fce_sem_simple_score(&funcs[a], &funcs[b]);
            }
        }
    }
    double score_ms = ms_since(t0);
    int unique_pairs = npairs / 2;
    printf(" simple_score %d pairs × %d %8.2f ms (%.1f µs/pair)\n",
           unique_pairs, iterations, score_ms,
           score_ms * 1000.0 / (unique_pairs * iterations));

    /* ── 8. Ranking ──────────────────────────────────────────── */
    fce_sem_ranked_t results[64];
    uint32_t count;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iterations; i++) {
        fce_sem_simple_rank(&funcs[0], funcs, (uint32_t)CORPUS_SIZE, 10, results, &count);
    }
    double rank_ms = ms_since(t0);
    printf(" simple_rank (top 10) × %d %8.2f ms (%.1f µs/call)\n",
           iterations, rank_ms, rank_ms * 1000.0 / iterations);

    /* ── Summary ─────────────────────────────────────────────── */
    printf("\n");
    printf(" Corpus size: %d functions\n", (int)CORPUS_SIZE);
    printf(" Vocabulary: %d tokens\n", fce_sem_corpus_token_count(corp_batch));
    printf(" Top match: results[0] = idx %d, score %.4f\n",
           results[0].index, results[0].score);
    printf(" score_sum: %.6f (prevents optimization)\n", score_sum);

    /* Cleanup */
    for (size_t c = 0; c < CORPUS_SIZE; c++) {
        free(funcs[c].tfidf_indices);
        free(funcs[c].tfidf_weights);
    }
    free(funcs);
    fce_sem_corpus_free(corp_single);
    fce_sem_corpus_free(corp_batch);

    return 0;
}
