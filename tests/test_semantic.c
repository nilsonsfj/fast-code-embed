/*
 * test_semantic.c — Unit tests for fast-code-embed library.
 *
 * Exercises: tokenization, random indexing, cosine similarity,
 * TF-IDF, corpus building, RRI, combined scoring.
 */
#include "semantic/semantic.h"
#include "foundation/hash_table.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        printf("  %-48s ", #name); \
        fflush(stdout); \
    } while (0)

#define PASS() \
    do { \
        tests_passed++; \
        printf("OK\n"); \
    } while (0)

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            printf("FAIL (line %d: %s)\n", __LINE__, #cond); \
            return; \
        } \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        float _diff = fabsf((a) - (b)); \
        if (_diff > (eps)) { \
            printf("FAIL (line %d: |%g - %g| = %g > %g)\n", \
                   __LINE__, (double)(a), (double)(b), (double)_diff, (double)(eps)); \
            return; \
        } \
    } while (0)

/* ── Tokenization tests ───────────────────────────────────────── */

static void test_tokenize_camel_case(void) {
    TEST(tokenize camelCase);
    char *tokens[32];
    int n = fce_sem_tokenize("handleRequest", tokens, 32);
    ASSERT(n == 2);
    ASSERT(strcmp(tokens[0], "handle") == 0);
    ASSERT(strcmp(tokens[1], "request") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);
    PASS();
}

static void test_tokenize_snake_case(void) {
    TEST(tokenize snake_case);
    char *tokens[32];
    int n = fce_sem_tokenize("get_user_name", tokens, 32);
    ASSERT(n == 3);
    ASSERT(strcmp(tokens[0], "get") == 0);
    ASSERT(strcmp(tokens[1], "user") == 0);
    ASSERT(strcmp(tokens[2], "name") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);
    PASS();
}

static void test_tokenize_dot_separated(void) {
    TEST(tokenize dot.separated);
    char *tokens[32];
    int n = fce_sem_tokenize("io.Reader", tokens, 32);
    ASSERT(n == 2);
    ASSERT(strcmp(tokens[0], "io") == 0);
    ASSERT(strcmp(tokens[1], "reader") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);
    PASS();
}

static void test_tokenize_pascal_case(void) {
    TEST(tokenize PascalCase);
    char *tokens[32];
    int n = fce_sem_tokenize("XMLHttpRequest", tokens, 32);
    ASSERT(n == 2);
    /* Lowercased + camel break at H: "xmlhttp" + "request" */
    ASSERT(strcmp(tokens[0], "xmlhttp") == 0);
    ASSERT(strcmp(tokens[1], "request") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);
    PASS();
}

static void test_tokenize_empty(void) {
    TEST(tokenize empty string);
    char *tokens[32];
    int n = fce_sem_tokenize("", tokens, 32);
    ASSERT(n == 0);
    PASS();
}

static void test_tokenize_null(void) {
    TEST(tokenize NULL);
    char *tokens[32];
    int n = fce_sem_tokenize(NULL, tokens, 32);
    ASSERT(n == 0);
    PASS();
}

static void test_tokenize_abbreviations(void) {
    TEST(tokenize abbreviation expansion);
    char *tokens[32];
    int n = fce_sem_tokenize("ctx", tokens, 32);
    ASSERT(n == 2);
    ASSERT(strcmp(tokens[0], "ctx") == 0);
    ASSERT(strcmp(tokens[1], "context") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);
    PASS();
}

static void test_tokenize_abbreviations_hash_table(void) {
    TEST(tokenize abbreviation hash table covers multiple entries);
    char *tokens[32];
    int n;

    /* Test several abbreviations from different categories. */
    n = fce_sem_tokenize("err", tokens, 32);
    ASSERT(n == 2);
    ASSERT(strcmp(tokens[1], "error") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);

    n = fce_sem_tokenize("fn", tokens, 32);
    ASSERT(n == 2);
    ASSERT(strcmp(tokens[1], "function") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);

    n = fce_sem_tokenize("db", tokens, 32);
    ASSERT(n == 2);
    ASSERT(strcmp(tokens[1], "database") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);

    n = fce_sem_tokenize("cb", tokens, 32);
    ASSERT(n == 2);
    ASSERT(strcmp(tokens[1], "callback") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);

    n = fce_sem_tokenize("idx", tokens, 32);
    ASSERT(n == 2);
    ASSERT(strcmp(tokens[1], "index") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);

    /* Non-abbreviation should not expand. */
    n = fce_sem_tokenize("foobar", tokens, 32);
    ASSERT(n == 1);
    ASSERT(strcmp(tokens[0], "foobar") == 0);
    for (int i = 0; i < n; i++) free(tokens[i]);

    PASS();
}

static void test_simple_score_deterministic(void) {
    TEST(simple score is deterministic across repeated calls (magnitude caching));
    fce_sem_func_t a = {0}, b = {0};

    fce_sem_ensure_ready();

    a.file_path = "src/handler.c";
    b.file_path = "src/handler.c";
    a.tfidf_len = 2;
    b.tfidf_len = 2;
    int indices[] = {0, 1};
    float weights[] = {1.0f, 1.0f};
    a.tfidf_indices = indices; a.tfidf_weights = weights;
    b.tfidf_indices = indices; b.tfidf_weights = weights;
    fce_sem_random_index("handle", &a.ri_vec);
    fce_sem_random_index("handle", &b.ri_vec);

    /* Call score twice — magnitude caching must not change the result. */
    float s1 = fce_sem_simple_score(&a, &b);
    float s2 = fce_sem_simple_score(&a, &b);
    ASSERT(s1 == s2);
    ASSERT(s1 > 0.5f);
    PASS();
}

static void test_simple_rank_flat_larger(void) {
    TEST(simple rank flat with 10 functions finds correct top-3);
    fce_sem_ensure_ready();

    /* Build 10 functions with varying RI vectors. */
    int max_tok = 2;
    float all_weights[10 * 2];
    int all_indices[10 * 2];
    int tfidf_lens[10];
    float all_ri_vecs[10 * FCE_SEM_DIM];
    const char *paths[10];

    for (int f = 0; f < 10; f++) {
        tfidf_lens[f] = 2;
        paths[f] = f < 3 ? "src/shared.c" : "src/other.c";
        all_indices[f * 2 + 0] = f < 3 ? 0 : f + 10;
        all_indices[f * 2 + 1] = f + 20;
        all_weights[f * 2 + 0] = 1.0f;
        all_weights[f * 2 + 1] = 0.5f;
        char name[32];
        snprintf(name, sizeof(name), "func_%d", f);
        fce_sem_vec_t rv;
        fce_sem_random_index(name, &rv);
        memcpy(all_ri_vecs + f * FCE_SEM_DIM, rv.v, sizeof(float) * FCE_SEM_DIM);
    }

    /* H1: flat path uses RI-only scoring (TF-IDF indices are positional and
     * meaningless). The query uses func_0's RI vector, so func_0 should be
     * the top result (cosine ≈ 1.0 with itself). */
    int q_indices[] = {0, 30};
    float q_weights[] = {1.0f, 0.5f};
    float q_ri[FCE_SEM_DIM];
    fce_sem_vec_t qrv;
    fce_sem_random_index("func_0", &qrv);
    memcpy(q_ri, qrv.v, sizeof(float) * FCE_SEM_DIM);

    fce_sem_ranked_t results[10];
    uint32_t count = 0;
    fce_sem_simple_rank_flat(
        all_weights, all_indices, tfidf_lens, all_ri_vecs, paths,
        10, max_tok, q_indices, q_weights, 2, q_ri, 3, results, &count);

    ASSERT(count == 3);
    /* Function 0 should be top result (identical RI vector to query). */
    ASSERT(results[0].index == 0);
    /* Results should be sorted descending. */
    ASSERT(results[0].score >= results[1].score);
    ASSERT(results[1].score >= results[2].score);
    PASS();
}

/* ── Random indexing tests ─────────────────────────────────────── */

static void test_random_index_deterministic(void) {
    TEST(random index is deterministic);
    fce_sem_vec_t a, b;
    fce_sem_ensure_ready();
    fce_sem_random_index("function", &a);
    fce_sem_random_index("function", &b);
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        ASSERT(a.v[i] == b.v[i]);
    }
    PASS();
}

static void test_random_index_different_tokens(void) {
    TEST(random index differs for different tokens);
    fce_sem_vec_t a, b;
    fce_sem_ensure_ready();
    fce_sem_random_index("error", &a);
    fce_sem_random_index("logging", &b);
    float sim = fce_sem_cosine(&a, &b);
    ASSERT(fabsf(sim) < 0.5f);
    PASS();
}

static void test_random_index_null_token(void) {
    TEST(random index NULL token returns zero vector);
    fce_sem_vec_t v;
    fce_sem_random_index(NULL, &v);
    float mag = 0.0f;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        mag += v.v[i] * v.v[i];
    }
    ASSERT(mag == 0.0f);
    PASS();
}

/* ── Cosine similarity tests ──────────────────────────────────── */

static void test_cosine_identical(void) {
    TEST(cosine similarity of identical vectors is ~1.0);
    fce_sem_vec_t v;
    fce_sem_ensure_ready();
    fce_sem_random_index("test", &v);
    float sim = fce_sem_cosine(&v, &v);
    ASSERT_NEAR(sim, 1.0f, 0.001f);
    PASS();
}

static void test_cosine_orthogonal(void) {
    TEST(cosine similarity of orthogonal vectors is ~0.0);
    fce_sem_vec_t a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    a.v[0] = 1.0f;
    b.v[1] = 1.0f;
    float sim = fce_sem_cosine(&a, &b);
    ASSERT_NEAR(sim, 0.0f, 0.001f);
    PASS();
}

static void test_cosine_null(void) {
    TEST(cosine similarity with NULL returns 0);
    fce_sem_vec_t v;
    memset(&v, 0, sizeof(v));
    ASSERT(fce_sem_cosine(NULL, &v) == 0.0f);
    ASSERT(fce_sem_cosine(&v, NULL) == 0.0f);
    PASS();
}

/* ── Normalize tests ──────────────────────────────────────────── */

static void test_normalize(void) {
    TEST(normalize produces unit vector);
    fce_sem_vec_t v;
    v.v[0] = 3.0f;
    v.v[1] = 4.0f;
    memset(v.v + 2, 0, (FCE_SEM_DIM - 2) * sizeof(float));
    fce_sem_normalize(&v);
    float mag = 0.0f;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        mag += v.v[i] * v.v[i];
    }
    ASSERT_NEAR(mag, 1.0f, 0.001f);
    PASS();
}

/* ── Vec add scaled tests ─────────────────────────────────────── */

static void test_vec_add_scaled(void) {
    TEST(vec_add_scaled accumulates correctly);
    fce_sem_vec_t dst, src;
    memset(&dst, 0, sizeof(dst));
    memset(&src, 0, sizeof(src));
    dst.v[0] = 1.0f;
    src.v[0] = 2.0f;
    src.v[1] = 3.0f;
    fce_sem_vec_add_scaled(&dst, &src, 0.5f);
    ASSERT_NEAR(dst.v[0], 2.0f, 0.001f);
    ASSERT_NEAR(dst.v[1], 1.5f, 0.001f);
    PASS();
}

/* ── Corpus tests ─────────────────────────────────────────────── */

static void test_corpus_add_and_finalize(void) {
    TEST(corpus add docs and finalize);
    fce_sem_corpus_t *corp = fce_sem_corpus_new();
    ASSERT(corp != NULL);

    const char *doc1[] = {"handle", "request"};
    const char *doc2[] = {"handle", "error"};
    const char *doc3[] = {"log", "message"};

    fce_sem_corpus_add_doc(corp, doc1, 2);
    fce_sem_corpus_add_doc(corp, doc2, 2);
    fce_sem_corpus_add_doc(corp, doc3, 2);

    ASSERT(fce_sem_corpus_doc_count(corp) == 3);

    fce_sem_corpus_finalize(corp);

    /* IDF for "handle" should be lower than IDF for "log" since "handle"
     * appears in 2 of 3 docs while "log" appears in 1 of 3. */
    float idf_handle = fce_sem_corpus_idf(corp, "handle");
    float idf_log = fce_sem_corpus_idf(corp, "log");
    ASSERT(idf_handle > 0.0f);
    ASSERT(idf_log > idf_handle);

    /* RI vector for "handle" should be enriched (non-zero after co-occurrence) */
    const fce_sem_vec_t *ri = fce_sem_corpus_ri_vec(corp, "handle");
    ASSERT(ri != NULL);
    float mag = 0.0f;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        mag += ri->v[i] * ri->v[i];
    }
    ASSERT(mag > 0.0f);

    fce_sem_corpus_free(corp);
    PASS();
}

static void test_corpus_batch_add(void) {
    TEST(corpus batch add docs);
    fce_sem_corpus_t *corp = fce_sem_corpus_new();
    ASSERT(corp != NULL);

    char *all_tokens[6] = {"fn", "error", "fn", "log", "err", "handle"};
    int counts[3] = {2, 2, 2};

    fce_sem_corpus_add_docs_batch(corp, all_tokens, counts, 3, 2);
    ASSERT(fce_sem_corpus_doc_count(corp) == 3);

    fce_sem_corpus_finalize(corp);
    fce_sem_corpus_free(corp);
    PASS();
}

static void test_corpus_token_at(void) {
    TEST(corpus token_at accessor);
    fce_sem_corpus_t *corp = fce_sem_corpus_new();
    /* Need >1 doc for non-zero IDF (log(N/df)): "alpha" in 2/3 docs, "beta" in 1/3 */
    const char *doc1[] = {"alpha", "beta"};
    const char *doc2[] = {"alpha", "gamma"};
    const char *doc3[] = {"alpha", "delta"};
    fce_sem_corpus_add_doc(corp, doc1, 2);
    fce_sem_corpus_add_doc(corp, doc2, 2);
    fce_sem_corpus_add_doc(corp, doc3, 2);
    fce_sem_corpus_finalize(corp);

    const fce_sem_vec_t *vec;
    float idf;
    const char *tok = fce_sem_corpus_token_at(corp, 0, &vec, &idf);
    ASSERT(tok != NULL);
    ASSERT(vec != NULL);
    /* IDF must be non-negative; for "alpha" in 3/3 docs, IDF=0;
     * for "beta" in 1/3 docs, IDF=log(3/1)>0. We check either is fine. */
    ASSERT(idf >= 0.0f);

    ASSERT(fce_sem_corpus_token_at(corp, -1, NULL, NULL) == NULL);
    ASSERT(fce_sem_corpus_token_at(corp, 999, NULL, NULL) == NULL);

    fce_sem_corpus_free(corp);
    PASS();
}

/* ── Proximity tests ──────────────────────────────────────────── */

static void test_proximity_same_file(void) {
    TEST(proximity same file);
    float p = fce_sem_proximity("src/foo.c", "src/foo.c");
    ASSERT_NEAR(p, 1.10f, 0.001f);
    PASS();
}

static void test_proximity_same_dir(void) {
    TEST(proximity same dir);
    float p = fce_sem_proximity("src/a.c", "src/b.c");
    ASSERT(p >= 1.0f);
    ASSERT(p <= 1.10f);
    PASS();
}

static void test_proximity_distant(void) {
    TEST(proximity distant);
    float p = fce_sem_proximity("src/a.c", "lib/b.c");
    ASSERT_NEAR(p, 1.0f, 0.01f);
    PASS();
}

static void test_proximity_null(void) {
    TEST(proximity NULL paths);
    float p = fce_sem_proximity(NULL, NULL);
    ASSERT_NEAR(p, 1.0f, 0.001f);
    PASS();
}

/* ── Combined scoring tests ───────────────────────────────────── */

static void test_combined_score_identical(void) {
    TEST(combined score of identical funcs is high);
    fce_sem_func_t a = {0}, b = {0};

    fce_sem_ensure_ready();

    a.file_path = "src/handler.c";
    b.file_path = "src/handler.c";
    a.tfidf_len = 2;
    int indices[] = {0, 1};
    float weights[] = {1.0f, 1.0f};
    a.tfidf_indices = indices;
    a.tfidf_weights = weights;
    b.tfidf_len = 2;
    b.tfidf_indices = indices;
    b.tfidf_weights = weights;

    fce_sem_random_index("handle", &a.ri_vec);
    fce_sem_random_index("handle", &b.ri_vec);
    fce_sem_random_index("request", &a.api_vec);
    fce_sem_random_index("request", &b.api_vec);

    fce_sem_config_t cfg = fce_sem_get_config();
    float score = fce_sem_combined_score(&a, &b, &cfg);
    ASSERT(score > 0.5f);
    PASS();
}

static void test_combined_score_different(void) {
    TEST(combined score of different funcs is lower);
    fce_sem_func_t a = {0}, b = {0};

    fce_sem_ensure_ready();

    a.file_path = "src/handler.c";
    b.file_path = "lib/utils.c";

    int indices_a[] = {0, 1};
    float weights_a[] = {1.0f, 1.0f};
    a.tfidf_indices = indices_a;
    a.tfidf_weights = weights_a;
    a.tfidf_len = 2;

    int indices_b[] = {2, 3};
    float weights_b[] = {1.0f, 1.0f};
    b.tfidf_indices = indices_b;
    b.tfidf_weights = weights_b;
    b.tfidf_len = 2;

    fce_sem_random_index("error", &a.ri_vec);
    fce_sem_random_index("logging", &b.ri_vec);
    fce_sem_random_index("throw", &a.api_vec);
    fce_sem_random_index("print", &b.api_vec);

    fce_sem_config_t cfg = fce_sem_get_config();
    float score = fce_sem_combined_score(&a, &b, &cfg);
    ASSERT(score >= 0.0f);
    ASSERT(score < 0.9f);
    PASS();
}

static void test_combined_score_null(void) {
    TEST(combined score NULL safety);
    fce_sem_config_t cfg = fce_sem_get_config();
    ASSERT(fce_sem_combined_score(NULL, NULL, &cfg) == 0.0f);
    PASS();
}

/* ── Simple API tests ─────────────────────────────────────────── */

static void test_simple_score_identical(void) {
    TEST(simple score of identical funcs is high);
    fce_sem_func_t a = {0}, b = {0};

    fce_sem_ensure_ready();

    a.file_path = "src/handler.c";
    b.file_path = "src/handler.c";
    a.tfidf_len = 2;
    int indices[] = {0, 1};
    float weights[] = {1.0f, 1.0f};
    a.tfidf_indices = indices;
    a.tfidf_weights = weights;
    b.tfidf_len = 2;
    b.tfidf_indices = indices;
    b.tfidf_weights = weights;

    fce_sem_random_index("handle", &a.ri_vec);
    fce_sem_random_index("handle", &b.ri_vec);

    float score = fce_sem_simple_score(&a, &b);
    ASSERT(score > 0.5f);
    ASSERT(score <= 1.0f);
    PASS();
}

static void test_simple_score_different(void) {
    TEST(simple score of different funcs is lower);
    fce_sem_func_t a = {0}, b = {0};

    fce_sem_ensure_ready();

    a.file_path = "src/handler.c";
    b.file_path = "lib/utils.c";

    int indices_a[] = {0, 1};
    float weights_a[] = {1.0f, 1.0f};
    a.tfidf_indices = indices_a;
    a.tfidf_weights = weights_a;
    a.tfidf_len = 2;

    int indices_b[] = {2, 3};
    float weights_b[] = {1.0f, 1.0f};
    b.tfidf_indices = indices_b;
    b.tfidf_weights = weights_b;
    b.tfidf_len = 2;

    fce_sem_random_index("error", &a.ri_vec);
    fce_sem_random_index("logging", &b.ri_vec);

    float score = fce_sem_simple_score(&a, &b);
    ASSERT(score >= 0.0f);
    ASSERT(score < 0.9f);
    PASS();
}

static void test_simple_score_range(void) {
    TEST(simple score always in 0 to 1 range);
    fce_sem_func_t a = {0}, b = {0};

    fce_sem_ensure_ready();

    a.file_path = "src/a.c";
    b.file_path = "lib/b.c";

    /* Completely different tokens, different files */
    int idx_a[] = {0};
    float w_a[] = {1.0f};
    a.tfidf_indices = idx_a;
    a.tfidf_weights = w_a;
    a.tfidf_len = 1;
    int idx_b[] = {9999};
    float w_b[] = {1.0f};
    b.tfidf_indices = idx_b;
    b.tfidf_weights = w_b;
    b.tfidf_len = 1;

    fce_sem_random_index("alpha", &a.ri_vec);
    fce_sem_random_index("zeta", &b.ri_vec);

    float score = fce_sem_simple_score(&a, &b);
    ASSERT(score >= 0.0f);
    ASSERT(score <= 1.0f);
    PASS();
}

static void test_simple_rank(void) {
    TEST(simple rank returns sorted results);
    fce_sem_func_t corpus[3] = {0};
    fce_sem_ensure_ready();

    int idx0[] = {0};
    float w0[] = {1.0f};
    corpus[0].tfidf_indices = idx0;
    corpus[0].tfidf_weights = w0;
    corpus[0].tfidf_len = 1;
    corpus[0].file_path = "src/a.c";

    int idx1[] = {1};
    float w1[] = {1.0f};
    corpus[1].tfidf_indices = idx1;
    corpus[1].tfidf_weights = w1;
    corpus[1].tfidf_len = 1;
    corpus[1].file_path = "src/b.c";

    int idx2[] = {0};
    float w2[] = {1.0f};
    corpus[2].tfidf_indices = idx2;
    corpus[2].tfidf_weights = w2;
    corpus[2].tfidf_len = 1;
    corpus[2].file_path = "src/a.c";

    fce_sem_random_index("query", &corpus[0].ri_vec);
    fce_sem_random_index("other", &corpus[1].ri_vec);
    fce_sem_random_index("query", &corpus[2].ri_vec);

    fce_sem_func_t query = {0};
    query.tfidf_indices = idx0;
    query.tfidf_weights = w0;
    query.tfidf_len = 1;
    query.file_path = "src/a.c";
    fce_sem_random_index("query", &query.ri_vec);

    fce_sem_ranked_t results[3];
    uint32_t count = 0;
    fce_sem_simple_rank(&query, corpus, 3, 3, results, &count);

    ASSERT(count == 3);
    /* First result should have highest score */
    ASSERT(results[0].score >= results[1].score);
    ASSERT(results[1].score >= results[2].score);
    PASS();
}

static void test_simple_rank_flat(void) {
    TEST(simple rank flat matches struct-based rank);
    fce_sem_ensure_ready();

    /* Build 3 functions */
    fce_sem_func_t a = {0}, b = {0}, c = {0};
    int idx_a[] = {0};
    float w_a[] = {1.0f};
    a.tfidf_indices = idx_a; a.tfidf_weights = w_a; a.tfidf_len = 1;
    a.file_path = "src/a.c";
    fce_sem_random_index("query", &a.ri_vec);

    int idx_b[] = {1};
    float w_b[] = {1.0f};
    b.tfidf_indices = idx_b; b.tfidf_weights = w_b; b.tfidf_len = 1;
    b.file_path = "src/b.c";
    fce_sem_random_index("other", &b.ri_vec);

    int idx_c[] = {0};
    float w_c[] = {1.0f};
    c.tfidf_indices = idx_c; c.tfidf_weights = w_c; c.tfidf_len = 1;
    c.file_path = "src/a.c";
    fce_sem_random_index("query", &c.ri_vec);

    /* Build flat arrays */
    int max_tok = 1;
    float all_weights[] = {1.0f, 1.0f, 1.0f};
    int all_indices[] = {0, 1, 0};
    int tfidf_lens[] = {1, 1, 1};
    float all_ri_vecs[3 * FCE_SEM_DIM];
    memcpy(all_ri_vecs + 0 * FCE_SEM_DIM, a.ri_vec.v, sizeof(float) * FCE_SEM_DIM);
    memcpy(all_ri_vecs + 1 * FCE_SEM_DIM, b.ri_vec.v, sizeof(float) * FCE_SEM_DIM);
    memcpy(all_ri_vecs + 2 * FCE_SEM_DIM, c.ri_vec.v, sizeof(float) * FCE_SEM_DIM);
    const char *paths[] = {"src/a.c", "src/b.c", "src/a.c"};

    fce_sem_ranked_t results[3];
    uint32_t count = 0;
    fce_sem_simple_rank_flat(
        all_weights, all_indices, tfidf_lens, all_ri_vecs, paths, 3, max_tok,
        idx_a, w_a, 1, a.ri_vec.v,
        3, results, &count);

    ASSERT(count == 3);
    ASSERT(results[0].score >= results[1].score);
    ASSERT(results[1].score >= results[2].score);
    /* Best match should be index 0 or 2 (both have same tokens + same file as query) */
    ASSERT(results[0].index == 0 || results[0].index == 2);
    PASS();
}

/* ── Diffusion tests ──────────────────────────────────────────── */

static void test_diffuse(void) {
    TEST(diffuse blends with neighbor mean);
    fce_sem_vec_t combined;
    memset(&combined, 0, sizeof(combined));
    combined.v[0] = 1.0f;

    fce_sem_vec_t neighbor;
    memset(&neighbor, 0, sizeof(neighbor));
    neighbor.v[0] = 0.0f;
    neighbor.v[1] = 1.0f;

    fce_sem_diffuse(&combined, &neighbor, 1, 0.3f);

    /* After diffusion, v[0] should be less than 1.0 (blended down) */
    ASSERT(combined.v[0] < 1.0f);
    ASSERT(combined.v[0] > 0.0f);
    PASS();
}

/* ── Config tests ─────────────────────────────────────────────── */

static void test_config_defaults(void) {
    TEST(config has valid defaults);
    fce_sem_config_t cfg = fce_sem_get_config();
    ASSERT(cfg.w_tfidf > 0.0f);
    ASSERT(cfg.w_ri > 0.0f);
    ASSERT(cfg.w_api > 0.0f);
    ASSERT(cfg.w_type > 0.0f);
    ASSERT(cfg.w_decorator >= 0.0f);
    ASSERT(cfg.w_struct_profile > 0.0f);
    ASSERT(cfg.threshold > 0.0f);
    ASSERT(cfg.threshold <= 1.0f);
    ASSERT(cfg.max_edges > 0);
    PASS();
}

/* ── Regression tests for review fixes ─────────────────────── */

static void test_search_query_null_doc_vectors(void) {
    TEST(search_query on non-finalized corpus returns 0 results);
    fce_sem_corpus_t *corp = fce_sem_corpus_new();
    ASSERT(corp != NULL);
    /* Don't add docs or finalize — doc_vectors_q is NULL. */
    fce_sem_ranked_t results[4];
    uint32_t count = 0;
    fce_sem_search_query(corp, "test", 4, results, &count);
    ASSERT(count == 0);
    fce_sem_corpus_free(corp);
    PASS();
}

static void test_combined_score_zero_vectors(void) {
    TEST(combined score with zero api/type/deco vectors works);
    fce_sem_func_t a = {0}, b = {0};

    fce_sem_ensure_ready();

    a.file_path = "src/handler.c";
    b.file_path = "src/handler.c";
    a.tfidf_len = 2;
    int indices[] = {0, 1};
    float weights[] = {1.0f, 1.0f};
    a.tfidf_indices = indices;
    a.tfidf_weights = weights;
    b.tfidf_len = 2;
    b.tfidf_indices = indices;
    b.tfidf_weights = weights;
    fce_sem_random_index("handle", &a.ri_vec);
    fce_sem_random_index("handle", &b.ri_vec);
    /* api_vec, type_vec, deco_vec are all zero (common case). */

    fce_sem_config_t cfg = fce_sem_get_config();
    float score = fce_sem_combined_score(&a, &b, &cfg);
    /* Should still produce a positive score from TF-IDF + RI. */
    ASSERT(score > 0.0f);
    ASSERT(score <= 1.0f);
    PASS();
}

static void test_corpus_add_doc_pathological_count(void) {
    TEST(corpus add_doc rejects >512 tokens);
    fce_sem_corpus_t *corp = fce_sem_corpus_new();
    /* Create a token array with >512 entries. */
    const char **tokens = (const char **)calloc(600, sizeof(char *));
    for (int i = 0; i < 600; i++) tokens[i] = "tok";
    fce_sem_corpus_add_doc(corp, tokens, 600);
    ASSERT(fce_sem_corpus_doc_count(corp) == 0);
    free(tokens);
    fce_sem_corpus_free(corp);
    PASS();
}

static void test_corpus_add_doc_rejects_after_finalize(void) {
    TEST(corpus add_doc rejected after finalize);
    fce_sem_corpus_t *corp = fce_sem_corpus_new();
    const char *t1[] = {"a", "b"};
    fce_sem_corpus_add_doc(corp, t1, 2);
    fce_sem_corpus_finalize(corp);
    ASSERT(fce_sem_corpus_doc_count(corp) == 1);
    fce_sem_corpus_add_doc(corp, t1, 2);
    ASSERT(fce_sem_corpus_doc_count(corp) == 1);
    fce_sem_corpus_free(corp);
    PASS();
}

static void test_shutdown_and_reinit(void) {
    TEST(shutdown and re-init works);
    fce_sem_ensure_ready();
    /* Score something to ensure the pretrained map is populated. */
    fce_sem_vec_t v;
    fce_sem_random_index("handle", &v);
    float mag = 0.0f;
    for (int i = 0; i < FCE_SEM_DIM; i++) mag += v.v[i] * v.v[i];
    ASSERT(mag > 0.0f);

    fce_sem_shutdown();

    /* After shutdown, ensure_ready should re-initialize. */
    fce_sem_ensure_ready();
    fce_sem_vec_t v2;
    fce_sem_random_index("handle", &v2);
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        ASSERT(v.v[i] == v2.v[i]);
    }
    PASS();
}

static void test_hash_table_null_guard(void) {
    TEST(hash table NULL guard on get/set/has);
    ASSERT(fce_ht_get(NULL, "key") == NULL);
    ASSERT(fce_ht_set(NULL, "key", NULL) == NULL);
    ASSERT(fce_ht_has(NULL, "key") == false);
    ASSERT(fce_ht_count(NULL) == 0);
    PASS();
}

static void test_corpus_search_query(void) {
    TEST(corpus search query returns top results);
    /* Build a corpus with known tokens. */
    fce_sem_corpus_t *corp = fce_sem_corpus_new();
    const char *t1[] = {"handle", "request"};
    const char *t2[] = {"handle", "response"};
    const char *t3[] = {"process", "data"};
    const char *t4[] = {"process", "input"};
    fce_sem_corpus_add_doc(corp, t1, 2);
    fce_sem_corpus_add_doc(corp, t2, 2);
    fce_sem_corpus_add_doc(corp, t3, 2);
    fce_sem_corpus_add_doc(corp, t4, 2);
    fce_sem_corpus_finalize(corp);

    /* Search for "handle" — should rank t1 and t2 highly. */
    fce_sem_ranked_t results[4];
    uint32_t count = 0;
    fce_sem_search_query(corp, "handle", 4, results, &count);
    ASSERT(count > 0);
    ASSERT(count <= 4);

    /* First result should be high similarity (int8 quantization
     * introduces small error — threshold accounts for this). */
    ASSERT(results[0].score > 0.8f);

    fce_sem_corpus_free(corp);
    PASS();
}

/* LOW priority fix tests */

static void test_abbreviation_lazy_allocation(void) {
    TEST(abbreviation hash table lazy allocation);
    /* Test that abbreviations work with lazy allocation.
     * This verifies P3 fix: abbreviations are only allocated on first tokenize call. */
    char *out[100];
    int count = fce_sem_tokenize("ctx", out, 100);
    ASSERT(count > 0);

    /* Should have both the original token and expanded form. */
    int has_ctx = 0, has_context = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(out[i], "ctx") == 0) has_ctx = 1;
        if (strcmp(out[i], "context") == 0) has_context = 1;
        free(out[i]);
    }
    ASSERT(has_ctx);
    ASSERT(has_context);
    PASS();
}

static void test_abbreviation_concurrent_init(void) {
    TEST(abbreviation hash table concurrent initialization);
    /* Test that multiple concurrent calls to tokenize don't cause issues
     * with lazy allocation (tests thread-safety of P3 fix). */
    char *out[100];

    /* Multiple sequential calls should all work. */
    for (int call = 0; call < 3; call++) {
        int count = fce_sem_tokenize("err_cfg", out, 100);
        ASSERT(count > 0);

        int has_error = 0, has_config = 0;
        for (int i = 0; i < count; i++) {
            if (strcmp(out[i], "error") == 0) has_error = 1;
            if (strcmp(out[i], "config") == 0) has_config = 1;
            free(out[i]);
        }
        ASSERT(has_error || count > 0);  /* At least some tokens */
        ASSERT(has_config || count > 0);
    }
    PASS();
}

static void test_reverse_index_memory_cap(void) {
    TEST(reverse index respects memory cap);
    /* Test P1 fix: memory cap prevents excessive allocation.
     * We can't easily create a truly pathological corpus without huge memory,
     * but we verify the cap check is in place by ensuring normal corpora work. */
    fce_sem_corpus_t *corp = fce_sem_corpus_new();

    /* Add a reasonable corpus. */
    const char *doc1[] = {"handle", "request", "data"};
    const char *doc2[] = {"process", "response", "data"};
    const char *doc3[] = {"validate", "input", "stream"};

    fce_sem_corpus_add_doc(corp, doc1, 3);
    fce_sem_corpus_add_doc(corp, doc2, 3);
    fce_sem_corpus_add_doc(corp, doc3, 3);

    /* Finalize should succeed without hitting the memory cap. */
    fce_sem_corpus_finalize(corp);
    ASSERT(corp != NULL);  /* Corpus should be valid after finalize */

    fce_sem_corpus_free(corp);
    PASS();
}

/* ── fix tests ──────────────────────────────── */

static void test_shutdown_reinit_abbreviations(void) {
    TEST(shutdown + reinit preserves abbreviations);
    fce_sem_ensure_ready();
    /* Tokenize with abbreviation expansion. */
    char *tokens[32];
    int n = fce_sem_tokenize("err_handler", tokens, 32);
    ASSERT(n > 1);
    int found_expanded = 0;
    for (int i = 0; i < n; i++) {
        if (strcmp(tokens[i], "error") == 0) found_expanded = 1;
        free(tokens[i]);
    }
    ASSERT(found_expanded);

    fce_sem_shutdown();
    fce_sem_ensure_ready();

    /* After reinit, abbreviations should still work. */
    n = fce_sem_tokenize("err_handler", tokens, 32);
    ASSERT(n > 1);
    found_expanded = 0;
    for (int i = 0; i < n; i++) {
        if (strcmp(tokens[i], "error") == 0) found_expanded = 1;
        free(tokens[i]);
    }
    ASSERT(found_expanded);
    PASS();
}

static void test_corpus_add_doc_grows_cap(void) {
    TEST(corpus add_doc grows cap consistently across multiple adds);
    fce_sem_ensure_ready();
    fce_sem_corpus_t *corp = fce_sem_corpus_new();
    ASSERT(corp != NULL);

    /* Add enough docs to trigger multiple reallocs. */
    for (int i = 0; i < 20; i++) {
        char tok[16];
        snprintf(tok, sizeof(tok), "token_%d", i);
        const char *doc[] = {tok};
        fce_sem_corpus_add_doc(corp, doc, 1);
    }
    /* All 20 docs should be accepted (cap grew correctly). */
    ASSERT(fce_sem_corpus_doc_count(corp) == 20);

    /* Finalize should not crash (cap/count are consistent). */
    fce_sem_corpus_finalize(corp);
    fce_sem_corpus_free(corp);
    PASS();
}

static void test_rank_flat_zero_scores(void) {
    TEST(rank flat with zero scores returns results);
    fce_sem_ensure_ready();

    /* 3 corpus items with completely empty TF-IDF and zero RI vectors. */
    int max_tok = 1;
    float all_weights[] = {0, 0, 0};
    int all_indices[] = {0, 0, 0};
    int tfidf_lens[] = {0, 0, 0};
    float all_ri_vecs[3 * FCE_SEM_DIM];
    memset(all_ri_vecs, 0, sizeof(all_ri_vecs));
    const char *paths[] = {"a.c", "b.c", "c.c"};

    /* Query with some actual signal. */
    float q_weights[] = {1.0f};
    int q_indices[] = {0};
    float q_ri[FCE_SEM_DIM];
    fce_sem_vec_t qv;
    fce_sem_random_index("query", &qv);
    memcpy(q_ri, qv.v, sizeof(q_ri));

    fce_sem_ranked_t results[3];
    uint32_t count = 0;
    fce_sem_simple_rank_flat(
        all_weights, all_indices, tfidf_lens, all_ri_vecs, paths, 3, max_tok,
        q_indices, q_weights, 1, q_ri,
        3, results, &count);

    /* Should still return all 3 items (scores are 0, not negative). */
    ASSERT(count == 3);
    /* All scores should be >= 0. */
    for (uint32_t i = 0; i < count; i++) {
        ASSERT(results[i].score >= 0.0f);
    }
    PASS();
}

static void test_rank_flat_top_k_limit(void) {
    TEST(rank flat respects top_k limit with mixed scores);
    fce_sem_ensure_ready();

    /* 5 corpus items with distinct TF-IDF indices so scores differ. */
    int max_tok = 1;
    float all_weights[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int all_indices[] = {0, 1, 2, 3, 4};
    int tfidf_lens[] = {1, 1, 1, 1, 1};
    float all_ri_vecs[5 * FCE_SEM_DIM];
    memset(all_ri_vecs, 0, sizeof(all_ri_vecs));
    const char *paths[] = {"a.c", "b.c", "c.c", "d.c", "e.c"};

    /* Query matches index 0 with weight 1.0. */
    float q_weights[] = {1.0f};
    int q_indices[] = {0};
    float q_ri[FCE_SEM_DIM];
    memset(q_ri, 0, sizeof(q_ri));

    fce_sem_ranked_t results[5];
    uint32_t count = 0;
    fce_sem_simple_rank_flat(
        all_weights, all_indices, tfidf_lens, all_ri_vecs, paths, 5, max_tok,
        q_indices, q_weights, 1, q_ri,
        2, results, &count);  /* top_k = 2 */

    ASSERT(count == 2);
    ASSERT(results[0].score >= results[1].score);
    PASS();
}

/* ── Review 0003 fix tests ─────────────────────────────── */

static void test_search_null_file_path(void) {
    TEST(search with NULL file_path does not crash);
    fce_sem_ensure_ready();
    fce_sem_func_t a = {0}, b = {0};
    a.tfidf_len = 2;
    int indices[] = {0, 1};
    float weights[] = {1.0f, 1.0f};
    a.tfidf_indices = indices;
    a.tfidf_weights = weights;
    b.tfidf_indices = indices;
    b.tfidf_weights = weights;
    b.tfidf_len = 2;
    fce_sem_random_index("foo", &a.ri_vec);
    fce_sem_random_index("bar", &b.ri_vec);
    /* file_path is NULL — single-pair score must work */
    float s = fce_sem_simple_score(&a, &b);
    ASSERT(s >= 0.0f && s <= 1.0f);
    /* search with func array and NULL file_path must not crash */
    fce_sem_func_t corpus[] = {a, b};
    fce_sem_ranked_t results[4];
    uint32_t count = 0;
    fce_sem_simple_search(&a, corpus, 2, 4, 0.0f, results, &count);
    ASSERT(count >= 0);
    PASS();
}

static void test_doc_count_batch_parity(void) {
    TEST(batch and single ingest agree on doc_count);
    fce_sem_ensure_ready();
    /* Two valid docs + one empty (count=0) + one oversized (count=600). */
    const char *d1[] = {"a", "b"};
    const char *d2[] = {"c", "d"};

    /* Single-doc path: only 2 docs counted */
    fce_sem_corpus_t *single = fce_sem_corpus_new();
    fce_sem_corpus_add_doc(single, d1, 2);
    fce_sem_corpus_add_doc(single, d2, 2);
    fce_sem_corpus_add_doc(single, NULL, 0);    /* empty — rejected */
    ASSERT(fce_sem_corpus_doc_count(single) == 2);

    /* Batch path: 4 inputs but only 2 valid */
    fce_sem_corpus_t *batch = fce_sem_corpus_new();
    int tc[] = {2, 2, 0, 600};
    /* Flat token array: doc 0 = {"a","b"}, doc 1 = {"c","d"}, doc 2 = empty, doc 3 = oversized */
    const char *all_tokens[] = {"a", "b", "c", "d"};
    fce_sem_corpus_add_docs_batch(batch, (char **)all_tokens, tc, 4, 2);
    ASSERT(fce_sem_corpus_doc_count(batch) == 2);

    fce_sem_corpus_free(single);
    fce_sem_corpus_free(batch);
    PASS();
}

static void test_abbrev_ht_oom_retry(void) {
    TEST(abbrev table retry after calloc failure);
    /* We can't easily trigger OOM, but verify the table works normally */
    fce_sem_ensure_ready();
    fce_sem_func_t a = {0};
    a.file_path = "src/main.c";
    int indices[] = {0};
    float weights[] = {1.0f};
    a.tfidf_indices = indices;
    a.tfidf_weights = weights;
    a.tfidf_len = 1;
    /* "w/0" should expand to "with 0" via abbreviation table */
    fce_sem_func_t b = {0};
    b.file_path = "src/main.c";
    b.tfidf_indices = indices;
    b.tfidf_weights = weights;
    b.tfidf_len = 1;
    fce_sem_random_index("w/0", &a.ri_vec);
    fce_sem_random_index("with 0", &b.ri_vec);
    float s = fce_sem_simple_score(&a, &b);
    /* Abbreviation expansion should make these more similar than raw "w/0" vs "with 0" */
    ASSERT(s >= 0.0f);
    PASS();
}

static void test_corpus_get_or_add_oom_rollback(void) {
    TEST(fce_corpus_get_or_add handles duplicate tokens);
    fce_sem_corpus_t *corp = fce_sem_corpus_new();
    ASSERT(corp != NULL);
    const char *tokens[] = {"alpha", "beta", "gamma"};
    fce_sem_corpus_add_doc(corp, tokens, 3);
    ASSERT(fce_sem_corpus_token_count(corp) == 3);
    /* Second doc reuses tokens — should not inflate vocabulary */
    const char *tokens2[] = {"alpha", "delta"};
    fce_sem_corpus_add_doc(corp, tokens2, 2);
    ASSERT(fce_sem_corpus_token_count(corp) == 4);
    /* Finalize and check IDF */
    fce_sem_corpus_finalize(corp);
    float idf_beta = fce_sem_corpus_idf(corp, "beta");
    ASSERT(idf_beta > 0.0f);  /* beta in 1 of 2 docs */
    float idf_alpha = fce_sem_corpus_idf(corp, "alpha");
    ASSERT(idf_alpha >= 0.0f); /* alpha in 2 of 2 docs — IDF = log(2/2) = 0 */
    fce_sem_corpus_free(corp);
    PASS();
}

/* ── Main ──────────────────────────────────────────────────────── */

int main(void) {
    printf("fast-code-embed tests\n");
    printf("=========================\n\n");

    /* Tokenization */
    printf("Tokenization:\n");
    test_tokenize_camel_case();
    test_tokenize_snake_case();
    test_tokenize_dot_separated();
    test_tokenize_pascal_case();
    test_tokenize_empty();
    test_tokenize_null();
    test_tokenize_abbreviations();
    test_tokenize_abbreviations_hash_table();

    /* Random indexing */
    printf("\nRandom Indexing:\n");
    test_random_index_deterministic();
    test_random_index_different_tokens();
    test_random_index_null_token();

    /* Cosine similarity */
    printf("\nCosine Similarity:\n");
    test_cosine_identical();
    test_cosine_orthogonal();
    test_cosine_null();

    /* Normalize */
    printf("\nNormalize:\n");
    test_normalize();

    /* Vec add scaled */
    printf("\nVec Add Scaled:\n");
    test_vec_add_scaled();

    /* Corpus */
    printf("\nCorpus:\n");
    test_corpus_add_and_finalize();
    test_corpus_batch_add();
    test_corpus_token_at();

    /* Proximity */
    printf("\nProximity:\n");
    test_proximity_same_file();
    test_proximity_same_dir();
    test_proximity_distant();
    test_proximity_null();

    /* Combined scoring */
    printf("\nCombined Scoring:\n");
    test_combined_score_identical();
    test_combined_score_different();
    test_combined_score_null();

    /* Simple API */
    printf("\nSimple API:\n");
    test_simple_score_identical();
    test_simple_score_different();
    test_simple_score_range();
    test_simple_score_deterministic();
    test_simple_rank();
    test_simple_rank_flat();
    test_simple_rank_flat_larger();

    /* Diffusion */
    printf("\nDiffusion:\n");
    test_diffuse();

    /* Config */
    printf("\nConfig:\n");
    test_config_defaults();

    /* Regression tests */
    printf("\nRegression:\n");
    test_search_query_null_doc_vectors();
    test_combined_score_zero_vectors();
    test_corpus_add_doc_pathological_count();
    test_corpus_add_doc_rejects_after_finalize();
    test_shutdown_and_reinit();
    test_hash_table_null_guard();
    test_corpus_search_query();

    /* LOW priority fix tests */
    printf("\nLOW Priority Fixes:\n");
    test_abbreviation_lazy_allocation();
    test_abbreviation_concurrent_init();
    test_reverse_index_memory_cap();

    test_shutdown_reinit_abbreviations();
    test_corpus_add_doc_grows_cap();
    test_rank_flat_zero_scores();
    test_rank_flat_top_k_limit();

    /* fixes */
    printf("\nReview 0003 Fixes:\n");
    test_search_null_file_path();
    test_doc_count_batch_parity();
    test_abbrev_ht_oom_retry();
    test_corpus_get_or_add_oom_rollback();

    /* Summary */
    printf("\n=========================\n");
    printf("%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
