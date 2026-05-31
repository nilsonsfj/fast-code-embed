/*
 * semantic.c — Algorithmic code embeddings: TF-IDF, Random Indexing,
 * API/Type/Decorator signatures, combined scoring, graph diffusion.
 *
 * All signals computed from graph buffer metadata — no source file reads.
 * Uses xxHash for deterministic random vectors. Pure C, zero dependencies.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "semantic/semantic.h"
#include "foundation/constants.h"
#include "foundation/hash_table.h"
#include "foundation/log.h"
#include "foundation/profile.h"
#include "foundation/platform.h"
#include "foundation/compat_thread.h"
#include "pipeline/worker_pool.h"
#include "foundation/simd_dot768.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <sys/stat.h>
#if defined(__linux__)
#include <malloc.h>
#endif

/* Portable: convert uint32_t to canonical little-endian for deterministic
 * hashing across architectures. On LE platforms (x86, ARM64) compiles to a
 * no-op. On BE, performs a byte swap. */
static inline uint32_t fce_htole32(uint32_t v) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return __builtin_bswap32(v);
#else
    return v;
#endif
}
#include <time.h>
#include <assert.h>
#include "embed/code_vectors.h"

#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"

#include <ctype.h>
#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sched.h>
#endif

/* ── Constants ───────────────────────────────────────────────────── */

enum {
    FCE_TOKEN_BUF_LEN = 128,
    FCE_CORPUS_INIT_CAP = 4096,
    FCE_DOC_TOKENS_INIT = 64,
    FCE_RI_SEED_BASE = 0x52494E44, /* "RIND" */
    FCE_PM_UNINIT = 0, FCE_PM_INIT = 1, FCE_PM_READY = 2,
};

/* Default signal weights for fce_sem_combined_score.
 * Applied weights sum to ~0.85; proximity multiplier is applied on top. */
#define FCE_SEM_W_TFIDF 0.20F
#define FCE_SEM_W_RI 0.25F
#define FCE_SEM_W_API 0.15F
#define FCE_SEM_W_TYPE 0.10F
#define FCE_SEM_W_DECORATOR 0.05F
#define FCE_SEM_W_STRUCT_PROFILE 0.10F

/* Threshold bounds for FCE_SEMANTIC_THRESHOLD env override. */
#define FCE_SEM_THRESHOLD_MIN 0.0F
#define FCE_SEM_THRESHOLD_MAX 1.0F

/* Epsilon for denominator guards in cosine math. */
#define FCE_SEM_DENOM_EPS 1e-10F

/* Count '/' characters in a path (for P3 proximity precomputation). */
static int fce_count_slashes(const char *path) {
    if (!path) return 0;
    int count = 0;
    for (const char *p = path; *p; p++) {
        if (*p == '/') count++;
    }
    return count;
}

/* Int8 quantization bounds for vector storage (maps [-1.0, 1.0] to [-127, 127]). */
#define FCE_SEM_INT8_MAX 127.0F
#define FCE_SEM_INT8_MIN (-127.0F)

/* Unit vector bounds (normalized similarity / cosine value limits). */
#define FCE_SEM_UNIT_POS 1.0F
#define FCE_SEM_UNIT_NEG (-1.0F)

/* Proximity boost: same-file gets 1.0 + FCE_SEM_PROX_MAX_BOOST, distant gets 1.0. */
#define FCE_SEM_PROX_MAX_BOOST 0.10F

/* Reflective Random Indexing blend factors for pass2 (context mixing). */
#define FCE_SEM_RRI_ALPHA 0.3F
#define FCE_SEM_RRI_BETA 0.7F

/* Strings of exactly two parts / dimensions used for co-occurrence window math. */
#define FCE_SEM_COOCCUR_STRIDE 2

/* Worker tile/chunk sizes for parallel cooccurrence passes. */
enum {
    FCE_SEM_WORKER_STACK_CAP = 256,
    FCE_SEM_TILE_SIZE = 40,
    FCE_SEM_SEEN_INIT_CAP = 256,
    FCE_SEM_RESOLVE_CHUNK = 64,
    FCE_SEM_COOCCUR_CHUNK = 32,
    FCE_SEM_RRI_TILE = 128,
    FCE_SEM_INT8_I_LO = -128,
    FCE_SEM_INT8_I_HI = 127,
    FCE_SEM_ATOMIC_INC = 1,
    FCE_SEM_RI_NONZERO_COUNT = 8,
};

/* Numeric conversion radix for strtol (base 10 decimal). */
enum { BASE_DECIMAL = 10 };

/* ── Forward declarations ──────────────────────────────────────────── */
static void heap_siftdown(fce_sem_ranked_t *arr, int root, int len);
static int fce_ranked_cmp_desc(const void *a, const void *b);

/* ── Configuration ───────────────────────────────────────────────── */

fce_sem_config_t fce_sem_get_config(void) {
    fce_sem_config_t cfg = {
        .w_tfidf = FCE_SEM_W_TFIDF,
        .w_ri = FCE_SEM_W_RI,
        .w_api = FCE_SEM_W_API,
        .w_type = FCE_SEM_W_TYPE,
        .w_decorator = FCE_SEM_W_DECORATOR,
        .w_struct_profile = FCE_SEM_W_STRUCT_PROFILE,
        .threshold = (float)FCE_SEM_EDGE_THRESHOLD,
        .max_edges = FCE_SEM_MAX_EDGES,
    };
    const char *thresh = getenv("FCE_SEMANTIC_THRESHOLD");
    if (thresh) {
        /* strtod reports errors via endptr; reject non-numeric input silently. */
        char *end = NULL;
        double parsed = strtod(thresh, &end);
        if (end != thresh && parsed > FCE_SEM_THRESHOLD_MIN && parsed <= FCE_SEM_THRESHOLD_MAX) {
            cfg.threshold = (float)parsed;
        }
    }
    return cfg;
}

bool fce_sem_is_enabled(void) {
    const char *val = getenv("FCE_SEMANTIC_ENABLED");
    return val && val[0] == '1';
}

/* ── Token extraction ────────────────────────────────────────────── */

/* True for characters that terminate a token regardless of case. */
static bool is_token_delim(char c) {
    return c == '.' || c == '/' || c == '_' || c == '-' || c == ' ' || c == '(' || c == ')' ||
           c == ',' || c == ':';
}

/* True for a camelCase transition: uppercase letter preceded by a lowercase. */
static bool is_camel_break(const char *name, int i) {
    if (i <= 0) {
        return false;
    }
    char c = name[i];
    char p = name[i - 1];
    return c >= 'A' && c <= 'Z' && p >= 'a' && p <= 'z';
}

/* Flush the current buffer as a token into out[]. */
static void fce_flush_token(char *buf, int *blen, char **out, int *count, int max_out) {
    if (*blen > 0 && *blen < FCE_TOKEN_BUF_LEN && *count < max_out) {
        buf[*blen] = '\0';
        char *tok = strdup(buf);
        if (tok) out[(*count)++] = tok;
    }
    *blen = 0;
}

/* Abbreviation hash table for O(1) lookup (same pattern as pretrained map). */
enum { ABBREV_HT_BITS = 8, ABBREV_HT_SIZE = 1 << ABBREV_HT_BITS };
typedef struct { uint64_t hash; const char *abbrev; const char *expanded; } abbrev_ht_entry_t;
static abbrev_ht_entry_t *g_abbrev_ht = NULL;
static _Atomic int g_abbrev_ht_state = FCE_PM_UNINIT;
typedef struct { const char *abbrev; const char *expanded; } abbrev_pair_t;
static void ensure_abbrev_ht(const abbrev_pair_t *abbrevs);

int fce_sem_tokenize(const char *name, char **out, int max_out) {
    if (!name || !out || max_out <= 0) {
        return 0;
    }
    int count = 0;
    char buf[FCE_TOKEN_BUF_LEN];
    int blen = 0;

    for (int i = 0; name[i] && count < max_out; i++) {
        char c = name[i];
        bool split = is_token_delim(c);
        bool camel = is_camel_break(name, i);
        if (split || camel) {
            fce_flush_token(buf, &blen, out, &count, max_out);
            if (split) {
                continue;
            }
        }
        if (isalnum((unsigned char)c)) {
            if (blen < FCE_TOKEN_BUF_LEN - 1) {
                buf[blen++] = (char)tolower((unsigned char)c);
            } else {
                /* Token exceeds buffer — mark overflow, fce_flush_token will discard it. */
                blen = FCE_TOKEN_BUF_LEN;
            }
        }
    }
    fce_flush_token(buf, &blen, out, &count, max_out);

    /* Abbreviation expansion: add expanded forms for common code abbreviations.
     * "err" → also add "error", "ctx" → "context", etc.
     * Uses a hash table for O(1) lookup instead of O(N·M) linear scan. */
    /* Cross-language abbreviation table — covers Go, Python, JS/TS, Rust,
     * Java, C/C++, Ruby, PHP, Kotlin, Swift, Scala, C#, and common patterns. */
    static const abbrev_pair_t abbrevs[] = {
        /* Error/exception handling */
        {"err", "error"},
        {"exc", "exception"},
        {"ex", "exception"},
        /* Context/config */
        {"ctx", "context"},
        {"cfg", "config"},
        {"conf", "configuration"},
        {"env", "environment"},
        {"opt", "option"},
        {"opts", "options"},
        /* Request/response (HTTP, RPC) */
        {"req", "request"},
        {"res", "response"},
        {"resp", "response"},
        {"rsp", "response"},
        {"hdr", "header"},
        {"hdrs", "headers"},
        /* Strings/formatting */
        {"str", "string"},
        {"fmt", "format"},
        {"msg", "message"},
        {"txt", "text"},
        {"lbl", "label"},
        {"desc", "description"},
        /* Data structures */
        {"buf", "buffer"},
        {"arr", "array"},
        {"vec", "vector"},
        {"lst", "list"},
        {"dict", "dictionary"},
        {"tbl", "table"},
        {"stk", "stack"},
        {"que", "queue"},
        /* Functions/callbacks */
        {"fn", "function"},
        {"func", "function"},
        {"cb", "callback"},
        {"proc", "procedure"},
        {"ctor", "constructor"},
        {"dtor", "destructor"},
        /* Database/storage */
        {"db", "database"},
        {"col", "column"},
        {"stmt", "statement"},
        {"txn", "transaction"},
        {"trx", "transaction"},
        {"repo", "repository"},
        /* Auth/security */
        {"auth", "authentication"},
        {"authz", "authorization"},
        {"perm", "permission"},
        {"cred", "credential"},
        {"tok", "token"},
        {"pwd", "password"},
        /* Values/types */
        {"val", "value"},
        {"num", "number"},
        {"int", "integer"},
        {"bool", "boolean"},
        {"flt", "float"},
        {"dbl", "double"},
        /* Indexing/iteration */
        {"idx", "index"},
        {"iter", "iterator"},
        {"elem", "element"},
        {"cnt", "count"},
        {"len", "length"},
        {"sz", "size"},
        {"pos", "position"},
        {"off", "offset"},
        {"cap", "capacity"},
        /* Lifecycle */
        {"init", "initialize"},
        {"deinit", "deinitialize"},
        {"alloc", "allocate"},
        {"dealloc", "deallocate"},
        {"del", "delete"},
        {"rm", "remove"},
        /* Implementation/interface */
        {"impl", "implementation"},
        {"iface", "interface"},
        {"abs", "abstract"},
        {"decl", "declaration"},
        /* Parameters/attributes */
        {"param", "parameter"},
        {"arg", "argument"},
        {"attr", "attribute"},
        {"prop", "property"},
        {"ret", "return"},
        /* Source/destination */
        {"src", "source"},
        {"dst", "destination"},
        {"tgt", "target"},
        {"orig", "original"},
        {"prev", "previous"},
        {"cur", "current"},
        {"tmp", "temporary"},
        {"temp", "temporary"},
        /* Networking/IO */
        {"conn", "connection"},
        {"sess", "session"},
        {"sock", "socket"},
        {"addr", "address"},
        {"url", "uniform"},
        {"srv", "server"},
        {"cli", "client"},
        {"svc", "service"},
        {"ep", "endpoint"},
        /* Management */
        {"mgr", "manager"},
        {"ctrl", "controller"},
        {"hdlr", "handler"},
        {"sched", "scheduler"},
        {"disp", "dispatcher"},
        {"reg", "registry"},
        /* Async/concurrent */
        {"chan", "channel"},
        {"sem", "semaphore"},
        {"mtx", "mutex"},
        {"wg", "waitgroup"},
        {"sig", "signal"},
        {"evt", "event"},
        {"sub", "subscriber"},
        {"pub", "publisher"},
        /* Testing */
        {"spec", "specification"},
        {"mock", "mock"},
        {"stub", "stub"},
        {"assert", "assertion"},
        /* Logging/monitoring */
        {"log", "logging"},
        {"lvl", "level"},
        {"dbg", "debug"},
        {"wrn", "warning"},
        {"inf", "info"},
        /* Time */
        {"ts", "timestamp"},
        {"dur", "duration"},
        {"ttl", "timetolive"},
        /* Miscellaneous */
        {"ver", "version"},
        {"ns", "namespace"},
        {"pkg", "package"},
        {"mod", "module"},
        {"lib", "library"},
        {"dep", "dependency"},
        {"ref", "reference"},
        {"ptr", "pointer"},
        {"obj", "object"},
        {"doc", "document"},
        {"cmd", "command"},
        {"ops", "operations"},
        {"util", "utility"},
        {"hlp", "helper"},
        {"ext", "extension"},
        {NULL, NULL},
    };
    _Static_assert(sizeof(abbrevs)/sizeof(abbrevs[0]) - 1 <= ABBREV_HT_SIZE * 3 / 4,
                   "abbrev table load factor too high — grow ABBREV_HT_BITS");
    ensure_abbrev_ht(abbrevs);

    int orig_count = count;
    if (g_abbrev_ht) {
        for (int t = 0; t < orig_count && count < max_out; t++) {
            uint64_t h = XXH3_64bits(out[t], strlen(out[t]) + 1);
            uint32_t idx = (uint32_t)(h & (ABBREV_HT_SIZE - 1));
            while (g_abbrev_ht[idx].expanded) {
                if (g_abbrev_ht[idx].hash == h &&
                    strcmp(g_abbrev_ht[idx].abbrev, out[t]) == 0) {
                    char *exp = strdup(g_abbrev_ht[idx].expanded);
                    if (exp) out[count++] = exp;
                    break;
                }
                idx = (idx + 1) & (ABBREV_HT_SIZE - 1);
            }
        }
    }

    return count;
}

void fce_sem_tokenize_batch(const char **names, int count,
                             char **all_tokens_out, int *token_counts_out,
                             int max_out) {
    if (!names || !all_tokens_out || !token_counts_out || count <= 0 || max_out <= 0) {
        return;
    }
    for (int f = 0; f < count; f++) {
        token_counts_out[f] = fce_sem_tokenize(names[f],
                                                all_tokens_out + f * max_out,
                                                max_out);
    }
}

void fce_sem_corpus_add_docs_tokenized(fce_sem_corpus_t *corpus,
                                        const char **names, int count) {
    if (!corpus || !names || count <= 0) return;
    int max_tok = FCE_SEM_MAX_TOKENS;
    size_t flat_size = (size_t)count * (size_t)max_tok;
    char **all_tokens = (char **)malloc(flat_size * sizeof(char *));
    int *token_counts = (int *)malloc((size_t)count * sizeof(int));
    if (!all_tokens || !token_counts) {
        free(all_tokens);
        free(token_counts);
        return;
    }
    fce_sem_tokenize_batch(names, count, all_tokens, token_counts, max_tok);
    fce_sem_corpus_add_docs_batch(corpus, all_tokens, token_counts, count, max_tok);
    /* Free the token strings (batch_add_docs_batch doesn't take ownership) */
    for (int f = 0; f < count; f++) {
        for (int t = 0; t < token_counts[f]; t++) {
            free(all_tokens[f * max_tok + t]);
        }
    }
    free(all_tokens);
    free(token_counts);
}

/* ── File-based corpus building ────────────────────────────────── */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int fce_sem_corpus_add_files(fce_sem_corpus_t *corpus,
                              const char **paths, int path_count,
                              int chunk_size, int *file_doc_counts,
                              int max_tokens_per_chunk) {
    if (!corpus || !paths || path_count <= 0 || chunk_size <= 0) return -1;
    /* L-7: cap chunk_size to prevent excessive per-file allocation. */
    if (chunk_size > 16 * 1024 * 1024) return -1;

    int max_tok = max_tokens_per_chunk > 0 ? max_tokens_per_chunk : FCE_SEM_MAX_TOKENS;
    if (max_tok > FCE_SEM_MAX_TOKENS) max_tok = FCE_SEM_MAX_TOKENS;
    /* Process files in batches to bound peak memory. */
    int batch_cap = 5000;
    char **all_tokens = (char **)malloc((size_t)batch_cap * max_tok * sizeof(char *));
    int *token_counts = (int *)malloc((size_t)batch_cap * sizeof(int));
    if (!all_tokens || !token_counts) {
        free(all_tokens); free(token_counts);
        return -1;
    }

    int total_docs = 0;
    int batch_used = 0;

    for (int fi = 0; fi < path_count; fi++) {
        if (file_doc_counts) file_doc_counts[fi] = 0;
        FILE *f = fopen(paths[fi], "rb");
        if (!f) continue;

        /* Read entire file into memory. Use fstat for portable file size
         * (ftell returns 32-bit long on Windows, wrong for files > 2 GB). */
        struct stat st;
        if (fstat(fileno(f), &st) != 0 || st.st_size <= 0) { fclose(f); continue; }
        /* L4: cap file size at 64 MB (configurable via FCE_MAX_FILE_SIZE env var)
         * to prevent OOM on adversarial/pathological input. */
        size_t max_file_size = 64 * 1024 * 1024;
        const char *env_sz = getenv("FCE_MAX_FILE_SIZE");
        if (env_sz) {
            long val = strtol(env_sz, NULL, 10);
            if (val > 0) max_file_size = (size_t)val;
        }
        if (st.st_size > (off_t)max_file_size) { fclose(f); continue; }
        size_t file_len = (size_t)st.st_size;
        char *file_buf = (char *)malloc(file_len);
        if (!file_buf) { fclose(f); continue; }
        size_t nread = fread(file_buf, 1, file_len, f);
        fclose(f);

        /* Chunk by } boundaries, tokenize each chunk.
         * Identical logic to bench_mem_query.c. */
        for (size_t offset = 0; offset < nread; ) {
            size_t end = offset + (size_t)chunk_size;
            if (end < nread) {
                size_t found = 0;
                for (size_t i = end; i < nread; i++) {
                    if (file_buf[i] == '}') { found = i + 1; break; }
                }
                end = found ? found : nread;
            } else {
                end = nread;
            }
            size_t chunk_len = end - offset;
            char *chunk = (char *)malloc(chunk_len + 1);
            if (!chunk) { offset = end; continue; }
            memcpy(chunk, file_buf + offset, chunk_len);
            chunk[chunk_len] = '\0';

            /* Tokenize this chunk. */
            char *tok_buf[FCE_SEM_MAX_TOKENS];
            int ntok = fce_sem_tokenize(chunk, tok_buf, max_tok);
            free(chunk);

            /* Store in flat batch array. */
            int base = batch_used * max_tok;
            for (int t = 0; t < ntok; t++) {
                all_tokens[base + t] = tok_buf[t];
            }
            token_counts[batch_used] = ntok;
            batch_used++;
            total_docs++;
            if (file_doc_counts) file_doc_counts[fi]++;

            /* Flush batch when full. */
            if (batch_used >= batch_cap) {
                fce_sem_corpus_add_docs_batch(corpus, all_tokens, token_counts,
                                               batch_used, max_tok);
                for (int i = 0; i < batch_used; i++) {
                    int b = i * max_tok;
                    for (int t = 0; t < token_counts[i]; t++) free(all_tokens[b + t]);
                }
                batch_used = 0;
            }

            offset = end;
        }
        free(file_buf);
    }

    /* Flush remaining. */
    if (batch_used > 0) {
        fce_sem_corpus_add_docs_batch(corpus, all_tokens, token_counts,
                                       batch_used, max_tok);
        for (int i = 0; i < batch_used; i++) {
            int b = i * max_tok;
            for (int t = 0; t < token_counts[i]; t++) free(all_tokens[b + t]);
        }
    }

    free(all_tokens);
    free(token_counts);
    return total_docs;
}

/* ── Dense vector operations ─────────────────────────────────────── */

float fce_sem_cosine(const fce_sem_vec_t *a, const fce_sem_vec_t *b) {
    if (!a || !b) {
        return 0.0F;
    }
    float dot, mag_a, mag_b;
    fce_dot768_mags3(a->v, b->v, &dot, &mag_a, &mag_b);
    float denom = sqrtf(mag_a) * sqrtf(mag_b);
    if (denom < FCE_SEM_DENOM_EPS) {
        return 0.0F;
    }
    return dot / denom;
}

/* Cosine similarity with precomputed magnitude for side a (the query).
 * Used in ranking hot loops to avoid recomputing the query magnitude
 * on every corpus item — roughly 1/3 of per-element FLOPs saved. */
static float fce_sem_cosine_aliased_with_mag(const float * restrict a, const float * restrict b, float mag_a) {
    if (!a || !b) return 0.0F;
    float dot, mag_b;
    fce_dot768_add_mag_b(a, b, &dot, &mag_b);
    float denom = sqrtf(mag_a) * sqrtf(mag_b);
    if (denom < FCE_SEM_DENOM_EPS) return 0.0F;
    return dot / denom;
}

/* ── Hash table integer value encoding ────────────────────────────
 * Store token indices directly as void* to avoid strdup+strtol per token.
 * Use idx+1 so that index 0 maps to non-NULL (1), and NULL means "not found".
 */
static inline void *token_idx_to_ptr(int idx) {
    return (void *)(intptr_t)(idx + 1);
}

static inline int ptr_to_token_idx(void *ptr) {
    if (!ptr) return FCE_NOT_FOUND;
    return (int)(intptr_t)ptr - 1;
}

/* Pretrained token lookup table — built lazily on first use. */
static FCEHashTable *g_pretrained_map = NULL;

static void init_pretrained_map(void) {
    /* L2: validate blob length before dereferencing blob_header, so the
     * checks stay sound if the blob ever becomes runtime-loaded. */
    if (FCE_PRETRAINED_VECTOR_BLOB_LEN < (size_t)(8 + (size_t)FCE_PRETRAINED_TOKEN_COUNT * FCE_PRETRAINED_DIM)) {
        fce_log_error("pretrained.blob.short", "detail", "blob too short");
        return;
    }
    const int32_t *blob_header = (const int32_t *)FCE_PRETRAINED_VECTOR_BLOB;
    int32_t blob_token_count = blob_header[0];
    int32_t blob_dim = blob_header[1];
    if (blob_token_count != FCE_PRETRAINED_TOKEN_COUNT) {
        char exp[16], act[16];
        snprintf(exp, sizeof exp, "%d", FCE_PRETRAINED_TOKEN_COUNT);
        snprintf(act, sizeof act, "%d", blob_token_count);
        fce_log_error("pretrained.blob.mismatch",
                      "field", "token_count",
                      "expected", exp, "actual", act);
        return;
    }
    if (blob_dim != FCE_PRETRAINED_DIM) {
        char exp[16], act[16];
        snprintf(exp, sizeof exp, "%d", FCE_PRETRAINED_DIM);
        snprintf(act, sizeof act, "%d", blob_dim);
        fce_log_error("pretrained.blob.mismatch",
                      "field", "dimension",
                      "expected", exp, "actual", act);
        return;
    }

    g_pretrained_map = fce_ht_create(FCE_PRETRAINED_TOKEN_COUNT);
    if (!g_pretrained_map) {
        return; /* OOM: table stays NULL, lookups will return NULL */
    }
    for (int i = 0; i < FCE_PRETRAINED_TOKEN_COUNT; i++) {
        const char *tok = FCE_PRETRAINED_TOKENS[i];
        if (tok && tok[0]) {
            /* Keys point directly into the static FCE_PRETRAINED_TOKENS array —
             * no strdup needed since the source outlives the table. */
            fce_ht_set(g_pretrained_map, tok, token_idx_to_ptr(i));
        }
    }
}

/* Thread-safe lazy init of the pretrained token lookup map.
 * Uses a tri-state atomic (UNINIT → INIT → READY) with CAS to ensure
 * init_pretrained_map runs exactly once even if multiple threads race.
 * A mutex serializes init to prevent hash-table resize races. */
static _Atomic int g_pretrained_state = FCE_PM_UNINIT;  /* 0=uninit, 1=init, 2=ready */
static _Atomic bool g_shutting_down = false;
static fce_mutex_t g_pretrained_mutex;
static fce_once_t g_pretrained_once = FCE_ONCE_INIT;

static void init_pretrained_mutex(void) {
    fce_mutex_init(&g_pretrained_mutex);
}

static void ensure_abbrev_ht(const abbrev_pair_t *abbrevs) {
    if (g_shutting_down) return;
    if (atomic_load_explicit(&g_abbrev_ht_state, memory_order_acquire) == FCE_PM_READY) {
        return;
    }
    fce_once(&g_pretrained_once, init_pretrained_mutex);
    fce_mutex_lock(&g_pretrained_mutex);
    if (atomic_load_explicit(&g_abbrev_ht_state, memory_order_acquire) != FCE_PM_READY) {
        g_abbrev_ht = calloc(ABBREV_HT_SIZE, sizeof(abbrev_ht_entry_t));
        if (g_abbrev_ht) {
            for (int a = 0; abbrevs[a].abbrev; a++) {
                uint64_t h = XXH3_64bits(abbrevs[a].abbrev, strlen(abbrevs[a].abbrev) + 1);
                uint32_t idx = (uint32_t)(h & (ABBREV_HT_SIZE - 1));
                while (g_abbrev_ht[idx].expanded) {
                    idx = (idx + 1) & (ABBREV_HT_SIZE - 1);
                }
                g_abbrev_ht[idx].hash = h;
                g_abbrev_ht[idx].abbrev = abbrevs[a].abbrev;
                g_abbrev_ht[idx].expanded = abbrevs[a].expanded;
            }
            atomic_store_explicit(&g_abbrev_ht_state, FCE_PM_READY, memory_order_release);
        }
        /* On calloc failure, leave state as FCE_PM_UNINIT so next call can retry. */
    }
    fce_mutex_unlock(&g_pretrained_mutex);
}

static void ensure_pretrained_map(void) {
    if (g_shutting_down) return;
    if (atomic_load_explicit(&g_pretrained_state, memory_order_acquire) == FCE_PM_READY) {
        return;
    }
    fce_once(&g_pretrained_once, init_pretrained_mutex);
    int expected = FCE_PM_UNINIT;
    if (atomic_compare_exchange_strong(&g_pretrained_state, &expected, FCE_PM_INIT)) {
        fce_mutex_lock(&g_pretrained_mutex);
        init_pretrained_map();
        fce_mutex_unlock(&g_pretrained_mutex);
        atomic_store_explicit(&g_pretrained_state, FCE_PM_READY, memory_order_release);
    } else {
        /* Another thread is initializing — yield until ready.
         * If shutdown resets to FCE_PM_UNINIT, retry the CAS. */
        while (1) {
            int state = atomic_load_explicit(&g_pretrained_state, memory_order_acquire);
            if (state == FCE_PM_READY) return;
            if (state == FCE_PM_UNINIT) {
                /* Shutdown ran — retry init instead of spinning forever. */
                expected = FCE_PM_UNINIT;
                if (atomic_compare_exchange_strong(&g_pretrained_state, &expected, FCE_PM_INIT)) {
                    fce_mutex_lock(&g_pretrained_mutex);
                    init_pretrained_map();
                    fce_mutex_unlock(&g_pretrained_mutex);
                    atomic_store_explicit(&g_pretrained_state, FCE_PM_READY, memory_order_release);
                    return;
                }
            }
#ifdef _WIN32
            SwitchToThread();
#else
            sched_yield();
#endif
        }
    }
}

void fce_sem_ensure_ready(void) {
    ensure_pretrained_map();
}

/* Forward declaration for fce_sem_shutdown (defined later with corpus_free). */
static void fce_free_ht_kv(const char *key, void *value, void *userdata);

/* No-op iterator for tables whose keys are static/arena-allocated. */
static void noop_ht_kv(const char *key, void *value, void *userdata) {
    (void)key; (void)value; (void)userdata;
}

void fce_sem_shutdown(void) {
    g_shutting_down = true;
    fce_once(&g_pretrained_once, init_pretrained_mutex);
    fce_mutex_lock(&g_pretrained_mutex);

    if (g_pretrained_map) {
        fce_ht_foreach(g_pretrained_map, noop_ht_kv, NULL);
        fce_ht_free(g_pretrained_map);
        g_pretrained_map = NULL;
    }
    atomic_store_explicit(&g_pretrained_state, FCE_PM_UNINIT, memory_order_release);

    if (g_abbrev_ht) {
        free(g_abbrev_ht);
        g_abbrev_ht = NULL;
    }
    atomic_store_explicit(&g_abbrev_ht_state, FCE_PM_UNINIT, memory_order_release);

    fce_mutex_unlock(&g_pretrained_mutex);
    fce_mutex_destroy(&g_pretrained_mutex);
    g_pretrained_once = (fce_once_t)FCE_ONCE_INIT;
    g_shutting_down = false;
}

void fce_sem_random_index(const char *token, fce_sem_vec_t *out) {
    memset(out, 0, sizeof(*out));
    if (!token) {
        return;
    }

    /* Try pretrained nomic-embed-code vector first (768d, distilled from 7B). */
    ensure_pretrained_map();
    if (g_pretrained_map) {
        void *idx_ptr = fce_ht_get(g_pretrained_map, token);
        if (idx_ptr) {
            int idx = ptr_to_token_idx(idx_ptr);
            if (idx >= 0 && idx < FCE_PRETRAINED_TOKEN_COUNT) {
                const int8_t *pvec = fce_pretrained_vec_at(idx);
                for (int d = 0; d < FCE_SEM_DIM && d < FCE_PRETRAINED_DIM; d++) {
                    out->v[d] = (float)pvec[d] / FCE_SEM_INT8_MAX;
                }
                return;
            }
        }
    }

    /* Fallback: sparse random vector for tokens not in pretrained vocab.
     * L4: hash includes NUL terminator for consistency with abbreviation and
     * token-dedup hashing. Position mod 768 gives slight bias (acceptable for RI). */
    uint64_t seed = XXH3_64bits(token, strlen(token) + 1);
    for (int i = 0; i < FCE_SEM_SPARSE_NNZE; i++) {
        uint32_t le_i = fce_htole32((uint32_t)i);
        uint64_t h = XXH3_64bits_withSeed(&le_i, sizeof(le_i), seed + FCE_RI_SEED_BASE);
        int pos = (int)(h % FCE_SEM_DIM);
        float sign = (h & 1) ? FCE_SEM_UNIT_POS : -FCE_SEM_UNIT_POS;
        out->v[pos] += sign;
    }
}

void fce_sem_normalize(fce_sem_vec_t *v) {
    if (!v) {
        return;
    }
    float mag = 0.0F;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        mag += v->v[i] * v->v[i];
    }
    mag = sqrtf(mag);
    if (mag < FCE_SEM_DENOM_EPS) {
        return;
    }
    float inv = FCE_SEM_UNIT_POS / mag;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        v->v[i] *= inv;
    }
}

void fce_sem_vec_add_scaled(fce_sem_vec_t *dst, const fce_sem_vec_t *src, float scale) {
    if (!dst || !src) return;
    /* Do NOT replace with fce_axpy_f32_768() here.
     * Clang -O2 auto-vectorizes this scalar loop into NEON/AVX2 at compile time.
     * Hand-written SIMD axpy kernels were benchmarked and showed a ~10% finalize
     * regression — the compiler's code generation (unrolling, scheduling, register
     * allocation) outperforms the manual intrinsics in this context. */
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        dst->v[i] += scale * src->v[i];
    }
}

/* ── Corpus (IDF + Random Indexing enrichment) ───────────────────── */

typedef struct {
    char *token;
    int doc_freq;
} corpus_entry_t;

struct fce_sem_corpus {
    FCEHashTable *token_map; /* token → index into entries[] */
    corpus_entry_t *entries;
    fce_sem_vec_t *enriched_vecs; /* separate allocation — NULL until finalize */
    int8_t *enriched_vecs_q;     /* R1: int8 quantized enriched vecs (768 bytes/entry) */
    fce_sem_vec_t *doc_vectors;  /* per-document vectors (sum of enriched token vecs) */
    int8_t *doc_vectors_q;       /* P0: quantized int8 doc vectors (768 bytes/doc) */
    float *doc_vectors_q_inv_mag; /* F2: reciprocal L2 magnitude of each quantized doc vector */
    int entry_count;
    int entry_cap;
    int doc_count;
    bool finalized;

    /* Per-document token lists for co-occurrence pass */
    int **doc_token_ids;
    int *doc_token_counts;
    int doc_cap;

    /* Inverted index: token_id → list of doc_ids containing that token.
     * Built during finalize. Enables fast keyword candidate retrieval. */
    int *inv_offsets;     /* inv_offsets[token_id] = start in inv_doc_ids */
    int *inv_doc_ids;     /* flat array of unique doc_ids per token */
};

static int fce_corpus_get_or_add(fce_sem_corpus_t *c, const char *token) {
    void *existing = fce_ht_get(c->token_map, token);
    if (existing) {
        return ptr_to_token_idx(existing);
    }
    if (c->entry_count >= c->entry_cap) {
        int new_cap = c->entry_cap < FCE_CORPUS_INIT_CAP ? FCE_CORPUS_INIT_CAP : c->entry_cap * 2;
        corpus_entry_t *grown = realloc(c->entries, (size_t)new_cap * sizeof(corpus_entry_t));
        if (!grown) {
            return FCE_NOT_FOUND;
        }
        c->entries = grown;
        c->entry_cap = new_cap;
    }
    /* Allocate key BEFORE bumping entry_count so OOM leaves state consistent. */
    char *key = strdup(token);
    if (!key) {
        return FCE_NOT_FOUND;
    }
    int idx = c->entry_count;
    fce_ht_set(c->token_map, key, token_idx_to_ptr(idx));
    /* C5: if fce_ht_get_key returns NULL, the insert failed (OOM).
     * Free the leaked strdup'd key and roll back entry_count. */
    const char *interned = fce_ht_get_key(c->token_map, token);
    if (!interned) {
        free(key);
        return FCE_NOT_FOUND;
    }
    c->entry_count++;
    /* Point entry at the hash table's interned key — avoids a second strdup.
     * The hash table key is freed by fce_free_ht_kv in fce_sem_corpus_free. */
    c->entries[idx].token = (char *)interned;
    c->entries[idx].doc_freq = 0;
    return idx;
}

fce_sem_corpus_t *fce_sem_corpus_new(void) {
    fce_sem_corpus_t *c = calloc(1, sizeof(fce_sem_corpus_t));
    if (c) {
        c->token_map = fce_ht_create(FCE_CORPUS_INIT_CAP);
        if (!c->token_map) {
            free(c);
            return NULL;
        }
    }
    return c;
}

void fce_sem_corpus_add_doc(fce_sem_corpus_t *corpus, const char **tokens, int count) {
    if (!corpus || !tokens || count <= 0 || corpus->finalized) {
        return;
    }
    /* Reject pathological documents to avoid O(N²) unique-token scan.
     * Real docs have ~10-50 tokens; minified JS can produce thousands. */
    enum { MAX_TOKENS_PER_DOC = 512 };
    if (count > MAX_TOKENS_PER_DOC) {
        return;
    }
    /* Track document for co-occurrence pass */
    if (corpus->doc_count >= corpus->doc_cap) {
        int new_cap =
            corpus->doc_cap < FCE_DOC_TOKENS_INIT ? FCE_DOC_TOKENS_INIT : corpus->doc_cap * 2;
        int **new_ids = realloc(corpus->doc_token_ids, (size_t)new_cap * sizeof(int *));
        int *new_counts = realloc(corpus->doc_token_counts, (size_t)new_cap * sizeof(int));
        if (!new_ids || !new_counts) {
            if (new_ids) corpus->doc_token_ids = new_ids;
            if (new_counts) corpus->doc_token_counts = new_counts;
            return;
        }
        corpus->doc_token_ids = new_ids;
        corpus->doc_token_counts = new_counts;
        corpus->doc_cap = new_cap;
    }
    int doc_idx = corpus->doc_count++;
    corpus->doc_token_ids[doc_idx] = malloc((size_t)count * sizeof(int));
    corpus->doc_token_counts[doc_idx] = count;
    if (!corpus->doc_token_ids[doc_idx]) {
        corpus->doc_count--;
        return;
    }

    /* Per-doc unique set for IDF */
    int *seen = malloc((size_t)count * sizeof(int));
    if (!seen) {
        free(corpus->doc_token_ids[doc_idx]);
        corpus->doc_token_ids[doc_idx] = NULL;
        corpus->doc_count--;
        return;
    }
    int seen_count = 0;

    for (int i = 0; i < count; i++) {
        int tid = fce_corpus_get_or_add(corpus, tokens[i]);
        corpus->doc_token_ids[doc_idx][i] = tid;
        if (tid < 0) {
            continue;
        }
        /* Check uniqueness for IDF (simple linear scan — tokens per doc is small) */
        bool is_new = true;
        for (int j = 0; j < seen_count; j++) {
            if (seen[j] == tid) {
                is_new = false;
                break;
            }
        }
        if (is_new) {
            seen[seen_count++] = tid;
            corpus->entries[tid].doc_freq++;
        }
    }
    free(seen);
}

/* ── Parallel corpus batch build ──────────────────────────────────── */
/* Strategy:
 *   Phase A (SEQUENTIAL): Scan all documents once to build the global
 *     token_map (inserts unique tokens, assigns global IDs). This is
 *     inherently sequential (hash table mutation), but much faster than
 *     the current per-doc add_doc because we avoid the per-doc malloc of
 *     the `seen` array and per-doc bookkeeping.
 *   Phase B (PARALLEL): Each worker processes a chunk of docs, translates
 *     tokens → global IDs via read-only token_map lookups, fills
 *     doc_token_ids[d], and accumulates doc_freq contributions via atomics.
 */

typedef struct {
    fce_sem_corpus_t *corpus;
    char **all_tokens;
    const int *token_counts;
    int max_tokens;
    int doc_count;
    int base_doc;  /* Offset into doc_token_ids/doc_token_counts arrays. */
    const int *doc_map; /* original doc_index → compacted array_idx, or NULL if identity. */
    _Atomic int *doc_freq_atomic; /* per-entry atomic counter (entry_count long) */
    _Atomic int next_idx;
    _Atomic int error;  /* set to 1 on OOM — caller checks after parallel_for */
} batch_resolve_ctx_t;

/* Resolve one document: look up each token's global ID, fill the corpus
 * doc_token_ids[d], and bump the per-token doc_freq counter atomically.  The
 * caller is responsible for ensuring `seen` has capacity for `count` ints
 * before calling (the worker grows its per-thread scratch buffer). */
static void batch_resolve_one_doc(batch_resolve_ctx_t *bc, int doc_index, int *seen) {
    int count = bc->token_counts[doc_index];
    int array_idx = bc->doc_map ? bc->base_doc + bc->doc_map[doc_index]
                                : bc->base_doc + doc_index;
    if (count <= 0 || count > 512) {
        bc->corpus->doc_token_ids[array_idx] = NULL;
        bc->corpus->doc_token_counts[array_idx] = 0;
        return;
    }
    int *ids = malloc((size_t)count * sizeof(int));
    if (!ids) {
        bc->corpus->doc_token_ids[array_idx] = NULL;
        bc->corpus->doc_token_counts[array_idx] = 0;
        return;
    }
    bc->corpus->doc_token_ids[array_idx] = ids;
    bc->corpus->doc_token_counts[array_idx] = count;

    int seen_count = 0;
    char **tokens = &bc->all_tokens[(ptrdiff_t)doc_index * bc->max_tokens];
    for (int i = 0; i < count; i++) {
        void *idx_ptr = fce_ht_get(bc->corpus->token_map, tokens[i]);
        int tid = ptr_to_token_idx(idx_ptr);
        ids[i] = tid;
        if (tid < 0) {
            continue;
        }
        /* Unique-per-doc check for IDF */
        bool is_new = true;
        for (int j = 0; j < seen_count; j++) {
            if (seen[j] == tid) {
                is_new = false;
                break;
            }
        }
        if (is_new) {
            seen[seen_count++] = tid;
            atomic_fetch_add_explicit(&bc->doc_freq_atomic[tid], FCE_SEM_ATOMIC_INC,
                                      memory_order_relaxed);
        }
    }
}

static void batch_resolve_worker(int worker_id, void *ctx_ptr) {
    (void)worker_id;
    batch_resolve_ctx_t *bc = ctx_ptr;
    /* Per-worker scratch for unique-per-doc tracking */
    int local_seen_cap = FCE_SEM_SEEN_INIT_CAP;
    int *seen = malloc((size_t)local_seen_cap * sizeof(int));
    if (!seen) {
        return;
    }

    while (true) {
        int start =
            atomic_fetch_add_explicit(&bc->next_idx, FCE_SEM_RESOLVE_CHUNK, memory_order_relaxed);
        if (start >= bc->doc_count) {
            break;
        }
        int end = start + FCE_SEM_RESOLVE_CHUNK;
        if (end > bc->doc_count) {
            end = bc->doc_count;
        }
        for (int d = start; d < end; d++) {
            if (atomic_load_explicit(&bc->error, memory_order_relaxed)) {
                free(seen);
                return;
            }
            if (bc->doc_map && bc->doc_map[d] < 0) continue;
            int count = bc->token_counts[d];
            if (count > local_seen_cap) {
                int *grown = realloc(seen, (size_t)count * sizeof(int));
                if (!grown) {
                    atomic_store_explicit(&bc->error, 1, memory_order_relaxed);
                    free(seen);
                    return;
                }
                seen = grown;
                local_seen_cap = count;
            }
            batch_resolve_one_doc(bc, d, seen);
        }
    }
    free(seen);
}

void fce_sem_corpus_add_docs_batch(fce_sem_corpus_t *corpus, char **all_tokens,
                                   const int *token_counts, int doc_count, int max_tokens_per_doc) {
    if (!corpus || !all_tokens || !token_counts || doc_count <= 0 || corpus->finalized) {
        return;
    }

    /* Phase A (SEQUENTIAL): Build token_map and allocate doc arrays.
     * Hash table mutation can't be parallelized; strdup+insert is the cost. */
    enum { MAX_TOKENS_PER_DOC = 512 };

    /* First pass: count valid docs and build compacted index mapping. */
    int valid_doc_count = 0;
    int *doc_map = (int *)malloc((size_t)doc_count * sizeof(int));
    if (!doc_map) return;
    for (int d = 0; d < doc_count; d++) {
        int count = token_counts[d];
        if (count > 0 && count <= MAX_TOKENS_PER_DOC) {
            doc_map[d] = valid_doc_count++;
        } else {
            doc_map[d] = -1; /* sentinel: skip this doc */
        }
    }

    if (valid_doc_count == 0) {
        free(doc_map);
        return;
    }

    if (corpus->doc_cap < corpus->doc_count + valid_doc_count) {
        int old_cap = corpus->doc_cap;
        int new_cap = corpus->doc_count + valid_doc_count;
        int **new_ids = realloc(corpus->doc_token_ids, (size_t)new_cap * sizeof(int *));
        int *new_counts = realloc(corpus->doc_token_counts, (size_t)new_cap * sizeof(int));
        if (!new_ids || !new_counts) {
            if (new_ids) corpus->doc_token_ids = new_ids;
            if (new_counts) corpus->doc_token_counts = new_counts;
            free(doc_map);
            return;
        }
        corpus->doc_token_ids = new_ids;
        corpus->doc_token_counts = new_counts;
        corpus->doc_cap = new_cap;
        /* Zero-init new slots so OOM rollback free() is safe on unprocessed entries. */
        memset(new_ids + old_cap, 0, (size_t)(new_cap - old_cap) * sizeof(int *));
        memset(new_counts + old_cap, 0, (size_t)(new_cap - old_cap) * sizeof(int));
    }
    int base_doc = corpus->doc_count;
    corpus->doc_count += valid_doc_count;

    /* Phase A (SEQUENTIAL): Insert all unique tokens into token_map and
     * compute doc_freq per token. Only process valid docs. */
    for (int d = 0; d < doc_count; d++) {
        if (doc_map[d] < 0) continue;
        int count = token_counts[d];
        char **tokens = &all_tokens[(ptrdiff_t)d * max_tokens_per_doc];
        for (int i = 0; i < count; i++) {
            (void)fce_corpus_get_or_add(corpus, tokens[i]);
        }
    }

    /* Phase B (PARALLEL): Resolve tokens → IDs and count doc_freq per entry.
     * token_map is now read-only; each worker owns its doc range (no writes
     * to shared state except atomic doc_freq counters). */
    _Atomic int *doc_freq_atomic = calloc((size_t)corpus->entry_count, sizeof(_Atomic int));
    if (!doc_freq_atomic) {
        /* OOM fallback: sequential path. Roll back doc_count first since
         * add_doc increments it itself. */
        corpus->doc_count = base_doc;
        for (int d = 0; d < doc_count; d++) {
            if (doc_map[d] < 0) continue;
            int count = token_counts[d];
            char **tokens = &all_tokens[(ptrdiff_t)d * max_tokens_per_doc];
            fce_sem_corpus_add_doc(corpus, (const char **)tokens, count);
        }
        free(doc_map);
        return;
    }

    int worker_count = fce_default_worker_count(true);
    batch_resolve_ctx_t bc = {
        .corpus = corpus,
        .all_tokens = all_tokens,
        .token_counts = token_counts,
        .max_tokens = max_tokens_per_doc,
        .doc_count = doc_count,
        .base_doc = base_doc,
        .doc_map = doc_map,
        .doc_freq_atomic = doc_freq_atomic,
    };
    atomic_init(&bc.next_idx, 0);
    atomic_init(&bc.error, 0);
    fce_parallel_for_opts_t opts = {.max_workers = worker_count};
    fce_parallel_for(worker_count, batch_resolve_worker, &bc, opts);

    /* If any worker hit OOM, the batch is incomplete — roll back doc_count. */
    if (atomic_load_explicit(&bc.error, memory_order_relaxed)) {
        fce_log_error("batch.resolve.oom", "detail", "rolling back documents");
        for (int d = base_doc; d < corpus->doc_count; d++) {
            free(corpus->doc_token_ids[d]);
        }
        corpus->doc_count = base_doc;
        free(doc_freq_atomic);
        free(doc_map);
        return;
    }

    /* Phase C (SEQUENTIAL reduce): atomic counters → entries[].doc_freq */
    for (int i = 0; i < corpus->entry_count; i++) {
        corpus->entries[i].doc_freq +=
            atomic_load_explicit(&doc_freq_atomic[i], memory_order_relaxed);
    }
    free(doc_freq_atomic);
    free(doc_map);
}

/* ── Parallel corpus_finalize ─────────────────────────────────────── */
/* Strategy:
 *   1. Precompute base RI vectors into a shared array (eliminates ~333M
 *      redundant fce_sem_random_index calls on kernel-scale corpora).
 *   2. Co-occurrence passes: partition TARGET tokens across workers so each
 *      worker writes to a disjoint range of fce_enriched_vec (zero contention).
 *      Each worker still scans all documents but only accumulates for targets
 *      in its range. Inner vector add is the parallelized work.
 *   3. Normalize/blend loops are trivially parallel per-entry.
 */

/* Reverse index: for each token id, a list of (doc_id, position_in_doc) pairs.
 * Built once, reused for both cooccur passes. Eliminates the O(num_chunks × doc_count)
 * redundant outer scan in the old algorithm. */
typedef struct {
    int32_t doc_id;
    int32_t pos;
} cooccur_pos_t;

typedef struct {
    int *offsets;        /* offsets[entry_count + 1], prefix sum of occurrences */
    cooccur_pos_t *flat; /* flat array of positions, total = offsets[entry_count] */
} reverse_index_t;

/* Tagged source vector: sparse (~30% of tokens, 8 inline nonzeros) or dense int8
 * reference into FCE_PRETRAINED_VECTOR_BLOB (no copy). Dense path converts int8→float
 * on the fly in the hot loop. Tested with both int8 and float32 blob storage
 * formats — cooccur passes are memory-bandwidth-bound, so the int8 format (4x
 * less source traffic) is equivalent in wall time despite the conversion cost,
 * while saving 90 MB of binary size. */
typedef struct {
    uint8_t is_sparse; /* 1 = sparse path, 0 = dense int8 reference */
    uint8_t nnz;       /* number of nonzeros used in sparse path */
    uint16_t _pad;
    uint16_t indices[FCE_SEM_SPARSE_NNZE]; /* 8 * 2 = 16 bytes */
    float values[FCE_SEM_SPARSE_NNZE];     /* 8 * 4 = 32 bytes */
    const int8_t *dense_int8;              /* points into FCE_PRETRAINED_VECTOR_BLOB */
} fce_sem_src_entry_t;

/* Inline helper: initialize a target vector from a sparse/dense source. */
static inline void sem_target_init_from_src(fce_sem_vec_t *dst, const fce_sem_src_entry_t *src) {
    memset(dst, 0, sizeof(*dst));
    if (src->is_sparse) {
        for (int k = 0; k < src->nnz; k++) {
            dst->v[src->indices[k]] = src->values[k];
        }
    } else {
        const int8_t *s = src->dense_int8;
        const float inv127 = FCE_SEM_UNIT_POS / FCE_SEM_INT8_MAX;
        /* Do NOT replace with fce_init_f32_from_i8_768() — see note in fce_sem_vec_add_scaled. */
        for (int d = 0; d < FCE_SEM_DIM; d++) {
            dst->v[d] = inv127 * (float)s[d];
        }
    }
}

/* Inline helper: add weighted source into target.
 * Sparse path: ~8 operations, ~48 bytes source memory traffic.
 * Dense path: 768 mul-adds with int8→float conversion, ~768 bytes traffic. */
static inline void sem_vec_add_src_scaled(fce_sem_vec_t *dst, const fce_sem_src_entry_t *src,
                                          float scale) {
    if (src->is_sparse) {
        for (int k = 0; k < src->nnz; k++) {
            dst->v[src->indices[k]] += scale * src->values[k];
        }
    } else {
        const int8_t *s = src->dense_int8;
        const float mul = scale * (FCE_SEM_UNIT_POS / FCE_SEM_INT8_MAX);
        /* Do NOT replace with fce_axpy_i8_768() — see note in fce_sem_vec_add_scaled. */
        for (int d = 0; d < FCE_SEM_DIM; d++) {
            dst->v[d] += mul * (float)s[d];
        }
    }
}

/* Pass 1 context: uses sparse/int8 tagged sources (most memory-efficient).
 * R4: writes directly to int8 pass1_q via per-worker scratch normalization.
 * Falls back to float32 enriched_vecs if enriched_vecs_q is NULL. */
typedef struct {
    fce_sem_vec_t *enriched_vecs;       /* float32 fallback (NULL in R4 path) */
    int8_t *enriched_vecs_q;            /* R4: int8 output (NULL in fallback path) */
    int8_t *pass1_q;                    /* R4: int8 pass1 output (NULL in fallback path) */
    fce_sem_vec_t *scratch;             /* R4: per-worker scratch for normalize-before-quantize */
    const fce_sem_src_entry_t *src_entries; /* sparse or int8-dense per token */
    int **doc_token_ids;
    const int *doc_token_counts;
    const reverse_index_t *rev;
    int doc_count;
    int entry_count;
    _Atomic int next_chunk;
    int num_chunks;
    int chunk_size;

    /* Cache-blocked tiling parameters */
    int tile_size; /* targets per L2-resident tile */
} cooccur_sparse_ctx_t;

/* Accumulate co-occurrence context for a single target token into `target`.
 * Reads neighbors within ±FCE_SEM_WINDOW positions across all documents that
 * reference this token id via the reverse index. */
static void cooccur_sparse_one_target(cooccur_sparse_ctx_t *cc, int tid, fce_sem_vec_t *target) {
    int occ_start = cc->rev->offsets[tid];
    int occ_end = cc->rev->offsets[tid + 1];
    for (int p = occ_start; p < occ_end; p++) {
        int d = cc->rev->flat[p].doc_id;
        int i = cc->rev->flat[p].pos;
        int *ids = cc->doc_token_ids[d];
        int len = cc->doc_token_counts[d];
        for (int w = -FCE_SEM_WINDOW; w <= FCE_SEM_WINDOW; w++) {
            if (w == 0) {
                continue;
            }
            int j = i + w;
            if (j < 0 || j >= len) {
                continue;
            }
            int nid = ids[j];
            if (nid < 0) {
                continue;
            }
            float weight = FCE_SEM_UNIT_POS / (float)abs(w);
            sem_vec_add_src_scaled(target, &cc->src_entries[nid], weight);
        }
    }
}

static void cooccur_worker_sparse(int worker_id, void *ctx_ptr) {
    cooccur_sparse_ctx_t *cc = ctx_ptr;
    while (true) {
        int ci =
            atomic_fetch_add_explicit(&cc->next_chunk, FCE_SEM_ATOMIC_INC, memory_order_relaxed);
        if (ci >= cc->num_chunks) {
            break;
        }
        int chunk_start = ci * cc->chunk_size;
        int chunk_end = chunk_start + cc->chunk_size;
        if (chunk_end > cc->entry_count) {
            chunk_end = cc->entry_count;
        }

        /* Cache-blocked target tiling: process tile_size targets at a time so
         * their vectors stay resident in L2 cache during their accumulation. */
        for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += cc->tile_size) {
            int tile_end = tile_start + cc->tile_size;
            if (tile_end > chunk_end) {
                tile_end = chunk_end;
            }
            for (int tid = tile_start; tid < tile_end; tid++) {
                if (cc->enriched_vecs_q && cc->pass1_q && cc->scratch) {
                    /* R4: accumulate into per-worker scratch, normalize, quantize
                     * directly to int8 pass1_q. No float32 enriched_vecs needed. */
                    fce_sem_vec_t *scratch = &cc->scratch[worker_id];
                    sem_target_init_from_src(scratch, &cc->src_entries[tid]);
                    cooccur_sparse_one_target(cc, tid, scratch);
                    fce_sem_normalize(scratch);
                    fce_quantize_f32_768(&cc->pass1_q[(size_t)tid * FCE_SEM_DIM], scratch->v);
                } else {
                    /* Fallback: accumulate into float32 enriched_vecs, normalize only.
                     * Quantization happens later in finalize_pass2. */
                    sem_target_init_from_src(&cc->enriched_vecs[tid], &cc->src_entries[tid]);
                    cooccur_sparse_one_target(cc, tid, &cc->enriched_vecs[tid]);
                    fce_sem_normalize(&cc->enriched_vecs[tid]);
                }
            }
        }
    }
}

/* Pass 2 (RRI) context: uses int8-quantized pass1 vectors as source.
 * Pass1 outputs are dense float32 post-normalized, values in [-1,1].
 * We quantize to int8 once (×127) to cut source memory traffic 4x.
 * R4: normalizes, quantizes, blends, and writes directly to enriched_vecs_q. */
typedef struct {
    fce_sem_vec_t *enriched_vecs;       /* float32 fallback (NULL in R4 path) */
    int8_t *enriched_vecs_q;            /* R4: int8 output (NULL in fallback path) */
    const int8_t *pass1_q; /* [entry_count × FCE_SEM_DIM] int8 quantized pass1 */
    int **doc_token_ids;
    const int *doc_token_counts;
    const reverse_index_t *rev;
    int doc_count;
    int entry_count;
    _Atomic int next_chunk;
    int num_chunks;
    int chunk_size;
    int tile_size;
} cooccur_int8_ctx_t;

static inline void sem_vec_add_int8_scaled(fce_sem_vec_t *dst, const int8_t *src, float scale) {
    const float mul = scale * (FCE_SEM_UNIT_POS / FCE_SEM_INT8_MAX);
    /* Do NOT replace with fce_axpy_i8_768() — see note in fce_sem_vec_add_scaled. */
    for (int d = 0; d < FCE_SEM_DIM; d++) {
        dst->v[d] += mul * (float)src[d];
    }
}

/* RRI pass 2 accumulator for a single target token, reading int8-quantized
 * pass1 vectors as the source. */
static void cooccur_int8_one_target(cooccur_int8_ctx_t *cc, int tid, fce_sem_vec_t *target) {
    int occ_start = cc->rev->offsets[tid];
    int occ_end = cc->rev->offsets[tid + 1];
    for (int p = occ_start; p < occ_end; p++) {
        int d = cc->rev->flat[p].doc_id;
        int i = cc->rev->flat[p].pos;
        int *ids = cc->doc_token_ids[d];
        int len = cc->doc_token_counts[d];
        for (int w = -FCE_SEM_WINDOW; w <= FCE_SEM_WINDOW; w++) {
            if (w == 0) {
                continue;
            }
            int j = i + w;
            if (j < 0 || j >= len) {
                continue;
            }
            int nid = ids[j];
            if (nid < 0) {
                continue;
            }
            float weight = FCE_SEM_UNIT_POS / (float)abs(w);
            sem_vec_add_int8_scaled(target, &cc->pass1_q[(size_t)nid * FCE_SEM_DIM], weight);
        }
    }
}

static void cooccur_worker_int8(int worker_id, void *ctx_ptr) {
    (void)worker_id;
    cooccur_int8_ctx_t *cc = ctx_ptr;
    while (true) {
        int ci =
            atomic_fetch_add_explicit(&cc->next_chunk, FCE_SEM_ATOMIC_INC, memory_order_relaxed);
        if (ci >= cc->num_chunks) {
            break;
        }
        int chunk_start = ci * cc->chunk_size;
        int chunk_end = chunk_start + cc->chunk_size;
        if (chunk_end > cc->entry_count) {
            chunk_end = cc->entry_count;
        }

        for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += cc->tile_size) {
            int tile_end = tile_start + cc->tile_size;
            if (tile_end > chunk_end) {
                tile_end = chunk_end;
            }
            for (int tid = tile_start; tid < tile_end; tid++) {
                /* RRI pass 2 starts from zero (no self-init) */
                fce_sem_vec_t acc = {0};
                cooccur_int8_one_target(cc, tid, &acc);

                if (cc->enriched_vecs_q) {
                    /* R4: quantize pass2 to int8, blend with normalized pass1,
                     * normalize the blend, quantize to int8, write to enriched_vecs_q.
                     * Matches original code's exact order of operations. */
                    int8_t p2_q[FCE_SEM_DIM];
                    fce_quantize_f32_768(p2_q, acc.v);
                    const int8_t *p1 = &cc->pass1_q[(size_t)tid * FCE_SEM_DIM];
                    int8_t *out = &cc->enriched_vecs_q[(size_t)tid * FCE_SEM_DIM];
                    /* Blend quantized pass2 with normalized pass1 in float32. */
                    fce_sem_vec_t blended = {0};
                    for (int d = 0; d < FCE_SEM_DIM; d++) {
                        blended.v[d] =
                            (FCE_SEM_RRI_BETA * (float)p1[d]) +
                            (FCE_SEM_RRI_ALPHA * (float)p2_q[d]);
                    }
                    fce_sem_normalize(&blended);
                    fce_quantize_f32_768(out, blended.v);
                } else {
                    /* Fallback: quantize, blend with normalized pass1,
                     * normalize into float32 enriched_vecs. */
                    int8_t p2_q[FCE_SEM_DIM];
                    fce_quantize_f32_768(p2_q, acc.v);
                    const int8_t *p1 = &cc->pass1_q[(size_t)tid * FCE_SEM_DIM];
                    for (int d = 0; d < FCE_SEM_DIM; d++) {
                        cc->enriched_vecs[tid].v[d] =
                            (FCE_SEM_RRI_BETA * (float)p1[d]) +
                            (FCE_SEM_RRI_ALPHA * (float)p2_q[d]);
                    }
                    fce_sem_normalize(&cc->enriched_vecs[tid]);
                }
            }
        }
    }
}

typedef struct {
    corpus_entry_t *entries;
    fce_sem_src_entry_t *src_entries;
    int entry_count;
    _Atomic int next_idx;
} src_build_ctx_t;

/* Build one src_entry for a token: dense float32 reference if in nomic vocab,
 * sparse inline representation otherwise. Collisions in the sparse hash are
 * merged and zeros filtered so the final representation is exactly the same
 * mathematical vector that the old dense path produced. */
static void build_src_entry(const char *token, fce_sem_src_entry_t *out) {
    memset(out, 0, sizeof(*out));
    if (!token) {
        out->is_sparse = 1;
        out->nnz = 0;
        return;
    }
    /* Dense path: direct int8 pointer into pretrained blob (zero-copy). */
    if (g_pretrained_map) {
        void *idx_ptr = fce_ht_get(g_pretrained_map, token);
        if (idx_ptr) {
            int idx = ptr_to_token_idx(idx_ptr);
            if (idx >= 0 && idx < FCE_PRETRAINED_TOKEN_COUNT) {
                out->is_sparse = 0;
                out->dense_int8 = fce_pretrained_vec_at(idx);
                return;
            }
        }
    }
    /* Sparse path: compute 8 hash positions with collision merging. */
    out->is_sparse = 1;
    uint16_t tmp_idx[FCE_SEM_SPARSE_NNZE];
    float tmp_val[FCE_SEM_SPARSE_NNZE];
    int count = 0;
    uint64_t seed = XXH3_64bits(token, strlen(token) + 1);
    for (int i = 0; i < FCE_SEM_SPARSE_NNZE; i++) {
        uint32_t le_i = fce_htole32((uint32_t)i);
        uint64_t h = XXH3_64bits_withSeed(&le_i, sizeof(le_i), seed + FCE_RI_SEED_BASE);
        int pos = (int)(h % FCE_SEM_DIM);
        float sign = (h & 1) ? FCE_SEM_UNIT_POS : -FCE_SEM_UNIT_POS;
        /* Merge collisions */
        int found = FCE_NOT_FOUND;
        for (int j = 0; j < count; j++) {
            if (tmp_idx[j] == (uint16_t)pos) {
                found = j;
                break;
            }
        }
        if (found >= 0) {
            tmp_val[found] += sign;
        } else {
            tmp_idx[count] = (uint16_t)pos;
            tmp_val[count] = sign;
            count++;
        }
    }
    /* Filter zeros */
    int nnz = 0;
    for (int j = 0; j < count; j++) {
        if (tmp_val[j] != 0.0F) {
            out->indices[nnz] = tmp_idx[j];
            out->values[nnz] = tmp_val[j];
            nnz++;
        }
    }
    out->nnz = (uint8_t)nnz;
}

static void src_build_worker(int worker_id, void *ctx_ptr) {
    (void)worker_id;
    src_build_ctx_t *sc = ctx_ptr;
    while (true) {
        int start = atomic_fetch_add_explicit(&sc->next_idx, FCE_SEM_WORKER_STACK_CAP,
                                              memory_order_relaxed);
        if (start >= sc->entry_count) {
            break;
        }
        int end = start + FCE_SEM_WORKER_STACK_CAP;
        if (end > sc->entry_count) {
            end = sc->entry_count;
        }
        for (int i = start; i < end; i++) {
            build_src_entry(sc->entries[i].token, &sc->src_entries[i]);
        }
    }
}

/* Build reverse index: token_id → list of (doc_id, position) pairs.
 * SEQUENTIAL (fast: just pointer arithmetic + flat array fill). */
static reverse_index_t *build_reverse_index(fce_sem_corpus_t *corpus) {
    reverse_index_t *rev = calloc(1, sizeof(reverse_index_t));
    if (!rev) {
        return NULL;
    }
    /* Phase A: count occurrences per token */
    uint32_t *counts = calloc((size_t)corpus->entry_count + 1, sizeof(uint32_t));
    if (!counts) {
        free(rev);
        return NULL;
    }
    int64_t total = 0;
    for (int d = 0; d < corpus->doc_count; d++) {
        int *ids = corpus->doc_token_ids[d];
        int len = corpus->doc_token_counts[d];
        for (int i = 0; i < len; i++) {
            int tid = ids[i];
            if (tid >= 0 && tid < corpus->entry_count) {
                if (counts[tid] == UINT32_MAX) {
                    fprintf(stderr, "semantic: per-token count overflow (token %d)\n", tid);
                    free(counts);
                    free(rev);
                    return NULL;
                }
                counts[tid]++;
                total++;
            }
        }
    }
    /* Sanity check: total occurrences must fit in int32 and not exceed reasonable
     * memory limits (1B occurrences ≈ 8GB for cooccur_pos_t). Exceeding this would
     * indicate either pathological input or adversarial DoS. */
    const int64_t MAX_OCCURRENCES = ((int64_t)1 << 30);
    if (total > MAX_OCCURRENCES) {
        fprintf(stderr, "semantic: corpus too large: %" PRId64 " occurrences > %" PRId64 " max\n", total, MAX_OCCURRENCES);
        free(counts);
        free(rev);
        return NULL;
    }
    if (total > (int64_t)INT32_MAX) {
        fprintf(stderr, "semantic: reverse index overflow (total=%" PRId64 ")\n", total);
        free(counts);
        free(rev);
        return NULL;
    }
    /* Phase B: exclusive prefix sum → offsets[] */
    rev->offsets = malloc(((size_t)corpus->entry_count + 1) * sizeof(int));
    if (!rev->offsets) {
        free(counts);
        free(rev);
        return NULL;
    }
    int64_t running = 0;
    for (int t = 0; t < corpus->entry_count; t++) {
        rev->offsets[t] = (int)running;
        running += counts[t];
        counts[t] = 0; /* reuse as per-token fill cursor */
    }
    rev->offsets[corpus->entry_count] = (int)running;
    /* Phase C: fill flat array. Ensure allocation size > 0 even for empty
     * corpora (avoids malloc(0) which is implementation-defined). */
    size_t flat_bytes = (total > 0 ? (size_t)total : 1) * sizeof(cooccur_pos_t);
    rev->flat = malloc(flat_bytes);
    if (!rev->flat) {
        free(rev->offsets);
        free(counts);
        free(rev);
        return NULL;
    }
    for (int d = 0; d < corpus->doc_count; d++) {
        int *ids = corpus->doc_token_ids[d];
        int len = corpus->doc_token_counts[d];
        for (int i = 0; i < len; i++) {
            int tid = ids[i];
            if (tid >= 0 && tid < corpus->entry_count) {
                int slot = rev->offsets[tid] + counts[tid]++;
                rev->flat[slot].doc_id = (int32_t)d;
                rev->flat[slot].pos = (int32_t)i;
            }
        }
    }
    free(counts);
    return rev;
}

static void free_reverse_index(reverse_index_t *rev) {
    if (!rev) {
        return;
    }
    free(rev->offsets);
    free(rev->flat);
    free(rev);
}

/* Bundle of parameters shared by the finalize sub-phases. */
typedef struct {
    fce_sem_corpus_t *corpus;
    reverse_index_t *rev;
    fce_sem_src_entry_t *src_entries;
    int worker_count;
    int num_chunks;
    int chunk_size;
    int tile_size;
    fce_parallel_for_opts_t opts;
    int8_t *_pass1_q; /* internal: pass1_q passed from pass1 to pass2 */
} finalize_params_t;

/* Sub-phase 1: build tagged source vectors (sparse or dense-int8) in parallel. */
static void finalize_build_sources(finalize_params_t *p) {
    src_build_ctx_t sc = {
        .entries = p->corpus->entries,
        .src_entries = p->src_entries,
        .entry_count = p->corpus->entry_count,
    };
    atomic_init(&sc.next_idx, 0);
    fce_parallel_for(p->worker_count, src_build_worker, &sc, p->opts);
}

/* Comparator for qsort of doc IDs (ascending). */
static int int_cmp_asc(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

/* Sub-phases 2+3: co-occurrence pass 1 + normalize + quantize.
 * R4: per-token normalize and quantize directly to int8 pass1_q.
 * Fallback: normalize to float32 enriched_vecs (quantization in pass2). */
static void finalize_pass1(finalize_params_t *p) {
    fce_sem_vec_t *scratch = NULL;
    int8_t *pass1_q = NULL;

    if (p->corpus->enriched_vecs_q) {
        /* R4 path: allocate per-worker scratch + int8 pass1 output.
         * scratch[i] is the float32 tile buffer for worker i. */
        scratch = calloc((size_t)p->worker_count, sizeof(fce_sem_vec_t));
        pass1_q = malloc((size_t)p->corpus->entry_count * FCE_SEM_DIM * sizeof(int8_t));
        if (!scratch || !pass1_q) {
            free(scratch);
            free(pass1_q);
            scratch = NULL;
            pass1_q = NULL;
            /* Fallback requires float32 enriched_vecs; allocate if not present. */
            if (!p->corpus->enriched_vecs) {
                p->corpus->enriched_vecs = calloc((size_t)p->corpus->entry_count,
                                                  sizeof(fce_sem_vec_t));
                if (!p->corpus->enriched_vecs) {
                    fce_log(FCE_LOG_WARN, "OOM during R4 fallback alloc; finalize will fail", NULL);
                    p->_pass1_q = NULL;
                    return;
                }
            }
            fce_log(FCE_LOG_WARN, "OOM during R4 pass1 scratch; using fallback path", NULL);
        }
    }

    cooccur_sparse_ctx_t cc = {
        .enriched_vecs = p->corpus->enriched_vecs,
        .enriched_vecs_q = p->corpus->enriched_vecs_q,
        .pass1_q = pass1_q,
        .scratch = scratch,
        .src_entries = p->src_entries,
        .doc_token_ids = p->corpus->doc_token_ids,
        .doc_token_counts = p->corpus->doc_token_counts,
        .rev = p->rev,
        .doc_count = p->corpus->doc_count,
        .entry_count = p->corpus->entry_count,
        .num_chunks = p->num_chunks,
        .chunk_size = p->chunk_size,
        .tile_size = p->tile_size,
    };
    atomic_init(&cc.next_chunk, 0);
    fce_parallel_for(p->worker_count, cooccur_worker_sparse, &cc, p->opts);

    free(scratch);

    /* Fallback path: worker already normalized enriched_vecs (line 1340).
     * No second normalization needed — normalizing a unit vector is idempotent
     * but wastes CPU (~150K FLOPs per entry). */

    /* Store pass1_q on corpus for pass2 to read. */
    p->_pass1_q = pass1_q;
}

/* Sub-phases 4+5: run RRI pass 2 + blend in one pass.
 * R4: quantizes, normalizes, blends, and writes directly to enriched_vecs_q.
 * Fallback: quantizes, blends, normalizes into float32 enriched_vecs. */
static void finalize_pass2(finalize_params_t *p) {
    int8_t *pass1_q = p->_pass1_q;

    if (!pass1_q) {
        /* Fallback: quantize normalized float32 pass1 to int8. */
        pass1_q = malloc((size_t)p->corpus->entry_count * FCE_SEM_DIM * sizeof(int8_t));
        if (!pass1_q) {
            fce_log(FCE_LOG_WARN, "OOM during pass2 quantization; using pass1 result", NULL);
            return;
        }
        for (int i = 0; i < p->corpus->entry_count; i++) {
            fce_quantize_f32_768(&pass1_q[(size_t)i * FCE_SEM_DIM],
                                 p->corpus->enriched_vecs[i].v);
        }
    }

    cooccur_int8_ctx_t cc = {
        .enriched_vecs = p->corpus->enriched_vecs,
        .enriched_vecs_q = p->corpus->enriched_vecs_q,
        .pass1_q = pass1_q,
        .doc_token_ids = p->corpus->doc_token_ids,
        .doc_token_counts = p->corpus->doc_token_counts,
        .rev = p->rev,
        .doc_count = p->corpus->doc_count,
        .entry_count = p->corpus->entry_count,
        .num_chunks = p->num_chunks,
        .chunk_size = p->chunk_size,
        .tile_size = p->tile_size,
    };
    atomic_init(&cc.next_chunk, 0);
    fce_parallel_for(p->worker_count, cooccur_worker_int8, &cc, p->opts);

    /* Don't free pass1_q here — caller frees params._pass1_q after this returns.
     * In fallback path, local pass1_q is freed above. In R4 path, caller frees. */
    if (!p->corpus->enriched_vecs_q) {
        free(pass1_q);
    }
}

/* Worker for parallel doc-vector construction in finalize.
 * R4: always uses int8 enriched_vecs_q (dequantizes on the fly).
 * Fallback: if enriched_vecs_q is NULL, uses float32 enriched_vecs. */
typedef struct {
    fce_sem_vec_t *doc_vectors;
    const fce_sem_vec_t *enriched_vecs;  /* float32 fallback (NULL in R4 path) */
    const int8_t *enriched_vecs_q;       /* R4: int8 path */
    int **doc_token_ids;
    int *doc_token_counts;
    int doc_count;
    int entry_count;
    _Atomic int next_doc;
} docvec_ctx_t;

static void docvec_build_worker(int wid, void *ctx_ptr) {
    (void)wid;
    docvec_ctx_t *dc = ctx_ptr;
    for (;;) {
        int d = atomic_fetch_add_explicit(&dc->next_doc, 1, memory_order_relaxed);
        if (d >= dc->doc_count) break;
        fce_sem_vec_t dv = {0};
        int *ids = dc->doc_token_ids[d];
        int ntok = dc->doc_token_counts[d];
        for (int t = 0; t < ntok; t++) {
            int tid = ids[t];
            if (tid >= 0 && tid < dc->entry_count) {
                if (dc->enriched_vecs_q) {
                    /* R4: dequantize int8 on the fly — ~5 tokens × 768 dims
                     * per doc, negligible vs the normalize at the end. */
                    const int8_t *src = &dc->enriched_vecs_q[(size_t)tid * FCE_SEM_DIM];
                    for (int i = 0; i < FCE_SEM_DIM; i++) {
                        dv.v[i] += (float)src[i];
                    }
                } else {
                    fce_sem_vec_add_scaled(&dv, &dc->enriched_vecs[tid], 1.0f);
                }
            }
        }
        fce_sem_normalize(&dv);
        dc->doc_vectors[d] = dv;
    }
}

/* P0: Worker for parallel doc-vector quantization (float32 → int8).
 * Also computes L2 magnitude of quantized vectors for proper cosine. */
typedef struct {
    const fce_sem_vec_t *doc_vectors;
    int8_t *doc_vectors_q;
    float *doc_vectors_q_inv_mag;
    int doc_count;
    _Atomic int next_doc;
} docvec_quant_ctx_t;

static void docvec_quantize_worker(int wid, void *ctx_ptr) {
    (void)wid;
    docvec_quant_ctx_t *dc = ctx_ptr;
    for (;;) {
        int d = atomic_fetch_add_explicit(&dc->next_doc, 1, memory_order_relaxed);
        if (d >= dc->doc_count) break;
        int8_t *dq = dc->doc_vectors_q + (size_t)d * FCE_SEM_DIM;
        fce_quantize_f32_768(dq, dc->doc_vectors[d].v);
        /* F2: store reciprocal magnitude for multiply-only hot loop. */
        int32_t mag_sq_i32 = fce_dot768_i8(dq, dq);
        float mag_sq = (float)mag_sq_i32;
        dc->doc_vectors_q_inv_mag[d] = (mag_sq > 0.0f) ? 1.0f / sqrtf(mag_sq) : 0.0f;
    }
}

int fce_sem_corpus_finalize(fce_sem_corpus_t *corpus) {
    if (!corpus || corpus->finalized) {
        return 0;
    }

    /* C2: Empty corpus — nothing to enrich. Avoids calloc(0) portability issue
     * (C permits calloc(0) to return NULL without it being an error). */
    if (corpus->entry_count == 0) {
        corpus->finalized = 1;
        return 0;
    }

    ensure_pretrained_map();

    int worker_count = fce_default_worker_count(true);
    if (worker_count < 1) worker_count = 1;
    fce_parallel_for_opts_t opts = {.max_workers = worker_count};

    /* Finer chunks = better load balancing for skewed token distributions. */
    int num_chunks = worker_count * FCE_SEM_COOCCUR_CHUNK;
    if (num_chunks > corpus->entry_count) {
        num_chunks = corpus->entry_count;
    }
    if (num_chunks < 1) {
        num_chunks = 1;
    }
    int chunk_size = (corpus->entry_count + num_chunks - 1) / num_chunks;

    reverse_index_t *rev = build_reverse_index(corpus);
    if (!rev) {
        return -1;
    }
    fce_sem_src_entry_t *src_entries =
        calloc((size_t)corpus->entry_count, sizeof(fce_sem_src_entry_t));
    if (!src_entries) {
        free_reverse_index(rev);
        return -1;
    }

    /* R4: Allocate int8 enriched_vecs_q directly (3 GB float32 eliminated).
     * If this allocation fails, fall back to float32 enriched_vecs and
     * quantize to int8 after pass2 (the old R1 path). */
    corpus->enriched_vecs_q = malloc((size_t)corpus->entry_count * FCE_SEM_DIM * sizeof(int8_t));
    if (!corpus->enriched_vecs_q) {
        /* Fallback: allocate float32 enriched_vecs; pass1+pass2 write here,
         * then R1 quantizes to int8 and frees the float32 array. */
        corpus->enriched_vecs = calloc((size_t)corpus->entry_count, sizeof(fce_sem_vec_t));
        if (!corpus->enriched_vecs) {
            free(src_entries);
            free_reverse_index(rev);
            return -1;
        }
    }

    finalize_params_t params = {
        .corpus = corpus,
        .rev = rev,
        .src_entries = src_entries,
        .worker_count = worker_count,
        .num_chunks = num_chunks,
        .chunk_size = chunk_size,
        .tile_size = FCE_SEM_TILE_SIZE,
        .opts = opts,
    };
    finalize_build_sources(&params);
    finalize_pass1(&params);
    finalize_pass2(&params);
    free(params._pass1_q);  /* R4 path: pass1_q consumed by pass2, no longer needed */

    /* Fallback path: quantize float32 enriched_vecs to int8, then free float32.
     * R4 path: enriched_vecs_q is already populated by pass1+pass2. */
    if (corpus->enriched_vecs && corpus->entry_count > 0) {
        if (!corpus->enriched_vecs_q) {
            corpus->enriched_vecs_q = malloc((size_t)corpus->entry_count * FCE_SEM_DIM * sizeof(int8_t));
        }
        if (corpus->enriched_vecs_q) {
            for (int i = 0; i < corpus->entry_count; i++) {
                fce_quantize_f32_768(&corpus->enriched_vecs_q[(size_t)i * FCE_SEM_DIM],
                                     corpus->enriched_vecs[i].v);
            }
            free(corpus->enriched_vecs);
            corpus->enriched_vecs = NULL;
        }
    }

    /* Compute document vectors before freeing doc_token_ids.
     * doc_vectors[d] = normalize(sum of enriched_vecs[t] for t in doc d).
     * Parallelized: each worker processes a disjoint range of documents.
     * If enriched_vecs_q is available, dequantizes int8 on the fly (saves
     * ~2.2 GB by not keeping float32 enriched_vecs during docvec). */
    corpus->doc_vectors = calloc((size_t)corpus->doc_count, sizeof(fce_sem_vec_t));
    if (!corpus->doc_vectors) {
        /* M3: doc_vectors is required for search — fail loudly rather than
         * producing a silently-degraded corpus that scores everything as 0.0. */
        free(src_entries);
        free_reverse_index(rev);
        fce_log_error("fce_sem_corpus_finalize", "detail", "doc_vectors calloc failed");
        return -1;
    }
    docvec_ctx_t dctx = {
        .doc_vectors = corpus->doc_vectors,
        .enriched_vecs = corpus->enriched_vecs,
        .enriched_vecs_q = corpus->enriched_vecs_q,
        .doc_token_ids = corpus->doc_token_ids,
        .doc_token_counts = corpus->doc_token_counts,
        .doc_count = corpus->doc_count,
        .entry_count = corpus->entry_count,
    };
    atomic_init(&dctx.next_doc, 0);
    fce_parallel_for(worker_count, docvec_build_worker, &dctx, opts);

    /* P0: Quantize float32 doc vectors to int8 for brute-force bandwidth reduction.
     * 4 bytes/doc → 1 byte/doc; SIMD int8 dot is ~4x faster than float32.
     * Also stores per-doc quantized magnitudes for proper cosine normalization. */
    if (corpus->doc_count > 0) {
        corpus->doc_vectors_q = calloc((size_t)corpus->doc_count * FCE_SEM_DIM, sizeof(int8_t));
        corpus->doc_vectors_q_inv_mag = calloc((size_t)corpus->doc_count, sizeof(float));
        if (corpus->doc_vectors_q && corpus->doc_vectors_q_inv_mag) {
            docvec_quant_ctx_t qctx = {
                .doc_vectors = corpus->doc_vectors,
                .doc_vectors_q = corpus->doc_vectors_q,
                .doc_vectors_q_inv_mag = corpus->doc_vectors_q_inv_mag,
                .doc_count = corpus->doc_count,
            };
            atomic_init(&qctx.next_doc, 0);
            fce_parallel_for(worker_count, docvec_quantize_worker, &qctx, opts);
        } else {
            /* H1+M2: partial quantization OOM — free both to prevent NULL deref in search.
             * Search functions check doc_vectors_q and return early if NULL. */
            free(corpus->doc_vectors_q); corpus->doc_vectors_q = NULL;
            free(corpus->doc_vectors_q_inv_mag); corpus->doc_vectors_q_inv_mag = NULL;
        }
    }

    /* F1: Free float32 doc_vectors — int8 is now the only representation.
     * Quantization is complete; no codepath reads float32 after this point. */
    free(corpus->doc_vectors);
    corpus->doc_vectors = NULL;

    free(src_entries);
    free_reverse_index(rev);

    /* Build inverted index: token_id → [doc_ids containing that token].
     * Used by the fast search path for keyword candidate retrieval. */
    if (corpus->doc_token_ids && corpus->entry_count > 0) {
        int ntok = corpus->entry_count;
        uint32_t *occ_counts = calloc((size_t)ntok + 1, sizeof(uint32_t));
        if (occ_counts) {
            /* Count occurrences per token across all docs. */
            for (int d = 0; d < corpus->doc_count; d++) {
                for (int i = 0; i < corpus->doc_token_counts[d]; i++) {
                    int t = corpus->doc_token_ids[d][i];
                    if (t < 0 || t >= ntok) continue;
                    occ_counts[t]++;
                }
            }
            /* Prefix sum → offsets for the un-deduped array. */
            uint32_t total_occ = 0;
            for (int t = 0; t < ntok; t++) {
                uint32_t c = occ_counts[t];
                occ_counts[t] = total_occ;
                total_occ += c;
            }
            occ_counts[ntok] = total_occ;

            int *tmp_ids = (int *)malloc((size_t)total_occ * sizeof(int));
            if (tmp_ids) {
                /* Fill doc_ids with duplicates. */
                uint32_t *cursor = malloc(((size_t)ntok + 1) * sizeof(uint32_t));
                if (cursor) {
                    for (int t = 0; t <= ntok; t++) cursor[t] = occ_counts[t];
                    for (int d = 0; d < corpus->doc_count; d++) {
                        for (int i = 0; i < corpus->doc_token_counts[d]; i++) {
                            int t = corpus->doc_token_ids[d][i];
                            if (t < 0 || t >= ntok) continue;
                            tmp_ids[cursor[t]++] = d;
                        }
                    }

                    /* Sort each token's doc list. */
                    for (int t = 0; t < ntok; t++) {
                        int start = (int)occ_counts[t];
                        int end = (int)occ_counts[t + 1];
                        int len = end - start;
                        if (len > 64) {
                            qsort(tmp_ids + start, (size_t)len, sizeof(int), int_cmp_asc);
                        } else {
                            for (int i = 1; i < len; i++) {
                                int key = tmp_ids[start + i];
                                int j = i - 1;
                                while (j >= 0 && tmp_ids[start + j] > key) {
                                    tmp_ids[start + j + 1] = tmp_ids[start + j];
                                    j--;
                                }
                                tmp_ids[start + j + 1] = key;
                            }
                        }
                    }
                    /* Count unique doc IDs per token (lightweight pass on sorted data). */
                    uint32_t total_uniq = 0;
                    for (int t = 0; t < ntok; t++) {
                        int start = (int)occ_counts[t];
                        int end = (int)occ_counts[t + 1];
                        int nu = (start < end) ? 1 : 0;
                        for (int i = start + 1; i < end; i++) {
                            if (tmp_ids[i] != tmp_ids[i - 1]) nu++;
                        }
                        total_uniq += (uint32_t)nu;
                    }
                    /* Dedup-copy: skip duplicates in sorted lists. */
                    corpus->inv_offsets = (int *)malloc(((size_t)ntok + 1) * sizeof(int));
                    corpus->inv_doc_ids = (int *)malloc((size_t)total_uniq * sizeof(int));
                    if (corpus->inv_offsets && corpus->inv_doc_ids) {
                        uint32_t w = 0;
                        for (int t = 0; t < ntok; t++) {
                            corpus->inv_offsets[t] = (int)w;
                            int start = (int)occ_counts[t];
                            int end = (int)occ_counts[t + 1];
                            int prev = -1;
                            for (int i = start; i < end; i++) {
                                if (tmp_ids[i] != prev) {
                                    corpus->inv_doc_ids[w++] = tmp_ids[i];
                                    prev = tmp_ids[i];
                                }
                            }
                        }
                        corpus->inv_offsets[ntok] = (int)w;
                    } else {
                        free(corpus->inv_offsets); corpus->inv_offsets = NULL;
                        free(corpus->inv_doc_ids); corpus->inv_doc_ids = NULL;
                    }
                    free(cursor);
                }
                free(tmp_ids);
            }
            free(occ_counts);
        }
    }

    /* P3: doc_token_ids is dead after finalize — free to reclaim RSS. */
    if (corpus->doc_token_ids) {
        for (int d = 0; d < corpus->doc_count; d++) {
            free(corpus->doc_token_ids[d]);
        }
        free(corpus->doc_token_ids);
        corpus->doc_token_ids = NULL;
    }
    free(corpus->doc_token_counts);
    corpus->doc_token_counts = NULL;

    /* Force the allocator to release freed pages back to the OS.
     * After finalize frees ~5 GB of transient buffers, the allocator may
     * retain pages in its free lists. This brings post-build RSS closer
     * to the actual live allocation (~1.1 GB). */
#if defined(__linux__)
    malloc_trim(0);
#if defined(M_PURGE)
    mallopt(M_PURGE, 0);
#endif
#elif defined(__APPLE__)
    extern void malloc_zone_pressure_relief(malloc_zone_t *zone, size_t goal);
    malloc_zone_pressure_relief(NULL, 0);
#endif

    corpus->finalized = true;
    return 0;
}

float fce_sem_corpus_idf(const fce_sem_corpus_t *corpus, const char *token) {
    if (!corpus || !token || corpus->doc_count == 0) {
        return 0.0F;
    }
    int idx = ptr_to_token_idx(fce_ht_get(corpus->token_map, token));
    if (idx < 0 || idx >= corpus->entry_count) {
        return 0.0F;
    }
    int df = corpus->entries[idx].doc_freq;
    if (df <= 0) {
        return 0.0F;
    }
    return logf((float)corpus->doc_count / (float)df);
}

const fce_sem_vec_t *fce_sem_corpus_ri_vec(const fce_sem_corpus_t *corpus, const char *token) {
    if (!corpus || !token) {
        return NULL;
    }
    int idx = ptr_to_token_idx(fce_ht_get(corpus->token_map, token));
    if (idx < 0 || idx >= corpus->entry_count) {
        return NULL;
    }
    /* R1: dequantize int8 vector into thread-local scratch on demand.
     * WARNING: the returned pointer is only valid until the next call to
     * this function from the SAME thread. Each function has its own
     * _Thread_local scratch, so calling fce_sem_corpus_token_at() does NOT
     * invalidate a pointer from this function. Safe when consumed
     * immediately (as in buildFunc / scoring), but NOT safe to store. */
    if (corpus->enriched_vecs_q) {
        static _Thread_local fce_sem_vec_t tl_dequant;
        const int8_t *src = &corpus->enriched_vecs_q[(size_t)idx * FCE_SEM_DIM];
        const float inv127 = 1.0f / FCE_SEM_INT8_MAX;
        for (int i = 0; i < FCE_SEM_DIM; i++) {
            tl_dequant.v[i] = inv127 * (float)src[i];
        }
        return &tl_dequant;
    }
    return corpus->enriched_vecs ? &corpus->enriched_vecs[idx] : NULL;
}

int fce_sem_corpus_doc_count(const fce_sem_corpus_t *corpus) {
    return corpus ? corpus->doc_count : 0;
}

int fce_sem_corpus_token_count(const fce_sem_corpus_t *corpus) {
    return corpus ? corpus->entry_count : 0;
}

const char *fce_sem_corpus_token_at(const fce_sem_corpus_t *corpus, int index,
                                    const fce_sem_vec_t **out_vec, float *out_idf) {
    if (!corpus || index < 0 || index >= corpus->entry_count) {
        return NULL;
    }
    if (out_vec) {
        if (corpus->enriched_vecs_q) {
            /* R1: dequantize int8 vector into thread-local scratch.
             * WARNING: valid only until the next call to this function from
             * the same thread. Each function has its own _Thread_local scratch,
             * so calling fce_sem_corpus_ri_vec() does NOT invalidate this pointer. */
            static _Thread_local fce_sem_vec_t tl_dequant;
            const int8_t *src = &corpus->enriched_vecs_q[(size_t)index * FCE_SEM_DIM];
            const float inv127 = 1.0f / FCE_SEM_INT8_MAX;
            for (int i = 0; i < FCE_SEM_DIM; i++) {
                tl_dequant.v[i] = inv127 * (float)src[i];
            }
            *out_vec = &tl_dequant;
        } else {
            *out_vec = corpus->enriched_vecs ? &corpus->enriched_vecs[index] : NULL;
        }
    }
    if (out_idf && corpus->doc_count > 0) {
        int df = corpus->entries[index].doc_freq;
        *out_idf = df > 0 ? logf((float)corpus->doc_count / (float)df) : 0.0F;
    }
    return corpus->entries[index].token;
}

static void fce_free_ht_kv(const char *key, void *value, void *userdata) {
    (void)userdata;
    (void)value; /* value is an encoded integer pointer, not heap-allocated */
    free((void *)key);
}

void fce_sem_corpus_free(fce_sem_corpus_t *corpus) {
    if (!corpus) {
        return;
    }
    /* Don't free entries[i].token — they point at hash table interned keys,
     * which are freed by fce_free_ht_kv below. */
    free(corpus->entries);
    free(corpus->enriched_vecs);
    free(corpus->enriched_vecs_q);
    free(corpus->doc_vectors_q);
    free(corpus->doc_vectors_q_inv_mag);
    if (corpus->doc_token_ids) {
        for (int d = 0; d < corpus->doc_count; d++) {
            free(corpus->doc_token_ids[d]);
        }
        free(corpus->doc_token_ids);
    }
    free(corpus->doc_token_counts);
    if (corpus->token_map) {
        fce_ht_foreach(corpus->token_map, fce_free_ht_kv, NULL);
        fce_ht_free(corpus->token_map);
    }
    free(corpus->inv_offsets);
    free(corpus->inv_doc_ids);
    free(corpus);
}

/* ── Combined scoring ────────────────────────────────────────────── */

/* Internal proximity with precomputed path_a slash count (avoids redundant walk). */
static float proximity_internal(const char *path_a, int total_dirs_a, const char *path_b) {
    if (!path_a || !path_b) {
        return FCE_SEM_UNIT_POS;
    }
    int shared_dirs = 0;
    const char *a = path_a;
    const char *b = path_b;
    while (1) {
        const char *ea = a;
        while (*ea && *ea != '/') ea++;
        const char *eb = b;
        while (*eb && *eb != '/') eb++;
        int len_a = (int)(ea - a);
        int len_b = (int)(eb - b);
        if (len_a == 0 && len_b == 0) break;
        if (len_a != len_b || memcmp(a, b, (size_t)len_a) != 0) break;
        shared_dirs++;
        a = *ea ? ea + 1 : ea;
        b = *eb ? eb + 1 : eb;
    }
    int total_dirs_b = fce_count_slashes(path_b);
    int max_dirs = total_dirs_a > total_dirs_b ? total_dirs_a : total_dirs_b;
    if (max_dirs == 0) {
        return FCE_SEM_UNIT_POS;
    }
    float ratio = (float)shared_dirs / (float)(max_dirs + 1);
    return FCE_SEM_UNIT_POS + (ratio * FCE_SEM_PROX_MAX_BOOST);
}

float fce_sem_proximity(const char *path_a, const char *path_b) {
    if (!path_a || !path_b) return FCE_SEM_UNIT_POS;
    int total_dirs_a = fce_count_slashes(path_a);
    return proximity_internal(path_a, total_dirs_a, path_b);
}

/* Sparse cosine over two pre-sorted (index, weight) vectors.
 * REQUIRES: tfidf_indices arrays must be sorted ascending.
 * The two-pointer merge silently produces wrong results if indices are unsorted.
 * mag_a_sq is the precomputed sum-of-squares of a's tfidf_weights (pass NAN
 * to compute it here; useful when the same 'a' is scored against many 'b's).
 * Returns 0 when either side is empty or the magnitude product is below epsilon. */
static float fce_sparse_tfidf_cosine(const fce_sem_func_t *a, const fce_sem_func_t *b,
                                  float mag_a_sq) {
    if (a->tfidf_len <= 0 || b->tfidf_indices == NULL || b->tfidf_len <= 0) {
        return 0.0F;
    }
    float dot = 0.0F;
    int ia = 0;
    int ib = 0;
    while (ia < a->tfidf_len && ib < b->tfidf_len) {
        if (a->tfidf_indices[ia] == b->tfidf_indices[ib]) {
            dot += a->tfidf_weights[ia] * b->tfidf_weights[ib];
            ia++;
            ib++;
        } else if (a->tfidf_indices[ia] < b->tfidf_indices[ib]) {
            ia++;
        } else {
            ib++;
        }
    }
    float ma = isnan(mag_a_sq) ? 0.0F : mag_a_sq;
    if (isnan(mag_a_sq)) {
        for (int i = 0; i < a->tfidf_len; i++) {
            ma += a->tfidf_weights[i] * a->tfidf_weights[i];
        }
    }
    float mb = 0.0F;
    for (int i = 0; i < b->tfidf_len; i++) {
        mb += b->tfidf_weights[i] * b->tfidf_weights[i];
    }
    float denom = sqrtf(ma) * sqrtf(mb);
    return denom > FCE_SEM_DENOM_EPS ? (dot / denom) : 0.0F;
}

/* H1: fce_sparse_tfidf_cosine_flat removed — positional indices (0,1,2,…) are not
 * global vocab IDs, making the flat TF-IDF cosine meaningless. The flat scoring
 * path (fce_score_flat) now uses RI-only. */

/* Internal combined scoring with precomputed proximity (P3) and precomputed
 * query-side magnitudes (P2). Avoids redundant per-element FLOPs.
 * Pass NAN for any magnitude to compute it inline (for single-call paths). */
static float score_combined_internal(const fce_sem_func_t *a, const fce_sem_func_t *b,
                                     const fce_sem_config_t *cfg, float prox,
                                     float q_tfidf_mag_sq, float q_ri_mag_sq,
                                     float q_api_mag_sq, float q_type_mag_sq,
                                     float q_deco_mag_sq, float q_sp_mag_sq) {
    if (!a || !b || !cfg) {
        return 0.0F;
    }

    float score = 0.0F;

    /* TF-IDF cosine — use precomputed query magnitude if provided. */
    float tfidf_mag = q_tfidf_mag_sq;
    if (isnan(tfidf_mag) && a->tfidf_len > 0) {
        float m = 0.0F;
        for (int i = 0; i < a->tfidf_len; i++) {
            m += a->tfidf_weights[i] * a->tfidf_weights[i];
        }
        tfidf_mag = m;
    }
    score += cfg->w_tfidf * fce_sparse_tfidf_cosine(a, b, tfidf_mag);

    /* RI cosine — use precomputed query magnitude. */
    if (isnan(q_ri_mag_sq)) {
        q_ri_mag_sq = 0.0F;
        for (int i = 0; i < FCE_SEM_DIM; i++) {
            q_ri_mag_sq += a->ri_vec.v[i] * a->ri_vec.v[i];
        }
    }
    score += cfg->w_ri * fce_sem_cosine_aliased_with_mag(a->ri_vec.v, b->ri_vec.v, q_ri_mag_sq);

    /* API cosine — skip when query signal is zero (P1: avoids 768 MACs). */
    if (isnan(q_api_mag_sq)) {
        q_api_mag_sq = 0.0F;
        for (int i = 0; i < FCE_SEM_DIM; i++) {
            q_api_mag_sq += a->api_vec.v[i] * a->api_vec.v[i];
        }
    }
    if (q_api_mag_sq > FCE_SEM_DENOM_EPS)
        score += cfg->w_api * fce_sem_cosine_aliased_with_mag(a->api_vec.v, b->api_vec.v, q_api_mag_sq);

    /* Type cosine — skip when query signal is zero (P1). */
    if (isnan(q_type_mag_sq)) {
        q_type_mag_sq = 0.0F;
        for (int i = 0; i < FCE_SEM_DIM; i++) {
            q_type_mag_sq += a->type_vec.v[i] * a->type_vec.v[i];
        }
    }
    if (q_type_mag_sq > FCE_SEM_DENOM_EPS)
        score += cfg->w_type * fce_sem_cosine_aliased_with_mag(a->type_vec.v, b->type_vec.v, q_type_mag_sq);

    /* Decorator cosine — skip when query signal is zero (P1). */
    if (isnan(q_deco_mag_sq)) {
        q_deco_mag_sq = 0.0F;
        for (int i = 0; i < FCE_SEM_DIM; i++) {
            q_deco_mag_sq += a->deco_vec.v[i] * a->deco_vec.v[i];
        }
    }
    if (q_deco_mag_sq > FCE_SEM_DENOM_EPS)
        score += cfg->w_decorator * fce_sem_cosine_aliased_with_mag(a->deco_vec.v, b->deco_vec.v, q_deco_mag_sq);

    /* Structural profile — use precomputed query-side magnitude if provided. */
    {
        float sp_mag = q_sp_mag_sq;
        if (isnan(sp_mag)) {
            sp_mag = 0.0F;
            for (int i = 0; i < FCE_SEM_AST_PROFILE_DIMS; i++)
                sp_mag += a->struct_profile[i] * a->struct_profile[i];
        }
        if (sp_mag > FCE_SEM_DENOM_EPS) {
            float dot = 0.0F;
            for (int i = 0; i < FCE_SEM_AST_PROFILE_DIMS; i++)
                dot += a->struct_profile[i] * b->struct_profile[i];
            float mb = 0.0F;
            for (int i = 0; i < FCE_SEM_AST_PROFILE_DIMS; i++)
                mb += b->struct_profile[i] * b->struct_profile[i];
            float denom = sqrtf(sp_mag) * sqrtf(mb);
            if (denom > FCE_SEM_DENOM_EPS)
                score += cfg->w_struct_profile * (dot / denom);
        }
    }

    /* Module proximity multiplier. */
    score *= prox;
    if (!isfinite(score)) return 0.0f;  /* C1: sanitize NaN/Inf from caller vectors */
    if (score > FCE_SEM_UNIT_POS) {
        score = FCE_SEM_UNIT_POS;
    }
    if (score < 0.0F) {
        score = 0.0F;
    }

    return score;
}

float fce_sem_combined_score(const fce_sem_func_t *a, const fce_sem_func_t *b,
                             const fce_sem_config_t *cfg) {
    float prox = fce_sem_proximity(a ? a->file_path : NULL, b ? b->file_path : NULL);
    return score_combined_internal(a, b, cfg, prox, NAN, NAN, NAN, NAN, NAN, NAN);
}

/* Min-heap helpers for top-k selection (O(log k) per push vs O(k log k) qsort). */
static void heap_siftdown(fce_sem_ranked_t *arr, int root, int len) {
    while (true) {
        int child = 2 * root + 1;
        if (child >= len) break;
        if (child + 1 < len && arr[child + 1].score < arr[child].score) child++;
        if (arr[root].score <= arr[child].score) break;
        fce_sem_ranked_t tmp = arr[root]; arr[root] = arr[child]; arr[child] = tmp;
        root = child;
    }
}

/* qsort comparator: descending by score. */
static int fce_ranked_cmp_desc(const void *a, const void *b) {
    float sa = ((const fce_sem_ranked_t *)a)->score;
    float sb = ((const fce_sem_ranked_t *)b)->score;
    if (sb > sa) return 1;
    if (sb < sa) return -1;
    return 0;
}

/* ── Quickselect for top-k candidate selection (P2) ────────────── */

/* Descending comparator for cand_t. */
typedef struct { int doc; float score; } cand_t;

static int cand_cmp_desc(const void *a, const void *b) {
    float sa = ((const cand_t *)a)->score;
    float sb = ((const cand_t *)b)->score;
    if (sb > sa) return 1;
    if (sb < sa) return -1;
    return 0;
}

/* Partial quickselect: rearrange scored[0..n-1] so that scored[0..k-1]
 * are the top-k by score (descending order, unsorted within partition).
 * O(n) average vs O(n·k) selection sort.
 * Depth guard: falls back to qsort when recursion depth exceeds 2*log2(n)
 * to prevent O(n²) worst case on adversarial inputs. */
static void quickselect_topk(cand_t *scored, int n, int k, int depth) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        if (depth-- <= 0) {
            /* Introsort fallback: sort entire array, then done. */
            qsort(scored, (size_t)n, sizeof(cand_t), cand_cmp_desc);
            return;
        }
        /* Median-of-three pivot. */
        int mid = lo + (hi - lo) / 2;
        if (scored[lo].score < scored[mid].score) {
            cand_t tmp = scored[lo]; scored[lo] = scored[mid]; scored[mid] = tmp;
        }
        if (scored[lo].score < scored[hi].score) {
            cand_t tmp = scored[lo]; scored[lo] = scored[hi]; scored[hi] = tmp;
        }
        if (scored[mid].score < scored[hi].score) {
            cand_t tmp = scored[mid]; scored[mid] = scored[hi]; scored[hi] = tmp;
        }
        /* Pivot at hi. */
        float piv = scored[hi].score;
        int i = lo;
        for (int j = lo; j < hi; j++) {
            if (scored[j].score > piv) {
                cand_t tmp = scored[i]; scored[i] = scored[j]; scored[j] = tmp;
                i++;
            }
        }
        cand_t tmp = scored[i]; scored[i] = scored[hi]; scored[hi] = tmp;
        if (i == k) break;
        if (i < k) lo = i + 1; else hi = i - 1;
    }
    /* Sort the top-k partition descending for deterministic output. */
    if (k > 1) qsort(scored, (size_t)k, sizeof(cand_t), cand_cmp_desc);
}

/* ── Serial top-k selection (L5) ───────────────────────────────── */

/* Score function for serial_topk: returns score for doc index i,
 * or NaN to signal the doc should be skipped. */
typedef float (*serial_score_fn)(int i, void *ctx);

/* Serial top-k: scan corpus, maintain min-heap of top_k results.
 * Used by all ranking functions for small corpora and OOM fallbacks.
 * Returns count of results written (<= top_k), sorted descending. */
static uint32_t serial_topk(
    serial_score_fn score_fn, void *ctx,
    uint32_t corpus_size, uint32_t top_k, float min_score,
    fce_sem_ranked_t *results_out) {

    uint32_t k = 0;
    for (uint32_t i = 0; i < corpus_size; i++) {
        float s = score_fn((int)i, ctx);
        if (s < min_score) continue;
        if (k < top_k) {
            results_out[k].index = i;
            results_out[k].score = s;
            k++;
            if (k == top_k) {
                for (int h = (int)top_k / 2 - 1; h >= 0; h--) {
                    heap_siftdown(results_out, h, (int)top_k);
                }
            }
        } else if (s > results_out[0].score) {
            results_out[0].index = i;
            results_out[0].score = s;
            heap_siftdown(results_out, 0, (int)top_k);
        }
    }
    if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
    return k;
}

/* Forward declarations for functions used by parallel workers. */
static float fce_score_flat(
    const int *q_idx, const float *q_w, int q_len, const float *q_ri,
    float q_ri_mag,
    const int *c_idx, const float *c_w, int c_len, const float *c_ri,
    const char *q_path, const char *c_path);
static float fce_score_simple_internal(fce_sem_func_t *a, fce_sem_func_t *b, float prox,
                                    float q_ri_mag_sq);

/* ── Serial score contexts (one per ranking function) ────────── */

typedef struct {
    const fce_sem_corpus_t *corpus;
    const int8_t *qvec_q;  /* P0: pre-quantized query vector for int8 brute-force */
    float qvec_q_inv_mag;   /* F2: reciprocal L2 magnitude of quantized query vector */
} sq_sctx_t;

static float sq_score(int i, void *ctx) {
    sq_sctx_t *c = ctx;
    const int8_t *dq = c->corpus->doc_vectors_q;
    float dot = (float)fce_dot768_i8(c->qvec_q, dq + (size_t)i * FCE_SEM_DIM);
    float inv_d_mag = c->corpus->doc_vectors_q_inv_mag[i];
    float cosine = (inv_d_mag > 0.0f && c->qvec_q_inv_mag > 0.0f) ? dot * c->qvec_q_inv_mag * inv_d_mag : 0.0f;
    return (cosine + FCE_SEM_UNIT_POS) * 0.5f;
}

typedef struct {
    fce_sem_func_t *query;
    fce_sem_func_t *corpus;
    float min_score;
    int q_prox;
    float q_tfidf_mag_sq;
    float q_ri_mag_sq;
    float q_api_mag_sq;
    float q_type_mag_sq;
    float q_deco_mag_sq;
    float q_sp_mag_sq;
    const fce_sem_config_t *cfg;
} sc_sctx_t;

static float sc_score(int i, void *ctx) {
    sc_sctx_t *c = ctx;
    float prox = proximity_internal(c->query->file_path, c->q_prox,
                                     c->corpus[i].file_path);
    return score_combined_internal(c->query, &c->corpus[i], c->cfg, prox,
                                   c->q_tfidf_mag_sq, c->q_ri_mag_sq,
                                   c->q_api_mag_sq, c->q_type_mag_sq,
                                   c->q_deco_mag_sq, c->q_sp_mag_sq);
}

typedef struct {
    fce_sem_func_t *query;
    fce_sem_func_t *corpus;
    float min_score;
    int q_prox;
    float q_ri_mag_sq;
} fce_ss_sctx_t;

static float fce_ss_score(int i, void *ctx) {
    fce_ss_sctx_t *c = ctx;
    float prox = proximity_internal(c->query->file_path, c->q_prox,
                                     c->corpus[i].file_path);
    return fce_score_simple_internal(c->query, &c->corpus[i], prox,
                                 c->q_ri_mag_sq);
}

typedef struct {
    const float *all_tfidf_weights;
    const int   *all_tfidf_indices;
    const int   *tfidf_lens;
    const float *all_ri_vecs;
    const char **file_paths;
    int max_tokens;
    const int   *q_tfidf_indices;
    const float *q_tfidf_weights;
    int q_tfidf_len;
    const float *q_ri_vec;
    float q_ri_mag;
} sf_sctx_t;

static float sf_score(int i, void *ctx) {
    sf_sctx_t *c = ctx;
    const int   *c_idx = c->all_tfidf_indices + (size_t)i * c->max_tokens;
    const float *c_w   = c->all_tfidf_weights + (size_t)i * c->max_tokens;
    int c_len = c->tfidf_lens ? c->tfidf_lens[i] : 0;
    if (c_len > c->max_tokens) c_len = c->max_tokens; /* defensive clamp (H1) */
    const float *c_ri  = c->all_ri_vecs + (size_t)i * FCE_SEM_DIM;
    const char  *c_path = c->file_paths ? c->file_paths[i] : "";
    return fce_score_flat(c->q_tfidf_indices, c->q_tfidf_weights, c->q_tfidf_len,
                      c->q_ri_vec, c->q_ri_mag,
                      c_idx, c_w, c_len, c_ri, "(query)", c_path);
}

/* Each ranking function dispatches N worker threads, each maintaining a local
 * top-k min-heap. After all workers finish, a merge step collects the partial
 * heaps and selects the global top-k. This is embarrassingly parallel because
 * each corpus item is scored independently. */

/* Merge per-worker top-k heaps into final results.
 * worker_results: [worker_count * top_k] flat array of per-worker results
 * worker_counts: [worker_count] result count per worker
 * Returns the number of results written to results_out (<= top_k). */
static uint32_t merge_worker_heaps(
    const fce_sem_ranked_t *worker_results,
    const int *worker_counts,
    int worker_count,
    uint32_t top_k,
    fce_sem_ranked_t *results_out) {
    uint32_t k = 0;
    for (int w = 0; w < worker_count; w++) {
        for (int i = 0; i < worker_counts[w]; i++) {
            fce_sem_ranked_t r = worker_results[(size_t)w * top_k + i];
            if (k < top_k) {
                results_out[k] = r;
                k++;
                if (k == top_k) {
                    for (int h = (int)top_k / 2 - 1; h >= 0; h--) {
                        heap_siftdown(results_out, h, (int)top_k);
                    }
                }
            } else if (r.score > results_out[0].score) {
                results_out[0] = r;
                heap_siftdown(results_out, 0, (int)top_k);
            }
        }
    }
    return k;
}

/* ── search_query parallel worker (removed — serial brute-force always wins) ── */
/* ── simple_rank_flat parallel worker ─────────────────────────── */
typedef struct {
    const float *all_tfidf_weights;
    const int   *all_tfidf_indices;
    const int   *tfidf_lens;
    const float *all_ri_vecs;
    const char **file_paths;
    int max_tokens;
    int corpus_size;
    const int   *q_tfidf_indices;
    const float *q_tfidf_weights;
    int q_tfidf_len;
    const float *q_ri_vec;
    float q_ri_mag;
    int top_k;
    _Atomic int next_doc;
    fce_sem_ranked_t *worker_results;
    int *worker_counts;
} flat_ctx_t;

static void flat_worker(int wid, void *ctx) {
    flat_ctx_t *w = ctx;
    fce_sem_ranked_t *local = w->worker_results + (size_t)wid * w->top_k;
    int n = 0;
    for (;;) {
        int f = atomic_fetch_add_explicit(&w->next_doc, 1, memory_order_relaxed);
        if (f >= w->corpus_size) break;
        const int   *c_idx = w->all_tfidf_indices + (size_t)f * w->max_tokens;
        const float *c_w   = w->all_tfidf_weights + (size_t)f * w->max_tokens;
        int c_len = w->tfidf_lens ? w->tfidf_lens[f] : 0;
        if (c_len > w->max_tokens) c_len = w->max_tokens; /* defensive clamp (H1) */
        const float *c_ri  = w->all_ri_vecs + (size_t)f * FCE_SEM_DIM;
        const char  *c_path = w->file_paths ? w->file_paths[f] : "";
        float s = fce_score_flat(w->q_tfidf_indices, w->q_tfidf_weights, w->q_tfidf_len,
                             w->q_ri_vec, w->q_ri_mag,
                             c_idx, c_w, c_len, c_ri, "(query)", c_path);
        if (n < w->top_k) {
            local[n].index = (uint32_t)f;
            local[n].score = s;
            n++;
            if (n == w->top_k) {
                for (int h = w->top_k / 2 - 1; h >= 0; h--) {
                    heap_siftdown(local, h, w->top_k);
                }
            }
        } else if (s > local[0].score) {
            local[0].index = (uint32_t)f;
            local[0].score = s;
            heap_siftdown(local, 0, w->top_k);
        }
    }
    w->worker_counts[wid] = n;
}

/* ── simple_search parallel worker ────────────────────────────── */
typedef struct {
    fce_sem_func_t *query;
    fce_sem_func_t *corpus;
    uint32_t corpus_size;
    float min_score;
    float q_prox; /* precomputed proximity for query (P3) */
    /* Precomputed query-side RI magnitude (P1) — avoids ~768 FLOPs per corpus item. */
    float q_ri_mag_sq;
    int top_k;
    _Atomic int next_doc;
    fce_sem_ranked_t *worker_results;
    int *worker_counts;
} fce_ss_ctx_t;

static void ss_worker(int wid, void *ctx) {
    fce_ss_ctx_t *w = ctx;
    fce_sem_ranked_t *local = w->worker_results + (size_t)wid * w->top_k;
    int n = 0;
    for (;;) {
        int i = atomic_fetch_add_explicit(&w->next_doc, 1, memory_order_relaxed);
        if (i >= (int)w->corpus_size) break;
        float prox = proximity_internal(w->query->file_path, w->q_prox,
                                         w->corpus[i].file_path);
        float s = fce_score_simple_internal(w->query, &w->corpus[i], prox,
                                         w->q_ri_mag_sq);
        if (s >= w->min_score) {
            if (n < w->top_k) {
                local[n].index = (uint32_t)i;
                local[n].score = s;
                n++;
                if (n == w->top_k) {
                    for (int h = w->top_k / 2 - 1; h >= 0; h--) {
                        heap_siftdown(local, h, w->top_k);
                    }
                }
            } else if (s > local[0].score) {
                local[0].index = (uint32_t)i;
                local[0].score = s;
                heap_siftdown(local, 0, w->top_k);
            }
        }
    }
    w->worker_counts[wid] = n;
}

/* ── search (combined score) parallel worker ──────────────────── */
typedef struct {
    fce_sem_func_t *query;
    fce_sem_func_t *corpus;
    uint32_t corpus_size;
    float min_score;
    float q_prox; /* precomputed proximity for query (P3) */
    /* Precomputed query-side magnitudes (P1) — avoids ~3072 FLOPs per corpus item. */
    float q_tfidf_mag_sq;
    float q_ri_mag_sq;
    float q_api_mag_sq;
    float q_type_mag_sq;
    float q_deco_mag_sq;
    float q_sp_mag_sq;
    const fce_sem_config_t *cfg;
    int top_k;
    _Atomic int next_doc;
    fce_sem_ranked_t *worker_results;
    int *worker_counts;
} sc_ctx_t;

static void sc_worker(int wid, void *ctx) {
    sc_ctx_t *w = ctx;
    fce_sem_ranked_t *local = w->worker_results + (size_t)wid * w->top_k;
    int n = 0;
    for (;;) {
        int i = atomic_fetch_add_explicit(&w->next_doc, 1, memory_order_relaxed);
        if (i >= (int)w->corpus_size) break;
        float prox = proximity_internal(w->query->file_path, w->q_prox,
                                         w->corpus[i].file_path);
        float s = score_combined_internal(w->query, &w->corpus[i], w->cfg, prox,
                                          w->q_tfidf_mag_sq, w->q_ri_mag_sq,
                                          w->q_api_mag_sq, w->q_type_mag_sq,
                                          w->q_deco_mag_sq, w->q_sp_mag_sq);
        if (s >= w->min_score) {
            if (n < w->top_k) {
                local[n].index = (uint32_t)i;
                local[n].score = s;
                n++;
                if (n == w->top_k) {
                    for (int h = w->top_k / 2 - 1; h >= 0; h--) {
                        heap_siftdown(local, h, w->top_k);
                    }
                }
            } else if (s > local[0].score) {
                local[0].index = (uint32_t)i;
                local[0].score = s;
                heap_siftdown(local, 0, w->top_k);
            }
        }
    }
    w->worker_counts[wid] = n;
}

/* ── Search query (tokenize + brute-force cosine on doc vectors) ── */

/* ── Inverted-index candidate retrieval + rerank ──────────────── */

/* Maximum candidates to retrieve before reranking. */
enum { FCE_CANDIDATE_CAP = 2048 };
/* N1: heap arrays sized 4096 must accommodate FCE_CANDIDATE_CAP. */
_Static_assert(FCE_CANDIDATE_CAP <= 4096, "FCE_CANDIDATE_CAP exceeds heap[4096]");

/* ── Unified candidate retrieval (Q1) ────────────────────────── */

/* Score callback for candidate retrieval. Returns a score for (corpus, doc_id). */
typedef float (*cand_score_fn)(const fce_sem_corpus_t *corpus, int doc_id, void *ctx);

/* IDF-sum scorer context: sum of IDF weights for matching query tokens. */
typedef struct { const int *q_toks; int q_ntok; const float *q_idf; } idf_score_ctx_t;
static float idf_score_fn(const fce_sem_corpus_t *corpus, int doc_id, void *vctx) {
    idf_score_ctx_t *c = vctx;
    float score = 0.0f;
    for (int t = 0; t < c->q_ntok; t++) {
        int tid = c->q_toks[t];
        int start = corpus->inv_offsets[tid];
        int end   = corpus->inv_offsets[tid + 1];
        int lo = start, hi = end;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (corpus->inv_doc_ids[mid] < doc_id) lo = mid + 1;
            else hi = mid;
        }
        if (lo < end && corpus->inv_doc_ids[lo] == doc_id) {
            score += c->q_idf[t];
        }
    }
    return score;
}

/* TF-IDF candidate scorer context: matched-IDF-mass heuristic.
 * This is NOT a true cosine — it accumulates qidf² for matched terms only,
 * so dot == doc_mag and the result reduces to sqrt(matched_mass / total_mass).
 * Used as a fast pre-filter for inverted-index candidate selection before
 * RI rerank; the naming reflects this is intentionally approximate. */
typedef struct { const float *q_idf; const int *q_toks; int q_ntok; float q_mag; } tfidf_mass_ctx_t;
static float tfidf_mass_score_fn(const fce_sem_corpus_t *corpus, int doc_id, void *vctx) {
    tfidf_mass_ctx_t *c = vctx;
    float dot = 0.0f;
    float doc_mag = 0.0f;
    for (int t = 0; t < c->q_ntok; t++) {
        int tid = c->q_toks[t];
        float qidf = c->q_idf[t];
        int start = corpus->inv_offsets[tid];
        int end   = corpus->inv_offsets[tid + 1];
        int lo = start, hi = end;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (corpus->inv_doc_ids[mid] < doc_id) lo = mid + 1;
            else hi = mid;
        }
        if (lo < end && corpus->inv_doc_ids[lo] == doc_id) {
            dot     += qidf * qidf;
            doc_mag += qidf * qidf;
        }
    }
    float denom = sqrtf(c->q_mag) * sqrtf(doc_mag);
    return (denom > FCE_SEM_DENOM_EPS) ? (dot / denom) : 0.0f;
}

/* P4: Thread-local scratch arena for collect_candidates — avoids per-query
 * calloc/malloc/free of bitmap (~3 KB), raw candidates, and scored array.
 * Buffers grow monotonically and are never freed (thread-local cleanup at exit). */
typedef struct {
    uint64_t *seen;
    int seen_nwords;
    int *raw;
    int raw_cap;
    cand_t *scored;
    int scored_cap;
} cand_scratch_t;

static _Thread_local cand_scratch_t tls_cand_scratch;
static _Thread_local bool tls_cand_scratch_initialized = false;

/* Destructor for thread-local scratch buffers. Registered via pthread_once. */
#ifndef _WIN32
static pthread_key_t tls_cand_key;
static void tls_cand_scratch_destructor(void *ptr) {
    cand_scratch_t *sc = (cand_scratch_t *)ptr;
    if (sc) {
        free(sc->seen);
        free(sc->raw);
        free(sc->scored);
        memset(sc, 0, sizeof(*sc));
    }
}
static void tls_cand_key_init(void) {
    pthread_key_create(&tls_cand_key, tls_cand_scratch_destructor);
}
#endif

/* Unified candidate retrieval: collect docs from inverted index, score with
 * callback, select top max_candidates. Returns count written to candidates_out. */
static int collect_candidates(const fce_sem_corpus_t *corpus,
                              const int *q_toks, int q_ntok,
                              cand_score_fn score_fn, void *score_ctx,
                              int *candidates_out, int max_candidates) {
    if (!corpus->inv_offsets || q_ntok == 0) return 0;

    int ndocs = corpus->doc_count;
    cand_scratch_t *sc = &tls_cand_scratch;

#ifndef _WIN32
    /* Register pthread destructor on first use so scratch is freed on thread exit. */
    if (!tls_cand_scratch_initialized) {
        static pthread_once_t once = PTHREAD_ONCE_INIT;
        pthread_once(&once, tls_cand_key_init);
        pthread_setspecific(tls_cand_key, sc);
        tls_cand_scratch_initialized = true;
    }
#endif

    /* Ensure seen bitmap is large enough. */
    int nwords = (ndocs + 63) / 64;
    if (nwords > sc->seen_nwords) {
        free(sc->seen);
        sc->seen = (uint64_t *)calloc((size_t)nwords, sizeof(uint64_t));
        if (!sc->seen) { sc->seen_nwords = 0; return 0; }
        sc->seen_nwords = nwords;
    } else {
        memset(sc->seen, 0, (size_t)nwords * sizeof(uint64_t));
    }

    long approx = 0;
    for (int t = 0; t < q_ntok; t++) {
        approx += corpus->entries[q_toks[t]].doc_freq;
    }
    if (approx > (long)max_candidates * 4) approx = (long)max_candidates * 4;
    if (approx < 256) approx = 256;

    /* Ensure raw buffer is large enough. */
    int raw_need = (int)approx;
    if (raw_need > sc->raw_cap) {
        free(sc->raw);
        sc->raw = (int *)malloc((size_t)raw_need * sizeof(int));
        if (!sc->raw) { sc->raw_cap = 0; return 0; }
        sc->raw_cap = raw_need;
    }
    int nraw = 0;

    for (int t = 0; t < q_ntok; t++) {
        int tid = q_toks[t];
        int start = corpus->inv_offsets[tid];
        int end   = corpus->inv_offsets[tid + 1];
        for (int i = start; i < end; i++) {
            int d = corpus->inv_doc_ids[i];
            int w = d / 64, b = d % 64;
            if (sc->seen[w] & ((uint64_t)1 << b)) continue;
            sc->seen[w] |= ((uint64_t)1 << b);
            if (nraw >= raw_need) {
                /* Grow buffer dynamically. */
                int new_cap = raw_need * 2;
                int *grown = (int *)realloc(sc->raw, (size_t)new_cap * sizeof(int));
                if (grown) {
                    sc->raw = grown;
                    sc->raw_cap = new_cap;
                    raw_need = new_cap;
                }
            }
            if (nraw < raw_need) {
                sc->raw[nraw++] = d;
            }
        }
    }

    if (nraw == 0) return 0;

    /* Ensure scored buffer is large enough. */
    if (nraw > sc->scored_cap) {
        free(sc->scored);
        sc->scored = (cand_t *)malloc((size_t)nraw * sizeof(cand_t));
        if (!sc->scored) { sc->scored_cap = 0; return 0; }
        sc->scored_cap = nraw;
    }
    for (int i = 0; i < nraw; i++) {
        sc->scored[i].doc = sc->raw[i];
        sc->scored[i].score = score_fn(corpus, sc->raw[i], score_ctx);
    }

    int nout = nraw;
    if (nout > max_candidates) nout = max_candidates;
    /* Depth guard: 2 * floor(log2(n)) prevents O(n²) worst case. */
    int depth = 0;
    { int tmp = nraw; while (tmp > 1) { depth++; tmp >>= 1; } }
    depth *= 2;
    quickselect_topk(sc->scored, nraw, nout, depth);
    for (int i = 0; i < nout; i++) {
        candidates_out[i] = sc->scored[i].doc;
    }
    return nout;
}

/* Retrieve candidate document IDs using IDF-sum scoring. */
static int keyword_candidates(const fce_sem_corpus_t *corpus,
                               const int *q_toks, int q_ntok,
                               int *candidates_out, int max_candidates) {
    float *q_idf = (float *)malloc((size_t)q_ntok * sizeof(float));
    if (!q_idf) return 0;
    for (int t = 0; t < q_ntok; t++) {
        int df = corpus->entries[q_toks[t]].doc_freq;
        q_idf[t] = (df > 0) ? logf((float)corpus->doc_count / (float)df) : 0.0f;
    }
    idf_score_ctx_t ctx = { .q_toks = q_toks, .q_ntok = q_ntok, .q_idf = q_idf };
    int n = collect_candidates(corpus, q_toks, q_ntok, idf_score_fn, &ctx,
                               candidates_out, max_candidates);
    free(q_idf);
    return n;
}

/* Retrieve candidate document IDs using TF-IDF cosine scoring. */
static int tfidf_keyword_candidates(const fce_sem_corpus_t *corpus,
                                     const int *q_toks, int q_ntok,
                                     int *candidates_out, int max_candidates) {
    float q_idf[FCE_SEM_MAX_TOKENS];
    float q_mag = 0.0f;
    for (int t = 0; t < q_ntok; t++) {
        int df = corpus->entries[q_toks[t]].doc_freq;
        q_idf[t] = (df > 0) ? logf((float)corpus->doc_count / (float)df) : 0.0f;
        q_mag += q_idf[t] * q_idf[t];
    }
    tfidf_mass_ctx_t ctx = { .q_idf = q_idf, .q_toks = q_toks, .q_ntok = q_ntok, .q_mag = q_mag };
    return collect_candidates(corpus, q_toks, q_ntok, tfidf_mass_score_fn, &ctx,
                              candidates_out, max_candidates);
}

/* ── Rerank scoring (RI cosine + proximity) ──────────────────── */

typedef struct {
    const fce_sem_corpus_t *corpus;
    const int8_t *qvec_q;       /* F2: pre-quantized query for int8 rerank */
    float qvec_q_inv_mag;       /* F2: reciprocal magnitude of quantized query */
    const char *q_path;
    float q_prox;
    int top_k;
    _Atomic int next_cand;
    fce_sem_ranked_t *worker_results;
    int *worker_counts;
    const int *candidates;
    int ncand;
} rerank_ctx_t;

static void rerank_worker(int wid, void *uctx) {
    rerank_ctx_t *w = uctx;
    fce_sem_ranked_t *local = w->worker_results + (size_t)wid * w->top_k;
    int n = 0;
    for (;;) {
        int ci = atomic_fetch_add_explicit(&w->next_cand, 1, memory_order_relaxed);
        if (ci >= w->ncand) break;
        int i = w->candidates[ci];
        float i8dot = (float)fce_dot768_i8(w->qvec_q,
                                    w->corpus->doc_vectors_q + (size_t)i * FCE_SEM_DIM);
        float inv_d_mag = w->corpus->doc_vectors_q_inv_mag[i];
        float dot = (inv_d_mag > 0.0f && w->qvec_q_inv_mag > 0.0f)
                ? i8dot * w->qvec_q_inv_mag * inv_d_mag : 0.0f;
        float s = (dot + FCE_SEM_UNIT_POS) * 0.5f;
        if (n < w->top_k) {
            local[n].index = (uint32_t)i;
            local[n].score = s;
            n++;
            if (n == w->top_k) {
                for (int h = w->top_k / 2 - 1; h >= 0; h--) {
                    heap_siftdown(local, h, w->top_k);
                }
            }
        } else if (s > local[0].score) {
            local[0].index = (uint32_t)i;
            local[0].score = s;
            heap_siftdown(local, 0, w->top_k);
        }
    }
    w->worker_counts[wid] = n;
}

/* ── Serial rerank helper (N1: deduplicated) ──────────────────── */

/* Score candidates with RI cosine, select top-k. Used by both fast and
 * tfidf search paths for serial reranking. */
static void rerank_serial(const fce_sem_corpus_t *corpus,
                          const int8_t *qvec_q,
                          float qvec_q_inv_mag,
                          const int *candidates, int ncand,
                          uint32_t top_k,
                          fce_sem_ranked_t *results_out,
                          uint32_t *count_out) {
    _Static_assert(FCE_CANDIDATE_CAP <= 4096, "heap[4096] must fit FCE_CANDIDATE_CAP");
    int heap_cap = (int)top_k;
    if (heap_cap > FCE_CANDIDATE_CAP) heap_cap = FCE_CANDIDATE_CAP;
    fce_sem_ranked_t heap[4096];
    int hn = 0;
    for (int ci = 0; ci < ncand; ci++) {
        int i = candidates[ci];
        float i8dot = (float)fce_dot768_i8(qvec_q,
                                    corpus->doc_vectors_q + (size_t)i * FCE_SEM_DIM);
        float inv_d_mag = corpus->doc_vectors_q_inv_mag[i];
        float dot = (inv_d_mag > 0.0f && qvec_q_inv_mag > 0.0f)
                ? i8dot * qvec_q_inv_mag * inv_d_mag : 0.0f;
        float s = (dot + FCE_SEM_UNIT_POS) * 0.5f;
        if (hn < heap_cap) {
            heap[hn].index = (uint32_t)i;
            heap[hn].score = s;
            hn++;
            if (hn == heap_cap) {
                for (int h = heap_cap / 2 - 1; h >= 0; h--) {
                    heap_siftdown(heap, h, heap_cap);
                }
            }
        } else if (s > heap[0].score) {
            heap[0].index = (uint32_t)i;
            heap[0].score = s;
            heap_siftdown(heap, 0, heap_cap);
        }
    }
    uint32_t k = 0;
    for (int i = 0; i < hn && k < (uint32_t)heap_cap; i++) {
        results_out[k++] = heap[i];
    }
    if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
    if (count_out) *count_out = k;
}

/* ── Brute-force search (P4: internal with pre-tokenized query) ── */

/* F5: Static-chunked parallel brute-force context. */
typedef struct {
    const fce_sem_corpus_t *corpus;
    const int8_t *qvec_q;
    float qvec_q_inv_mag;
    uint32_t top_k;
    int chunk_start;
    int chunk_end;
    fce_sem_ranked_t *local_heap;  /* per-worker local top-k heap */
    int local_count;
} bf_chunk_ctx_t;

static void bf_chunk_worker(int idx, void *ctx) {
    bf_chunk_ctx_t *all = (bf_chunk_ctx_t *)ctx;
    bf_chunk_ctx_t *c = &all[idx];
    const fce_sem_corpus_t *corpus = c->corpus;
    const int8_t *qvec_q = c->qvec_q;
    float qvec_q_inv_mag = c->qvec_q_inv_mag;
    uint32_t top_k = c->top_k;
    fce_sem_ranked_t *local = c->local_heap;
    int n = 0;

    for (int i = c->chunk_start; i < c->chunk_end; i++) {
        float dot = (float)fce_dot768_i8(qvec_q, corpus->doc_vectors_q + (size_t)i * FCE_SEM_DIM);
        float inv_d_mag = corpus->doc_vectors_q_inv_mag[i];
        float cosine = (inv_d_mag > 0.0f && qvec_q_inv_mag > 0.0f) ? dot * qvec_q_inv_mag * inv_d_mag : 0.0f;
        float s = (cosine + FCE_SEM_UNIT_POS) * 0.5f;

        if (n < (int)top_k) {
            local[n].index = (uint32_t)i;
            local[n].score = s;
            n++;
            if (n == (int)top_k) {
                for (int h = (int)top_k / 2 - 1; h >= 0; h--) {
                    heap_siftdown(local, h, (int)top_k);
                }
            }
        } else if (s > local[0].score) {
            local[0].index = (uint32_t)i;
            local[0].score = s;
            heap_siftdown(local, 0, (int)top_k);
        }
    }
    c->local_count = n;
}

/* Merge multiple local heaps into results_out (serial). */
static uint32_t merge_local_heaps(bf_chunk_ctx_t *chunks, int nchunks,
                                   uint32_t top_k, fce_sem_ranked_t *results_out) {
    /* Collect all candidates from local heaps. */
    int total = 0;
    for (int c = 0; c < nchunks; c++) total += chunks[c].local_count;

    if (total == 0) return 0;
    if ((uint32_t)total <= top_k) {
        /* Fewer candidates than top_k — just copy and sort. */
        uint32_t k = 0;
        for (int c = 0; c < nchunks; c++) {
            for (int i = 0; i < chunks[c].local_count; i++) {
                results_out[k++] = chunks[c].local_heap[i];
            }
        }
        if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
        return k;
    }

    /* More candidates than top_k — use a combined min-heap.
     * Heap-allocate to avoid stack overflow when top_k > 4096. */
    fce_sem_ranked_t *heap = malloc((size_t)top_k * sizeof(fce_sem_ranked_t));
    if (!heap) {
        /* OOM: collect ALL candidates from all local heaps, sort, take top-k. */
        fce_sem_ranked_t *all = malloc((size_t)total * sizeof(fce_sem_ranked_t));
        if (!all) {
            /* Double OOM: best-effort take first top_k items. */
            uint32_t k = 0;
            for (int c = 0; c < nchunks && k < top_k; c++) {
                for (int i = 0; i < chunks[c].local_count && k < top_k; i++) {
                    results_out[k++] = chunks[c].local_heap[i];
                }
            }
            if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
            return k;
        }
        uint32_t k = 0;
        for (int c = 0; c < nchunks; c++) {
            for (int i = 0; i < chunks[c].local_count; i++) {
                all[k++] = chunks[c].local_heap[i];
            }
        }
        if (k > 1) qsort(all, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
        uint32_t out = k < top_k ? k : top_k;
        memcpy(results_out, all, out * sizeof(fce_sem_ranked_t));
        free(all);
        return out;
    }
    int hn = 0;
    for (int c = 0; c < nchunks; c++) {
        for (int i = 0; i < chunks[c].local_count; i++) {
            fce_sem_ranked_t r = chunks[c].local_heap[i];
            if (hn < (int)top_k) {
                heap[hn++] = r;
                if (hn == (int)top_k) {
                    for (int h = (int)top_k / 2 - 1; h >= 0; h--) {
                        heap_siftdown(heap, h, (int)top_k);
                    }
                }
            } else if (r.score > heap[0].score) {
                heap[0] = r;
                heap_siftdown(heap, 0, (int)top_k);
            }
        }
    }

    uint32_t k = 0;
    for (int i = 0; i < hn && k < top_k; i++) {
        results_out[k++] = heap[i];
    }
    free(heap);
    if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
    return k;
}

static void bruteforce_precomputed(const fce_sem_corpus_t *corpus,
                                   const fce_sem_vec_t *qvec,
                                   const int8_t *qvec_q_pre,
                                   float qvec_q_inv_mag_pre,
                                   uint32_t top_k,
                                   fce_sem_ranked_t *results_out,
                                   uint32_t *count_out) {
    if (count_out) *count_out = 0;
    if (!corpus || !qvec || top_k == 0 || !results_out) return;
    if (!corpus->doc_vectors_q) return;

    int n = corpus->doc_count;
    if (n == 0) return;
    if ((int)top_k > n) top_k = (uint32_t)n;

    /* P0: pre-quantize query vector for int8 brute-force path.
     * F2: store reciprocal query magnitude for multiply-only hot loop.
     * L3: reuse pre-quantized query if caller already computed it. */
    int8_t qvec_q_buf[FCE_SEM_DIM];
    float qvec_q_inv_mag = 0.0f;
    const int8_t *qvec_q = NULL;
    if (qvec_q_pre) {
        qvec_q = qvec_q_pre;
        qvec_q_inv_mag = qvec_q_inv_mag_pre;
    } else if (corpus->doc_vectors_q) {
        fce_quantize_f32_768(qvec_q_buf, qvec->v);
        float mag_sq = 0.0f;
        for (int i = 0; i < FCE_SEM_DIM; i++) mag_sq += (float)qvec_q_buf[i] * (float)qvec_q_buf[i];
        qvec_q_inv_mag = (mag_sq > 0.0f) ? 1.0f / sqrtf(mag_sq) : 0.0f;
        qvec_q = qvec_q_buf;
    }

    /* F5: Static-chunked parallel brute-force.
     * Previous parallel path used atomic_fetch_add per doc (contention).
     * Static chunking: each worker gets a contiguous doc range, zero atomics.
     * Heuristic: use parallel when scan > 50 MB (≈66K docs × 768B).
     * Default: total_cores / 4 (min 1). Env var FCE_BRUTE_WORKERS overrides. */
    size_t scan_bytes = (size_t)n * FCE_SEM_DIM;
    int total_cores = fce_system_info().total_cores;
    int nworkers = total_cores / 4;
    if (nworkers < 1) nworkers = 1;

    /* Check for env var override. */
    char env_buf[32];
    const char *env_val = fce_safe_getenv("FCE_BRUTE_WORKERS", env_buf, sizeof(env_buf), "");
    if (env_val && env_val[0]) {
        int v = atoi(env_val);
        if (v >= 1 && v <= 64) nworkers = v;
    }

    if (nworkers > 1 && scan_bytes > 50 * 1024 * 1024) {
        /* Parallel path: split docs into nworkers contiguous chunks. */
        int total_chunks = nworkers;  /* workers + main thread */
        int chunk_size = n / total_chunks;
        int remainder = n % total_chunks;

        bf_chunk_ctx_t *chunks = (bf_chunk_ctx_t *)calloc((size_t)total_chunks, sizeof(bf_chunk_ctx_t));
        if (!chunks) goto fallback_serial;

        int offset = 0;
        for (int c = 0; c < total_chunks; c++) {
            int sz = chunk_size + (c < remainder ? 1 : 0);
            chunks[c] = (bf_chunk_ctx_t){
                .corpus = corpus, .qvec_q = qvec_q, .qvec_q_inv_mag = qvec_q_inv_mag,
                .top_k = top_k, .chunk_start = offset, .chunk_end = offset + sz,
                .local_heap = (fce_sem_ranked_t *)calloc(top_k, sizeof(fce_sem_ranked_t)),
                .local_count = 0
            };
            if (!chunks[c].local_heap) {
                for (int j = 0; j < c; j++) free(chunks[j].local_heap);
                free(chunks);
                goto fallback_serial;
            }
            offset += sz;
        }

        /* Launch parallel workers (nworkers-1 threads + main thread). */
        fce_parallel_for_opts_t popts = { .max_workers = nworkers - 1 };
        fce_parallel_for_static(total_chunks, bf_chunk_worker, chunks, popts);

        /* Merge local heaps into final results. */
        *count_out = merge_local_heaps(chunks, total_chunks, top_k, results_out);

        for (int c = 0; c < total_chunks; c++) free(chunks[c].local_heap);
        free(chunks);
        return;
    }

fallback_serial:
    {
        sq_sctx_t ctx = { .corpus = corpus, .qvec_q = qvec_q, .qvec_q_inv_mag = qvec_q_inv_mag };
        if (count_out) *count_out = serial_topk(sq_score, &ctx, (uint32_t)n, top_k, -1.0f, results_out);
    }
}

/* Public API: tokenize query then delegate to internal brute-force. */
void fce_sem_search_query_bruteforce(const fce_sem_corpus_t *corpus,
                           const char *query,
                           uint32_t top_k,
                           fce_sem_ranked_t *results_out,
                           uint32_t *count_out) {
    if (count_out) *count_out = 0;
    if (!corpus || !query || top_k == 0 || !results_out) return;
    if (!corpus->doc_vectors_q) return;

    char *q_toks[FCE_SEM_MAX_TOKENS];
    int q_ntok = fce_sem_tokenize(query, q_toks, FCE_SEM_MAX_TOKENS);
    if (q_ntok == 0) goto cleanup;

    fce_sem_vec_t qvec;
    memset(&qvec, 0, sizeof(qvec));
    for (int t = 0; t < q_ntok; t++) {
        const fce_sem_vec_t *rv = fce_sem_corpus_ri_vec(corpus, q_toks[t]);
        if (rv) fce_sem_vec_add_scaled(&qvec, rv, 1.0f);
    }
    fce_sem_normalize(&qvec);

    bruteforce_precomputed(corpus, &qvec, NULL, 0.0f, top_k, results_out, count_out);

cleanup:
    for (int t = 0; t < q_ntok; t++) free(q_toks[t]);
}

/* ── Fast search: inverted index candidate retrieval + rerank ─── */

void fce_sem_search_query(const fce_sem_corpus_t *corpus,
                           const char *query,
                           uint32_t top_k,
                           fce_sem_ranked_t *results_out,
                           uint32_t *count_out) {
    if (count_out) *count_out = 0;
    if (!corpus || !query || top_k == 0 || !results_out) return;
    if (!corpus->doc_vectors_q) return;

    int n = corpus->doc_count;
    if (n == 0) return;
    if ((int)top_k > n) top_k = (uint32_t)n;

    /* Tokenize query. */
    char *q_toks[FCE_SEM_MAX_TOKENS];
    int q_ntok = fce_sem_tokenize(query, q_toks, FCE_SEM_MAX_TOKENS);
    if (q_ntok == 0) goto cleanup;

    /* Build query vector for RI cosine. */
    fce_sem_vec_t qvec;
    memset(&qvec, 0, sizeof(qvec));
    for (int t = 0; t < q_ntok; t++) {
        const fce_sem_vec_t *rv = fce_sem_corpus_ri_vec(corpus, q_toks[t]);
        if (rv) {
            fce_sem_vec_add_scaled(&qvec, rv, 1.0f);
        }
    }
    fce_sem_normalize(&qvec);

    /* Convert query tokens to token IDs for inverted index lookup. */
    int q_tok_ids[FCE_SEM_MAX_TOKENS];
    int q_id_count = 0;
    for (int t = 0; t < q_ntok; t++) {
        void *ptr = fce_ht_get(corpus->token_map, q_toks[t]);
        if (ptr) {
            q_tok_ids[q_id_count++] = ptr_to_token_idx(ptr);
        }
    }

    /* F2: pre-quantize query for int8 rerank path. */
    int8_t qvec_q_buf[FCE_SEM_DIM];
    float qvec_q_inv_mag = 0.0f;
    const int8_t *qvec_q = NULL;
    if (corpus->doc_vectors_q) {
        fce_quantize_f32_768(qvec_q_buf, qvec.v);
        float mag_sq = 0.0f;
        for (int i = 0; i < FCE_SEM_DIM; i++) mag_sq += (float)qvec_q_buf[i] * (float)qvec_q_buf[i];
        qvec_q_inv_mag = (mag_sq > 0.0f) ? 1.0f / sqrtf(mag_sq) : 0.0f;
        qvec_q = qvec_q_buf;
    }

    /* Try inverted index candidate retrieval. */
    if (corpus->inv_offsets && q_id_count > 0) {
        int candidates[FCE_CANDIDATE_CAP];
        int ncand = keyword_candidates(corpus, q_tok_ids, q_id_count,
                                        candidates, FCE_CANDIDATE_CAP);

        /* Fall back to brute-force if too few candidates. */
        if (ncand < (int)top_k) {
            goto brute_force;
        }

        /* Rerank candidates: score each with RI cosine, select top-k. */
        int worker_count = fce_default_worker_count(true);
        if (worker_count < 1) worker_count = 1;

        if (worker_count <= 1 || ncand / 2 <= (int)top_k) {
            rerank_serial(corpus, qvec_q, qvec_q_inv_mag,
                          candidates, ncand, top_k, results_out, count_out);
        } else {
            /* Parallel: partition candidates across workers. */
            if (ncand / (int)top_k < worker_count) worker_count = ncand / (int)top_k;
            if (worker_count < 1) worker_count = 1;
            fce_sem_ranked_t *worker_results = malloc((size_t)worker_count * top_k * sizeof(fce_sem_ranked_t));
            int *worker_counts = calloc((size_t)worker_count, sizeof(int));
            if (!worker_results || !worker_counts) {
                free(worker_results);
                free(worker_counts);
                goto brute_force;
            }

            rerank_ctx_t ctx = {
                .corpus = corpus,
                .qvec_q = qvec_q,
                .qvec_q_inv_mag = qvec_q_inv_mag,
                .q_path = NULL,
                .q_prox = 1.0f,
                .top_k = (int)top_k,
                .candidates = candidates,
                .ncand = ncand,
                .worker_results = worker_results,
                .worker_counts = worker_counts,
            };
            atomic_init(&ctx.next_cand, 0);

            fce_parallel_for_opts_t opts = {.max_workers = worker_count};
            fce_parallel_for(worker_count, rerank_worker, &ctx, opts);

            uint32_t k = merge_worker_heaps(worker_results, worker_counts, worker_count, top_k, results_out);
            if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
            if (count_out) *count_out = k;
            free(worker_results);
            free(worker_counts);
        }
        goto cleanup;
    }

brute_force:
    /* P4: reuse pre-tokenized query instead of re-tokenizing.
     * L3: pass pre-quantized query to avoid redundant quantization. */
    bruteforce_precomputed(corpus, &qvec, qvec_q, qvec_q_inv_mag,
                           top_k, results_out, count_out);

cleanup:
    for (int t = 0; t < q_ntok; t++) free(q_toks[t]);
}

/* ── TF-IDF hybrid search: TF-IDF cosine candidates + RI rerank ── */

void fce_sem_search_query_tfidf(const fce_sem_corpus_t *corpus,
                                 const char *query,
                                 uint32_t top_k,
                                 fce_sem_ranked_t *results_out,
                                 uint32_t *count_out) {
    if (count_out) *count_out = 0;
    if (!corpus || !query || top_k == 0 || !results_out) return;
    if (!corpus->doc_vectors_q) return;

    int n = corpus->doc_count;
    if (n == 0) return;
    if ((int)top_k > n) top_k = (uint32_t)n;

    char *q_toks[FCE_SEM_MAX_TOKENS];
    int q_ntok = fce_sem_tokenize(query, q_toks, FCE_SEM_MAX_TOKENS);
    if (q_ntok == 0) goto cleanup_tf;

    fce_sem_vec_t qvec;
    memset(&qvec, 0, sizeof(qvec));
    for (int t = 0; t < q_ntok; t++) {
        const fce_sem_vec_t *rv = fce_sem_corpus_ri_vec(corpus, q_toks[t]);
        if (rv) fce_sem_vec_add_scaled(&qvec, rv, 1.0f);
    }
    fce_sem_normalize(&qvec);

    int q_tok_ids[FCE_SEM_MAX_TOKENS];
    int q_id_count = 0;
    for (int t = 0; t < q_ntok; t++) {
        void *ptr = fce_ht_get(corpus->token_map, q_toks[t]);
        if (ptr) q_tok_ids[q_id_count++] = ptr_to_token_idx(ptr);
    }

    /* F2: pre-quantize query for int8 rerank path. */
    int8_t qvec_q_buf_tf[FCE_SEM_DIM];
    float qvec_q_inv_mag_tf = 0.0f;
    const int8_t *qvec_q_tf = NULL;
    if (corpus->doc_vectors_q) {
        fce_quantize_f32_768(qvec_q_buf_tf, qvec.v);
        float mag_sq = 0.0f;
        for (int i = 0; i < FCE_SEM_DIM; i++) mag_sq += (float)qvec_q_buf_tf[i] * (float)qvec_q_buf_tf[i];
        qvec_q_inv_mag_tf = (mag_sq > 0.0f) ? 1.0f / sqrtf(mag_sq) : 0.0f;
        qvec_q_tf = qvec_q_buf_tf;
    }

    if (corpus->inv_offsets && q_id_count > 0) {
        int candidates[FCE_CANDIDATE_CAP];
        int ncand = tfidf_keyword_candidates(corpus, q_tok_ids, q_id_count,
                                              candidates, FCE_CANDIDATE_CAP);
        if (ncand < (int)top_k) goto brute_tf;

        int worker_count = fce_default_worker_count(true);
        if (worker_count < 1) worker_count = 1;

        if (worker_count <= 1 || ncand / 2 <= (int)top_k) {
            rerank_serial(corpus, qvec_q_tf, qvec_q_inv_mag_tf,
                          candidates, ncand, top_k, results_out, count_out);
        } else {
            if (ncand / (int)top_k < worker_count) worker_count = ncand / (int)top_k;
            if (worker_count < 1) worker_count = 1;
            fce_sem_ranked_t *worker_results = malloc((size_t)worker_count * top_k * sizeof(fce_sem_ranked_t));
            int *worker_counts = calloc((size_t)worker_count, sizeof(int));
            if (!worker_results || !worker_counts) {
                free(worker_results); free(worker_counts);
                goto brute_tf;
            }
            rerank_ctx_t ctx = {
                .corpus = corpus,
                .qvec_q = qvec_q_tf, .qvec_q_inv_mag = qvec_q_inv_mag_tf,
                .q_path = NULL, .q_prox = 1.0f,
                .top_k = (int)top_k, .candidates = candidates,
                .ncand = ncand, .worker_results = worker_results,
                .worker_counts = worker_counts,
            };
            atomic_init(&ctx.next_cand, 0);
            fce_parallel_for_opts_t opts = {.max_workers = worker_count};
            fce_parallel_for(worker_count, rerank_worker, &ctx, opts);
            uint32_t k = merge_worker_heaps(worker_results, worker_counts, worker_count, top_k, results_out);
            if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
            if (count_out) *count_out = k;
            free(worker_results); free(worker_counts);
        }
        goto cleanup_tf;
    }

brute_tf:
    /* P4: reuse pre-tokenized query instead of re-tokenizing.
     * L3: pass pre-quantized query to avoid redundant quantization. */
    bruteforce_precomputed(corpus, &qvec, qvec_q_tf, qvec_q_inv_mag_tf,
                           top_k, results_out, count_out);

cleanup_tf:
    for (int t = 0; t < q_ntok; t++) free(q_toks[t]);
}

int fce_sem_search_candidate_count(const fce_sem_corpus_t *corpus,
                                    const char *query) {
    if (!corpus || !query) return 0;
    if (!corpus->inv_offsets) return 0;

    char *q_toks[FCE_SEM_MAX_TOKENS];
    int q_ntok = fce_sem_tokenize(query, q_toks, FCE_SEM_MAX_TOKENS);
    if (q_ntok == 0) return 0;

    int q_tok_ids[FCE_SEM_MAX_TOKENS];
    int q_id_count = 0;
    for (int t = 0; t < q_ntok; t++) {
        void *ptr = fce_ht_get(corpus->token_map, q_toks[t]);
        if (ptr) q_tok_ids[q_id_count++] = ptr_to_token_idx(ptr);
    }

    int candidates[FCE_CANDIDATE_CAP];
    int ncand = keyword_candidates(corpus, q_tok_ids, q_id_count,
                                    candidates, FCE_CANDIDATE_CAP);

    for (int t = 0; t < q_ntok; t++) free(q_toks[t]);
    return ncand;
}

/* ── Ranking / Search ────────────────────────────────────────────── */

void fce_sem_rank(fce_sem_func_t *query, fce_sem_func_t *corpus,
                   uint32_t corpus_size, uint32_t top_k,
                   const fce_sem_config_t *cfg,
                   fce_sem_ranked_t *results_out, uint32_t *count_out) {
    fce_sem_search(query, corpus, corpus_size, top_k, 0.0f, cfg, results_out, count_out);
}

void fce_sem_search(fce_sem_func_t *query, fce_sem_func_t *corpus,
                     uint32_t corpus_size, uint32_t top_k, float min_score,
                     const fce_sem_config_t *cfg,
                     fce_sem_ranked_t *results_out, uint32_t *count_out) {
    if (!query || !corpus || corpus_size == 0 || top_k == 0) {
        if (count_out) *count_out = 0;
        return;
    }
    /* Validate query-side TF-IDF sort invariant once at entry. */
    for (int i = 1; i < query->tfidf_len; i++) {
        if (query->tfidf_indices[i] < query->tfidf_indices[i-1]) {
            fce_log_error("fce_sem_search: unsorted query TF-IDF indices at %d", i);
            if (count_out) *count_out = 0;
            return;
        }
    }

    int n = (int)corpus_size;
    if ((int)top_k > n) top_k = (uint32_t)n;

    int worker_count = fce_default_worker_count(true);
    if (worker_count < 1) worker_count = 1;

    /* Small corpus or single worker: serial path. */
    if (worker_count <= 1 || n / 2 <= (int)top_k) {
        sc_sctx_t ctx = {
            .query = query, .corpus = corpus, .cfg = cfg,
            .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
            .q_tfidf_mag_sq = NAN, .q_ri_mag_sq = NAN,
            .q_api_mag_sq = NAN, .q_type_mag_sq = NAN, .q_deco_mag_sq = NAN,
            .q_sp_mag_sq = NAN,
        };
        if (count_out) *count_out = serial_topk(sc_score, &ctx, corpus_size, top_k, min_score, results_out);
        return;
    }

    /* Parallel path: partition corpus across workers, each builds a local top-k. */
    if (n / (int)top_k < worker_count) worker_count = n / (int)top_k;
    if (worker_count < 1) worker_count = 1;
    fce_sem_ranked_t *worker_results = malloc((size_t)worker_count * top_k * sizeof(fce_sem_ranked_t));
    int *worker_counts = calloc((size_t)worker_count, sizeof(int));
    if (!worker_results || !worker_counts) {
        free(worker_results);
        free(worker_counts);
        /* Fallback to serial on OOM. */
        sc_sctx_t ctx = {
            .query = query, .corpus = corpus, .cfg = cfg,
            .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
            .q_tfidf_mag_sq = NAN, .q_ri_mag_sq = NAN,
            .q_api_mag_sq = NAN, .q_type_mag_sq = NAN, .q_deco_mag_sq = NAN,
            .q_sp_mag_sq = NAN,
        };
        if (count_out) *count_out = serial_topk(sc_score, &ctx, corpus_size, top_k, min_score, results_out);
        return;
    }

    float q_prox = fce_count_slashes(query->file_path);

    /* Precompute query-side magnitudes once (P1) — avoids ~3072 FLOPs per corpus item. */
    float q_tfidf_mag_sq = NAN;
    if (query->tfidf_len > 0) {
        float m = 0.0F;
        for (int i = 0; i < query->tfidf_len; i++) {
            m += query->tfidf_weights[i] * query->tfidf_weights[i];
        }
        q_tfidf_mag_sq = m;
    }
    float q_ri_mag_sq = 0.0F;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        q_ri_mag_sq += query->ri_vec.v[i] * query->ri_vec.v[i];
    }
    float q_api_mag_sq = 0.0F;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        q_api_mag_sq += query->api_vec.v[i] * query->api_vec.v[i];
    }
    float q_type_mag_sq = 0.0F;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        q_type_mag_sq += query->type_vec.v[i] * query->type_vec.v[i];
    }
    float q_deco_mag_sq = 0.0F;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        q_deco_mag_sq += query->deco_vec.v[i] * query->deco_vec.v[i];
    }
    float q_sp_mag_sq = 0.0F;
    for (int i = 0; i < FCE_SEM_AST_PROFILE_DIMS; i++) {
        q_sp_mag_sq += query->struct_profile[i] * query->struct_profile[i];
    }

    sc_ctx_t ctx = {
        .query = query,
        .corpus = corpus,
        .corpus_size = corpus_size,
        .min_score = min_score,
        .q_prox = q_prox,
        .q_tfidf_mag_sq = q_tfidf_mag_sq,
        .q_ri_mag_sq = q_ri_mag_sq,
        .q_api_mag_sq = q_api_mag_sq,
        .q_type_mag_sq = q_type_mag_sq,
        .q_deco_mag_sq = q_deco_mag_sq,
        .q_sp_mag_sq = q_sp_mag_sq,
        .cfg = cfg,
        .top_k = (int)top_k,
        .worker_results = worker_results,
        .worker_counts = worker_counts,
    };
    atomic_init(&ctx.next_doc, 0);
    fce_parallel_for_opts_t opts = {.max_workers = worker_count};
    fce_parallel_for(worker_count, sc_worker, &ctx, opts);

    uint32_t k = merge_worker_heaps(worker_results, worker_counts, worker_count, top_k, results_out);
    if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
    if (count_out) *count_out = k;
    free(worker_results);
    free(worker_counts);
}

/* ── Simple API ──────────────────────────────────────────────────── */

/* Internal simple scoring with precomputed proximity (P3) and precomputed
 * query-side magnitudes (P1). Pass NAN to compute inline. */
static float fce_score_simple_internal(fce_sem_func_t *a, fce_sem_func_t *b, float prox,
                                    float q_ri_mag_sq) {
    if (!a || !b) return 0.0f;

    /* RI cosine — use precomputed query magnitude if provided.
     * H1: TF-IDF dropped from the simple API because buildFunc uses positional
     * indices (0,1,2,...) not global vocab IDs, making the sparse cosine merge
     * meaningless.  The RI half is correct. */
    float ri_mag = q_ri_mag_sq;
    if (isnan(ri_mag)) {
        ri_mag = 0.0f;
        for (int i = 0; i < FCE_SEM_DIM; i++) {
            ri_mag += a->ri_vec.v[i] * a->ri_vec.v[i];
        }
    }
    float ri_raw = fce_sem_cosine_aliased_with_mag(a->ri_vec.v, b->ri_vec.v, ri_mag);
    float ri = (ri_raw + FCE_SEM_UNIT_POS) * 0.5f;

    ri *= prox;
    if (!isfinite(ri)) return 0.0f;  /* C1: sanitize NaN/Inf from caller vectors */
    if (ri > FCE_SEM_UNIT_POS) ri = FCE_SEM_UNIT_POS;
    if (ri < 0.0f) ri = 0.0f;
    return ri;
}

float fce_sem_simple_score(fce_sem_func_t *a, fce_sem_func_t *b) {
    float prox = fce_sem_proximity(a ? a->file_path : NULL, b ? b->file_path : NULL);
    return fce_score_simple_internal(a, b, prox, NAN);
}

void fce_sem_simple_rank(fce_sem_func_t *query, fce_sem_func_t *corpus,
                          uint32_t corpus_size, uint32_t top_k,
                          fce_sem_ranked_t *results_out, uint32_t *count_out) {
    fce_sem_simple_search(query, corpus, corpus_size, top_k, 0.0f, results_out, count_out);
}

void fce_sem_simple_search(fce_sem_func_t *query, fce_sem_func_t *corpus,
                            uint32_t corpus_size, uint32_t top_k,
                            float min_score,
                            fce_sem_ranked_t *results_out, uint32_t *count_out) {
    if (!query || !corpus || corpus_size == 0 || top_k == 0) {
        if (count_out) *count_out = 0;
        return;
    }
    /* Validate query-side TF-IDF sort invariant once at entry. */
    for (int i = 1; i < query->tfidf_len; i++) {
        if (query->tfidf_indices[i] < query->tfidf_indices[i-1]) {
            fce_log_error("fce_sem_simple_search: unsorted query TF-IDF indices at %d", i);
            if (count_out) *count_out = 0;
            return;
        }
    }

    int n = (int)corpus_size;
    if ((int)top_k > n) top_k = (uint32_t)n;

    int worker_count = fce_default_worker_count(true);
    if (worker_count < 1) worker_count = 1;

    /* Small corpus or single worker: serial path. */
    if (worker_count <= 1 || n / 2 <= (int)top_k) {
        fce_ss_sctx_t ctx = {
            .query = query, .corpus = corpus,
            .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
            .q_ri_mag_sq = NAN,
        };
        if (count_out) *count_out = serial_topk(fce_ss_score, &ctx, corpus_size, top_k, min_score, results_out);
        return;
    }

    /* Parallel path. */
    if (n / (int)top_k < worker_count) worker_count = n / (int)top_k;
    if (worker_count < 1) worker_count = 1;
    fce_sem_ranked_t *worker_results = malloc((size_t)worker_count * top_k * sizeof(fce_sem_ranked_t));
    int *worker_counts = calloc((size_t)worker_count, sizeof(int));
    if (!worker_results || !worker_counts) {
        free(worker_results);
        free(worker_counts);
        /* Fallback to serial on OOM. */
        fce_ss_sctx_t ctx = {
            .query = query, .corpus = corpus,
            .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
            .q_ri_mag_sq = NAN,
        };
        if (count_out) *count_out = serial_topk(fce_ss_score, &ctx, corpus_size, top_k, min_score, results_out);
        return;
    }

    float q_prox = fce_count_slashes(query->file_path);

    /* Precompute query-side RI magnitude once (P1). */
    float q_ri_mag_sq = 0.0F;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        q_ri_mag_sq += query->ri_vec.v[i] * query->ri_vec.v[i];
    }

    fce_ss_ctx_t ctx = {
        .query = query,
        .corpus = corpus,
        .corpus_size = corpus_size,
        .min_score = min_score,
        .q_prox = q_prox,
        .q_ri_mag_sq = q_ri_mag_sq,
        .top_k = (int)top_k,
        .worker_results = worker_results,
        .worker_counts = worker_counts,
    };
    atomic_init(&ctx.next_doc, 0);
    fce_parallel_for_opts_t opts = {.max_workers = worker_count};
    fce_parallel_for(worker_count, ss_worker, &ctx, opts);

    uint32_t k = merge_worker_heaps(worker_results, worker_counts, worker_count, top_k, results_out);
    if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
    if (count_out) *count_out = k;
    free(worker_results);
    free(worker_counts);
}

/* Flat-data scorer: avoids 3KB memcpy per iteration by aliasing into the
 * flat arrays directly. Reads only the fields used by fce_sem_simple_score.
 * q_ri_mag is a precomputed query-side RI magnitude passed in to avoid
 * recomputing it on every corpus item (see M4).
 *
 * NOTE: the flat API has no query file path, so proximity is always 1.0.
 * This scorer applies RI only — TF-IDF is dropped because positional
 * indices (0,1,2,…) are not global vocab IDs, making the sparse cosine
 * merge meaningless (H1).
 * For proximity-weighted results, use fce_score_simple_internal with structs. */
static float fce_score_flat(
    const int *q_idx, const float *q_w, int q_len, const float *q_ri,
    float q_ri_mag,
    const int *c_idx, const float *c_w, int c_len, const float *c_ri,
    const char *q_path, const char *c_path) {
    (void)q_idx; (void)q_w; (void)q_len;
    (void)c_idx; (void)c_w; (void)c_len;
    (void)q_path; (void)c_path;
    float ri_raw = fce_sem_cosine_aliased_with_mag(q_ri, c_ri, q_ri_mag);
    float ri = (ri_raw + FCE_SEM_UNIT_POS) * 0.5f;
    if (!isfinite(ri)) return 0.0f;
    if (ri > FCE_SEM_UNIT_POS) ri = FCE_SEM_UNIT_POS;
    if (ri < 0.0f) ri = 0.0f;
    return ri;
}

void fce_sem_simple_rank_flat(
    const float *all_tfidf_weights,
    const int   *all_tfidf_indices,
    const int   *tfidf_lens,
    const float *all_ri_vecs,
    const char **file_paths,
    uint32_t corpus_size,
    int max_tokens,
    const int   *q_tfidf_indices,
    const float *q_tfidf_weights,
    int q_tfidf_len,
    const float *q_ri_vec,
    uint32_t top_k,
    fce_sem_ranked_t *results_out,
    uint32_t *count_out) {

    if (!all_tfidf_weights || corpus_size == 0 || top_k == 0) {
        if (count_out) *count_out = 0;
        return;
    }

    /* Precompute query-side RI magnitude once. */
    float q_ri_mag = 0.0F;
    if (q_ri_vec) {
        for (int i = 0; i < FCE_SEM_DIM; i++) {
            q_ri_mag += q_ri_vec[i] * q_ri_vec[i];
        }
    }

    int n = (int)corpus_size;
    if ((int)top_k > n) top_k = (uint32_t)n;

    /* P1: Validate query-side TF-IDF indices once here, instead of per-corpus-item
     * inside fce_sparse_tfidf_cosine_flat. Query array is identical across all items. */
    for (int i = 1; i < q_tfidf_len; i++) {
        if (q_tfidf_indices[i] < q_tfidf_indices[i-1]) {
            fce_log_error("fce_sem_simple_rank_flat: unsorted query TF-IDF indices at %d", i);
            if (count_out) *count_out = 0;
            return;
        }
    }

    int worker_count = fce_default_worker_count(true);
    if (worker_count < 1) worker_count = 1;

    /* Small corpus or single worker: serial path. */
    if (worker_count <= 1 || n / 2 <= (int)top_k) {
        sf_sctx_t ctx = {
            .all_tfidf_weights = all_tfidf_weights, .all_tfidf_indices = all_tfidf_indices,
            .tfidf_lens = tfidf_lens, .all_ri_vecs = all_ri_vecs, .file_paths = file_paths,
            .max_tokens = max_tokens,
            .q_tfidf_indices = q_tfidf_indices, .q_tfidf_weights = q_tfidf_weights,
            .q_tfidf_len = q_tfidf_len, .q_ri_vec = q_ri_vec, .q_ri_mag = q_ri_mag,
        };
        if (count_out) *count_out = serial_topk(sf_score, &ctx, corpus_size, top_k, -1.0f, results_out);
        return;
    }

    /* Parallel path. */
    if (n / (int)top_k < worker_count) worker_count = n / (int)top_k;
    if (worker_count < 1) worker_count = 1;
    fce_sem_ranked_t *worker_results = malloc((size_t)worker_count * top_k * sizeof(fce_sem_ranked_t));
    int *worker_counts = calloc((size_t)worker_count, sizeof(int));
    if (!worker_results || !worker_counts) {
        free(worker_results);
        free(worker_counts);
        /* Fallback to serial on OOM. */
        sf_sctx_t ctx = {
            .all_tfidf_weights = all_tfidf_weights, .all_tfidf_indices = all_tfidf_indices,
            .tfidf_lens = tfidf_lens, .all_ri_vecs = all_ri_vecs, .file_paths = file_paths,
            .max_tokens = max_tokens,
            .q_tfidf_indices = q_tfidf_indices, .q_tfidf_weights = q_tfidf_weights,
            .q_tfidf_len = q_tfidf_len, .q_ri_vec = q_ri_vec, .q_ri_mag = q_ri_mag,
        };
        if (count_out) *count_out = serial_topk(sf_score, &ctx, corpus_size, top_k, -1.0f, results_out);
        return;
    }

    flat_ctx_t ctx = {
        .all_tfidf_weights = all_tfidf_weights,
        .all_tfidf_indices = all_tfidf_indices,
        .tfidf_lens = tfidf_lens,
        .all_ri_vecs = all_ri_vecs,
        .file_paths = file_paths,
        .max_tokens = max_tokens,
        .corpus_size = (int)corpus_size,
        .q_tfidf_indices = q_tfidf_indices,
        .q_tfidf_weights = q_tfidf_weights,
        .q_tfidf_len = q_tfidf_len,
        .q_ri_vec = q_ri_vec,
        .q_ri_mag = q_ri_mag,
        .top_k = (int)top_k,
        .worker_results = worker_results,
        .worker_counts = worker_counts,
    };
    atomic_init(&ctx.next_doc, 0);
    fce_parallel_for_opts_t opts = {.max_workers = worker_count};
    fce_parallel_for(worker_count, flat_worker, &ctx, opts);

    uint32_t k = merge_worker_heaps(worker_results, worker_counts, worker_count, top_k, results_out);
    if (k > 1) qsort(results_out, k, sizeof(fce_sem_ranked_t), fce_ranked_cmp_desc);
    if (count_out) *count_out = k;
    free(worker_results);
    free(worker_counts);
}

/* ── Graph diffusion ─────────────────────────────────────────────── */

void fce_sem_diffuse(fce_sem_vec_t *combined, const fce_sem_vec_t *neighbors, int neighbor_count,
                     float alpha) {
    if (!combined || !neighbors || neighbor_count <= 0) {
        return;
    }
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;
    /* Blend: combined = (1-α) × combined + α × mean(neighbors) */
    fce_sem_vec_t mean;
    memset(&mean, 0, sizeof(mean));
    for (int n = 0; n < neighbor_count; n++) {
        for (int i = 0; i < FCE_SEM_DIM; i++) {
            mean.v[i] += neighbors[n].v[i];
        }
    }
    float inv_n = FCE_SEM_UNIT_POS / (float)neighbor_count;
    float one_minus_alpha = FCE_SEM_UNIT_POS - alpha;
    for (int i = 0; i < FCE_SEM_DIM; i++) {
        combined->v[i] = (one_minus_alpha * combined->v[i]) + (alpha * mean.v[i] * inv_n);
    }
    fce_sem_normalize(combined);
}
