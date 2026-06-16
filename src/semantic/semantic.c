/* * semantic.c — Algorithmic code embeddings: TF-IDF, Random Indexing,
 * API/Type/Decorator signatures, combined scoring, graph diffusion.
 *
 * All signals computed from graph buffer metadata — no source file reads.
 * Uses xxHash for deterministic random vectors. Pure C, zero dependencies. */
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
#ifdef FCE_SEM_DIM_256
#include "embed/pca_projection.h"
#endif
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <limits.h>
#include <sys/stat.h>
#include <time.h>
#include <fenv.h>
#if defined(__linux__)
#include <malloc.h>
#endif
#if defined(__APPLE__)
#include <dlfcn.h>
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
#include <assert.h>
#include "embed/code_vectors.h"

#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#ifndef _WIN32
#include <sched.h>
#endif

/* ── Constants ───────────────────────────────────────────────────── */

enum {
 /* tokens >= 256 bytes are silently discarded
 * (not truncated). Only ASCII alphanumeric bytes [a-zA-Z0-9] are kept;
 * non-ASCII bytes (>= 0x80) are dropped. Behavior is locale-independent
 * (explicit ASCII-range checks, not isalnum/tolower). */
 FCE_TOKEN_BUF_LEN = 256,
 FCE_CORPUS_INIT_CAP = 4096,
 FCE_DOC_TOKENS_INIT = 64,
 FCE_RI_SEED_BASE = 0x52494E44, /* "RIND" */
 FCE_PM_UNINIT = 0, FCE_PM_INIT = 1, FCE_PM_READY = 2, FCE_PM_FAILED = 3,
 /* hard caps on vocabulary and document count.
 * ~4x the measured linux-source corpus (193K docs, ~1M vocab tokens).
 * Prevents a hostile/untrusted caller from OOMing the JVM host. */
 FCE_SEM_MAX_ENTRY_COUNT = 5000000, /* 5 M vocabulary tokens */
 FCE_SEM_MAX_DOC_COUNT = 1000000, /* 1 M documents */
};

/* Default signal weights for fce_sem_combined_score.
 * Applied weights sum to ~0.85; proximity multiplier is applied on top. */
#define FCE_SEM_W_TFIDF 0.20F
#define FCE_SEM_W_RI 0.25F
#define FCE_SEM_W_API 0.15F
#define FCE_SEM_W_TYPE 0.10F
#define FCE_SEM_W_DECORATOR 0.05F
#define FCE_SEM_W_STRUCT_PROFILE 0.10F

/* 256-dim mode: boost dimension-independent signals, reduce RI weight.
 * The truncated pretrained vectors and compressed enriched vectors produce
 * less reliable RI cosine similarity at 256 dims. Compensate by leaning
 * harder on TF-IDF (token overlap), API/type/decorator signatures, and
 * structural profile — none of which depend on embedding dimensionality. */
#ifdef FCE_SEM_DIM_256
#undef FCE_SEM_W_TFIDF
#define FCE_SEM_W_TFIDF 0.25F
#undef FCE_SEM_W_RI
#define FCE_SEM_W_RI 0.15F
#undef FCE_SEM_W_API
#define FCE_SEM_W_API 0.18F
#undef FCE_SEM_W_TYPE
#define FCE_SEM_W_TYPE 0.12F
#undef FCE_SEM_W_DECORATOR
#define FCE_SEM_W_DECORATOR 0.07F
#undef FCE_SEM_W_STRUCT_PROFILE
#define FCE_SEM_W_STRUCT_PROFILE 0.10F
#endif

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
 .query_mode = FCE_QUERY_AUTO,
 .sparse_vectors = false,
 .sparse_nnz = FCE_SPARSE_NNZ_DEFAULT,
 };
 /* use fce_safe_getenv instead of raw
 * getenv() — getenv() is not thread-safe per POSIX (the returned pointer
 * can be invalidated by concurrent setenv/putenv). fce_safe_getenv copies
 * to a stack buffer, which is safe for concurrent getenv() calls.
 * NOTE: fce_safe_getenv iterates environ directly and is NOT safe against
 * concurrent setenv/putenv. This is acceptable here because
 * fce_sem_get_config is only called when the caller passes cfg=NULL
 * (infrequent; typically once per search call). The hot-path env vars
 * (FCE_BRUTE_WORKERS, FCE_STACK_SIZE) are cached at init via pthread_once
 * instead. */
 char thresh_buf[32];
 const char *thresh = fce_safe_getenv("FCE_SEMANTIC_THRESHOLD", thresh_buf, sizeof(thresh_buf), NULL);
 if (thresh && thresh[0]) {
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
 /* use fce_safe_getenv instead of raw
 * getenv() for thread-safety (see fce_sem_get_config). */
 char val_buf[32];
 const char *val = fce_safe_getenv("FCE_SEMANTIC_ENABLED", val_buf, sizeof(val_buf), NULL);
 return val && val[0] == '1';
}

/* ── Token extraction ────────────────────────────────────────────── */

/* True for characters that terminate a token regardless of case.
 * LIMITATIONS:
 * - Does NOT include `'`, `@`, `|` (rare in source code but present in
 * template strings, decorators, and bitwise-or patterns respectively).
 * - Only ASCII alphanumeric bytes [a-zA-Z0-9] are accepted into tokens;
 * non-ASCII bytes (>= 0x80) are dropped (not kept as part of the token).
 * Add additional delimiters here if your target corpus uses them. */
static bool is_token_delim(char c) {
 return c == '.' || c == '/' || c == '_' || c == '-' || c == ' ' || c == '(' || c == ')' ||
 c == ',' || c == ':';
}

/* True for a token break point at position i in an identifier.
 * Splits on:
 * - lowercase → uppercase (camelCase: "parseHTTP" → "parse", "http")
 * - digit → uppercase letter (parse2JSON → "parse2", "json")
 * - uppercase → uppercase + lowercase (acronym boundary: "HTTPServer" → "http", "server")
 * Does NOT split letter→digit — letter+digit runs stay together so
 * identifiers like utf8, sha256, base64, int32, http2 are kept as one
 * token (utf8Decode → "utf8", "decode"; parse2 → "parse2"). */
static bool is_camel_break(const char *name, int i) {
 if (i <= 0) return false;
 /* self-defending guard — name[i] must be non-NUL
 * for name[i+1] to be a valid read. The caller's loop guarantees this
 * today, but a future caller passing the last-char index would read OOB
 * without this guard. */
 if (!name[i]) return false;
 char c = name[i];
 char p = name[i - 1];
 bool c_up = c >= 'A' && c <= 'Z';
 bool p_up = p >= 'A' && p <= 'Z';
 bool p_dg = p >= '0' && p <= '9';
 bool p_lo = p >= 'a' && p <= 'z';
 /* lowercase → uppercase: camelCase */
 if (c_up && p_lo) return true;
 /* digit → uppercase letter: parse2JSON */
 if (p_dg && c_up) return true;
 /* acronym boundary — two+ uppercase followed by
 * uppercase + lowercase (e.g. HTTPServer → HTTP, Server). name[i] is
 * non-NUL (loop condition), so name[i+1] is always a valid read (worst
 * case the NUL terminator). The strlen() per character was O(n²). */
 if (c_up && p_up) {
 char n = name[i + 1];
 if (n >= 'a' && n <= 'z') return true;
 }
 return false;
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

/* Shutdown state and reader count — shared by pretrained map and abbrev HT.
 * Must be declared before fce_sem_tokenize which uses them for C-2 bracketing. */
static _Atomic bool g_shutting_down = false;
static _Atomic int g_reader_count = 0;

int fce_sem_tokenize(const char *name, char **out, int max_out) {
 if (!name || !out || max_out <= 0) {
 return 0;
 }
 /* fce_sem_tokenize now uses the same Dekker-style
 * seq_cst reader-count bracket as fce_sem_random_index (500-510).
 * The public contract (semantic.h:118) states that fce_sem_shutdown
 * "MUST NOT be called concurrently with any fce_sem_* operation", so
 * the bracket is defensive — protecting against a future contract
 * relaxation rather than a current race. */
 /* Zero the entire output array so unwritten slots are deterministically
 * NULL. The abbreviation loop and downstream strlen/strcmp calls assume
 * every slot in [0, count) is a valid pointer; this defends against
 * external callers that pre-allocated a large buffer and pass max_out
 * larger than the actual token count. */
 memset(out, 0, (size_t)max_out * sizeof(char *));
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
 bool c_lo = (c >= 'a' && c <= 'z');
 bool c_up = (c >= 'A' && c <= 'Z');
 bool c_dg = (c >= '0' && c <= '9');
 if (c_lo || c_up || c_dg) {
 char lc = c_up ? (char)(c + ('a' - 'A')) : c;
 if (blen < FCE_TOKEN_BUF_LEN - 1) {
 buf[blen++] = lc;
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

 /* g_abbrev_ht is freed by fce_sem_shutdown
 * without the Dekker reader-count bracket. We bracket the read here
 * (same pattern as fce_sem_random_index vs g_pretrained_map) to
 * prevent UAF if a caller violates the no-concurrent-shutdown contract.
 * C2: the bracket now uses seq_cst + fence (matching
 * random_index) instead of the weaker acquire-only ordering.
 * increment → fence → check state → read → decrement; shutdown spins
 * on count==0.
 * M-5: ensure_abbrev_ht itself runs BEFORE the reader
 * bracket. Safety here rests on g_init_mutex serialization (both
 * ensure_abbrev_ht and fce_sem_shutdown hold g_init_mutex during their
 * critical sections), NOT on the reader bracket. A future refactor that
 * moves the free out from under the mutex would open a UAF window. */
 int orig_count = count;
 /* Promote to seq_cst + fence to match the Dekker
 * handshake in fce_sem_random_index (1055-1057). With acquire-only on the
 * RMW and no fence before the flag load, the increment and the flag-load
 * are not globally ordered against shutdown's seq_cst store/load pair.
 * In principle shutdown can observe count==0, free g_abbrev_ht, while this
 * thread has incremented but still reads a stale g_shutting_down==false —
 * a UAF that the random_index path specifically prevents. */
 atomic_fetch_add_explicit(&g_reader_count, 1, memory_order_seq_cst);
 atomic_thread_fence(memory_order_seq_cst);
 if (g_shutting_down || atomic_load_explicit(&g_abbrev_ht_state, memory_order_seq_cst) != FCE_PM_READY) {
 atomic_fetch_sub_explicit(&g_reader_count, 1, memory_order_seq_cst);
 } else if (g_abbrev_ht) {
 for (int t = 0; t < orig_count && count < max_out; t++) {
 uint64_t h = XXH3_64bits(out[t], strlen(out[t]) + 1);
 uint32_t idx = (uint32_t)(h & (ABBREV_HT_SIZE - 1));
 /* bound the probe to ABBREV_HT_SIZE iterations.
 * The _Static_assert load-factor guard keeps the table ≤75% full so
 * an empty slot is always reachable within a short probe, but a
 * missed _Static_assert after future edits could silently turn this
 * into an infinite loop. The cap converts the failure mode from a
 * hang to a silently-missed abbreviation expansion. */
 for (uint32_t p = 0; p < ABBREV_HT_SIZE && g_abbrev_ht[idx].expanded; p++) {
 if (g_abbrev_ht[idx].hash == h &&
 strcmp(g_abbrev_ht[idx].abbrev, out[t]) == 0) {
 char *exp = strdup(g_abbrev_ht[idx].expanded);
 if (exp) out[count++] = exp;
 break;
 }
 idx = (idx + 1) & (ABBREV_HT_SIZE - 1);
 }
 }
 atomic_fetch_sub_explicit(&g_reader_count, 1, memory_order_seq_cst);
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
 fce_sem_corpus_add_docs_batch(corpus, all_tokens, token_counts, count, max_tok, NULL);
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
 /* cap chunk_size to prevent excessive per-file allocation. */
 if (chunk_size > 16 * 1024 * 1024) return -1;

 int max_tok = max_tokens_per_chunk > 0 ? max_tokens_per_chunk : FCE_SEM_MAX_TOKENS;
 if (max_tok > FCE_SEM_MAX_TOKENS) max_tok = FCE_SEM_MAX_TOKENS;
 /* Process files in batches to bound peak memory. */
 int batch_cap = 5000;
 char **all_tokens = (char **)malloc((size_t)batch_cap * max_tok * sizeof(char *));
 int *token_counts = (int *)malloc((size_t)batch_cap * sizeof(int));
 int *batch_file_idx = (int *)malloc((size_t)batch_cap * sizeof(int));
 if (!all_tokens || !token_counts || !batch_file_idx) {
 free(all_tokens); free(token_counts); free(batch_file_idx);
 return -1;
 }

 /* int is safe here — FCE_SEM_MAX_DOC_COUNT (1M) fits in int (2.1B). */
 int total_docs = 0;
 int batch_used = 0;

 /* resolve FCE_MAX_FILE_SIZE ONCE up front.
 * - getenv is not thread-safe in glibc per man 3 getenv; calling it inside
 * the per-file loop exposed a UB surface if a future caller invokes
 * add_files concurrently with setenv. Hoisting fixes the race and the
 * per-call getenv overhead.
 * - errno == ERANGE means strtol overflowed LONG_MAX; previously this
 * passed the `> 0` check and silently set max_file_size to LONG_MAX,
 * defeating the cap. Now rejected. */
 size_t max_file_size = 64 * 1024 * 1024;
 char env_buf[32];
 const char *env_sz = fce_safe_getenv("FCE_MAX_FILE_SIZE", env_buf, sizeof(env_buf), "");
 if (env_sz && env_sz[0]) {
 char *endp = NULL;
 errno = 0;
 long val = strtol(env_sz, &endp, 10);
 if (endp != env_sz && *endp == '\0' && errno == 0 && val > 0) {
 /* cap at 256 MB to prevent a misconfigured
 * env var from driving a huge malloc(file_len). */
 enum { FCE_MAX_FILE_SIZE_UPPER = 256 * 1024 * 1024 };
 if (val > FCE_MAX_FILE_SIZE_UPPER) {
 fce_log(FCE_LOG_WARN,
 "FCE_MAX_FILE_SIZE exceeds 256 MB upper bound; capping", NULL);
 val = FCE_MAX_FILE_SIZE_UPPER;
 }
 max_file_size = (size_t)val;
 }
 }

 for (int fi = 0; fi < path_count; fi++) {
 if (file_doc_counts) file_doc_counts[fi] = 0;
 FILE *f = fopen(paths[fi], "rb");
 if (!f) {
 /* log skip so batch indexers can surface failures. */
 fce_log_debug("add_files.skip", "path", paths[fi], "reason", "fopen_failed");
 continue;
 }

 /* Read entire file into memory. Use fstat for portable file size
 * (ftell returns 32-bit long on Windows, wrong for files > 2 GB). */
 struct stat st;
 if (fstat(fileno(f), &st) != 0 || st.st_size <= 0) {
 fce_log_debug("add_files.skip", "path", paths[fi], "reason", "fstat_or_empty");
 fclose(f); continue;
 }
 /* cast to uint64_t (not off_t) for portable bounds
 * check. On 32-bit Windows-with-MSVC, off_t is 32-bit, and a >2 GB
 * file's st_size can wrap to negative. The uint64_t cast handles
 * 64-bit and 32-bit off_t consistently. */
 if ((uint64_t)st.st_size > (uint64_t)max_file_size) {
 fce_log_debug("add_files.skip", "path", paths[fi], "reason", "exceeds_max_file_size");
 fclose(f); continue;
 }
 size_t file_len = (size_t)st.st_size;
 char *file_buf = (char *)malloc(file_len);
 if (!file_buf) {
 fce_log_debug("add_files.skip", "path", paths[fi], "reason", "malloc_failed");
 fclose(f); continue;
 }
 size_t nread = fread(file_buf, 1, file_len, f);
 /* C-MED-5 / C-HIGH-4: ferror must be checked BEFORE fclose, since
 * fclose invalidates the FILE handle. A short read with ferror set
 * means an I/O error (not EOF); skip this file rather than indexing
 * truncated data. L-3: a short read WITHOUT
 * ferror (file truncated between fstat and fread) yields nread <
 * file_len but read_err == 0 — the partial buffer is indexed as
 * if complete. This is benign for offline indexing of trusted input
 * but worth noting: truncation mid-read produces a partially-indexed
 * file with no diagnostic. */
 int read_err = (nread != file_len && ferror(f));
 fclose(f);
 if (read_err) {
 fce_log_debug("add_files.skip", "path", paths[fi], "reason", "read_error");
 free(file_buf); continue;
 }
 /* short read without ferror (file truncated
 * between fstat and fread) yields partial content indexed as if complete.
 * Warn once so batch indexers can detect corruption. */
 if (nread != file_len) {
 fce_log_warn("add_files.truncated",
 "path", paths[fi]);
 fce_log_int(FCE_LOG_WARN, "add_files.truncated",
 "expected", (int64_t)file_len);
 fce_log_int(FCE_LOG_WARN, "add_files.truncated",
 "got", (int64_t)nread);
 }

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
 batch_file_idx[batch_used] = fi;
 batch_used++;
 total_docs++;
 if (file_doc_counts) file_doc_counts[fi]++;

 /* Flush batch when full. */
 if (batch_used >= batch_cap) {
 int before = fce_sem_corpus_doc_count(corpus);
 fce_sem_corpus_add_docs_batch(corpus, all_tokens, token_counts,
 batch_used, max_tok, NULL);
 /* distribute rejected-doc deficit across
 * the files that contributed to this batch, going from the most-
 * recent file backwards, clamping each at 0. This prevents any
 * file_doc_counts[] from going negative when a multi-file batch
 * is rolled back (OOM or cap).
 *
 * M2: Known limitation — the deficit is
 * attributed by recency of contribution within the batch, not by
 * which file's docs were actually rejected (the cap/OOM path
 * drops docs by valid-index order, not file order). So
 * file_doc_counts[] can be skewed per-file under partial batch
 * rejection when a flush spans multiple files. The aggregate
 * total_docs is always correct. Callers that map doc ids back to
 * files should treat per-file counts as approximate. */
 int accepted = fce_sem_corpus_doc_count(corpus) - before;
 if (accepted < batch_used) {
 int deficit = batch_used - accepted;
 total_docs -= deficit;
 if (file_doc_counts) {
 for (int i = batch_used - 1; i >= 0 && deficit > 0; i--) {
 int f = batch_file_idx[i];
 int adj = file_doc_counts[f] < deficit ? file_doc_counts[f] : deficit;
 file_doc_counts[f] -= adj;
 deficit -= adj;
 }
 }
 }
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
 int before = fce_sem_corpus_doc_count(corpus);
 fce_sem_corpus_add_docs_batch(corpus, all_tokens, token_counts,
 batch_used, max_tok, NULL);
 int accepted = fce_sem_corpus_doc_count(corpus) - before;
 if (accepted < batch_used) {
 int deficit = batch_used - accepted;
 total_docs -= deficit;
 if (file_doc_counts) {
 for (int i = batch_used - 1; i >= 0 && deficit > 0; i--) {
 int f = batch_file_idx[i];
 int adj = file_doc_counts[f] < deficit ? file_doc_counts[f] : deficit;
 file_doc_counts[f] -= adj;
 deficit -= adj;
 }
 }
 }
 for (int i = 0; i < batch_used; i++) {
 int b = i * max_tok;
 for (int t = 0; t < token_counts[i]; t++) free(all_tokens[b + t]);
 }
 }

 free(all_tokens);
 free(token_counts);
 free(batch_file_idx);
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
 * Use idx+1 so that index 0 maps to non-NULL (1), and NULL means "not found". */
static inline void *token_idx_to_ptr(int idx) {
 return (void *)(intptr_t)(idx + 1);
}

static inline int ptr_to_token_idx(void *ptr) {
 if (!ptr) return FCE_NOT_FOUND;
 return (int)(intptr_t)ptr - 1;
}

/* Pretrained token lookup table — built lazily on first use. */
static FCEHashTable *g_pretrained_map = NULL;

/* one-shot gate for pretrained-map OOM during population.
 * If fce_ht_set fails mid-insert (resize-OOM), affected tokens silently fall
 * through to sparse RI fallback, degrading embedding quality. */
static _Atomic int g_pretrained_map_warned = 0;

static void init_pretrained_map(void) {
 /* validate blob length before dereferencing blob_header, so the
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
 bool inserted = false;
 fce_ht_set(g_pretrained_map, tok, token_idx_to_ptr(i), &inserted);
 /* detect resize-OOM mid-population.
 * A failed insert means this token falls through to sparse RI,
 * degrading embedding quality. One-shot to avoid log flood. */
 if (!inserted && atomic_exchange_explicit(&g_pretrained_map_warned, 1, memory_order_acq_rel) == 0) {
  fce_log_warn("pretrained_map.insert_failed", "token", tok, "index", "see M3");
 }
 }
 }
}

/* Thread-safe lazy init of the pretrained token lookup map.
 * Uses a tri-state atomic (UNINIT → INIT → READY) with CAS to ensure
 * init_pretrained_map runs exactly once even if multiple threads race.
 * A mutex serializes init to prevent hash-table resize races. */
static _Atomic int g_pretrained_state = FCE_PM_UNINIT; /* 0=uninit, 1=init, 2=ready */
/* the mutex name was misleading — it guards BOTH the
 * pretrained token map AND the abbreviation hash table. Renamed to
 * g_init_mutex to reflect its broader role. */
static fce_mutex_t g_init_mutex;
static fce_once_t g_pretrained_once = FCE_ONCE_INIT;

/* cache FCE_BRUTE_WORKERS at init.
 * fce_safe_getenv iterates environ directly and is NOT safe against
 * concurrent setenv/putenv. Read once at startup, store the result,
 * and never touch environ on the hot path again.
 * L-4: the fce_once latch is never reset by
 * fce_sem_shutdown, so a changed FCE_BRUTE_WORKERS env var after
 * shutdown/re-init is silently ignored. This is consistent with the
 * env-var-read-once semantics but inconsistent with the re-init contract
 * advertised in semantic.h:148-149. Acceptable because env vars are a
 * build-time tuning mechanism, not a runtime knob. */
static int g_cached_brute_workers = 0; /* 0 = not yet cached */
static fce_once_t g_brute_workers_once = FCE_ONCE_INIT;
static void init_brute_workers(void) {
 char env_buf[32];
 const char *env_val = fce_safe_getenv("FCE_BRUTE_WORKERS", env_buf, sizeof(env_buf), "");
 if (env_val && env_val[0]) {
 char *endp = NULL;
 errno = 0;
 long v = strtol(env_val, &endp, 10);
 if (endp != env_val && *endp == '\0' && errno == 0 &&
 v >= 1 && v <= 64) {
 g_cached_brute_workers = (int)v;
 }
 }
 if (g_cached_brute_workers < 1) {
 int total_cores = fce_system_info().total_cores;
 g_cached_brute_workers = total_cores / 4;
 if (g_cached_brute_workers < 1) g_cached_brute_workers = 1;
 if (g_cached_brute_workers > 64) g_cached_brute_workers = 64;
 }
}

/* one-shot log gate. Score functions are
 * hot — emitting a warning per NaN/Inf item would flood the log when
 * a caller passes a fully-NaN corpus. atomic_exchange gives us a
 * single edge transition (0→1) per process lifetime. */
static _Atomic int g_nonfinite_warned = 0;

/* one-shot gate for the vocabulary-cap warning.
 * On a hostile corpus the per-token fce_log_warn would flood the log;
 * atomic_exchange fires it exactly once. */
static _Atomic int g_vocab_cap_warned = 0;

/* one-shot guard for document-cap warning to prevent
 * log flooding when indexing continues past the 1M cap. */
static _Atomic int g_doc_cap_warned = 0;

static void init_pretrained_mutex(void) {
 fce_mutex_init(&g_init_mutex);
}

static void ensure_abbrev_ht(const abbrev_pair_t *abbrevs) {
 if (g_shutting_down) return;
 if (atomic_load_explicit(&g_abbrev_ht_state, memory_order_acquire) == FCE_PM_READY) {
 return;
 }
 fce_once(&g_pretrained_once, init_pretrained_mutex);
 fce_mutex_lock(&g_init_mutex);
 if (atomic_load_explicit(&g_abbrev_ht_state, memory_order_acquire) != FCE_PM_READY
 && atomic_load_explicit(&g_abbrev_ht_state, memory_order_acquire) != FCE_PM_FAILED) {
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
 } else {
 /* on calloc failure, mark as FAILED
 * so subsequent calls don't retry on every tokenization (perf degrad).
 * Abbreviation expansion is simply skipped — tokens pass through
 * unchanged. */
 atomic_store_explicit(&g_abbrev_ht_state, FCE_PM_FAILED, memory_order_release);
 }
 }
 fce_mutex_unlock(&g_init_mutex);
}

static void ensure_pretrained_map(void) {
 /* the g_shutting_down read here is intentionally
 * relaxed. The flag is a hint — even if a stale `true` is observed
 * after shutdown has cleared it, the CAS in this function will succeed
 * (state == UNINIT) and we'll re-init. The mutex serialises the real
 * work, so no UAF is possible. The release/acquire on g_pretrained_state
 * is what actually synchronises the data path. */
 if (g_shutting_down) return;
 if (atomic_load_explicit(&g_pretrained_state, memory_order_acquire) == FCE_PM_READY) {
 return;
 }
 fce_once(&g_pretrained_once, init_pretrained_mutex);
 int expected = FCE_PM_UNINIT;
 if (atomic_compare_exchange_strong(&g_pretrained_state, &expected, FCE_PM_INIT)) {
 fce_mutex_lock(&g_init_mutex);
 init_pretrained_map();
 fce_mutex_unlock(&g_init_mutex);
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
 fce_mutex_lock(&g_init_mutex);
 init_pretrained_map();
 fce_mutex_unlock(&g_init_mutex);
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
 /* Dekker-style handshake.
 * Store g_shutting_down with seq_cst BEFORE reading g_reader_count.
 * Any reader that increments the count after this store will observe
 * the flag and back off (sparse fallback). Any reader that incremented
 * before this store is visible here because seq_cst prevents reordering.
 * The spin-wait ensures all in-flight readers complete before we free
 * g_pretrained_map. */
 atomic_store_explicit(&g_shutting_down, true, memory_order_seq_cst);
 atomic_thread_fence(memory_order_seq_cst);
 fce_once(&g_pretrained_once, init_pretrained_mutex);
 while (atomic_load_explicit(&g_reader_count, memory_order_seq_cst) > 0) {
 /* Yield to avoid burning a core while waiting for readers.
 * Readers are fast (one hash lookup), so this converges quickly,
 * but a reader thread may be descheduled by the OS while inside
 * the bracket; spinning without yield can cause priority inversion
 * on oversubscribed hosts. */
#ifdef _WIN32
 SwitchToThread();
#else
 sched_yield();
#endif
 }
 fce_mutex_lock(&g_init_mutex);

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

 fce_mutex_unlock(&g_init_mutex);
 /* IMPORTANT: do NOT destroy g_init_mutex here.
 *
 * ensure_pretrained_map / ensure_abbrev_ht spin on the state atomic and
 * may call fce_mutex_lock(&g_init_mutex) after observing
 * FCE_PM_UNINIT (e.g., a thread that lost the original CAS races with
 * shutdown). Destroying the mutex would mean those late lock attempts
 * touch a destroyed synchronization primitive — undefined behavior
 * (crash, pthread corruption, deadlock on Windows). The mutex is a
 * tiny allocation; we keep it alive for the lifetime of the process.
 * The data it protects is reset above; spinning threads will
 * re-initialize it via the existing once-flag. */
 /* use seq_cst for the clear, matching the seq_cst
 * entry store, so the flag participates in a single total order. */
 atomic_store_explicit(&g_shutting_down, false, memory_order_seq_cst);
}

void fce_sem_random_index(const char *token, fce_sem_vec_t *out) {
 memset(out, 0, sizeof(*out));
 if (!token) {
 return;
 }

 /* Dekker-style handshake with shutdown.
 * Increment reader count BEFORE ensure_pretrained_map and check shutdown
 * AFTER, with seq_cst ordering on both sides. If shutdown observes count==0
 * then this reader's increment hasn't happened yet — but the reader will
 * then observe g_shutting_down==true (because shutdown stored it before
 * reading the count) and back off. If the reader observes count > 0 after
 * incrementing, shutdown's spin-wait will block until this reader finishes.
 * The seq_cst ordering prevents both stores from being reordered past each
 * other (Dekker-style), closing the TOCTOU hole. */
 atomic_fetch_add_explicit(&g_reader_count, 1, memory_order_seq_cst);
 atomic_thread_fence(memory_order_seq_cst);
 if (atomic_load_explicit(&g_shutting_down, memory_order_seq_cst)) {
 atomic_fetch_sub_explicit(&g_reader_count, 1, memory_order_seq_cst);
 goto sparse_fallback;
 }

 ensure_pretrained_map();
 /* surface a one-shot warning on first lookup
 * if the pretrained map failed to initialize (corrupt/short blob).
 * This makes the silent degradation to sparse vectors visible. */
 static _Atomic bool warned_short_blob = false;
 if (!g_pretrained_map && !atomic_exchange_explicit(&warned_short_blob, true, memory_order_relaxed)) {
 fce_log_warn("pretrained vector map unavailable — using sparse fallback", "cause", "blob too short or OOM");
 }
 if (g_pretrained_map) {
 void *idx_ptr = fce_ht_get(g_pretrained_map, token);
 if (idx_ptr) {
 int idx = ptr_to_token_idx(idx_ptr);
 if (idx >= 0 && idx < FCE_PRETRAINED_TOKEN_COUNT) {
 const int8_t *pvec = fce_pretrained_vec_at(idx);
#ifdef FCE_SEM_DIM_256
 /* PCA projection: center → multiply by 768x256 matrix.
 * Preserves 84.5% variance vs 65% for naive truncation. */
 for (int d = 0; d < FCE_SEM_DIM; d++) {
 float sum = 0.0f;
 for (int k = 0; k < FCE_PRETRAINED_DIM; k++) {
 sum += ((float)pvec[k] / FCE_SEM_INT8_MAX - fce_pca_mean[k]) * fce_pca_proj[k][d];
 }
 out->v[d] = sum;
 }
#else
 for (int d = 0; d < FCE_SEM_DIM && d < FCE_PRETRAINED_DIM; d++) {
 out->v[d] = (float)pvec[d] / FCE_SEM_INT8_MAX;
 }
#endif
 atomic_fetch_sub_explicit(&g_reader_count, 1, memory_order_seq_cst);
 return;
 }
 }
 }
 atomic_fetch_sub_explicit(&g_reader_count, 1, memory_order_seq_cst);

 /* Fallback: sparse random vector for tokens not in pretrained vocab.
 * H-02: Use collision-merging like build_src_entry
 * so both code paths produce the same vector for the same token. Without
 * merging, colliding hash positions accumulate (e.g. +1 + +1 = +2),
 * producing magnitude > 1.0 inconsistent with pretrained vectors.
 * L4: hash includes NUL terminator for consistency with abbreviation and
 * token-dedup hashing. Position mod 768 gives slight bias (acceptable for RI). */
sparse_fallback:;
 uint64_t seed = XXH3_64bits(token, strlen(token) + 1);
 uint16_t tmp_idx[FCE_SEM_SPARSE_NNZE];
 float tmp_val[FCE_SEM_SPARSE_NNZE];
 int count = 0;
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
 /* Write non-zero entries */
 for (int j = 0; j < count; j++) {
 if (tmp_val[j] != 0.0F) {
 out->v[tmp_idx[j]] = tmp_val[j];
 }
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
 int8_t *enriched_vecs_q; /* int8 quantized enriched vecs (768 bytes/entry) */
 fce_sem_vec_t *doc_vectors; /* per-document vectors (sum of enriched token vecs) */
 int8_t *doc_vectors_q; /* quantized int8 doc vectors (768 bytes/doc) */
 float *doc_vectors_q_inv_mag; /* reciprocal L2 magnitude of each quantized doc vector */
 int entry_count;
 int entry_cap;
 int doc_count;
 bool finalized;
 /* track if finalize was attempted but failed.
 * On failure, doc_vectors is freed but doc_token_ids may still be valid.
 * Retrying finalize would crash (NULL deref on doc_vectors). Block retry. */
 bool finalize_failed;

 /* Sparse vector storage (top-K NNZ per vector). */
 int sparse_nnz; /* non-zero entries per vector (0 = dense mode) */
 uint16_t *enriched_sparse_idx; /* [entry_count * sparse_nnz] sorted dim indices */
 int8_t *enriched_sparse_val; /* [entry_count * sparse_nnz] quantized values */
 uint16_t *doc_sparse_idx; /* [doc_count * sparse_nnz] sorted dim indices */
 int8_t *doc_sparse_val; /* [doc_count * sparse_nnz] quantized values */

 /* Per-document token lists for co-occurrence pass */
 int **doc_token_ids;
 int *doc_token_counts;
 int doc_cap;

 /* Inverted index: token_id → list of doc_ids containing that token.
 * Built during finalize. Enables fast keyword candidate retrieval. */
 int *inv_offsets; /* inv_offsets[token_id] = start in inv_doc_ids */
 int *inv_doc_ids; /* flat array of unique doc_ids per token */
};

/* These assertions protect the corpus_mirror_t layout in tests/test_semantic.c,
 * which casts fce_sem_corpus_t* to a mirror struct to set private fields in tests.
 * A field insertion or reorder that shifts any of these fields will fail to compile. */
_Static_assert(offsetof(struct fce_sem_corpus, entry_count) == 7 * sizeof(void *),
 "corpus_mirror_t: entry_count must be 7 pointers from start");
_Static_assert(offsetof(struct fce_sem_corpus, entry_cap) ==
 offsetof(struct fce_sem_corpus, entry_count) + sizeof(int),
 "corpus_mirror_t: entry_cap must immediately follow entry_count");
_Static_assert(offsetof(struct fce_sem_corpus, doc_count) ==
 offsetof(struct fce_sem_corpus, entry_cap) + sizeof(int),
 "corpus_mirror_t: doc_count must immediately follow entry_cap");
_Static_assert(offsetof(struct fce_sem_corpus, finalized) ==
 offsetof(struct fce_sem_corpus, doc_count) + sizeof(int),
 "corpus_mirror_t: finalized must immediately follow doc_count");
_Static_assert(offsetof(struct fce_sem_corpus, finalize_failed) ==
 offsetof(struct fce_sem_corpus, finalized) + sizeof(bool),
 "corpus_mirror_t: finalize_failed must immediately follow finalized");

static int fce_corpus_get_or_add(fce_sem_corpus_t *c, const char *token) {
 /* fce_ht_set now exposes an 'inserted' out-param
 * that disambiguates "new key" (inserted=true) from "OOM" (inserted=false),
 * eliminating the previous fce_ht_get_key disambiguation dance. */
 void *existing = fce_ht_get(c->token_map, token);
 if (existing) {
 return ptr_to_token_idx(existing);
 }
 /* reject new tokens beyond the hard vocabulary cap. */
 if (c->entry_count >= FCE_SEM_MAX_ENTRY_COUNT) {
 /* rate-limit to one-shot to prevent log flooding
 * on hostile corpora with millions of unique tokens over the cap. */
 if (atomic_exchange_explicit(&g_vocab_cap_warned, 1, memory_order_acq_rel) == 0) {
 fce_log_warn("vocabulary cap reached", "limit", "5000000");
 }
 return FCE_NOT_FOUND;
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
 bool inserted = false;
 fce_ht_set(c->token_map, key, token_idx_to_ptr(idx), &inserted);
 if (!inserted) {
 /* OOM: fce_ht_set failed to insert (resize failed).
 * Free the leaked strdup'd key and roll back entry_count. */
 free(key);
 return FCE_NOT_FOUND;
 }
 c->entry_count++;
 /* Point entry at the hash table's interned key — avoids a second strdup.
 * The hash table does NOT copy the key pointer (borrowed), so 'key' is
 * the same pointer stored in the table. The key is freed by
 * fce_free_ht_kv in fce_sem_corpus_free. */
 c->entries[idx].token = key;
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
 * Real docs have ~10-50 tokens; minified JS can produce thousands.
 * The cap is centralised in the public header (FCE_SEM_MAX_TOKENS);
 * both fce_sem_corpus_add_doc and fce_sem_corpus_add_docs_batch use
 * the same value to keep the rejection criterion consistent. */
 if (count > FCE_SEM_MAX_TOKENS) {
 return;
 }
 /* reject new documents beyond the hard doc cap. */
 if (corpus->doc_count >= FCE_SEM_MAX_DOC_COUNT) {
 /* one-shot to prevent log flooding past 1M docs. */
 if (atomic_exchange_explicit(&g_doc_cap_warned, 1, memory_order_acq_rel) == 0) {
 fce_log_warn("document cap reached", "limit", "1000000");
 }
 return;
 }
 /* Track document for co-occurrence pass */
 if (corpus->doc_count >= corpus->doc_cap) {
 int new_cap =
 corpus->doc_cap < FCE_DOC_TOKENS_INIT ? FCE_DOC_TOKENS_INIT : corpus->doc_cap * 2;
 /* commit each realloc result immediately on success.
 * If only one realloc succeeds, the successfully-grown buffer is
 * larger than doc_cap but both pointers remain valid. doc_cap is
 * NOT bumped, so the next call retries the grow. This avoids the
 * previous bug where free(new_ids) freed the only live copy while
 * corpus->doc_token_ids still referenced the old (freed) block. */
 int **new_ids = realloc(corpus->doc_token_ids, (size_t)new_cap * sizeof(int *));
 if (new_ids) corpus->doc_token_ids = new_ids;
 int *new_counts = realloc(corpus->doc_token_counts, (size_t)new_cap * sizeof(int));
 if (new_counts) corpus->doc_token_counts = new_counts;
 if (!new_ids || !new_counts) {
 return;
 }
 corpus->doc_cap = new_cap;
 }
 int doc_idx = corpus->doc_count++;
 corpus->doc_token_ids[doc_idx] = malloc((size_t)count * sizeof(int));
 /* set doc_token_counts AFTER the NULL check
 * so a failed malloc doesn't leave a stale count. */
 if (!corpus->doc_token_ids[doc_idx]) {
 corpus->doc_count--;
 return;
 }
 corpus->doc_token_counts[doc_idx] = count;

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
 * Phase A (SEQUENTIAL): Scan all documents once to build the global
 * token_map (inserts unique tokens, assigns global IDs). This is
 * inherently sequential (hash table mutation), but much faster than
 * the current per-doc add_doc because we avoid the per-doc malloc of
 * the `seen` array and per-doc bookkeeping.
 * Phase B (PARALLEL): Each worker processes a chunk of docs, translates
 * tokens → global IDs via read-only token_map lookups, fills
 * doc_token_ids[d], and accumulates doc_freq contributions via atomics. */

typedef struct {
 fce_sem_corpus_t *corpus;
 char **all_tokens;
 const int *token_counts;
 int max_tokens;
 int doc_count;
 int base_doc; /* Offset into doc_token_ids/doc_token_counts arrays. */
 const int *doc_map; /* original doc_index → compacted array_idx, or NULL if identity. */
 _Atomic int *doc_freq_atomic; /* per-entry atomic counter (entry_count long) */
 _Atomic int next_idx;
 _Atomic int error; /* set to 1 on OOM — caller checks after parallel_for */
} batch_resolve_ctx_t;

/* Resolve one document: look up each token's global ID, fill the corpus
 * doc_token_ids[d], and bump the per-token doc_freq counter atomically. The
 * caller is responsible for ensuring `seen` has capacity for `count` ints
 * before calling (the worker grows its per-thread scratch buffer). */
static void batch_resolve_one_doc(batch_resolve_ctx_t *bc, int doc_index, int *seen) {
 int count = bc->token_counts[doc_index];
 int array_idx = bc->doc_map ? bc->base_doc + bc->doc_map[doc_index]
 : bc->base_doc + doc_index;
 if (count <= 0 || count > FCE_SEM_MAX_TOKENS) {
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
 /* must signal batch-wide error so the caller
 * rolls back doc_count; without this the caller sees error==0 and
 * proceeds with unresolved docs (silent data loss / UAF with C2). */
 atomic_store_explicit(&bc->error, 1, memory_order_relaxed);
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
 /* realloc failure — the
 * error flag triggers a full batch rollback at the caller.
 * Partial IDF loss from unprocessed documents is expected
 * before rollback; the error flag ensures the caller knows
 * the batch is incomplete and rolls back doc_count. */
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

/* fill caller's doc_map_out with -1 so every
 * rejection/rollback path leaves a consistent "nothing accepted" map. */
static void invalidate_doc_map(int *doc_map_out, int doc_count) {
 if (!doc_map_out) return;
 for (int _d = 0; _d < doc_count; _d++) doc_map_out[_d] = -1;
}

void fce_sem_corpus_add_docs_batch(fce_sem_corpus_t *corpus, char **all_tokens,
 const int *token_counts, int doc_count,
 int max_tokens_per_doc, int *doc_map_out) {
 if (!corpus || !all_tokens || !token_counts || doc_count <= 0 || corpus->finalized) {
 return;
 }
 if (max_tokens_per_doc <= 0) {
 fce_log_int(FCE_LOG_ERROR,
 "fce_sem_corpus_add_docs_batch.max_tokens_out_of_range",
 "value", max_tokens_per_doc);
 return;
 }
 /* REJECT batches where max_tokens_per_doc exceeds
 * FCE_SEM_MAX_TOKENS rather than silently clamping. The caller allocates
 * all_tokens with stride max_tokens_per_doc; if we clamp it to 512 here
 * but the caller's stride is larger, every document d > 0 reads token
 * pointers from the wrong row — silent corpus corruption. All in-tree
 * callers (JNI, add_files, index_dir) already use stride ≤ 512, so this
 * is a no-op for them; direct C API callers get a clear error. */
 if (max_tokens_per_doc > FCE_SEM_MAX_TOKENS) {
 fce_log_int(FCE_LOG_ERROR,
 "fce_sem_corpus_add_docs_batch.max_tokens_exceeds_cap",
 "value", max_tokens_per_doc);
 return;
 }

 /* Phase A (SEQUENTIAL): Build token_map and allocate doc arrays.
 * Hash table mutation can't be parallelized; strdup+insert is the cost.
 * Per-doc cap is FCE_SEM_MAX_TOKENS (public header, see §8.3). */

 /* First pass: count valid docs and build compacted index mapping. */
 int valid_doc_count = 0;
 int *doc_map = (int *)malloc((size_t)doc_count * sizeof(int));
 if (!doc_map) return;
 for (int d = 0; d < doc_count; d++) {
 int count = token_counts[d];
 if (count > 0 && count <= FCE_SEM_MAX_TOKENS) {
 doc_map[d] = valid_doc_count++;
 } else {
 doc_map[d] = -1; /* sentinel: skip this doc */
 }
 }

 /* if caller wants the doc_map, copy it
 * so they can map paths to the correct docs (rejected docs are skipped). */
 if (doc_map_out) {
 memcpy(doc_map_out, doc_map, (size_t)doc_count * sizeof(int));
 }

 if (valid_doc_count == 0) {
 free(doc_map);
 return;
 }

 /* reject batch if it would exceed the hard doc cap. */
 if (corpus->doc_count > FCE_SEM_MAX_DOC_COUNT - valid_doc_count) {
 fce_log_warn("document cap exceeded by batch", "limit", "1000000");
 invalidate_doc_map(doc_map_out, doc_count);
 free(doc_map);
 return;
 }

 if (corpus->doc_cap < corpus->doc_count + valid_doc_count) {
 int old_cap = corpus->doc_cap;
 int new_cap = corpus->doc_count + valid_doc_count;
 /* commit each realloc result immediately on success.
 * If only one realloc succeeds, the successfully-grown buffer is
 * larger than doc_cap but both pointers remain valid. doc_cap is
 * NOT bumped, so the next call retries the grow. This avoids the
 * previous bug where free(new_ids) freed the only live copy while
 * corpus->doc_token_ids still referenced the old (freed) block. */
 int **new_ids = realloc(corpus->doc_token_ids, (size_t)new_cap * sizeof(int *));
 if (new_ids) corpus->doc_token_ids = new_ids;
 int *new_counts = realloc(corpus->doc_token_counts, (size_t)new_cap * sizeof(int));
 if (new_counts) corpus->doc_token_counts = new_counts;
 if (!new_ids || !new_counts) {
 invalidate_doc_map(doc_map_out, doc_count);
 free(doc_map);
 return;
 }
 corpus->doc_cap = new_cap;
 /* Zero-init new slots so OOM rollback free() is safe on unprocessed entries. */
 memset(corpus->doc_token_ids + old_cap, 0, (size_t)(new_cap - old_cap) * sizeof(int *));
 memset(corpus->doc_token_counts + old_cap, 0, (size_t)(new_cap - old_cap) * sizeof(int));
 }
 int base_doc = corpus->doc_count;
 int base_entry_count = corpus->entry_count; /* remember for rollback */
 corpus->doc_count += valid_doc_count;

 /* zero-init the newly-claimed slots unconditionally.
 * When doc_cap >= new doc_count the no-realloc path is taken and the
 * memset inside the growth block is skipped. After a prior rollback,
 * those slots may hold dangling freed pointers from the rolled-back
 * batch; without zeroing, a partial-worker OOM leaves UAF/double-free
 * landmines for the next finalize or rollback. */
 memset(corpus->doc_token_ids + base_doc, 0,
 (size_t)valid_doc_count * sizeof(int *));
 memset(corpus->doc_token_counts + base_doc, 0,
 (size_t)valid_doc_count * sizeof(int));

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
 * add_doc increments it itself. Each add_doc call can fail per-doc,
 * so the indices the caller already received in doc_map_out may not
 * match what actually gets added. Invalidate to prevent a stale map. */
 invalidate_doc_map(doc_map_out, doc_count);
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

 /* If any worker hit OOM, the batch is incomplete — roll back doc_count
 * and Phase-A vocabulary inserts. */
 if (atomic_load_explicit(&bc.error, memory_order_relaxed)) {
 fce_log_error("batch.resolve.oom", "detail", "rolling back documents");
 for (int d = base_doc; d < corpus->doc_count; d++) {
 free(corpus->doc_token_ids[d]);
 /* NULL the slot so a subsequent batch that
 * fits inside doc_cap doesn't inherit a dangling pointer (the
 * no-realloc path skips memset). */
 corpus->doc_token_ids[d] = NULL;
 }
 corpus->doc_count = base_doc;
 /* Roll back Phase-A vocabulary inserts. Truncate entries[]
 * and rebuild the hash table so tokens inserted this batch are
 * removed and doc_freq for existing tokens stays at 0 (Phase C
 * was skipped). */
 if (corpus->entry_count > base_entry_count) {
 for (int i = base_entry_count; i < corpus->entry_count; i++) {
 free((char *)corpus->entries[i].token);
 }
 corpus->entry_count = base_entry_count;
 fce_ht_clear(corpus->token_map);
 bool ins_ok = true;
 for (int i = 0; i < corpus->entry_count; i++) {
 void *vp = fce_ht_set(corpus->token_map, corpus->entries[i].token,
 token_idx_to_ptr(i), &ins_ok);
 (void)vp;
 if (!ins_ok) break;
 }
 /* if the reinsert loop failed (second OOM),
 * token_map is partially rebuilt — some entries are invisible to
 * fce_ht_get, causing silent corpus corruption (duplicate vocab,
 * wrong IDF). Clear the map and entries so subsequent operations
 * see an empty (but not corrupt) corpus. */
 if (!ins_ok) {
 fce_log_error("batch.rollback.double_oom",
 "detail", "hash table reinsert failed; "
 "corpus vocabulary cleared");
 for (int i = 0; i < corpus->entry_count; i++) {
 free((char *)corpus->entries[i].token);
 }
 corpus->entry_count = 0;
 fce_ht_clear(corpus->token_map);
 }
 }
 invalidate_doc_map(doc_map_out, doc_count);
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
 * 1. Precompute base RI vectors into a shared array (eliminates ~333M
 * redundant fce_sem_random_index calls on kernel-scale corpora).
 * 2. Co-occurrence passes: partition TARGET tokens across workers so each
 * worker writes to a disjoint range of fce_enriched_vec (zero contention).
 * Each worker still scans all documents but only accumulates for targets
 * in its range. Inner vector add is the parallelized work.
 * 3. Normalize/blend loops are trivially parallel per-entry. */

/* Reverse index: for each token id, a list of (doc_id, position_in_doc) pairs.
 * Built once, reused for both cooccur passes. Eliminates the O(num_chunks × doc_count)
 * redundant outer scan in the old algorithm. */
typedef struct {
 int32_t doc_id;
 int32_t pos;
} cooccur_pos_t;

typedef struct {
 int *offsets; /* offsets[entry_count + 1], prefix sum of occurrences.
 * INVARIANT: allocated length is `entry_count + 1`, so
 * a read of `offsets[tid + 1]` is safe for any
 * `tid ∈ [0, entry_count)`. The cursor in
 * cooccur_sparse_one_target / cooccur_int8_one_target
 * is built from corpus token ids (also in
 * [0, entry_count)), so the +1 read is in-bounds.
 * If you ever change `tid` to a value that can be
 * `entry_count`, add the explicit guard. */
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
 uint8_t nnz; /* number of nonzeros used in sparse path */
 uint16_t _pad;
 uint16_t indices[FCE_SEM_SPARSE_NNZE]; /* 8 * 2 = 16 bytes */
 float values[FCE_SEM_SPARSE_NNZE]; /* 8 * 4 = 32 bytes */
 const int8_t *dense_int8; /* points into FCE_PRETRAINED_VECTOR_BLOB */
} fce_sem_src_entry_t;

/* Inline helper: initialize a target vector from a sparse/dense source.
 * C1: In 256-dim mode, the dense path must use the same PCA
 * projection as fce_sem_random_index so that corpus vectors and RI queries
 * live in the same basis. Truncation would produce incompatible vectors. */
static inline void sem_target_init_from_src(fce_sem_vec_t *dst, const fce_sem_src_entry_t *src) {
 memset(dst, 0, sizeof(*dst));
 if (src->is_sparse) {
 for (int k = 0; k < src->nnz; k++) {
 dst->v[src->indices[k]] = src->values[k];
 }
 } else {
 const int8_t *s = src->dense_int8;
#ifdef FCE_SEM_DIM_256
 /* PCA projection: center → multiply by 768×256 matrix.
 * Same as fce_sem_random_index so corpus vectors and RI queries
 * are in the same basis. */
 for (int d = 0; d < FCE_SEM_DIM; d++) {
 float sum = 0.0f;
 for (int k = 0; k < FCE_PRETRAINED_DIM; k++) {
 sum += ((float)s[k] / FCE_SEM_INT8_MAX - fce_pca_mean[k]) * fce_pca_proj[k][d];
 }
 dst->v[d] = sum;
 }
#else
 const float inv127 = FCE_SEM_UNIT_POS / FCE_SEM_INT8_MAX;
 /* Do NOT replace with fce_init_f32_from_i8_768() — see note in fce_sem_vec_add_scaled. */
 for (int d = 0; d < FCE_SEM_DIM; d++) {
 dst->v[d] = inv127 * (float)s[d];
 }
#endif
 }
}

/* Inline helper: add weighted source into target.
 * C1: In 256-dim mode, the dense path must apply PCA projection
 * so that corpus vectors and RI queries live in the same basis.
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
#ifdef FCE_SEM_DIM_256
 /* PCA projection: center → multiply by 768×256 matrix, then scale. */
 for (int d = 0; d < FCE_SEM_DIM; d++) {
 float sum = 0.0f;
 for (int k = 0; k < FCE_PRETRAINED_DIM; k++) {
 sum += ((float)s[k] / FCE_SEM_INT8_MAX - fce_pca_mean[k]) * fce_pca_proj[k][d];
 }
 dst->v[d] += scale * sum;
 }
#else
 const float mul = scale * (FCE_SEM_UNIT_POS / FCE_SEM_INT8_MAX);
 /* Do NOT replace with fce_axpy_i8_768() — see note in fce_sem_vec_add_scaled. */
 for (int d = 0; d < FCE_SEM_DIM; d++) {
 dst->v[d] += mul * (float)s[d];
 }
#endif
 }
}

/* Pass 1 context: uses sparse/int8 tagged sources (most memory-efficient).
 * R4: writes directly to int8 pass1_q via per-worker scratch normalization.
 * Falls back to float32 enriched_vecs if enriched_vecs_q is NULL. */
typedef struct {
 fce_sem_vec_t *enriched_vecs; /* float32 fallback (NULL in R4 path) */
 int8_t *enriched_vecs_q; /* int8 output (NULL in fallback path) */
 int8_t *pass1_q; /* int8 pass1 output (NULL in fallback path) */
 fce_sem_vec_t *scratch; /* per-worker scratch for normalize-before-quantize */
 int scratch_count; /* size of scratch[] for bounds checking */
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
 if (nid < 0 || nid >= cc->entry_count) {
 continue;
 }
 float weight = FCE_SEM_UNIT_POS / (float)abs(w);
 sem_vec_add_src_scaled(target, &cc->src_entries[nid], weight);
 }
 }
}

static void cooccur_worker_sparse(int worker_id, void *ctx_ptr) {
 /* CONTRACT: worker_id is the iteration index from fce_parallel_for, and
 * the caller invokes fce_parallel_for(count=worker_count, ...). This means
 * worker_id ∈ [0, worker_count) which is exactly the valid index range
 * for cc->scratch[worker_id]. Do NOT change the parallel_for count to a
 * different value, and do not interpret worker_id as a thread number —
 * it's an array index, period. */
 cooccur_sparse_ctx_t *cc = ctx_ptr;
 /* Latent-2: enforce scratch[worker_id] bounds at runtime.
 * The scratch array is sized to scratch_count (= worker_count at the
 * call site). A future caller that passes a different parallel_for count
 * would silently write OOB — this guard converts that into a safe return
 * instead of a heap corruption. */
 if (worker_id < 0 || worker_id >= cc->scratch_count) return;
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
 /* accumulate into per-worker scratch, normalize, quantize
 * directly to int8 pass1_q. No float32 enriched_vecs needed. */
 fce_sem_vec_t *scratch = &cc->scratch[worker_id];
 sem_target_init_from_src(scratch, &cc->src_entries[tid]);
 cooccur_sparse_one_target(cc, tid, scratch);
 fce_sem_normalize(scratch);
 /* the scalar NaN/Inf sweep is
 * redundant in the R4 path because fce_quantize_f32_768
 * already neutralizes NaN/Inf lanes (simd_dot768.h).
 * The fallback path (float32 enriched_vecs) has no such
 * clamp and still needs its sweep. */
 fce_quantize_f32_768(&cc->pass1_q[(size_t)tid * FCE_SEM_DIM], scratch->v);
 } else {
 /* Fallback: accumulate into float32 enriched_vecs, normalize only.
 * Quantization happens later in finalize_pass2. */
 sem_target_init_from_src(&cc->enriched_vecs[tid], &cc->src_entries[tid]);
 cooccur_sparse_one_target(cc, tid, &cc->enriched_vecs[tid]);
 /* detect float accumulator overflow
 * (very dense corpora can produce non-finite values in the
 * R4-fallback float32 path). The R4 path quantizes each
 * pass1 vector to int8 immediately, which clamps any Inf
 * to ±127 before it propagates; the fallback path has no
 * such clamp. Sanitize here so a NaN/Inf target doesn't
 * survive into pass2 and poison the int8 quantizer. */
 fce_sem_normalize(&cc->enriched_vecs[tid]);
 bool nonfinite = false;
 for (int d = 0; d < FCE_SEM_DIM; d++) {
 if (!isfinite(cc->enriched_vecs[tid].v[d])) {
 nonfinite = true;
 break;
 }
 }
 if (nonfinite) {
 memset(&cc->enriched_vecs[tid], 0, sizeof(cc->enriched_vecs[tid]));
 }
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
 fce_sem_vec_t *enriched_vecs; /* float32 fallback (NULL in R4 path) */
 int8_t *enriched_vecs_q; /* int8 output (NULL in fallback path) */
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
 if (nid < 0 || nid >= cc->entry_count) {
 continue;
 }
 float weight = FCE_SEM_UNIT_POS / (float)abs(w);
 sem_vec_add_int8_scaled(target, &cc->pass1_q[(size_t)nid * FCE_SEM_DIM], weight);
 }
 }
}

static void cooccur_worker_int8(int worker_id, void *ctx_ptr) {
 /* cooccur_worker_sparse has a bounds guard
 * on worker_id vs scratch_count, but cooccur_worker_int8 does NOT need one
 * because it doesn't use the scratch array at all — work is dispatched via
 * the atomic next_chunk counter, not worker_id. The (void)worker_id cast
 * confirms this parameter is unused. */
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
 /* quantize pass2 to int8, blend with normalized pass1,
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
 /* redundant NaN/Inf sweep removed —
 * fce_quantize_f32_768 already handles NaN/Inf lanes. */
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
 /* Dense path: direct int8 pointer into pretrained blob (zero-copy).
 * L2: participate in the reader-count bracket
 * for consistency with the Dekker handshake in fce_sem_random_index.
 * This function is called from parallel workers during finalize, so
 * shutdown cannot occur (finalize owns the corpus), but the bracket
 * is cheap and makes the safety model uniform. */
 atomic_fetch_add_explicit(&g_reader_count, 1, memory_order_seq_cst);
 atomic_thread_fence(memory_order_seq_cst);
 if (atomic_load_explicit(&g_shutting_down, memory_order_seq_cst)) {
 atomic_fetch_sub_explicit(&g_reader_count, 1, memory_order_seq_cst);
 goto sparse_entry;
 }
 if (g_pretrained_map) {
 void *idx_ptr = fce_ht_get(g_pretrained_map, token);
 if (idx_ptr) {
 int idx = ptr_to_token_idx(idx_ptr);
 if (idx >= 0 && idx < FCE_PRETRAINED_TOKEN_COUNT) {
 out->is_sparse = 0;
 out->dense_int8 = fce_pretrained_vec_at(idx);
 atomic_fetch_sub_explicit(&g_reader_count, 1, memory_order_seq_cst);
 return;
 }
 }
 }
 atomic_fetch_sub_explicit(&g_reader_count, 1, memory_order_seq_cst);

sparse_entry:
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
 /* defensive NULL guard on doc_token_ids[d].
 * In practice this can't happen because OOM rollback in add_docs_batch
 * resets doc_count, but protects against future refactors. */
 if (!ids) continue;
 int len = corpus->doc_token_counts[d];
 for (int i = 0; i < len; i++) {
 int tid = ids[i];
 if (tid >= 0 && tid < corpus->entry_count) {
 /* per-token UINT32_MAX guard removed —
 * MAX_OCCURRENCES below caps total at 1B, so no single counter can
 * reach UINT32_MAX. The INT32_MAX check is also unreachable. */
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
 char total_buf[32], max_buf[32];
 snprintf(total_buf, sizeof total_buf, "%" PRId64, total);
 snprintf(max_buf, sizeof max_buf, "%" PRId64, MAX_OCCURRENCES);
 fce_log_error("reverse_index.overflow", "kind", "corpus_too_large", "total", total_buf, "max", max_buf, NULL);
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
 if (!ids) continue;
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
 int pass1_failed; /* set by pass1 on double-OOM; finalize skips pass2 */
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
 p->pass1_failed = 1;
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
 .scratch_count = p->worker_count,
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
 bool local_alloc = false;

 if (!pass1_q) {
 /* Fallback: quantize normalized float32 pass1 to int8. */
 if (!p->corpus->enriched_vecs) {
 fce_log(FCE_LOG_WARN, "pass1 failed and no enriched_vecs; skipping pass2", NULL);
 return;
 }
 pass1_q = malloc((size_t)p->corpus->entry_count * FCE_SEM_DIM * sizeof(int8_t));
 if (!pass1_q) {
 fce_log(FCE_LOG_WARN, "OOM during pass2 quantization; using pass1 result", NULL);
 return;
 }
 local_alloc = true;
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

 /* free only locally-allocated pass1_q.
 * In the pure-R4 path (p->_pass1_q != NULL) local_alloc is false —
 * the caller frees it. In the pure-fallback path (enriched_vecs_q == NULL)
 * and in the hybrid path (R4 malloc succeeded but pass1 malloc failed,
 * so we re-quantized here), local_alloc is true and we free here. */
 if (local_alloc) free(pass1_q);
}

/* Worker for parallel doc-vector construction in finalize.
 * R4: always uses int8 enriched_vecs_q (dequantizes on the fly).
 * Fallback: if enriched_vecs_q is NULL, uses float32 enriched_vecs. */
typedef struct {
 fce_sem_vec_t *doc_vectors;
 const fce_sem_vec_t *enriched_vecs; /* float32 fallback (NULL in R4 path) */
 const int8_t *enriched_vecs_q; /* int8 path */
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
 if (!ids) goto skip_doc; /* M2 */
 for (int t = 0; t < ntok; t++) {
 int tid = ids[t];
 if (tid >= 0 && tid < dc->entry_count) {
 if (dc->enriched_vecs_q) {
 /* dequantize int8 on the fly — ~5 tokens × 768 dims
 * per doc, negligible vs the normalize at the end. */
 const int8_t *src = &dc->enriched_vecs_q[(size_t)tid * FCE_SEM_DIM];
 for (int i = 0; i < FCE_SEM_DIM; i++) {
 dv.v[i] += (float)src[i] / FCE_SEM_INT8_MAX;
 }
 } else {
 fce_sem_vec_add_scaled(&dv, &dc->enriched_vecs[tid], 1.0f);
 }
 }
 }
 skip_doc:
 fce_sem_normalize(&dv);
 dc->doc_vectors[d] = dv;
 }
}

/* Worker for parallel doc-vector quantization (float32 → int8).
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
 /* store reciprocal magnitude for multiply-only hot loop. */
 int32_t mag_sq_i32 = fce_dot768_i8(dq, dq);
 float mag_sq = (float)mag_sq_i32;
 dc->doc_vectors_q_inv_mag[d] = (mag_sq > 0.0f) ? 1.0f / sqrtf(mag_sq) : 0.0f;
 }
}

/* Forward declarations for sparse vector functions. */
static bool fce_sparsify_topk(uint16_t *out_idx, int8_t *out_val,
 const int8_t *dense, int dim, int k);

void fce_sem_corpus_set_sparse(fce_sem_corpus_t *corpus, int nnz) {
 if (!corpus || corpus->finalized) return;
 /* clamp nnz to FCE_SEM_DIM — a vector cannot
 * have more non-zero entries than its dimension. Values above DIM are
 * pure 0xFFFF padding (wasted memory) and could be a memory-amplification
 * vector if set_sparse is reachable from untrusted configuration. */
 if (nnz > FCE_SEM_DIM) nnz = FCE_SEM_DIM;
 corpus->sparse_nnz = nnz > 0 ? nnz : 0;
}

int fce_sem_corpus_finalize(fce_sem_corpus_t *corpus) {
 /* block retry after failure — the corpus
 * is in an inconsistent state (doc_vectors freed but doc_token_ids may
 * still be valid). Retrying would crash on NULL deref. */
 if (!corpus || corpus->finalized) return 0;
 if (corpus->finalize_failed) return -1;

 /* the per-vector FP rounding pin in
 * fce_quantize_f32_768_avx2 (simd_dot768.h) already ensures the SIMD
 * path produces the same int8 vectors as the scalar path regardless of
 * the ambient MXCSR. The old assertion/warning here was contradictory:
 * it warned about divergence that the pin already prevents. Removed to
 * avoid misleading future readers. */

 /* Empty corpus — nothing to enrich. Avoids calloc(0) portability issue
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
 corpus->finalize_failed = true;
 return -1;
 }
 fce_sem_src_entry_t *src_entries =
 calloc((size_t)corpus->entry_count, sizeof(fce_sem_src_entry_t));
 if (!src_entries) {
 free_reverse_index(rev);
 corpus->finalize_failed = true;
 return -1;
 }

 /* Allocate int8 enriched_vecs_q directly (3 GB float32 eliminated).
 * If this allocation fails, fall back to float32 enriched_vecs and
 * quantize to int8 after pass2 (the old R1 path). */
 corpus->enriched_vecs_q = malloc((size_t)corpus->entry_count * FCE_SEM_DIM * sizeof(int8_t));
 if (!corpus->enriched_vecs_q) {
 /* Fallback: allocate float32 enriched_vecs; pass1+pass2 write here,
 * then R1 quantizes to int8 and frees the float32 array. */
 corpus->enriched_vecs = calloc((size_t)corpus->entry_count, sizeof(fce_sem_vec_t));
 if (!corpus->enriched_vecs) {
 free(corpus->enriched_vecs_q);
 corpus->enriched_vecs_q = NULL;
 free(src_entries);
 free_reverse_index(rev);
 corpus->finalize_failed = true;
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
 if (params.pass1_failed) {
 free(params._pass1_q);
 /* free/NULL partial enrichment arrays so a retry
 * of fce_sem_corpus_finalize doesn't leak the previous allocation. */
 free(corpus->enriched_vecs_q);
 corpus->enriched_vecs_q = NULL;
 free(corpus->enriched_vecs);
 corpus->enriched_vecs = NULL;
 free(src_entries);
 free_reverse_index(rev);
 fce_log_error("fce_sem_corpus_finalize", "detail", "pass1 double-OOM; finalize failed");
 corpus->finalize_failed = true;
 return -1;
 }
 finalize_pass2(&params);
 free(params._pass1_q); /* R4 path: pass1_q consumed by pass2, no longer needed */

 /* Fallback path: quantize float32 enriched_vecs to int8, then free float32.
 * R4 path: enriched_vecs_q is already populated by pass1+pass2.
 * Review 0007 §1.2: only re-quantize when enriched_vecs_q is NULL (pure
 * float32 fallback). When enriched_vecs_q is already populated (R4 partial
 * OOM), enriched_vecs has stale pass1-only data and must NOT overwrite the
 * correct pass1⊕pass2 blend. */
 if (corpus->enriched_vecs && corpus->entry_count > 0) {
 if (!corpus->enriched_vecs_q) {
 corpus->enriched_vecs_q = malloc((size_t)corpus->entry_count * FCE_SEM_DIM * sizeof(int8_t));
 if (corpus->enriched_vecs_q) {
 for (int i = 0; i < corpus->entry_count; i++) {
 fce_quantize_f32_768(&corpus->enriched_vecs_q[(size_t)i * FCE_SEM_DIM],
 corpus->enriched_vecs[i].v);
 }
 /* only free float32 after int8
 * re-quantization succeeds. If malloc failed, keep enriched_vecs
 * for docvec_build_worker's float32 fallback path (line 2062). */
 free(corpus->enriched_vecs);
 corpus->enriched_vecs = NULL;
 }
 /* else: int8 malloc failed — keep float32 enriched_vecs;
 * docvec_build_worker handles both representations. */
 } else {
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
 /* doc_vectors is required for search — fail loudly rather than
 * producing a silently-degraded corpus that scores everything as 0.0.
 * also free enrichment arrays to make retry safe. */
 free(corpus->enriched_vecs_q);
 corpus->enriched_vecs_q = NULL;
 free(corpus->enriched_vecs);
 corpus->enriched_vecs = NULL;
 free(src_entries);
 free_reverse_index(rev);
 fce_log_error("fce_sem_corpus_finalize", "detail", "doc_vectors calloc failed");
 corpus->finalize_failed = true;
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

 /* Quantize float32 doc vectors to int8 for brute-force bandwidth reduction.
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
 /* partial quantization OOM — free both to
 * prevent NULL deref in search. Return -1 so the caller knows the
 * corpus is not searchable (doc_vectors_q == NULL means every
 * search silently returns empty). */
 free(corpus->doc_vectors_q); corpus->doc_vectors_q = NULL;
 free(corpus->doc_vectors_q_inv_mag); corpus->doc_vectors_q_inv_mag = NULL;
 free(corpus->doc_vectors); corpus->doc_vectors = NULL;
 free(src_entries);
 free_reverse_index(rev);
 corpus->finalize_failed = true;
 return -1;
 }
 }

 /* Free float32 doc_vectors — int8 is now the only representation.
 * Quantization is complete; no codepath reads float32 after this point. */
 free(corpus->doc_vectors);
 corpus->doc_vectors = NULL;

 /* Sparse vector storage: sparsify enriched and doc vectors to top-K NNZ.
 * Must happen after docvec quantization and before freeing dense buffers.
 * M-4: if sparsification OOMs, fail finalize hard
 * rather than silently producing a corpus where every vector is the zero/
 * sentinel value and all scores collapse to 0.5. */
 if (corpus->sparse_nnz > 0 && corpus->enriched_vecs_q && corpus->doc_vectors_q) {
 int nnz = corpus->sparse_nnz;
 int dim = FCE_SEM_DIM;
 bool sparsify_ok = true;
 /* Sparsify enriched vectors. */
 corpus->enriched_sparse_idx = malloc((size_t)corpus->entry_count * nnz * sizeof(uint16_t));
 corpus->enriched_sparse_val = malloc((size_t)corpus->entry_count * nnz);
 if (corpus->enriched_sparse_idx && corpus->enriched_sparse_val) {
 for (int i = 0; i < corpus->entry_count; i++) {
 if (!fce_sparsify_topk(
 &corpus->enriched_sparse_idx[(size_t)i * nnz],
 &corpus->enriched_sparse_val[(size_t)i * nnz],
 &corpus->enriched_vecs_q[(size_t)i * dim], dim, nnz)) {
 sparsify_ok = false;
 break;
 }
 }
 if (sparsify_ok) {
 free(corpus->enriched_vecs_q);
 corpus->enriched_vecs_q = NULL;
 }
 } else {
 sparsify_ok = false;
 }
 /* Sparsify doc vectors. */
 if (sparsify_ok) {
 corpus->doc_sparse_idx = malloc((size_t)corpus->doc_count * nnz * sizeof(uint16_t));
 corpus->doc_sparse_val = malloc((size_t)corpus->doc_count * nnz);
 if (corpus->doc_sparse_idx && corpus->doc_sparse_val) {
 for (int i = 0; i < corpus->doc_count; i++) {
 if (!fce_sparsify_topk(
 &corpus->doc_sparse_idx[(size_t)i * nnz],
 &corpus->doc_sparse_val[(size_t)i * nnz],
 &corpus->doc_vectors_q[(size_t)i * dim], dim, nnz)) {
 sparsify_ok = false;
 break;
 }
 }
 if (sparsify_ok) {
 free(corpus->doc_vectors_q);
 corpus->doc_vectors_q = NULL;
 /* Recompute inv_mag from the SPARSIFIED
 * values so the cosine score is self-consistent over the sparse
 * representation. The previous code kept the full-dense inv_mag,
 * producing partial_dot / full_magnitude — a biased, non-monotonic
 * distortion that penalizes documents with diffuse vectors. */
 if (corpus->doc_vectors_q_inv_mag) {
 for (int i = 0; i < corpus->doc_count; i++) {
 const uint16_t *di = &corpus->doc_sparse_idx[(size_t)i * nnz];
 const int8_t *dv = &corpus->doc_sparse_val[(size_t)i * nnz];
 int32_t mag_sq = 0;
 for (int k = 0; k < nnz; k++) {
 if (di[k] == 0xFFFF) break;
 mag_sq += (int32_t)dv[k] * (int32_t)dv[k];
 }
 float msq = (float)mag_sq;
 corpus->doc_vectors_q_inv_mag[i] = (msq > 0.0f) ? 1.0f / sqrtf(msq) : 0.0f;
 }
 }
 /* ASYMMETRIC COSINE — the
 * document-side magnitude above is recomputed from the
 * sparsified values (top-K dims only), but the query-side
 * magnitude (qvec_q_inv_mag in bruteforce_precomputed /
 * fce_sem_search_query) is computed from all 768 dims.
 * The cosine becomes:
 * partial_dot / (full_query_mag × sparse_doc_mag)
 *
 * WARNING: this is NOT a monotone transform of the true
 * cosine. Documents whose mass is concentrated in dims
 * the query also keeps are scored very differently from
 * documents whose mass lies in dropped dimensions, so
 * two docs can swap rank order versus dense mode.
 * Sparse mode changes RANKING, not just precision.
 * Callers that need faithful rank-order should use dense
 * mode or accept the distortion as a memory/speed trade-off.
 * Restricting the query magnitude to the sparse intersection
 * would lose the fast pre-quantized query path. */
 }
 } else {
 sparsify_ok = false;
 }
 }
 if (!sparsify_ok) {
 fce_log_error("fce_sem_corpus_finalize: sparse vector allocation/sparsification failed (OOM)");
 /* The original goto "fail_free" landed on the
 * success path — the corpus was marked finalized with half-built sparse
 * buffers, causing NULL deref / OOB reads on the next query. Free all
 * partial sparse state, fall through to normal src_entries/rev cleanup,
 * then return -1 without marking finalized. */
 free(corpus->enriched_sparse_idx); corpus->enriched_sparse_idx = NULL;
 free(corpus->enriched_sparse_val); corpus->enriched_sparse_val = NULL;
 free(corpus->doc_sparse_idx); corpus->doc_sparse_idx = NULL;
 free(corpus->doc_sparse_val); corpus->doc_sparse_val = NULL;
 free(corpus->enriched_vecs_q); corpus->enriched_vecs_q = NULL;
 free(corpus->doc_vectors_q); corpus->doc_vectors_q = NULL;
 free(corpus->doc_vectors_q_inv_mag); corpus->doc_vectors_q_inv_mag = NULL;
 corpus->sparse_nnz = 0;
 goto fail_finalize;
 }
 } else if (corpus->sparse_nnz > 0 && !corpus->enriched_vecs_q) {
 /* one-shot warning when sparse was requested
 * but int8 re-quant failed (enriched_vecs_q is NULL). The corpus falls
 * back to dense storage silently — correct but wastes memory. */
 static _Atomic int warned = 0;
 if (atomic_exchange_explicit(&warned, 1, memory_order_relaxed) == 0) {
 /* fce_log_warn treats all variadic args as
 * const char*, so passing int corpus->sparse_nnz is undefined
 * behaviour (read back as pointer → SIGSEGV). Use fce_log_int
 * which correctly formats an int64_t. */
 fce_log_int(FCE_LOG_WARN,
 "sparse requested but int8 quantization failed; "
 "falling back to dense storage",
 "nnz", (int64_t)corpus->sparse_nnz);
 }
 }

 free(src_entries);
 free_reverse_index(rev);

 /* Build inverted index: token_id → [doc_ids containing that token].
 * Used by the fast search path for keyword candidate retrieval.
 * FCE_BRUTE_ONLY (compile-time): skip inverted index build — saves
 * ~1.5 GB memory and ~10-15% finalize time.
 * FCE_SEM_SKIP_INV_INDEX (runtime env): same effect, for benchmarking
 * without recompilation. */
 {
 int build_inv_index = 1;
#ifndef FCE_BRUTE_ONLY
 {
 char skip_buf[8];
 const char *skip = fce_safe_getenv("FCE_SEM_SKIP_INV_INDEX", skip_buf, sizeof(skip_buf), NULL);
 if (skip && skip[0] == '1') build_inv_index = 0;
 }
#else
 build_inv_index = 0;
#endif

 if (build_inv_index && corpus->doc_token_ids && corpus->entry_count > 0) {
 int ntok = corpus->entry_count;
 uint32_t *occ_counts = calloc((size_t)ntok + 1, sizeof(uint32_t));
 if (occ_counts) {
 /* Count occurrences per token across all docs. */
 for (int d = 0; d < corpus->doc_count; d++) {
 int *ids = corpus->doc_token_ids[d];
 int ntok_d = corpus->doc_token_counts[d];
 if (!ids) continue; /* batch resolver
 sets ids=NULL when per-doc malloc fails. */
 for (int i = 0; i < ntok_d; i++) {
 int t = ids[i];
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

 /* Independent total_occ cap for the
 * inverted-index path. The reverse-index build already caps at
 * MAX_OCCURRENCES, but coupling that invariant to build order is
 * fragile. This local check ensures the ~2 GB transient
 * allocation stays bounded regardless of build ordering. */
 if (total_occ > (1U << 30)) {
 /* benign capacity limit — search falls
 * back to brute force. Use WARN not ERROR. */
 fce_log_warn("inverted_index.overflow", "kind", "occurrences_exceed_cap",
 "total", "exceeds limit", NULL);
 } else {

 int *tmp_ids = (int *)malloc((size_t)total_occ * sizeof(int));
 if (tmp_ids) {
 /* Fill doc_ids with duplicates. */
 uint32_t *cursor = malloc(((size_t)ntok + 1) * sizeof(uint32_t));
 if (cursor) {
 for (int t = 0; t <= ntok; t++) cursor[t] = occ_counts[t];
 for (int d = 0; d < corpus->doc_count; d++) {
 int *ids = corpus->doc_token_ids[d];
 if (!ids) continue;
 for (int i = 0; i < corpus->doc_token_counts[d]; i++) {
 int t = ids[i];
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
 /* debug-only assertion that
 * inv_doc_ids is sorted ascending and deduplicated.
 * The binary search in idf_score_fn / tfidf_mass_score_fn
 * assumes this; violation produces silently-missed
 * candidates. Checked once at build time, not per query. */
#ifndef NDEBUG
 for (int t = 0; t < ntok; t++) {
 int s = corpus->inv_offsets[t];
 int e = corpus->inv_offsets[t + 1];
 for (int j = s + 1; j < e; j++) {
 assert(corpus->inv_doc_ids[j] > corpus->inv_doc_ids[j - 1]);
 }
 }
#endif
 } else {
 free(corpus->inv_offsets); corpus->inv_offsets = NULL;
 free(corpus->inv_doc_ids); corpus->inv_doc_ids = NULL;
 }
 free(cursor);
 }
 free(tmp_ids);
 }
 } /* end else (total_occ within cap) */
 free(occ_counts);
 }
 }
 } /* end build_inv_index */

 /* doc_token_ids is dead after finalize — free to reclaim RSS. */
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
#if defined(__GLIBC__)
 malloc_trim(0);
#if defined(M_PURGE)
 mallopt(M_PURGE, 0);
#endif
#elif defined(__APPLE__)
 /* malloc_zone_pressure_relief was added in macOS
 * 10.12 (Sierra). The extern declaration would cause a link failure on
 * older deployment targets. Resolve via dlsym so the call is a safe no-op
 * when the symbol is absent. */
 {
 typedef void (*relief_fn)(malloc_zone_t *, size_t);
 static relief_fn fn = NULL;
 static int resolved = 0;
 if (!resolved) {
 resolved = 1;
 fn = (relief_fn)dlsym(RTLD_DEFAULT, "malloc_zone_pressure_relief");
 }
 if (fn) fn(NULL, 0);
 }
#endif

 /* post-condition assertion — exactly one of the
 * doc-vector representations should be active after finalize. If both
 * are NULL, the corpus is non-functional (all scores will be zero).
 * If both are non-NULL, the scorer will pick one arbitrarily.
 * The enriched-side representations are more permissive (float32
 * enriched_vecs can coexist with int8 doc_vectors_q) because
 * fce_sem_corpus_ri_vec falls through both. */
 /* short-circuit when doc_count==0 — both pointers are
 * legitimately NULL (no documents to allocate vectors for). */
 if (corpus->doc_count > 0) {
 assert((corpus->doc_vectors_q != NULL) != (corpus->doc_sparse_idx != NULL)
 && "finalize post-condition: exactly one of doc_vectors_q / doc_sparse_idx");
 }

 corpus->finalized = true;
 return 0;

fail_finalize:
 /* sparse OOM error exit — free transient buffers
 * and return -1 WITHOUT marking finalized. The corpus is left in a
 * clean (non-sparse, non-finalized) state; the caller can destroy it
 * or retry with sparse disabled. */
 free(src_entries);
 free_reverse_index(rev);
 corpus->finalize_failed = true;
 return -1;
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
 /* dequantize int8 vector into thread-local scratch on demand.
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
 } else if (corpus->enriched_sparse_idx) {
 /* Sparse decompression: reconstruct dense vector from top-K NNZ. */
 static _Thread_local fce_sem_vec_t tl_dequant;
 memset(&tl_dequant, 0, sizeof(tl_dequant));
 int nnz = corpus->sparse_nnz;
 const uint16_t *sp_idx = &corpus->enriched_sparse_idx[(size_t)idx * nnz];
 const int8_t *sp_val = &corpus->enriched_sparse_val[(size_t)idx * nnz];
 const float inv127 = 1.0f / FCE_SEM_INT8_MAX;
 for (int i = 0; i < nnz; i++) {
 if (sp_idx[i] == 0xFFFF) break;
 tl_dequant.v[sp_idx[i]] = inv127 * (float)sp_val[i];
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
 /* dequantize int8 vector into thread-local scratch.
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
 } else if (corpus->enriched_sparse_idx) {
 /* Sparse decompression: reconstruct dense vector from top-K NNZ. */
 static _Thread_local fce_sem_vec_t tl_dequant;
 memset(&tl_dequant, 0, sizeof(tl_dequant));
 int nnz = corpus->sparse_nnz;
 const uint16_t *idx = &corpus->enriched_sparse_idx[(size_t)index * nnz];
 const int8_t *val = &corpus->enriched_sparse_val[(size_t)index * nnz];
 const float inv127 = 1.0f / FCE_SEM_INT8_MAX;
 for (int i = 0; i < nnz; i++) {
 if (idx[i] == 0xFFFF) break;
 tl_dequant.v[idx[i]] = inv127 * (float)val[i];
 }
 *out_vec = &tl_dequant;
 } else {
 *out_vec = corpus->enriched_vecs ? &corpus->enriched_vecs[index] : NULL;
 }
 }
 if (out_idf) *out_idf = 0.0F;
 if (corpus->doc_count > 0) {
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
 free(corpus->doc_vectors);
 free(corpus->doc_vectors_q);
 free(corpus->doc_vectors_q_inv_mag);
 free(corpus->enriched_sparse_idx);
 free(corpus->enriched_sparse_val);
 free(corpus->doc_sparse_idx);
 free(corpus->doc_sparse_val);
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

/* precompute per-corpus-item slash counts so that
 * proximity_internal avoids redundant O(path-length) walks on every scored
 * pair. Caller must free() the returned array. Returns NULL on OOM. */
static int *precompute_corpus_prox(const fce_sem_func_t *corpus, int corpus_size) {
 int *prox = malloc((size_t)corpus_size * sizeof(int));
 if (!prox) return NULL;
 for (int i = 0; i < corpus_size; i++) {
 prox[i] = corpus[i].file_path ? fce_count_slashes(corpus[i].file_path) : 0;
 }
 return prox;
}

/* ── Combined scoring ────────────────────────────────────────────── */

/* use size_t for path component lengths to handle
 * pathological multi-GB paths without overflow. Current callers are bounded
 * by filesystem path limits, but defensiveness costs nothing here. */
static float proximity_internal(const char *path_a, int total_dirs_a,
 const char *path_b, int total_dirs_b) {
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
 size_t len_a = (size_t)(ea - a);
 size_t len_b = (size_t)(eb - b);
 if (len_a == 0 && len_b == 0) break;
 if (len_a != len_b || memcmp(a, b, len_a) != 0) break;
 /* only count a component as shared when BOTH
 * sides are directory components (slash-terminated). Previously,
 * the break required BOTH to be terminal (filename), which inflated
 * proximity when one side had a filename matching the other's
 * directory component. */
 bool is_dir_a = (*ea == '/');
 bool is_dir_b = (*eb == '/');
 if (!is_dir_a || !is_dir_b) break;
 shared_dirs++;
 a = *ea ? ea + 1 : ea;
 b = *eb ? eb + 1 : eb;
 }
 int max_dirs = total_dirs_a > total_dirs_b ? total_dirs_a : total_dirs_b;
 if (max_dirs == 0) {
 return FCE_SEM_UNIT_POS;
 }
 /* The +1 is intentional smoothing — it caps the ratio at
 * max_dirs/(max_dirs+1) < 1.0 so that same-directory pairs never
 * reach the full FCE_SEM_PROX_MAX_BOOST. With A1's directory-only
 * counting, shared_dirs ≤ max_dirs, so this invariant holds for all
 * path pairs including identical paths. */
 float ratio = (float)shared_dirs / (float)(max_dirs + 1);
 return FCE_SEM_UNIT_POS + (ratio * FCE_SEM_PROX_MAX_BOOST);
}

/* one-shot warning callback for backslash paths. */
static fce_once_t g_backslash_warn_once = FCE_ONCE_INIT;
static void do_backslash_warn(void) {
 fce_log_warn("fce_sem_proximity.backslash_paths",
 "note", "backslash-separated paths yield flat proximity (1.0); "
 "normalize to forward slash for proximity weighting");
}

float fce_sem_proximity(const char *path_a, const char *path_b) {
 if (!path_a || !path_b) return FCE_SEM_UNIT_POS;
 if (strchr(path_a, '\\') || strchr(path_b, '\\')) {
 fce_once(&g_backslash_warn_once, do_backslash_warn);
 }
 int total_dirs_a = fce_count_slashes(path_a);
 int total_dirs_b = fce_count_slashes(path_b);
 return proximity_internal(path_a, total_dirs_a, path_b, total_dirs_b);
}

/* sentinel for "compute magnitude inline."
 * Pre-NaN-sentinel rationale: the previous API used `isnan(mag_a_sq)` to
 * mean "caller didn't precompute." A caller could legitimately pass NaN
 * by accident (e.g. `NaN/df` arithmetic on a corpus with df=0). Negative
 * magnitude is impossible for any real sum-of-squares, so `< 0` is a
 * safe out-of-band signal. */
#define FCE_MAG_COMPUTE_INLINE (-1.0F)

/* Sparse cosine over two pre-sorted (index, weight) vectors.
 * REQUIRES: tfidf_indices arrays must be sorted ascending.
 * The two-pointer merge silently produces wrong results if indices are unsorted.
 * mag_a_sq is the precomputed sum-of-squares of a's tfidf_weights; pass
 * FCE_MAG_COMPUTE_INLINE to compute it here (useful for single-call paths
 * where the same 'a' is scored against only one 'b'). Avoids the previous
 * `isnan` sentinel, which a caller could collide with by accident.
 * Returns 0 when either side is empty or the magnitude product is below epsilon. */
static float fce_sparse_tfidf_cosine(const fce_sem_func_t *a, const fce_sem_func_t *b,
 float mag_a_sq) {
 if (a->tfidf_len <= 0 || b->tfidf_indices == NULL || b->tfidf_weights == NULL || b->tfidf_len <= 0) {
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
 float ma = mag_a_sq;
 if (ma < 0.0F) {
 ma = 0.0F;
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

/* fce_sparse_tfidf_cosine_flat removed — positional indices (0,1,2,…) are not
 * global vocab IDs, making the flat TF-IDF cosine meaningless. The flat scoring
 * path (fce_score_flat) now uses RI-only. */

/* Internal combined scoring with precomputed proximity (P3) and precomputed
 * query-side magnitudes (P2). Avoids redundant per-element FLOPs.
 * Pass FCE_MAG_COMPUTE_INLINE for any magnitude to compute it inline
 * (for single-call paths).
 * when any magnitude is FCE_MAG_COMPUTE_INLINE, the
 * corresponding vector is read from `a` — callers MUST pass the query
 * descriptor as `a` and the corpus descriptor as `b`. Both the public
 * single-pair entry (fce_sem_combined_score) and the parallel worker
 * (sc_worker) honour this contract. A future caller that passes the
 * corpus item as `a` would silently mis-normalize cosines (wrong scores,
 * no crash).
 * B5: the query-is-a contract is enforced by convention
 * only. No runtime assertion is possible because magnitudes can legitimately
 * be either FCE_MAG_COMPUTE_INLINE or precomputed depending on the call
 * site. All current callers (sc_score, sc_worker, fce_sem_combined_score)
 * honour this contract. */
static float score_combined_internal(const fce_sem_func_t *a, const fce_sem_func_t *b,
 const fce_sem_config_t *cfg, float prox,
 float q_tfidf_mag_sq, float q_ri_mag_sq,
 float q_api_mag_sq, float q_type_mag_sq,
 float q_deco_mag_sq, float q_sp_mag_sq) {
 if (!a || !b || !cfg) {
 return 0.0F;
 }
 /* /C3: validate tfidf_weights invariant at the internal
 * scoring entry point so all public scorers (fce_sem_combined_score,
 * fce_sem_search, fce_sem_rank) are protected. A direct C caller that
 * builds a descriptor with indices but no weights would otherwise crash
 * in the TF-IDF magnitude loop. */
 if (a->tfidf_len > 0 && (a->tfidf_indices == NULL || a->tfidf_weights == NULL)) {
 return 0.0F;
 }
 if (b->tfidf_len > 0 && (b->tfidf_indices == NULL || b->tfidf_weights == NULL)) {
 return 0.0F;
 }

 float score = 0.0F;

 /* TF-IDF cosine — use precomputed query magnitude if provided. */
 float tfidf_mag = q_tfidf_mag_sq;
 if (tfidf_mag < 0.0F && a->tfidf_len > 0) {
 /* FCE_MAG_COMPUTE_INLINE sentinel — compute the magnitude here. */
 float m = 0.0F;
 for (int i = 0; i < a->tfidf_len; i++) {
 m += a->tfidf_weights[i] * a->tfidf_weights[i];
 }
 tfidf_mag = m;
 }
 score += cfg->w_tfidf * fce_sparse_tfidf_cosine(a, b, tfidf_mag);

 /* RI cosine — use precomputed query magnitude. */
 if (q_ri_mag_sq < 0.0F) {
 q_ri_mag_sq = 0.0F;
 for (int i = 0; i < FCE_SEM_DIM; i++) {
 q_ri_mag_sq += a->ri_vec.v[i] * a->ri_vec.v[i];
 }
 }
 score += cfg->w_ri * fce_sem_cosine_aliased_with_mag(a->ri_vec.v, b->ri_vec.v, q_ri_mag_sq);

 /* API cosine — skip when query signal is zero (P1: avoids 768 MACs). */
 if (q_api_mag_sq < 0.0F) {
 q_api_mag_sq = 0.0F;
 for (int i = 0; i < FCE_SEM_DIM; i++) {
 q_api_mag_sq += a->api_vec.v[i] * a->api_vec.v[i];
 }
 }
 if (q_api_mag_sq > FCE_SEM_DENOM_EPS)
 score += cfg->w_api * fce_sem_cosine_aliased_with_mag(a->api_vec.v, b->api_vec.v, q_api_mag_sq);

 /* Type cosine — skip when query signal is zero (P1). */
 if (q_type_mag_sq < 0.0F) {
 q_type_mag_sq = 0.0F;
 for (int i = 0; i < FCE_SEM_DIM; i++) {
 q_type_mag_sq += a->type_vec.v[i] * a->type_vec.v[i];
 }
 }
 if (q_type_mag_sq > FCE_SEM_DENOM_EPS)
 score += cfg->w_type * fce_sem_cosine_aliased_with_mag(a->type_vec.v, b->type_vec.v, q_type_mag_sq);

 /* Decorator cosine — skip when query signal is zero (P1). */
 if (q_deco_mag_sq < 0.0F) {
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
 if (sp_mag < 0.0F) {
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
 if (!isfinite(score)) {
 /* surface a one-shot warning instead of
 * silently returning 0. Repeated NaN/Inf inputs would otherwise
 * produce repeated silent zero scores with no diagnostic. */
 if (atomic_exchange_explicit(&g_nonfinite_warned, 1, memory_order_acq_rel) == 0) {
 fce_log_warn("combined_score.nonfinite_input", NULL);
 }
 return 0.0f;
 }
 if (score > FCE_SEM_UNIT_POS) {
 score = FCE_SEM_UNIT_POS;
 }
 if (score < 0.0F) {
 score = 0.0F;
 }

 return score;
}

/* cheap ascending-sort check for the caller-supplied
 * tfidf_indices arrays. Only active in debug builds; release builds skip the
 * cost. The contract is documented on fce_sem_func_t in semantic.h. */
#ifdef NDEBUG
#define FCE_ASSERT_TFIDF_SORTED(f) ((void)0)
#else
static inline void fce_assert_tfidf_sorted(const fce_sem_func_t *f, const char *where) {
 if (!f || f->tfidf_len <= 1) return;
 for (int i = 1; i < f->tfidf_len; i++) {
 /* use <= to reject duplicates — the two-pointer
 * merge in fce_sparse_tfidf_cosine desynchronizes on equal indices,
 * producing an incorrect dot product. */
 if (f->tfidf_indices[i] <= f->tfidf_indices[i-1]) {
 /* use fce_log instead of fprintf(stderr)
 * for consistency with the rest of the logging infrastructure. */
 fce_log_error("unsorted tfidf_indices", "function", where, "slot", "i");
 /* Don't abort — a corrupt ranking result is worse than a crash
 * for users who get a clear diagnostic in the log. */
 return;
 }
 }
}
#define FCE_ASSERT_TFIDF_SORTED(f) fce_assert_tfidf_sorted((f), __func__)
#endif

float fce_sem_combined_score(const fce_sem_func_t *a, const fce_sem_func_t *b,
 const fce_sem_config_t *cfg) {
 FCE_ASSERT_TFIDF_SORTED(a);
 FCE_ASSERT_TFIDF_SORTED(b);
 float prox = fce_sem_proximity(a ? a->file_path : NULL, b ? b->file_path : NULL);
 return score_combined_internal(a, b, cfg, prox,
 FCE_MAG_COMPUTE_INLINE,
 FCE_MAG_COMPUTE_INLINE,
 FCE_MAG_COMPUTE_INLINE,
 FCE_MAG_COMPUTE_INLINE,
 FCE_MAG_COMPUTE_INLINE,
 FCE_MAG_COMPUTE_INLINE);
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

/* qsort comparator: descending by score, ascending by index for ties
 *: ensures deterministic output across runs and
 * thread counts. */
static int fce_ranked_cmp_desc(const void *a, const void *b) {
 float sa = ((const fce_sem_ranked_t *)a)->score;
 float sb = ((const fce_sem_ranked_t *)b)->score;
 if (sb > sa) return 1;
 if (sb < sa) return -1;
 uint32_t ia = ((const fce_sem_ranked_t *)a)->index;
 uint32_t ib = ((const fce_sem_ranked_t *)b)->index;
 return (ia > ib) - (ia < ib);
}

/* ── Quickselect for top-k candidate selection (P2) ────────────── */

/* Descending comparator for cand_t. */
typedef struct { int doc; float score; } cand_t;

static int cand_cmp_desc(const void *a, const void *b) {
 float sa = ((const cand_t *)a)->score;
 float sb = ((const cand_t *)b)->score;
 if (sb > sa) return 1;
 if (sb < sa) return -1;
 int da = ((const cand_t *)a)->doc;
 int db = ((const cand_t *)b)->doc;
 return (da > db) - (da < db);
}

/* lightweight xorshift32 for pivot randomization.
 * Avoids stdlib rand() which holds a global lock. State is per-call (stack),
 * so no thread-safety concerns. */
static inline uint32_t qs_xorshift32(uint32_t *state) {
 uint32_t x = *state;
 x ^= x << 13;
 x ^= x >> 17;
 x ^= x << 5;
 *state = x;
 return x;
}

/* per-thread monotonic counter for pivot
 * randomization. Prevents adversarial inputs from triggering O(n²)
 * quickselect by ensuring different queries produce different pivot
 * sequences even when called from the same indices. */
static _Thread_local uint32_t tls_qselect_counter = 0;

/* Partial quickselect: rearrange scored[0..n-1] so that scored[0..k-1]
 * are the top-k by score (descending order, unsorted within partition).
 * O(n) average vs O(n·k) selection sort.
 * Depth guard: falls back to qsort when recursion depth exceeds 2*log2(n)
 * to prevent O(n²) worst case on adversarial inputs.
 * C4: pivot is randomized (median-of-three + random swap) to eliminate
 * adversarial-input O(n²) paths even before the depth guard triggers.
 * C6: seed incorporates a per-thread counter to vary pivot sequences
 * across calls, defeating adversarial input crafting. */
static void quickselect_topk(cand_t *scored, int n, int k, int depth) {
 int lo = 0, hi = n - 1;
 /* seed xorshift with lo/hi/k AND a per-thread monotonic counter
 * so different queries produce different pivot sequences.
 * L4: the `| 1u` is load-bearing — xorshift32 has a
 * fixed point at 0 (all-zero state never leaves 0). Setting bit 0
 * guarantees a non-zero seed regardless of the arithmetic above. */
 uint32_t rng_state = (uint32_t)(lo + 31 * k + 7 * hi + tls_qselect_counter++) | 1u;
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
 /* swap median-of-three pivot with a random position in [lo, hi]
 * to break adversarial patterns that exploit deterministic pivot. */
 if (hi - lo > 2) {
 int range = hi - lo + 1;
 int r = lo + (int)(qs_xorshift32(&rng_state) % (uint32_t)range);
 if (r != hi) {
 cand_t tmp = scored[r]; scored[r] = scored[hi]; scored[hi] = tmp;
 }
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
 if (!isfinite(s)) continue;
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

/* Clamp int8 cosine-derived score to [0,1].
 * Cauchy-Schwarz guarantees |cosine| <= 1 in exact arithmetic, but float
 * rounding of reciprocal magnitudes can produce cosine slightly outside [-1,1],
 * yielding scores outside [0,1]. The struct-based scorers already clamp;
 * these int8 fast paths did not. */
static inline float fce_clamp_unit(float s) {
 if (s < 0.0f) return 0.0f;
 if (s > 1.0f) return 1.0f;
 return s;
}

/* ── Sparse vector operations ──────────────────────────────────── */

/* Sparsify a dense int8 vector to top-K non-zero entries.
 * Uses a min-heap of size K for O(dim × log(K)) performance.
 * Writes sorted (index, value) pairs to out_idx/out_val.
 * Returns true on success, false on OOM (output is zeroed/sentinel-filled). */
static bool fce_sparsify_topk(uint16_t *out_idx, int8_t *out_val,
 const int8_t *dense, int dim, int k) {
 /* Min-heap: (magnitude, index) — smallest magnitude at root. */
 typedef struct { int16_t mag; int16_t idx; } heap_entry_t;
 heap_entry_t *heap = (heap_entry_t *)malloc((size_t)k * sizeof(heap_entry_t));
 if (!heap) { for (int i = 0; i < k; i++) { out_idx[i] = 0xFFFF; out_val[i] = 0; } return false; }
 int hsize = 0;

 #define HEAP_SIFTUP(h, n) do { \
 int _i = (n); \
 while (_i > 0) { \
 int _p = (_i - 1) / 2; \
 if (h[_p].mag <= h[_i].mag) break; \
 heap_entry_t _t = h[_p]; h[_p] = h[_i]; h[_i] = _t; \
 _i = _p; \
 } \
 } while(0)
 #define HEAP_SIFTDOWN(h, n) do { \
 int _n = (n), _i = 0; \
 while (1) { \
 int _l = 2*_i+1, _r = 2*_i+2, _s = _i; \
 if (_l < _n && h[_l].mag < h[_s].mag) _s = _l; \
 if (_r < _n && h[_r].mag < h[_s].mag) _s = _r; \
 if (_s == _i) break; \
 heap_entry_t _t = h[_i]; h[_i] = h[_s]; h[_s] = _t; \
 _i = _s; \
 } \
 } while(0)

 for (int d = 0; d < dim; d++) {
 int mag = dense[d] >= 0 ? dense[d] : -dense[d];
 if (mag == 0) continue;
 if (hsize < k) {
 heap[hsize] = (heap_entry_t){(int16_t)mag, (int16_t)d};
 hsize++;
 HEAP_SIFTUP(heap, hsize - 1);
 } else if (mag > heap[0].mag) {
 heap[0] = (heap_entry_t){(int16_t)mag, (int16_t)d};
 HEAP_SIFTDOWN(heap, k);
 }
 }

 /* Extract from heap and sort by index for merge-join. */
 for (int i = 0; i < k; i++) {
 if (i < hsize) {
 out_idx[i] = (uint16_t)heap[i].idx;
 out_val[i] = dense[heap[i].idx];
 } else {
 out_idx[i] = 0xFFFF;
 out_val[i] = 0;
 }
 }
 /* Sort by index (insertion sort — k is small). */
 for (int i = 1; i < k; i++) {
 uint16_t ki = out_idx[i]; int8_t vi = out_val[i];
 int j = i - 1;
 while (j >= 0 && out_idx[j] > ki) {
 out_idx[j+1] = out_idx[j]; out_val[j+1] = out_val[j];
 j--;
 }
 out_idx[j+1] = ki; out_val[j+1] = vi;
 }
 #undef HEAP_SIFTUP
 #undef HEAP_SIFTDOWN
 free(heap);
 return true;
}

typedef struct {
 const fce_sem_corpus_t *corpus;
 const int8_t *qvec_q; /* pre-quantized query vector for int8 brute-force */
 float qvec_q_inv_mag; /* reciprocal L2 magnitude of quantized query vector */
} sq_sctx_t;

static float sq_score(int i, void *ctx) {
 sq_sctx_t *c = ctx;
 float dot;
 if (c->corpus->sparse_nnz > 0 && c->corpus->doc_sparse_idx) {
 /* Dense×sparse dot product. */
 int nnz = c->corpus->sparse_nnz;
 const uint16_t *di = &c->corpus->doc_sparse_idx[(size_t)i * nnz];
 const int8_t *dv = &c->corpus->doc_sparse_val[(size_t)i * nnz];
 int32_t acc = 0;
 for (int k = 0; k < nnz; k++) {
 if (di[k] == 0xFFFF) break;
 acc += (int32_t)c->qvec_q[di[k]] * (int32_t)dv[k];
 }
 dot = (float)acc;
 } else {
 /* Dense path. */
 const int8_t *dq = c->corpus->doc_vectors_q;
 dot = (float)fce_dot768_i8(c->qvec_q, dq + (size_t)i * FCE_SEM_DIM);
 }
 float inv_d_mag = c->corpus->doc_vectors_q_inv_mag ? c->corpus->doc_vectors_q_inv_mag[i] : 0.0f;
 float cosine = (inv_d_mag > 0.0f && c->qvec_q_inv_mag > 0.0f) ? dot * c->qvec_q_inv_mag * inv_d_mag : 0.0f;
 return fce_clamp_unit((cosine + FCE_SEM_UNIT_POS) * 0.5f);
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
 const int *corpus_prox; /* precomputed corpus slash counts (L3) */
} sc_sctx_t;

static float sc_score(int i, void *ctx) {
 sc_sctx_t *c = ctx;
 float prox = proximity_internal(c->query->file_path, c->q_prox,
 c->corpus[i].file_path,
 c->corpus_prox ? c->corpus_prox[i] : fce_count_slashes(c->corpus[i].file_path));
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
 const int *corpus_prox; /* precomputed corpus slash counts (L3) */
} fce_ss_sctx_t;

static float fce_ss_score(int i, void *ctx) {
 fce_ss_sctx_t *c = ctx;
 float prox = proximity_internal(c->query->file_path, c->q_prox,
 c->corpus[i].file_path,
 c->corpus_prox ? c->corpus_prox[i] : fce_count_slashes(c->corpus[i].file_path));
 return fce_score_simple_internal(c->query, &c->corpus[i], prox,
 c->q_ri_mag_sq);
}

typedef struct {
 const float *all_tfidf_weights;
 const int *all_tfidf_indices;
 const int *tfidf_lens;
 const float *all_ri_vecs;
 const char **file_paths;
 int max_tokens;
 const int *q_tfidf_indices;
 const float *q_tfidf_weights;
 int q_tfidf_len;
 const float *q_ri_vec;
 float q_ri_mag;
} sf_sctx_t;

static float sf_score(int i, void *ctx) {
 sf_sctx_t *c = ctx;
 const int *c_idx = c->all_tfidf_indices + (size_t)i * c->max_tokens;
 const float *c_w = c->all_tfidf_weights + (size_t)i * c->max_tokens;
 int c_len = c->tfidf_lens ? c->tfidf_lens[i] : 0;
 if (c_len > c->max_tokens) c_len = c->max_tokens; /* defensive clamp (H1) */
 const float *c_ri = c->all_ri_vecs + (size_t)i * FCE_SEM_DIM;
 const char *c_path = c->file_paths ? c->file_paths[i] : "";
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
 /* assert worker_counts[w] <= top_k.
 * The heap logic guarantees each worker produces at most top_k results,
 * but a bug in the heap could cause OOB reads on worker_results. */
 assert(worker_counts[w] <= (int)top_k);
 for (int i = 0; i < worker_counts[w]; i++) {
 fce_sem_ranked_t r = worker_results[(size_t)w * top_k + i];
 if (!isfinite(r.score)) continue;
 if (k < top_k) {
 results_out[k] = r;
 k++;
 if (k == top_k) {
 for (int h = (int)top_k / 2 - 1; h >= 0; h--) {
 heap_siftdown(results_out, h, (int)top_k);
 }
 }
 } else if (r.score > results_out[0].score ||
 (r.score == results_out[0].score && r.index < results_out[0].index)) {
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
 const int *all_tfidf_indices;
 const int *tfidf_lens;
 const float *all_ri_vecs;
 const char **file_paths;
 int max_tokens;
 int corpus_size;
 const int *q_tfidf_indices;
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
 const int *c_idx = w->all_tfidf_indices + (size_t)f * w->max_tokens;
 const float *c_w = w->all_tfidf_weights + (size_t)f * w->max_tokens;
 int c_len = w->tfidf_lens ? w->tfidf_lens[f] : 0;
 if (c_len > w->max_tokens) c_len = w->max_tokens; /* defensive clamp (H1) */
 const float *c_ri = w->all_ri_vecs + (size_t)f * FCE_SEM_DIM;
 const char *c_path = w->file_paths ? w->file_paths[f] : "";
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
 int corpus_size;
 float min_score;
 int q_prox; /* precomputed proximity for query (P3) — fce_count_slashes returns int */
 /* Precomputed query-side RI magnitude (P1) — avoids ~768 FLOPs per corpus item. */
 float q_ri_mag_sq;
 const int *corpus_prox; /* precomputed corpus slash counts (L3) */
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
 if (i >= w->corpus_size) break;
 float prox = proximity_internal(w->query->file_path, w->q_prox,
 w->corpus[i].file_path,
 w->corpus_prox ? w->corpus_prox[i] : fce_count_slashes(w->corpus[i].file_path));
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
 } else if (s > local[0].score ||
 (s == local[0].score && (uint32_t)i < local[0].index)) {
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
 int corpus_size;
 float min_score;
 int q_prox; /* precomputed proximity for query (P3) — fce_count_slashes returns int */
 /* Precomputed query-side magnitudes (P1) — avoids ~3072 FLOPs per corpus item. */
 float q_tfidf_mag_sq;
 float q_ri_mag_sq;
 float q_api_mag_sq;
 float q_type_mag_sq;
 float q_deco_mag_sq;
 float q_sp_mag_sq;
 const fce_sem_config_t *cfg;
 const int *corpus_prox; /* precomputed corpus slash counts (L3) */
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
 if (i >= w->corpus_size) break;
 float prox = proximity_internal(w->query->file_path, w->q_prox,
 w->corpus[i].file_path,
 w->corpus_prox ? w->corpus_prox[i] : fce_count_slashes(w->corpus[i].file_path));
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
 } else if (s > local[0].score ||
 (s == local[0].score && (uint32_t)i < local[0].index)) {
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

/* candidate-heap size on stack.
 * FCE_CANDIDATE_HEAP_SZ is the upper bound on heap[] array size — set to
 * 4096 for headroom. The runtime cap is FCE_CANDIDATE_CAP (2048), so the
 * stack array is twice as large as needed; this trades a bit of stack
 * (sizeof(fce_sem_ranked_t) * 4096 = 65 536 bytes = 64 KB on most
 * platforms with fce_sem_ranked_t = {int, float}) for the simplicity of
 * a static bound. Do not raise FCE_CANDIDATE_CAP past FCE_CANDIDATE_HEAP_SZ
 * without checking stack depth at all call sites (collect_candidates,
 * fce_sem_search, fce_sem_simple_search, fce_sem_combined_score). */
#define FCE_CANDIDATE_HEAP_SZ 4096
_Static_assert(FCE_CANDIDATE_CAP <= FCE_CANDIDATE_HEAP_SZ,
 "FCE_CANDIDATE_CAP exceeds FCE_CANDIDATE_HEAP_SZ");

/* ── Unified candidate retrieval (Q1) ────────────────────────── */

/* Score callback for candidate retrieval. Returns a score for (corpus, doc_id). */
typedef float (*cand_score_fn)(const fce_sem_corpus_t *corpus, int doc_id, void *ctx);

/* IDF-sum scorer context: sum of IDF weights for matching query tokens.
 * C-1: defensive `tid` bounds check at the top of the
 * inner loop. Today every public entry point filters NULL/negative
 * (q_tok_ids are built from ptr_to_token_idx, which is non-negative), but
 * this guard costs nothing and protects against future extensions that
 * might construct q_tok_ids from untrusted input. */
typedef struct { const int *q_toks; int q_ntok; const float *q_idf; } idf_score_ctx_t;
static float idf_score_fn(const fce_sem_corpus_t *corpus, int doc_id, void *vctx) {
 /* defensive NULL guard — collect_candidates checks
 * inv_offsets before dispatch, but these functions are file-static and a
 * future refactor could call them directly. Matching the per-tid bounds
 * checks they already carry. */
 if (!corpus->inv_offsets || !corpus->inv_doc_ids) return 0.0f;
 idf_score_ctx_t *c = vctx;
 float score = 0.0f;
 for (int t = 0; t < c->q_ntok; t++) {
 int tid = c->q_toks[t];
 if (tid < 0 || tid >= corpus->entry_count) continue;
 int start = corpus->inv_offsets[tid];
 int end = corpus->inv_offsets[tid + 1];
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
/* intentionally simplified TF-IDF overlap
 * heuristic — NOT a proper cosine. dot and doc_mag accumulate the same value
 * (qidf²), so the function returns sqrt(matched_mass / total_mass). This is
 * adequate for candidate retrieval (the rerank stage uses proper RI cosine).
 * Do NOT "fix" this to compute a real cosine — it would change candidate
 * quality and is unnecessary given the rerank step. */
static float tfidf_mass_score_fn(const fce_sem_corpus_t *corpus, int doc_id, void *vctx) {
 /* defensive NULL guard — see idf_score_fn above. */
 if (!corpus->inv_offsets || !corpus->inv_doc_ids) return 0.0f;
 tfidf_mass_ctx_t *c = vctx;
 float dot = 0.0f;
 float doc_mag = 0.0f;
 for (int t = 0; t < c->q_ntok; t++) {
 int tid = c->q_toks[t];
 if (tid < 0 || tid >= corpus->entry_count) continue;
 float qidf = c->q_idf[t];
 int start = corpus->inv_offsets[tid];
 int end = corpus->inv_offsets[tid + 1];
 int lo = start, hi = end;
 while (lo < hi) {
 int mid = lo + (hi - lo) / 2;
 if (corpus->inv_doc_ids[mid] < doc_id) lo = mid + 1;
 else hi = mid;
 }
 if (lo < end && corpus->inv_doc_ids[lo] == doc_id) {
 dot += qidf * qidf;
 doc_mag += qidf * qidf;
 }
 }
 float denom = sqrtf(c->q_mag) * sqrtf(doc_mag);
 return (denom > FCE_SEM_DENOM_EPS) ? (dot / denom) : 0.0f;
}

/* Thread-local scratch arena for collect_candidates — avoids per-query
 * calloc/malloc/free of bitmap (~3 KB), raw candidates, and scored array.
 * Buffers grow monotonically and are never freed (thread-local cleanup at exit).
 * C3: known limitation — in JVM thread pools,
 * threads are long-lived and reused across queries, so buffers grow to the
 * high-water mark and stay there. The cost is bounded by peak_corpus_size/8 ×
 * thread_pool_size (e.g., ~500 KB for a 4-thread pool with 1M docs). */
typedef struct {
 uint64_t *seen;
 int seen_nwords;
 int *raw;
 int raw_cap;
 cand_t *scored;
 int scored_cap;
} cand_scratch_t;

/* Destructor for thread-local scratch buffers. Registered via pthread_once.
 * L-2: On Windows, this entire block is compiled out
 * (#ifndef _WIN32). Worker threads leak their seen/raw/scored buffers at
 * thread exit. Acceptable for a long-lived JVM process where threads are
 * pooled; documented for completeness. A Windows fix would use
 * FlsAlloc/FlsSetValue with a destructor callback.
 *
 * FIX: the previous design stored a pointer to a
 * `static _Thread_local cand_scratch_t` in the pthread key. On macOS the
 * C runtime reclaims _Thread_local storage BEFORE pthread key destructors
 * run, so the destructor received a dangling pointer whose inner
 * seen/raw/scored fields were already garbage → libmalloc abort
 * (SIGTRAP / ___BUG_IN_CLIENT_OF_LIBMALLOC_POINTER_BEING_FREED_WAS_NOT_ALLOCATED).
 *
 * Fix: allocate the cand_scratch_t itself on the heap and store it ONLY
 * in the pthread key. The destructor frees the inner buffers AND the
 * struct. No _Thread_local is used. */
#ifndef _WIN32
static pthread_key_t tls_cand_key;
static void tls_cand_scratch_destructor(void *ptr) {
 cand_scratch_t *sc = (cand_scratch_t *)ptr;
 if (sc) {
 free(sc->seen);
 free(sc->raw);
 free(sc->scored);
 free(sc);
 }
}
static void tls_cand_key_init(void) {
 /* pthread_key_create can fail if the system
 * runs out of TLS keys. On failure tls_cand_key is 0 (zero-init) and
 * subsequent pthread_getspecific/setspecific on slot 0 is UB. The
 * documented maximum is PTHREAD_KEYS_MAX (typically 128-512); in practice
 * a JVM already uses ~50 keys. We cannot recover gracefully, so log and
 * leave tls_cand_key as 0 — tls_cand_scratch_get will return NULL which
 * callers already handle (collect_candidates checks for NULL). */
 if (pthread_key_create(&tls_cand_key, tls_cand_scratch_destructor) != 0) {
 fce_log_warn("tls_cand_key_init",
 "pthread_key_create failed (TLS key exhaustion)");
 }
}
static cand_scratch_t *tls_cand_scratch_get(void) {
 static pthread_once_t once = PTHREAD_ONCE_INIT;
 pthread_once(&once, tls_cand_key_init);
 if (!tls_cand_key) return NULL; /* key creation failed */
 cand_scratch_t *sc = (cand_scratch_t *)pthread_getspecific(tls_cand_key);
 if (!sc) {
 sc = (cand_scratch_t *)calloc(1, sizeof(cand_scratch_t));
 pthread_setspecific(tls_cand_key, sc);
 }
 return sc;
}
#endif
#ifdef _WIN32
static _Thread_local cand_scratch_t tls_cand_scratch_win;
static cand_scratch_t *tls_cand_scratch_get(void) {
 return &tls_cand_scratch_win;
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
 cand_scratch_t *sc = tls_cand_scratch_get();
 if (!sc) return 0; /* TLS key creation failed */

 /* Ensure seen bitmap is large enough.
 * INVARIANT: `seen_nwords` is monotonically
 * non-decreasing within a thread (this is the only call site that
 * grows it). On a smaller subsequent corpus we memset the first
 * `nwords` words and never read beyond — so leaving the trailing
 * words untouched is safe. If you ever add a shrink path, call
 * `calloc` instead of `realloc` (realloc copies stale bits). */
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
 int tid = q_toks[t];
 /* bounds check mirrors idf_score_fn / tfidf_mass_score_fn
 * — self-defending against out-of-range token ids. */
 if (tid < 0 || tid >= corpus->entry_count) continue;
 approx += corpus->entries[tid].doc_freq;
 }
 if (approx > (long)max_candidates * 4) approx = (long)max_candidates * 4;
 if (approx < 256) approx = 256;

 /* Ensure raw buffer is large enough.
 * C13: use a more conservative initial
 * estimate (max of approx and 2× max_candidates) to reduce realloc
 * calls on corpora with high token overlap. The 2× factor provides
 * headroom for the common case where initial doc_freq sums underestimate
 * the actual unique candidate count. */
 int raw_need = (int)approx;
 if (raw_need < max_candidates * 2) raw_need = max_candidates * 2;
 if (raw_need > sc->raw_cap) {
 free(sc->raw);
 sc->raw = (int *)malloc((size_t)raw_need * sizeof(int));
 if (!sc->raw) { sc->raw_cap = 0; return 0; }
 sc->raw_cap = raw_need;
 }
 raw_need = sc->raw_cap; /* use actual capacity so mid-loop grows don't shrink */
 int nraw = 0;

 for (int t = 0; t < q_ntok; t++) {
 int tid = q_toks[t];
 /* bounds check — out-of-range tokens are skipped
 * to match the guard in idf_score_fn / tfidf_mass_score_fn. */
 if (tid < 0 || tid >= corpus->entry_count) continue;
 int start = corpus->inv_offsets[tid];
 int end = corpus->inv_offsets[tid + 1];
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
 } else {
 /* realloc failure — abort
 * collection so the caller falls back to brute-force. */
 return 0;
 }
 }
 sc->raw[nraw++] = d;
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
 int tid = q_toks[t];
 /* bounds-check token id before indexing entries[].
 * Consistent with collect_candidates / idf_score_fn / tfidf_mass_score_fn.
 * Guards against FCE_NOT_FOUND (-1) or out-of-range ids from future callers. */
 if (tid < 0 || tid >= corpus->entry_count) { q_idf[t] = 0.0f; continue; }
 int df = corpus->entries[tid].doc_freq;
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
 /* heap allocation instead of VLA to match
 * keyword_candidates and avoid stack pressure on JNI threads. */
 float *q_idf = (float *)malloc((size_t)q_ntok * sizeof(float));
 if (!q_idf) return 0;
 float q_mag = 0.0f;
 for (int t = 0; t < q_ntok; t++) {
 int tid = q_toks[t];
 /* bounds-check token id before indexing entries[].
 * Consistent with collect_candidates / idf_score_fn / tfidf_mass_score_fn. */
 if (tid < 0 || tid >= corpus->entry_count) { q_idf[t] = 0.0f; continue; }
 int df = corpus->entries[tid].doc_freq;
 q_idf[t] = (df > 0) ? logf((float)corpus->doc_count / (float)df) : 0.0f;
 q_mag += q_idf[t] * q_idf[t];
 }
 tfidf_mass_ctx_t ctx = { .q_idf = q_idf, .q_toks = q_toks, .q_ntok = q_ntok, .q_mag = q_mag };
 int n = collect_candidates(corpus, q_toks, q_ntok, tfidf_mass_score_fn, &ctx,
 candidates_out, max_candidates);
 free(q_idf);
 return n;
}

/* ── Rerank scoring (RI cosine + proximity) ──────────────────── */

typedef struct {
 const fce_sem_corpus_t *corpus;
 const int8_t *qvec_q; /* pre-quantized query for int8 rerank */
 float qvec_q_inv_mag; /* reciprocal magnitude of quantized query */
 int top_k;
 _Atomic int next_cand;
 fce_sem_ranked_t *worker_results;
 int *worker_counts;
 const int *candidates;
 int ncand;
} rerank_ctx_t;

static void rerank_worker(int wid, void *uctx) {
 rerank_ctx_t *w = uctx;
 if (!w->corpus->doc_vectors_q && !w->corpus->doc_sparse_idx) return;
 fce_sem_ranked_t *local = w->worker_results + (size_t)wid * w->top_k;
 int n = 0;
 int sparse = (w->corpus->sparse_nnz > 0 && w->corpus->doc_sparse_idx);
 int nnz = w->corpus->sparse_nnz;
 for (;;) {
 int ci = atomic_fetch_add_explicit(&w->next_cand, 1, memory_order_relaxed);
 if (ci >= w->ncand) break;
 int i = w->candidates[ci];
 float dot;
 if (sparse) {
 /* Dense×sparse dot product. */
 const uint16_t *di = &w->corpus->doc_sparse_idx[(size_t)i * nnz];
 const int8_t *dv = &w->corpus->doc_sparse_val[(size_t)i * nnz];
 int32_t acc = 0;
 for (int k = 0; k < nnz; k++) {
 if (di[k] == 0xFFFF) break;
 acc += (int32_t)w->qvec_q[di[k]] * (int32_t)dv[k];
 }
 dot = (float)acc;
 } else {
 dot = (float)fce_dot768_i8(w->qvec_q,
 w->corpus->doc_vectors_q + (size_t)i * FCE_SEM_DIM);
 }
 float inv_d_mag = w->corpus->doc_vectors_q_inv_mag ? w->corpus->doc_vectors_q_inv_mag[i] : 0.0f;
 float cosine = (inv_d_mag > 0.0f && w->qvec_q_inv_mag > 0.0f)
 ? dot * w->qvec_q_inv_mag * inv_d_mag : 0.0f;
 float s = fce_clamp_unit((cosine + FCE_SEM_UNIT_POS) * 0.5f);
 if (n < w->top_k) {
 local[n].index = (uint32_t)i;
 local[n].score = s;
 n++;
 if (n == w->top_k) {
 for (int h = w->top_k / 2 - 1; h >= 0; h--) {
 heap_siftdown(local, h, w->top_k);
 }
 }
 } else if (s > local[0].score ||
 (s == local[0].score && (uint32_t)i < local[0].index)) {
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
 _Static_assert(FCE_CANDIDATE_CAP <= FCE_CANDIDATE_HEAP_SZ,
 "FCE_CANDIDATE_CAP exceeds FCE_CANDIDATE_HEAP_SZ");
 /* rerank_serial's heap is bounded by FCE_CANDIDATE_CAP.
 * If top_k exceeds it, results are silently truncated. Clamp and warn once so
 * callers learn their request was oversized rather than getting mysteriously
 * fewer results than expected. */
 if (top_k > FCE_CANDIDATE_CAP) {
 /* _Atomic to avoid data race when
 * multiple threads call rerank_serial concurrently. At worst a
 * duplicate warning is emitted — the race is benign but UB. */
 static _Atomic int warned = 0;
 if (!warned) {
 /* fce_log_warn expects all variadic
 * args to be const char*. Passing (int)top_k and FCE_CANDIDATE_CAP
 * as values was UB (reads int as pointer). Use fce_log_int which
 * correctly formats integer values. */
 fce_log_int(FCE_LOG_WARN, "rerank_serial.truncated", "top_k", (int64_t)top_k);
 warned = 1;
 }
 top_k = FCE_CANDIDATE_CAP;
 }
 int heap_cap = (int)top_k;
 /* heap-allocate the rerank scratch instead of stack-allocating
 * 64 KB. These functions are reachable from JNI caller threads whose stack size
 * is uncontrolled (small -Xss). The heap allocation is amortized across the
 * search call and avoids a ~50 KB stack frame. */
 fce_sem_ranked_t *heap = malloc((size_t)FCE_CANDIDATE_HEAP_SZ * sizeof(fce_sem_ranked_t));
 if (!heap) {
 if (count_out) *count_out = 0;
 return;
 }
 int hn = 0;
 int sparse = (corpus->sparse_nnz > 0 && corpus->doc_sparse_idx);
 int nnz = corpus->sparse_nnz;
 /* Build query bitset once for bitset-accelerated dot product. */
 for (int ci = 0; ci < ncand; ci++) {
 int i = candidates[ci];
 float dot;
 if (sparse) {
 /* Dense×sparse dot product. */
 const uint16_t *di = &corpus->doc_sparse_idx[(size_t)i * nnz];
 const int8_t *dv = &corpus->doc_sparse_val[(size_t)i * nnz];
 int32_t acc = 0;
 for (int k = 0; k < nnz; k++) {
 if (di[k] == 0xFFFF) break;
 acc += (int32_t)qvec_q[di[k]] * (int32_t)dv[k];
 }
 dot = (float)acc;
 } else {
 dot = (float)fce_dot768_i8(qvec_q,
 corpus->doc_vectors_q + (size_t)i * FCE_SEM_DIM);
 }
 float inv_d_mag = corpus->doc_vectors_q_inv_mag ? corpus->doc_vectors_q_inv_mag[i] : 0.0f;
 float cosine = (inv_d_mag > 0.0f && qvec_q_inv_mag > 0.0f)
 ? dot * qvec_q_inv_mag * inv_d_mag : 0.0f;
 float s = fce_clamp_unit((cosine + FCE_SEM_UNIT_POS) * 0.5f);
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
 free(heap);
}

/* ── Brute-force search (P4: internal with pre-tokenized query) ── */

/* Static-chunked parallel brute-force context. */
typedef struct {
 const fce_sem_corpus_t *corpus;
 const int8_t *qvec_q;
 float qvec_q_inv_mag;
 uint32_t top_k;
 int chunk_start;
 int chunk_end;
 fce_sem_ranked_t *local_heap; /* per-worker local top-k heap */
 int local_count;
} bf_chunk_ctx_t;

static void bf_chunk_worker(int idx, void *ctx) {
 bf_chunk_ctx_t *all = (bf_chunk_ctx_t *)ctx;
 /* idx must be a valid index into all[].
 * The caller uses fce_parallel_for_static(total_chunks, …), which
 * dispatches idx ∈ [0, total_chunks). Assert for defense-in-depth. */
 bf_chunk_ctx_t *c = &all[idx];
 const fce_sem_corpus_t *corpus = c->corpus;
 const int8_t *qvec_q = c->qvec_q;
 float qvec_q_inv_mag = c->qvec_q_inv_mag;
 uint32_t top_k = c->top_k;
 fce_sem_ranked_t *local = c->local_heap;
 int n = 0;
 int sparse = (corpus->sparse_nnz > 0 && corpus->doc_sparse_idx);
 int nnz = corpus->sparse_nnz;

 for (int i = c->chunk_start; i < c->chunk_end; i++) {
 float dot;
 if (sparse) {
 /* Dense×sparse dot product: iterate doc's K entries, look up
 * values in the dense query vector. O(K) with no merge-join. */
 const uint16_t *di = &corpus->doc_sparse_idx[(size_t)i * nnz];
 const int8_t *dv = &corpus->doc_sparse_val[(size_t)i * nnz];
 int32_t acc = 0;
 for (int k = 0; k < nnz; k++) {
 if (di[k] == 0xFFFF) break;
 acc += (int32_t)qvec_q[di[k]] * (int32_t)dv[k];
 }
 dot = (float)acc;
 } else {
 dot = (float)fce_dot768_i8(qvec_q, corpus->doc_vectors_q + (size_t)i * FCE_SEM_DIM);
 }
 float inv_d_mag = corpus->doc_vectors_q_inv_mag ? corpus->doc_vectors_q_inv_mag[i] : 0.0f;
 float cosine = (inv_d_mag > 0.0f && qvec_q_inv_mag > 0.0f) ? dot * qvec_q_inv_mag * inv_d_mag : 0.0f;
 float s = fce_clamp_unit((cosine + FCE_SEM_UNIT_POS) * 0.5f);

 if (n < (int)top_k) {
 local[n].index = (uint32_t)i;
 local[n].score = s;
 n++;
 if (n == (int)top_k) {
 for (int h = (int)top_k / 2 - 1; h >= 0; h--) {
 heap_siftdown(local, h, (int)top_k);
 }
 }
 } else if (s > local[0].score ||
 (s == local[0].score && (uint32_t)i < local[0].index)) {
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
 /* Double OOM — scan all local heaps
 * to extract the global top-k without any allocation. Each iteration
 * finds the highest-scored item across all heaps, appends it to
 * results_out, then removes it by replacing with the last item and
 * shrinking the count. This is O(top_k * total) but correct — the
 * old code simply copied the first top_k items from the first chunks,
 * which could miss better candidates in later chunks. */
 assert(nchunks <= 64); /* init_brute_workers caps at 64 */
 int rem[64];
 for (int c = 0; c < nchunks; c++) rem[c] = chunks[c].local_count;
 uint32_t k = 0;
 for (uint32_t take = 0; take < top_k; take++) {
 int best_c = -1, best_i = -1;
 float best_s = -1.0f;
 for (int c = 0; c < nchunks; c++) {
 for (int i = 0; i < rem[c]; i++) {
 float s = chunks[c].local_heap[i].score;
 if (s > best_s ||
 (s == best_s && best_c >= 0 &&
 chunks[c].local_heap[i].index < chunks[best_c].local_heap[best_i].index)) {
 best_s = s;
 best_c = c;
 best_i = i;
 }
 }
 }
 if (best_c < 0) break;
 results_out[k++] = chunks[best_c].local_heap[best_i];
 chunks[best_c].local_heap[best_i] =
 chunks[best_c].local_heap[--rem[best_c]];
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
 } else if (r.score > heap[0].score ||
 (r.score == heap[0].score && r.index < heap[0].index)) {
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
 if (!corpus->doc_vectors_q && !corpus->doc_sparse_idx) return;

 int n = corpus->doc_count;
 if (n == 0) return;
 if ((int)top_k > n) top_k = (uint32_t)n;

 /* pre-quantize query vector for int8 brute-force path.
 * F2: store reciprocal query magnitude for multiply-only hot loop.
 * L3: reuse pre-quantized query if caller already computed it.
 * M-2: use a stack buffer owned by the call frame
 * instead of _Thread_local storage. The previous _Thread_local buffer
 * was shared with worker threads via pointer — correct only because
 * the main thread joined before the buffer was reused, but fragile.
 * A 768-byte stack buffer is well within any sane stack, including
 * JNI threads with small -Xss. */
 int8_t qvec_q_buf[FCE_SEM_DIM];
 float qvec_q_inv_mag = 0.0f;
 const int8_t *qvec_q = NULL;
 if (qvec_q_pre) {
 qvec_q = qvec_q_pre;
 qvec_q_inv_mag = qvec_q_inv_mag_pre;
 } else if (corpus->doc_vectors_q || corpus->doc_sparse_idx) {
 fce_quantize_f32_768(qvec_q_buf, qvec->v);
 float mag_sq = 0.0f;
 for (int i = 0; i < FCE_SEM_DIM; i++) mag_sq += (float)qvec_q_buf[i] * (float)qvec_q_buf[i];
 qvec_q_inv_mag = (mag_sq > 0.0f) ? 1.0f / sqrtf(mag_sq) : 0.0f;
 qvec_q = qvec_q_buf;
 }

 /* Static-chunked parallel brute-force.
 * Previous parallel path used atomic_fetch_add per doc (contention).
 * Static chunking: each worker gets a contiguous doc range, zero atomics.
 * Heuristic: use parallel when scan > 50 MB (≈66K docs × 768B).
 * Default: total_cores / 4 (min 1). Env var FCE_BRUTE_WORKERS overrides. */
 size_t scan_bytes = (size_t)n * FCE_SEM_DIM;

 /* use cached value instead of calling
 * fce_safe_getenv on every search (not safe from concurrent threads). */
 fce_once(&g_brute_workers_once, init_brute_workers);
 int nworkers = g_cached_brute_workers;

 if (nworkers > 1 && scan_bytes > 50 * 1024 * 1024) {
 /* Parallel path: split docs into nworkers contiguous chunks. */
 int total_chunks = nworkers;
 int chunk_size = n / total_chunks;
 int remainder = n % total_chunks;

 bf_chunk_ctx_t *chunks = (bf_chunk_ctx_t *)calloc((size_t)total_chunks, sizeof(bf_chunk_ctx_t));
 if (!chunks) goto fallback_serial;

 int offset = 0;
 for (int c = 0; c < total_chunks; c++) {
 int sz = chunk_size + (c < remainder ? 1 : 0);
 /* cap per-chunk heap to min(top_k, chunk_size)
 * to avoid nworkers × top_k memory amplification when top_k ≈ doc_count. */
 uint32_t chunk_k = (uint32_t)sz < top_k ? (uint32_t)sz : top_k;
 chunks[c] = (bf_chunk_ctx_t){
 .corpus = corpus, .qvec_q = qvec_q, .qvec_q_inv_mag = qvec_q_inv_mag,
 .top_k = chunk_k, .chunk_start = offset, .chunk_end = offset + sz,
 .local_heap = (fce_sem_ranked_t *)calloc(chunk_k, sizeof(fce_sem_ranked_t)),
 .local_count = 0
 };
 if (!chunks[c].local_heap) {
 for (int j = 0; j < c; j++) free(chunks[j].local_heap);
 free(chunks);
 goto fallback_serial;
 }
 offset += sz;
 }

 /* fce_parallel_for_static treats max_workers
 * as total parallelism (spawns max_workers-1 threads + main = max_workers
 * executors). Passing nworkers-1 here gave nworkers-1 total executors for
 * nworkers chunks — one thread ran two chunks back-to-back, halving
 * throughput. With nworkers==2 (the default on 8-core), max_workers==1
 * hit the serial guard and the entire scan ran single-threaded. */
 fce_parallel_for_opts_t popts = { .max_workers = nworkers };
 fce_parallel_for_static(total_chunks, bf_chunk_worker, chunks, popts);

 /* Merge local heaps into final results. */
 *count_out = merge_local_heaps(chunks, total_chunks, top_k, results_out);

 for (int c = 0; c < total_chunks; c++) free(chunks[c].local_heap);
 free(chunks);
 return;
 }

fallback_serial:
 {
 sq_sctx_t ctx = {
 .corpus = corpus, .qvec_q = qvec_q, .qvec_q_inv_mag = qvec_q_inv_mag,
 };
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
 if (!corpus->doc_vectors_q && !corpus->doc_sparse_idx) return;

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

 /* check for zero query vector before scoring. */
 float qmag = 0.0f;
 for (int i = 0; i < FCE_SEM_DIM; i++) qmag += qvec.v[i] * qvec.v[i];
 if (qmag == 0.0f) goto cleanup;

 bruteforce_precomputed(corpus, &qvec, NULL, 0.0f, top_k, results_out, count_out);

cleanup:
 for (int t = 0; t < q_ntok; t++) free(q_toks[t]);
}

/* ── Fast search: inverted index candidate retrieval + rerank ─── */

void fce_sem_search_query(const fce_sem_corpus_t *corpus,
 const char *query,
 uint32_t top_k,
 fce_sem_ranked_t *results_out,
 uint32_t *count_out,
 const fce_sem_config_t *cfg) {
 if (count_out) *count_out = 0;
 if (!corpus || !query || top_k == 0 || !results_out) return;
 if (!corpus->doc_vectors_q && !corpus->doc_sparse_idx) return;

 fce_query_mode_t mode = cfg ? cfg->query_mode : FCE_QUERY_AUTO;

 /* FCE_QUERY_BRUTE: skip inverted index, go straight to brute-force. */
 if (mode == FCE_QUERY_BRUTE) {
 char *q_toks[FCE_SEM_MAX_TOKENS];
 int q_ntok = fce_sem_tokenize(query, q_toks, FCE_SEM_MAX_TOKENS);
 if (q_ntok == 0) return;
 fce_sem_vec_t qvec;
 memset(&qvec, 0, sizeof(qvec));
 for (int t = 0; t < q_ntok; t++) {
 const fce_sem_vec_t *rv = fce_sem_corpus_ri_vec(corpus, q_toks[t]);
 if (rv) fce_sem_vec_add_scaled(&qvec, rv, 1.0f);
 }
 fce_sem_normalize(&qvec);
 bruteforce_precomputed(corpus, &qvec, NULL, 0.0f, top_k, results_out, count_out);
 for (int t = 0; t < q_ntok; t++) free(q_toks[t]);
 return;
 }

 /* FCE_QUERY_TFIDF: redirect to TF-IDF candidate path (use AUTO inside
 * so it uses its own fast path without circular redirect). */
 if (mode == FCE_QUERY_TFIDF) {
 fce_sem_config_t auto_cfg = cfg ? *cfg : (fce_sem_config_t){.query_mode = FCE_QUERY_AUTO};
 auto_cfg.query_mode = FCE_QUERY_AUTO;
 fce_sem_search_query_tfidf(corpus, query, top_k, results_out, count_out, &auto_cfg);
 return;
 }

 int n = corpus->doc_count;
 if (n == 0) return;
 if ((int)top_k > n) top_k = (uint32_t)n;

 /* stack footprint is ~10 KB total —
 * char *q_toks[512] (4 KB), int q_tok_ids[512] (2 KB),
 * fce_sem_vec_t qvec (3 KB), int8_t qvec_q_buf[768] (768 B),
 * plus candidates heap-allocated. Safe for default 1 MB worker stacks. */
 int *candidates = (int *)malloc(sizeof(int) * FCE_CANDIDATE_CAP);
 if (!candidates) {
 if (count_out) *count_out = 0;
 return;
 }

 /* Tokenize query. */
 char *q_toks[FCE_SEM_MAX_TOKENS];
 int q_ntok = fce_sem_tokenize(query, q_toks, FCE_SEM_MAX_TOKENS);
 if (q_ntok == 0) { free(candidates); return; }

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

 /* pre-quantize query for int8 rerank path. */
 int8_t qvec_q_buf[FCE_SEM_DIM];
 float qvec_q_inv_mag = 0.0f;
 const int8_t *qvec_q = NULL;
 if (corpus->doc_vectors_q || corpus->doc_sparse_idx) {
 fce_quantize_f32_768(qvec_q_buf, qvec.v);
 float mag_sq = 0.0f;
 for (int i = 0; i < FCE_SEM_DIM; i++) mag_sq += (float)qvec_q_buf[i] * (float)qvec_q_buf[i];
 qvec_q_inv_mag = (mag_sq > 0.0f) ? 1.0f / sqrtf(mag_sq) : 0.0f;
 qvec_q = qvec_q_buf;
 }
 /* a zero/degenerate query vector (empty tokenization
 * or all-zero embeddings) produces qvec_q_inv_mag == 0, causing every document
 * to score exactly 0.5 — a silent quality cliff. Return an empty result set
 * instead so callers don't mistake degenerate scores for real results.
 * H-1: goto cleanup so q_toks[0..q_ntok-1] are freed;
 * the earlier bare return leaked those strdup'd tokens. */
 if (qvec_q_inv_mag == 0.0f) {
 if (count_out) *count_out = 0;
 goto cleanup;
 }

 /* FCE_QUERY_FAST: force inverted index path, skip fallback to brute if too few. */
 if (mode == FCE_QUERY_FAST) {
 if (!corpus->inv_offsets || q_id_count == 0) goto brute_force;
 }

 /* Try inverted index candidate retrieval. */
 if (corpus->inv_offsets && q_id_count > 0) {
 int ncand = keyword_candidates(corpus, q_tok_ids, q_id_count,
 candidates, FCE_CANDIDATE_CAP);

 /* Fall back to brute-force if too few candidates, unless FAST mode
 * explicitly suppresses the fallback.
 * M-3: log once when FAST mode returns fewer
 * results than requested, so callers can detect an oversized request. */
 if (ncand < (int)top_k && mode != FCE_QUERY_FAST) {
 goto brute_force;
 }
 if (ncand < (int)top_k && mode == FCE_QUERY_FAST) {
 /* _Atomic to avoid data race. */
 static _Atomic int warned = 0;
 if (!warned) {
 /* fce_log_warn consumes all variadic
 * args as const char* — passing int is UB (reads int as pointer).
 * Use fce_log_int which correctly formats integer values. */
 fce_log_int(FCE_LOG_WARN, "query_fast.few_results",
 "ncand", (int64_t)ncand);
 fce_log_int(FCE_LOG_WARN, "query_fast.few_results",
 "top_k", (int64_t)top_k);
 warned = 1;
 }
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
 /* worker_count is clamped to ≥1 here, which
 * bounds the malloc below (worker_count * top_k). The earlier
 * comment about "worker_count ≤ n/top_k ≤ doc_count" is the
 * INT_MAX overflow guard, NOT the allocation-size bound — the
 * allocation is safe because worker_count * top_k ≤ (ncand/top_k+1) * top_k
 * ≤ ncand + top_k, and ncand ≤ FCE_CANDIDATE_CAP. */
 assert(worker_count >= 1);
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
 /* reuse pre-tokenized query instead of re-tokenizing.
 * L3: pass pre-quantized query to avoid redundant quantization. */
 bruteforce_precomputed(corpus, &qvec, qvec_q, qvec_q_inv_mag,
 top_k, results_out, count_out);

cleanup:
 free(candidates);
 for (int t = 0; t < q_ntok; t++) free(q_toks[t]);
}

/* ── TF-IDF hybrid search: TF-IDF cosine candidates + RI rerank ── */

void fce_sem_search_query_tfidf(const fce_sem_corpus_t *corpus,
 const char *query,
 uint32_t top_k,
 fce_sem_ranked_t *results_out,
 uint32_t *count_out,
 const fce_sem_config_t *cfg) {
 if (count_out) *count_out = 0;
 if (!corpus || !query || top_k == 0 || !results_out) return;
 if (!corpus->doc_vectors_q && !corpus->doc_sparse_idx) return;

 fce_query_mode_t mode = cfg ? cfg->query_mode : FCE_QUERY_AUTO;

 /* FCE_QUERY_BRUTE: skip TF-IDF candidates, go straight to brute-force. */
 if (mode == FCE_QUERY_BRUTE) {
 char *q_toks[FCE_SEM_MAX_TOKENS];
 int q_ntok = fce_sem_tokenize(query, q_toks, FCE_SEM_MAX_TOKENS);
 if (q_ntok == 0) return;
 fce_sem_vec_t qvec;
 memset(&qvec, 0, sizeof(qvec));
 for (int t = 0; t < q_ntok; t++) {
 const fce_sem_vec_t *rv = fce_sem_corpus_ri_vec(corpus, q_toks[t]);
 if (rv) fce_sem_vec_add_scaled(&qvec, rv, 1.0f);
 }
 fce_sem_normalize(&qvec);
 bruteforce_precomputed(corpus, &qvec, NULL, 0.0f, top_k, results_out, count_out);
 for (int t = 0; t < q_ntok; t++) free(q_toks[t]);
 return;
 }

 /* FCE_QUERY_FAST: redirect to inverted-index fast path. */
 if (mode == FCE_QUERY_FAST) {
 fce_sem_search_query(corpus, query, top_k, results_out, count_out, cfg);
 return;
 }

 int n = corpus->doc_count;
 if (n == 0) return;
 if ((int)top_k > n) top_k = (uint32_t)n;

 /* heap-allocate candidates to reduce stack
 * footprint from ~15 KB to ~2 KB (only q_tok_ids on stack now). */
 int *candidates_tf = (int *)malloc(sizeof(int) * FCE_CANDIDATE_CAP);
 if (!candidates_tf) {
 if (count_out) *count_out = 0;
 return;
 }

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

 /* pre-quantize query for int8 rerank path. */
 int8_t qvec_q_buf_tf[FCE_SEM_DIM];
 float qvec_q_inv_mag_tf = 0.0f;
 const int8_t *qvec_q_tf = NULL;
 if (corpus->doc_vectors_q || corpus->doc_sparse_idx) {
 fce_quantize_f32_768(qvec_q_buf_tf, qvec.v);
 float mag_sq = 0.0f;
 for (int i = 0; i < FCE_SEM_DIM; i++) mag_sq += (float)qvec_q_buf_tf[i] * (float)qvec_q_buf_tf[i];
 qvec_q_inv_mag_tf = (mag_sq > 0.0f) ? 1.0f / sqrtf(mag_sq) : 0.0f;
 qvec_q_tf = qvec_q_buf_tf;
 }
 /* zero/degenerate query vector — return empty results. */
 if (qvec_q_inv_mag_tf == 0.0f) {
 if (count_out) *count_out = 0;
 goto cleanup_tf;
 }

 if (corpus->inv_offsets && q_id_count > 0) {
 int ncand = tfidf_keyword_candidates(corpus, q_tok_ids, q_id_count,
 candidates_tf, FCE_CANDIDATE_CAP);
 if (ncand < (int)top_k) goto brute_tf;

 int worker_count = fce_default_worker_count(true);
 if (worker_count < 1) worker_count = 1;

 if (worker_count <= 1 || ncand / 2 <= (int)top_k) {
 rerank_serial(corpus, qvec_q_tf, qvec_q_inv_mag_tf,
 candidates_tf, ncand, top_k, results_out, count_out);
 } else {
 if (ncand / (int)top_k < worker_count) worker_count = ncand / (int)top_k;
 if (worker_count < 1) worker_count = 1;
 assert(worker_count >= 1);
 fce_sem_ranked_t *worker_results = malloc((size_t)worker_count * top_k * sizeof(fce_sem_ranked_t));
 int *worker_counts = calloc((size_t)worker_count, sizeof(int));
 if (!worker_results || !worker_counts) {
 free(worker_results); free(worker_counts);
 goto brute_tf;
 }
 rerank_ctx_t ctx = {
 .corpus = corpus,
 .qvec_q = qvec_q_tf, .qvec_q_inv_mag = qvec_q_inv_mag_tf,
 .top_k = (int)top_k, .candidates = candidates_tf,
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
 /* reuse pre-tokenized query instead of re-tokenizing.
 * L3: pass pre-quantized query to avoid redundant quantization. */
 bruteforce_precomputed(corpus, &qvec, qvec_q_tf, qvec_q_inv_mag_tf,
 top_k, results_out, count_out);

cleanup_tf:
 free(candidates_tf);
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

 /* heap-allocate to reduce stack footprint. */
 int *candidates = (int *)malloc(sizeof(int) * FCE_CANDIDATE_CAP);
 if (!candidates) {
 for (int t = 0; t < q_ntok; t++) free(q_toks[t]);
 return 0;
 }
 int ncand = keyword_candidates(corpus, q_tok_ids, q_id_count,
 candidates, FCE_CANDIDATE_CAP);

 free(candidates);
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
 /* default to fce_sem_get_config() when cfg is NULL,
 * matching fce_sem_search_query* which treats NULL as AUTO. */
 fce_sem_config_t default_cfg;
 if (!cfg) {
 default_cfg = fce_sem_get_config();
 cfg = &default_cfg;
 }
 if (!query || !corpus || corpus_size == 0 || top_k == 0) {
 if (count_out) *count_out = 0;
 return;
 }
 /* Defend against a NULL
 * tfidf_indices or tfidf_weights with tfidf_len > 0. The struct's
 * invariant is "tfidf_indices != NULL iff tfidf_len > 0" but a
 * direct C caller could violate it. The sort loop and the TF-IDF
 * magnitude loop below would NULL-deref. */
 if (query->tfidf_len > 0 && (query->tfidf_indices == NULL || query->tfidf_weights == NULL)) {
 fce_log_error("fce_sem_search: tfidf_len > 0 with NULL tfidf_indices or tfidf_weights");
 if (count_out) *count_out = 0;
 return;
 }
 /* Validate query-side TF-IDF sort invariant once at entry.
 * M3: use <= to reject duplicates (not just <).
 * fce_sparse_tfidf_cosine's two-pointer merge desynchronizes on equal
 * indices, producing an incorrect dot product. The simple-search path
 * and JNI marshal already use <=; the combined path must match. */
 for (int i = 1; i < query->tfidf_len; i++) {
 if (query->tfidf_indices[i] <= query->tfidf_indices[i-1]) {
 fce_log_int(FCE_LOG_ERROR,
 "fce_sem_search.duplicate_query_tfidf",
 "slot", i);
 if (count_out) *count_out = 0;
 return;
 }
 }

 /* clamp top_k to INT_MAX symmetrically with
 * corpus_size. Without this, top_k > INT_MAX casts to negative int,
 * bypassing the clamp and causing an unbounded linear fill of results. */
 if (top_k > (uint32_t)INT_MAX) top_k = (uint32_t)INT_MAX;
 /* Clamp to INT_MAX to avoid undefined behavior when
 * corpus_size > INT_MAX (implementation-defined cast to int). */
 int n = corpus_size > (uint32_t)INT_MAX ? INT_MAX : (int)corpus_size;
 if ((int)top_k > n) top_k = (uint32_t)n;

 int worker_count = fce_default_worker_count(true);
 if (worker_count < 1) worker_count = 1;

 /* Small corpus or single worker: serial path. */
 if (worker_count <= 1 || n / 2 <= (int)top_k) {
 int *cprox = precompute_corpus_prox(corpus, n);
 sc_sctx_t ctx = {
 .query = query, .corpus = corpus, .cfg = cfg,
 .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
 .q_tfidf_mag_sq = FCE_MAG_COMPUTE_INLINE, .q_ri_mag_sq = FCE_MAG_COMPUTE_INLINE,
 .q_api_mag_sq = FCE_MAG_COMPUTE_INLINE, .q_type_mag_sq = FCE_MAG_COMPUTE_INLINE,
 .q_deco_mag_sq = FCE_MAG_COMPUTE_INLINE, .q_sp_mag_sq = FCE_MAG_COMPUTE_INLINE,
 .corpus_prox = cprox,
 };
 if (count_out) *count_out = serial_topk(sc_score, &ctx, (uint32_t)n, top_k, min_score, results_out);
 free(cprox);
 return;
 }

 /* Parallel path: partition corpus across workers, each builds a local top-k. */
 if (n / (int)top_k < worker_count) worker_count = n / (int)top_k;
 if (worker_count < 1) worker_count = 1;
 /* precompute cprox before worker allocations so the
 * OOM fallback to serial can reuse it instead of re-walking every path. */
 int *cprox = precompute_corpus_prox(corpus, n);
 fce_sem_ranked_t *worker_results = malloc((size_t)worker_count * top_k * sizeof(fce_sem_ranked_t));
 int *worker_counts = calloc((size_t)worker_count, sizeof(int));
 if (!worker_results || !worker_counts) {
 free(worker_results);
 free(worker_counts);
 free(cprox);
 /* Fallback to serial on OOM. */
 sc_sctx_t ctx = {
 .query = query, .corpus = corpus, .cfg = cfg,
 .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
 .q_tfidf_mag_sq = FCE_MAG_COMPUTE_INLINE, .q_ri_mag_sq = FCE_MAG_COMPUTE_INLINE,
 .q_api_mag_sq = FCE_MAG_COMPUTE_INLINE, .q_type_mag_sq = FCE_MAG_COMPUTE_INLINE,
 .q_deco_mag_sq = FCE_MAG_COMPUTE_INLINE, .q_sp_mag_sq = FCE_MAG_COMPUTE_INLINE,
 };
 if (count_out) *count_out = serial_topk(sc_score, &ctx, (uint32_t)n, top_k, min_score, results_out);
 return;
 }
 if (!cprox) {
 free(worker_results);
 free(worker_counts);
 /* Fallback to serial on OOM (no precomputed proximity). */
 sc_sctx_t ctx = {
 .query = query, .corpus = corpus, .cfg = cfg,
 .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
 .q_tfidf_mag_sq = FCE_MAG_COMPUTE_INLINE, .q_ri_mag_sq = FCE_MAG_COMPUTE_INLINE,
 .q_api_mag_sq = FCE_MAG_COMPUTE_INLINE, .q_type_mag_sq = FCE_MAG_COMPUTE_INLINE,
 .q_deco_mag_sq = FCE_MAG_COMPUTE_INLINE, .q_sp_mag_sq = FCE_MAG_COMPUTE_INLINE,
 };
 if (count_out) *count_out = serial_topk(sc_score, &ctx, (uint32_t)n, top_k, min_score, results_out);
 return;
 }

 int q_prox = fce_count_slashes(query->file_path);

 /* Precompute query-side magnitudes once (P1) — avoids ~3072 FLOPs per corpus item. */
 float q_tfidf_mag_sq = FCE_MAG_COMPUTE_INLINE;
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
 .corpus_size = n,
 .min_score = min_score,
 .q_prox = q_prox,
 .q_tfidf_mag_sq = q_tfidf_mag_sq,
 .q_ri_mag_sq = q_ri_mag_sq,
 .q_api_mag_sq = q_api_mag_sq,
 .q_type_mag_sq = q_type_mag_sq,
 .q_deco_mag_sq = q_deco_mag_sq,
 .q_sp_mag_sq = q_sp_mag_sq,
 .cfg = cfg,
 .corpus_prox = cprox,
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
 * query-side magnitudes (P1). Pass FCE_MAG_COMPUTE_INLINE to compute inline. */
static float fce_score_simple_internal(fce_sem_func_t *a, fce_sem_func_t *b, float prox,
 float q_ri_mag_sq) {
 if (!a || !b) return 0.0f;

 /* RI cosine — use precomputed query magnitude if provided.
 * H1: TF-IDF dropped from the simple API because buildFunc uses positional
 * indices (0,1,2,...) not global vocab IDs, making the sparse cosine merge
 * meaningless. The RI half is correct. */
 float ri_mag = q_ri_mag_sq;
 if (ri_mag < 0.0f) {
 ri_mag = 0.0f;
 for (int i = 0; i < FCE_SEM_DIM; i++) {
 ri_mag += a->ri_vec.v[i] * a->ri_vec.v[i];
 }
 }
 float ri_raw = fce_sem_cosine_aliased_with_mag(a->ri_vec.v, b->ri_vec.v, ri_mag);
 float ri = (ri_raw + FCE_SEM_UNIT_POS) * 0.5f;

 ri *= prox;
 if (!isfinite(ri)) {
 /* one-shot warning, same gate as
 * score_combined_internal. */
 if (atomic_exchange_explicit(&g_nonfinite_warned, 1, memory_order_acq_rel) == 0) {
 fce_log_warn("simple_score.nonfinite_input", NULL);
 }
 return 0.0f;
 }
 if (ri > FCE_SEM_UNIT_POS) ri = FCE_SEM_UNIT_POS;
 if (ri < 0.0f) ri = 0.0f;
 return ri;
}

float fce_sem_simple_score(fce_sem_func_t *a, fce_sem_func_t *b) {
 FCE_ASSERT_TFIDF_SORTED(a);
 FCE_ASSERT_TFIDF_SORTED(b);
 float prox = fce_sem_proximity(a ? a->file_path : NULL, b ? b->file_path : NULL);
 return fce_score_simple_internal(a, b, prox, FCE_MAG_COMPUTE_INLINE);
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
 /* NULL tfidf_indices or
 * tfidf_weights with tfidf_len > 0 is a caller-contract violation;
 * the sort loop below would NULL-deref. */
 if (query->tfidf_len > 0 && (query->tfidf_indices == NULL || query->tfidf_weights == NULL)) {
 fce_log_error("fce_sem_simple_search: tfidf_len > 0 with NULL tfidf_indices or tfidf_weights");
 if (count_out) *count_out = 0;
 return;
 }
 /* validate query-side TF-IDF sort invariant once at
 * entry. Use <= to reject duplicates — the two-pointer merge in
 * fce_sparse_tfidf_cosine desynchronizes on equal indices. */
 for (int i = 1; i < query->tfidf_len; i++) {
 if (query->tfidf_indices[i] <= query->tfidf_indices[i-1]) {
 fce_log_int(FCE_LOG_ERROR,
 "fce_sem_simple_search.duplicate_query_tfidf",
 "slot", i);
 if (count_out) *count_out = 0;
 return;
 }
 }

 /* clamp top_k to INT_MAX. */
 if (top_k > (uint32_t)INT_MAX) top_k = (uint32_t)INT_MAX;
 /* clamp to INT_MAX. */
 int n = corpus_size > (uint32_t)INT_MAX ? INT_MAX : (int)corpus_size;
 if ((int)top_k > n) top_k = (uint32_t)n;

 int worker_count = fce_default_worker_count(true);
 if (worker_count < 1) worker_count = 1;

 /* Small corpus or single worker: serial path. */
 if (worker_count <= 1 || n / 2 <= (int)top_k) {
 int *cprox = precompute_corpus_prox(corpus, n);
 fce_ss_sctx_t ctx = {
 .query = query, .corpus = corpus,
 .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
 .q_ri_mag_sq = FCE_MAG_COMPUTE_INLINE,
 .corpus_prox = cprox,
 };
 if (count_out) *count_out = serial_topk(fce_ss_score, &ctx, (uint32_t)n, top_k, min_score, results_out);
 free(cprox);
 return;
 }

 /* Parallel path. */
 if (n / (int)top_k < worker_count) worker_count = n / (int)top_k;
 if (worker_count < 1) worker_count = 1;
 /* precompute cprox before worker allocations so the
 * OOM fallback to serial can reuse it instead of re-walking every path. */
 int *cprox = precompute_corpus_prox(corpus, n);
 fce_sem_ranked_t *worker_results = malloc((size_t)worker_count * top_k * sizeof(fce_sem_ranked_t));
 int *worker_counts = calloc((size_t)worker_count, sizeof(int));
 if (!worker_results || !worker_counts) {
 free(worker_results);
 free(worker_counts);
 free(cprox);
 /* Fallback to serial on OOM. */
 fce_ss_sctx_t ctx = {
 .query = query, .corpus = corpus,
 .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
 .q_ri_mag_sq = FCE_MAG_COMPUTE_INLINE,
 };
 if (count_out) *count_out = serial_topk(fce_ss_score, &ctx, (uint32_t)n, top_k, min_score, results_out);
 return;
 }
 if (!cprox) {
 free(worker_results);
 free(worker_counts);
 /* Fallback to serial on OOM (no precomputed proximity). */
 fce_ss_sctx_t ctx = {
 .query = query, .corpus = corpus,
 .min_score = min_score, .q_prox = fce_count_slashes(query->file_path),
 .q_ri_mag_sq = FCE_MAG_COMPUTE_INLINE,
 };
 if (count_out) *count_out = serial_topk(fce_ss_score, &ctx, (uint32_t)n, top_k, min_score, results_out);
 return;
 }

 int q_prox = fce_count_slashes(query->file_path);

 /* Precompute query-side RI magnitude once (P1). */
 float q_ri_mag_sq = 0.0F;
 for (int i = 0; i < FCE_SEM_DIM; i++) {
 q_ri_mag_sq += query->ri_vec.v[i] * query->ri_vec.v[i];
 }

 fce_ss_ctx_t ctx = {
 .query = query,
 .corpus = corpus,
 .corpus_size = n,
 .min_score = min_score,
 .q_prox = q_prox,
 .q_ri_mag_sq = q_ri_mag_sq,
 .corpus_prox = cprox,
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
 const int *all_tfidf_indices,
 const int *tfidf_lens,
 const float *all_ri_vecs,
 const char **file_paths,
 uint32_t corpus_size,
 int max_tokens,
 const int *q_tfidf_indices,
 const float *q_tfidf_weights,
 int q_tfidf_len,
 const float *q_ri_vec,
 uint32_t top_k,
 fce_sem_ranked_t *results_out,
 uint32_t *count_out) {

 if (!all_ri_vecs || corpus_size == 0 || top_k == 0) {
 if (count_out) *count_out = 0;
 return;
 }

 /* TF-IDF corpus arrays are optional in the flat path.
 * The flat scorer (fce_score_flat) intentionally drops TF-IDF and uses RI only,
 * so callers building RI-only corpora should not be forced to allocate and pin
 * large corpusSize × maxTokens weight/index arrays that are then ignored.
 * When TF-IDF arrays are NULL, set max_tokens = 0 so that sf_score's
 * pointer arithmetic is a harmless NULL + 0 = NULL. */
 bool have_tfidf = (all_tfidf_weights && all_tfidf_indices && tfidf_lens);
 if (!have_tfidf) {
 max_tokens = 0;
 }

 /* defend against a NULL q_tfidf_indices with
 * q_tfidf_len > 0. The sort loop below would NULL-deref. The
 * JNI path always pairs indices with weights, but a direct C
 * caller could supply only weights. */
 if (q_tfidf_len > 0 && q_tfidf_indices == NULL) {
 fce_log_error("fce_sem_simple_rank_flat: q_tfidf_len > 0 with NULL q_tfidf_indices");
 if (count_out) *count_out = 0;
 return;
 }

 if (!q_ri_vec) {
 if (count_out) *count_out = 0;
 return;
 }

 /* Precompute query-side RI magnitude once.
 * q_ri_vec is guaranteed non-NULL by the guard above. */
 float q_ri_mag = 0.0F;
 for (int i = 0; i < FCE_SEM_DIM; i++) {
 q_ri_mag += q_ri_vec[i] * q_ri_vec[i];
 }

 /* clamp top_k to INT_MAX. */
 if (top_k > (uint32_t)INT_MAX) top_k = (uint32_t)INT_MAX;
 /* clamp to INT_MAX. */
 int n = corpus_size > (uint32_t)INT_MAX ? INT_MAX : (int)corpus_size;
 if ((int)top_k > n) top_k = (uint32_t)n;

 /* validate query-side TF-IDF indices for
 * diagnostic purposes. The flat scorer (fce_score_flat) does NOT
 * consume TF-IDF arrays — it casts them away — so unsorted/duplicate
 * indices do NOT affect the score. Downgrade from hard error to
 * debug log so callers passing TF-IDF "for layout symmetry" get
 * correct results instead of an empty result set. */
 for (int i = 1; i < q_tfidf_len; i++) {
 if (q_tfidf_indices[i] <= q_tfidf_indices[i-1]) {
 fce_log_debug("fce_sem_simple_rank_flat.unsorted_query_tfidf",
 "slot", i);
 break; /* one-shot: don't spam on every invocation */
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
 if (count_out) *count_out = serial_topk(sf_score, &ctx, (uint32_t)n, top_k, -1.0f, results_out);
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
 if (count_out) *count_out = serial_topk(sf_score, &ctx, (uint32_t)n, top_k, -1.0f, results_out);
 return;
 }

 flat_ctx_t ctx = {
 .all_tfidf_weights = all_tfidf_weights,
 .all_tfidf_indices = all_tfidf_indices,
 .tfidf_lens = tfidf_lens,
 .all_ri_vecs = all_ri_vecs,
 .file_paths = file_paths,
 .max_tokens = max_tokens,
 .corpus_size = n,
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
 /* assert no-aliasing between `combined` and
 * `neighbors[]`. The function blends combined = (1-α)·combined + α·mean,
 * where mean is built by summing neighbors into a local. If `combined`
 * is also a member of `neighbors[]`, the `mean` accumulator reads the
 * *original* combined value (safe) but the final blend writes back into
 * a slot that was just summed into mean — that's fine, but a future
 * refactor that reads from combined after the mean loop would silently
 * observe corrupted values. The contract is documented on the prototype
 * in semantic.h; this assert enforces it in debug builds. */
#ifndef NDEBUG
 for (int n = 0; n < neighbor_count; n++) {
 assert(neighbors[n].v != combined->v);
 }
#endif
 /* detect aliasing at runtime and skip aliased
 * neighbors. The old one-shot warning hid systemic misuse after the
 * first occurrence; silently blending a self-referencing vector is a
 * "wrong answers, no crash" failure mode. We skip any neighbor whose
 * data pointer aliases combined, which is the safe default. */
 /* Blend: combined = (1-α) × combined + α × mean(neighbors) */
 /* use float accumulator instead of double to
 * reduce stack usage from ~9 KB to ~3 KB. The double accumulator bought
 * ~10 ULP extra precision, negligible after the final normalize(). */
 fce_sem_vec_t mean;
 for (int i = 0; i < FCE_SEM_DIM; i++) mean.v[i] = 0.0f;
 int valid_count = 0;
 for (int n = 0; n < neighbor_count; n++) {
 /* skip aliased neighbors — including
 * combined in its own mean would corrupt the result. */
 if (neighbors[n].v == combined->v) continue;
 for (int i = 0; i < FCE_SEM_DIM; i++) {
 mean.v[i] += neighbors[n].v[i];
 }
 valid_count++;
 }
 /* If all neighbors were aliased (degenerate case), skip the blend. */
 if (valid_count == 0) return;
 float inv_n = FCE_SEM_UNIT_POS / (float)valid_count;
 float one_minus_alpha = FCE_SEM_UNIT_POS - alpha;
 for (int i = 0; i < FCE_SEM_DIM; i++) {
 combined->v[i] = (one_minus_alpha * combined->v[i]) + (alpha * mean.v[i] * inv_n);
 }
 fce_sem_normalize(combined);
}
