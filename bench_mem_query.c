/* * bench_mem_query.c — Index directory, measure memory, benchmark queries.
 *
 * Build: cc -O2 -std=c11 -Isrc bench_mem_query.c -Lbuild -lfast_code_embed -lpthread -lm -o bench_mem_query
 * Run: ./bench_mem_query <directory> [chunk_size] */

/* Expose POSIX/GNU symbols (clock_gettime, strdup, fileno, lstat, setenv)
 * which are hidden under strict -std=c11 on glibc. Must precede all includes. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "semantic/semantic.h"
#include "foundation/platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <time.h>
#include <math.h>
#include <fnmatch.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#define DEFAULT_CHUNK_SIZE 5000
#define BATCH_SIZE 50000
#define MAX_PATH_LEN 1024
#define MAX_TOKENS_PER_CHUNK 256

static const char *INCLUDE_EXTS[] = { ".c", ".h", ".cpp", ".hpp", ".java", ".py", ".rs", ".go", ".js", ".ts", NULL };

static int should_include(const char *path) {
 size_t len = strlen(path);
 for (const char **ext = INCLUDE_EXTS; *ext; ext++) {
 size_t extlen = strlen(*ext);
 if (len >= extlen && strcmp(path + len - extlen, *ext) == 0) return 1;
 }
 return 0;
}

static char *read_file(const char *path, size_t *out_len) {
 FILE *f = fopen(path, "rb");
 if (!f) return NULL;
 struct stat st;
 if (fstat(fileno(f), &st) != 0 || st.st_size <= 0) { fclose(f); return NULL; }
 char *buf = (char *)malloc((size_t)st.st_size);
 if (!buf) { fclose(f); return NULL; }
 size_t nread = fread(buf, 1, (size_t)st.st_size, f);
 /* check ferror BEFORE fclose, mirroring the hardened
 * pattern in semantic.c:551-557 and index_dir.c. A short read with
 * ferror set means I/O error (not EOF); discard truncated data. */
 int read_err = (nread != (size_t)st.st_size && ferror(f));
 fclose(f);
 if (read_err) { free(buf); *out_len = 0; return NULL; }
 *out_len = nread;
 return buf;
}

typedef struct { char **paths; int count; int capacity; } file_list_t;
static void file_list_add(file_list_t *list, const char *path) {
 if (list->count >= list->capacity) {
 int new_cap = list->capacity ? list->capacity * 2 : 10000;
 char **grown = (char **)realloc(list->paths, (size_t)new_cap * sizeof(char *));
 if (!grown) return;
 list->paths = grown;
 list->capacity = new_cap;
 }
 list->paths[list->count] = strdup(path);
 if (!list->paths[list->count]) return; /* skip on OOM */
 list->count++;
}

/* iterative directory walk — prevents stack overflow
 * on deep directory trees. Mirrors the hardened walk_dir in index_dir.c. */
static void walk_dir(const char *root, file_list_t *list) {
 int stack_cap = 256;
 int stack_len = 0;
 char **dir_stack = (char **)malloc((size_t)stack_cap * sizeof(char *));
 if (!dir_stack) return;
 dir_stack[stack_len++] = strdup(root);
 if (!dir_stack[0]) { free(dir_stack); return; }

 while (stack_len > 0) {
 char *dirpath = dir_stack[--stack_len];
 DIR *dir = opendir(dirpath);
 if (!dir) { free(dirpath); continue; }
 struct dirent *entry;
 while ((entry = readdir(dir)) != NULL) {
 if (entry->d_name[0] == '.') continue;
 char fullpath[MAX_PATH_LEN];
 int written = snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
 if (written < 0 || (size_t)written >= sizeof(fullpath)) continue;
 struct stat st;
 if (lstat(fullpath, &st) != 0) continue;
 if (S_ISLNK(st.st_mode)) continue;
 if (S_ISDIR(st.st_mode)) {
 if (stack_len == stack_cap) {
 int new_cap = stack_cap * 2;
 char **grown = (char **)realloc(dir_stack, (size_t)new_cap * sizeof(char *));
 if (!grown) {
 fprintf(stderr, "warning: dir_stack realloc OOM, skipping subtree under %s\n", fullpath);
 continue;
 }
 dir_stack = grown;
 stack_cap = new_cap;
 }
 dir_stack[stack_len] = strdup(fullpath);
 if (!dir_stack[stack_len]) continue; /* don't bump on OOM */
 stack_len++;
 } else if (S_ISREG(st.st_mode) && should_include(fullpath)) {
 file_list_add(list, fullpath);
 }
 }
 closedir(dir);
 free(dirpath);
 }
 free(dir_stack);
}

static double ms_since(struct timespec start) {
 struct timespec now;
 clock_gettime(CLOCK_MONOTONIC, &now);
 return (now.tv_sec - start.tv_sec) * 1000.0 + (now.tv_nsec - start.tv_nsec) / 1e6;
}

static long get_peak_rss_bytes(void) {
 struct rusage ru;
 getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
 return ru.ru_maxrss;
#else
 return ru.ru_maxrss * 1024;
#endif
}

static long get_current_rss_bytes(void) {
#if defined(__APPLE__)
 struct mach_task_basic_info info;
 mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
 kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
 (task_info_t)&info, &count);
 if (kr == KERN_SUCCESS) return (long)info.resident_size;
 return -1;
#elif defined(__linux__)
 FILE *f = fopen("/proc/self/status", "r");
 if (!f) return -1;
 char line[256];
 long rss_kb = -1;
 while (fgets(line, sizeof(line), f)) {
 if (sscanf(line, "VmRSS: %ld kB", &rss_kb) == 1) break;
 }
 fclose(f);
 /* return -1 on parse failure, not -1024.
 * rss_kb stays -1 if no VmRSS line was found. */
 return rss_kb > 0 ? rss_kb * 1024 : -1;
#else
 return -1;
#endif
}

/* Fixed query set shared between the in-memory dump (this file) and the
 * standalone loadquery tool, so their outputs can be diffed byte-for-byte. */
static const char *const DUMP_QUERIES[] = {
 "gpu display drivers",
 "user mode scheduling",
 "pcie ethernet code",
 "memory allocation pages",
 "file system inode",
 "network socket buffer",
 "interrupt handler irq",
 "crypto aes cipher",
};
#define DUMP_QUERY_COUNT ((int)(sizeof(DUMP_QUERIES) / sizeof(DUMP_QUERIES[0])))

/* Deterministic, process-independent dump of query results: result lines go to
 * stdout (idx + score to 6 dp, so two processes can be diffed), per-query timing
 * goes to stderr. Identical logic lives in loadquery.c. */
static void dump_query_results(const fce_sem_corpus_t *corp) {
 printf("=== QUERY DUMP (vocab=%d docs=%d) ===\n",
 fce_sem_corpus_token_count(corp), fce_sem_corpus_doc_count(corp));
 for (int i = 0; i < DUMP_QUERY_COUNT; i++) {
 const char *q = DUMP_QUERIES[i];
 fce_sem_ranked_t fr[15], br[15];
 uint32_t fn = 0, bn = 0;
 struct timespec ts;
 clock_gettime(CLOCK_MONOTONIC, &ts);
 fce_sem_search_query(corp, q, 15, fr, &fn, NULL);
 double fast_ms = ms_since(ts);
 clock_gettime(CLOCK_MONOTONIC, &ts);
 fce_sem_search_query_bruteforce(corp, q, 15, br, &bn);
 double brute_ms = ms_since(ts);

 printf("Q fast %s\n", q);
 for (uint32_t j = 0; j < fn; j++) printf(" %u %.6f\n", fr[j].index, fr[j].score);
 printf("Q brute %s\n", q);
 for (uint32_t j = 0; j < bn; j++) printf(" %u %.6f\n", br[j].index, br[j].score);
 fprintf(stderr, " query %-26s fast=%6.3f ms brute=%6.3f ms\n", q, fast_ms, brute_ms);
 }
}

int main(int argc, char **argv) {
 setbuf(stdout, NULL); /* Disable buffering for incremental output */
 if (argc < 2) {
 fprintf(stderr, "Usage: %s <directory> [chunk_size] [--brute-only] [--sparse[=N]] [--save-load[=path]]\n", argv[0]);
 return 1;
 }
 const char *root_dir = argv[1];
 int chunk_size = DEFAULT_CHUNK_SIZE;
 int brute_only = 0;
 int sparse_nnz = 0;
 int save_load = 0;
 const char *save_load_path = NULL;
 int dump_queries = 0;
 for (int i = 2; i < argc; i++) {
 if (strcmp(argv[i], "--brute-only") == 0) {
 brute_only = 1;
 } else if (strcmp(argv[i], "--sparse") == 0) {
 sparse_nnz = 32; /* default top-32 NNZ */
 } else if (strncmp(argv[i], "--sparse=", 9) == 0) {
 sparse_nnz = atoi(argv[i] + 9);
 if (sparse_nnz <= 0) sparse_nnz = 32;
 } else if (strcmp(argv[i], "--save-load") == 0) {
 save_load = 1;
 } else if (strncmp(argv[i], "--save-load=", 12) == 0) {
 save_load = 1;
 save_load_path = argv[i] + 12;
 } else if (strcmp(argv[i], "--dump-queries") == 0) {
 dump_queries = 1;
 } else {
 chunk_size = atoi(argv[i]);
 if (chunk_size <= 0) chunk_size = DEFAULT_CHUNK_SIZE;
 }
 }

 struct timespec t_total, t0, t1;
 clock_gettime(CLOCK_MONOTONIC, &t_total);

 printf("fast-code-embed memory + query benchmark\n");
 printf("================================================\n");
 printf("Directory: %s\n", root_dir);
 printf("Chunk size: %d bytes\n", chunk_size);
 if (sparse_nnz > 0)
 printf("Mode: sparse (nnz=%d)\n\n", sparse_nnz);
 else if (brute_only)
 printf("Mode: brute-only\n\n");
 else
 printf("Mode: normal\n\n");

 /* ── 1. Walk directory ────────────────────────────────────── */
 clock_gettime(CLOCK_MONOTONIC, &t0);
 file_list_t files = {0};
 walk_dir(root_dir, &files);
 printf(" Walk directory: %8.1f ms (%d files)\n", ms_since(t0), files.count);

 if (files.count == 0) {
 printf("No source files found.\n");
 free(files.paths);
 return 0;
 }

 /* ── 2. Build corpus ─────────────────────────────────────── */
 clock_gettime(CLOCK_MONOTONIC, &t0);
 char **all_tokens = (char **)malloc((size_t)BATCH_SIZE * MAX_TOKENS_PER_CHUNK * sizeof(char *));
 int *token_counts = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));
 if (!all_tokens || !token_counts) {
 fprintf(stderr, "malloc failed for tokens\n");
 free(all_tokens);
 free(token_counts);
 return 1;
 }
 int batch_used = 0, total_chunks = 0, files_processed = 0;

 int doc_path_cap = 1000000;
 char **doc_paths = (char **)malloc(doc_path_cap * sizeof(char *));
 if (!doc_paths) {
 fprintf(stderr, "malloc failed for doc_paths\n");
 free(all_tokens);
 free(token_counts);
 return 1;
 }
 int doc_path_count = 0;

 fce_sem_corpus_t *corp = fce_sem_corpus_new();
 if (!corp) {
 fprintf(stderr, "fce_sem_corpus_new failed (OOM)\n");
 free(doc_paths);
 free(all_tokens);
 free(token_counts);
 for (int i = 0; i < files.count; i++) free(files.paths[i]);
 free(files.paths);
 return 1;
 }

 for (int f = 0; f < files.count; f++) {
 size_t len;
 char *content = read_file(files.paths[f], &len);
 if (!content) continue;
 files_processed++;
 for (size_t offset = 0; offset < len; ) {
 /* Find next } at or after offset+chunk_size for semantic split. */
 size_t end = offset + (size_t)chunk_size;
 if (end < len) {
 size_t found = 0;
 for (size_t i = end; i < len; i++) {
 if (content[i] == '}') { found = i + 1; break; }
 }
 end = found ? found : len; /* no } left → take rest of file */
 } else {
 end = len; /* last chunk */
 }
 size_t chunk_len = end - offset;
 char *chunk = (char *)malloc(chunk_len + 1);
 if (!chunk) { offset = end; continue; }
 memcpy(chunk, content + offset, chunk_len);
 chunk[chunk_len] = '\0';
 char *tok_buf[MAX_TOKENS_PER_CHUNK];
 int ntok = fce_sem_tokenize(chunk, tok_buf, MAX_TOKENS_PER_CHUNK);
 free(chunk);

 /* skip zero-token chunks — they are rejected by
 * add_docs_batch (doc_map[d] = -1) so recording a doc_paths entry
 * would misalign the index mapping. */
 if (ntok == 0) { offset = end; continue; }

 int base = batch_used * MAX_TOKENS_PER_CHUNK;
 for (int t = 0; t < ntok; t++) all_tokens[base + t] = tok_buf[t];
 token_counts[batch_used] = ntok;

 if (doc_path_count < doc_path_cap) {
 doc_paths[doc_path_count++] = files.paths[f];
 }

 batch_used++;
 total_chunks++;
 offset = end;
 if (batch_used >= BATCH_SIZE) {
 /* track how many docs the batch actually
 * accepted so doc_paths stays aligned with the corpus doc index.
 * Rejected docs (vocab cap, doc cap, OOM rollback) don't appear in
 * the corpus, so their path entries must be removed. */
 int before = fce_sem_corpus_doc_count(corp);
 fce_sem_corpus_add_docs_batch(corp, all_tokens, token_counts, batch_used, MAX_TOKENS_PER_CHUNK, NULL);
 int accepted = fce_sem_corpus_doc_count(corp) - before;
 if (accepted < batch_used && doc_path_count >= (batch_used - accepted)) {
 doc_path_count -= (batch_used - accepted);
 }
 for (int i = 0; i < batch_used; i++) {
 int base2 = i * MAX_TOKENS_PER_CHUNK;
 for (int t = 0; t < token_counts[i]; t++) free(all_tokens[base2 + t]);
 }
 batch_used = 0;
 }
 }
 free(content);
 }
 if (batch_used > 0) {
 int before = fce_sem_corpus_doc_count(corp);
 fce_sem_corpus_add_docs_batch(corp, all_tokens, token_counts, batch_used, MAX_TOKENS_PER_CHUNK, NULL);
 int accepted = fce_sem_corpus_doc_count(corp) - before;
 if (accepted < batch_used && doc_path_count >= (batch_used - accepted)) {
 doc_path_count -= (batch_used - accepted);
 }
 for (int i = 0; i < batch_used; i++) {
 int base = i * MAX_TOKENS_PER_CHUNK;
 for (int t = 0; t < token_counts[i]; t++) free(all_tokens[base + t]);
 }
 }
 free(all_tokens);
 free(token_counts);
 printf(" Read + chunk + tokenize: %8.1f ms (%d chunks from %d files)\n",
 ms_since(t0), total_chunks, files_processed);

 /* ── 3. Finalize corpus ─────────────────────────────────────── */
 if (brute_only) setenv("FCE_SEM_SKIP_INV_INDEX", "1", 1);
 if (sparse_nnz > 0) fce_sem_corpus_set_sparse(corp, sparse_nnz);
 long rss_before_finalize = get_current_rss_bytes();
 clock_gettime(CLOCK_MONOTONIC, &t0);
 fce_sem_corpus_finalize(corp);
 double build_ms = ms_since(t0);
 long rss_after_finalize = get_current_rss_bytes();
 long peak_rss = get_peak_rss_bytes();

 printf(" Corpus finalize: %8.1f ms\n", build_ms);
 fflush(stdout);
 printf("\n");
 printf(" ── Memory ──────────────────────────────────\n");
 printf(" RSS before finalize: %8.1f GB\n", rss_before_finalize / 1073741824.0);
 printf(" RSS after finalize: %8.1f GB\n", rss_after_finalize / 1073741824.0);
 printf(" Peak RSS: %8.1f GB\n", peak_rss / 1073741824.0);

 int vocab = fce_sem_corpus_token_count(corp);
 int ndocs = fce_sem_corpus_doc_count(corp);
 printf("\n");
 printf(" ── Corpus ──────────────────────────────────\n");
 printf(" Vocabulary: %d tokens\n", vocab);
 printf(" Documents: %d\n", ndocs);

 /* ── 3a. Deterministic query dump (for cross-process parity check) ──
  * Saves the cache if requested, dumps in-memory query results to stdout,
  * then exits before the timing benchmark so the output is diff-friendly. */
 if (dump_queries) {
 if (save_load) {
 char dpath[1024];
 const char *cpath = save_load_path;
 if (!cpath) {
 const char *dir = getenv("TMPDIR");
 if (!dir || !*dir) dir = "/tmp";
 size_t dl = strlen(dir);
 const char *sep = (dl > 0 && dir[dl - 1] == '/') ? "" : "/";
 snprintf(dpath, sizeof(dpath), "%s%sfce_bench_corpus.fce", dir, sep);
 cpath = dpath;
 }
 int rc = fce_sem_corpus_save(corp, cpath,
 (const char *const *)doc_paths, doc_path_count);
 if (rc != 0) {
 /* Save failed: emitting the in-memory dump anyway would let a
  * caller diff it against a stale cache from a previous run and
  * believe cross-process load parity held. Fail loudly instead,
  * and unlink the destination so no stale cache is reused. */
 fprintf(stderr, "ERROR: save of %s failed (rc=%d); not dumping queries\n",
 cpath, rc);
 remove(cpath);
 free(doc_paths);
 for (int i = 0; i < files.count; i++) free(files.paths[i]);
 free(files.paths);
 fce_sem_corpus_free(corp);
 return 1;
 }
 fprintf(stderr, "SAVED %s (rc=%d)\n", cpath, rc);
 }
 dump_query_results(corp);
 free(doc_paths);
 for (int i = 0; i < files.count; i++) free(files.paths[i]);
 free(files.paths);
 fce_sem_corpus_free(corp);
 return 0;
 }

 /* ── 3b. Save / load (mmap cache) ─────────────────────────── */
 int parity_failed = 0;
 if (save_load) {
 char default_path[1024];
 const char *cpath = save_load_path;
 if (!cpath) {
 const char *dir = getenv("TMPDIR");
 if (!dir || !*dir) dir = "/tmp";
 size_t dl = strlen(dir);
 const char *sep = (dl > 0 && dir[dl - 1] == '/') ? "" : "/";
 snprintf(default_path, sizeof(default_path), "%s%sfce_bench_corpus.fce", dir, sep);
 cpath = default_path;
 }
 printf("\n");
 printf(" ── Save / Load (mmap cache) ────────────────\n");
 printf(" Path: %s\n", cpath);
 clock_gettime(CLOCK_MONOTONIC, &t0);
 int save_rc = fce_sem_corpus_save(corp, cpath,
 (const char *const *)doc_paths, doc_path_count);
 double save_ms = ms_since(t0);
 if (save_rc != 0) {
 printf(" Save: FAILED\n");
 } else {
 struct stat st;
 double fsz = (stat(cpath, &st) == 0) ? (double)st.st_size : 0.0;
 printf(" Save: %8.1f ms (%.2f GB, %.0f MB/s)\n",
 save_ms, fsz / 1073741824.0, fsz / 1e6 / (save_ms / 1000.0));

 clock_gettime(CLOCK_MONOTONIC, &t0);
 fce_sem_corpus_t *ldc = fce_sem_corpus_load(cpath);
 double load_ms = ms_since(t0);
 if (!ldc) {
 printf(" Load: FAILED\n");
 } else {
 printf(" Load: %8.1f ms (%.0f MB/s, %.1fx faster than finalize)\n",
 load_ms, fsz / 1e6 / (load_ms / 1000.0), build_ms / load_ms);
 printf(" Loaded: vocab=%d docs=%d labels=%d\n",
 fce_sem_corpus_token_count(ldc), fce_sem_corpus_doc_count(ldc),
 fce_sem_corpus_doc_label_count(ldc));

 /* Confirm query results match the in-memory corpus exactly. */
 const char *pq[] = {"gpu display drivers", "memory allocation pages", "file system inode"};
 int npq = (int)(sizeof(pq) / sizeof(pq[0]));
 int identical = 0;
 for (int i = 0; i < npq; i++) {
 fce_sem_ranked_t a[10], b[10];
 uint32_t na = 0, nb = 0;
 fce_sem_search_query(corp, pq[i], 10, a, &na, NULL);
 fce_sem_search_query(ldc, pq[i], 10, b, &nb, NULL);
 int same = (na == nb);
 for (uint32_t j = 0; same && j < na; j++) {
 if (a[j].index != b[j].index || fabsf(a[j].score - b[j].score) > 1e-6f) same = 0;
 }
 identical += same;
 }
 printf(" Query parity vs in-memory corpus: %s (%d/%d identical)\n",
 identical == npq ? "OK" : "MISMATCH", identical, npq);

 if (identical != npq) {
 /* This is a release-validation tool: a save/load that visibly
  * fails parity must not have its later timings presented as if the
  * loaded representation were trustworthy. Drop the loaded corpus,
  * keep the in-memory one, and propagate a non-zero exit code. */
 parity_failed = 1;
 fce_sem_corpus_free(ldc);
 printf(" PARITY MISMATCH: discarding loaded corpus; benchmarks below "
 "run on the in-memory corpus; exit code will be non-zero.\n");
 } else {
 /* Continue the query benchmarks on the mmap-loaded corpus. Doc paths
  * are now served zero-copy from the loaded corpus's label table. */
 fce_sem_corpus_free(corp);
 corp = ldc;
 printf(" (query benchmarks below run on the mmap-loaded corpus)\n");
 }
 }
 }
 fflush(stdout);
 }

 /* ── 4. Query benchmarks ──────────────────────────────────── */
 printf("\n");
 printf(" ── Query Benchmarks ───────────────────────\n");
 fflush(stdout);

 fce_sem_config_t fast_cfg = fce_sem_get_config();
 fce_sem_config_t tfidf_cfg = fce_sem_get_config();
 if (brute_only) {
 fast_cfg.query_mode = FCE_QUERY_BRUTE;
 tfidf_cfg.query_mode = FCE_QUERY_BRUTE;
 }

 const char *queries[] = {
 "gpu display drivers",
 "user mode scheduling",
 "pcie ethernet code",
 "memory allocation pages",
 "file system inode",
 };
 int nqueries = sizeof(queries) / sizeof(queries[0]);

 int top_ks[] = {10, 15};
 int ntop_ks = sizeof(top_ks) / sizeof(top_ks[0]);

 for (int ki = 0; ki < ntop_ks; ki++) {
 int top_k = top_ks[ki];
 printf("\n top_k = %d\n", top_k);
 double total_fast_ms = 0;
 double total_tfidf_ms = 0;
 double total_brute_ms = 0;

 for (int qi = 0; qi < nqueries; qi++) {
 const char *qstr = queries[qi];

 fce_sem_ranked_t *fast_results = (fce_sem_ranked_t *)malloc(top_k * sizeof(fce_sem_ranked_t));
 fce_sem_ranked_t *tfidf_results = (fce_sem_ranked_t *)malloc(top_k * sizeof(fce_sem_ranked_t));
 fce_sem_ranked_t *brute_results = (fce_sem_ranked_t *)malloc(top_k * sizeof(fce_sem_ranked_t));
 /* check all three malloc results before
 * passing them to search functions which write directly into them. */
 if (!fast_results || !tfidf_results || !brute_results) {
 fprintf(stderr, "OOM allocating result buffers, skipping query %d\n", qi);
 free(fast_results); free(tfidf_results); free(brute_results);
 continue;
 }
 uint32_t fast_count = 0, tfidf_count = 0, brute_count = 0;

 /* Warm up all paths */
 fce_sem_search_query(corp, qstr, top_k, fast_results, &fast_count, &fast_cfg);
 fce_sem_search_query_tfidf(corp, qstr, top_k, tfidf_results, &tfidf_count, &tfidf_cfg);
 fce_sem_search_query_bruteforce(corp, qstr, top_k, brute_results, &brute_count);

 int ncand = fce_sem_search_candidate_count(corp, qstr);

 int iters = 20;

 /* Benchmark fast path (inverted index + IDF sum) */
 clock_gettime(CLOCK_MONOTONIC, &t1);
 for (int iter = 0; iter < iters; iter++)
 fce_sem_search_query(corp, qstr, top_k, fast_results, &fast_count, &fast_cfg);
 double fast_ms = ms_since(t1) / iters;
 total_fast_ms += fast_ms;

 /* Benchmark TF-IDF hybrid path */
 clock_gettime(CLOCK_MONOTONIC, &t1);
 for (int iter = 0; iter < iters; iter++)
 fce_sem_search_query_tfidf(corp, qstr, top_k, tfidf_results, &tfidf_count, &tfidf_cfg);
 double tfidf_ms = ms_since(t1) / iters;
 total_tfidf_ms += tfidf_ms;

 /* Benchmark brute-force path */
 clock_gettime(CLOCK_MONOTONIC, &t1);
 for (int iter = 0; iter < iters; iter++)
 fce_sem_search_query_bruteforce(corp, qstr, top_k, brute_results, &brute_count);
 double brute_ms = ms_since(t1) / iters;
 total_brute_ms += brute_ms;

 /* Count overlaps with brute-force.
 * iterate only over returned counts, not top_k,
 * to avoid reading uninitialized slots when a path returns
 * fewer results. */
 int overlap_fast = 0, overlap_tfidf = 0;
 for (int i = 0; i < (int)fast_count; i++) {
 for (int j = 0; j < (int)brute_count; j++) {
 if (fast_results[i].index == brute_results[j].index) { overlap_fast++; break; }
 }
 }
 for (int i = 0; i < (int)tfidf_count; i++) {
 for (int j = 0; j < (int)brute_count; j++) {
 if (tfidf_results[i].index == brute_results[j].index) { overlap_tfidf++; break; }
 }
 }

 printf("\n %-30s fast=%5.0fms tfidf=%5.0fms brute=%5.0fms cands=%d\n",
 qstr, fast_ms, tfidf_ms, brute_ms, ncand);
 printf(" overlap with brute: fast=%d/%d tfidf=%d/%d\n",
 overlap_fast, top_k, overlap_tfidf, top_k);

 /* Print side-by-side results (top 5). */
 int show = (top_k < 5) ? top_k : 5;
 printf(" %-4s %-28s %-4s %-28s %-4s %-28s\n",
 "Fast", "", "TFIDF", "", "Brute", "");
 for (int r = 0; r < show; r++) {
 const char *ffname = "?", *tfname = "?", *bfname = "?";
 float fscore = 0, tfscore = 0, bscore = 0;

 if (r < (int)fast_count) {
 uint32_t idx = fast_results[r].index;
 const char *p = (idx < (uint32_t)doc_path_count) ? doc_paths[idx] : "?";
 ffname = strrchr(p, '/'); ffname = ffname ? ffname + 1 : p;
 fscore = fast_results[r].score;
 }
 if (r < (int)tfidf_count) {
 uint32_t idx = tfidf_results[r].index;
 const char *p = (idx < (uint32_t)doc_path_count) ? doc_paths[idx] : "?";
 tfname = strrchr(p, '/'); tfname = tfname ? tfname + 1 : p;
 tfscore = tfidf_results[r].score;
 }
 if (r < (int)brute_count) {
 uint32_t idx = brute_results[r].index;
 const char *p = (idx < (uint32_t)doc_path_count) ? doc_paths[idx] : "?";
 bfname = strrchr(p, '/'); bfname = bfname ? bfname + 1 : p;
 bscore = brute_results[r].score;
 }

 int match_bf = (r < (int)fast_count && r < (int)brute_count &&
 fast_results[r].index == brute_results[r].index);
 int match_tf = (r < (int)tfidf_count && r < (int)brute_count &&
 tfidf_results[r].index == brute_results[r].index);
 printf(" [%d]%.3f %-22s [%d]%.3f %-22s [%d]%.3f %-22s%s%s\n",
 r, fscore, ffname, r, tfscore, tfname, r, bscore, bfname,
 match_bf ? " <<" : "", match_tf ? " <<" : "");
 }

 free(fast_results);
 free(tfidf_results);
 free(brute_results);
 }
 printf("\n Average: fast=%5.0fms tfidf=%5.0fms brute=%5.0fms\n",
 total_fast_ms / nqueries, total_tfidf_ms / nqueries, total_brute_ms / nqueries);
 }

 /* ── Summary ─────────────────────────────────────────────── */
 double total_ms = ms_since(t_total);
 printf("\n ── Final Summary ─────────────────────────────\n");
 printf(" Total time: %.1f ms\n", total_ms);
 printf(" Peak RSS: %.1f GB\n", peak_rss / 1073741824.0);
 printf(" Post-build RSS: %.1f GB\n", rss_after_finalize / 1073741824.0);

 /* Cleanup */
 free(doc_paths);
 for (int i = 0; i < files.count; i++) free(files.paths[i]);
 free(files.paths);
 fce_sem_corpus_free(corp);
 return parity_failed ? 1 : 0;
}
