/* * worker_pool.c — Parallel-for dispatch with pthreads.
 *
 * Uses pthreads with configurable stacks and atomic work-stealing index.
 * Each worker pulls indices from a shared atomic counter — zero
 * contention, natural load balancing across heterogeneous cores. */
#include "pipeline/worker_pool.h"
#include "foundation/constants.h"

enum { WP_TRUE = 1, WP_MIN = 1, WP_STEP = 1 };
#include "foundation/platform.h"
#include "foundation/compat_thread.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <errno.h>

/* Embedding workloads need far less stack than AST recursion (tree-sitter).
 * Default 1 MB; override via FCE_STACK_SIZE env var (in bytes). */
#define FCE_DEFAULT_WORKER_STACK ((size_t)1 * FCE_SZ_1K * FCE_SZ_1K)

/* M1: cache FCE_STACK_SIZE at init.
 * fce_safe_getenv iterates environ directly and is NOT safe against
 * concurrent setenv/putenv. Read once, store the result. */
static size_t g_cached_stack_size = 0; /* 0 = not yet cached */
static fce_once_t g_stack_size_once = FCE_ONCE_INIT;
static void init_stack_size(void) {
 size_t stack_size = FCE_DEFAULT_WORKER_STACK;
 char env_buf[32];
 const char *env_val = fce_safe_getenv("FCE_STACK_SIZE", env_buf, sizeof(env_buf), "");
 if (env_val && env_val[0]) {
 char *endptr;
 errno = 0;
 unsigned long v = strtoul(env_val, &endptr, 10);
 if (errno == 0 && endptr != env_val && *endptr == '\0' &&
 v >= 65536 && v <= (unsigned long)64 * FCE_SZ_1K * FCE_SZ_1K) {
 stack_size = (size_t)v;
 }
 }
 g_cached_stack_size = stack_size;
}

/* ── Serial fallback ─────────────────────────────────────────────── */

static void run_serial(int count, fce_parallel_fn fn, void *ctx) {
 for (int i = 0; i < count; i++) {
 fn(i, ctx);
 }
}

/* ── Static-chunked parallel-for (F5) ──────────────────────────── */

typedef struct {
 fce_parallel_fn fn;
 void *ctx;
 int start;
 int end;
} static_chunk_arg_t;

static void *static_chunk_worker(void *arg) {
 static_chunk_arg_t *a = arg;
 for (int i = a->start; i < a->end; i++) {
 a->fn(i, a->ctx);
 }
 return NULL;
}

static void run_pthreads_static(int count, fce_parallel_fn fn, void *ctx, int nworkers) {
 /* Split [0, count) into nworkers+1 contiguous chunks (workers + main thread). */
 int total_workers = nworkers + 1;
 int chunk_size = count / total_workers;
 int remainder = count % total_workers;

 fce_thread_t *threads = (fce_thread_t *)malloc((size_t)nworkers * sizeof(fce_thread_t));
 static_chunk_arg_t *args = (static_chunk_arg_t *)malloc((size_t)total_workers * sizeof(static_chunk_arg_t));
 if (!threads || !args) {
 free(threads); free(args);
 run_serial(count, fn, ctx);
 return;
 }

 /* M1: use cached stack size. */
 fce_once(&g_stack_size_once, init_stack_size);
 size_t stack_size = g_cached_stack_size;

 /* Assign contiguous chunks to each worker. */
 int offset = 0;
 for (int i = 0; i < total_workers; i++) {
 int sz = chunk_size + (i < remainder ? 1 : 0);
 args[i] = (static_chunk_arg_t){ .fn = fn, .ctx = ctx, .start = offset, .end = offset + sz };
 offset += sz;
 }

 /* Spawn worker threads (all but the last chunk, which runs on main thread). */
 bool serial_fallback = false;
 for (int i = 0; i < nworkers; i++) {
 if (fce_thread_create(&threads[i], stack_size, static_chunk_worker, &args[i]) != 0) {
 /* Failed to create thread — run remaining chunks serially
 * (including the last chunk, which would normally run on main thread). */
 for (int j = i; j < total_workers; j++) {
 for (int k = args[j].start; k < args[j].end; k++) fn(k, ctx);
 }
 nworkers = i;
 serial_fallback = true;
 break;
 }
 }

 if (!serial_fallback) {
 /* Main thread runs the last chunk. */
 static_chunk_arg_t *last = &args[total_workers - 1];
 for (int i = last->start; i < last->end; i++) {
 fn(i, ctx);
 }
 }

 for (int i = 0; i < nworkers; i++) {
 fce_thread_join(&threads[i]);
 }

 free(threads);
 free(args);
}

/* ── pthreads backend ────────────────────────────────────────────── */

typedef struct {
 fce_parallel_fn fn;
 void *ctx;
 _Atomic int *next_idx;
 int count;
} pthread_worker_arg_t;

static void *pthread_worker(void *arg) {
 pthread_worker_arg_t *wa = arg;
 while (WP_TRUE) {
 int idx = atomic_fetch_add_explicit(wa->next_idx, WP_STEP, memory_order_relaxed);
 if (idx >= wa->count) {
 break;
 }
 wa->fn(idx, wa->ctx);
 }
 return NULL;
}

static void run_pthreads(int count, fce_parallel_fn fn, void *ctx, int nworkers) {
 _Atomic int next_idx = 0;

 pthread_worker_arg_t wa = {
 .fn = fn,
 .ctx = ctx,
 .next_idx = &next_idx,
 .count = count,
 };

 /* M1: use cached stack size. */
 fce_once(&g_stack_size_once, init_stack_size);
 size_t stack_size = g_cached_stack_size;

 fce_thread_t *threads = (fce_thread_t *)malloc((size_t)nworkers * sizeof(fce_thread_t));
 if (!threads) {
 run_serial(count, fn, ctx);
 return;
 }

 for (int i = 0; i < nworkers; i++) {
 if (fce_thread_create(&threads[i], stack_size, pthread_worker, &wa) != 0) {
 /* Failed to create thread — let remaining work run in main thread */
 nworkers = i;
 break;
 }
 }

 /* Main thread also participates */
 while (WP_TRUE) {
 int idx = atomic_fetch_add_explicit(&next_idx, WP_STEP, memory_order_relaxed);
 if (idx >= count) {
 break;
 }
 fn(idx, ctx);
 }

 for (int i = 0; i < nworkers; i++) {
 fce_thread_join(&threads[i]);
 }

 free(threads);
}

/* ── Public API ──────────────────────────────────────────────────── */

void fce_parallel_for(int count, fce_parallel_fn fn, void *ctx, fce_parallel_for_opts_t opts) {
 if (count <= 0 || !fn) {
 return;
 }

 /* Determine worker count */
 int nworkers = opts.max_workers;
 if (nworkers <= 0) {
 nworkers = fce_default_worker_count(true);
 }
 if (nworkers < WP_MIN) {
 nworkers = 1;
 }

 /* Serial fallback: single worker or trivially small workload */
 if (nworkers <= WP_MIN || count <= WP_MIN) {
 run_serial(count, fn, ctx);
 return;
 }

 /* P5: main thread participates, so spawn one fewer to avoid one idle runner. */
 run_pthreads(count, fn, ctx, nworkers - 1);
}

void fce_parallel_for_static(int count, fce_parallel_fn fn, void *ctx, fce_parallel_for_opts_t opts) {
 if (count <= 0 || !fn) {
 return;
 }

 int nworkers = opts.max_workers;
 if (nworkers <= 0) {
 nworkers = fce_default_worker_count(true);
 }
 if (nworkers < WP_MIN) {
 nworkers = 1;
 }

 if (nworkers <= WP_MIN || count <= WP_MIN) {
 run_serial(count, fn, ctx);
 return;
 }

 run_pthreads_static(count, fn, ctx, nworkers - 1);
}
