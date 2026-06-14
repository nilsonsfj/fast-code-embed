/* * worker_pool.h — Generic parallel-for dispatch.
 *
 * Backend: pthreads with configurable stacks and atomic work-stealing index.
 * Each worker pulls from a shared counter — zero contention, natural
 * load balancing across heterogeneous cores (P/E on Apple Silicon).
 *
 * Serial fallback when count <= 1 or max_workers <= 1. */
#ifndef FCE_WORKER_POOL_H
#define FCE_WORKER_POOL_H

/* Worker callback: called once per iteration with index [0..count-1]. */
typedef void (*fce_parallel_fn)(int idx, void *ctx);

/* Options for parallel dispatch. */
typedef struct {
    int max_workers; /* 0 = auto-detect from fce_default_worker_count */
} fce_parallel_for_opts_t;

/* Dispatch `count` iterations of `fn(idx, ctx)` across worker threads.
 * Each index [0..count-1] is visited exactly once.
 * Blocks until all iterations complete.
 *
 * If count <= 0, this is a no-op.
 * If count <= 1 or workers <= 1, runs single-threaded. */
void fce_parallel_for(int count, fce_parallel_fn fn, void *ctx, fce_parallel_for_opts_t opts);

/* Static-chunked parallel-for: each worker gets a contiguous range [start, end).
 * No atomic operations in the hot loop — zero synchronization overhead.
 * Ideal for bandwidth-bound scans where work is uniform.
 *
 * If count <= 0, this is a no-op.
 * If count <= 1 or workers <= 1, runs single-threaded. */
void fce_parallel_for_static(int count, fce_parallel_fn fn, void *ctx, fce_parallel_for_opts_t opts);

#endif /* FCE_WORKER_POOL_H */
