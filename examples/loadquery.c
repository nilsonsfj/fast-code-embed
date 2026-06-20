/* loadquery.c — clean-process verification tool.
 *
 * Loads a corpus cache via fce_sem_corpus_load (zero-copy mmap) WITHOUT any
 * build step, times the load and each query, and dumps deterministic query
 * results to stdout in the exact same format as bench_mem_query's
 * --dump-queries, so the two can be diffed byte-for-byte. Timings go to stderr.
 */
/* clock_gettime / CLOCK_MONOTONIC / struct timespec are POSIX; request them
 * explicitly so this example builds under strict ISO C (e.g. gcc -std=c11)
 * where they are otherwise hidden. Must precede any system header. */
#if defined(__linux__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#include "semantic/semantic.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double ms_since(struct timespec s) {
    struct timespec n;
    clock_gettime(CLOCK_MONOTONIC, &n);
    return (double)(n.tv_sec - s.tv_sec) * 1000.0 + (double)(n.tv_nsec - s.tv_nsec) / 1e6;
}

/* MUST match bench_mem_query.c's DUMP_QUERIES exactly. */
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

/* MUST match bench_mem_query.c's dump_query_results exactly (stdout side). */
static void dump_query_results(const fce_sem_corpus_t *corp) {
    printf("=== QUERY DUMP (vocab=%d docs=%d) ===\n",
           fce_sem_corpus_token_count(corp), fce_sem_corpus_doc_count(corp));
    for (int i = 0; i < DUMP_QUERY_COUNT; i++) {
        const char *q = DUMP_QUERIES[i];
        fce_sem_ranked_t fr[15], br[15];
        uint32_t fn = 0, bn = 0;
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        fce_sem_search_query_fast(corp, q, 15, fr, &fn);
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
    if (argc < 2) {
        fprintf(stderr, "usage: %s <cachefile>\n", argv[0]);
        return 2;
    }
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    fce_sem_corpus_t *corp = fce_sem_corpus_load(argv[1]);
    double load_ms = ms_since(t0);
    if (!corp) {
        fprintf(stderr, "LOAD FAILED: %s (missing/corrupt/incompatible)\n", argv[1]);
        return 1;
    }
    fprintf(stderr, "LOAD %s: %.3f ms (vocab=%d docs=%d labels=%d)\n",
            argv[1], load_ms,
            fce_sem_corpus_token_count(corp),
            fce_sem_corpus_doc_count(corp),
            fce_sem_corpus_doc_label_count(corp));
    dump_query_results(corp);
    fce_sem_corpus_free(corp);
    return 0;
}
