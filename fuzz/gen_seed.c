/* Seed-corpus generator for the cache-loader fuzzer.
 *
 * Writes one valid cache file (as produced by fce_sem_corpus_save) to the path
 * given on the command line. A structurally-valid seed lets libFuzzer reach the
 * deeper validation paths in fce_sem_corpus_load by mutating a real file rather
 * than starting from random noise that is rejected at the magic check.
 *
 * Usage: gen_seed <output-cache-path>
 */
#include "semantic/semantic.h"

#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <output-cache-path>\n", argv[0]);
        return 2;
    }

    fce_sem_corpus_t *c = fce_sem_corpus_new();
    if (!c) {
        fprintf(stderr, "corpus_new failed\n");
        return 1;
    }

    /* A small but real corpus: tokens that resolve against the embedded
     * pretrained table, so finalize produces non-empty enriched vectors. */
    static const char *docs[][3] = {
        {"handle", "request", "parse"},
        {"open", "file", "read"},
        {"read", "file", "buffer"},
        {"alloc", "memory", "buffer"},
        {"free", "memory", "page"},
    };
    int ndocs = (int)(sizeof(docs) / sizeof(docs[0]));
    for (int i = 0; i < ndocs; i++) {
        fce_sem_corpus_add_doc(c, docs[i], 3);
    }
    if (fce_sem_corpus_finalize(c) != 0) {
        fprintf(stderr, "finalize failed\n");
        fce_sem_corpus_free(c);
        return 1;
    }

    const char *labels[] = {"a.c", "b.c", "c.c", "d.c", "e.c"};
    int rc = fce_sem_corpus_save(c, argv[1], labels, ndocs);
    fce_sem_corpus_free(c);
    if (rc != 0) {
        fprintf(stderr, "save failed (%d)\n", rc);
        return 1;
    }
    printf("wrote seed cache: %s\n", argv[1]);
    return 0;
}
