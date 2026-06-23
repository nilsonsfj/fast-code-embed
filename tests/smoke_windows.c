/* Windows cross-compile smoke link.
 *
 * This is NOT a functional test — it is never executed (the build artifact is a
 * Windows PE binary produced by a mingw-w64 cross toolchain on a Linux host).
 * Its sole purpose is to force the public API symbols and the embedded vector
 * blob to resolve at link time, so the `windows-cross` build catches any
 * _WIN32 code path that fails to compile or leaves a symbol undefined. */
#include "semantic/semantic.h"
#include "version.h"
#include <stdio.h>

int main(void) {
    printf("fast-code-embed %s, active dim %d\n",
           fce_version(), fce_sem_active_dim());
    fce_sem_corpus_t *c = fce_sem_corpus_new();
    if (c) fce_sem_corpus_free(c);
    return 0;
}
