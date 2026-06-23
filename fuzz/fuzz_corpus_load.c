/* mkstemp()/ftruncate() need a POSIX feature macro to be declared under strict
 * -std=c11 (glibc hides them otherwise, unlike Apple's headers). glibc only
 * exposes mkstemp at _POSIX_C_SOURCE >= 200809L, so request that level. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1
#endif

/* libFuzzer harness for fce_sem_corpus_load — the untrusted-cache parser.
 *
 * fce_sem_corpus_load mmaps a path and is designed to range-validate every
 * offset/length/count/index/float read from the file and reject anything
 * malformed rather than trust it. This harness continuously verifies that
 * design: each fuzz input is written to a temporary file and loaded; the only
 * contract under test is that no input — however corrupt or truncated — causes
 * a crash, out-of-bounds access, or leak (the latter two caught by the
 * ASan/UBSan instrumentation in the fuzzer build). A load that succeeds is
 * immediately freed.
 *
 * Build/run: make fuzz && ./build/fuzz/fuzz_corpus_load build/fuzz/corpus
 */
#include "semantic/semantic.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* One reusable temp file for the whole fuzzing run — libFuzzer drives this
 * function in a tight loop, so creating a fresh file per input would dominate
 * the runtime and churn the filesystem. The template lives under $TMPDIR (or
 * /tmp) so it works in sandboxed environments that only grant $TMPDIR. */
static int g_fd = -1;
static char g_path[4096];

static void ensure_tmp(void) {
    if (g_fd >= 0) {
        return;
    }
    const char *dir = getenv("TMPDIR");
    if (!dir || !*dir) {
        dir = "/tmp";
    }
    size_t dlen = strlen(dir);
    int has_slash = (dlen > 0 && dir[dlen - 1] == '/');
    int n = snprintf(g_path, sizeof(g_path), "%s%sfce_fuzz_cache_XXXXXX",
                     dir, has_slash ? "" : "/");
    if (n < 0 || (size_t)n >= sizeof(g_path)) {
        fprintf(stderr, "TMPDIR path too long\n");
        abort();
    }
    g_fd = mkstemp(g_path);
    if (g_fd < 0) {
        perror("mkstemp");
        abort();
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    ensure_tmp();

    /* Overwrite the temp file with this input. */
    if (ftruncate(g_fd, 0) != 0) {
        return 0;
    }
    if (lseek(g_fd, 0, SEEK_SET) != 0) {
        return 0;
    }
    size_t off = 0;
    while (off < size) {
        ssize_t w = write(g_fd, data + off, size - off);
        if (w <= 0) {
            return 0;
        }
        off += (size_t)w;
    }

    /* The loader opens and mmaps the path independently; the bytes written
     * above are visible to it via the page cache (same inode). */
    fce_sem_corpus_t *c = fce_sem_corpus_load(g_path);
    if (c) {
        fce_sem_corpus_free(c);
    }
    return 0;
}
