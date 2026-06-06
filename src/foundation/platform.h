/*
 * platform.h — OS abstractions.
 *
 * Provides cross-platform wrappers for:
 *   - Memory-mapped files (mmap / VirtualAlloc)
 *   - High-resolution monotonic clock
 *   - CPU core count
 *   - File existence check
 */
#ifndef FCE_PLATFORM_H
#define FCE_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

/* ── Safe memory ──────────────────────────────────────────────── */

/* realloc_or_free: frees old pointer on failure instead of leaking it.
 * Returns NULL on allocation failure (old memory is freed).
 *
 * WARNING: This is only safe for arrays whose elements do NOT own additional
 * heap memory. If `ptr` points to an array of heap-allocated pointers
 * (e.g. char *[]) and the realloc fails, every string the old array owned
 * is leaked — realloc_or_free frees the array itself but not the strings
 * each element points at. For such cases, use a manual
 *     tmp = realloc(ptr, new_size);
 *     if (!tmp) { / * keep ptr, free individual elements * / }
 * pattern instead. The safe_grow macro below inherits this restriction. */
static inline void *realloc_or_free(void *ptr, size_t size) {
    enum { REALLOC_OR_FREE_MIN = 1 };
    if (size == 0) {
        size = REALLOC_OR_FREE_MIN;
    }
    void *tmp = realloc(ptr, size);
    if (!tmp) {
        free(ptr);
    }
    return tmp;
}

/* Safe free: frees and NULLs a pointer to prevent double-free / use-after-free.
 * Use via the safe_free() macro so the caller's pointer is actually cleared. */
static inline void safe_free_impl(void **pp) {
    if (pp && *pp) {
        free(*pp);
        *pp = NULL;
    }
}
#define safe_free(ptr) safe_free_impl((void **)(void *)&(ptr))

/* Safe const string free: frees a const char* and NULLs it.
 * Casts away const so callers don't repeat the (void *) dance. */
static inline void safe_str_free(const char **sp) {
    if (sp && *sp) {
        free((void *)*sp);
        *sp = NULL;
    }
}

/* Safe buffer free: frees a heap array and zeros its element count.
 * Use for dynamic arrays paired with a size_t count. */
static inline void safe_buf_free_impl(void **buf, size_t *count) {
    if (buf && *buf) {
        free(*buf);
        *buf = NULL;
    }
    if (count) {
        *count = 0;
    }
}
#define safe_buf_free(buf, countp) safe_buf_free_impl((void **)(void *)&(buf), (countp))

/* Safe grow: doubles capacity and reallocs when count reaches cap.
 * Note: uses realloc_or_free which frees the old buffer on failure, so this is
 * only appropriate for arrays whose elements don't own additional heap memory.
 * For arrays of heap-allocated pointers (e.g. char *[]), use a manual
 * realloc+cleanup pattern instead — see warning on realloc_or_free above.
 * Usage: safe_grow(arr, count, cap, growth_factor)
 * After the call, arr is the new buffer (NULL on OOM). */
#define safe_grow(arr, n, cap, factor)                                                             \
    do {                                                                                           \
        if ((size_t)(n) >= (size_t)(cap)) {                                                        \
            (cap) *= (factor);                                                                     \
            (arr) = realloc_or_free((arr), (size_t)(cap) * sizeof(*(arr)));                        \
        }                                                                                          \
    } while (0)

/* ── Memory mapping ────────────────────────────────────────────── */

/* Map a file read-only into memory. Returns NULL on error.
 * *out_size is set to the file size. */
void *fce_mmap_read(const char *path, size_t *out_size);

/* Unmap a previously mapped region. */
void fce_munmap(void *addr, size_t size);

/* ── Timing ────────────────────────────────────────────────────── */

/* Monotonic nanosecond timestamp (for elapsed time measurement). */
uint64_t fce_now_ns(void);

/* Monotonic millisecond timestamp. */
uint64_t fce_now_ms(void);

/* ── System info ───────────────────────────────────────────────── */

/* Number of available CPU cores. */
int fce_nprocs(void);

/* System topology: core types and RAM (only fields with production consumers). */
typedef struct {
    int total_cores;  /* hw.ncpu (all cores) */
    int perf_cores;   /* P-cores (Apple) or total_cores (others) */
    size_t total_ram; /* total physical RAM in bytes */
} fce_system_info_t;

/* Query system information. Results are cached after first call. */
fce_system_info_t fce_system_info(void);

/* Recommended worker count for parallel indexing.
 * initial=true:  all cores (user is waiting for initial index)
 * initial=false: max(1, perf_cores-1) (leave headroom for user apps) */
int fce_default_worker_count(bool initial);

/* ── Environment variables ──────────────────────────────────────── */

/* Thread-safe getenv: copies the value into a caller-provided buffer.
 * Returns buf on success, or fallback if the variable is unset.
 * Returns NULL when the variable is unset and fallback is NULL.
 *
 * THREAD-SAFETY (review 0002 §3.7): this function reads the process `environ`
 * array directly (not via glibc getenv), so it's safe to call concurrently
 * with other fce_safe_getenv calls. It is NOT safe to call concurrently
 * with setenv/putenv that may reallocate the environ array itself.  Safe for
 * init paths and infrequent calls.  NOT safe for hot concurrent paths —
 * cache the result at init via pthread_once instead.  See FCE_BRUTE_WORKERS
 * / FCE_STACK_SIZE in semantic.c / worker_pool.c for examples. */
const char *fce_safe_getenv(const char *name, char *buf, size_t buf_sz, const char *fallback);

/* ── Home directory ─────────────────────────────────────────────── */

/* Cross-platform home directory: tries HOME first, then USERPROFILE (Windows).
 * Returns NULL when neither is set. */
const char *fce_get_home_dir(void);

/* ── App config directories ────────────────────────────────────── */

/* Cross-platform app config directory (static buffer, not thread-safe).
 * Windows: %APPDATA% (e.g. C:/Users/.../AppData/Roaming)
 * macOS:   $HOME (callers append Library/Application Support/...)
 * Linux:   $XDG_CONFIG_HOME or ~/.config */
const char *fce_app_config_dir(void);

/* Windows: %LOCALAPPDATA% (e.g. C:/Users/.../AppData/Local)
 * macOS/Linux: same as fce_app_config_dir(). */
const char *fce_app_local_dir(void);

/* ── Cache directory ────────────────────────────────────────────── */

/* Resolve the database cache directory. All project indexes are stored here.
 * Priority: FCE_CACHE_DIR env var > ~/.cache/fast-code-embed (default).
 * Returns static buffer or NULL if home is unavailable. */
const char *fce_resolve_cache_dir(void);

/* ── File system ───────────────────────────────────────────────── */

/* Check if a path exists. */
bool fce_file_exists(const char *path);

/* Check if path is a directory. */
bool fce_is_dir(const char *path);

/* Get file size. Returns -1 on error. */
int64_t fce_file_size(const char *path);

/* Normalize path separators to forward slashes (in-place).
 * On Windows, converts backslashes to forward slashes.
 * On POSIX, this is a no-op. Returns the input pointer.
 * WARNING: Mutates the buffer in place — only call on heap/stack allocated
 * mutable buffers, never on string literals or read-only memory. */
char *fce_normalize_path_sep(char *path);

#endif /* FCE_PLATFORM_H */
