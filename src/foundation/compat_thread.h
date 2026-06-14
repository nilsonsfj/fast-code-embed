/* * compat_thread.h — Portable threading: pthreads on POSIX, Win32 threads on Windows.
 *
 * Provides: thread create/join, mutex, aligned allocation.
 * All have zero overhead on POSIX (thin inlines or macros). */
#ifndef FCE_COMPAT_THREAD_H
#define FCE_COMPAT_THREAD_H

#include <stddef.h>

/* ── Thread ───────────────────────────────────────────────────── */

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef struct {
 HANDLE handle;
} fce_thread_t;

#else /* POSIX */

#include <pthread.h>

typedef struct {
 pthread_t handle;
} fce_thread_t;

#endif

/* Create a thread with the given stack size (0 = OS default).
 * fn receives arg. Returns 0 on success. */
int fce_thread_create(fce_thread_t *t, size_t stack_size, void *(*fn)(void *), void *arg);

/* Wait for thread to finish. Returns 0 on success. */
int fce_thread_join(fce_thread_t *t);

/* Detach thread so resources are freed on exit. Returns 0 on success. */
int fce_thread_detach(fce_thread_t *t);

/* ── Mutex ────────────────────────────────────────────────────── */

#ifdef _WIN32

typedef struct {
 CRITICAL_SECTION cs;
} fce_mutex_t;

#else

typedef struct {
 pthread_mutex_t mtx;
} fce_mutex_t;

#endif

void fce_mutex_init(fce_mutex_t *m);
void fce_mutex_lock(fce_mutex_t *m);
void fce_mutex_unlock(fce_mutex_t *m);
void fce_mutex_destroy(fce_mutex_t *m);

/* ── Once ─────────────────────────────────────────────────────── */

#ifdef _WIN32

typedef INIT_ONCE fce_once_t;
#define FCE_ONCE_INIT INIT_ONCE_STATIC_INIT

#else

typedef pthread_once_t fce_once_t;
#define FCE_ONCE_INIT PTHREAD_ONCE_INIT

#endif

/* Run `fn` exactly once, thread-safely. Returns 0 on success. */
int fce_once(fce_once_t *once, void (*fn)(void));

/* ── Aligned allocation ───────────────────────────────────────── */

/* Allocate size bytes aligned to alignment boundary.
 * Returns 0 on success, non-zero on failure. *ptr receives the allocation. */
int fce_aligned_alloc(void **ptr, size_t alignment, size_t size);

/* Free memory from fce_aligned_alloc. */
void fce_aligned_free(void *ptr);

#endif /* FCE_COMPAT_THREAD_H */
