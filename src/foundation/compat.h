/* * compat.h — Cross-platform compatibility macros and shims.
 *
 * Provides portable TLS, sleep, strdup/strndup, and getline across
 * POSIX (macOS/Linux) and Windows. Include this instead of using
 * platform-specific macros directly. */
#ifndef FCE_COMPAT_H
#define FCE_COMPAT_H

#include <stddef.h>
#include <stdio.h>

/* ── Thread-local storage ─────────────────────────────────────── */
/* _Thread_local is C11 standard — works on GCC, Clang, and MSVC (2019+).
 * __declspec(thread) is MSVC-only and doesn't work on MinGW GCC. */
#define FCE_TLS _Thread_local

/* ── Sleep ────────────────────────────────────────────────────── */
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define fce_usleep(us) Sleep((DWORD)((us) / 1000))
#else
#include <unistd.h>
#define fce_usleep(us) usleep((useconds_t)(us))
#endif

/* ── strdup / strndup ─────────────────────────────────────────── */
#ifdef _WIN32
#define fce_strdup _strdup
/* Implemented in compat.c */
char *fce_strndup(const char *s, size_t n);
#else
#define fce_strdup strdup
#define fce_strndup strndup
#endif

/* ── getline (Windows lacks it) ───────────────────────────────── */
#ifdef _WIN32
/* Implemented in compat.c */
ssize_t fce_getline(char **lineptr, size_t *n, FILE *stream);
#else
#define fce_getline getline
#endif

/* ── fileno ───────────────────────────────────────────────────── */
#ifdef _WIN32
#define fce_fileno _fileno
#else
#define fce_fileno fileno
#endif

/* ── strcasestr (Windows lacks it) ────────────────────────────── */
#ifdef _WIN32
/* Implemented in compat.c */
char *fce_strcasestr(const char *haystack, const char *needle);
#else
#define fce_strcasestr strcasestr
#endif

/* ── mkdir portability ───────────────────────────────────────── */
#ifdef _WIN32
#include <direct.h>
#define fce_mkdir(path) _mkdir(path)
#else
#include <sys/stat.h>
#define fce_mkdir(path) mkdir(path, 0755)
#endif

/* ── clock_gettime / nanosleep (Windows lacks them) ──────────── */
#include <time.h>
#ifdef _WIN32
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
/* Implemented in compat.c */
int fce_clock_gettime(int clk_id, struct timespec *tp);
static inline int fce_nanosleep(const struct timespec *req, struct timespec *rem) {
    (void)rem;
    Sleep((DWORD)(req->tv_sec * 1000 + req->tv_nsec / 1000000));
    return 0;
}
#else
#define fce_clock_gettime clock_gettime
#define fce_nanosleep nanosleep
#endif

/* ── gmtime_r (Windows lacks it) ─────────────────────────────── */
#ifdef _WIN32
static inline struct tm *fce_gmtime_r(const time_t *timep, struct tm *result) {
    return gmtime_s(result, timep) == 0 ? result : NULL;
}
#else
#define fce_gmtime_r gmtime_r
#endif

/* ── mkdtemp (Windows lacks it) ──────────────────────────────── */
#ifdef _WIN32
/* Translates /tmp/ to %TEMP%\ and copies result back to tmpl.
 * Callers MUST use char buf[FCE_SZ_256] or larger. */
char *fce_mkdtemp(char *tmpl);
#else
#define fce_mkdtemp mkdtemp
#endif

/* ── mkstemp (Windows lacks it) ──────────────────────────────── */
#ifdef _WIN32
int fce_mkstemp(char *tmpl);
#else
#define fce_mkstemp mkstemp
#endif

/* ── setenv / unsetenv (Windows lacks them) ──────────────────── */
#ifdef _WIN32
static inline int fce_setenv(const char *name, const char *value, int overwrite) {
    /* Match POSIX setenv(): when overwrite == 0, leave an existing var unchanged. */
    if (!overwrite) {
        size_t needed = 0;
        if (getenv_s(&needed, NULL, 0, name) == 0 && needed > 0) return 0;
    }
    return _putenv_s(name, value);
}
static inline int fce_unsetenv(const char *name) {
    return _putenv_s(name, "");
}
#else
#define fce_setenv setenv
#define fce_unsetenv unsetenv
#endif

/* ── pipe (Windows uses _pipe) ───────────────────────────────── */
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define fce_pipe(fds) _pipe(fds, 4096, _O_BINARY)
#else
#define fce_pipe(fds) pipe(fds)
#endif

/* ── Temp directory helper ───────────────────────────────────── */
static inline const char *fce_tmpdir(void) {
#ifdef _WIN32
    /* Uses raw getenv instead of fce_safe_getenv
     * for consistency. On Windows, getenv is thread-safe per MSVC docs, and
     * fce_safe_getenv requires a caller-owned buffer which is awkward for a
     * function returning a pointer. This is the only getenv usage in the
     * codebase — documented here for audit purposes. */
    const char *t = getenv("TEMP");
    if (!t)
        t = getenv("TMP");
    return t ? t : ".";
#else
    return "/tmp";
#endif
}

/* ── Signal handling ──────────────────────────────────────────── */
/* Windows doesn't have sigaction; provide macro to select signal API. */
#ifdef _WIN32
#define FCE_HAS_SIGACTION 0
#else
#define FCE_HAS_SIGACTION 1
#endif

#endif /* FCE_COMPAT_H */
