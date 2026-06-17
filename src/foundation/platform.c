/* * platform.c — OS abstraction implementations.
 *
 * macOS, Linux, and Windows. Platform-specific code behind #ifdef guards. */
/* Define POSIX feature level before any system headers for clock_gettime support.
 * On macOS, clock_gettime is available without restricting feature macros,
 * and restricting them hides BSD types (u_int, u_char, etc.) used by
 * system headers like sys/proc.h. Only define on Linux. */
#if defined(__linux__)
#define _POSIX_C_SOURCE 200112L
#endif

#include "foundation/platform.h"

#include "foundation/compat_thread.h"
#include "foundation/constants.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32

/* ── Windows implementation ───────────────────────────────────── */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <sys/stat.h>

void *fce_mmap_read(const char *path, size_t *out_size) {
 if (!path || !out_size) {
 return NULL;
 }
 *out_size = 0;

 HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
 FILE_ATTRIBUTE_NORMAL, NULL);
 if (file == INVALID_HANDLE_VALUE) {
 return NULL;
 }
 LARGE_INTEGER sz;
 if (!GetFileSizeEx(file, &sz) || sz.QuadPart == 0) {
 CloseHandle(file);
 return NULL;
 }
 HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
 if (!mapping) {
 CloseHandle(file);
 return NULL;
 }
 void *addr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
 CloseHandle(mapping);
 CloseHandle(file);
 if (!addr) {
 return NULL;
 }
 *out_size = (size_t)sz.QuadPart;
 return addr;
}

void fce_munmap(void *addr, size_t size) {
 (void)size;
 if (addr) {
 UnmapViewOfFile(addr);
 }
}

int fce_atomic_replace(const char *tmp_path, const char *final_path) {
 if (!tmp_path || !final_path) {
 return -1;
 }
 return MoveFileExA(tmp_path, final_path, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
}

uint64_t fce_now_ns(void) {
 LARGE_INTEGER freq, count;
 QueryPerformanceFrequency(&freq);
 QueryPerformanceCounter(&count);
 return (uint64_t)count.QuadPart * 1000000000ULL / (uint64_t)freq.QuadPart;
}

uint64_t fce_now_ms(void) {
 /* FCE_NSEC_PER_MSEC is defined in constants.h. The local FCE_USEC_PER_SEC
 * was a misleadingly-named alias for the same value (1,000,000). */
 return fce_now_ns() / FCE_NSEC_PER_MSEC;
}

int fce_nprocs(void) {
 SYSTEM_INFO si;
 GetSystemInfo(&si);
 return (int)si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}

bool fce_file_exists(const char *path) {
 DWORD attr = GetFileAttributesA(path);
 return attr != INVALID_FILE_ATTRIBUTES;
}

bool fce_is_dir(const char *path) {
 DWORD attr = GetFileAttributesA(path);
 return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
}

int64_t fce_file_size(const char *path) {
 WIN32_FILE_ATTRIBUTE_DATA fad;
 if (!GetFileAttributesExA(path, GetFileExInfoStandard, &fad)) {
 return FCE_NOT_FOUND;
 }
 LARGE_INTEGER sz;
 sz.HighPart = (LONG)fad.nFileSizeHigh; // cppcheck-suppress unreadVariable
 sz.LowPart = fad.nFileSizeLow; // cppcheck-suppress unreadVariable
 return (int64_t)sz.QuadPart;
}

char *fce_normalize_path_sep(char *path) {
 if (path) {
 for (char *p = path; *p; p++) {
 if (*p == '\\') {
 *p = '/';
 }
 }
 }
 return path;
}

#else /* POSIX (macOS + Linux) */

/* ── POSIX implementation ─────────────────────────────────────── */

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#else
#include <sched.h>
#endif

/* ── Memory mapping ────────────────────────────────────────────── */

void *fce_mmap_read(const char *path, size_t *out_size) {
 if (!path || !out_size) {
 return NULL;
 }
 *out_size = 0;

 int fd = open(path, O_RDONLY);
 if (fd < 0) {
 return NULL;
 }

 struct stat st;
 if (fstat(fd, &st) != 0 || st.st_size == 0) {
 close(fd);
 return NULL;
 }

 void *addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
 close(fd);

 if (addr == MAP_FAILED) {
 return NULL;
 }
 *out_size = (size_t)st.st_size;
 return addr;
}

void fce_munmap(void *addr, size_t size) {
 if (addr && size > 0) {
 munmap(addr, size);
 }
}

int fce_atomic_replace(const char *tmp_path, const char *final_path) {
 if (!tmp_path || !final_path) {
 return -1;
 }
 /* rename() is atomic within a filesystem and replaces an existing
  * destination, so a concurrent reader sees either the old file or the
  * fully-written new one — never a torn intermediate. */
 return rename(tmp_path, final_path) == 0 ? 0 : -1;
}

/* ── Timing ────────────────────────────────────────────────────── */

#ifdef __APPLE__
static mach_timebase_info_data_t timebase_info;
static fce_once_t timebase_once = FCE_ONCE_INIT;
static void init_timebase(void) { mach_timebase_info(&timebase_info); }

uint64_t fce_now_ns(void) {
 fce_once(&timebase_once, init_timebase);
 uint64_t ticks = mach_absolute_time();
 /* ticks * numer can overflow uint64_t on
 * long-running macOS servers (~4.7 years on Apple Silicon where numer=125).
 * Use __uint128_t to avoid overflow; supported by GCC/Clang on 64-bit. */
#if defined(__SIZEOF_INT128__)
 __uint128_t product = (__uint128_t)ticks * (__uint128_t)timebase_info.numer;
 return (uint64_t)(product / timebase_info.denom);
#else
 return ticks * timebase_info.numer / timebase_info.denom;
#endif
}
#else
uint64_t fce_now_ns(void) {
 struct timespec ts;
 clock_gettime(CLOCK_MONOTONIC, &ts);
 return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#endif

uint64_t fce_now_ms(void) {
 /* FCE_NSEC_PER_MSEC is defined in constants.h. */
 return fce_now_ns() / FCE_NSEC_PER_MSEC;
}

/* ── System info ───────────────────────────────────────────────── */

int fce_nprocs(void) {
#ifdef __APPLE__
 int ncpu = 0;
 size_t len = sizeof(ncpu);
 if (sysctlbyname("hw.ncpu", &ncpu, &len, NULL, 0) == 0 && ncpu > 0) {
 return ncpu;
 }
 enum { MIN_NPROCS = 1 };
 return MIN_NPROCS;
#else
 long n = sysconf(_SC_NPROCESSORS_ONLN);
 return n > 0 ? (int)n : 1;
#endif
}

/* ── File system ───────────────────────────────────────────────── */

bool fce_file_exists(const char *path) {
 struct stat st;
 return stat(path, &st) == 0;
}

bool fce_is_dir(const char *path) {
 struct stat st;
 return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

int64_t fce_file_size(const char *path) {
 struct stat st;
 if (stat(path, &st) != 0) {
 return FCE_NOT_FOUND;
 }
 return (int64_t)st.st_size;
}

char *fce_normalize_path_sep(char *path) {
 /* Normalize on ALL platforms — backslash paths can arrive via stored
 * data, cross-platform DB files, or Windows-style arguments. */
 if (path) {
 for (char *p = path; *p; p++) {
 if (*p == '\\') {
 *p = '/';
 }
 }
 }
 return path;
}

#endif /* _WIN32 */

/* ── Environment variables ────────────────────────────────────── */

/* Thread-safe getenv: iterates environ directly instead of calling getenv().
 * getenv() is flagged by concurrency-mt-unsafe because the returned pointer
 * can be invalidated by setenv/putenv in another thread. We copy to a
 * caller-owned buffer immediately. */
#ifdef _WIN32
#include <stdlib.h>
#define FCE_ENVIRON _environ
#elif defined(__APPLE__)
#include <crt_externs.h>
#define FCE_ENVIRON (*_NSGetEnviron())
#else
extern char **environ;
#define FCE_ENVIRON environ
#endif

/* Safe getenv: iterates environ directly (NOT safe with concurrent setenv/putenv).
 * the environ array can be reallocated by concurrent
 * setenv/putenv — a concurrent reader would dereference freed memory. This
 * function is safe against concurrent getenv() calls (each copies to its own
 * buffer) but NOT against concurrent setenv/putenv.
 * Safe for: init paths, infrequent calls, or calls where the caller accepts
 * the risk. NOT safe for hot concurrent paths — cache the result at init
 * instead (see FCE_BRUTE_WORKERS / FCE_STACK_SIZE in semantic.c / worker_pool.c). */
const char *fce_safe_getenv(const char *name, char *buf, size_t buf_sz, const char *fallback) {
 if (!name || buf_sz == 0) return NULL; /* NULL name -> strlen UB; buf_sz 0 -> underflow */
 char **env = FCE_ENVIRON;
 if (env) {
 size_t nlen = strlen(name);
 for (; *env; env++) {
 if (strncmp(*env, name, nlen) == 0 && (*env)[nlen] == '=') {
 const char *val = *env + nlen + 1;
 snprintf(buf, buf_sz, "%s", val);
 /* detect truncation — if the source
 * value is longer than the buffer can hold, reject it to
 * prevent strtol/strtod from parsing a truncated number.
 * A value of exactly buf_sz-1 bytes fits (with NUL), so the
 * check is > buf_sz-1 (i.e. >= buf_sz). */
 if (strlen(val) > buf_sz - 1) {
 buf[0] = '\0';
 return NULL;
 }
 return buf;
 }
 }
 }
 if (fallback) {
 snprintf(buf, buf_sz, "%s", fallback);
 return buf;
 }
 buf[0] = '\0';
 return NULL;
}

/* ── Home directory (cross-platform) ──────────────────────────── */

/* NOTE: These path functions return pointers to static buffers.
 * They are NOT thread-safe — concurrent calls from multiple threads
 * can race on the buffer. They are also NOT safe for consecutive calls
 * on the same thread if the result of one is still in use when the next
 * is called (e.g. `printf("%s %s", fce_get_home_dir(), fce_app_config_dir())`).
 * These functions are only called from single-threaded initialization paths. */

const char *fce_get_home_dir(void) {
 static char buf[FCE_SZ_1K];
 char tmp[FCE_SZ_256] = "";

 fce_safe_getenv("HOME", tmp, sizeof(tmp), NULL);
 if (tmp[0]) {
 snprintf(buf, sizeof(buf), "%s", tmp);
 fce_normalize_path_sep(buf);
 return buf;
 }

 fce_safe_getenv("USERPROFILE", tmp, sizeof(tmp), NULL);
 if (tmp[0]) {
 snprintf(buf, sizeof(buf), "%s", tmp);
 fce_normalize_path_sep(buf);
 return buf;
 }
 return NULL;
}

/* ── App config directories (cross-platform) ─────────────────── */

const char *fce_app_config_dir(void) {
 static char buf[FCE_SZ_1K];
 char tmp[FCE_SZ_256] = "";
#ifdef _WIN32
 fce_safe_getenv("APPDATA", tmp, sizeof(tmp), NULL);
 if (tmp[0]) {
 snprintf(buf, sizeof(buf), "%s", tmp);
 fce_normalize_path_sep(buf);
 return buf;
 }
 const char *home = fce_get_home_dir();
 if (home) {
 snprintf(buf, sizeof(buf), "%s/AppData/Roaming", home);
 return buf;
 }
 return NULL;
#else
 /* Linux: XDG_CONFIG_HOME or ~/.config */
 fce_safe_getenv("XDG_CONFIG_HOME", tmp, sizeof(tmp), NULL);
 if (tmp[0]) {
 snprintf(buf, sizeof(buf), "%s", tmp);
 return buf;
 }
 const char *home = fce_get_home_dir();
 if (home) {
 snprintf(buf, sizeof(buf), "%s/.config", home);
 return buf;
 }
 return NULL;
#endif /* _WIN32 */
}

const char *fce_app_local_dir(void) {
#ifdef _WIN32
 static char buf[FCE_SZ_1K];
 char tmp[FCE_SZ_256] = "";
 fce_safe_getenv("LOCALAPPDATA", tmp, sizeof(tmp), NULL);
 if (tmp[0]) {
 snprintf(buf, sizeof(buf), "%s", tmp);
 fce_normalize_path_sep(buf);
 return buf;
 }
 const char *home = fce_get_home_dir();
 if (home) {
 snprintf(buf, sizeof(buf), "%s/AppData/Local", home);
 return buf;
 }
 return NULL;
#else
 return fce_app_config_dir();
#endif
}

/* ── Cache directory ─────────────────────────────────────────── */

const char *fce_resolve_cache_dir(void) {
 static char buf[FCE_SZ_1K];
 char tmp[FCE_SZ_256] = "";
 fce_safe_getenv("FCE_CACHE_DIR", tmp, sizeof(tmp), NULL);
 if (tmp[0]) {
 snprintf(buf, sizeof(buf), "%s", tmp);
 fce_normalize_path_sep(buf);
 return buf;
 }
 const char *home = fce_get_home_dir();
 if (!home) {
 return NULL;
 }
 snprintf(buf, sizeof(buf), "%s/.cache/fast-code-embed", home);
 return buf;
}
