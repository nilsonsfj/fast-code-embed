/* * system_info.c — CPU core count and RAM detection.
 *
 * macOS: sysctlbyname for core counts, hw.memsize for RAM.
 * BSD: sysconf + sysctl(HW_PHYSMEM64 / HW_PHYSMEM).
 * Linux: sysconf + sysinfo().
 * Windows: GetSystemInfo + GlobalMemoryStatusEx.
 *
 * Results are cached after first call (immutable hardware properties). */
#include "foundation/constants.h"

enum { DEFAULT_CORES = 1, MIN_WORKERS = 1 };
#include "foundation/platform.h"
#include <stdint.h> // uint64_t
#include <string.h>
#include "foundation/compat_thread.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__NetBSD__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#else /* Linux */
#include <unistd.h>
#include <sys/sysinfo.h>

#endif

/* ── macOS detection ─────────────────────────────────────────────── */

#ifdef __APPLE__

static int sysctl_int(const char *name, int fallback) {
 int val = 0;
 size_t len = sizeof(val);
 if (sysctlbyname(name, &val, &len, NULL, 0) == 0 && val > 0) {
 return val;
 }
 return fallback;
}

static size_t sysctl_size(const char *name, size_t fallback) {
 size_t val = 0;
 size_t len = sizeof(val);
 if (sysctlbyname(name, &val, &len, NULL, 0) == 0 && val > 0) {
 return val;
 }
 /* Try FCE_SZ_64-bit variant */
 uint64_t val64 = 0;
 len = sizeof(val64);
 if (sysctlbyname(name, &val64, &len, NULL, 0) == 0 && val64 > 0) {
 return (size_t)val64;
 }
 return fallback;
}

static fce_system_info_t detect_system_macos(void) {
 fce_system_info_t info;
 memset(&info, 0, sizeof(info));

 info.total_cores = sysctl_int("hw.ncpu", DEFAULT_CORES);
 info.perf_cores = sysctl_int("hw.perflevel0.physicalcpu", info.total_cores);

 /* If perflevel sysctls fail (Intel Mac), perf = total */
 int eff = sysctl_int("hw.perflevel1.physicalcpu", 0);
 if (info.perf_cores + eff > info.total_cores) {
 info.perf_cores = info.total_cores;
 }

 info.total_ram = sysctl_size("hw.memsize", 0);
 return info;
}

#elif defined(__NetBSD__) || defined(__FreeBSD__) || defined(__OpenBSD__)

static fce_system_info_t detect_system_bsd(void) {
 fce_system_info_t info;
 memset(&info, 0, sizeof(info));

 long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
 info.total_cores = nprocs > 0 ? (int)nprocs : 1;
 info.perf_cores = info.total_cores;

#if defined(__OpenBSD__)
 int mib[2] = { CTL_HW, HW_PHYSMEM };
#else
 int mib[2] = { CTL_HW, HW_PHYSMEM64 };
#endif
 uint64_t physmem = 0;
 size_t len = sizeof(physmem);
 if (sysctl(mib, 2, &physmem, &len, NULL, 0) == 0 && physmem > 0) {
 info.total_ram = (size_t)physmem;
 }

 return info;
}

#else /* Linux */

static fce_system_info_t detect_system_linux(void) {
 fce_system_info_t info;
 memset(&info, 0, sizeof(info));

 long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
 info.total_cores = nprocs > 0 ? (int)nprocs : 1;
 info.perf_cores = info.total_cores; /* Linux doesn't distinguish P/E */

 struct sysinfo si;
 if (sysinfo(&si) == 0) {
 info.total_ram = (size_t)si.totalram * (size_t)si.mem_unit;
 }

 return info;
}

#endif /* __APPLE__ / BSD / Linux */

/* ── Windows detection ───────────────────────────────────────────── */

#ifdef _WIN32
static fce_system_info_t detect_system_windows(void) {
 fce_system_info_t info;
 memset(&info, 0, sizeof(info));

 SYSTEM_INFO si;
 GetSystemInfo(&si);
 info.total_cores = (int)si.dwNumberOfProcessors;
 if (info.total_cores < 1) {
 info.total_cores = 1;
 }
 info.perf_cores = info.total_cores;

 MEMORYSTATUSEX ms;
 ms.dwLength = sizeof(ms);
 if (GlobalMemoryStatusEx(&ms)) {
 info.total_ram = (size_t)ms.ullTotalPhys;
 }

 return info;
}
#endif

/* ── Public API ──────────────────────────────────────────────────── */

static fce_system_info_t cached_info;
static fce_once_t info_once = FCE_ONCE_INIT;

static void init_system_info(void) {
#ifdef _WIN32
 cached_info = detect_system_windows();
#elif defined(__APPLE__)
 cached_info = detect_system_macos();
#elif defined(__NetBSD__) || defined(__FreeBSD__) || defined(__OpenBSD__)
 cached_info = detect_system_bsd();
#else
 cached_info = detect_system_linux();
#endif
}

fce_system_info_t fce_system_info(void) {
 fce_once(&info_once, init_system_info);
 return cached_info;
}

int fce_default_worker_count(bool initial) {
 /* fce_system_info() is thread-safe by way of
 * the underlying fce_once(&info_once, init_system_info). After the
 * one-time init, the returned `cached_info` is a plain struct copy
 * (by value), so concurrent reads on this path cannot tear. Calls
 * before the first init race benignly — every racing thread runs
 * init_system_info, which is idempotent in practice (the sysctl/grpc
 * calls produce the same data). */
 fce_system_info_t info = fce_system_info();
 if (initial) {
 /* Use all cores for initial indexing — user is waiting */
 return info.total_cores;
 }
 /* Incremental: leave headroom for user's apps */
 int workers = info.perf_cores - 1;
 return workers > 0 ? workers : MIN_WORKERS;
}
