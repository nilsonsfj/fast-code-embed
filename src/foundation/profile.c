/*
 * profile.c — Activatable profiling implementation.
 */
/* Define POSIX feature level on Linux for clock_gettime support.
 * On macOS, clock_gettime is available by default. */
#if defined(__linux__)
#define _POSIX_C_SOURCE 200112L
#endif
#include "foundation/profile.h"
#include "foundation/log.h"
#include "foundation/compat.h"
#include "foundation/platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {
    PROF_BUF_LEN = 32,
    PROF_NS_PER_US = 1000,
    PROF_US_PER_MS = 1000,
    PROF_US_PER_SEC = 1000000,
};
#define PROF_US_PER_SEC_D 1000000.0

bool fce_profile_active = false;

void fce_profile_init(void) {
    /* C-8: use fce_safe_getenv instead of raw
     * getenv() to be consistent with the rest of the codebase and avoid
     * a data race if another thread calls setenv/putenv concurrently. */
    char env_buf[4];
    const char *env = fce_safe_getenv("FCE_PROFILE", env_buf, sizeof(env_buf), NULL);
    if (env && env[0] != '\0' && env[0] != '0') {
        fce_profile_active = true;
    }
}

void fce_profile_enable(void) {
    fce_profile_active = true;
}

void fce_profile_now(struct timespec *ts) {
    fce_clock_gettime(CLOCK_MONOTONIC, ts);
}

void fce_profile_log_elapsed(const char *phase, const char *sub, const struct timespec *start,
                             long items) {
    struct timespec now;
    fce_clock_gettime(CLOCK_MONOTONIC, &now);

    long us = ((long)(now.tv_sec - start->tv_sec) * PROF_US_PER_SEC) +
              ((now.tv_nsec - start->tv_nsec) / PROF_NS_PER_US);
    long ms = us / PROF_US_PER_MS;

    char ms_buf[PROF_BUF_LEN];
    char us_buf[PROF_BUF_LEN];
    char items_buf[PROF_BUF_LEN];
    snprintf(ms_buf, sizeof(ms_buf), "%ld", ms);
    snprintf(us_buf, sizeof(us_buf), "%ld", us);

    if (items > 0 && us > 0) {
        long rate = (long)((double)items * PROF_US_PER_SEC_D / (double)us);
        char rate_buf[PROF_BUF_LEN];
        snprintf(items_buf, sizeof(items_buf), "%ld", items);
        snprintf(rate_buf, sizeof(rate_buf), "%ld", rate);
        fce_log_info("prof", "phase", phase, "sub", sub, "ms", ms_buf, "us", us_buf, "items",
                     items_buf, "rate_per_s", rate_buf);
    } else if (items > 0) {
        snprintf(items_buf, sizeof(items_buf), "%ld", items);
        fce_log_info("prof", "phase", phase, "sub", sub, "ms", ms_buf, "us", us_buf, "items",
                     items_buf);
    } else {
        fce_log_info("prof", "phase", phase, "sub", sub, "ms", ms_buf, "us", us_buf);
    }
}
