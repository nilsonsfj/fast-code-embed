/*
 * log.c — Structured key-value logging to stderr.
 */
#include "foundation/log.h"
#include "foundation/constants.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

static FCELogLevel g_log_level = FCE_LOG_INFO;
static fce_log_sink_fn g_log_sink = NULL;

void fce_log_set_sink(fce_log_sink_fn fn) {
    g_log_sink = fn;
}

void fce_log_set_level(FCELogLevel level) {
    g_log_level = level;
}

FCELogLevel fce_log_get_level(void) {
    return g_log_level;
}

static const char *level_str(FCELogLevel level) {
    switch (level) {
    case FCE_LOG_DEBUG:
        return "debug";
    case FCE_LOG_INFO:
        return "info";
    case FCE_LOG_WARN:
        return "warn";
    case FCE_LOG_ERROR:
        return "error";
    default:
        return "unknown";
    }
}

void fce_log(FCELogLevel level, const char *msg, ...) {
    if (level < g_log_level) {
        return;
    }

    /* Build the log line into a buffer ONCE — no double va_list iteration */
    char line_buf[FCE_SZ_512];
    int pos =
        snprintf(line_buf, sizeof(line_buf), "level=%s msg=%s", level_str(level), msg ? msg : "");

    va_list args;
    va_start(args, msg);
    for (;;) {
        const char *key = va_arg(args, const char *);
        if (!key) {
            break;
        }
        const char *val = va_arg(args, const char *);
        if (!val) {
            val = "";
        }
        if ((size_t)pos < sizeof(line_buf) - 1) {
            pos += snprintf(line_buf + pos, sizeof(line_buf) - (size_t)pos, " %s=%s", key, val);
        }
    }
    va_end(args);

    /* When a sink is registered it takes over all output (exclusive).
     * Otherwise write structured log to stderr. */
    if (g_log_sink) {
        g_log_sink(line_buf);
    } else {
        (void)fprintf(stderr, "%s\n", line_buf);
    }
}

void fce_log_int(FCELogLevel level, const char *msg, const char *key, int64_t value) {
    if (level < g_log_level) {
        return;
    }

    char line_buf[FCE_SZ_256];
    snprintf(line_buf, sizeof(line_buf), "level=%s msg=%s %s=%" PRId64, level_str(level),
             msg ? msg : "", key ? key : "?", value);

    if (g_log_sink) {
        g_log_sink(line_buf);
    } else {
        (void)fprintf(stderr, "%s\n", line_buf);
    }
}
