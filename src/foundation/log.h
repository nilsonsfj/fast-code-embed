/*
 * log.h — Structured key-value logging to stderr.
 *
 * Design:
 *   - All output goes to stderr (stdout is reserved for MCP JSON-RPC)
 *   - Structured format: "level=info msg=pass.timing pass=defs elapsed_ms=42"
 *   - Levels: DEBUG, INFO, WARN, ERROR
 *   - Level filtering at compile time (FCE_LOG_MIN_LEVEL) and runtime
 *   - Thread-safe (each fprintf is atomic on POSIX for lines < PIPE_BUF)
 */
#ifndef FCE_LOG_H
#define FCE_LOG_H

#include <stdint.h>

typedef enum {
    FCE_LOG_DEBUG = 0,
    FCE_LOG_INFO = 1,
    FCE_LOG_WARN = 2,
    FCE_LOG_ERROR = 3,
    FCE_LOG_NONE = 4 /* disable all logging */
} FCELogLevel;

/* Set minimum log level (default: INFO). */
void fce_log_set_level(FCELogLevel level);

/* Get current log level. */
FCELogLevel fce_log_get_level(void);

/* Core logging function. msg is a short semantic tag.
 * Variadic args are key-value pairs: (const char *key, const char *value)...
 * Terminated by NULL key.
 *
 * Example:
 *   fce_log(FCE_LOG_INFO, "pass.timing",
 *           "pass", "defs", "elapsed_ms", "42", NULL);
 *
 * Output:
 *   level=info msg=pass.timing pass=defs elapsed_ms=42
 */
void fce_log(FCELogLevel level, const char *msg, ...);

/* Convenience macros. */
#define fce_log_debug(msg, ...) fce_log(FCE_LOG_DEBUG, msg __VA_OPT__(,) __VA_ARGS__, NULL)
#define fce_log_info(msg, ...) fce_log(FCE_LOG_INFO, msg __VA_OPT__(,) __VA_ARGS__, NULL)
#define fce_log_warn(msg, ...) fce_log(FCE_LOG_WARN, msg __VA_OPT__(,) __VA_ARGS__, NULL)
#define fce_log_error(msg, ...) fce_log(FCE_LOG_ERROR, msg __VA_OPT__(,) __VA_ARGS__, NULL)

/* Log with integer value (avoids sprintf for common case). */
void fce_log_int(FCELogLevel level, const char *msg, const char *key, int64_t value);

/* Optional log sink callback — called with the formatted log line. */
typedef void (*fce_log_sink_fn)(const char *line);
void fce_log_set_sink(fce_log_sink_fn fn);

#endif /* FCE_LOG_H */
