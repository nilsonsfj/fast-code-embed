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

/* Convenience macros. C-2 (review 0002 §5.8 / 0001 §5.7):
 * __VA_OPT__ is a C23 feature, but the Makefile uses -std=c11. The GNU
 * `, ##__VA_ARGS__` extension is supported by GCC, Clang, and most other
 * compilers we target, and lets the macros work in C11 mode. With the
 * extension, passing only `msg` (no varargs) suppresses the comma entirely;
 * passing `msg, k, v` produces the variadic expansion.
 *
 * The `-Wgnu-zero-variadic-macro-arguments` warning is suppressed globally
 * in the Makefile (CFLAGS) because Clang does not honour `_Pragma` push/pop
 * around macro-expansion sites the way GCC does — the warning is attributed
 * to the call site, not the macro definition. The suppression is narrow
 * (this single warning) and intentional. */
#define fce_log_debug(msg, ...) fce_log(FCE_LOG_DEBUG, msg, ##__VA_ARGS__, NULL)
#define fce_log_info(msg, ...)  fce_log(FCE_LOG_INFO,  msg, ##__VA_ARGS__, NULL)
#define fce_log_warn(msg, ...)  fce_log(FCE_LOG_WARN,  msg, ##__VA_ARGS__, NULL)
#define fce_log_error(msg, ...) fce_log(FCE_LOG_ERROR, msg, ##__VA_ARGS__, NULL)

/* Log with integer value (avoids sprintf for common case). */
void fce_log_int(FCELogLevel level, const char *msg, const char *key, int64_t value);

/* Optional log sink callback — called with the formatted log line.
 *
 * CONTRACT (review 0002 §7.3): the sink pointer MUST remain valid for the
 * lifetime of any thread that might call fce_log() after set_sink. The
 * library dereferences the pointer on every log call. There is no
 * reference counting — unregister the sink (pass NULL) before freeing the
 * memory it points to, or use a function pointer that lives for the whole
 * process (e.g. a static function). */
typedef void (*fce_log_sink_fn)(const char *line);
void fce_log_set_sink(fce_log_sink_fn fn);

#endif /* FCE_LOG_H */
