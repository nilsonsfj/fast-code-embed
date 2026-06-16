/* * log.h — Structured key-value logging to stderr.
 *
 * Design:
 * - All output goes to stderr (stdout is left free for the host program)
 * - Structured format: "level=info msg=pass.timing pass=defs elapsed_ms=42"
 * - Levels: DEBUG, INFO, WARN, ERROR
 * - Level filtering at compile time (FCE_LOG_MIN_LEVEL) and runtime
 * - Thread-safe (each fprintf is atomic on POSIX for lines < PIPE_BUF) */
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
 * fce_log(FCE_LOG_INFO, "pass.timing",
 * "pass", "defs", "elapsed_ms", "42", NULL);
 *
 * Output:
 * level=info msg=pass.timing pass=defs elapsed_ms=42 */
void fce_log(FCELogLevel level, const char *msg, ...);

/* Convenience macros. The msg tag is folded into __VA_ARGS__ so these stay
 * valid ISO C11: there is no trailing `, ##__VA_ARGS__` (a GNU extension that
 * trips -Wpedantic) and no need for __VA_OPT__ (which is C23). Every call site
 * passes at least the msg tag; the trailing NULL terminates the key/value
 * list. Example: fce_log_warn("pass.oom", "pass", "defs"). */
#define fce_log_debug(...) fce_log(FCE_LOG_DEBUG, __VA_ARGS__, NULL)
#define fce_log_info(...) fce_log(FCE_LOG_INFO, __VA_ARGS__, NULL)
#define fce_log_warn(...) fce_log(FCE_LOG_WARN, __VA_ARGS__, NULL)
#define fce_log_error(...) fce_log(FCE_LOG_ERROR, __VA_ARGS__, NULL)

/* Log with integer value (avoids sprintf for common case). */
void fce_log_int(FCELogLevel level, const char *msg, const char *key, int64_t value);

/* Optional log sink callback — called with the formatted log line.
 *
 * CONTRACT: the sink pointer MUST remain valid for the
 * lifetime of any thread that might call fce_log() after set_sink. The
 * library dereferences the pointer on every log call. There is no
 * reference counting — unregister the sink (pass NULL) before freeing the
 * memory it points to, or use a function pointer that lives for the whole
 * process (e.g. a static function). */
typedef void (*fce_log_sink_fn)(const char *line);
void fce_log_set_sink(fce_log_sink_fn fn);

#endif /* FCE_LOG_H */
