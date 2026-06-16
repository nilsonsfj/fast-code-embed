/* * log.c — Structured key-value logging to stderr. */
#include "foundation/log.h"
#include "foundation/constants.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>

/* g_log_level and g_log_sink are _Atomic so concurrent
 * fce_log / fce_log_set_level / fce_log_set_sink from different threads do
 * not constitute a data race (UB on weakly-ordered architectures). */
static _Atomic FCELogLevel g_log_level = FCE_LOG_INFO;
static _Atomic fce_log_sink_fn g_log_sink = NULL;

void fce_log_set_sink(fce_log_sink_fn fn) {
 atomic_store_explicit(&g_log_sink, fn, memory_order_release);
}

void fce_log_set_level(FCELogLevel level) {
 atomic_store_explicit(&g_log_level, level, memory_order_release);
}

FCELogLevel fce_log_get_level(void) {
 return atomic_load_explicit(&g_log_level, memory_order_acquire);
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
 if (level < atomic_load_explicit(&g_log_level, memory_order_acquire)) {
 return;
 }

 /* Build the log line into a buffer ONCE — no double va_list iteration.
 * Silently truncates at FCE_SZ_512 bytes. This is
 * acceptable for the structured-key=value format because the loss is
 * always in the last value, and the keys/values passed are typically
 * short (pass=defs elapsed_ms=42, etc.). */
 char line_buf[FCE_SZ_512];
 int pos =
 snprintf(line_buf, sizeof(line_buf), "level=%s msg=%s", level_str(level), msg ? msg : "");
 /* handle negative snprintf return (encoding error). */
 if (pos < 0) pos = 0;

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
 /* snprintf can return negative on encoding
 * error. Clamp to 0 so the next snprintf doesn't compute an underflowed
 * pointer. Also clamp to buffer size so the next write is in-bounds. */
 if (pos < 0) pos = 0;
 /* snprintf returns the number of chars that
 * WOULD have been written, so pos can exceed sizeof(line_buf). Normalize
 * after each write so the next snprintf call computes a valid size. */
 if (pos > (int)sizeof(line_buf) - 1) pos = (int)sizeof(line_buf) - 1;
 }
 va_end(args);

 /* When a sink is registered it takes over all output (exclusive).
 * Snapshot the sink pointer atomically; calling through a stale
 * non-NULL pointer is fine because the sink contract is "lives for the
 * whole process lifetime". */
 fce_log_sink_fn sink = atomic_load_explicit(&g_log_sink, memory_order_acquire);
 if (sink) {
 sink(line_buf);
 } else {
 (void)fprintf(stderr, "%s\n", line_buf);
 }
}

void fce_log_int(FCELogLevel level, const char *msg, const char *key, int64_t value) {
 if (level < atomic_load_explicit(&g_log_level, memory_order_acquire)) {
 return;
 }

 char line_buf[FCE_SZ_256];
 snprintf(line_buf, sizeof(line_buf), "level=%s msg=%s %s=%" PRId64, level_str(level),
 msg ? msg : "", key ? key : "?", value);

 fce_log_sink_fn sink = atomic_load_explicit(&g_log_sink, memory_order_acquire);
 if (sink) {
 sink(line_buf);
 } else {
 (void)fprintf(stderr, "%s\n", line_buf);
 }
}
