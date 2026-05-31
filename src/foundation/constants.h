/*
 * constants.h — Project-wide named constants.
 */
#ifndef FCE_CONSTANTS_H
#define FCE_CONSTANTS_H

/* ── Byte / character constants ──────────────────────────────── */
enum {
    FCE_BYTE_RANGE = 256, /* full byte range 0x00–0xFF */
    FCE_QUOTE_PAIR = 2,   /* two quote characters (open + close) */
    FCE_QUOTE_OFFSET = 1, /* skip opening quote */
};

/* ── Size units (powers of 2) ────────────────────────────────── */
enum {
    FCE_SZ_2 = 2,
    FCE_SZ_3 = 3,
    FCE_SZ_4 = 4,
    FCE_SZ_5 = 5,
    FCE_SZ_6 = 6,
    FCE_SZ_7 = 7,
    FCE_SZ_8 = 8,
    FCE_SZ_16 = 16,
    FCE_SZ_32 = 32,
    FCE_SZ_64 = 64,
    FCE_SZ_128 = 128,
    FCE_SZ_256 = 256,
    FCE_SZ_512 = 512,
    FCE_SZ_1K = 1024,
    FCE_SZ_2K = 2048,
    FCE_SZ_4K = 4096,
    FCE_SZ_8K = 8192,
    FCE_SZ_16K = 16384,
    FCE_SZ_32K = 32768,
    FCE_SZ_64K = 65536,
};

/* ── Numeric bases and common factors ────────────────────────── */
enum {
    FCE_DECIMAL_BASE = 10,
    FCE_HEX_BASE = 16,
    FCE_PERCENT = 100,
};

/* ── Tree-sitter line offset ─────────────────────────────────── */
/* ts_node row is 0-based; source lines are 1-based. */
enum { TS_LINE_OFFSET = 1 };

/* ── Sentinel values ─────────────────────────────────────────── */
enum {
    FCE_NOT_FOUND = -1, /* search miss, invalid index */
    FCE_INIT_DONE = 1,  /* initialization flag */
};

/* ── Default pagination limits ───────────────────────────────── */
/* Default page size for search_graph and the underlying store-layer search.
 * Chosen so a typical broad query (e.g. file_pattern="**" on a 12k-node
 * project) stays well within MCP tool-result size budgets. Callers that
 * want more results paginate via offset+limit; the response always carries
 * 'total' and 'has_more' so agents can detect truncation. */
enum { FCE_DEFAULT_SEARCH_LIMIT = 200 };

/* ── Time conversion factors ─────────────────────────────────── */
#define FCE_NSEC_PER_SEC 1000000000ULL
#define FCE_USEC_PER_SEC 1000000ULL
#define FCE_MSEC_PER_SEC 1000ULL
#define FCE_NSEC_PER_USEC 1000ULL
#define FCE_NSEC_PER_MSEC 1000000ULL

/* ── Common string/buffer sizes ──────────────────────────────── */
enum {
    FCE_SMALL_BUF = 3,   /* small scratch buffers */
    FCE_NAME_BUF = 4,    /* name buffer slots */
    FCE_PATH_MAX = 1024, /* path buffer size */
    FCE_LINE_BUF = 512,  /* line read buffer */
};

#endif /* FCE_CONSTANTS_H */
