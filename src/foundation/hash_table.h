/* * hash_table.h — Robin Hood open-addressing hash table (string → void*).
 *
 * Design decisions:
 * - Keys are interned or arena-allocated strings (NOT copied by the table)
 * - Open addressing with Robin Hood insertion for bounded probe distance
 * - Power-of-2 capacity with 75% load factor trigger for resize
 * - Tombstone-free deletion via backward shift */
#ifndef FCE_HASH_TABLE_H
#define FCE_HASH_TABLE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    const char *key; /* borrowed pointer — caller owns the string */
    void *value;
    uint32_t hash; /* cached hash */
    uint32_t psl;  /* probe sequence length (0 = empty slot) */
} FCEHTEntry;

typedef struct {
    FCEHTEntry *entries;
    uint32_t capacity; /* always power of 2 */
    uint32_t count;    /* number of live entries */
    uint32_t mask;     /* capacity - 1, for fast modulo */
    uint32_t seed;     /* hash seed for DoS mitigation */
} FCEHashTable;

/* Create a hash table with initial capacity (rounded up to power of 2). */
FCEHashTable *fce_ht_create(uint32_t initial_capacity);

/* Free the hash table (does NOT free keys or values). */
void fce_ht_free(FCEHashTable *ht);

/* THREAD-SAFETY: fce_ht_* is NOT thread-safe. Concurrent
 * fce_ht_set / fce_ht_get / fce_ht_delete on the same table will corrupt the
 * probe sequence (the resize check inside set is not atomic with the
 * insert). Externalise synchronization if needed. Read-mostly workloads
 * (one writer at a time, many concurrent readers) are not safe either —
 * the Robin Hood probe assumes a stable layout. The corpus token_map and
 * pretrained map are designed for single-threaded mutation then concurrent
 * reads via the lazy-init state machine. */

/* Insert or update. Returns previous value (NULL if new key or OOM).
 * If 'inserted' is non-NULL, set to true when a new key was added,
 * false on update or OOM — disambiguates the NULL return.
 * B4: callers MUST check 'inserted' to distinguish
 * "new insert, no previous value" (inserted=true, return=NULL) from
 * "OOM during resize" (inserted unchanged, return=NULL). Ignoring
 * 'inserted' and relying solely on the return value is an API trap. */
void *fce_ht_set(FCEHashTable *ht, const char *key, void *value, bool *inserted);

/* Lookup. Returns NULL if not found. */
void *fce_ht_get(const FCEHashTable *ht, const char *key);

/* Check if key exists. */
bool fce_ht_has(const FCEHashTable *ht, const char *key);

/* Return the stored key pointer for a given lookup key, or NULL.
 * Useful when you need the canonical (heap-owned) key string
 * rather than your own local copy. Returns NULL only when the key is not
 * in the table. */
const char *fce_ht_get_key(const FCEHashTable *ht, const char *key);

/* Delete. Returns removed value (NULL if not found). */
void *fce_ht_delete(FCEHashTable *ht, const char *key);

/* Number of entries. */
uint32_t fce_ht_count(const FCEHashTable *ht);

/* Iteration: call fn(key, value, userdata) for each entry. */
typedef void (*fce_ht_iter_fn)(const char *key, void *value, void *userdata);
void fce_ht_foreach(const FCEHashTable *ht, fce_ht_iter_fn fn, void *userdata);

/* Clear all entries (keeps allocated memory). */
void fce_ht_clear(FCEHashTable *ht);

#endif /* FCE_HASH_TABLE_H */
