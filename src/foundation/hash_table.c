/*
 * hash_table.c — Robin Hood open-addressing hash table.
 *
 * Key design choices:
 *   - FNV-1a hash (fast, good distribution for strings)
 *   - Per-table random seed to mitigate hash-flooding DoS attacks
 *   - Robin Hood insertion: on collision, the key with shorter probe
 *     distance yields its slot. This bounds max probe distance.
 *   - Backward-shift deletion: no tombstones needed.
 *   - Load factor 75% triggers 2x resize.
 */
#include "foundation/hash_table.h"
#include "foundation/constants.h"
#include <stdlib.h> // arc4random

enum {
    HT_MIN_CAP = 8,     /* minimum hash table capacity */
    HT_INITIAL_PSL = 1, /* initial probe sequence length */
    HT_LOAD_NUM = 3,    /* load factor numerator (75%) */
    HT_LOAD_DEN = 4,    /* load factor denominator */
    HT_SHIFT_1 = 1,     /* bit shift amounts for next_pow2 */
    HT_SHIFT_2 = 2,
    HT_SHIFT_4 = 4,
    HT_SHIFT_8 = 8,
    HT_SHIFT_16 = 16,
};
#include <stdbool.h>
#include <stdint.h> // uint32_t
#include <stdlib.h>
#include <string.h>

/* FNV-1a hash constants (published by Fowler/Noll/Vo) */
#define FNV_OFFSET_BASIS 2166136261U
#define FNV_PRIME 16777619U

/* Seeded FNV-1a: XORs seed into basis before hashing to prevent hash-flooding.
 * The seed is stored in FCEHashTable.seed and initialized with arc4random()
 * at table creation time. This makes corpus token map resistant to
 * adversarial inputs that would otherwise degrade Robin Hood performance. */
static uint32_t fnv1a_with_seed(const char *key, uint32_t seed) {
    uint32_t h = FNV_OFFSET_BASIS ^ seed;
    for (const unsigned char *p = (const unsigned char *)key; *p; p++) {
        h ^= *p;
        h *= FNV_PRIME;
    }
    return h;
}

/* Round up to next power of 2 */
static uint32_t next_pow2(uint32_t v) {
    if (v < HT_MIN_CAP) {
        return HT_MIN_CAP;
    }
    v--;
    v |= v >> HT_SHIFT_1;
    v |= v >> HT_SHIFT_2;
    v |= v >> HT_SHIFT_4;
    v |= v >> HT_SHIFT_8;
    v |= v >> HT_SHIFT_16;
    return v + 1;
}

FCEHashTable *fce_ht_create(uint32_t initial_capacity) {
    FCEHashTable *ht = (FCEHashTable *)calloc(1, sizeof(FCEHashTable));
    if (!ht) {
        return NULL;
    }
    ht->capacity = next_pow2(initial_capacity);
    ht->mask = ht->capacity - 1;
    ht->entries = (FCEHTEntry *)calloc(ht->capacity, sizeof(FCEHTEntry));
    if (!ht->entries) {
        free(ht);
        return NULL;
    }
    ht->seed = arc4random();
    return ht;
}

void fce_ht_free(FCEHashTable *ht) {
    if (!ht) {
        return;
    }
    free(ht->entries);
    free(ht);
}

static bool ht_resize(FCEHashTable *ht) {
    if (ht->capacity > UINT32_MAX / 2) return false; /* already at max */
    uint32_t new_cap = ht->capacity * 2;
    uint32_t new_mask = new_cap - 1;
    FCEHTEntry *new_entries = (FCEHTEntry *)calloc(new_cap, sizeof(FCEHTEntry));
    if (!new_entries) {
        return false; /* OOM: keep old table */
    }

    for (uint32_t i = 0; i < ht->capacity; i++) {
        const FCEHTEntry *e = &ht->entries[i];
        if (e->psl == 0) {
            continue; /* empty slot */
        }

        /* Re-insert into new table */
        uint32_t idx = e->hash & new_mask;
        FCEHTEntry cur = {.key = e->key, .value = e->value, .hash = e->hash, .psl = HT_INITIAL_PSL};
        for (;;) {
            FCEHTEntry *slot = &new_entries[idx];
            if (slot->psl == 0) {
                *slot = cur;
                break;
            }
            /* Robin Hood: steal from rich (shorter probe) */
            if (cur.psl > slot->psl) {
                FCEHTEntry tmp = *slot;
                *slot = cur;
                cur = tmp;
            }
            cur.psl++;
            idx = (idx + 1) & new_mask;
        }
    }

    free(ht->entries);
    ht->entries = new_entries;
    ht->capacity = new_cap;
    ht->mask = new_mask;
    return true;
}

void *fce_ht_set(FCEHashTable *ht, const char *key, void *value) {
    if (!ht || !key) return NULL;
    /* Resize at 75% load */
    if ((uint64_t)ht->count * HT_LOAD_DEN >= (uint64_t)ht->capacity * HT_LOAD_NUM) {
        if (!ht_resize(ht)) return NULL;  /* OOM: insert fails */
    }

    uint32_t h = fnv1a_with_seed(key, ht->seed);
    uint32_t idx = h & ht->mask;
    FCEHTEntry cur = {.key = key, .value = value, .hash = h, .psl = HT_INITIAL_PSL};
    void *prev_value = NULL;

    for (uint32_t probe = 0; probe < ht->capacity; probe++) {
        FCEHTEntry *slot = &ht->entries[idx];

        if (slot->psl == 0) {
            /* Empty slot — insert here */
            *slot = cur;
            ht->count++;
            return prev_value;
        }

        /* Check for existing key */
        if (slot->hash == cur.hash && strcmp(slot->key, cur.key) == 0) {
            prev_value = slot->value;
            slot->value = cur.value;
            /* Don't update slot->key — the table already holds a valid pointer,
             * and overwriting it would leak the old allocation. */
            return prev_value;
        }

        /* Robin Hood: steal from rich */
        if (cur.psl > slot->psl) {
            FCEHTEntry tmp = *slot;
            *slot = cur;
            cur = tmp;
        }

        cur.psl++;
        idx = (idx + 1) & ht->mask;
    }
    /* Should never reach here — load factor < 1 guarantees empty slot exists.
     * Guard against infinite loop if resize OOMs repeatedly. */
    return NULL;
}

void *fce_ht_get(const FCEHashTable *ht, const char *key) {
    if (!ht || !key) return NULL;
    uint32_t h = fnv1a_with_seed(key, ht->seed);
    uint32_t idx = h & ht->mask;
    uint32_t psl = 1;

    for (;;) {
        const FCEHTEntry *slot = &ht->entries[idx];
        if (slot->psl == 0) {
            return NULL; /* empty — not found */
        }
        if (psl > slot->psl) {
            return NULL; /* Robin Hood guarantee */
        }
        if (slot->hash == h && strcmp(slot->key, key) == 0) {
            return slot->value;
        }
        psl++;
        idx = (idx + 1) & ht->mask;
    }
}

bool fce_ht_has(const FCEHashTable *ht, const char *key) {
    if (!ht || !key) return false;
    return fce_ht_get(ht, key) != NULL;
}

const char *fce_ht_get_key(const FCEHashTable *ht, const char *key) {
    if (!ht || !key) {
        return NULL;
    }
    uint32_t h = fnv1a_with_seed(key, ht->seed);
    uint32_t idx = h & ht->mask;
    uint32_t psl = 1;
    for (;;) {
        const FCEHTEntry *slot = &ht->entries[idx];
        if (slot->psl == 0) {
            return NULL;
        }
        if (psl > slot->psl) {
            return NULL;
        }
        if (slot->hash == h && strcmp(slot->key, key) == 0) {
            return slot->key;
        }
        psl++;
        idx = (idx + 1) & ht->mask;
    }
}

void *fce_ht_delete(FCEHashTable *ht, const char *key) {
    if (!ht || !key) return NULL;
    uint32_t h = fnv1a_with_seed(key, ht->seed);
    uint32_t idx = h & ht->mask;
    uint32_t psl = 1;

    /* Find the entry */
    for (;;) {
        FCEHTEntry *slot = &ht->entries[idx];
        if (slot->psl == 0) {
            return NULL;
        }
        if (psl > slot->psl) {
            return NULL;
        }
        if (slot->hash == h && strcmp(slot->key, key) == 0) {
            void *removed = slot->value;
            ht->count--;

            /* Backward shift: fill the hole */
            for (;;) {
                uint32_t next_idx = (idx + 1) & ht->mask;
                const FCEHTEntry *next = &ht->entries[next_idx];
                if (next->psl <= HT_INITIAL_PSL) {
                    /* Next slot is empty or at home — stop */
                    ht->entries[idx] = (FCEHTEntry){0};
                    break;
                }
                /* Shift next entry back */
                ht->entries[idx] = *next;
                ht->entries[idx].psl--;
                idx = next_idx;
            }
            return removed;
        }
        psl++;
        idx = (idx + 1) & ht->mask;
    }
}

uint32_t fce_ht_count(const FCEHashTable *ht) {
    return ht ? ht->count : 0;
}

void fce_ht_foreach(const FCEHashTable *ht, fce_ht_iter_fn fn, void *userdata) {
    if (!ht || !fn) {
        return;
    }
    for (uint32_t i = 0; i < ht->capacity; i++) {
        if (ht->entries[i].psl > 0) {
            fn(ht->entries[i].key, ht->entries[i].value, userdata);
        }
    }
}

void fce_ht_clear(FCEHashTable *ht) {
    if (!ht) {
        return;
    }
    memset(ht->entries, 0, ht->capacity * sizeof(FCEHTEntry));
    ht->count = 0;
}
