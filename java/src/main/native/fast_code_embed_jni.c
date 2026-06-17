/* * fast_code_embed_jni.c — JNI bridge for fast-code-embed.
 *
 * Covers: corpus lifecycle (including batch), simple scoring, ranking, search.
 * FuncDescriptor fields are marshalled into temporary fce_sem_func_t on the C side.
 *
 * Design notes:
 * - JNI_OnLoad caches jclass and jfieldID once; marshal_func reuses the cached
 *   IDs (no GetFieldID per call)
 * - Batch loops cache jstring refs and reuse them for release
 * - DeleteLocalRef after every intermediate ref to avoid local-ref table churn
 * - ExceptionCheck after all JNI calls that can throw; early return on count==0
 * - marshal_func releases pinned arrays on error; all malloc/calloc are checked
 *   for NULL and jump to a single cleanup path on failure
 *
 * JNI exception-throwing pattern:
 * - For NullPointerException, use the cached `cls_npe` global ref
 * (initialised in JNI_OnLoad via FindClass("java/lang/NullPointerException")).
 * - For IllegalArgumentException, use the cached `cls_illegal_arg` global ref.
 * - For any other exception class, FindClass + ThrowNew is acceptable
 * (call sites are rare; the cost is hidden in the error path).
 * - All JNI calls that can throw are followed by an ExceptionCheck
 * before continuing. The macro CHECK_EXCEPTION_RETURN(env, retval)
 * is provided for short-circuit return on pending exception. */
#include <jni.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "semantic/semantic.h"
#include "foundation/log.h"

/* ── Opaque handle table ──────────────── *
 * Converts a Java-level double-close / stale-handle bug from JVM heap
 * corruption into a catchable IllegalStateException. Every corpus
 * handle is encoded as (index | (generation << 16)) where generation
 * is incremented on each free, making stale handles detectable. */
#include <pthread.h>

/* ── Opaque handle table ─────────────────────────────────────────────
 *
 * The handle table converts opaque Java-level jlong handles into C
 * pointers, with generation counters that detect stale/duplicate handles
 * and turn double-close bugs into safe NULL returns.
 *
 * LIFETIME CONTRACT:
 *
 * Double-free is prevented at the C level: nFreeCorpus uses take_handle()
 * which atomically decodes AND clears the slot under a single lock, then
 * waits for in-flight users to drain (refcount drops to zero) before
 * returning the pointer for the caller to free.
 *
 * Use-against-free is prevented at the C level via per-slot refcounts:
 * - acquire_handle() increments the refcount under the mutex.
 * - release_handle() decrements it; if the slot was already taken
 * (ptr == NULL) and this was the last user, it signals drain_cv
 * so take_handle() can proceed with the free.
 * - take_handle() clears the pointer, then waits for refcount to
 * reach zero before returning — guaranteeing no outstanding user
 * holds a dangling pointer.
 *
 * Every JNI entry point that decodes a handle MUST call acquire_handle()
 * on success and release_handle() on EVERY return path (normal, early,
 * or error). Omission causes a stuck refcount (take_handle never returns)
 * or a use-after-free (take_handle returns before the user is done).
 *
 * The generation counter's sole purpose is catching stale/duplicate
 * handles, NOT lifetime management against concurrent free.
 * ──────────────────────────────────────────────────────────────────── */

#define HANDLE_TABLE_CAP 4096
#define HANDLE_SLOT_BITS 16                                         /* bits for slot index */
#define HANDLE_GEN_MASK ((1ULL << (64 - HANDLE_SLOT_BITS - 1)) - 1) /* 47 bits, bit 63 always clear */

typedef struct {
    void *ptr;    /* live fce_sem_corpus_t* or NULL */
    uint64_t gen; /* monotonically increasing; gen==0 → slot free */
    int refcount; /* active users */
} handle_slot_t;

static handle_slot_t g_handle_table[HANDLE_TABLE_CAP];
static pthread_mutex_t g_handle_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_handle_drain_cv = PTHREAD_COND_INITIALIZER;

/* Encode a (slot, generation) pair into a jlong handle.
 * HANDLE_GEN_MASK is 47 bits so bit 63 (sign bit)
 * is always clear — the encoded handle is always positive as a jlong. */
static inline jlong encode_handle(int slot, uint64_t gen) {
    return (jlong)(((uint64_t)slot) | ((gen & HANDLE_GEN_MASK) << HANDLE_SLOT_BITS));
}

/* Allocate a handle table slot. Returns encoded handle, or -1 on OOM/full. */
static jlong alloc_handle(void *ptr) {
    pthread_mutex_lock(&g_handle_mutex);
    for (int i = 0; i < HANDLE_TABLE_CAP; i++) {
        /* A slot is reusable only when it is both unowned (ptr==NULL) AND fully
         * drained (refcount==0). Requiring ptr==NULL alone is unsafe: take_handle()
         * clears ptr BEFORE it has waited for in-flight acquirers to finish, so a
         * slot can momentarily have ptr==NULL while refcount>1. Reusing it then
         * would reset refcount=1, let the draining closer observe its
         * "refcount<=1" predicate prematurely, and free a corpus another thread is
         * still querying (use-after-free). take_handle() sets refcount=0 only after
         * the drain completes, so this guard closes that window. */
        if (g_handle_table[i].ptr == NULL && g_handle_table[i].refcount == 0) {
            g_handle_table[i].gen++;
            if (g_handle_table[i].gen == 0) g_handle_table[i].gen = 1;
            g_handle_table[i].ptr = ptr;
            g_handle_table[i].refcount = 1; /* initial owner */
            jlong h = encode_handle(i, g_handle_table[i].gen);
            pthread_mutex_unlock(&g_handle_mutex);
            return h;
        }
    }
    pthread_mutex_unlock(&g_handle_mutex);
    return -1; /* table full */
}

/* Decode handle and increment refcount.
 * Returns the corpus pointer if the generation matches, or NULL.
 * The caller MUST call release_handle() on every return path. */
static void *acquire_handle(jlong handle) {
    if (handle <= 0) return NULL;
    uint64_t encoded = (uint64_t)handle;
    int slot = (int)(encoded & ((1 << HANDLE_SLOT_BITS) - 1));
    uint64_t gen = encoded >> HANDLE_SLOT_BITS;
    if (slot < 0 || slot >= HANDLE_TABLE_CAP) return NULL;
    pthread_mutex_lock(&g_handle_mutex);
    void *ptr = g_handle_table[slot].ptr;
    uint64_t live_gen = g_handle_table[slot].gen;
    if (ptr && gen == live_gen) {
        g_handle_table[slot].refcount++;
        pthread_mutex_unlock(&g_handle_mutex);
        return ptr;
    }
    pthread_mutex_unlock(&g_handle_mutex);
    return NULL;
}

/* Decrement refcount. If the slot was already taken
 * (ptr==NULL by take_handle) and this was the last user, signal drain_cv
 * so take_handle() can proceed with the free. The condition is refcount<=1
 * (not ==0) because take_handle consumes the owner reference after the drain
 * completes, so it waits for refcount to drop to 1 (all acquirers done).
 * validate generation to prevent decrements from
 * stale or double-released handles — a double-release or stale-handle whose
 * slot has been reallocated to a different corpus would otherwise decrement
 * the wrong slot's refcount, causing use-after-free. After take_handle
 * clears ptr, the generation remains unchanged, so in-flight acquirers
 * still validate successfully.
 * guard against refcount underflow — if
 * release_handle is called without a matching acquire_handle (e.g., error
 * path), skip the decrement when refcount is already 0 to prevent
 * negative refcount that could let take_handle proceed prematurely. */
static void release_handle(jlong handle) {
    if (handle <= 0) return;
    uint64_t encoded = (uint64_t)handle;
    int slot = (int)(encoded & ((1 << HANDLE_SLOT_BITS) - 1));
    uint64_t gen = encoded >> HANDLE_SLOT_BITS;
    if (slot < 0 || slot >= HANDLE_TABLE_CAP) return;
    pthread_mutex_lock(&g_handle_mutex);
    if (gen == g_handle_table[slot].gen) {
        int rc = g_handle_table[slot].refcount;
        if (rc > 0) {
            rc = --g_handle_table[slot].refcount;
            if (g_handle_table[slot].ptr == NULL && rc <= 1) {
                /* broadcast so all waiters in
                 * take_handle() re-check their predicate. signal() wakes
                 * exactly one arbitrary waiter — if that waiter is draining a
                 * different slot its predicate is false, and the waiter whose
                 * predicate is now true is never woken. */
                pthread_cond_broadcast(&g_handle_drain_cv);
            }
        }
    }
    pthread_mutex_unlock(&g_handle_mutex);
}

/* Atomically decode AND clear the handle slot,
 * then wait for all in-flight users to drain (refcount drops to 1, meaning
 * only the owner baseline remains). Returns the pointer only if the generation
 * matches; the winning caller gets a non-NULL pointer and the slot is NULLed
 * under the same lock, so a second concurrent close() on the same handle
 * safely gets NULL. The caller is responsible for freeing the returned pointer. */
static void *take_handle(jlong handle) {
    if (handle <= 0) return NULL;
    uint64_t encoded = (uint64_t)handle;
    int slot = (int)(encoded & ((1 << HANDLE_SLOT_BITS) - 1));
    uint64_t gen = encoded >> HANDLE_SLOT_BITS;
    if (slot < 0 || slot >= HANDLE_TABLE_CAP) return NULL;
    pthread_mutex_lock(&g_handle_mutex);
    void *ptr = g_handle_table[slot].ptr;
    uint64_t live_gen = g_handle_table[slot].gen;
    if (ptr && gen == live_gen) {
        g_handle_table[slot].ptr = NULL; /* clear — no new acquires possible */
        /* Wait for in-flight users to finish. */
        while (g_handle_table[slot].refcount > 1) {
            pthread_cond_wait(&g_handle_drain_cv, &g_handle_mutex);
        }
        g_handle_table[slot].refcount = 0;
    } else {
        ptr = NULL;
    }
    pthread_mutex_unlock(&g_handle_mutex);
    return ptr;
}

/* ── Cached IDs (initialized in JNI_OnLoad) ────────────────────── */

static jclass cls_func;
static jfieldID fid_file_path;
static jfieldID fid_tfidf_indices;
static jfieldID fid_tfidf_weights;
static jfieldID fid_ri_vec;

static jclass cls_search_result;
static jmethodID ctor_search_result;

static jclass cls_illegal_arg;
static jclass cls_npe;
static jclass cls_oom;

/* ── Helpers ────────────────────────────────────────────────────── */

#define CHECK_EXCEPTION_RETURN(env, retval)                 \
    do {                                                    \
        if ((*env)->ExceptionCheck(env)) { return retval; } \
    } while (0)

/* Raise OutOfMemoryError for a native allocation failure, so a failed
 * malloc/calloc/strdup surfaces in Java as a thrown exception rather than a
 * silent no-op (a void method that adds nothing) or a generic sentinel. Uses
 * the cached cls_oom; if a JNI exception is already pending, leave it in place.
 * Caller still returns its sentinel afterwards. */
static void throw_oom(JNIEnv *env, const char *ctx) {
    if ((*env)->ExceptionCheck(env)) return; /* don't mask an existing one */
    if (cls_oom) (*env)->ThrowNew(env, cls_oom, ctx ? ctx : "native allocation failed");
}

/* Build a fce_sem_func_t from a Java FuncDescriptor object.
 *
 * This function returns a zeroed descriptor with
 * file_path==NULL on OOM *and* throws OutOfMemoryError. Every caller MUST
 * check ExceptionCheck() before using the returned struct. This "return
 * half-built struct + pending exception" pattern is brittle — the flat API
 * (nSimpleRankFlat) is the preferred model for new code.
 *
 * Ownership contract:
 * - Caller must free(*path_out) when done.
 * - If tfidf_indices/tfidf_weights are pinned (non-NULL in the returned struct),
 * caller must call unmarshal_func(env, &f, jindices, jweights, path) to release
 * them back to the JVM. Forgetting to call unmarshal_func leaks pinned JNI arrays.
 * - ri_vec is copied into the struct (not pinned), so no release needed for it.
 * - *jindices_out / *jweights_out are set to the JNI array refs so the caller can
 * pass them to unmarshal_func without re-fetching from the JVM.
 *
 * Uses cached field IDs — no GetFieldID calls.
 * On JNI exception, releases any already-acquired pinned arrays before returning. */
static fce_sem_func_t marshal_func(JNIEnv *env, jobject obj, char **path_out,
                                   jintArray *jindices_out, jfloatArray *jweights_out) {
    /* JNI-MED-2: null-guard the Java object before any Get*Field call.
     * GetObjectField(env, NULL, fid) is undefined per JNI spec — some JVMs
     * crash immediately. Throw NPE explicitly so callers see a clear error. */
    if (!obj) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "FuncDescriptor is null");
        fce_sem_func_t f;
        memset(&f, 0, sizeof(f));
        if (path_out) *path_out = NULL;
        if (jindices_out) *jindices_out = NULL;
        if (jweights_out) *jweights_out = NULL;
        return f;
    }
    fce_sem_func_t f;
    memset(&f, 0, sizeof(f));

    /* filePath */
    jstring jpath = (jstring)(*env)->GetObjectField(env, obj, fid_file_path);
    if (jpath) {
        const char *p = (*env)->GetStringUTFChars(env, jpath, NULL);
        /* GetStringUTFChars can return NULL on OOM without
         * necessarily raising a pending exception (some JVMs return NULL and
         * set the pending OOM only on the *next* JNI call). strdup(NULL) is
         * UB, almost certainly a crash. Check for both: */
        if (!p || (*env)->ExceptionCheck(env)) {
            if (p) (*env)->ReleaseStringUTFChars(env, jpath, p);
            (*env)->DeleteLocalRef(env, jpath);
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            /* throw OOM so callers see the failure
             * via ExceptionCheck rather than silently scoring against a zeroed
             * descriptor. use cached cls_oom to avoid
             * FindClass under memory pressure. If cls_oom is NULL, the pending
             * OOM from GetStringUTFChars is sufficient — leave it. */
            if (cls_oom) (*env)->ThrowNew(env, cls_oom,
                                          "marshal_func: GetStringUTFChars OOM");
            return (fce_sem_func_t){0};
        }
        /* check strdup return — on OOM,
         * *path_out is NULL, which causes downstream scoring to silently
         * degrade (no proximity boost). Release jpath, delete jpath ref,
         * and return zeroed func so caller sees the failure. */
        *path_out = strdup(p);
        (*env)->ReleaseStringUTFChars(env, jpath, p);
        if (!*path_out) {
            (*env)->DeleteLocalRef(env, jpath);
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            /* throw OOM so callers see the failure
             * via ExceptionCheck rather than silently scoring against a zeroed
             * descriptor. use cached cls_oom. */
            if (cls_oom) (*env)->ThrowNew(env, cls_oom,
                                          "marshal_func: strdup OOM");
            return (fce_sem_func_t){0};
        }
        f.file_path = *path_out;
    } else {
        *path_out = NULL;
    }
    if (jpath) (*env)->DeleteLocalRef(env, jpath);

    /* tfidfIndices */
    jintArray jindices = (jintArray)(*env)->GetObjectField(env, obj, fid_tfidf_indices);
    /* tfidfWeights */
    jfloatArray jweights = (jfloatArray)(*env)->GetObjectField(env, obj, fid_tfidf_weights);

    *jindices_out = jindices;
    *jweights_out = jweights;

    /* A2: exactly one of indices/weights present is a
     * silent quality degradation (struct path drops TF-IDF silently);
     * the flat path already rejects this with IllegalArgumentException.
     * Match that behaviour here for consistency. */
    if ((jindices != NULL) != (jweights != NULL)) {
        if (cls_illegal_arg)
            (*env)->ThrowNew(env, cls_illegal_arg,
                             "tfidfIndices and tfidfWeights must both be present or both absent");
        if (jindices) (*env)->DeleteLocalRef(env, jindices);
        if (jweights) (*env)->DeleteLocalRef(env, jweights);
        free(*path_out);
        *path_out = NULL;
        *jindices_out = NULL;
        *jweights_out = NULL;
        return (fce_sem_func_t){0};
    }

    if (jindices && jweights) {
        jint ilen = (*env)->GetArrayLength(env, jindices);
        jint wlen = (*env)->GetArrayLength(env, jweights);
        f.tfidf_indices = (int *)(*env)->GetIntArrayElements(env, jindices, NULL);
        /* J1: check for NULL return and pending exception
         * from GetIntArrayElements *before* issuing GetFloatArrayElements.
         * Calling further JNI functions with a pending exception is undefined
         * per the JNI spec. */
        if (!f.tfidf_indices || (*env)->ExceptionCheck(env)) {
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (jindices) (*env)->DeleteLocalRef(env, jindices);
            if (jweights) (*env)->DeleteLocalRef(env, jweights);
            free(*path_out);
            *path_out = NULL;
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            return (fce_sem_func_t){0};
        }
        f.tfidf_weights = (float *)(*env)->GetFloatArrayElements(env, jweights, NULL);
        if (!f.tfidf_weights || (*env)->ExceptionCheck(env)) {
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (f.tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
            if (jindices) (*env)->DeleteLocalRef(env, jindices);
            if (jweights) (*env)->DeleteLocalRef(env, jweights);
            free(*path_out);
            *path_out = NULL;
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            return (fce_sem_func_t){0};
        }
        /* J-1: tfidf_indices MUST be sorted ascending —
         * fce_sparse_tfidf_cosine uses a two-pointer merge that produces
         * wrong scores on unsorted input. The struct-based API is
         * deprecated, but "deprecated" ≠ "removed", so we validate at the
         * JNI boundary where untrusted Java arrays cross into C.
         * Cost is O(n) relative to the per-corpus O(n·k) scoring that
         * follows — negligible. */
        {
            bool sorted = true;
            jint tfidf_len = ilen < wlen ? ilen : wlen;
            if (tfidf_len > 1) {
                for (jint k = 1; k < tfidf_len; k++) {
                    if (f.tfidf_indices[k] <= f.tfidf_indices[k - 1]) {
                        sorted = false;
                        break;
                    }
                }
            }
            if (!sorted) {
                (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
                (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
                jclass iae = cls_illegal_arg ? cls_illegal_arg : (*env)->FindClass(env, "java/lang/IllegalArgumentException");
                (*env)->ThrowNew(env, iae, "tfidf_indices must be sorted ascending");
                if (jindices) (*env)->DeleteLocalRef(env, jindices);
                if (jweights) (*env)->DeleteLocalRef(env, jweights);
                free(*path_out);
                *path_out = NULL;
                if (jindices_out) *jindices_out = NULL;
                if (jweights_out) *jweights_out = NULL;
                return (fce_sem_func_t){0};
            }
        }
        /* reject tfidf index/weight length mismatch.
         * Silently truncating to the shorter length discards data with no
         * diagnostic. Throwing IllegalArgumentException matches the strictness
         * of the flat path's query-side checks. */
        if (ilen != wlen) {
            (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
            jclass iae = cls_illegal_arg ? cls_illegal_arg : (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            (*env)->ThrowNew(env, iae, "tfidfIndices and tfidfWeights must have the same length");
            if (jindices) (*env)->DeleteLocalRef(env, jindices);
            if (jweights) (*env)->DeleteLocalRef(env, jweights);
            free(*path_out);
            *path_out = NULL;
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            return (fce_sem_func_t){0};
        }
        f.tfidf_len = ilen;
    }

    /* riVec */
    jfloatArray jri = (jfloatArray)(*env)->GetObjectField(env, obj, fid_ri_vec);
    if (jri) {
        jfloat *elems = (*env)->GetFloatArrayElements(env, jri, NULL);
        /* check both NULL return AND pending
         * exception. The JNI spec allows GetFloatArrayElements to return
         * NULL without a pending exception on some JVMs. Mirror the
         * filePath guard (line 92). */
        if (!elems || (*env)->ExceptionCheck(env)) {
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (f.tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
            /* B3: delete jindices/jweights local refs for
             * symmetry with the success path (unmarshal_func expects them or
             * NULL). Without this, they leak until the native frame pops. */
            if (jindices) (*env)->DeleteLocalRef(env, jindices);
            if (jweights) (*env)->DeleteLocalRef(env, jweights);
            (*env)->DeleteLocalRef(env, jri);
            free(*path_out);
            *path_out = NULL;
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            return (fce_sem_func_t){0};
        }
        /* Validate array length — must be >= FCE_SEM_DIM. */
        jint ri_len = (*env)->GetArrayLength(env, jri);
        if (ri_len < FCE_SEM_DIM) {
            (*env)->ReleaseFloatArrayElements(env, jri, elems, JNI_ABORT);
            (*env)->DeleteLocalRef(env, jri);
            jclass iae = cls_illegal_arg ? cls_illegal_arg : (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            char ri_len_msg[64];
            snprintf(ri_len_msg, sizeof(ri_len_msg), "riVec length must be >= %d", FCE_SEM_DIM);
            (*env)->ThrowNew(env, iae, ri_len_msg);
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (f.tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
            if (jindices) (*env)->DeleteLocalRef(env, jindices);
            if (jweights) (*env)->DeleteLocalRef(env, jweights);
            free(*path_out);
            *path_out = NULL;
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            return (fce_sem_func_t){0};
        }
        memcpy(f.ri_vec.v, elems, sizeof(float) * FCE_SEM_DIM);
        (*env)->ReleaseFloatArrayElements(env, jri, elems, JNI_ABORT);
    }
    (*env)->DeleteLocalRef(env, jri);

    return f;
}

/* Release JNI array refs after marshalling is done.
 * Receives the JNI array refs directly from marshal_func — no re-fetch.
 * J1: marshal_func NULLs *jindices_out and *jweights_out on
 * every error return. unmarshal_func guards on non-NULL before release/delete,
 * so calling it after a marshal_func error is safe (all refs are NULL, nothing
 * is released). This contract is binding: any future marshal_func error branch
 * that forgets to NULL-out the output pointers would cause a double-release. */
static void unmarshal_func(JNIEnv *env, jobject obj, fce_sem_func_t *f,
                           jintArray jindices, jfloatArray jweights, char *path) {
    (void)obj;
    if (jindices && f->tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f->tfidf_indices, JNI_ABORT);
    if (jweights && f->tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f->tfidf_weights, JNI_ABORT);

    if (jindices) (*env)->DeleteLocalRef(env, jindices);
    if (jweights) (*env)->DeleteLocalRef(env, jweights);

    free(path);
}

/* ── Library load ────────────────────────────────────────────────── */

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    (void)reserved;
    JNIEnv *env;
    if ((*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    /* Cache FuncDescriptor class + field IDs */
    jclass local_func = (*env)->FindClass(env, "io/github/nilsonsfj/fastcodeembed/FuncDescriptor");
    if (!local_func) goto fail;
    cls_func = (*env)->NewGlobalRef(env, local_func);
    (*env)->DeleteLocalRef(env, local_func);
    if (!cls_func) goto fail;

    fid_file_path = (*env)->GetFieldID(env, cls_func, "filePath", "Ljava/lang/String;");
    fid_tfidf_indices = (*env)->GetFieldID(env, cls_func, "tfidfIndices", "[I");
    fid_tfidf_weights = (*env)->GetFieldID(env, cls_func, "tfidfWeights", "[F");
    fid_ri_vec = (*env)->GetFieldID(env, cls_func, "riVec", "[F");
    if (!fid_file_path || !fid_tfidf_indices || !fid_tfidf_weights || !fid_ri_vec) {
        goto fail;
    }

    /* Cache SearchResult class + constructor */
    jclass local_sr = (*env)->FindClass(env, "io/github/nilsonsfj/fastcodeembed/SearchResult");
    if (!local_sr) goto fail;
    cls_search_result = (*env)->NewGlobalRef(env, local_sr);
    (*env)->DeleteLocalRef(env, local_sr);
    if (!cls_search_result) goto fail;

    ctor_search_result = (*env)->GetMethodID(env, cls_search_result, "<init>", "(IF)V");
    if (!ctor_search_result) goto fail;

    /* Cache IllegalArgumentException class (avoid FindClass in error path).
     * hard-fail if core exception classes can't load — using
     * a NULL cls at ThrowNew is UB. */
    jclass local_iae = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
    if (!local_iae) goto fail;
    cls_illegal_arg = (*env)->NewGlobalRef(env, local_iae);
    (*env)->DeleteLocalRef(env, local_iae);
    if (!cls_illegal_arg) goto fail;

    /* Cache NullPointerException class for explicit NPE throws. */
    jclass local_npe = (*env)->FindClass(env, "java/lang/NullPointerException");
    if (!local_npe) goto fail;
    cls_npe = (*env)->NewGlobalRef(env, local_npe);
    (*env)->DeleteLocalRef(env, local_npe);
    if (!cls_npe) goto fail;

    /* cache OutOfMemoryError class so marshal_func's
     * OOM paths don't call FindClass under memory pressure (FindClass can itself
     * fail with OOM, making ThrowNew(env, NULL, msg) UB). */
    jclass local_oom = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
    if (!local_oom) goto fail;
    cls_oom = (*env)->NewGlobalRef(env, local_oom);
    (*env)->DeleteLocalRef(env, local_oom);
    if (!cls_oom) goto fail;

    return JNI_VERSION_1_6;

fail:
    /* clean up any global refs already created before
     * aborting library load. Symmetric with JNI_OnUnload. */
    if (cls_func) {
        (*env)->DeleteGlobalRef(env, cls_func);
        cls_func = NULL;
    }
    if (cls_search_result) {
        (*env)->DeleteGlobalRef(env, cls_search_result);
        cls_search_result = NULL;
    }
    if (cls_illegal_arg) {
        (*env)->DeleteGlobalRef(env, cls_illegal_arg);
        cls_illegal_arg = NULL;
    }
    if (cls_npe) {
        (*env)->DeleteGlobalRef(env, cls_npe);
        cls_npe = NULL;
    }
    if (cls_oom) {
        (*env)->DeleteGlobalRef(env, cls_oom);
        cls_oom = NULL;
    }
    return JNI_ERR;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    (void)vm;
    (void)reserved;
    /* Do NOT call fce_sem_shutdown() here.
     * JNI_OnUnload runs when the defining classloader is unloaded, which
     * may be concurrent with application threads mid-search. The C API
     * contract (semantic.h:112) requires all fce_sem_* operations to have
     * quiesced before shutdown — OnUnload provides no such guarantee.
     * The static maps (~30 MB) are reclaimed by process teardown. */
    JNIEnv *env;
    if ((*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6) == JNI_OK) {
        if (cls_func) (*env)->DeleteGlobalRef(env, cls_func);
        if (cls_search_result) (*env)->DeleteGlobalRef(env, cls_search_result);
        if (cls_illegal_arg) (*env)->DeleteGlobalRef(env, cls_illegal_arg);
        if (cls_npe) (*env)->DeleteGlobalRef(env, cls_npe);
        if (cls_oom) (*env)->DeleteGlobalRef(env, cls_oom);
    }
    cls_func = NULL;
    cls_search_result = NULL;
    cls_illegal_arg = NULL;
    cls_npe = NULL;
    cls_oom = NULL;
}

/* ── Corpus JNI ─────────────────────────────────────────────────── */

JNIEXPORT jlong JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nCreateCorpus(
    JNIEnv *env, jclass cls) {
    (void)env;
    (void)cls;
    fce_sem_corpus_t *c = fce_sem_corpus_new();
    /* fce_sem_corpus_new returns NULL on OOM. The old
     * scheme returned 0 (= NULL handle) for both "valid empty corpus" and
     * "OOM", which made Corpus treat OOM as a successful create-then-close
     * and surface a confusing "Corpus is closed" IllegalStateException on
     * the first method call. The Java side now reserves 0 for "closed" and
     * uses -1 (any sentinel non-zero, non-positive) for "OOM". */
    if (!c) {
        return (jlong)-1;
    }
    /* register the pointer in the handle table and
     * return an opaque (index | generation) encoded handle. A stale positive
     * handle from a previously-freed Corpus will fail the generation check
     * and decode as NULL instead of being cast to a live pointer. */
    jlong h = alloc_handle(c);
    if (h < 0) {
        /* Table full or OOM — free the corpus and report OOM. */
        fce_sem_corpus_free(c);
        return (jlong)-1;
    }
    return h;
}

JNIEXPORT void JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nFreeCorpus(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env;
    (void)cls;
    /* decode handle via the handle table. A stale
     * or duplicated handle fails the generation check and returns NULL.
     * B1: use take_handle() — atomic decode+clear under
     * a single lock. If two threads race close() on the same handle, only
     * the first gets a non-NULL pointer; the second safely gets NULL. */
    void *ptr = take_handle(handle);
    if (!ptr) return;
    fce_sem_corpus_free((fce_sem_corpus_t *)ptr);
}

JNIEXPORT jint JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSaveCorpus(
    JNIEnv *env, jclass cls, jlong handle, jstring jpath, jobjectArray jlabels) {
    (void)cls;
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return -1;
    if (!jpath) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "path is null");
        release_handle(handle);
        return -1;
    }
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    if (!path || (*env)->ExceptionCheck(env)) {
        if (path) (*env)->ReleaseStringUTFChars(env, jpath, path);
        release_handle(handle);
        return -1;
    }

    const char **labels = NULL;
    int label_count = 0;
    int pinned = 0;
    int rc = -1;

    if (jlabels) {
        label_count = (*env)->GetArrayLength(env, jlabels);
        if (label_count > 0) {
            /* Deep-copy each label and release its JNI string + local ref
             * immediately, so at most one local reference is live at a time.
             * Caching one jstring ref per label would accumulate `label_count`
             * references (one per doc) and overflow the JNI local reference
             * table on large corpora (ART hard-caps it at 512). */
            labels = (const char **)calloc((size_t)label_count, sizeof(char *));
            if (!labels) {
                goto save_cleanup;
            }
            for (int i = 0; i < label_count; i++) {
                jstring js = (jstring)(*env)->GetObjectArrayElement(env, jlabels, i);
                if ((*env)->ExceptionCheck(env)) {
                    if (js) (*env)->DeleteLocalRef(env, js);
                    goto save_cleanup;
                }
                if (!js) {
                    if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null label in array");
                    goto save_cleanup;
                }
                const char *s = (*env)->GetStringUTFChars(env, js, NULL);
                if (!s || (*env)->ExceptionCheck(env)) {
                    if (s) (*env)->ReleaseStringUTFChars(env, js, s);
                    (*env)->DeleteLocalRef(env, js);
                    goto save_cleanup;
                }
                labels[i] = strdup(s);
                (*env)->ReleaseStringUTFChars(env, js, s);
                (*env)->DeleteLocalRef(env, js);
                if (!labels[i]) goto save_cleanup; /* strdup OOM */
                pinned = i + 1;
            }
        }
    }

    rc = fce_sem_corpus_save(corp, path, labels, label_count);

save_cleanup:
    for (int i = 0; i < pinned; i++) {
        free((void *)labels[i]);
    }
    free(labels);
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    release_handle(handle);
    return rc;
}

JNIEXPORT jlong JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nLoadCorpus(
    JNIEnv *env, jclass cls, jstring jpath) {
    (void)cls;
    if (!jpath) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "path is null");
        return 0;
    }
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    if (!path || (*env)->ExceptionCheck(env)) {
        if (path) (*env)->ReleaseStringUTFChars(env, jpath, path);
        return -1; /* OOM marshaling the path */
    }
    fce_sem_corpus_t *c = fce_sem_corpus_load(path);
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    if (!c) {
        return 0; /* missing/corrupt/foreign file → Java throws IOException */
    }
    jlong h = alloc_handle(c);
    if (h < 0) {
        fce_sem_corpus_free(c);
        return -1; /* handle table full / OOM */
    }
    return h;
}

JNIEXPORT jobjectArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetDocLabels(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return NULL;
    int n = fce_sem_corpus_doc_label_count(corp);
    jclass strcls = (*env)->FindClass(env, "java/lang/String");
    if (!strcls) {
        release_handle(handle);
        return NULL;
    }
    jobjectArray arr = (*env)->NewObjectArray(env, n, strcls, NULL);
    (*env)->DeleteLocalRef(env, strcls);
    if (!arr) {
        release_handle(handle);
        return NULL; /* pending OOM */
    }
    for (int i = 0; i < n; i++) {
        const char *lbl = fce_sem_corpus_doc_label(corp, i);
        /* Labels are bytes from a (possibly untrusted) mmap'd cache file. The
         * loader guarantees NUL-termination but not valid modified UTF-8;
         * NewStringUTF has undefined behavior / can throw on malformed input.
         * Clear any thrown exception and substitute an empty string so a crafted
         * cache cannot abort label retrieval. */
        jstring js = (*env)->NewStringUTF(env, lbl ? lbl : "");
        if ((*env)->ExceptionCheck(env)) {
            (*env)->ExceptionClear(env);
            js = NULL;
        }
        if (!js) {
            js = (*env)->NewStringUTF(env, "");
            if (!js) {
                release_handle(handle);
                return NULL; /* pending OOM */
            }
        }
        (*env)->SetObjectArrayElement(env, arr, i, js);
        (*env)->DeleteLocalRef(env, js);
        /* SetObjectArrayElement can throw (e.g. OOM); making further JNI calls
         * with a pending exception is undefined, so bail out immediately. */
        if ((*env)->ExceptionCheck(env)) {
            release_handle(handle);
            return NULL;
        }
    }
    release_handle(handle);
    return arr;
}

JNIEXPORT void JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddDoc(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jtokens) {
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return;
    /* null-guard the array argument. GetArrayLength on
     * NULL is UB per JNI spec and SIGSEGVs the JVM on HotSpot. */
    if (!jtokens) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "tokens is null");
        release_handle(handle);
        return;
    }
    int count = (*env)->GetArrayLength(env, jtokens);
    /* malloc(0) is implementation-defined and may return
     * NULL, which the OOM guard below would treat as an allocation failure.
     * Return early on empty input — matching nAddDocsBatch (:329). */
    if (count == 0) {
        release_handle(handle);
        return;
    }
    /* Deep-copy each token and release its JNI string + local ref immediately,
     * so at most one local reference is live at a time. Caching one jstring ref
     * per token would accumulate `count` references for a single document and
     * overflow the JNI local reference table on token-heavy docs (ART hard-caps
     * it at 512). calloc so unfilled slots are NULL and free(NULL)-safe. */
    const char **tokens = (const char **)calloc(count, sizeof(char *));
    if (!tokens) {
        throw_oom(env, "nAddDoc: token array allocation failed");
        release_handle(handle);
        return;
    }
    int pinned = 0;
    for (int i = 0; i < count; i++) {
        jstring jtok = (jstring)(*env)->GetObjectArrayElement(env, jtokens, i);
        if ((*env)->ExceptionCheck(env)) {
            /* GetObjectArrayElement may throw
             * (e.g., ArrayIndexOutOfBoundsException) while jtok holds a valid
             * local ref. Release it before jumping to cleanup to prevent
             * leaking a JNI local reference on each exception. */
            if (jtok) (*env)->DeleteLocalRef(env, jtok);
            goto adddoc_cleanup;
        }
        if (!jtok) {
            if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null token in array");
            goto adddoc_cleanup;
        }
        const char *s = (*env)->GetStringUTFChars(env, jtok, NULL);
        if (!s || (*env)->ExceptionCheck(env)) {
            if (s) (*env)->ReleaseStringUTFChars(env, jtok, s);
            (*env)->DeleteLocalRef(env, jtok);
            goto adddoc_cleanup;
        }
        tokens[i] = strdup(s);
        (*env)->ReleaseStringUTFChars(env, jtok, s);
        (*env)->DeleteLocalRef(env, jtok);
        if (!tokens[i]) {
            throw_oom(env, "nAddDoc: token copy failed");
            goto adddoc_cleanup;
        }
        pinned = i + 1;
    }
    fce_sem_corpus_add_doc(corp, tokens, count);
adddoc_cleanup:
    for (int i = 0; i < pinned; i++) {
        free((void *)tokens[i]);
    }
    free(tokens);
    release_handle(handle);
}

JNIEXPORT jintArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddDocsBatch(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jdocs, jint maxTokensPerDoc) {
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return NULL;
    /* null-guard the array argument. */
    if (!jdocs) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "docs is null");
        release_handle(handle);
        return NULL;
    }
    int docCount = (*env)->GetArrayLength(env, jdocs);
    if (docCount == 0 || maxTokensPerDoc <= 0) {
        release_handle(handle);
        return NULL;
    }

    /* J-05: Clamp docCount for trusted callers.
     * Note: 1M docs × 512 tokens × sizeof(char*) ≈ 4 GB in all_tokens alone,
     * so the clamp limits worst-case allocation but is not a hard memory bound.
     * The C layer enforces the real corpus limits. */
    enum { MAX_BATCH_DOCS = 1000000 };
    if (docCount > MAX_BATCH_DOCS) {
        fce_log_warn("nAddDocsBatch.clamped", "docCount", "exceeds 1M limit");
        docCount = MAX_BATCH_DOCS;
    }

    /* clamp maxTokensPerDoc to FCE_SEM_MAX_TOKENS
     * (512). Without this, a single doc with >512 tokens causes the C batch
     * function to reject the entire batch silently. Matching the single-doc
     * path (addDoc) which truncates per-doc, not per-batch. */
    if (maxTokensPerDoc > FCE_SEM_MAX_TOKENS) maxTokensPerDoc = FCE_SEM_MAX_TOKENS;

    /* Build the flat token array: all_tokens[doc * maxTokensPerDoc + tok].
     * Each token is deep-copied (strdup) and its JNI string + local ref are
     * released immediately, so at most one local reference is live at a time.
     * Caching one jstring local ref per token would accumulate up to
     * docCount*maxTokensPerDoc references and overflow the JNI local reference
     * table (hard-capped at 512 on Android ART), crashing the JVM on large
     * batches.
     * Clamp per-doc token count to maxTokensPerDoc to prevent heap overflow. */
    size_t flat = (size_t)docCount * (size_t)maxTokensPerDoc;
    char **all_tokens = (char **)calloc(flat, sizeof(char *));
    int *token_counts = (int *)malloc(sizeof(int) * docCount);
    if (!all_tokens || !token_counts) {
        free(all_tokens);
        free(token_counts);
        throw_oom(env, "nAddDocsBatch: token buffer allocation failed");
        release_handle(handle);
        return NULL;
    }

    /* hoist both result pointers to NULL before
     * the first goto so all error paths return a defined value and doc_map
     * is always safe to free in the cleanup block. gcc -Wmaybe-uninitialized
     * catches this class of bug; clang's sometimes-uninitialized does not. */
    jintArray jresult = NULL;
    int *doc_map = NULL;

    for (int d = 0; d < docCount; d++) {
        jobjectArray jdoc = (jobjectArray)(*env)->GetObjectArrayElement(env, jdocs, d);
        if ((*env)->ExceptionCheck(env)) {
            if (jdoc) (*env)->DeleteLocalRef(env, jdoc);
            goto addbatch_cleanup;
        }
        /* null-guard the inner sub-array before GetArrayLength.
         * A null element inside jdocs yields jdoc == NULL with no pending exception,
         * then GetArrayLength(env, NULL) is UB. Currently unreachable because
         * Corpus.addDocsBatch iterates in Java first, but defends against future
         * callers of the package-private path. */
        if (!jdoc) {
            if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null document array");
            goto addbatch_cleanup;
        }
        int len = (*env)->GetArrayLength(env, jdoc);
        if (len > maxTokensPerDoc) len = maxTokensPerDoc;
        token_counts[d] = len;
        for (int t = 0; t < len; t++) {
            jstring jtok = (jstring)(*env)->GetObjectArrayElement(env, jdoc, t);
            if ((*env)->ExceptionCheck(env)) {
                if (jtok) (*env)->DeleteLocalRef(env, jtok);
                (*env)->DeleteLocalRef(env, jdoc);
                goto addbatch_cleanup;
            }
            if (!jtok) {
                if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null token in doc array");
                (*env)->DeleteLocalRef(env, jdoc);
                goto addbatch_cleanup;
            }
            const char *tok = (*env)->GetStringUTFChars(env, jtok, NULL);
            if (!tok || (*env)->ExceptionCheck(env)) {
                if (tok) (*env)->ReleaseStringUTFChars(env, jtok, tok);
                (*env)->DeleteLocalRef(env, jtok);
                (*env)->DeleteLocalRef(env, jdoc);
                goto addbatch_cleanup;
            }
            char *copy = strdup(tok);
            (*env)->ReleaseStringUTFChars(env, jtok, tok);
            (*env)->DeleteLocalRef(env, jtok);
            if (!copy) {
                /* strdup OOM: no pending JNI exception, but cannot proceed. The C
                 * batch add has not run yet, so nothing was added — surface OOM. */
                throw_oom(env, "nAddDocsBatch: token copy failed");
                (*env)->DeleteLocalRef(env, jdoc);
                goto addbatch_cleanup;
            }
            all_tokens[(size_t)d * maxTokensPerDoc + t] = copy;
        }
        (*env)->DeleteLocalRef(env, jdoc);
    }

    /* allocate doc_map to track which docs
     * were accepted by the C side, so Java can map paths correctly. */
    doc_map = (int *)malloc(sizeof(int) * docCount);
    if (!doc_map) {
        /* C batch add has not run yet, so nothing was added — surface OOM
         * rather than returning NULL (which Java reads as the docs-added
         * "use sequential mapping" fallback). */
        throw_oom(env, "nAddDocsBatch: doc map allocation failed");
        goto addbatch_cleanup;
    }

    fce_sem_corpus_add_docs_batch(corp, all_tokens, token_counts, docCount, maxTokensPerDoc, doc_map);

    /* Convert doc_map to jintArray for Java. */
    jresult = (*env)->NewIntArray(env, docCount);
    if (!jresult) {
        /* On some JVMs NewIntArray returns NULL without a pending exception.
         * Java reads a NULL result as "use sequential mapping", which would
         * misassign file-path labels; throw OOM so the caller fails loudly. */
        if (!(*env)->ExceptionCheck(env)) {
            throw_oom(env, "nAddDocsBatch: NewIntArray failed");
        }
    } else {
        (*env)->SetIntArrayRegion(env, jresult, 0, docCount, doc_map);
    }

addbatch_cleanup:
    /* free(NULL) is safe — doc_map is NULL on every error path that jumps
     * here before the malloc above, or after a failed malloc. */
    free(doc_map);
    /* Free the deep-copied token strings. all_tokens was calloc'd, so slots
     * never filled (a mid-doc bail) are NULL and free(NULL) is safe. The JNI
     * strings + local refs were already released inside the extraction loop,
     * so no JNI calls are needed here. */
    for (size_t idx = 0; idx < flat; idx++) {
        free(all_tokens[idx]);
    }
    free(all_tokens);
    free(token_counts);
    release_handle(handle);
    return jresult;
}

JNIEXPORT jboolean JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nFinalizeCorpus(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env;
    (void)cls;
    /* use acquire_handle / release_handle.
     * The refcount protects the corpus during the long-running finalize
     * even if close() is called concurrently from another thread. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return JNI_FALSE;
    int rc = fce_sem_corpus_finalize(corp);
    release_handle(handle);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddDocsTokenized(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jnames) {
    (void)cls;
    /* Partial-state semantics — on mid-batch failure
     * (null element, OOM), earlier batches that were already committed via
     * fce_sem_corpus_add_docs_tokenized remain in the corpus. The caller
     * sees an exception and must decide whether to continue, roll back, or
     * discard the corpus. This is a deliberate trade-off: atomic rollback
     * across potentially millions of tokens would require a full corpus
     * snapshot and copy, which is not justified for a batch-add API. */
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return;
    /* null-guard the array argument. */
    if (!jnames) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "names is null");
        release_handle(handle);
        return;
    }
    int total = (*env)->GetArrayLength(env, jnames);
    if (total == 0) {
        release_handle(handle);
        return;
    }

    /* reduce BATCH from 10000 to 1000 to cut
     * per-call heap allocation from ~160 KB to ~16 KB. This bounds allocation
     * pressure on constrained environments (Android JNI threads with small
     * heaps) while still providing good batching throughput. */
    const int BATCH = 1000;
    const char **names = (const char **)malloc(sizeof(char *) * BATCH);
    if (!names) {
        throw_oom(env, "nAddDocsTokenized: name buffer allocation failed");
        release_handle(handle);
        return;
    }

    for (int start = 0; start < total; start += BATCH) {
        int end = start + BATCH;
        if (end > total) end = total;
        int batch = end - start;

        /* Extract batch of strings: deep-copy each and release its JNI string +
         * local ref immediately, so at most one local reference is live at a time
         * (caching them would overflow the local reference table, hard-capped at
         * 512 on Android ART). */
        int pinned = 0;
        for (int i = 0; i < batch; i++) {
            jstring jname = (jstring)(*env)->GetObjectArrayElement(env, jnames, start + i);
            if ((*env)->ExceptionCheck(env)) {
                if (jname) (*env)->DeleteLocalRef(env, jname);
                goto addtok_cleanup;
            }
            if (!jname) {
                if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null token in array");
                goto addtok_cleanup;
            }
            const char *s = (*env)->GetStringUTFChars(env, jname, NULL);
            if (!s || (*env)->ExceptionCheck(env)) {
                if (s) (*env)->ReleaseStringUTFChars(env, jname, s);
                (*env)->DeleteLocalRef(env, jname);
                goto addtok_cleanup;
            }
            names[i] = strdup(s);
            (*env)->ReleaseStringUTFChars(env, jname, s);
            (*env)->DeleteLocalRef(env, jname);
            if (!names[i]) {
                throw_oom(env, "nAddDocsTokenized: name copy failed");
                goto addtok_cleanup;
            }
            pinned = i + 1;
        }

        /* Tokenize + add to corpus */
        fce_sem_corpus_add_docs_tokenized(corp, names, batch);

    addtok_cleanup:
        /* Free the deep-copied strings (JNI refs already released above). */
        for (int i = 0; i < pinned; i++) {
            free((void *)names[i]);
            names[i] = NULL;
        }
        /* if an exception is pending (NPE from
         * null element, OOM from GetStringUTFChars, or AIOOBE from
         * GetObjectArrayElement), the outer loop must not re-enter
         * JNI — calling GetObjectArrayElement / GetStringUTFChars with
         * a pending exception is UB per JNI spec §"Exceptions" and
         * crashes the JVM on HotSpot. */
        if ((*env)->ExceptionCheck(env)) break;
    }
    free(names);
    release_handle(handle);
}

JNIEXPORT jint JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddFiles(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jpaths, jint chunkSize,
    jintArray jFileDocCounts, jint maxTokensPerChunk) {
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return 0;
    /* null jpaths → GetArrayLength is UB and
     * SIGSEGVs the JVM on HotSpot. Throw NPE before reaching the JNI call. */
    if (!jpaths) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "paths is null");
        release_handle(handle);
        return -1;
    }
    int count = (*env)->GetArrayLength(env, jpaths);
    if (count == 0 || chunkSize <= 0) {
        release_handle(handle);
        return 0;
    }

    /* Extract all path strings: deep-copy each and release its JNI string +
     * local ref immediately, so at most one local reference is live at a time.
     * Caching one jstring ref per file would accumulate `count` references and
     * overflow the JNI local reference table on large trees (ART hard-caps it
     * at 512), crashing the JVM. */
    const char **paths = (const char **)calloc(count, sizeof(char *));
    if (!paths) {
        throw_oom(env, "nAddFiles: path buffer allocation failed");
        release_handle(handle);
        return -1;
    }
    /* hoist file_doc_counts declaration above the
     * for-loop to prevent goto from jumping over its initialization. */
    int *file_doc_counts = NULL;

    /* initialize result before any goto can bypass the assignment. */
    int result = -1;
    int pinned = 0;
    for (int i = 0; i < count; i++) {
        jstring jpath = (jstring)(*env)->GetObjectArrayElement(env, jpaths, i);
        if ((*env)->ExceptionCheck(env)) {
            if (jpath) (*env)->DeleteLocalRef(env, jpath);
            goto addfiles_cleanup;
        }
        if (!jpath) {
            if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null path in array");
            goto addfiles_cleanup;
        }
        const char *s = (*env)->GetStringUTFChars(env, jpath, NULL);
        if (!s || (*env)->ExceptionCheck(env)) {
            /* OOM: a pending OutOfMemoryError may be set. Bail before any
             * further JNI calls (which would be UB) or C function calls
             * (which would be invoked with a pending exception). */
            if (s) (*env)->ReleaseStringUTFChars(env, jpath, s);
            (*env)->DeleteLocalRef(env, jpath);
            goto addfiles_cleanup;
        }
        paths[i] = strdup(s);
        (*env)->ReleaseStringUTFChars(env, jpath, s);
        (*env)->DeleteLocalRef(env, jpath);
        if (!paths[i]) {
            throw_oom(env, "nAddFiles: path copy failed");
            goto addfiles_cleanup;
        }
        pinned = i + 1;
    }

    /* Per-file doc counts output buffer. */
    if (jFileDocCounts && (*env)->GetArrayLength(env, jFileDocCounts) >= count) {
        file_doc_counts = (int *)malloc(sizeof(int) * count);
        if (file_doc_counts) memset(file_doc_counts, 0, sizeof(int) * count);
    }
    /* One C call: read + chunk + tokenize + add to corpus. */
    result = fce_sem_corpus_add_files(corp, paths, count, chunkSize, file_doc_counts,
                                      maxTokensPerChunk);

    /* Copy per-file counts back to Java array. */
    if (file_doc_counts && jFileDocCounts) {
        (*env)->SetIntArrayRegion(env, jFileDocCounts, 0, count, file_doc_counts);
    }
    free(file_doc_counts);

addfiles_cleanup:
    for (int i = 0; i < pinned; i++) {
        free((void *)paths[i]);
    }
    free(paths);
    release_handle(handle);
    return result;
}

JNIEXPORT jfloat JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetIdf(
    JNIEnv *env, jclass cls, jlong handle, jstring jtoken) {
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return 0.0f;
    /* null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM on HotSpot. Throw NPE before reaching the JNI call. */
    if (!jtoken) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "token is null");
        release_handle(handle);
        return 0.0f;
    }
    const char *tok = (*env)->GetStringUTFChars(env, jtoken, NULL);
    if (!tok || (*env)->ExceptionCheck(env)) {
        /* if GetStringUTFChars returned non-NULL
         * but an exception is pending (some JVMs set OOM and still return a
         * copy), release the string to avoid a native memory leak. */
        if (tok) (*env)->ReleaseStringUTFChars(env, jtoken, tok);
        release_handle(handle);
        return 0.0f;
    }
    float idf = fce_sem_corpus_idf(corp, tok);
    (*env)->ReleaseStringUTFChars(env, jtoken, tok);
    release_handle(handle);
    return idf;
}

JNIEXPORT jfloatArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetRiVec(
    JNIEnv *env, jclass cls, jlong handle, jstring jtoken) {
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return NULL;
    /* null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM on HotSpot. Throw NPE before reaching the JNI call. */
    if (!jtoken) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "token is null");
        release_handle(handle);
        return NULL;
    }
    const char *tok = (*env)->GetStringUTFChars(env, jtoken, NULL);
    if (!tok || (*env)->ExceptionCheck(env)) {
        if (tok) (*env)->ReleaseStringUTFChars(env, jtoken, tok);
        release_handle(handle);
        return NULL;
    }
    /* fce_sem_corpus_ri_vec returns a pointer to a
     * _Thread_local fce_sem_vec_t (semantic.c tl_dequant). The data is
     * immediately copied out via SetFloatArrayRegion below, so the TLS
     * pointer's lifetime does not matter once the Java array is filled
     * — the JVM owns its own storage. No need to copy into a fresh
     * allocation. The function is safe across JNI calls because the
     * _Thread_local scratch is per-Java-thread (and the C side, being
     * a JNI_OnLoad-attached dylib, is always called from a Java
     * thread that has a stable _Thread_local key). */
    const fce_sem_vec_t *vec = fce_sem_corpus_ri_vec(corp, tok);
    (*env)->ReleaseStringUTFChars(env, jtoken, tok);
    if (!vec) {
        release_handle(handle);
        return NULL;
    }
    jfloatArray result = (*env)->NewFloatArray(env, FCE_SEM_DIM);
    /* NewFloatArray can fail with pending OOM.
     * release_handle MUST be called on every path after acquire_handle.
     * J-01: NewFloatArray may return NULL without
     * setting a pending exception on some JVMs — check for NULL explicitly. */
    if (!result || (*env)->ExceptionCheck(env)) {
        release_handle(handle);
        return NULL;
    }
    (*env)->SetFloatArrayRegion(env, result, 0, FCE_SEM_DIM, vec->v);
    release_handle(handle);
    return result;
}

JNIEXPORT jint JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetDocCount(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env;
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return 0;
    int count = fce_sem_corpus_doc_count(corp);
    release_handle(handle);
    return count;
}

JNIEXPORT jint JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetTokenCount(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env;
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return 0;
    int count = fce_sem_corpus_token_count(corp);
    release_handle(handle);
    return count;
}

JNIEXPORT jint JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetDim(
    JNIEnv *env, jclass cls) {
    (void)env;
    (void)cls;
    /* Expose the compile-time dimension to Java so the bindings work for
     * both the default 768-dim build and 256-dim builds compiled with
     * -DFCE_SEM_DIM_256. */
    return (jint)FCE_SEM_DIM;
}

/* ── Static Nomic Search JNI ────────────────────────────────────── */

JNIEXPORT void JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_init(
    JNIEnv *env, jclass cls) {
    (void)env;
    (void)cls;
    fce_sem_ensure_ready();
}

JNIEXPORT jobjectArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_tokenize(
    JNIEnv *env, jclass cls, jstring jname) {
    (void)cls;
    /* null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call. */
    if (!jname) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "name is null");
        return NULL;
    }
    const char *name = (*env)->GetStringUTFChars(env, jname, NULL);
    if (!name || (*env)->ExceptionCheck(env)) {
        if (name) (*env)->ReleaseStringUTFChars(env, jname, name);
        return NULL;
    }

    char *tokens[FCE_SEM_MAX_TOKENS];
    /* Zero the array so the cleanup path can safely free every slot up to count,
     * even if the count is less than FCE_SEM_MAX_TOKENS. */
    memset(tokens, 0, sizeof(tokens));
    int count = fce_sem_tokenize(name, tokens, FCE_SEM_MAX_TOKENS);
    /* fce_sem_tokenize now uses explicit ASCII-range
     * checks ([a-zA-Z0-9] only), so tokens contain only ASCII bytes.
     * NewStringUTF is therefore safe — no non-ASCII bytes can reach it. */

    jclass strCls = (*env)->FindClass(env, "java/lang/String");
    if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup;
    jobjectArray result = (*env)->NewObjectArray(env, count, strCls, NULL);
    (*env)->DeleteLocalRef(env, strCls);
    if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup;
    for (int i = 0; i < count; i++) {
        jstring jtok = (*env)->NewStringUTF(env, tokens[i]);
        if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup;
        (*env)->SetObjectArrayElement(env, result, i, jtok);
        (*env)->DeleteLocalRef(env, jtok);
        free(tokens[i]);
        tokens[i] = NULL;
    }

    (*env)->ReleaseStringUTFChars(env, jname, name);
    return result;

tokenize_cleanup:
    /* Latent-3: safety of this cleanup loop depends on two
     * invariants working together:
     * 1. tokens[] is zeroed up front (memset at :699), so every slot
     * holds NULL unless fce_sem_tokenize wrote to it.
     * 2. The success loop (:707-713) sets tokens[i] = NULL after free,
     * so already-consumed slots are harmless to free() again.
     * A future edit that removes either the up-front memset or the
     * per-iteration NULLing will introduce a double-free here.
     * Guard: only free non-NULL slots. */
    for (int i = 0; i < count; i++) {
        if (tokens[i]) free(tokens[i]);
    }
    (*env)->ReleaseStringUTFChars(env, jname, name);
    return NULL;
}

JNIEXPORT jobjectArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nTokenizeBatch(
    JNIEnv *env, jclass cls, jobjectArray jnames) {
    (void)cls;
    /* null-guard the array argument. */
    if (!jnames) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "names is null");
        return NULL;
    }
    int count = (*env)->GetArrayLength(env, jnames);
    if (count == 0) {
        jclass strArrCls = (*env)->FindClass(env, "[Ljava/lang/String;");
        CHECK_EXCEPTION_RETURN(env, NULL);
        jobjectArray empty = (*env)->NewObjectArray(env, 0, strArrCls, NULL);
        (*env)->DeleteLocalRef(env, strArrCls);
        return empty;
    }

    /* Extract input strings: deep-copy each and release its JNI string +
     * local ref immediately, so at most one local reference is live at a time.
     * Caching one jstring ref per name would accumulate `count` references and
     * overflow the JNI local reference table on large batches (ART hard-caps
     * it at 512). */
    const char **names = (const char **)malloc(sizeof(char *) * count);
    if (!names) { return NULL; }
    int pinned = 0;
    for (int i = 0; i < count; i++) {
        jstring jname = (jstring)(*env)->GetObjectArrayElement(env, jnames, i);
        if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup_input;
        if (!jname) {
            if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null token in array");
            goto tokenize_cleanup_input;
        }
        const char *s = (*env)->GetStringUTFChars(env, jname, NULL);
        if (!s || (*env)->ExceptionCheck(env)) {
            if (s) (*env)->ReleaseStringUTFChars(env, jname, s);
            (*env)->DeleteLocalRef(env, jname);
            goto tokenize_cleanup_input;
        }
        names[i] = strdup(s);
        (*env)->ReleaseStringUTFChars(env, jname, s);
        (*env)->DeleteLocalRef(env, jname);
        if (!names[i]) {
            /* Surface OOM as an exception. Returning a bare NULL result is
             * indistinguishable from "empty input" to the Java caller. */
            throw_oom(env, "nTokenizeBatch: name copy failed");
            goto tokenize_cleanup_input;
        }
        pinned = i + 1;
    }

    /* Tokenize all in C */
    int max_out = FCE_SEM_MAX_TOKENS;
    /* defense-in-depth overflow check.
     * On 64-bit (required by _Static_assert in semantic.h), count * max_out
     * fits in size_t, but this explicit check prevents issues if the
     * requirement is ever relaxed. */
    size_t flat_sz = (size_t)count * (size_t)max_out;
    if (flat_sz > (SIZE_MAX / sizeof(char *))) goto tokenize_cleanup_input;
    char **all_tokens = (char **)calloc(flat_sz, sizeof(char *));
    int *token_counts = (int *)malloc(count * sizeof(int));
    if (!all_tokens || !token_counts) {
        free(all_tokens);
        free(token_counts);
        goto tokenize_cleanup_input;
    }
    fce_sem_tokenize_batch(names, count, all_tokens, token_counts, max_out);

    /* Free copied input strings (JNI refs already released during extraction). */
    for (int i = 0; i < count; i++) {
        free((void *)names[i]);
    }
    free(names);

    /* Build Java String[][] result */
    {
        jobjectArray result = NULL;
        jclass strCls = (*env)->FindClass(env, "java/lang/String");
        if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup_tokens;
        jclass strArrCls = (*env)->FindClass(env, "[Ljava/lang/String;");
        if ((*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, strCls);
            goto tokenize_cleanup_tokens;
        }
        result = (*env)->NewObjectArray(env, count, strArrCls, NULL);
        (*env)->DeleteLocalRef(env, strArrCls);
        if ((*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, strCls);
            goto tokenize_cleanup_tokens;
        }
        /* Track the most recent docTokens array so an OOM/error exit from
         * the inner loop does not leak its local reference. */
        jobjectArray pending_doc_tokens = NULL;
        for (int i = 0; i < count; i++) {
            jobjectArray docTokens = (*env)->NewObjectArray(env, token_counts[i], strCls, NULL);
            if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup_result;
            pending_doc_tokens = docTokens;
            for (int t = 0; t < token_counts[i]; t++) {
                jstring jtok = (*env)->NewStringUTF(env, all_tokens[(size_t)i * max_out + t]);
                /* NewStringUTF can return NULL and raise
                 * OutOfMemoryError. Without an ExceptionCheck, the subsequent
                 * SetObjectArrayElement / next NewStringUTF would be JNI UB.
                 * Mirror the guard already used by single-doc tokenize (:620-627). */
                if (!jtok || (*env)->ExceptionCheck(env)) {
                    free(all_tokens[(size_t)i * max_out + t]);
                    all_tokens[(size_t)i * max_out + t] = NULL;
                    goto tokenize_cleanup_result;
                }
                (*env)->SetObjectArrayElement(env, docTokens, t, jtok);
                (*env)->DeleteLocalRef(env, jtok);
                free(all_tokens[(size_t)i * max_out + t]);
                all_tokens[(size_t)i * max_out + t] = NULL;
            }
            (*env)->SetObjectArrayElement(env, result, i, docTokens);
            if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup_result;
            (*env)->DeleteLocalRef(env, docTokens);
            pending_doc_tokens = NULL;
        }
        (*env)->DeleteLocalRef(env, strCls);
        free(all_tokens);
        free(token_counts);
        return result;

    tokenize_cleanup_result:
        if (result) (*env)->DeleteLocalRef(env, result);
        if (pending_doc_tokens) (*env)->DeleteLocalRef(env, pending_doc_tokens);
        (*env)->DeleteLocalRef(env, strCls);
    }

tokenize_cleanup_tokens:
    for (int i = 0; i < count; i++) {
        for (int t = 0; t < token_counts[i]; t++) {
            char *tok = all_tokens[(size_t)i * max_out + t];
            if (tok) {
                free(tok);
                all_tokens[(size_t)i * max_out + t] = NULL;
            }
        }
    }
    free(all_tokens);
    free(token_counts);
    return NULL;

tokenize_cleanup_input:
    for (int i = 0; i < pinned; i++) {
        free((void *)names[i]);
    }
    free(names);
    return NULL;
}

JNIEXPORT jfloat JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_proximity(
    JNIEnv *env, jclass cls, jstring jpathA, jstring jpathB) {
    (void)cls;
    /* null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call. */
    if (!jpathA) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "pathA is null");
        return 0.0f;
    }
    if (!jpathB) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "pathB is null");
        return 0.0f;
    }
    const char *a = (*env)->GetStringUTFChars(env, jpathA, NULL);
    /* NULL check on GetStringUTFChars (pending OOM). */
    if (!a || (*env)->ExceptionCheck(env)) {
        if (a) (*env)->ReleaseStringUTFChars(env, jpathA, a);
        return 0.0f;
    }
    const char *b = (*env)->GetStringUTFChars(env, jpathB, NULL);
    if (!b || (*env)->ExceptionCheck(env)) {
        if (b) (*env)->ReleaseStringUTFChars(env, jpathB, b);
        (*env)->ReleaseStringUTFChars(env, jpathA, a);
        return 0.0f;
    }
    float score = fce_sem_proximity(a, b);
    (*env)->ReleaseStringUTFChars(env, jpathA, a);
    (*env)->ReleaseStringUTFChars(env, jpathB, b);
    return score;
}

/* ── Simple API ─────────────────────────────────────────────────── */

/* NOTE: simpleRank/simpleSearch hold 2N JNI local references (index + weight
 * arrays) alive until the cleanup pass. HotSpot auto-grows the local ref table
 * but may abort under sustained pressure. For large corpora (N > 1000), use
 * nSimpleRankFlat which bypasses per-item JNI marshaling entirely.
 * This object-array path is the older, non-preferred model (the Java docs
 * steer callers to extractFlat + simpleRankBatch); it stays supported for
 * small corpora and is hard-gated below at corpusSize > 4096. */

JNIEXPORT jfloat JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_simpleScore(
    JNIEnv *env, jclass cls, jobject ja, jobject jb) {
    (void)cls;
    char *pathA = NULL, *pathB = NULL;
    jintArray jindA = NULL, jindB = NULL;
    jfloatArray jwgtA = NULL, jwgtB = NULL;
    fce_sem_func_t a = marshal_func(env, ja, &pathA, &jindA, &jwgtA);
    /* marshal_func can throw (NullPointerException,
     * IllegalArgumentException, OOM). Must check before calling marshal_func
     * again — JNI spec forbids most Get* calls with a pending exception. */
    if ((*env)->ExceptionCheck(env)) {
        unmarshal_func(env, ja, &a, jindA, jwgtA, pathA);
        return 0.0f;
    }
    fce_sem_func_t b = marshal_func(env, jb, &pathB, &jindB, &jwgtB);
    float score = 0.0f;
    if (!(*env)->ExceptionCheck(env)) {
        score = fce_sem_simple_score(&a, &b);
    }
    unmarshal_func(env, ja, &a, jindA, jwgtA, pathA);
    unmarshal_func(env, jb, &b, jindB, jwgtB, pathB);
    return score;
}

JNIEXPORT jobjectArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_simpleRank(
    JNIEnv *env, jclass cls, jobject jquery, jobjectArray jcorpus, jint topK) {
    (void)cls;

    /* null-guard the array argument. */
    if (!jcorpus) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "corpus is null");
        return NULL;
    }
    int corpusSize = (*env)->GetArrayLength(env, jcorpus);
    if (corpusSize <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* this deprecated path holds 2N JNI local refs alive
     * until cleanup, which can exhaust the local-ref table on large corpora.
     * Reject corpora larger than 4096 — callers should use nSimpleRankFlat. */
    if (corpusSize > 4096) {
        if (cls_illegal_arg) {
            (*env)->ThrowNew(env, cls_illegal_arg,
                             "corpus too large for simpleRank; use simpleRankBatch (nSimpleRankFlat) for corpora > 4096");
        }
        return NULL;
    }
    /* clamp topK to corpus size before allocating
     * result buffers to prevent allocation-amplification from hostile callers. */
    if (topK > corpusSize) topK = corpusSize;
    /* zero-initialization is LOAD-BEARING for the cleanup
     * path — the *Cleanup labels iterate corpusSize and skip slots where
     * jindices_arr[i] || jweights_arr[i] is NULL (i.e., never marshaled).
     * A future switch to malloc would turn cleanup into wild-pointer reads
     * and free() of indeterminate pointers. Do NOT change to malloc. */
    fce_sem_func_t *corpus = (fce_sem_func_t *)calloc(corpusSize, sizeof(fce_sem_func_t));
    char **paths = (char **)calloc(corpusSize, sizeof(char *));
    jobject *corp_refs = (jobject *)calloc(corpusSize, sizeof(jobject));
    jintArray *jindices_arr = (jintArray *)calloc(corpusSize, sizeof(jintArray));
    jfloatArray *jweights_arr = (jfloatArray *)calloc(corpusSize, sizeof(jfloatArray));
    fce_sem_ranked_t *results = NULL;
    jobjectArray jresults = NULL;

    if (!corpus || !paths || !corp_refs || !jindices_arr || !jweights_arr) {
        free(corpus);
        free(paths);
        free(corp_refs);
        free(jindices_arr);
        free(jweights_arr);
        return NULL;
    }

    /* Marshal query */
    char *qpath = NULL;
    jintArray qjindices = NULL;
    jfloatArray qjweights = NULL;
    fce_sem_func_t query = marshal_func(env, jquery, &qpath, &qjindices, &qjweights);
    if ((*env)->ExceptionCheck(env)) goto simpleRank_cleanup;

    /* Marshal corpus — cache jobject refs for cleanup. */
    for (int i = 0; i < corpusSize; i++) {
        jobject obj = (*env)->GetObjectArrayElement(env, jcorpus, i);
        if ((*env)->ExceptionCheck(env)) goto simpleRank_cleanup;
        corp_refs[i] = obj;
        corpus[i] = marshal_func(env, obj, &paths[i], &jindices_arr[i], &jweights_arr[i]);
        (*env)->DeleteLocalRef(env, obj);
        corp_refs[i] = NULL;
        if ((*env)->ExceptionCheck(env)) goto simpleRank_cleanup;
    }

    results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) {
        throw_oom(env, "simpleRank: results allocation failed");
        goto simpleRank_cleanup;
    }
    uint32_t count = 0;
    fce_sem_simple_rank(&query, corpus, corpusSize, topK, results, &count);

    /* Build Java array using cached class/method IDs */
    jresults = (*env)->NewObjectArray(env, count, cls_search_result, NULL);
    if ((*env)->ExceptionCheck(env)) {
        jresults = NULL;
        goto simpleRank_cleanup;
    }
    for (uint32_t i = 0; i < count; i++) {
        jobject jres = (*env)->NewObject(env, cls_search_result, ctor_search_result,
                                         results[i].index, results[i].score);
        /* NewObject can throw OOM and return NULL.
         * Continuing JNI calls with a pending exception is UB per the spec.
         * delete partially-built jresults before
         * jumping to cleanup. */
        if (!jres || (*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, jresults);
            jresults = NULL;
            goto simpleRank_cleanup;
        }
        (*env)->SetObjectArrayElement(env, jresults, i, jres);
        (*env)->DeleteLocalRef(env, jres);
        /* SetObjectArrayElement can throw; bail before any further JNI call. */
        if ((*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, jresults);
            jresults = NULL;
            goto simpleRank_cleanup;
        }
    }

simpleRank_cleanup:
    unmarshal_func(env, jquery, &query, qjindices, qjweights, qpath);
    for (int i = 0; i < corpusSize; i++) {
        if (jindices_arr[i] || jweights_arr[i]) {
            unmarshal_func(env, corp_refs[i], &corpus[i], jindices_arr[i], jweights_arr[i], paths[i]);
            paths[i] = NULL; /* path ownership transferred to unmarshal_func */
        }
        /* free the path unconditionally. The path is
         * always allocated by marshal_func but unmarshal_func only frees it when
         * called (gated on having TF-IDF arrays). For RI-only descriptors with
         * no setTfidf(), paths[i] leaks. */
        free(paths[i]);
        if (corp_refs[i]) (*env)->DeleteLocalRef(env, corp_refs[i]);
    }
    free(corp_refs);
    free(jindices_arr);
    free(jweights_arr);
    free(corpus);
    free(paths);
    free(results);
    return jresults;
}

JNIEXPORT jobjectArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_simpleSearch(
    JNIEnv *env, jclass cls, jobject jquery, jobjectArray jcorpus,
    jint topK, jfloat minScore) {
    (void)cls;

    /* null-guard the array argument. */
    if (!jcorpus) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "corpus is null");
        return NULL;
    }
    int corpusSize = (*env)->GetArrayLength(env, jcorpus);
    if (corpusSize <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* this deprecated path holds 2N JNI local refs alive
     * until cleanup, which can exhaust the local-ref table on large corpora.
     * Reject corpora larger than 4096 — callers should use nSimpleRankFlat. */
    if (corpusSize > 4096) {
        if (cls_illegal_arg) {
            (*env)->ThrowNew(env, cls_illegal_arg,
                             "corpus too large for simpleSearch; use simpleRankBatch (nSimpleRankFlat) for corpora > 4096");
        }
        return NULL;
    }
    /* clamp topK to corpus size, mirroring simpleRank. */
    if (topK > corpusSize) topK = corpusSize;
    /* zero-initialization is LOAD-BEARING for the cleanup
     * path — see comment in simpleRank above. Do NOT change to malloc. */
    fce_sem_func_t *corpus = (fce_sem_func_t *)calloc(corpusSize, sizeof(fce_sem_func_t));
    char **paths = (char **)calloc(corpusSize, sizeof(char *));
    jobject *corp_refs = (jobject *)calloc(corpusSize, sizeof(jobject));
    jintArray *jindices_arr = (jintArray *)calloc(corpusSize, sizeof(jintArray));
    jfloatArray *jweights_arr = (jfloatArray *)calloc(corpusSize, sizeof(jfloatArray));
    fce_sem_ranked_t *results = NULL;
    jobjectArray jresults = NULL;

    if (!corpus || !paths || !corp_refs || !jindices_arr || !jweights_arr) {
        free(corpus);
        free(paths);
        free(corp_refs);
        free(jindices_arr);
        free(jweights_arr);
        return NULL;
    }

    char *qpath = NULL;
    jintArray qjindices = NULL;
    jfloatArray qjweights = NULL;
    fce_sem_func_t query = marshal_func(env, jquery, &qpath, &qjindices, &qjweights);
    if ((*env)->ExceptionCheck(env)) goto simpleSearch_cleanup;

    for (int i = 0; i < corpusSize; i++) {
        jobject obj = (*env)->GetObjectArrayElement(env, jcorpus, i);
        if ((*env)->ExceptionCheck(env)) goto simpleSearch_cleanup;
        corp_refs[i] = obj;
        corpus[i] = marshal_func(env, obj, &paths[i], &jindices_arr[i], &jweights_arr[i]);
        (*env)->DeleteLocalRef(env, obj);
        corp_refs[i] = NULL;
        if ((*env)->ExceptionCheck(env)) goto simpleSearch_cleanup;
    }

    results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) {
        throw_oom(env, "simpleSearch: results allocation failed");
        goto simpleSearch_cleanup;
    }
    uint32_t count = 0;
    fce_sem_simple_search(&query, corpus, corpusSize, topK, minScore, results, &count);

    jresults = (*env)->NewObjectArray(env, count, cls_search_result, NULL);
    if ((*env)->ExceptionCheck(env)) {
        jresults = NULL;
        goto simpleSearch_cleanup;
    }
    for (uint32_t i = 0; i < count; i++) {
        jobject jres = (*env)->NewObject(env, cls_search_result, ctor_search_result,
                                         results[i].index, results[i].score);
        /* NewObject can throw OOM and return NULL.
         * Continuing JNI calls with a pending exception is UB per the spec.
         * delete partially-built jresults before
         * jumping to cleanup. */
        if (!jres || (*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, jresults);
            jresults = NULL;
            goto simpleSearch_cleanup;
        }
        (*env)->SetObjectArrayElement(env, jresults, i, jres);
        (*env)->DeleteLocalRef(env, jres);
        /* SetObjectArrayElement can throw; bail before any further JNI call. */
        if ((*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, jresults);
            jresults = NULL;
            goto simpleSearch_cleanup;
        }
    }

simpleSearch_cleanup:
    unmarshal_func(env, jquery, &query, qjindices, qjweights, qpath);
    for (int i = 0; i < corpusSize; i++) {
        if (jindices_arr[i] || jweights_arr[i]) {
            unmarshal_func(env, corp_refs[i], &corpus[i], jindices_arr[i], jweights_arr[i], paths[i]);
            paths[i] = NULL; /* path ownership transferred to unmarshal_func */
        }
        /* free the path unconditionally — same as simpleRank_cleanup. */
        free(paths[i]);
        if (corp_refs[i]) (*env)->DeleteLocalRef(env, corp_refs[i]);
    }
    free(corp_refs);
    free(jindices_arr);
    free(jweights_arr);
    free(corpus);
    free(paths);
    free(results);
    return jresults;
}

/* ── Batch rank (flat arrays, zero per-item JNI overhead) ──────── */

JNIEXPORT jobjectArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSimpleRankFlat(
    JNIEnv *env, jclass cls,
    /* corpus flat arrays */
    jfloatArray j_all_weights,
    jintArray j_all_indices,
    jintArray j_tfidf_lens,
    jfloatArray j_all_ri_vecs,
    jobjectArray j_file_paths,
    jint maxTokens,
    /* query */
    jintArray j_q_indices,
    jfloatArray j_q_weights,
    jfloatArray j_q_ri_vec,
    /* output */
    jint topK) {
    (void)cls;

    jobjectArray jresults = NULL;

    /* declare all pin pointers upfront, NULL-initialized,
     * so that goto flat_cleanup_query before any pin assignment does not read
     * indeterminate stack values. Each pointer is assigned below at its pin site.
     * J2: also declare results here — goto flat_cleanup_query
     * can be reached before the original declaration site, and jumping over
     * an uninitialized declaration is a maintenance trap. */
    const float *all_weights = NULL, *all_ri_vecs = NULL;
    const int *all_indices = NULL, *tfidf_lens = NULL;
    const float *q_weights = NULL, *q_ri_vec = NULL;
    const int *q_indices = NULL;
    fce_sem_ranked_t *results = NULL;

    /* Null-guard the required array arguments BEFORE any GetArrayLength calls.
     * JNI spec says GetArrayLength(env, NULL) is undefined and crashes on HotSpot.
     * j_all_weights, j_all_indices, and j_tfidf_lens are
     * optional — the flat scorer uses RI only and callers building RI-only corpora
     * should not be forced to allocate TF-IDF arrays that are then ignored. */
    if (!j_file_paths || !j_all_ri_vecs) {
        if ((*env)->ExceptionCheck(env)) return NULL;
        /* J1: guard cls_npe for uniformity with every other
         * throw site in this file. Today JNI_OnLoad hard-fails if cls_npe
         * is NULL, but the invariant is fragile to future relaxations. */
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null array argument to nSimpleRankFlat");
        return NULL;
    }

    int corpusSize = (*env)->GetArrayLength(env, j_file_paths);
    if (corpusSize <= 0 || topK <= 0 || maxTokens <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* clamp topK to corpus size. */
    if (topK > corpusSize) topK = corpusSize;

    /* Get primitive array pointers. check ExceptionCheck
     * and NULL between each pin — Get*ArrayElements may throw OOM if it
     * needs to copy the array, and continuing JNI calls with a pending
     * exception is undefined behavior per the JNI spec.
     * TF-IDF corpus arrays are optional (flat scorer
     * uses RI only). Pin them only when provided. */
    if (j_all_weights) {
        all_weights = (*env)->GetFloatArrayElements(env, j_all_weights, NULL);
        if (!all_weights || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    }
    if (j_all_indices) {
        all_indices = (*env)->GetIntArrayElements(env, j_all_indices, NULL);
        if (!all_indices || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    }
    if (j_tfidf_lens) {
        tfidf_lens = (*env)->GetIntArrayElements(env, j_tfidf_lens, NULL);
        if (!tfidf_lens || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    }
    all_ri_vecs = (*env)->GetFloatArrayElements(env, j_all_ri_vecs, NULL);
    if (!all_ri_vecs || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;

    /* Validate array sizes against caller-supplied dimensions.
     * Compute products in size_t to avoid 32-bit overflow on large corpora.
     * TF-IDF arrays are optional. When provided, validate
     * their sizes; when absent, skip TF-IDF validation. */
    int w_len = j_all_weights ? (*env)->GetArrayLength(env, j_all_weights) : 0;
    int i_len = j_all_indices ? (*env)->GetArrayLength(env, j_all_indices) : 0;
    int tl_len = j_tfidf_lens ? (*env)->GetArrayLength(env, j_tfidf_lens) : 0;
    int rv_len = (*env)->GetArrayLength(env, j_all_ri_vecs);
    size_t needed_ri = (size_t)corpusSize * (size_t)FCE_SEM_DIM;
    if ((size_t)rv_len < needed_ri) {
        if (cls_illegal_arg) {
            char ri_size_msg[80];
            snprintf(ri_size_msg, sizeof(ri_size_msg),
                     "array size mismatch: all_ri_vecs too small for corpusSize * %d", FCE_SEM_DIM);
            (*env)->ThrowNew(env, cls_illegal_arg, ri_size_msg);
        }
        goto flat_cleanup_query;
    }
    if (j_all_weights || j_all_indices || j_tfidf_lens) {
        /* At least one TF-IDF array was provided — they must all be present
         * and correctly sized. */
        if (!j_all_weights || !j_all_indices || !j_tfidf_lens) {
            if (cls_illegal_arg) {
                (*env)->ThrowNew(env, cls_illegal_arg,
                                 "if any TF-IDF corpus array is provided, all three (weights, indices, lens) must be");
            }
            goto flat_cleanup_query;
        }
        size_t needed_tfidf = (size_t)corpusSize * (size_t)maxTokens;
        if ((size_t)w_len < needed_tfidf || (size_t)i_len < needed_tfidf ||
            (size_t)tl_len < (size_t)corpusSize) {
            if (cls_illegal_arg) {
                (*env)->ThrowNew(env, cls_illegal_arg,
                                 "array size mismatch: corpus arrays too small for corpusSize * maxTokens");
            }
            goto flat_cleanup_query;
        }
    }

    if (j_q_weights) {
        q_weights = (*env)->GetFloatArrayElements(env, j_q_weights, NULL);
        if (!q_weights || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    }
    if (j_q_indices) {
        q_indices = (*env)->GetIntArrayElements(env, j_q_indices, NULL);
        if (!q_indices || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    }
    if (j_q_ri_vec) {
        q_ri_vec = (*env)->GetFloatArrayElements(env, j_q_ri_vec, NULL);
        if (!q_ri_vec || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    }

    int q_len = j_q_indices ? (*env)->GetArrayLength(env, j_q_indices) : 0;

    /* Validate query-side arrays against expected dimensions.
     * q_ri_vec must have at least FCE_SEM_DIM floats, and q_weights
     * must be at least as long as q_indices (one weight per token).
     * Mismatched (indices without weights) is invalid and would crash.
     * also reject the inverse case (weights without
     * indices). The C scorer receives q_tfidf_len = 0 in that case and
     * silently ignores the weights, which is a logic bug: the caller
     * likely passed weights by mistake and expected them to contribute. */
    /* explicit NULL-return check on query arrays,
     * mirroring the corpus-side guard at :1041. Get*ArrayElements can return
     * NULL without a pending exception on some JVMs. Without this, a genuine
     * JVM OOM on query arrays would be silently masked as "no query signal". */
    if ((j_q_weights && !q_weights) || (j_q_indices && !q_indices) || (j_q_ri_vec && !q_ri_vec)) goto flat_cleanup_query;
    /* j_q_ri_vec is required — the flat
     * scorer uses q_ri_vec to compute RI cosine and will dereference it.
     * Throw IllegalArgumentException if the caller passes a non-NULL
     * j_q_ri_vec that failed to pin (OOM), or if the argument is missing. */
    if (!j_q_ri_vec) {
        if (cls_illegal_arg) {
            (*env)->ThrowNew(env, cls_illegal_arg,
                             "q_ri_vec is required for nSimpleRankFlat");
        }
        goto flat_cleanup_query;
    }
    if (j_q_indices && !j_q_weights) {
        if (cls_illegal_arg) {
            (*env)->ThrowNew(env, cls_illegal_arg,
                             "q_tfidf_indices provided without q_tfidf_weights");
        }
        goto flat_cleanup_query;
    }
    if (j_q_weights && !j_q_indices) {
        if (cls_illegal_arg) {
            (*env)->ThrowNew(env, cls_illegal_arg,
                             "q_tfidf_weights provided without q_tfidf_indices");
        }
        goto flat_cleanup_query;
    }
    int q_ri_len = j_q_ri_vec ? (*env)->GetArrayLength(env, j_q_ri_vec) : 0;
    int q_w_len = j_q_weights ? (*env)->GetArrayLength(env, j_q_weights) : 0;
    /* J-2: the check `q_w_len < q_len` is the strict
     * direction (too-short weights). A *longer* `q_weights` is harmless:
     * the C scorer uses `q_tfidf_len` (built from q_indices) to bound the
     * reads on `q_tfidf_weights`, not the array length. The query-side
     * loop at fce_sparse_tfidf_cosine reads `c->tfidf_weights[i]` only
     * for `i < c->tfidf_len`, and tfidf_len is constructed from
     * `tfidf_lens[f]` for the corpus side. So a longer q_weights array
     * does not cause out-of-bounds reads and is silently tolerated. */
    if ((j_q_ri_vec && q_ri_len < FCE_SEM_DIM) ||
        (j_q_indices && j_q_weights && q_w_len < q_len)) {
        if (cls_illegal_arg) {
            char q_size_msg[96];
            snprintf(q_size_msg, sizeof(q_size_msg),
                     "query array size mismatch: q_ri_vec < %d or q_weights < q_indices", FCE_SEM_DIM);
            (*env)->ThrowNew(env, cls_illegal_arg, q_size_msg);
        }
        goto flat_cleanup_query;
    }

    /* flat scorer does not use file paths — skip extraction entirely. */

    /* Score all pairs in C */
    results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) {
        throw_oom(env, "nSimpleRankFlat: results allocation failed");
        goto flat_cleanup_query;
    }
    uint32_t count = 0;
    fce_sem_simple_rank_flat(
        all_weights, all_indices, tfidf_lens, all_ri_vecs,
        NULL, corpusSize, maxTokens,
        q_indices, q_weights, q_len, q_ri_vec,
        (uint32_t)topK, results, &count);

    /* Build Java result array using cached class/method IDs */
    jresults = (*env)->NewObjectArray(env, count, cls_search_result, NULL);
    if ((*env)->ExceptionCheck(env)) {
        jresults = NULL;
        goto flat_cleanup_results;
    }
    for (uint32_t i = 0; i < count; i++) {
        jobject jres = (*env)->NewObject(env, cls_search_result, ctor_search_result,
                                         results[i].index, results[i].score);
        /* NewObject can throw OOM and return NULL.
         * Continuing JNI calls with a pending exception is UB per the spec.
         * delete partially-built jresults before
         * jumping to cleanup. */
        if (!jres || (*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, jresults);
            jresults = NULL;
            goto flat_cleanup_results;
        }
        (*env)->SetObjectArrayElement(env, jresults, i, jres);
        (*env)->DeleteLocalRef(env, jres);
        /* SetObjectArrayElement can throw; bail before any further JNI call. */
        if ((*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, jresults);
            jresults = NULL;
            goto flat_cleanup_results;
        }
    }

flat_cleanup_results:
    free(results);

flat_cleanup_query:
    if (q_weights) (*env)->ReleaseFloatArrayElements(env, j_q_weights, (jfloat *)q_weights, JNI_ABORT);
    if (q_indices) (*env)->ReleaseIntArrayElements(env, j_q_indices, (jint *)q_indices, JNI_ABORT);
    if (q_ri_vec) (*env)->ReleaseFloatArrayElements(env, j_q_ri_vec, (jfloat *)q_ri_vec, JNI_ABORT);

    if (all_ri_vecs) (*env)->ReleaseFloatArrayElements(env, j_all_ri_vecs, (float *)all_ri_vecs, JNI_ABORT);
    if (tfidf_lens) (*env)->ReleaseIntArrayElements(env, j_tfidf_lens, (jint *)tfidf_lens, JNI_ABORT);
    if (all_indices) (*env)->ReleaseIntArrayElements(env, j_all_indices, (jint *)all_indices, JNI_ABORT);
    if (all_weights) (*env)->ReleaseFloatArrayElements(env, j_all_weights, (jfloat *)all_weights, JNI_ABORT);

    return jresults;
}

/* ── Corpus search (high-level query API) ──────────────────────── */

/* Helper: build Java SearchResult[] from C results + docPaths. */
static jobjectArray build_search_results(JNIEnv *env, fce_sem_ranked_t *results,
                                         uint32_t count) {
    jobjectArray jresults = (*env)->NewObjectArray(env, count, cls_search_result, NULL);
    if ((*env)->ExceptionCheck(env)) return NULL;
    for (uint32_t i = 0; i < count; i++) {
        jobject jres = (*env)->NewObject(env, cls_search_result, ctor_search_result,
                                         results[i].index, results[i].score);
        /* NewObject can throw OOM and return NULL.
         * Continuing JNI calls with a pending exception is UB per the spec.
         * delete the partially-built jresults before
         * returning NULL — consistent with the file's DeleteLocalRef-on-every-path
         * style. The JVM would reclaim it on return anyway, but this makes the
         * contract explicit and is required under PushLocalFrame. */
        if (!jres || (*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, jresults);
            return NULL;
        }
        (*env)->SetObjectArrayElement(env, jresults, i, jres);
        (*env)->DeleteLocalRef(env, jres);
        /* SetObjectArrayElement can throw; bail before any further JNI call. */
        if ((*env)->ExceptionCheck(env)) {
            (*env)->DeleteLocalRef(env, jresults);
            return NULL;
        }
    }
    return jresults;
}

JNIEXPORT jobjectArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchQuery(
    JNIEnv *env, jclass cls, jlong handle,
    jstring jquery, jint topK) {
    (void)cls;
    /* use acquire_handle / release_handle.
     * The refcount protects the corpus during the parallel search
     * even if close() is called concurrently from another thread. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    if (topK <= 0) {
        release_handle(handle);
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* clamp topK to corpus size.
     * early-return on empty corpus — avoids malloc
     * amplification when doc_count == 0 and topK is large. */
    int doc_count = fce_sem_corpus_doc_count(corp);
    if (doc_count <= 0) {
        release_handle(handle);
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    if (topK > doc_count) topK = doc_count;
    /* null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM on HotSpot. Throw NPE before reaching the JNI call.
     * return NULL (not empty array) for
     * consistency with other methods. */
    if (!jquery) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "query is null");
        release_handle(handle);
        return NULL;
    }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query || (*env)->ExceptionCheck(env)) {
        if (query) (*env)->ReleaseStringUTFChars(env, jquery, query);
        release_handle(handle);
        return NULL;
    }

    fce_sem_ranked_t *results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) {
        (*env)->ReleaseStringUTFChars(env, jquery, query);
        release_handle(handle);
        return NULL;
    }
    uint32_t count = 0;
    fce_sem_search_query(corp, query, topK, results, &count, NULL);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    jobjectArray jresults = build_search_results(env, results, count);
    free(results);
    release_handle(handle);
    return jresults;
}

JNIEXPORT jobjectArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchQueryTfidf(
    JNIEnv *env, jclass cls, jlong handle,
    jstring jquery, jint topK) {
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    if (topK <= 0) {
        release_handle(handle);
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* clamp topK to corpus size.
     * early-return on empty corpus — avoids malloc
     * amplification when doc_count == 0 and topK is large. */
    int doc_count = fce_sem_corpus_doc_count(corp);
    if (doc_count <= 0) {
        release_handle(handle);
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    if (topK > doc_count) topK = doc_count;
    /* null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM on HotSpot. Throw NPE before reaching the JNI call.
     * return NULL (not empty array) for
     * consistency with other methods. */
    if (!jquery) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "query is null");
        release_handle(handle);
        return NULL;
    }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query || (*env)->ExceptionCheck(env)) {
        if (query) (*env)->ReleaseStringUTFChars(env, jquery, query);
        release_handle(handle);
        return NULL;
    }

    fce_sem_ranked_t *results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) {
        (*env)->ReleaseStringUTFChars(env, jquery, query);
        release_handle(handle);
        return NULL;
    }
    uint32_t count = 0;
    fce_sem_search_query_tfidf(corp, query, topK, results, &count, NULL);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    jobjectArray jresults = build_search_results(env, results, count);
    free(results);
    release_handle(handle);
    return jresults;
}

JNIEXPORT jobjectArray JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchQueryBruteforce(
    JNIEnv *env, jclass cls, jlong handle,
    jstring jquery, jint topK) {
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    if (topK <= 0) {
        release_handle(handle);
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* clamp topK to corpus size.
     * early-return on empty corpus — avoids malloc
     * amplification when doc_count == 0 and topK is large. */
    int doc_count = fce_sem_corpus_doc_count(corp);
    if (doc_count <= 0) {
        release_handle(handle);
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    if (topK > doc_count) topK = doc_count;
    /* null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM on HotSpot. Throw NPE before reaching the JNI call.
     * return NULL (not empty array) for
     * consistency with other methods. */
    if (!jquery) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "query is null");
        release_handle(handle);
        return NULL;
    }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query || (*env)->ExceptionCheck(env)) {
        if (query) (*env)->ReleaseStringUTFChars(env, jquery, query);
        release_handle(handle);
        return NULL;
    }

    fce_sem_ranked_t *results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) {
        (*env)->ReleaseStringUTFChars(env, jquery, query);
        release_handle(handle);
        return NULL;
    }
    uint32_t count = 0;
    fce_sem_search_query_bruteforce(corp, query, topK, results, &count);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    jobjectArray jresults = build_search_results(env, results, count);
    free(results);
    release_handle(handle);
    return jresults;
}

JNIEXPORT jint JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchCandidateCount(
    JNIEnv *env, jclass cls, jlong handle, jstring jquery) {
    (void)cls;
    /* use acquire_handle / release_handle. */
    fce_sem_corpus_t *corp = acquire_handle(handle);
    if (!corp) return 0;
    /* null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM on HotSpot. Throw NPE before reaching the JNI call. */
    if (!jquery) {
        if (cls_npe) (*env)->ThrowNew(env, cls_npe, "query is null");
        release_handle(handle);
        return 0;
    }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query || (*env)->ExceptionCheck(env)) {
        if (query) (*env)->ReleaseStringUTFChars(env, jquery, query);
        release_handle(handle);
        return 0;
    }
    int count = fce_sem_search_candidate_count(corp, query);
    (*env)->ReleaseStringUTFChars(env, jquery, query);
    release_handle(handle);
    return count;
}

/* ── Memory measurement (getrusage) ────────────────────────────── */

#include <sys/resource.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#if defined(__linux__)
#include <stdio.h>
#endif

JNIEXPORT jlong JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_getPeakRssBytes(
    JNIEnv *env, jclass cls) {
    (void)env;
    (void)cls;
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return (jlong)-1;
#if defined(__APPLE__)
    return (jlong)ru.ru_maxrss;
#elif defined(__linux__)
    return (jlong)ru.ru_maxrss * 1024;
#else
    /* ru_maxrss semantics vary across Unixes (FreeBSD=KiB,
     * some=pages). Return -1 for unknown platforms to avoid silently wrong
     * telemetry. */
    return (jlong)-1;
#endif
}

JNIEXPORT jlong JNICALL Java_io_github_nilsonsfj_fastcodeembed_FastCodeEmbed_getCurrentRssBytes(
    JNIEnv *env, jclass cls) {
    (void)env;
    (void)cls;
#if defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                 (task_info_t)&info, &count);
    if (kr == KERN_SUCCESS) return (jlong)info.resident_size;
    return -1;
#elif defined(__linux__)
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long rss_kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmRSS: %ld kB", &rss_kb) == 1) break;
    }
    fclose(f);
    return rss_kb > 0 ? (jlong)rss_kb * 1024 : -1;
#else
    return -1;
#endif
}
