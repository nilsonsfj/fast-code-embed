/*
 * fast_code_embed_jni.c — JNI bridge for fast-code-embed.
 *
 * Covers: corpus lifecycle (including batch), simple scoring, ranking, search.
 * FuncDescriptor fields are marshalled into temporary fce_sem_func_t on the C side.
 *
 * Fixes applied:
 * - JNI_OnLoad caches jclass and jfieldID once (fixes #4)
 * - marshal_func uses cached IDs (no GetFieldID per call)
 * - Batch loops cache jstring refs, reuse for release (fixes #2)
 * - DeleteLocalRef after every intermediate ref (fixes #1)
 * - ExceptionCheck after all JNI calls that can throw (fixes #3)
 * - Early return on count==0 (fixes #25)
 * - marshal_func releases pinned arrays on error (fixes J3)
 * - All malloc/calloc checked for NULL, goto cleanup on failure (fixes J2)
 *
 * JNI exception-throwing pattern (review 0002 §8.4):
 * - For NullPointerException, use the cached `cls_npe` global ref
 *   (initialised in JNI_OnLoad via FindClass("java/lang/NullPointerException")).
 * - For IllegalArgumentException, use the cached `cls_illegal_arg` global ref.
 * - For any other exception class, FindClass + ThrowNew is acceptable
 *   (call sites are rare; the cost is hidden in the error path).
 * - All JNI calls that can throw are followed by an ExceptionCheck
 *   before continuing. The macro CHECK_EXCEPTION_RETURN(env, retval)
 *   is provided for short-circuit return on pending exception.
 */
#include <jni.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "semantic/semantic.h"

/* ── Cached IDs (initialized in JNI_OnLoad) ────────────────────── */

static jclass    cls_func;
static jfieldID  fid_file_path;
static jfieldID  fid_tfidf_indices;
static jfieldID  fid_tfidf_weights;
static jfieldID  fid_ri_vec;

static jclass    cls_search_result;
static jmethodID ctor_search_result;

static jclass    cls_illegal_arg;
static jclass    cls_npe;

/* ── Helpers ────────────────────────────────────────────────────── */

#define CHECK_EXCEPTION_RETURN(env, retval) do { \
    if ((*env)->ExceptionCheck(env)) { return retval; } \
} while (0)

/* Build a fce_sem_func_t from a Java FuncDescriptor object.
 *
 * Ownership contract:
 *   - Caller must free(*path_out) when done.
 *   - If tfidf_indices/tfidf_weights are pinned (non-NULL in the returned struct),
 *     caller must call unmarshal_func(env, &f, jindices, jweights, path) to release
 *     them back to the JVM. Forgetting to call unmarshal_func leaks pinned JNI arrays.
 *   - ri_vec is copied into the struct (not pinned), so no release needed for it.
 *   - *jindices_out / *jweights_out are set to the JNI array refs so the caller can
 *     pass them to unmarshal_func without re-fetching from the JVM.
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
        /* Review 0001 §1.1: GetStringUTFChars can return NULL on OOM without
         * necessarily raising a pending exception (some JVMs return NULL and
         * set the pending OOM only on the *next* JNI call). strdup(NULL) is
         * UB, almost certainly a crash. Check for both: */
        if (!p || (*env)->ExceptionCheck(env)) {
            if (p) (*env)->ReleaseStringUTFChars(env, jpath, p);
            (*env)->DeleteLocalRef(env, jpath);
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            return (fce_sem_func_t){0};
        }
        /* C2: check strdup return — on OOM,
         * *path_out is NULL, which causes downstream scoring to silently
         * degrade (no proximity boost). Release jpath, delete jpath ref,
         * and return zeroed func so caller sees the failure. */
        *path_out = strdup(p);
        (*env)->ReleaseStringUTFChars(env, jpath, p);
        if (!*path_out) {
            (*env)->DeleteLocalRef(env, jpath);
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            return (fce_sem_func_t){0};
        }
        f.file_path = *path_out;
    } else {
        *path_out = NULL;
    }
    (*env)->DeleteLocalRef(env, jpath);

    /* tfidfIndices */
    jintArray jindices = (jintArray)(*env)->GetObjectField(env, obj, fid_tfidf_indices);
    /* tfidfWeights */
    jfloatArray jweights = (jfloatArray)(*env)->GetObjectField(env, obj, fid_tfidf_weights);

    *jindices_out = jindices;
    *jweights_out = jweights;

    if (jindices && jweights) {
        jint ilen = (*env)->GetArrayLength(env, jindices);
        jint wlen = (*env)->GetArrayLength(env, jweights);
        f.tfidf_indices = (int *)(*env)->GetIntArrayElements(env, jindices, NULL);
        /* J1 (review 0003 §2.1): check for NULL return and pending exception
         * from GetIntArrayElements *before* issuing GetFloatArrayElements.
         * Calling further JNI functions with a pending exception is undefined
         * per the JNI spec. */
        if (!f.tfidf_indices || (*env)->ExceptionCheck(env)) {
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (jindices) (*env)->DeleteLocalRef(env, jindices);
            if (jweights) (*env)->DeleteLocalRef(env, jweights);
            free(*path_out); *path_out = NULL;
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
            free(*path_out); *path_out = NULL;
            if (jindices_out) *jindices_out = NULL;
            if (jweights_out) *jweights_out = NULL;
            return (fce_sem_func_t){0};
        }
        /* J-1 (review 0005 §J-1): tfidf_indices MUST be sorted ascending —
         * fce_sparse_tfidf_cosine uses a two-pointer merge that produces
         * wrong scores on unsorted input.  The struct-based API is
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
                free(*path_out); *path_out = NULL;
                if (jindices_out) *jindices_out = NULL;
                if (jweights_out) *jweights_out = NULL;
                return (fce_sem_func_t){0};
            }
        }
        f.tfidf_len = ilen < wlen ? ilen : wlen;
    }

    /* riVec */
    jfloatArray jri = (jfloatArray)(*env)->GetObjectField(env, obj, fid_ri_vec);
    if (jri) {
        jfloat *elems = (*env)->GetFloatArrayElements(env, jri, NULL);
        /* C3 (review 0002-0002 §2.3): check both NULL return AND pending
         * exception. The JNI spec allows GetFloatArrayElements to return
         * NULL without a pending exception on some JVMs. Mirror the
         * filePath guard (line 92). */
        if (!elems || (*env)->ExceptionCheck(env)) {
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (f.tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
            /* B3 (review 0010 §B3): delete jindices/jweights local refs for
             * symmetry with the success path (unmarshal_func expects them or
             * NULL). Without this, they leak until the native frame pops. */
            if (jindices) (*env)->DeleteLocalRef(env, jindices);
            if (jweights) (*env)->DeleteLocalRef(env, jweights);
            (*env)->DeleteLocalRef(env, jri);
            free(*path_out); *path_out = NULL;
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
            (*env)->ThrowNew(env, iae, "riVec length must be >= 768");
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (f.tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
            if (jindices) (*env)->DeleteLocalRef(env, jindices);
            if (jweights) (*env)->DeleteLocalRef(env, jweights);
            free(*path_out); *path_out = NULL;
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
 * Receives the JNI array refs directly from marshal_func — no re-fetch. */
static void unmarshal_func(JNIEnv *env, jobject obj, fce_sem_func_t *f,
                            jintArray jindices, jfloatArray jweights, char *path) {
    (void)obj;
    if (jindices && f->tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f->tfidf_indices, JNI_ABORT);
    if (jweights && f->tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f->tfidf_weights, JNI_ABORT);

    (*env)->DeleteLocalRef(env, jindices);
    (*env)->DeleteLocalRef(env, jweights);

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
    jclass local_func = (*env)->FindClass(env, "com/github/nilsonsfj/fastcodeembed/FuncDescriptor");
    if (!local_func) goto fail;
    cls_func = (*env)->NewGlobalRef(env, local_func);
    (*env)->DeleteLocalRef(env, local_func);

    fid_file_path       = (*env)->GetFieldID(env, cls_func, "filePath", "Ljava/lang/String;");
    fid_tfidf_indices   = (*env)->GetFieldID(env, cls_func, "tfidfIndices", "[I");
    fid_tfidf_weights   = (*env)->GetFieldID(env, cls_func, "tfidfWeights", "[F");
    fid_ri_vec          = (*env)->GetFieldID(env, cls_func, "riVec", "[F");
    if (!fid_file_path || !fid_tfidf_indices || !fid_tfidf_weights || !fid_ri_vec) {
        goto fail;
    }

    /* Cache SearchResult class + constructor */
    jclass local_sr = (*env)->FindClass(env, "com/github/nilsonsfj/fastcodeembed/SearchResult");
    if (!local_sr) goto fail;
    cls_search_result = (*env)->NewGlobalRef(env, local_sr);
    (*env)->DeleteLocalRef(env, local_sr);

    ctor_search_result = (*env)->GetMethodID(env, cls_search_result, "<init>", "(IF)V");
    if (!ctor_search_result) goto fail;

    /* Cache IllegalArgumentException class (M-5: avoid FindClass in error path).
     * Review 0007 §2.5: hard-fail if core exception classes can't load — using
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

    return JNI_VERSION_1_6;

fail:
    /* L3 (review 0007 §L3): clean up any global refs already created before
     * aborting library load. Symmetric with JNI_OnUnload. */
    if (cls_func) { (*env)->DeleteGlobalRef(env, cls_func); cls_func = NULL; }
    if (cls_search_result) { (*env)->DeleteGlobalRef(env, cls_search_result); cls_search_result = NULL; }
    if (cls_illegal_arg) { (*env)->DeleteGlobalRef(env, cls_illegal_arg); cls_illegal_arg = NULL; }
    if (cls_npe) { (*env)->DeleteGlobalRef(env, cls_npe); cls_npe = NULL; }
    return JNI_ERR;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    (void)vm; (void)reserved;
    /* C4 (review 0002-0002 §2.4): Do NOT call fce_sem_shutdown() here.
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
    }
    cls_func = NULL;
    cls_search_result = NULL;
    cls_illegal_arg = NULL;
    cls_npe = NULL;
}

/* ── Corpus JNI ─────────────────────────────────────────────────── */

JNIEXPORT jlong JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nCreateCorpus(
    JNIEnv *env, jclass cls) {
    (void)cls;
    fce_sem_corpus_t *c = fce_sem_corpus_new();
    /* Review 0001 §2.6: fce_sem_corpus_new returns NULL on OOM. The old
     * scheme returned 0 (= NULL handle) for both "valid empty corpus" and
     * "OOM", which made Corpus treat OOM as a successful create-then-close
     * and surface a confusing "Corpus is closed" IllegalStateException on
     * the first method call. The Java side now reserves 0 for "closed" and
     * uses -1 (any sentinel non-zero, non-positive) for "OOM". */
    if (!c) {
        return (jlong)-1;
    }
    return (jlong)(uintptr_t)c;
}

JNIEXPORT void JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nFreeCorpus(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;
    /* C1: guard against invalid (stale/corrupt)
     * handle — matches every other JNI function's guard pattern. A stale
     * handle from a previously-freed Corpus would otherwise be cast to a
     * pointer and passed to free(), causing double-free / heap corruption. */
    if (handle <= 0) return;
    fce_sem_corpus_free((fce_sem_corpus_t *)(uintptr_t)handle);
}

JNIEXPORT void JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddDoc(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jtokens) {
    (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return;
    /* H1 (review 0007 §H1): null-guard the array argument. GetArrayLength on
     * NULL is UB per JNI spec and SIGSEGV the JVM on HotSpot. */
    if (!jtokens) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "tokens is null"); return; }
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    int count = (*env)->GetArrayLength(env, jtokens);
    /* L4 (review 0005 §L4): malloc(0) is implementation-defined and may return
     * NULL, which the OOM guard below would treat as an allocation failure.
     * Return early on empty input — matching nAddDocsBatch (:329). */
    if (count == 0) return;
    const char **tokens = (const char **)malloc(sizeof(char *) * count);
    jstring *refs = (jstring *)malloc(sizeof(jstring) * count);
    if (!tokens || !refs) { free(tokens); free(refs); return; }
    int pinned = 0;
    for (int i = 0; i < count; i++) {
        jstring jtok = (jstring)(*env)->GetObjectArrayElement(env, jtokens, i);
        if ((*env)->ExceptionCheck(env)) {
            /* C9: GetObjectArrayElement may throw
             * (e.g., ArrayIndexOutOfBoundsException) while jtok holds a valid
             * local ref. Release it before jumping to cleanup to prevent
             * leaking a JNI local reference on each exception. */
            if (jtok) (*env)->DeleteLocalRef(env, jtok);
            goto adddoc_cleanup;
        }
        if (!jtok) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null token in array"); goto adddoc_cleanup; }
        tokens[i] = (*env)->GetStringUTFChars(env, jtok, NULL);
        if (!tokens[i]) {
            (*env)->DeleteLocalRef(env, jtok);
            goto adddoc_cleanup;
        }
        refs[i] = jtok;
        pinned = i + 1;
    }
    fce_sem_corpus_add_doc(corp, tokens, count);
adddoc_cleanup:
    for (int i = 0; i < pinned; i++) {
        (*env)->ReleaseStringUTFChars(env, refs[i], tokens[i]);
        (*env)->DeleteLocalRef(env, refs[i]);
    }
    free(refs);
    free(tokens);
}

JNIEXPORT void JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddDocsBatch(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jdocs, jint maxTokensPerDoc) {
    (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return;
    /* H1 (review 0007 §H1): null-guard the array argument. */
    if (!jdocs) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "docs is null"); return; }
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    int docCount = (*env)->GetArrayLength(env, jdocs);
    if (docCount == 0 || maxTokensPerDoc <= 0) return;

    /* H1 (review 0001-0001 §1): clamp maxTokensPerDoc to FCE_SEM_MAX_TOKENS
     * (512). Without this, a single doc with >512 tokens causes the C batch
     * function to reject the entire batch silently. Matching the single-doc
     * path (addDoc) which truncates per-doc, not per-batch. */
    if (maxTokensPerDoc > FCE_SEM_MAX_TOKENS) maxTokensPerDoc = FCE_SEM_MAX_TOKENS;

    /* Build the flat token array: all_tokens[doc * maxTokensPerDoc + tok].
     * Cache jstring refs so release pass doesn't re-fetch.
     * Clamp per-doc token count to maxTokensPerDoc to prevent heap overflow. */
    size_t flat = (size_t)docCount * (size_t)maxTokensPerDoc;
    char **all_tokens = (char **)malloc(sizeof(char *) * flat);
    jstring *all_refs = (jstring *)calloc(flat, sizeof(jstring));
    int *token_counts = (int *)malloc(sizeof(int) * docCount);
    if (!all_tokens || !all_refs || !token_counts) {
        free(all_tokens); free(all_refs); free(token_counts);
        return;
    }
    memset(all_tokens, 0, sizeof(char *) * flat);

    for (int d = 0; d < docCount; d++) {
        jobjectArray jdoc = (jobjectArray)(*env)->GetObjectArrayElement(env, jdocs, d);
        if ((*env)->ExceptionCheck(env)) {
            if (jdoc) (*env)->DeleteLocalRef(env, jdoc);
            goto addbatch_cleanup;
        }
        /* L2 (review 0007 §L2): null-guard the inner sub-array before GetArrayLength.
         * A null element inside jdocs yields jdoc == NULL with no pending exception,
         * then GetArrayLength(env, NULL) is UB. Currently unreachable because
         * Corpus.addDocsBatch iterates in Java first, but defends against future
         * callers of the package-private path. */
        if (!jdoc) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null document array"); goto addbatch_cleanup; }
        int len = (*env)->GetArrayLength(env, jdoc);
        if (len > maxTokensPerDoc) len = maxTokensPerDoc;
        token_counts[d] = len;
        for (int t = 0; t < len; t++) {
            jstring jtok = (jstring)(*env)->GetObjectArrayElement(env, jdoc, t);
            if ((*env)->ExceptionCheck(env)) {
                (*env)->DeleteLocalRef(env, jdoc);
                goto addbatch_cleanup;
            }
            if (!jtok) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null token in doc array"); (*env)->DeleteLocalRef(env, jdoc); goto addbatch_cleanup; }
            const char *tok = (*env)->GetStringUTFChars(env, jtok, NULL);
            if (!tok) {
                (*env)->DeleteLocalRef(env, jtok);
                (*env)->DeleteLocalRef(env, jdoc);
                goto addbatch_cleanup;
            }
            all_tokens[(size_t)d * maxTokensPerDoc + t] = (char *)tok;
            all_refs[(size_t)d * maxTokensPerDoc + t] = jtok;
        }
        (*env)->DeleteLocalRef(env, jdoc);
    }

    fce_sem_corpus_add_docs_batch(corp, all_tokens, token_counts, docCount, maxTokensPerDoc);

addbatch_cleanup:
    /* Release JNI strings using cached refs. Iterate the full flat array so
     * partially-pinned docs (where the inner loop bailed mid-doc) are also
     * cleaned up. all_refs was calloc'd, so unprocessed entries are NULL. */
    for (size_t idx = 0; idx < flat; idx++) {
        if (all_refs[idx]) {
            (*env)->ReleaseStringUTFChars(env, all_refs[idx], all_tokens[idx]);
            (*env)->DeleteLocalRef(env, all_refs[idx]);
            all_refs[idx] = NULL;
        }
    }

    free(all_refs);
    free(all_tokens);
    free(token_counts);
}

JNIEXPORT jboolean JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nFinalizeCorpus(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return JNI_FALSE;
    int rc = fce_sem_corpus_finalize((fce_sem_corpus_t *)(uintptr_t)handle);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddDocsTokenized(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jnames) {
    (void)cls;
    /* L3 (review 0006 §L3): Partial-state semantics — on mid-batch failure
     * (null element, OOM), earlier batches that were already committed via
     * fce_sem_corpus_add_docs_tokenized remain in the corpus. The caller
     * sees an exception and must decide whether to continue, roll back, or
     * discard the corpus. This is a deliberate trade-off: atomic rollback
     * across potentially millions of tokens would require a full corpus
     * snapshot and copy, which is not justified for a batch-add API. */
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return;
    /* H1 (review 0007 §H1): null-guard the array argument. */
    if (!jnames) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "names is null"); return; }
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    int total = (*env)->GetArrayLength(env, jnames);
    if (total == 0) return;

    /* C5: reduce BATCH from 10000 to 1000 to cut
     * per-call heap allocation from ~160 KB to ~16 KB. This bounds allocation
     * pressure on constrained environments (Android JNI threads with small
     * heaps) while still providing good batching throughput. */
    const int BATCH = 1000;
    const char **names = (const char **)malloc(sizeof(char *) * BATCH);
    jstring *refs = (jstring *)malloc(sizeof(jstring) * BATCH);
    if (!names || !refs) { free(names); free(refs); return; }

    for (int start = 0; start < total; start += BATCH) {
        int end = start + BATCH;
        if (end > total) end = total;
        int batch = end - start;

        /* Extract batch of strings + cache refs */
        int pinned = 0;
        for (int i = 0; i < batch; i++) {
            jstring jname = (jstring)(*env)->GetObjectArrayElement(env, jnames, start + i);
            if ((*env)->ExceptionCheck(env)) goto addtok_cleanup;
            if (!jname) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null token in array"); goto addtok_cleanup; }
            names[i] = (*env)->GetStringUTFChars(env, jname, NULL);
            if (!names[i]) {
                (*env)->DeleteLocalRef(env, jname);
                goto addtok_cleanup;
            }
            refs[i] = jname;
            pinned = i + 1;
        }

        /* Tokenize + add to corpus */
        fce_sem_corpus_add_docs_tokenized(corp, names, batch);

addtok_cleanup:
        /* Release strings using cached refs */
        for (int i = 0; i < pinned; i++) {
            (*env)->ReleaseStringUTFChars(env, refs[i], names[i]);
            (*env)->DeleteLocalRef(env, refs[i]);
        }
        /* H1 (review 0006 §H1): if an exception is pending (NPE from
         * null element, OOM from GetStringUTFChars, or AIOOBE from
         * GetObjectArrayElement), the outer loop must not re-enter
         * JNI — calling GetObjectArrayElement / GetStringUTFChars with
         * a pending exception is UB per JNI spec §"Exceptions" and
         * crashes the JVM on HotSpot. */
        if ((*env)->ExceptionCheck(env)) break;
    }
    free(refs);
    free(names);
}

JNIEXPORT jint JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddFiles(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jpaths, jint chunkSize,
    jintArray jFileDocCounts, jint maxTokensPerChunk) {
    (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return 0;
    /* C1: null jpaths → GetArrayLength is UB and
     * SIGSEGVs the JVM on HotSpot. Throw NPE before reaching the JNI call. */
    if (!jpaths) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "paths is null"); return -1; }
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    int count = (*env)->GetArrayLength(env, jpaths);
    if (count == 0 || chunkSize <= 0) return 0;

    /* Extract all path strings in one pass, cache refs for cleanup. */
    const char **paths = (const char **)malloc(sizeof(char *) * count);
    jstring *refs = (jstring *)malloc(sizeof(jstring) * count);
    if (!paths || !refs) { free(paths); free(refs); return -1; }
    /* H2: initialize result before any goto can bypass the assignment. */
    int result = -1;
    int pinned = 0;
    for (int i = 0; i < count; i++) {
        jstring jpath = (jstring)(*env)->GetObjectArrayElement(env, jpaths, i);
        if ((*env)->ExceptionCheck(env)) {
            if (jpath) (*env)->DeleteLocalRef(env, jpath);
            goto addfiles_cleanup;
        }
        if (!jpath) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null path in array"); goto addfiles_cleanup; }
        paths[i] = (*env)->GetStringUTFChars(env, jpath, NULL);
        if (!paths[i]) {
            /* OOM: a pending OutOfMemoryError is now set. Bail before any
             * further JNI calls (which would be UB) or C function calls
             * (which would be invoked with a pending exception). */
            (*env)->DeleteLocalRef(env, jpath);
            goto addfiles_cleanup;
        }
        refs[i] = jpath;
        pinned = i + 1;
    }

    /* Per-file doc counts output buffer. */
    int *file_doc_counts = NULL;
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
        (*env)->ReleaseStringUTFChars(env, refs[i], paths[i]);
        (*env)->DeleteLocalRef(env, refs[i]);
    }
    free(refs);
    free(paths);
    return result;
}

JNIEXPORT jfloat JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetIdf(
    JNIEnv *env, jclass cls, jlong handle, jstring jtoken) {
    (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return 0.0f;
    /* H1 (review 0005 §H1): null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call. */
    if (!jtoken) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "token is null"); return 0.0f; }
    const char *tok = (*env)->GetStringUTFChars(env, jtoken, NULL);
    if (!tok) {
        /* OOM: OutOfMemoryError is pending. Calling the C function with a
         * pending exception is undefined JNI behavior — return early. */
        return 0.0f;
    }
    float idf = fce_sem_corpus_idf((fce_sem_corpus_t *)(uintptr_t)handle, tok);
    (*env)->ReleaseStringUTFChars(env, jtoken, tok);
    return idf;
}

JNIEXPORT jfloatArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetRiVec(
    JNIEnv *env, jclass cls, jlong handle, jstring jtoken) {
    (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return NULL;
    /* H1 (review 0005 §H1): null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call. */
    if (!jtoken) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "token is null"); return NULL; }
    const char *tok = (*env)->GetStringUTFChars(env, jtoken, NULL);
    if (!tok) {
        /* OOM: OutOfMemoryError is pending. Bail before calling C. */
        return NULL;
    }
    /* Review 0001 §2.5: fce_sem_corpus_ri_vec returns a pointer to a
     * _Thread_local fce_sem_vec_t (semantic.c tl_dequant). The data is
     * immediately copied out via SetFloatArrayRegion below, so the TLS
     * pointer's lifetime does not matter once the Java array is filled
     * — the JVM owns its own storage. No need to copy into a fresh
     * allocation. The function is safe across JNI calls because the
     * _Thread_local scratch is per-Java-thread (and the C side, being
     * a JNI_OnLoad-attached dylib, is always called from a Java
     * thread that has a stable _Thread_local key). */
    const fce_sem_vec_t *vec = fce_sem_corpus_ri_vec((fce_sem_corpus_t *)(uintptr_t)handle, tok);
    (*env)->ReleaseStringUTFChars(env, jtoken, tok);
    if (!vec) return NULL;
    jfloatArray result = (*env)->NewFloatArray(env, FCE_SEM_DIM);
    CHECK_EXCEPTION_RETURN(env, NULL);
    (*env)->SetFloatArrayRegion(env, result, 0, FCE_SEM_DIM, vec->v);
    return result;
}

JNIEXPORT jint JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetDocCount(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return 0;
    return fce_sem_corpus_doc_count((fce_sem_corpus_t *)(uintptr_t)handle);
}

JNIEXPORT jint JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetTokenCount(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return 0;
    return fce_sem_corpus_token_count((fce_sem_corpus_t *)(uintptr_t)handle);
}

/* ── Static Nomic Search JNI ────────────────────────────────────── */

JNIEXPORT void JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_init(
    JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    fce_sem_ensure_ready();
}

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_tokenize(
    JNIEnv *env, jclass cls, jstring jname) {
    (void)cls;
    /* H1 (review 0005 §H1): null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call. */
    if (!jname) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "name is null"); return NULL; }
    const char *name = (*env)->GetStringUTFChars(env, jname, NULL);
    if (!name) {
        /* OOM: pending OutOfMemoryError. */
        return NULL;
    }

    char *tokens[FCE_SEM_MAX_TOKENS];
    /* Zero the array so the cleanup path can safely free every slot up to count,
     * even if the count is less than FCE_SEM_MAX_TOKENS. */
    memset(tokens, 0, sizeof(tokens));
    int count = fce_sem_tokenize(name, tokens, FCE_SEM_MAX_TOKENS);

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
    /* I1 (review 0007 §I1): safety of this cleanup loop depends on two
     * invariants working together:
     *   1. tokens[] is zeroed up front (memset at :699), so every slot
     *      holds NULL unless fce_sem_tokenize wrote to it.
     *   2. The success loop (:707-713) sets tokens[i] = NULL after free,
     *      so already-consumed slots are harmless to free() again.
     * A future edit that removes either the up-front memset or the
     * per-iteration NULLing will introduce a double-free here. */
    for (int i = 0; i < count; i++) {
        free(tokens[i]);
    }
    (*env)->ReleaseStringUTFChars(env, jname, name);
    return NULL;
}

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nTokenizeBatch(
    JNIEnv *env, jclass cls, jobjectArray jnames) {
    (void)cls;
    /* H1 (review 0007 §H1): null-guard the array argument. */
    if (!jnames) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "names is null"); return NULL; }
    int count = (*env)->GetArrayLength(env, jnames);
    if (count == 0) {
        jclass strArrCls = (*env)->FindClass(env, "[Ljava/lang/String;");
        CHECK_EXCEPTION_RETURN(env, NULL);
        jobjectArray empty = (*env)->NewObjectArray(env, 0, strArrCls, NULL);
        (*env)->DeleteLocalRef(env, strArrCls);
        return empty;
    }

    /* Extract input strings */
    const char **names = (const char **)malloc(sizeof(char *) * count);
    jstring *refs = (jstring *)malloc(sizeof(jstring) * count);
    if (!names || !refs) { free(names); free(refs); return NULL; }
    int pinned = 0;
    for (int i = 0; i < count; i++) {
        jstring jname = (jstring)(*env)->GetObjectArrayElement(env, jnames, i);
        if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup_input;
        if (!jname) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "null token in array"); goto tokenize_cleanup_input; }
        names[i] = (*env)->GetStringUTFChars(env, jname, NULL);
        /* Review 0001 §1.1: OOM check on GetStringUTFChars (a pending
         * OutOfMemoryError is set on NULL return; the next JNI call
         * would crash). */
        if (!names[i]) {
            (*env)->DeleteLocalRef(env, jname);
            goto tokenize_cleanup_input;
        }
        refs[i] = jname;
        pinned = i + 1;
    }

    /* Tokenize all in C */
    int max_out = FCE_SEM_MAX_TOKENS;
    /* C18: defense-in-depth overflow check.
     * On 64-bit (required by _Static_assert in semantic.h), count * max_out
     * fits in size_t, but this explicit check prevents issues if the
     * requirement is ever relaxed. */
    size_t flat_sz = (size_t)count * (size_t)max_out;
    if (flat_sz > (SIZE_MAX / sizeof(char *))) return NULL;
    char **all_tokens = (char **)calloc(flat_sz, sizeof(char *));
    int *token_counts = (int *)malloc(count * sizeof(int));
    if (!all_tokens || !token_counts) {
        free(all_tokens); free(token_counts);
        goto tokenize_cleanup_input;
    }
    fce_sem_tokenize_batch(names, count, all_tokens, token_counts, max_out);

    /* Release input strings using cached refs */
    for (int i = 0; i < count; i++) {
        (*env)->ReleaseStringUTFChars(env, refs[i], names[i]);
        (*env)->DeleteLocalRef(env, refs[i]);
    }
    free(refs);
    free(names);

    /* Build Java String[][] result */
    {
    jclass strCls = (*env)->FindClass(env, "java/lang/String");
    if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup_tokens;
    jclass strArrCls = (*env)->FindClass(env, "[Ljava/lang/String;");
    if ((*env)->ExceptionCheck(env)) { (*env)->DeleteLocalRef(env, strCls); goto tokenize_cleanup_tokens; }
    jobjectArray result = (*env)->NewObjectArray(env, count, strArrCls, NULL);
    (*env)->DeleteLocalRef(env, strArrCls);
    if ((*env)->ExceptionCheck(env)) { (*env)->DeleteLocalRef(env, strCls); goto tokenize_cleanup_tokens; }
    for (int i = 0; i < count; i++) {
        jobjectArray docTokens = (*env)->NewObjectArray(env, token_counts[i], strCls, NULL);
        if ((*env)->ExceptionCheck(env)) goto tokenize_cleanup_result;
        for (int t = 0; t < token_counts[i]; t++) {
            jstring jtok = (*env)->NewStringUTF(env, all_tokens[(size_t)i * max_out + t]);
            /* M1 (review 0006 §M1): NewStringUTF can return NULL and raise
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
        (*env)->DeleteLocalRef(env, docTokens);
    }
    (*env)->DeleteLocalRef(env, strCls);
    free(all_tokens);
    free(token_counts);
    return result;

tokenize_cleanup_result:
    (*env)->DeleteLocalRef(env, strCls);
    }

tokenize_cleanup_tokens:
    for (int i = 0; i < count; i++) {
        for (int t = 0; t < token_counts[i]; t++) {
            char *tok = all_tokens[(size_t)i * max_out + t];
            if (tok) { free(tok); all_tokens[(size_t)i * max_out + t] = NULL; }
        }
    }
    free(all_tokens);
    free(token_counts);
    return NULL;

tokenize_cleanup_input:
    for (int i = 0; i < pinned; i++) {
        (*env)->ReleaseStringUTFChars(env, refs[i], names[i]);
        (*env)->DeleteLocalRef(env, refs[i]);
    }
    free(refs);
    free(names);
    return NULL;
}

JNIEXPORT jfloat JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_proximity(
    JNIEnv *env, jclass cls, jstring jpathA, jstring jpathB) {
    (void)cls;
    /* H1 (review 0005 §H1): null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call. */
    if (!jpathA) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "pathA is null"); return 0.0f; }
    if (!jpathB) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "pathB is null"); return 0.0f; }
    const char *a = (*env)->GetStringUTFChars(env, jpathA, NULL);
    /* Review 0001 §1.1: NULL check on GetStringUTFChars (pending OOM). */
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
 * Review 0007 §2.3. this path is @Deprecated from Java;
 * consider gating behind a size assertion or removing once all callers
 * have migrated to the flat API. */

JNIEXPORT jfloat JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_simpleScore(
    JNIEnv *env, jclass cls, jobject ja, jobject jb) {
    (void)cls;
    char *pathA = NULL, *pathB = NULL;
    jintArray jindA = NULL, jindB = NULL;
    jfloatArray jwgtA = NULL, jwgtB = NULL;
    fce_sem_func_t a = marshal_func(env, ja, &pathA, &jindA, &jwgtA);
    /* Review 0007 §2.2: marshal_func can throw (NullPointerException,
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

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_simpleRank(
    JNIEnv *env, jclass cls, jobject jquery, jobjectArray jcorpus, jint topK) {
    (void)cls;

    /* H1 (review 0007 §H1): null-guard the array argument. */
    if (!jcorpus) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "corpus is null"); return NULL; }
    int corpusSize = (*env)->GetArrayLength(env, jcorpus);
    if (corpusSize <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* L4 (review 0006): this deprecated path holds 2N JNI local refs alive
     * until cleanup, which can exhaust the local-ref table on large corpora.
     * Reject corpora larger than 4096 — callers should use nSimpleRankFlat. */
    if (corpusSize > 4096) {
        if (cls_illegal_arg) {
            (*env)->ThrowNew(env, cls_illegal_arg,
                "corpus too large for simpleRank (deprecated); use nSimpleRankFlat for corpora > 4096");
        }
        return NULL;
    }
    /* C5 (review 0002-0002 §2.5): clamp topK to corpus size before allocating
     * result buffers to prevent allocation-amplification from hostile callers. */
    if (topK > corpusSize) topK = corpusSize;
    fce_sem_func_t *corpus = (fce_sem_func_t *)calloc(corpusSize, sizeof(fce_sem_func_t));
    char **paths = (char **)calloc(corpusSize, sizeof(char *));
    jobject *corp_refs = (jobject *)calloc(corpusSize, sizeof(jobject));
    jintArray *jindices_arr = (jintArray *)calloc(corpusSize, sizeof(jintArray));
    jfloatArray *jweights_arr = (jfloatArray *)calloc(corpusSize, sizeof(jfloatArray));
    fce_sem_ranked_t *results = NULL;
    jobjectArray jresults = NULL;

    if (!corpus || !paths || !corp_refs || !jindices_arr || !jweights_arr) {
        free(corpus); free(paths); free(corp_refs); free(jindices_arr); free(jweights_arr);
        return NULL;
    }

    /* Marshal query */
    char *qpath = NULL;
    jintArray qjindices = NULL;
    jfloatArray qjweights = NULL;
    fce_sem_func_t query = marshal_func(env, jquery, &qpath, &qjindices, &qjweights);
    if ((*env)->ExceptionCheck(env)) goto simpleRank_cleanup;

    /* Marshal corpus — cache jobject refs for cleanup.
     * corp_refs[i] is not used by unmarshal_func (only jindices/jweights are),
     * so delete it immediately to bound local ref pressure.
     * J-2 (review 0002 §2.2): the invariant is "corp_refs[i] is non-NULL
     * iff the local ref has NOT been DeleteLocalRef'd". We set it to NULL
     * on the success path, and the cleanup loop walks the array looking
     * for non-NULL slots. If marshal_func throws partway through (e.g.,
     * OOM in GetIntArrayElements), the loop bails out at the
     * ExceptionCheck. Slots [0, i) are NULL (already cleared); slot i is
     * the obj reference still pinned (NOT DeleteLocalRef'd yet) — the
     * cleanup at simpleRank_cleanup will release it. Slots [i+1, N) are
     * untouched and the cleanup loop skips them (NULL). The current
     * code is correct; this comment makes the invariant explicit. */
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
    if (!results) goto simpleRank_cleanup;
    uint32_t count = 0;
    fce_sem_simple_rank(&query, corpus, corpusSize, topK, results, &count);

    /* Build Java array using cached class/method IDs */
    jresults = (*env)->NewObjectArray(env, count, cls_search_result, NULL);
    if ((*env)->ExceptionCheck(env)) { jresults = NULL; goto simpleRank_cleanup; }
    for (uint32_t i = 0; i < count; i++) {
        jobject jres = (*env)->NewObject(env, cls_search_result, ctor_search_result,
            results[i].index, results[i].score);
        (*env)->SetObjectArrayElement(env, jresults, i, jres);
        (*env)->DeleteLocalRef(env, jres);
    }

simpleRank_cleanup:
    unmarshal_func(env, jquery, &query, qjindices, qjweights, qpath);
    for (int i = 0; i < corpusSize; i++) {
        if (jindices_arr[i] || jweights_arr[i]) {
            unmarshal_func(env, corp_refs[i], &corpus[i], jindices_arr[i], jweights_arr[i], paths[i]);
            paths[i] = NULL; /* path ownership transferred to unmarshal_func */
        }
        /* M1 (review 0001-0001 §3): free the path unconditionally. The path is
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

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_simpleSearch(
    JNIEnv *env, jclass cls, jobject jquery, jobjectArray jcorpus,
    jint topK, jfloat minScore) {
    (void)cls;

    /* H1 (review 0007 §H1): null-guard the array argument. */
    if (!jcorpus) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "corpus is null"); return NULL; }
    int corpusSize = (*env)->GetArrayLength(env, jcorpus);
    if (corpusSize <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* L4 (review 0006): this deprecated path holds 2N JNI local refs alive
     * until cleanup, which can exhaust the local-ref table on large corpora.
     * Reject corpora larger than 4096 — callers should use nSimpleRankFlat. */
    if (corpusSize > 4096) {
        if (cls_illegal_arg) {
            (*env)->ThrowNew(env, cls_illegal_arg,
                "corpus too large for simpleSearch (deprecated); use nSimpleRankFlat for corpora > 4096");
        }
        return NULL;
    }
    /* M2 (review 0004 §M2): clamp topK to corpus size, mirroring simpleRank C5. */
    if (topK > corpusSize) topK = corpusSize;
    fce_sem_func_t *corpus = (fce_sem_func_t *)calloc(corpusSize, sizeof(fce_sem_func_t));
    char **paths = (char **)calloc(corpusSize, sizeof(char *));
    jobject *corp_refs = (jobject *)calloc(corpusSize, sizeof(jobject));
    jintArray *jindices_arr = (jintArray *)calloc(corpusSize, sizeof(jintArray));
    jfloatArray *jweights_arr = (jfloatArray *)calloc(corpusSize, sizeof(jfloatArray));
    fce_sem_ranked_t *results = NULL;
    jobjectArray jresults = NULL;

    if (!corpus || !paths || !corp_refs || !jindices_arr || !jweights_arr) {
        free(corpus); free(paths); free(corp_refs); free(jindices_arr); free(jweights_arr);
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
    if (!results) goto simpleSearch_cleanup;
    uint32_t count = 0;
    fce_sem_simple_search(&query, corpus, corpusSize, topK, minScore, results, &count);

    jresults = (*env)->NewObjectArray(env, count, cls_search_result, NULL);
    if ((*env)->ExceptionCheck(env)) { jresults = NULL; goto simpleSearch_cleanup; }
    for (uint32_t i = 0; i < count; i++) {
        jobject jres = (*env)->NewObject(env, cls_search_result, ctor_search_result,
            results[i].index, results[i].score);
        (*env)->SetObjectArrayElement(env, jresults, i, jres);
        (*env)->DeleteLocalRef(env, jres);
    }

simpleSearch_cleanup:
    unmarshal_func(env, jquery, &query, qjindices, qjweights, qpath);
    for (int i = 0; i < corpusSize; i++) {
        if (jindices_arr[i] || jweights_arr[i]) {
            unmarshal_func(env, corp_refs[i], &corpus[i], jindices_arr[i], jweights_arr[i], paths[i]);
            paths[i] = NULL; /* path ownership transferred to unmarshal_func */
        }
        /* M1 (review 0001-0001 §3): free the path unconditionally — same as simpleRank_cleanup. */
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

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSimpleRankFlat(
    JNIEnv *env, jclass cls,
    /* corpus flat arrays */
    jfloatArray j_all_weights,
    jintArray   j_all_indices,
    jintArray   j_tfidf_lens,
    jfloatArray j_all_ri_vecs,
    jobjectArray j_file_paths,
    jint maxTokens,
    /* query */
    jintArray   j_q_indices,
    jfloatArray j_q_weights,
    jfloatArray j_q_ri_vec,
    /* output */
    jint topK) {
    (void)cls;

    jobjectArray jresults = NULL;

    /* H1 (review 0003 §H1): declare all pin pointers upfront, NULL-initialized,
     * so that goto flat_cleanup_query before any pin assignment does not read
     * indeterminate stack values. Each pointer is assigned below at its pin site. */
    const float *all_weights = NULL, *all_ri_vecs = NULL;
    const int   *all_indices = NULL, *tfidf_lens  = NULL;
    const float *q_weights   = NULL, *q_ri_vec    = NULL;
    const int   *q_indices   = NULL;

    /* Null-guard the required array arguments BEFORE any GetArrayLength calls.
     * JNI spec says GetArrayLength(env, NULL) is undefined and crashes on HotSpot.
     * L2 (review 0007 §L2): j_all_weights, j_all_indices, and j_tfidf_lens are
     * optional — the flat scorer uses RI only and callers building RI-only corpora
     * should not be forced to allocate TF-IDF arrays that are then ignored. */
    if (!j_file_paths || !j_all_ri_vecs) {
        if ((*env)->ExceptionCheck(env)) return NULL;
        (*env)->ThrowNew(env, cls_npe, "null array argument to nSimpleRankFlat");
        return NULL;
    }

    int corpusSize = (*env)->GetArrayLength(env, j_file_paths);
    if (corpusSize <= 0 || topK <= 0 || maxTokens <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* C5 (review 0002-0002 §2.5): clamp topK to corpus size. */
    if (topK > corpusSize) topK = corpusSize;

    /* Get primitive array pointers. check ExceptionCheck
     * and NULL between each pin — Get*ArrayElements may throw OOM if it
     * needs to copy the array, and continuing JNI calls with a pending
     * exception is undefined behavior per the JNI spec.
     * L2 (review 0007 §L2): TF-IDF corpus arrays are optional (flat scorer
     * uses RI only). Pin them only when provided. */
    if (j_all_weights) {
        all_weights = (*env)->GetFloatArrayElements(env, j_all_weights, NULL);
        if (!all_weights || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    }
    if (j_all_indices) {
        all_indices = (*env)->GetIntArrayElements (env, j_all_indices, NULL);
        if (!all_indices || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    }
    if (j_tfidf_lens) {
        tfidf_lens  = (*env)->GetIntArrayElements (env, j_tfidf_lens, NULL);
        if (!tfidf_lens || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    }
    all_ri_vecs = (*env)->GetFloatArrayElements(env, j_all_ri_vecs, NULL);
    if (!all_ri_vecs || (*env)->ExceptionCheck(env)) goto flat_cleanup_query;

    /* Validate array sizes against caller-supplied dimensions.
     * Compute products in size_t to avoid 32-bit overflow on large corpora.
     * L2 (review 0007 §L2): TF-IDF arrays are optional. When provided, validate
     * their sizes; when absent, skip TF-IDF validation. */
    int w_len  = j_all_weights ? (*env)->GetArrayLength(env, j_all_weights) : 0;
    int i_len  = j_all_indices ? (*env)->GetArrayLength(env, j_all_indices) : 0;
    int tl_len = j_tfidf_lens  ? (*env)->GetArrayLength(env, j_tfidf_lens) : 0;
    int rv_len = (*env)->GetArrayLength(env, j_all_ri_vecs);
    size_t needed_ri    = (size_t)corpusSize * (size_t)FCE_SEM_DIM;
    if ((size_t)rv_len < needed_ri) {
        if (cls_illegal_arg) {
            (*env)->ThrowNew(env, cls_illegal_arg,
                "array size mismatch: all_ri_vecs too small for corpusSize * 768");
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
     * Review 0001 §2.3: also reject the inverse case (weights without
     * indices). The C scorer receives q_tfidf_len = 0 in that case and
     * silently ignores the weights, which is a logic bug: the caller
     * likely passed weights by mistake and expected them to contribute. */
    /* L1 (review 0007 §L1): explicit NULL-return check on query arrays,
     * mirroring the corpus-side guard at :1041. Get*ArrayElements can return
     * NULL without a pending exception on some JVMs. Without this, a genuine
     * JVM OOM on query arrays would be silently masked as "no query signal". */
    if ((j_q_weights && !q_weights) || (j_q_indices && !q_indices) || (j_q_ri_vec && !q_ri_vec)) goto flat_cleanup_query;
    /* C15: j_q_ri_vec is required — the flat
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
    int q_w_len  = j_q_weights ? (*env)->GetArrayLength(env, j_q_weights) : 0;
    /* J-2 (review 0002 §2.1): the check `q_w_len < q_len` is the strict
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
            (*env)->ThrowNew(env, cls_illegal_arg,
                "query array size mismatch: q_ri_vec < 768 or q_weights < q_indices");
        }
        goto flat_cleanup_query;
    }

    /* flat scorer does not use file paths — skip extraction entirely. */
    fce_sem_ranked_t *results = NULL;

    /* Score all pairs in C */
    results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) goto flat_cleanup_query;
    uint32_t count = 0;
    fce_sem_simple_rank_flat(
        all_weights, all_indices, tfidf_lens, all_ri_vecs,
        NULL, corpusSize, maxTokens,
        q_indices, q_weights, q_len, q_ri_vec,
        (uint32_t)topK, results, &count);

    /* Build Java result array using cached class/method IDs */
    jresults = (*env)->NewObjectArray(env, count, cls_search_result, NULL);
    if ((*env)->ExceptionCheck(env)) { jresults = NULL; goto flat_cleanup_results; }
    for (uint32_t i = 0; i < count; i++) {
        jobject jres = (*env)->NewObject(env, cls_search_result, ctor_search_result,
            results[i].index, results[i].score);
        (*env)->SetObjectArrayElement(env, jresults, i, jres);
        (*env)->DeleteLocalRef(env, jres);
    }

flat_cleanup_results:
    free(results);

flat_cleanup_query:
    if (q_weights) (*env)->ReleaseFloatArrayElements(env, j_q_weights, (jfloat *)q_weights, JNI_ABORT);
    if (q_indices) (*env)->ReleaseIntArrayElements(env, j_q_indices, (jint *)q_indices, JNI_ABORT);
    if (q_ri_vec)  (*env)->ReleaseFloatArrayElements(env, j_q_ri_vec, (jfloat *)q_ri_vec, JNI_ABORT);

    if (all_ri_vecs) (*env)->ReleaseFloatArrayElements(env, j_all_ri_vecs, (float *)all_ri_vecs, JNI_ABORT);
    if (tfidf_lens)  (*env)->ReleaseIntArrayElements(env, j_tfidf_lens, (jint *)tfidf_lens, JNI_ABORT);
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
        (*env)->SetObjectArrayElement(env, jresults, i, jres);
        (*env)->DeleteLocalRef(env, jres);
    }
    return jresults;
}

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchQuery(
    JNIEnv *env, jclass cls, jlong handle,
    jstring jquery, jint topK) {
    (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* C5 (review 0002-0002 §2.5): clamp topK to corpus size.
     * M1 (review 0005 §1.2): early-return on empty corpus — avoids malloc
     * amplification when doc_count == 0 and topK is large. */
    int doc_count = fce_sem_corpus_doc_count(corp);
    if (doc_count <= 0) return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    if (topK > doc_count) topK = doc_count;
    /* H1 (review 0005 §H1): null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call.
     * C7: return NULL (not empty array) for
     * consistency with other methods. */
    if (!jquery) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "query is null"); return NULL; }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query || (*env)->ExceptionCheck(env)) {
        if (query) (*env)->ReleaseStringUTFChars(env, jquery, query);
        return NULL;
    }

    fce_sem_ranked_t *results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) { (*env)->ReleaseStringUTFChars(env, jquery, query); return NULL; }
    uint32_t count = 0;
    fce_sem_search_query(corp, query, topK, results, &count, NULL);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    jobjectArray jresults = build_search_results(env, results, count);
    free(results);
    return jresults;
}

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchQueryTfidf(
    JNIEnv *env, jclass cls, jlong handle,
    jstring jquery, jint topK) {
    (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* C5 (review 0002-0002 §2.5): clamp topK to corpus size.
     * M1 (review 0005 §1.2): early-return on empty corpus — avoids malloc
     * amplification when doc_count == 0 and topK is large. */
    int doc_count = fce_sem_corpus_doc_count(corp);
    if (doc_count <= 0) return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    if (topK > doc_count) topK = doc_count;
    /* H1 (review 0005 §H1): null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call.
     * C7: return NULL (not empty array) for
     * consistency with other methods. */
    if (!jquery) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "query is null"); return NULL; }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query || (*env)->ExceptionCheck(env)) {
        if (query) (*env)->ReleaseStringUTFChars(env, jquery, query);
        return NULL;
    }

    fce_sem_ranked_t *results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) { (*env)->ReleaseStringUTFChars(env, jquery, query); return NULL; }
    uint32_t count = 0;
    fce_sem_search_query_tfidf(corp, query, topK, results, &count, NULL);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    jobjectArray jresults = build_search_results(env, results, count);
    free(results);
    return jresults;
}

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchQueryBruteforce(
    JNIEnv *env, jclass cls, jlong handle,
    jstring jquery, jint topK) {
    (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    /* C5 (review 0002-0002 §2.5): clamp topK to corpus size.
     * M1 (review 0005 §1.2): early-return on empty corpus — avoids malloc
     * amplification when doc_count == 0 and topK is large. */
    int doc_count = fce_sem_corpus_doc_count(corp);
    if (doc_count <= 0) return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    if (topK > doc_count) topK = doc_count;
    /* H1 (review 0005 §H1): null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call.
     * C7: return NULL (not empty array) for
     * consistency with other methods. */
    if (!jquery) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "query is null"); return NULL; }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query || (*env)->ExceptionCheck(env)) {
        if (query) (*env)->ReleaseStringUTFChars(env, jquery, query);
        return NULL;
    }

    fce_sem_ranked_t *results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) { (*env)->ReleaseStringUTFChars(env, jquery, query); return NULL; }
    uint32_t count = 0;
    fce_sem_search_query_bruteforce(corp, query, topK, results, &count);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    jobjectArray jresults = build_search_results(env, results, count);
    free(results);
    return jresults;
}

JNIEXPORT jint JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchCandidateCount(
    JNIEnv *env, jclass cls, jlong handle, jstring jquery) {
    (void)cls;
    /* M3 (review 0004 §M3): guard against invalid (OOM sentinel / stale) handle. */
    if (handle <= 0) return 0;
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    /* H1 (review 0005 §H1): null jstring → GetStringUTFChars(env, NULL, …) is UB
     * and crashes the JVM. Throw NPE before reaching the JNI call. */
    if (!jquery) { if (cls_npe) (*env)->ThrowNew(env, cls_npe, "query is null"); return 0; }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    if (!query || (*env)->ExceptionCheck(env)) {
        if (query) (*env)->ReleaseStringUTFChars(env, jquery, query);
        return 0;
    }
    int count = fce_sem_search_candidate_count(corp, query);
    (*env)->ReleaseStringUTFChars(env, jquery, query);
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

JNIEXPORT jlong JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_getPeakRssBytes(
    JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
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

JNIEXPORT jlong JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_getCurrentRssBytes(
    JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
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
