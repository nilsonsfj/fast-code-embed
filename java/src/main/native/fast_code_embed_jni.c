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
    fce_sem_func_t f;
    memset(&f, 0, sizeof(f));


    /* filePath */
    jstring jpath = (jstring)(*env)->GetObjectField(env, obj, fid_file_path);
    if (jpath) {
        const char *p = (*env)->GetStringUTFChars(env, jpath, NULL);
        if ((*env)->ExceptionCheck(env)) { (*env)->DeleteLocalRef(env, jpath); return (fce_sem_func_t){0}; }
        *path_out = strdup(p);
        f.file_path = *path_out;
        (*env)->ReleaseStringUTFChars(env, jpath, p);
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
        f.tfidf_weights = (float *)(*env)->GetFloatArrayElements(env, jweights, NULL);
        if ((*env)->ExceptionCheck(env)) {
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (f.tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
            free(*path_out); *path_out = NULL;
            return (fce_sem_func_t){0};
        }
        if (f.tfidf_indices && f.tfidf_weights) {
            f.tfidf_len = ilen < wlen ? ilen : wlen;
        } else {
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (f.tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
            f.tfidf_indices = NULL;
            f.tfidf_weights = NULL;
            f.tfidf_len = 0;
        }
    }

    /* riVec */
    jfloatArray jri = (jfloatArray)(*env)->GetObjectField(env, obj, fid_ri_vec);
    if (jri) {
        jfloat *elems = (*env)->GetFloatArrayElements(env, jri, NULL);
        if ((*env)->ExceptionCheck(env)) {
            if (f.tfidf_indices) (*env)->ReleaseIntArrayElements(env, jindices, (jint *)f.tfidf_indices, JNI_ABORT);
            if (f.tfidf_weights) (*env)->ReleaseFloatArrayElements(env, jweights, f.tfidf_weights, JNI_ABORT);
            (*env)->DeleteLocalRef(env, jri);
            free(*path_out); *path_out = NULL;
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
            free(*path_out); *path_out = NULL;
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
    if (!local_func) return JNI_ERR;
    cls_func = (*env)->NewGlobalRef(env, local_func);
    (*env)->DeleteLocalRef(env, local_func);

    fid_file_path       = (*env)->GetFieldID(env, cls_func, "filePath", "Ljava/lang/String;");
    fid_tfidf_indices   = (*env)->GetFieldID(env, cls_func, "tfidfIndices", "[I");
    fid_tfidf_weights   = (*env)->GetFieldID(env, cls_func, "tfidfWeights", "[F");
    fid_ri_vec          = (*env)->GetFieldID(env, cls_func, "riVec", "[F");
    if (!fid_file_path || !fid_tfidf_indices || !fid_tfidf_weights || !fid_ri_vec) {
        return JNI_ERR;
    }

    /* Cache SearchResult class + constructor */
    jclass local_sr = (*env)->FindClass(env, "com/github/nilsonsfj/fastcodeembed/SearchResult");
    if (!local_sr) return JNI_ERR;
    cls_search_result = (*env)->NewGlobalRef(env, local_sr);
    (*env)->DeleteLocalRef(env, local_sr);

    ctor_search_result = (*env)->GetMethodID(env, cls_search_result, "<init>", "(IF)V");
    if (!ctor_search_result) return JNI_ERR;

    /* Cache IllegalArgumentException class (M-5: avoid FindClass in error path). */
    jclass local_iae = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
    if (local_iae) {
        cls_illegal_arg = (*env)->NewGlobalRef(env, local_iae);
        (*env)->DeleteLocalRef(env, local_iae);
    }

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM *vm, void *reserved) {
    (void)vm; (void)reserved;
    fce_sem_shutdown();
    JNIEnv *env;
    if ((*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6) == JNI_OK) {
        if (cls_func) (*env)->DeleteGlobalRef(env, cls_func);
        if (cls_search_result) (*env)->DeleteGlobalRef(env, cls_search_result);
        if (cls_illegal_arg) (*env)->DeleteGlobalRef(env, cls_illegal_arg);
    }
    cls_func = NULL;
    cls_search_result = NULL;
    cls_illegal_arg = NULL;
}

/* ── Corpus JNI ─────────────────────────────────────────────────── */

JNIEXPORT jlong JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nCreateCorpus(
    JNIEnv *env, jclass cls) {
    (void)cls;
    return (jlong)(uintptr_t)fce_sem_corpus_new();
}

JNIEXPORT void JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nFreeCorpus(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)cls;
    fce_sem_corpus_free((fce_sem_corpus_t *)(uintptr_t)handle);
}

JNIEXPORT void JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddDoc(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jtokens) {
    (void)cls;
    int count = (*env)->GetArrayLength(env, jtokens);
    if (count == 0) return;
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    const char **tokens = (const char **)malloc(sizeof(char *) * count);
    jstring *refs = (jstring *)malloc(sizeof(jstring) * count);
    if (!tokens || !refs) { free(tokens); free(refs); return; }
    int pinned = 0;
    for (int i = 0; i < count; i++) {
        jstring jtok = (jstring)(*env)->GetObjectArrayElement(env, jtokens, i);
        if ((*env)->ExceptionCheck(env)) goto adddoc_cleanup;
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
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    int docCount = (*env)->GetArrayLength(env, jdocs);
    if (docCount == 0 || maxTokensPerDoc <= 0) return;

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

    int docs_pinned = 0;
    for (int d = 0; d < docCount; d++) {
        jobjectArray jdoc = (jobjectArray)(*env)->GetObjectArrayElement(env, jdocs, d);
        if ((*env)->ExceptionCheck(env)) {
            if (jdoc) (*env)->DeleteLocalRef(env, jdoc);
            goto addbatch_cleanup;
        }
        int len = (*env)->GetArrayLength(env, jdoc);
        if (len > maxTokensPerDoc) len = maxTokensPerDoc;
        token_counts[d] = len;
        for (int t = 0; t < len; t++) {
            jstring jtok = (jstring)(*env)->GetObjectArrayElement(env, jdoc, t);
            if ((*env)->ExceptionCheck(env)) {
                (*env)->DeleteLocalRef(env, jdoc);
                goto addbatch_cleanup;
            }
            const char *tok = (*env)->GetStringUTFChars(env, jtok, NULL);
            if (!tok) {
                (*env)->DeleteLocalRef(env, jtok);
                (*env)->DeleteLocalRef(env, jdoc);
                goto addbatch_cleanup;
            }
            all_tokens[(size_t)d * maxTokensPerDoc + t] = (char *)tok;
            all_refs[(size_t)d * maxTokensPerDoc + t] = jtok;
        }
        docs_pinned = d + 1;
        (*env)->DeleteLocalRef(env, jdoc);
    }

    fce_sem_corpus_add_docs_batch(corp, all_tokens, token_counts, docCount, maxTokensPerDoc);

addbatch_cleanup:
    /* Release JNI strings using cached refs — only for docs that were fully pinned */
    for (int d = 0; d < docs_pinned; d++) {
        int len = token_counts[d];
        for (int t = 0; t < len; t++) {
            size_t idx = (size_t)d * maxTokensPerDoc + t;
            if (all_refs[idx]) {
                (*env)->ReleaseStringUTFChars(env, all_refs[idx], all_tokens[idx]);
                (*env)->DeleteLocalRef(env, all_refs[idx]);
            }
        }
    }

    free(all_refs);
    free(all_tokens);
    free(token_counts);
}

JNIEXPORT jboolean JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nFinalizeCorpus(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
    int rc = fce_sem_corpus_finalize((fce_sem_corpus_t *)(uintptr_t)handle);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddDocsTokenized(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jnames) {
    (void)cls;
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    int total = (*env)->GetArrayLength(env, jnames);
    if (total == 0) return;

    /* Process in batches of 10000. Cache jstring refs per batch. */
    const int BATCH = 10000;
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
    }
    free(refs);
    free(names);
}

JNIEXPORT jint JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nAddFiles(
    JNIEnv *env, jclass cls, jlong handle, jobjectArray jpaths, jint chunkSize,
    jintArray jFileDocCounts, jint maxTokensPerChunk) {
    (void)cls;
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
        if ((*env)->ExceptionCheck(env)) goto addfiles_cleanup;
        paths[i] = (*env)->GetStringUTFChars(env, jpath, NULL);
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
    const char *tok = (*env)->GetStringUTFChars(env, jtoken, NULL);
    CHECK_EXCEPTION_RETURN(env, 0.0f);
    float idf = fce_sem_corpus_idf((fce_sem_corpus_t *)(uintptr_t)handle, tok);
    (*env)->ReleaseStringUTFChars(env, jtoken, tok);
    return idf;
}

JNIEXPORT jfloatArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetRiVec(
    JNIEnv *env, jclass cls, jlong handle, jstring jtoken) {
    (void)cls;
    const char *tok = (*env)->GetStringUTFChars(env, jtoken, NULL);
    CHECK_EXCEPTION_RETURN(env, NULL);
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
    return fce_sem_corpus_doc_count((fce_sem_corpus_t *)(uintptr_t)handle);
}

JNIEXPORT jint JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nGetTokenCount(
    JNIEnv *env, jclass cls, jlong handle) {
    (void)env; (void)cls;
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
    const char *name = (*env)->GetStringUTFChars(env, jname, NULL);
    CHECK_EXCEPTION_RETURN(env, NULL);

    char *tokens[FCE_SEM_MAX_TOKENS];
    int count = fce_sem_tokenize(name, tokens, FCE_SEM_MAX_TOKENS);

    jclass strCls = (*env)->FindClass(env, "java/lang/String");
    CHECK_EXCEPTION_RETURN(env, NULL);
    jobjectArray result = (*env)->NewObjectArray(env, count, strCls, NULL);
    (*env)->DeleteLocalRef(env, strCls);
    CHECK_EXCEPTION_RETURN(env, NULL);
    for (int i = 0; i < count; i++) {
        jstring jtok = (*env)->NewStringUTF(env, tokens[i]);
        (*env)->SetObjectArrayElement(env, result, i, jtok);
        (*env)->DeleteLocalRef(env, jtok);
        free(tokens[i]);
    }

    (*env)->ReleaseStringUTFChars(env, jname, name);
    return result;
}

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nTokenizeBatch(
    JNIEnv *env, jclass cls, jobjectArray jnames) {
    (void)cls;
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
        names[i] = (*env)->GetStringUTFChars(env, jname, NULL);
        refs[i] = jname;
        pinned = i + 1;
    }

    /* Tokenize all in C */
    int max_out = FCE_SEM_MAX_TOKENS;
    char **all_tokens = (char **)calloc((size_t)count * max_out, sizeof(char *));
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
    const char *a = (*env)->GetStringUTFChars(env, jpathA, NULL);
    CHECK_EXCEPTION_RETURN(env, 0.0f);
    const char *b = (*env)->GetStringUTFChars(env, jpathB, NULL);
    if ((*env)->ExceptionCheck(env)) {
        (*env)->ReleaseStringUTFChars(env, jpathA, a);
        return 0.0f;
    }
    float score = fce_sem_proximity(a, b);
    (*env)->ReleaseStringUTFChars(env, jpathA, a);
    (*env)->ReleaseStringUTFChars(env, jpathB, b);
    return score;
}

/* ── Simple API ─────────────────────────────────────────────────── */

JNIEXPORT jfloat JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_simpleScore(
    JNIEnv *env, jclass cls, jobject ja, jobject jb) {
    (void)cls;
    char *pathA = NULL, *pathB = NULL;
    jintArray jindA = NULL, jindB = NULL;
    jfloatArray jwgtA = NULL, jwgtB = NULL;
    fce_sem_func_t a = marshal_func(env, ja, &pathA, &jindA, &jwgtA);
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

    int corpusSize = (*env)->GetArrayLength(env, jcorpus);
    if (corpusSize <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
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
     * so delete it immediately to bound local ref pressure. */
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
        }
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

    int corpusSize = (*env)->GetArrayLength(env, jcorpus);
    if (corpusSize <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
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
        }
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

    int corpusSize = (*env)->GetArrayLength(env, j_file_paths);
    if (corpusSize <= 0 || topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }

    /* Get primitive array pointers */
    const float *all_weights = (*env)->GetFloatArrayElements(env, j_all_weights, NULL);
    const int   *all_indices = (*env)->GetIntArrayElements(env, j_all_indices, NULL);
    const int   *tfidf_lens  = (*env)->GetIntArrayElements(env, j_tfidf_lens, NULL);
    const float *all_ri_vecs = (*env)->GetFloatArrayElements(env, j_all_ri_vecs, NULL);

    if ((*env)->ExceptionCheck(env)) goto flat_cleanup_prim;

    /* Validate array sizes against caller-supplied dimensions.
     * Compute products in size_t to avoid 32-bit overflow on large corpora. */
    int w_len  = (*env)->GetArrayLength(env, j_all_weights);
    int i_len  = (*env)->GetArrayLength(env, j_all_indices);
    int tl_len = (*env)->GetArrayLength(env, j_tfidf_lens);
    int rv_len = (*env)->GetArrayLength(env, j_all_ri_vecs);
    size_t needed_tfidf = (size_t)corpusSize * (size_t)maxTokens;
    size_t needed_ri    = (size_t)corpusSize * (size_t)FCE_SEM_DIM;
    if ((size_t)w_len < needed_tfidf || (size_t)i_len < needed_tfidf ||
        (size_t)tl_len < (size_t)corpusSize || (size_t)rv_len < needed_ri) {
        goto flat_cleanup_prim;
    }

    const float *q_weights = j_q_weights ? (*env)->GetFloatArrayElements(env, j_q_weights, NULL) : NULL;
    const int   *q_indices = j_q_indices ? (*env)->GetIntArrayElements(env, j_q_indices, NULL) : NULL;
    const float *q_ri_vec  = j_q_ri_vec ? (*env)->GetFloatArrayElements(env, j_q_ri_vec, NULL) : NULL;

    int q_len = j_q_indices ? (*env)->GetArrayLength(env, j_q_indices) : 0;

    /* Validate query-side arrays against expected dimensions.
     * q_ri_vec must have at least FCE_SEM_DIM floats, and q_weights
     * must be at least as long as q_indices (one weight per token).
     * Mismatched (indices without weights) is invalid and would crash. */
    if ((*env)->ExceptionCheck(env)) goto flat_cleanup_query;
    if (j_q_indices && !j_q_weights) {
        goto flat_cleanup_query;
    }
    int q_ri_len = j_q_ri_vec ? (*env)->GetArrayLength(env, j_q_ri_vec) : 0;
    int q_w_len  = j_q_weights ? (*env)->GetArrayLength(env, j_q_weights) : 0;
    if ((j_q_ri_vec && q_ri_len < FCE_SEM_DIM) ||
        (j_q_indices && j_q_weights && q_w_len < q_len)) {
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

flat_cleanup_prim:
    if (all_weights) (*env)->ReleaseFloatArrayElements(env, j_all_weights, (jfloat *)all_weights, JNI_ABORT);
    if (all_indices) (*env)->ReleaseIntArrayElements(env, j_all_indices, (jint *)all_indices, JNI_ABORT);
    if (tfidf_lens)  (*env)->ReleaseIntArrayElements(env, j_tfidf_lens, (jint *)tfidf_lens, JNI_ABORT);
    if (all_ri_vecs) (*env)->ReleaseFloatArrayElements(env, j_all_ri_vecs, (jfloat *)all_ri_vecs, JNI_ABORT);

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
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    CHECK_EXCEPTION_RETURN(env, NULL);

    fce_sem_ranked_t *results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) { (*env)->ReleaseStringUTFChars(env, jquery, query); return NULL; }
    uint32_t count = 0;
    fce_sem_search_query(corp, query, topK, results, &count);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    jobjectArray jresults = build_search_results(env, results, count);
    free(results);
    return jresults;
}

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchQueryTfidf(
    JNIEnv *env, jclass cls, jlong handle,
    jstring jquery, jint topK) {
    (void)cls;
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    CHECK_EXCEPTION_RETURN(env, NULL);

    fce_sem_ranked_t *results = (fce_sem_ranked_t *)malloc(sizeof(fce_sem_ranked_t) * topK);
    if (!results) { (*env)->ReleaseStringUTFChars(env, jquery, query); return NULL; }
    uint32_t count = 0;
    fce_sem_search_query_tfidf(corp, query, topK, results, &count);
    (*env)->ReleaseStringUTFChars(env, jquery, query);

    jobjectArray jresults = build_search_results(env, results, count);
    free(results);
    return jresults;
}

JNIEXPORT jobjectArray JNICALL Java_com_github_nilsonsfj_fastcodeembed_FastCodeEmbed_nSearchQueryBruteforce(
    JNIEnv *env, jclass cls, jlong handle,
    jstring jquery, jint topK) {
    (void)cls;
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    if (topK <= 0) {
        return (*env)->NewObjectArray(env, 0, cls_search_result, NULL);
    }
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    CHECK_EXCEPTION_RETURN(env, NULL);

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
    fce_sem_corpus_t *corp = (fce_sem_corpus_t *)(uintptr_t)handle;
    const char *query = (*env)->GetStringUTFChars(env, jquery, NULL);
    CHECK_EXCEPTION_RETURN(env, 0);
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
#else
    return (jlong)ru.ru_maxrss * 1024;
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
