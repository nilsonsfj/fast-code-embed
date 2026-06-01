package com.github.nilsonsfj.fastcodeembed;

/**
 * Java binding for the <a href="https://github.com/nilsonsfj/fast-code-embed">fast-code-embed</a> C library.
 *
 * <p>Provides two APIs:</p>
 * <ul>
 *   <li><b>Full API</b> — requires populating all vector fields manually (API, Type, Decorator, Structural Profile).</li>
 *   <li><b>Simple API</b> — zero-config, uses only TF-IDF + Random Indexing, returns [0.0, 1.0].</li>
 * </ul>
 *
 * <p>All functions are static. Call {@link #init()} once before any scoring.</p>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * FastCodeEmbed.init();
 *
 * Corpus corp = new Corpus();
 * corp.addDocsBatch(new String[][]{{"handle", "request"}, {"validate", "user"}});
 * corp.complete();
 *
 * FuncDescriptor a = corp.buildFunc("src/handler.c", new String[]{"handle", "request"});
 * FuncDescriptor b = corp.buildFunc("src/auth.c", new String[]{"validate", "user"});
 *
 * float score = FastCodeEmbed.simpleScore(a, b);  // [0.0, 1.0]
 * }</pre>
 *
 * @since 0.0.1
 */
public final class FastCodeEmbed {

    /** Library version string, e.g. "0.0.1". */
    public static final String VERSION = "0.0.1";

    static {
        NativeLibrary.load();
    }

    private FastCodeEmbed() {}

    // ── Initialization ────────────────────────────────────────────

    /**
     * Initialize the pretrained token lookup table.
     * Must be called once before any scoring functions.
     * Safe to call multiple times (idempotent).
     */
    public static native void init();

    // ── Tokenization ──────────────────────────────────────────────

    /**
     * Tokenize a name into camelCase / snake_case / dot-separated tokens.
     * Tokens are lowercased. Abbreviations are expanded (e.g. "getJSON" → ["get", "json"]).
     *
     * @param name the identifier to tokenize
     * @return array of lowercased tokens
     */
    public static native String[] tokenize(String name);

    /**
     * Batch-tokenize multiple names in one JNI call.
     * Much faster than calling tokenize() in a loop (avoids per-name JNI overhead).
     *
     * @param names array of identifiers to tokenize
     * @return array of token arrays (same length as input)
     */
    public static String[][] tokenizeBatch(String[] names) {
        return nTokenizeBatch(names);
    }

    // ── Proximity ─────────────────────────────────────────────────

    /**
     * Compute module proximity multiplier based on file paths.
     * Uses continuous scaling: shared path components increase the boost
     * proportionally, up to a maximum of 1.10 for identical paths.
     * Same file → ~1.10, same directory → depends on depth,
     * completely different → 1.0.
     *
     * @param pathA file path of first function
     * @param pathB file path of second function
     * @return proximity multiplier [1.0, 1.10]
     */
    public static native float proximity(String pathA, String pathB);

    // ── Simple API ────────────────────────────────────────────────

    /**
     * Score two functions using TF-IDF + Random Indexing only.
     * Returns [0.0, 1.0] — no user-provided vectors needed.
     *
     * @param a first function descriptor
     * @param b second function descriptor
     * @return similarity score [0.0 = unrelated, 1.0 = identical]
     */
    public static native float simpleScore(FuncDescriptor a, FuncDescriptor b);

    /**
     * Rank corpus against a query using simple scoring.
     * Returns results sorted by score descending.
     *
     * @param query  the query function
     * @param corpus array of function descriptors to rank
     * @param topK   maximum number of results to return
     * @return ranked results (length ≤ topK)
     */
    public static native SearchResult[] simpleRank(FuncDescriptor query, FuncDescriptor[] corpus, int topK);

    /**
     * Search corpus with a minimum score threshold.
     * Returns results sorted by score descending.
     *
     * @param query    the query function
     * @param corpus   array of function descriptors to search
     * @param topK     maximum number of results to return
     * @param minScore minimum score threshold (inclusive)
     * @return ranked results passing the threshold
     */
    public static native SearchResult[] simpleSearch(FuncDescriptor query, FuncDescriptor[] corpus,
                                                     int topK, float minScore);

    /**
     * Rank corpus against a query using pre-extracted flat arrays.
     * All scoring happens in C — no per-item JNI marshaling overhead.
     *
     * <p>Use {@link Corpus#extractFlat(FuncDescriptor[])} to build the flat arrays
     * from an array of FuncDescriptors. The flat arrays can be reused across queries.</p>
     *
     * @param allWeights  flat IDF weights: [func * maxTokens + token]
     * @param allIndices  flat token indices: [func * maxTokens + token]
     * @param tfidfLens   per-function token count: [func]
     * @param allRiVecs   flat RI vectors: [func * 768 + dim]
     * @param filePaths   per-function file paths: [func]
     * @param maxTokens  stride for flat arrays (longest TF-IDF in corpus)
     * @param qIndices   query token indices
     * @param qWeights   query IDF weights
     * @param qRiVec     query RI vector (768 floats)
     * @param topK       maximum results
     * @return ranked results sorted by score descending
     */
    public static SearchResult[] simpleRankBatch(
            float[] allWeights, int[] allIndices, int[] tfidfLens,
            float[] allRiVecs, String[] filePaths, int maxTokens,
            int[] qIndices, float[] qWeights, float[] qRiVec,
            int topK) {
        return nSimpleRankFlat(allWeights, allIndices, tfidfLens,
                allRiVecs, filePaths, maxTokens,
                qIndices, qWeights, qRiVec, topK);
    }

    // ── Corpus search (high-level query API) ─────────────────────

    /**
     * High-level search: tokenize a query string, build query vector from
     * enriched token vectors, return top-k ranked results.
     * Fast path: uses inverted index for candidate retrieval, then reranks
     * with RI cosine similarity.
     *
     * @param corpus finalized corpus to search
     * @param query  natural language query string
     * @param topK   maximum results
     * @return ranked results sorted by score descending
     */
    public static SearchResult[] searchQuery(Corpus corpus,
                                             String query, int topK) {
        return nSearchQuery(corpus.getHandle(), query, topK);
    }

    /**
     * TF-IDF hybrid search: uses TF-IDF sparse cosine for candidate retrieval,
     * then reranks with RI cosine.
     *
     * @param corpus finalized corpus to search
     * @param query  natural language query string
     * @param topK   maximum results
     * @return ranked results sorted by score descending
     */
    public static SearchResult[] searchQueryTfidf(Corpus corpus,
                                                  String query, int topK) {
        return nSearchQueryTfidf(corpus.getHandle(), query, topK);
    }

    /**
     * Brute-force search: scans ALL document vectors with cosine similarity.
     * Slower but guaranteed to find the global top-k.
     *
     * @param corpus finalized corpus to search
     * @param query  natural language query string
     * @param topK   maximum results
     * @return ranked results sorted by score descending
     */
    public static SearchResult[] searchQueryBruteforce(Corpus corpus,
                                                       String query, int topK) {
        return nSearchQueryBruteforce(corpus.getHandle(), query, topK);
    }

    /**
     * Return the number of candidates the inverted index would retrieve
     * for a given query. Useful for understanding search selectivity.
     *
     * @param corpus finalized corpus
     * @param query  query string
     * @return candidate count
     */
    public static int searchCandidateCount(Corpus corpus, String query) {
        return nSearchCandidateCount(corpus.getHandle(), query);
    }

    // ── Memory measurement ────────────────────────────────────────

    /**
     * Get peak RSS (resident set size) in bytes via getrusage().
     * This is the same measurement as the C benchmark uses.
     *
     * @return peak RSS in bytes, or -1 on error
     */
    public static native long getPeakRssBytes();

    /**
     * Get current RSS (resident set size) in bytes.
     * macOS: task_info, Linux: /proc/self/status VmRSS.
     *
     * @return current RSS in bytes, or -1 on error
     */
    public static native long getCurrentRssBytes();

    // ── JNI internals (called from C, not user-facing) ────────────

    /** Review 0001 §2.6: -1 is the OOM sentinel from nCreateCorpus. The
     *  Corpus constructor converts it to an OutOfMemoryError; callers
     *  that reach this method directly see the raw -1. */
    static final long CORPUS_OOM = -1L;

    static long createCorpus() { return nCreateCorpus(); }
    static void freeCorpus(long handle) { nFreeCorpus(handle); }
    static void addDoc(long handle, String[] tokens) { nAddDoc(handle, tokens); }
    static void addDocsBatch(long handle, String[][] docs, int maxTokensPerDoc) {
        nAddDocsBatch(handle, docs, maxTokensPerDoc);
    }
    static boolean finalizeCorpus(long handle) { return nFinalizeCorpus(handle); }
    static void addDocsTokenized(long handle, String[] names) { nAddDocsTokenized(handle, names); }
    static int addFiles(long handle, String[] paths, int chunkSize, int[] fileDocCounts, int maxTokensPerChunk) {
        return nAddFiles(handle, paths, chunkSize, fileDocCounts, maxTokensPerChunk);
    }
    static float getIdf(long handle, String token) { return nGetIdf(handle, token); }
    static float[] getRiVec(long handle, String token) { return nGetRiVec(handle, token); }
    static int getDocCount(long handle) { return nGetDocCount(handle); }
    static int getTokenCount(long handle) { return nGetTokenCount(handle); }

    private static native long nCreateCorpus();
    private static native void nFreeCorpus(long handle);
    private static native void nAddDoc(long handle, String[] tokens);
    private static native void nAddDocsBatch(long handle, String[][] docs, int maxTokensPerDoc);
    private static native boolean nFinalizeCorpus(long handle);
    private static native void nAddDocsTokenized(long handle, String[] names);
    private static native int nAddFiles(long handle, String[] paths, int chunkSize, int[] fileDocCounts, int maxTokensPerChunk);
    private static native float nGetIdf(long handle, String token);
    private static native float[] nGetRiVec(long handle, String token);
    private static native int nGetDocCount(long handle);
    private static native int nGetTokenCount(long handle);
    private static native String[][] nTokenizeBatch(String[] names);
    private static native SearchResult[] nSimpleRankFlat(
            float[] allWeights, int[] allIndices, int[] tfidfLens,
            float[] allRiVecs, String[] filePaths, int maxTokens,
            int[] qIndices, float[] qWeights, float[] qRiVec, int topK);
    private static native SearchResult[] nSearchQuery(long handle,
        String query, int topK);

    private static native SearchResult[] nSearchQueryTfidf(long handle,
        String query, int topK);

    private static native SearchResult[] nSearchQueryBruteforce(long handle,
        String query, int topK);

    private static native int nSearchCandidateCount(long handle, String query);
}
