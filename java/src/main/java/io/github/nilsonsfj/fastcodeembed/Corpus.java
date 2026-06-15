package io.github.nilsonsfj.fastcodeembed;

/**
 * Wraps a native {@code fce_sem_corpus_t} for building IDF weights and
 * enriched Random Indexing vectors from a collection of tokenized documents.
 *
 * <p>Workflow: create → add docs → {@link #complete()} → query.</p>
 *
 * <h2>Example</h2>
 * <pre>{@code
 * try (Corpus corp = new Corpus()) {
 *     corp.addDocsBatch(new String[][]{
 *         {"handle", "request", "parse"},
 *         {"validate", "user", "check"},
 *         {"handle", "response", "send"}
 *     });
 *     corp.complete();
 *
 *     float idf = corp.getIdf("handle");      // log(N/df)
 *     float[] vec = corp.getRiVec("handle");   // enriched {@link FastCodeEmbed#SEM_DIM}-dim vector
 *
 *     FuncDescriptor a = corp.buildFunc("src/handler.c", new String[]{"handle", "request"});
 *     FuncDescriptor b = corp.buildFunc("src/auth.c", new String[]{"validate", "user"});
 *     float score = FastCodeEmbed.simpleScore(a, b);
 * }
 * }</pre>
 *
 * <p>Implements {@link AutoCloseable} for try-with-resources.</p>
 *
 * <h2>Thread safety</h2>
 * <p><b>close() must not race with any query or mutation method.</b>  The native
 * handle table uses per-slot refcounts to protect against use-after-free during
 * long-running parallel operations (search, finalize), but the close() method
 * blocks until all in-flight operations complete.  In practice this means:
 * close from the same thread that last used the corpus, or use the
 * try-with-resources pattern which serializes access.  Calling close() from
 * a different thread while search/finalize is running will block that thread
 * until the operation completes.</p>
 *
 * @since 0.0.1
 */
public class Corpus implements AutoCloseable {
    /* H-1: volatile so that a racing close() is visible to
     * query threads.  close() is synchronized to guarantee atomic test-and-clear
     * — only one thread can ever observe a non-zero handle and free it. */
    private volatile long handle;
    private boolean finalized;
    private final java.util.ArrayList<String> docPaths = new java.util.ArrayList<>();

    /* L-2: Cleaner backstop. If the caller forgets
     * close() / try-with-resources, the Cleaner reclaims native memory on
     * GC.  close() clears `handle` before freeing, so the action sees 0
     * and becomes a no-op — no double-free. */
    private static final java.lang.ref.Cleaner CLEANER = java.lang.ref.Cleaner.create();
    private final java.lang.ref.Cleaner.Cleanable cleanable;
    private final CloseAction closeAction;

    /** Action registered with the Cleaner. Holds the native handle only. */
    private static final class CloseAction implements Runnable {
        private final java.util.concurrent.atomic.AtomicLong h;
        CloseAction(long handle) { this.h = new java.util.concurrent.atomic.AtomicLong(handle); }
        @Override public void run() {
            /* JA-05: Use getAndSet(0) for atomic
             * check-and-clear. The old volatile check-then-act had a TOCTOU
             * race: the Cleaner thread could read h as non-zero, then the
             * explicit close() thread sets h to 0 and frees the corpus, then
             * the Cleaner thread also frees it — double-free. getAndSet(0)
             * guarantees exactly one thread sees the non-zero handle. */
            long v = h.getAndSet(0);
            if (v != 0) { FastCodeEmbed.freeCorpus(v); }
        }
    }

    /**
     * Create a new empty corpus.
     *
     * @throws OutOfMemoryError if the native library could not allocate
     *         the corpus: previously the constructor *         silently produced a 0-handle that surfaced as a confusing
     *         "Corpus is closed" IllegalStateException on the first call).
     * @throws UnsatisfiedLinkError if native library is not loaded
     */
    public Corpus() {
        this.handle = FastCodeEmbed.createCorpus();
        if (this.handle == FastCodeEmbed.CORPUS_OOM) {
            this.handle = 0;
            throw new OutOfMemoryError("fce_sem_corpus_new returned NULL (calloc/ht_create OOM)");
        }
        this.finalized = false;
        /* L-2: register Cleaner backstop. CloseAction holds
         * a private copy of the handle; close() clears both `this.handle` and
         * `closeAction.h` before freeing, so if GC runs the action after close(),
         * it sees h==0 and skips. */
        this.closeAction = new CloseAction(this.handle);
        this.cleanable = CLEANER.register(this, closeAction);
    }

    /**
     * Add a single document's tokens to the corpus.
     * Can be called multiple times before {@link #complete()}.
     *
     * @param tokens array of token strings for this document
     * @throws IllegalStateException if corpus is closed
     */
    public void addDoc(String[] tokens) {
        checkNotClosed();
        int before = FastCodeEmbed.getDocCount(handle);
        FastCodeEmbed.addDoc(handle, tokens);
        int after = FastCodeEmbed.getDocCount(handle);
        if (after > before) docPaths.add("");
        /* If native rejected the doc (count<=0 or >512 tokens), docPaths
         * stays in sync because we only append when the native count grew. */
    }

    /**
     * Add a single document's tokens with a file path.
     * The path is tracked for use in search result display.
     *
     * @param tokens array of token strings for this document
     * @param filePath source file path for this document
     * @throws IllegalStateException if corpus is closed
     */
    public void addDoc(String[] tokens, String filePath) {
        checkNotClosed();
        int before = FastCodeEmbed.getDocCount(handle);
        FastCodeEmbed.addDoc(handle, tokens);
        int after = FastCodeEmbed.getDocCount(handle);
        if (after > before) docPaths.add(filePath != null ? filePath : "");
    }

    /**
     * Batch-add documents. More efficient than repeated {@link #addDoc} calls.
     * All docs are internally padded to the length of the longest doc.
     *
     * @param docs array of token arrays, one per document
     * @throws IllegalStateException if corpus is closed
     */
    public void addDocsBatch(String[][] docs) {
        checkNotClosed();
        int maxLen = 0;
        for (String[] doc : docs) {
            if (doc.length > maxLen) maxLen = doc.length;
        }
        /* C21: reject all-empty input early —
         * the C side silently succeeds with doc_count unchanged, which can
         * confuse callers who expect an error for empty input. */
        if (maxLen == 0) throw new IllegalArgumentException("All documents are empty");
        int before = FastCodeEmbed.getDocCount(handle);
        FastCodeEmbed.addDocsBatch(handle, docs, maxLen);
        int added = FastCodeEmbed.getDocCount(handle) - before;
        /* No path mapping needed — add empty strings for all accepted docs. */
        for (int i = 0; i < added; i++) docPaths.add("");
    }

    /**
     * Batch-add documents with file paths. Paths are tracked for search result display.
     *
     * @param docs array of token arrays, one per document
     * @param paths per-document file paths
     * @throws IllegalStateException if corpus is closed
     * @throws IllegalArgumentException if docs.length != paths.length
     */
    public void addDocsBatch(String[][] docs, String[] paths) {
        checkNotClosed();
        if (docs.length != paths.length) {
            throw new IllegalArgumentException("docs.length (" + docs.length + ") != paths.length (" + paths.length + ")");
        }
        int maxLen = 0;
        for (String[] doc : docs) {
            if (doc.length > maxLen) maxLen = doc.length;
        }
        if (maxLen == 0) throw new IllegalArgumentException("All documents are empty");
        int before = FastCodeEmbed.getDocCount(handle);
        int[] docMap = FastCodeEmbed.addDocsBatch(handle, docs, maxLen);
        int added = FastCodeEmbed.getDocCount(handle) - before;
        /* M-07: use docMap to map paths correctly.
         * docMap[d] is the index of doc d in the valid docs list, or -1 if
         * rejected. We iterate all docs and only add paths for accepted ones. */
        if (docMap != null) {
            for (int d = 0; d < docMap.length && added > 0; d++) {
                if (docMap[d] >= 0) {
                    String p = paths[d] != null ? paths[d] : "";
                    docPaths.add(p);
                    added--;
                }
            }
        } else {
            /* Fallback: if docMap is NULL (OOM), use sequential mapping. */
            for (int i = 0; i < added; i++) {
                String p = paths[i] != null ? paths[i] : "";
                docPaths.add(p);
            }
        }
    }

    /**
     * Batch-add documents by raw names. Tokenization happens in C (one JNI call).
     * Much faster than tokenizeBatch + addDocsBatch for large corpora.
     *
     * @param names raw identifiers/names to tokenize and index
     */
    public void addDocsTokenized(String[] names) {
        checkNotClosed();
        int before = FastCodeEmbed.getDocCount(handle);
        FastCodeEmbed.addDocsTokenized(handle, names);
        int added = FastCodeEmbed.getDocCount(handle) - before;
        for (int i = 0; i < added; i++) docPaths.add("");
    }

    /**
     * Batch-add documents by raw names with file paths.
     *
     * @param names raw identifiers/names to tokenize and index
     * @param paths per-document file paths
     */
    public void addDocsTokenized(String[] names, String[] paths) {
        checkNotClosed();
        if (names.length != paths.length) {
            throw new IllegalArgumentException("names.length != paths.length");
        }
        int before = FastCodeEmbed.getDocCount(handle);
        FastCodeEmbed.addDocsTokenized(handle, names);
        int added = FastCodeEmbed.getDocCount(handle) - before;
        for (int i = 0; i < added; i++) {
            String p = paths[i] != null ? paths[i] : "";
            docPaths.add(p);
        }
    }

    /**
     * Read source files, chunk by } boundaries, tokenize, and add to corpus.
     * All work happens in C — no intermediate Java String objects created.
     * This is the fastest way to build a corpus from source files.
     *
     * <p>Best-effort semantics: if some files are
     * rejected by the C side (oversize, IO error, empty), the returned
     * {@code fileDocCounts[i]} for those indices is 0 and {@code docPaths}
     * is not extended for them. The {@code result} return value is the
     * total number of chunks added (sum of {@code fileDocCounts}).
     * Callers that need strict per-file success can iterate the inputs
     * and check {@code fileDocCounts[i] > 0}.</p>
     *
     * <p>Pre-validation: a {@code chunkSize <= 0} is
     * rejected by the C side with a -1 return; no exception is thrown.
     * Callers are expected to pre-validate user input if needed.</p>
     *
     * @param paths file paths to read and index
     * @param chunkSize target bytes per chunk (chunks split at } boundaries)
     * @param maxTokensPerChunk max tokens per chunk (0 = 512 default)
     * @return number of documents added, or -1 on error
     */
    public int addFiles(String[] paths, int chunkSize, int maxTokensPerChunk) {
        checkNotClosed();
        if (paths == null) {
            throw new IllegalArgumentException("paths must not be null");
        }
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize must be > 0, got " + chunkSize);
        }
        int[] fileDocCounts = new int[paths.length];
        int result = FastCodeEmbed.addFiles(handle, paths, chunkSize, fileDocCounts, maxTokensPerChunk);
        /* Build docPaths: one entry per document (chunk), using file path.
         * Rejected files (fileDocCounts[i] == 0) contribute no entries, so
         * docPaths stays in sync with the corpus's actual doc count. */
        for (int i = 0; i < paths.length; i++) {
            String p = paths[i] != null ? paths[i] : "";
            for (int d = 0; d < fileDocCounts[i]; d++) {
                docPaths.add(p);
            }
        }
        return result;
    }

    /**
     * Read source files with default max tokens per chunk (512).
     */
    public int addFiles(String[] paths, int chunkSize) {
        return addFiles(paths, chunkSize, 0);
    }

    /**
     * Finalize the corpus: compute IDF weights and enriched RI vectors.
     * Must be called before any querying. Idempotent.
     *
     * @throws IllegalStateException if corpus is closed
     */
    public void complete() {
        checkNotClosed();
        if (!FastCodeEmbed.finalizeCorpus(handle)) {
            throw new IllegalStateException("Corpus finalization failed (out of memory)");
        }
        this.finalized = true;
    }

    /**
     * Get IDF weight for a token. Returns 0.0 for unknown tokens
     * and for tokens that appear in every document (IDF = log(1) = 0).
     *
     * @param token the token to look up
     * @return IDF weight ≥ 0.0
     * @throws IllegalStateException if {@link #complete()} has not been called
     */
    public float getIdf(String token) {
        checkFinalized();
        return FastCodeEmbed.getIdf(handle, token);
    }

    /**
     * Get the enriched Random Indexing vector for a token (after co-occurrence).
     *
     * @param token the token to look up
     * @return {@link FastCodeEmbed#SEM_DIM}-dimensional float array, or null if token is unknown
     * @throws IllegalStateException if {@link #complete()} has not been called
     */
    public float[] getRiVec(String token) {
        checkFinalized();
        return FastCodeEmbed.getRiVec(handle, token);
    }

    /**
     * Number of documents registered in the corpus.
     *
     * @return document count
     */
    public int getDocCount() {
        checkNotClosed();
        return FastCodeEmbed.getDocCount(handle);
    }

    /**
     * Number of unique tokens (vocabulary size) after finalization.
     *
     * @return vocabulary size
     */
    public int getTokenCount() {
        checkNotClosed();
        return FastCodeEmbed.getTokenCount(handle);
    }

    /**
     * Clear tracked file paths to free memory.
     * Call after finalization if paths are no longer needed.
     */
    public void clearDocPaths() {
        docPaths.clear();
        docPaths.trimToSize();
    }

    /**
     * Build a FuncDescriptor for querying against this corpus.
     * Tokens not in the corpus get IDF = 0.0 and zero RI vectors.
     * <p>
     * <b>WARNING — positional indices:</b>
     * the TF-IDF indices in the resulting FuncDescriptor are positional
     * (0, 1, 2, …) rather than corpus vocabulary IDs. This is safe for use
     * with {@code simpleSearch} and {@code simpleRank} (which use RI-based
     * scoring and ignore the TF-IDF indices), but the TF-IDF cosine
     * component will <b>not</b> produce meaningful results with
     * struct-based scoring APIs ({@code fce_sem_combined_score} /
     * {@code nCombinedScore} / similar). Use {@link #extractFlat} +
     * {@code simpleRankBatch} (the flat-array scoring path) for the
     * intended use case.
     * </p>
     *
     * @param filePath file path for the function
     * @param tokens   token strings for this function
     * @return a ready-to-score FuncDescriptor (positional indices)
     * @throws IllegalStateException if {@link #complete()} has not been called
     * @deprecated Use {@code extractFlat} + {@code simpleRankBatch} for the
     *             fast path; this method's FuncDescriptor is positional
     *             and only safe with the simple* scoring family.
     */
    @Deprecated
    public FuncDescriptor buildFunc(String filePath, String[] tokens) {
        checkFinalized();
        int[] indices = new int[tokens.length];
        float[] weights = new float[tokens.length];
        float[] riVec = new float[FastCodeEmbed.SEM_DIM];

        for (int i = 0; i < tokens.length; i++) {
            indices[i] = i;
            weights[i] = getIdf(tokens[i]);
            float[] rv = getRiVec(tokens[i]);
            if (rv != null) {
                for (int d = 0; d < FastCodeEmbed.SEM_DIM; d++) {
                    riVec[d] += rv[d];
                }
            }
        }

        FuncDescriptor fd = new FuncDescriptor(filePath);
        fd.setTfidf(indices, weights);
        fd.setRiVec(riVec);
        return fd;
    }

    /**
     * Extract flat arrays from FuncDescriptors for use with
     * {@link FastCodeEmbed#simpleRankBatch}. Call once after building
     * all FuncDescriptors; reuse the result across multiple queries.
     *
     * @param funcs array of built FuncDescriptors
     * @return flat arrays ready for batch ranking
     */
    public static FlatCorpus extractFlat(FuncDescriptor[] funcs) {
        int maxTokens = 0;
        for (FuncDescriptor f : funcs) {
            if (f.getTfidfLen() > maxTokens) maxTokens = f.getTfidfLen();
        }

        int n = funcs.length;
        /* n * maxTokens and n * SEM_DIM are computed in long
         * arithmetic so very large query-side corpora fail loudly with an
         * IllegalArgumentException instead of silently corrupting the flat
         * layout via 32-bit overflow. */
        long tfidfTotalLong = (long) n * (long) maxTokens;
        long riTotalLong = (long) n * (long) FastCodeEmbed.SEM_DIM;
        if (tfidfTotalLong > Integer.MAX_VALUE || riTotalLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                "flat corpus too large: n=" + n + ", maxTokens=" + maxTokens +
                ", dim=" + FastCodeEmbed.SEM_DIM);
        }
        int tfidfTotal = (int) tfidfTotalLong;
        int riTotal = (int) riTotalLong;

        float[] allWeights = new float[tfidfTotal];
        int[] allIndices = new int[tfidfTotal];
        int[] tfidfLens = new int[n];
        float[] allRiVecs = new float[riTotal];
        String[] filePaths = new String[n];

        for (int f = 0; f < n; f++) {
            FuncDescriptor fd = funcs[f];
            filePaths[f] = fd.getFilePath();
            tfidfLens[f] = fd.getTfidfLen();

            int[] idx = fd.getTfidfIndices();
            float[] w = fd.getTfidfWeights();
            if (idx != null && w != null) {
                System.arraycopy(idx, 0, allIndices, f * maxTokens, idx.length);
                System.arraycopy(w, 0, allWeights, f * maxTokens, w.length);
            }

            float[] ri = fd.getRiVec();
            if (ri != null) {
                System.arraycopy(ri, 0, allRiVecs, f * FastCodeEmbed.SEM_DIM, FastCodeEmbed.SEM_DIM);
            }
        }

        return new FlatCorpus(allWeights, allIndices, tfidfLens, allRiVecs, filePaths, maxTokens);
    }

    /**
     * Get the file path for a document by its corpus index.
     *
     * @param index document index (0-based)
     * @return file path, or empty string if not set
     */
    public String getDocPath(int index) {
        if (index < 0 || index >= docPaths.size()) return "";
        return docPaths.get(index);
    }

    /**
     * Get all tracked file paths (one per document, in corpus order).
     *
     * @return array of file paths
     */
    public String[] getDocPaths() {
        return docPaths.toArray(new String[0]);
    }

    /**
     * Get the native corpus handle (for JNI use).
     *
     * @return native handle
     */
    long getHandle() {
        return handle;
    }

    /**
     * Free native resources. Safe to call multiple times.
     *
     * <p><b>Must not be called concurrently with any query or mutation method</b>
     * on the same Corpus instance.  The native handle table does not prevent
     * use-after-free when close() races with a query (see class Thread safety
     * section).  Use try-with-resources or ensure all queries complete before
     * calling close.</p>
     *
     * <p>H-1: clears both the field handle and the
     * Cleaner action's handle, then frees via the Cleaner (idempotent).
     * The volatile write to closeAction.h ensures the Cleaner thread
     * sees 0 and skips if it races with this call.</p>
     */
    public synchronized void close() {
        long h = handle;
        handle = 0;
        closeAction.h.set(0);
        if (h != 0) FastCodeEmbed.freeCorpus(h);
        /* Deregister the Cleaner so it doesn't run during JVM shutdown.
         * Without this, the Cleaner thread may attempt to call into native
         * code after the JNI library is being unloaded, causing SIGTRAP. */
        cleanable.clean();
    }

    private void checkNotClosed() {
        if (handle == 0) throw new IllegalStateException("Corpus is closed");
    }

    private void checkFinalized() {
        checkNotClosed();
        if (!finalized) throw new IllegalStateException("Call complete() before querying");
    }

    /**
     * Pre-extracted flat arrays for batch ranking.
     * Created by {@link #extractFlat(FuncDescriptor[])}.
     * Reusable across multiple queries.
     * RI vectors are laid out as {@code [func * FastCodeEmbed.SEM_DIM + dim]}.
     */
    public static class FlatCorpus {
        public final float[] allWeights;
        public final int[] allIndices;
        public final int[] tfidfLens;
        public final float[] allRiVecs;
        public final String[] filePaths;
        public final int maxTokens;

        FlatCorpus(float[] allWeights, int[] allIndices, int[] tfidfLens,
                   float[] allRiVecs, String[] filePaths, int maxTokens) {
            this.allWeights = allWeights;
            this.allIndices = allIndices;
            this.tfidfLens = tfidfLens;
            this.allRiVecs = allRiVecs;
            this.filePaths = filePaths;
            this.maxTokens = maxTokens;
        }

        /** Number of functions in this corpus. */
        public int size() { return filePaths.length; }
    }
}
