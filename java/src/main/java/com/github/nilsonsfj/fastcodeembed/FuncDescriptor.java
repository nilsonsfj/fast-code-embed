package com.github.nilsonsfj.fastcodeembed;

import java.util.Objects;

/**
 * Java-side representation of a function's semantic data for scoring.
 *
 * <p>At minimum, set {@code filePath} and TF-IDF ({@link #setTfidf}).
 * RI vectors are optional — if not set, scoring uses TF-IDF only.</p>
 *
 * <h2>Example</h2>
 * <pre>{@code
 * FuncDescriptor fd = new FuncDescriptor("src/handler.c");
 * fd.setTfidf(new int[]{0, 1}, new float[]{0.8f, 1.2f});
 * fd.setRiVec(enrichedVec768d);
 * }</pre>
 *
 * <p>Use {@link Corpus#buildFunc} as a convenience to populate all fields
 * from a finalized corpus.</p>
 *
 * @since 0.0.1
 */
public class FuncDescriptor {
    private final String filePath;
    private int[] tfidfIndices;
    private float[] tfidfWeights;
    private float[] riVec;

    /**
     * Create a function descriptor for the given file path.
     *
     * @param filePath source file path (used for module proximity)
     * @throws NullPointerException if filePath is null
     */
    public FuncDescriptor(String filePath) {
        this.filePath = Objects.requireNonNull(filePath, "filePath must not be null");
    }

    /**
     * Set TF-IDF sparse vector as parallel index/weight arrays.
     *
     * <p><b>Important:</b> {@code indices} must be sorted in strictly ascending
     * order. The C-side sparse TF-IDF cosine scorer
     * ({@code fce_sparse_tfidf_cosine}) relies on this invariant for its
     * merge-loop correctness. Unsorted indices produce silently too-low scores
     * with no error in release builds.</p>
     *
     * @param indices token indices (into the corpus vocabulary), ascending
     * @param weights corresponding IDF weights
     * @throws NullPointerException with named arg if either input is null
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public void setTfidf(int[] indices, float[] weights) {
        /* J-2 (review 0002 §2.6): the previous code would NPE on null input
         * with no argument name in the stack. Objects.requireNonNull gives
         * a useful diagnostic ("indices" or "weights") for the common
         * "I passed getIdf() == null" mistake. */
        Objects.requireNonNull(indices, "indices");
        Objects.requireNonNull(weights, "weights");
        if (indices.length != weights.length) {
            throw new IllegalArgumentException("indices and weights must have same length");
        }
        /* C23: validate sorted ascending order.
         * The C-side merge-loop in fce_sparse_tfidf_cosine assumes this
         * invariant. Unsorted indices silently produce too-low scores.
         * O(n) scan is cheap and catches the bug early. */
        for (int i = 1; i < indices.length; i++) {
            if (indices[i] <= indices[i - 1]) {
                throw new IllegalArgumentException(
                    "tfidf indices must be sorted ascending (found index[" + i + "]=" +
                    indices[i] + " <= index[" + (i-1) + "]=" + indices[i-1] + ")");
            }
        }
        this.tfidfIndices = indices;
        this.tfidfWeights = weights;
    }

    /**
     * Set the 768-dimensional Random Indexing vector.
     * <p>
     * L-9: The dimension is hardcoded to 768 to match
     * {@code FCE_SEM_DIM} in the C library. If the library is compiled with
     * {@code FCE_SEM_DIM_256}, the JNI layer still uses 768 (the preprocessor
     * dimension), so the Java bindings only support the default 768-dim build.
     * Typically built by summing enriched token vectors from a finalized corpus.
     *
     * @param vec float array of length 768
     * @throws NullPointerException with named arg if vec is null
     * @throws IllegalArgumentException if vec.length != 768
     */
    public void setRiVec(float[] vec) {
        /* J-2 (review 0002 §2.6): same as setTfidf. */
        Objects.requireNonNull(vec, "vec");
        if (vec.length != 768) {
            throw new IllegalArgumentException("RI vector must be 768 dimensions, got " + vec.length);
        }
        this.riVec = vec;
    }

    /** @return source file path */
    public String getFilePath() { return filePath; }

    /** @return TF-IDF token indices, or null if not set */
    public int[] getTfidfIndices() { return tfidfIndices; }

    /** @return TF-IDF weights, or null if not set */
    public float[] getTfidfWeights() { return tfidfWeights; }

    /** @return 768-dimensional RI vector, or null if not set */
    public float[] getRiVec() { return riVec; }

    /** @return number of TF-IDF entries, or 0 if not set */
    public int getTfidfLen() {
        return tfidfIndices != null ? tfidfIndices.length : 0;
    }
}
