package com.github.nilsonsfj.fastcodeembed;

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
     */
    public FuncDescriptor(String filePath) {
        this.filePath = filePath;
    }

    /**
     * Set TF-IDF sparse vector as parallel index/weight arrays.
     *
     * @param indices token indices (into the corpus vocabulary)
     * @param weights corresponding IDF weights
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public void setTfidf(int[] indices, float[] weights) {
        if (indices.length != weights.length) {
            throw new IllegalArgumentException("indices and weights must have same length");
        }
        this.tfidfIndices = indices;
        this.tfidfWeights = weights;
    }

    /**
     * Set the 768-dimensional Random Indexing vector.
     * Typically built by summing enriched token vectors from a finalized corpus.
     *
     * @param vec float array of length 768
     * @throws IllegalArgumentException if vec.length != 768
     */
    public void setRiVec(float[] vec) {
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
