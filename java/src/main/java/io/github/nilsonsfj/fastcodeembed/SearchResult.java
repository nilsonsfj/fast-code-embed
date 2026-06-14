package io.github.nilsonsfj.fastcodeembed;

/**
 * A ranked search result: corpus index + similarity score.
 *
 * <p>Returned by {@link FastCodeEmbed#simpleRank} and
 * {@link FastCodeEmbed#simpleSearch}. Results are sorted by
 * score descending (best match first).</p>
 *
 * @since 0.0.1
 */
public final class SearchResult {
    private final int index;
    private final float score;

    /**
     * @param index index into the corpus array that was searched
     * @param score similarity score [0.0, 1.0] for simple APIs
     */
    public SearchResult(int index, float score) {
        this.index = index;
        this.score = score;
    }

    /**
     * Index into the corpus array that was searched.
     * Use this to look up the original function metadata.
     *
     * @return corpus index
     */
    public int getIndex() { return index; }

    /**
     * Similarity score. For the simple API this is normalized to [0.0, 1.0].
     *
     * @return score
     */
    public float getScore() { return score; }

    /** @return human-readable representation */
    @Override
    public String toString() {
        return String.format("SearchResult{index=%d, score=%.4f}", index, score);
    }
}
