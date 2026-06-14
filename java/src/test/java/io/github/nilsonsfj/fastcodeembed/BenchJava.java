package io.github.nilsonsfj.fastcodeembed;

/**
 * Benchmark fast-code-embed Java binding using the project's own source code.
 *
 * Measures: init, tokenization, corpus build (batch), IDF lookup,
 *           RI vector access, simple scoring, ranking.
 *
 * Build + run: cd java && ./build.sh bench
 */
public class BenchJava {

    static final String[][] CORPUS = {
        /* semantic/semantic.c */
        {"src/semantic/semantic.c", "fce", "sem", "get", "config"},
        {"src/semantic/semantic.c", "fce", "sem", "is", "enabled"},
        {"src/semantic/semantic.c", "fce", "sem", "tokenize"},
        {"src/semantic/semantic.c", "fce", "sem", "cosine"},
        {"src/semantic/semantic.c", "fce", "sem", "random", "index"},
        {"src/semantic/semantic.c", "fce", "sem", "ensure", "ready"},
        {"src/semantic/semantic.c", "fce", "sem", "normalize"},
        {"src/semantic/semantic.c", "fce", "sem", "vec", "add", "scaled"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "new"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "add", "doc"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "add", "docs", "batch"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "finalize"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "idf"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "ri", "vec"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "doc", "count"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "token", "count"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "token", "at"},
        {"src/semantic/semantic.c", "fce", "sem", "corpus", "free"},
        {"src/semantic/semantic.c", "fce", "sem", "combined", "score"},
        {"src/semantic/semantic.c", "fce", "sem", "proximity"},
        {"src/semantic/semantic.c", "fce", "sem", "rank"},
        {"src/semantic/semantic.c", "fce", "sem", "search"},
        {"src/semantic/semantic.c", "fce", "sem", "simple", "score"},
        {"src/semantic/semantic.c", "fce", "sem", "simple", "rank"},
        {"src/semantic/semantic.c", "fce", "sem", "simple", "search"},
        {"src/semantic/semantic.c", "fce", "sem", "diffuse"},
        {"src/semantic/semantic.c", "sparse", "tfidf", "cosine"},
        {"src/semantic/semantic.c", "small", "cosine"},
        {"src/semantic/semantic.c", "ranked", "cmp"},
        {"src/semantic/semantic.c", "flush", "token"},

        /* foundation/hash_table.c */
        {"src/foundation/hash_table.c", "fce", "ht", "create"},
        {"src/foundation/hash_table.c", "fce", "ht", "free"},
        {"src/foundation/hash_table.c", "fce", "ht", "set"},
        {"src/foundation/hash_table.c", "fce", "ht", "get"},
        {"src/foundation/hash_table.c", "fce", "ht", "has"},
        {"src/foundation/hash_table.c", "fce", "ht", "delete"},
        {"src/foundation/hash_table.c", "fce", "ht", "count"},
        {"src/foundation/hash_table.c", "fce", "ht", "foreach"},
        {"src/foundation/hash_table.c", "fce", "ht", "clear"},
        {"src/foundation/hash_table.c", "ht", "resize"},

        /* foundation/platform.c */
        {"src/foundation/platform.c", "fce", "mmap", "read"},
        {"src/foundation/platform.c", "fce", "munmap"},
        {"src/foundation/platform.c", "fce", "now", "ns"},
        {"src/foundation/platform.c", "fce", "now", "ms"},
        {"src/foundation/platform.c", "fce", "nprocs"},
        {"src/foundation/platform.c", "fce", "file", "exists"},
        {"src/foundation/platform.c", "fce", "is", "dir"},
        {"src/foundation/platform.c", "fce", "file", "size"},
        {"src/foundation/platform.c", "fce", "normalize", "path", "sep"},
        {"src/foundation/platform.c", "fce", "safe", "getenv"},
        {"src/foundation/platform.c", "fce", "get", "home", "dir"},
        {"src/foundation/platform.c", "fce", "app", "config", "dir"},
        {"src/foundation/platform.c", "fce", "app", "local", "dir"},
        {"src/foundation/platform.c", "fce", "resolve", "cache", "dir"},

        /* foundation/system_info.c */
        {"src/foundation/system_info.c", "fce", "system", "info"},
        {"src/foundation/system_info.c", "fce", "default", "worker", "count"},

        /* foundation/log.c */
        {"src/foundation/log.c", "fce", "log", "set", "sink"},
        {"src/foundation/log.c", "fce", "log", "set", "level"},
        {"src/foundation/log.c", "fce", "log", "get", "level"},
        {"src/foundation/log.c", "fce", "log"},

        /* foundation/profile.c */
        {"src/foundation/profile.c", "fce", "profile", "init"},
        {"src/foundation/profile.c", "fce", "profile", "enable"},
        {"src/foundation/profile.c", "fce", "profile", "now"},
        {"src/foundation/profile.c", "fce", "profile", "log", "elapsed"},

        /* pipeline/worker_pool.c */
        {"src/pipeline/worker_pool.c", "fce", "parallel", "for"},
        {"src/pipeline/worker_pool.c", "pthread", "worker"},
        {"src/pipeline/worker_pool.c", "run", "serial"},
        {"src/pipeline/worker_pool.c", "run", "pthreads"},

        /* version.c */
        {"src/version.c", "fce", "version"},
    };

    public static void main(String[] args) {
        System.out.println("fast-code-embed Java benchmark");
        System.out.println("==================================");
        System.out.printf("Corpus: %d functions from this project's source%n%n", CORPUS.length);

        int iterations = 100;

        /* ── 1. Init ──────────────────────────────────────────── */
        long t0 = System.nanoTime();
        FastCodeEmbed.init();
        double initMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  init (ensure_ready)         %8.2f ms%n", initMs);

        /* ── 2. Tokenize (individual calls) ─────────────────────── */
        t0 = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            for (String[] row : CORPUS) {
                FastCodeEmbed.tokenize(row[1]);
            }
        }
        double tokMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  tokenize individual × %d      %8.2f ms  (%.1f µs/call)%n",
                iterations, tokMs, tokMs * 1000.0 / (CORPUS.length * iterations));

        /* ── 2b. Tokenize (batch) ─────────────────────────────── */
        String[] rawNames = new String[CORPUS.length];
        for (int i = 0; i < CORPUS.length; i++) rawNames[i] = CORPUS[i][1];
        t0 = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            FastCodeEmbed.tokenizeBatch(rawNames);
        }
        double tokBatchMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  tokenize batch × %d           %8.2f ms  (%.1f µs/call)  [%.1fx faster]%n",
                iterations, tokBatchMs, tokBatchMs * 1000.0 / iterations,
                tokMs / tokBatchMs);

        /* ── 3. Corpus build (batch) ──────────────────────────── */
        String[][] docs = new String[CORPUS.length][];
        for (int i = 0; i < CORPUS.length; i++) {
            docs[i] = new String[CORPUS[i].length - 1];
            System.arraycopy(CORPUS[i], 1, docs[i], 0, docs[i].length);
        }

        t0 = System.nanoTime();
        Corpus corp = new Corpus();
        corp.addDocsBatch(docs);
        corp.complete();
        double buildMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  corpus build (batch)         %8.2f ms%n", buildMs);

        /* ── 4. IDF lookup ────────────────────────────────────── */
        t0 = System.nanoTime();
        float idfSum = 0;
        for (int i = 0; i < iterations; i++) {
            for (String[] row : CORPUS) {
                idfSum += corp.getIdf(row[1]);
            }
        }
        double idfMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  IDF lookup %d × %d            %8.2f ms  (%.1f µs/call)%n",
                CORPUS.length, iterations, idfMs,
                idfMs * 1000.0 / (CORPUS.length * iterations));

        /* ── 5. RI vector access ──────────────────────────────── */
        t0 = System.nanoTime();
        float vecSum = 0;
        for (int i = 0; i < iterations; i++) {
            for (String[] row : CORPUS) {
                float[] v = corp.getRiVec(row[1]);
                if (v != null) vecSum += v[0];
            }
        }
        double riMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  RI vec access %d × %d         %8.2f ms  (%.1f µs/call)%n",
                CORPUS.length, iterations, riMs,
                riMs * 1000.0 / (CORPUS.length * iterations));

        /* ── 6. Build FuncDescriptors ────────────────────────── */
        FuncDescriptor[] funcs = new FuncDescriptor[CORPUS.length];
        for (int i = 0; i < CORPUS.length; i++) {
            String[] toks = new String[CORPUS[i].length - 1];
            System.arraycopy(CORPUS[i], 1, toks, 0, toks.length);
            funcs[i] = corp.buildFunc(CORPUS[i][0], toks);
        }

        /* ── 7. Simple scoring (all pairs) ────────────────────── */
        int uniquePairs = CORPUS.length * (CORPUS.length - 1) / 2;
        t0 = System.nanoTime();
        float scoreSum = 0;
        for (int i = 0; i < iterations; i++) {
            for (int a = 0; a < CORPUS.length; a++) {
                for (int b = a + 1; b < CORPUS.length; b++) {
                    scoreSum += FastCodeEmbed.simpleScore(funcs[a], funcs[b]);
                }
            }
        }
        double scoreMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  simple_score %d pairs × %d    %8.2f ms  (%.1f µs/pair)%n",
                uniquePairs, iterations, scoreMs,
                scoreMs * 1000.0 / (uniquePairs * iterations));

        /* ── 8. Ranking (old API — per-item JNI marshaling) ──── */
        t0 = System.nanoTime();
        SearchResult[] results = null;
        for (int i = 0; i < iterations; i++) {
            results = FastCodeEmbed.simpleRank(funcs[0], funcs, 10);
        }
        double rankMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  simple_rank old (top 10) × %d  %7.2f ms  (%.1f µs/call)%n",
                iterations, rankMs, rankMs * 1000.0 / iterations);

        /* ── 9. Ranking (batch/flat — zero per-item overhead) ── */
        Corpus.FlatCorpus flat = Corpus.extractFlat(funcs);
        int[] qIdxInt = funcs[0].getTfidfIndices();

        t0 = System.nanoTime();
        SearchResult[] batchResults = null;
        for (int i = 0; i < iterations; i++) {
            batchResults = FastCodeEmbed.simpleRankBatch(
                flat.allWeights, flat.allIndices, flat.tfidfLens,
                flat.allRiVecs, flat.filePaths, flat.maxTokens,
                qIdxInt, funcs[0].getTfidfWeights(), funcs[0].getRiVec(), 10);
        }
        double batchRankMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  simple_rank flat (top 10) × %d %6.2f ms  (%.1f µs/call)  [%.1fx faster]%n",
                iterations, batchRankMs, batchRankMs * 1000.0 / iterations,
                rankMs / batchRankMs);

        /* ── Summary ──────────────────────────────────────────── */
        System.out.println();
        System.out.printf("  Corpus size: %d functions%n", CORPUS.length);
        System.out.printf("  Vocabulary:  %d tokens%n", corp.getTokenCount());
        System.out.printf("  Top match:   results[0] = idx %d, score %.4f%n",
                results[0].getIndex(), results[0].getScore());
        System.out.printf("  score_sum:   %.6f (prevents optimization)%n", scoreSum);

        corp.close();
    }
}
