package com.github.nilsonsfj.fastcodeembed;

/**
 * Standalone test runner for Java JNI binding — no test framework needed.
 * Covers: batch corpus ops, simple scoring, ranking, search.
 */
public class FastCodeEmbedTest {
    static int passed = 0;
    static int failed = 0;

    static void test(String name, Runnable r) {
        System.out.printf("  %-50s ", name);
        System.out.flush();
        try {
            r.run();
            System.out.println("OK");
            passed++;
        } catch (Throwable t) {
            System.out.println("FAIL: " + t.getMessage());
            failed++;
        }
    }

    static void assertTrue(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }

    static void assertEquals(float expected, float actual, float eps, String msg) {
        if (Math.abs(expected - actual) > eps)
            throw new AssertionError(msg + " (expected " + expected + ", got " + actual + ")");
    }

    static void assertEquals(String expected, String actual, String msg) {
        if (!expected.equals(actual))
            throw new AssertionError(msg + " (expected '" + expected + "', got '" + actual + "')");
    }

    static void assertEquals(int expected, int actual, String msg) {
        if (expected != actual) throw new AssertionError(msg + " (expected " + expected + ", got " + actual + ")");
    }

    static void assertNotNull(Object obj, String msg) {
        if (obj == null) throw new AssertionError(msg);
    }

    /* ── Tests ────────────────────────────────────────────────── */

    public static void main(String[] args) {
        FastCodeEmbed.init();

        System.out.println("fast-code-embed Java binding tests");
        System.out.println("=======================================");

        System.out.println("\nTokenization:");
        test("tokenize camelCase", () -> {
            String[] t = FastCodeEmbed.tokenize("handleRequest");
            assertEquals(2, t.length, "token count");
            assertEquals("handle", t[0], "token 0");
            assertEquals("request", t[1], "token 1");
        });
        test("tokenize snake_case", () -> {
            String[] t = FastCodeEmbed.tokenize("handle_request");
            assertEquals(2, t.length, "token count");
            assertEquals("handle", t[0], "token 0");
            assertEquals("request", t[1], "token 1");
        });

        System.out.println("\nCorpus (batch):");
        test("batch add + finalize + IDF", () -> {
            try (Corpus corp = new Corpus()) {
                corp.addDocsBatch(new String[][]{
                    {"handle", "request", "parse"},
                    {"validate", "user", "check"},
                    {"handle", "response", "send"}
                });
                assertEquals(3, corp.getDocCount(), "doc count");

                corp.complete();

                float idfHandle = corp.getIdf("handle");  // 2/3 docs
                float idfReq = corp.getIdf("request");    // 1/3 docs
                assertTrue(idfHandle > 0.0f, "IDF handle > 0");
                assertTrue(idfReq > idfHandle, "IDF request > IDF handle");
                assertEquals(0.0f, corp.getIdf("nonexistent"), 0.001f, "unknown token IDF = 0");
            }
        });

        test("single doc add", () -> {
            try (Corpus corp = new Corpus()) {
                corp.addDoc(new String[]{"alpha", "beta"});
                corp.complete();
                assertEquals(1, corp.getDocCount(), "doc count");
                // IDF is log(N/df) = log(1/1) = 0 for single doc — that's correct
                assertEquals(0.0f, corp.getIdf("alpha"), 0.001f, "IDF alpha in single doc");
            }
        });

        test("RI vector enriched", () -> {
            try (Corpus corp = new Corpus()) {
                corp.addDocsBatch(new String[][]{{"foo", "bar"}, {"baz", "bar"}});
                corp.complete();

                float[] vec = corp.getRiVec("bar");
                assertNotNull(vec, "RI vec not null");
                assertEquals(768, vec.length, "vec dim");

                boolean nonZero = false;
                for (float v : vec) if (v != 0.0f) { nonZero = true; break; }
                assertTrue(nonZero, "enriched vec should be non-zero");

                float[] unknown = corp.getRiVec("nonexistent");
                assertTrue(unknown == null, "unknown token vec should be null");
            }
        });

        test("buildFunc convenience", () -> {
            try (Corpus corp = new Corpus()) {
                corp.addDocsBatch(new String[][]{{"handle", "request"}, {"validate", "user"}});
                corp.complete();

                FuncDescriptor a = corp.buildFunc("src/handler.c", new String[]{"handle", "request"});
                assertEquals("src/handler.c", a.getFilePath(), "filepath");
                assertEquals(2, a.getTfidfLen(), "tfidf len");
                assertTrue(a.getTfidfWeights()[0] > 0.0f, "tfidf weight > 0");
                assertNotNull(a.getRiVec(), "ri vec not null");
            }
        });

        System.out.println("\nSimple scoring:");
        test("identical funcs → high score", () -> {
            try (Corpus corp = new Corpus()) {
                corp.addDocsBatch(new String[][]{{"handle", "request"}, {"validate", "user"}});
                corp.complete();
                FuncDescriptor a = corp.buildFunc("src/handler.c", new String[]{"handle", "request"});
                FuncDescriptor b = corp.buildFunc("src/handler.c", new String[]{"handle", "request"});
                float score = FastCodeEmbed.simpleScore(a, b);
                assertTrue(score > 0.5f, "expected > 0.5, got " + score);
                assertTrue(score <= 1.0f, "expected <= 1.0");
            }
        });

        test("different funcs → lower score", () -> {
            try (Corpus corp = new Corpus()) {
                corp.addDocsBatch(new String[][]{{"handle", "request"}, {"validate", "user"}});
                corp.complete();
                FuncDescriptor a = corp.buildFunc("src/handler.c", new String[]{"handle", "request"});
                FuncDescriptor b = corp.buildFunc("lib/utils.c", new String[]{"validate", "user"});
                float score = FastCodeEmbed.simpleScore(a, b);
                assertTrue(score < 0.9f, "expected < 0.9, got " + score);
            }
        });

        test("score always in [0,1]", () -> {
            FuncDescriptor a = new FuncDescriptor("a.c");
            a.setTfidf(new int[]{}, new float[]{});
            a.setRiVec(new float[768]);
            FuncDescriptor b = new FuncDescriptor("b.c");
            b.setTfidf(new int[]{}, new float[]{});
            b.setRiVec(new float[768]);
            float score = FastCodeEmbed.simpleScore(a, b);
            assertTrue(score >= 0.0f, "score >= 0, got " + score);
            assertTrue(score <= 1.0f, "score <= 1, got " + score);
        });

        System.out.println("\nRanking:");
        test("simpleRank returns sorted results", () -> {
            try (Corpus corp = new Corpus()) {
                corp.addDocsBatch(new String[][]{
                    {"handle", "request"},
                    {"validate", "user"},
                    {"handle", "response"}
                });
                corp.complete();

                FuncDescriptor query = corp.buildFunc("q.c", new String[]{"handle", "request"});
                FuncDescriptor[] corpus = {
                    corp.buildFunc("a.c", new String[]{"handle", "request"}),
                    corp.buildFunc("b.c", new String[]{"validate", "user"}),
                    corp.buildFunc("c.c", new String[]{"handle", "response"})
                };

                SearchResult[] results = FastCodeEmbed.simpleRank(query, corpus, 10);
                assertTrue(results.length > 0, "should have results");
                for (int i = 1; i < results.length; i++) {
                    assertTrue(results[i - 1].getScore() >= results[i].getScore(),
                        "descending order violated at index " + i);
                }
            }
        });

        test("simpleSearch respects minScore", () -> {
            try (Corpus corp = new Corpus()) {
                corp.addDocsBatch(new String[][]{
                    {"handle", "request"},
                    {"validate", "user"},
                    {"handle", "response"}
                });
                corp.complete();

                FuncDescriptor query = corp.buildFunc("q.c", new String[]{"handle", "request"});
                FuncDescriptor[] corpus = {
                    corp.buildFunc("a.c", new String[]{"handle", "request"}),
                    corp.buildFunc("b.c", new String[]{"validate", "user"}),
                    corp.buildFunc("c.c", new String[]{"handle", "response"})
                };

                SearchResult[] high = FastCodeEmbed.simpleSearch(query, corpus, 10, 0.8f);
                SearchResult[] low = FastCodeEmbed.simpleSearch(query, corpus, 10, 0.0f);
                assertTrue(high.length <= low.length,
                    "high threshold should return fewer/equal results");
            }
        });

        test("simpleRank respects topK", () -> {
            try (Corpus corp = new Corpus()) {
                String[][] docs = new String[5][];
                docs[0] = new String[]{"a", "b"};
                docs[1] = new String[]{"c", "d"};
                docs[2] = new String[]{"e", "f"};
                docs[3] = new String[]{"g", "h"};
                docs[4] = new String[]{"i", "j"};
                corp.addDocsBatch(docs);
                corp.complete();

                FuncDescriptor query = corp.buildFunc("q.c", new String[]{"a", "b"});
                FuncDescriptor[] corpus = new FuncDescriptor[5];
                for (int i = 0; i < 5; i++) {
                    corpus[i] = corp.buildFunc("c" + i + ".c", docs[i]);
                }

                SearchResult[] results = FastCodeEmbed.simpleRank(query, corpus, 2);
                assertTrue(results.length <= 2, "should return at most topK results");
            }
        });

        System.out.println("\nBatch ranking (flat arrays):");
        test("simpleRankBatch returns correct top results", () -> {
            try (Corpus corp = new Corpus()) {
                corp.addDocsBatch(new String[][]{
                    {"handle", "request"},
                    {"validate", "user"},
                    {"handle", "response"},
                    {"parse", "json"},
                    {"validate", "email"}
                });
                corp.complete();

                FuncDescriptor query = corp.buildFunc("q.c", new String[]{"handle", "request"});
                FuncDescriptor[] corpus = {
                    corp.buildFunc("a.c", new String[]{"handle", "request"}),
                    corp.buildFunc("b.c", new String[]{"validate", "user"}),
                    corp.buildFunc("c.c", new String[]{"handle", "response"}),
                    corp.buildFunc("d.c", new String[]{"parse", "json"}),
                    corp.buildFunc("e.c", new String[]{"validate", "email"})
                };

                Corpus.FlatCorpus flat = Corpus.extractFlat(corpus);
                SearchResult[] batch = FastCodeEmbed.simpleRankBatch(
                    flat.allWeights, flat.allIndices, flat.tfidfLens,
                    flat.allRiVecs, flat.filePaths, flat.maxTokens,
                    query.getTfidfIndices(), query.getTfidfWeights(), query.getRiVec(), 3);

                // Verify: 3 results, descending order, best match first
                assertEquals(3, batch.length, "result count");
                assertTrue(batch[0].getScore() >= batch[1].getScore(), "descending 0>=1");
                assertTrue(batch[1].getScore() >= batch[2].getScore(), "descending 1>=2");

                // Best match should be index 0 (identical tokens + same file)
                assertEquals(0, batch[0].getIndex(), "best match index");
                assertTrue(batch[0].getScore() > 0.5f, "best match score > 0.5");
            }
        });

        test("simpleRankBatch respects topK", () -> {
            try (Corpus corp = new Corpus()) {
                String[][] docs = new String[5][];
                docs[0] = new String[]{"a", "b"};
                docs[1] = new String[]{"c", "d"};
                docs[2] = new String[]{"e", "f"};
                docs[3] = new String[]{"g", "h"};
                docs[4] = new String[]{"i", "j"};
                corp.addDocsBatch(docs);
                corp.complete();

                FuncDescriptor query = corp.buildFunc("q.c", new String[]{"a", "b"});
                FuncDescriptor[] corpus = new FuncDescriptor[5];
                for (int i = 0; i < 5; i++) {
                    corpus[i] = corp.buildFunc("c" + i + ".c", docs[i]);
                }

                Corpus.FlatCorpus flat = Corpus.extractFlat(corpus);
                SearchResult[] results = FastCodeEmbed.simpleRankBatch(
                    flat.allWeights, flat.allIndices, flat.tfidfLens,
                    flat.allRiVecs, flat.filePaths, flat.maxTokens,
                    query.getTfidfIndices(), query.getTfidfWeights(), query.getRiVec(), 2);
                assertTrue(results.length <= 2, "should return at most topK results");
            }
        });

        test("simpleRankBatch empty corpus", () -> {
            FuncDescriptor q = new FuncDescriptor("q.c");
            q.setTfidf(new int[]{}, new float[]{});
            q.setRiVec(new float[768]);
            SearchResult[] results = FastCodeEmbed.simpleRankBatch(
                new float[]{}, new int[]{}, new int[]{},
                new float[]{}, new String[]{}, 1,
                new int[]{}, new float[]{}, new float[768], 10);
            assertEquals(0, results.length, "empty corpus should return 0 results");
        });

        System.out.println("\nProximity:");
        test("same file → 1.10", () -> {
            float p = FastCodeEmbed.proximity("src/handler.c", "src/handler.c");
            assertEquals(1.10f, p, 0.001f, "proximity");
        });
        test("different files → 1.0", () -> {
            float p = FastCodeEmbed.proximity("src/handler.c", "lib/utils.c");
            assertEquals(1.0f, p, 0.001f, "proximity");
        });

        System.out.println("\n=========================");
        System.out.printf("%d/%d tests passed%n", passed, passed + failed);
        System.exit(failed > 0 ? 1 : 0);
    }
}
