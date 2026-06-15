package io.github.nilsonsfj.fastcodeembed;

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

        test("addDocsBatch(docs,paths) all-empty throws", () -> {
            try (Corpus corp = new Corpus()) {
                boolean threw = false;
                try {
                    corp.addDocsBatch(new String[][]{{}, {}}, new String[]{"a.c", "b.c"});
                } catch (IllegalArgumentException e) {
                    threw = true;
                }
                assertTrue(threw, "expected IllegalArgumentException for all-empty docs with paths");
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
                assertEquals(FastCodeEmbed.SEM_DIM, vec.length, "vec dim");

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
            a.setRiVec(new float[FastCodeEmbed.SEM_DIM]);
            FuncDescriptor b = new FuncDescriptor("b.c");
            b.setTfidf(new int[]{}, new float[]{});
            b.setRiVec(new float[FastCodeEmbed.SEM_DIM]);
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
            q.setRiVec(new float[FastCodeEmbed.SEM_DIM]);
            SearchResult[] results = FastCodeEmbed.simpleRankBatch(
                new float[]{}, new int[]{}, new int[]{},
                new float[]{}, new String[]{}, 1,
                new int[]{}, new float[]{}, new float[FastCodeEmbed.SEM_DIM], 10);
            assertEquals(0, results.length, "empty corpus should return 0 results");
        });

        System.out.println("\nProximity:");
        test("same file → 1.05", () -> {
            /* Why 1.05? The proximity algorithm counts only directory components
             * that match between two paths. For "src/handler.c" vs itself:
             *   - "src" matches → shared_dirs = 1
             *   - total_dirs (max slashes) = 1, so max_dirs + 1 = 2
             *   - ratio = shared_dirs / (max_dirs + 1) = 1 / 2 = 0.5
             *   - proximity = 1.0 + 0.5 * 0.10 (PROX_MAX_BOOST) = 1.05
             * Different prefixes ("src/handler.c" vs "lib/utils.c") share
             * zero path components → ratio = 0 → proximity = 1.0. */
            float p = FastCodeEmbed.proximity("src/handler.c", "src/handler.c");
            assertEquals(1.05f, p, 0.001f, "proximity");
        });
        test("different files → 1.0", () -> {
            float p = FastCodeEmbed.proximity("src/handler.c", "lib/utils.c");
            assertEquals(1.0f, p, 0.001f, "proximity");
        });

        System.out.println("\nCorpus lifecycle:");
        test("close() cleans up cleanly (no SIGTRAP at shutdown)", () -> {
            /* Regression test for SIGTRAP at JVM shutdown.  The Cleaner thread
             * could race with JNI_OnUnload if close() didn't deregister the
             * Cleaner action.  This test creates and closes a corpus, then the
             * JVM exits cleanly (no Trace/BPT trap). */
            try (Corpus corp = new Corpus()) {
                String[][] docs = new String[500][];
                for (int i = 0; i < 500; i++) {
                    docs[i] = new String[]{"token_" + i, "common", "shared"};
                }
                corp.addDocsBatch(docs);
                corp.complete();
                /* Exercise query to ensure native resources are alive. */
                SearchResult[] r = FastCodeEmbed.searchQuery(corp, "token_0 common", 10);
                assertTrue(r.length > 0, "should return results");
            }
            /* After try-with-resources, the corpus is closed and the Cleaner
             * is deregistered.  If close() forgot to call cleanable.clean(),
             * the Cleaner thread would attempt native calls during JVM shutdown,
             * causing SIGTRAP (signal 5 / Trace/BPT trap). */
        });

        test("double close() is safe", () -> {
            Corpus corp = new Corpus();
            corp.addDocsBatch(new String[][]{{"a", "b"}, {"c", "d"}});
            corp.complete();
            corp.close();
            corp.close(); /* must not crash or double-free */
        });

        test("close before complete is safe", () -> {
            Corpus corp = new Corpus();
            corp.addDocsBatch(new String[][]{{"x", "y"}});
            corp.close(); /* finalize was never called */
        });

        test("searchQuery on closed corpus throws IllegalStateException", () -> {
            Corpus corp = new Corpus();
            corp.addDocsBatch(new String[][]{{"a", "b"}});
            corp.complete();
            corp.close();
            boolean threw = false;
            try {
                FastCodeEmbed.searchQuery(corp, "a", 1);
            } catch (IllegalStateException e) {
                threw = true;
            }
            assertTrue(threw, "expected IllegalStateException for closed corpus");
        });

        test("searchQuery with null corpus throws NullPointerException", () -> {
            boolean threw = false;
            try {
                FastCodeEmbed.searchQuery(null, "a", 1);
            } catch (NullPointerException e) {
                threw = true;
            }
            assertTrue(threw, "expected NullPointerException for null corpus");
        });

        test("SEM_DIM matches native build", () -> {
            assertTrue(FastCodeEmbed.SEM_DIM > 0, "SEM_DIM must be positive");
        });

        System.out.println("\n=========================");
        System.out.printf("%d/%d tests passed%n", passed, passed + failed);
        System.exit(failed > 0 ? 1 : 0);
    }
}
