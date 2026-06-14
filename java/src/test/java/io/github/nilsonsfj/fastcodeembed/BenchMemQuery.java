package io.github.nilsonsfj.fastcodeembed;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Java equivalent of bench_mem_query.c — index directory, measure memory, benchmark queries.
 *
 * Build + run: cd java && ./build.sh memquery <directory> [chunk_size]
 */
public class BenchMemQuery {

    static final int DEFAULT_CHUNK_SIZE = 5000;
    static final int BATCH_SIZE = 50000;
    static final int MAX_TOKENS_PER_CHUNK = 256;

    static final Set<String> INCLUDE_EXTS = Set.of(
        ".c", ".h", ".cpp", ".hpp", ".java", ".py", ".rs", ".go", ".js", ".ts"
    );

    static void walkDir(Path dir, List<Path> out) throws IOException {
        if (!Files.isDirectory(dir)) return;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                String name = entry.getFileName().toString();
                if (name.startsWith(".")) continue;
                if (Files.isSymbolicLink(entry)) continue;
                if (Files.isDirectory(entry)) {
                    walkDir(entry, out);
                } else if (Files.isRegularFile(entry)) {
                    String ext = "";
                    int dot = name.lastIndexOf('.');
                    if (dot >= 0) ext = name.substring(dot);
                    if (INCLUDE_EXTS.contains(ext)) {
                        out.add(entry);
                    }
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: BenchMemQuery <directory> [chunk_size]");
            System.exit(1);
        }

        String rootDir = args[0];
        int chunkSize = DEFAULT_CHUNK_SIZE;
        if (args.length >= 2) {
            try { chunkSize = Integer.parseInt(args[1]); } catch (NumberFormatException ignored) {}
            if (chunkSize <= 0) chunkSize = DEFAULT_CHUNK_SIZE;
        }

        long tTotal = System.nanoTime();

        System.out.println("fast-code-embed memory + query benchmark");
        System.out.println("================================================");
        System.out.printf("Directory: %s%n", rootDir);
        System.out.printf("Chunk size: %d bytes%n%n", chunkSize);

        /* ── 1. Walk directory ────────────────────────────────────── */
        long t0 = System.nanoTime();
        List<Path> files = new ArrayList<>();
        walkDir(Path.of(rootDir), files);
        double walkMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  Walk directory:           %8.1f ms  (%d files)%n", walkMs, files.size());

        if (files.isEmpty()) {
            System.out.println("No source files found.");
            return;
        }

        /* ── 2. Build corpus ─────────────────────────────────────── */
        t0 = System.nanoTime();
        int filesProcessed = 0;
        /* Collect file paths only — all reading/chunking/tokenizing happens in C. */
        String[] filePaths = new String[files.size()];
        for (int i = 0; i < files.size(); i++) filePaths[i] = files.get(i).toString();
        filesProcessed = files.size();

        /* Single JNI call: read + chunk + tokenize + add to corpus. */
        Corpus corp = new Corpus();
        int totalChunks = corp.addFiles(filePaths, chunkSize, 256);

        double buildMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  Read + chunk + tokenize:  %8.1f ms  (%d chunks from %d files)%n",
                           buildMs, totalChunks, filesProcessed);

        /* ── 3. Finalize corpus ─────────────────────────────────────── */
        String[] docPaths = corp.getDocPaths();
        corp.clearDocPaths(); /* #3: free path tracking memory */

        long rssBefore = FastCodeEmbed.getCurrentRssBytes();
        t0 = System.nanoTime();
        corp.complete();
        double finalizeMs = (System.nanoTime() - t0) / 1e6;

        /* #4: hint GC to reclaim intermediate token strings */
        System.gc();
        Thread.sleep(100);
        System.gc();

        long rssAfter = FastCodeEmbed.getCurrentRssBytes();
        long peakRss = FastCodeEmbed.getPeakRssBytes();

        System.out.printf("  Corpus finalize:          %8.1f ms%n", finalizeMs);
        System.out.println();
        System.out.println("  -- Memory -----------------------------------------");
        System.out.printf("  RSS before finalize:      %8.1f GB%n", rssBefore / 1073741824.0);
        System.out.printf("  RSS after finalize:       %8.1f GB%n", rssAfter / 1073741824.0);
        System.out.printf("  Peak RSS:                 %8.1f GB%n", peakRss / 1073741824.0);

        int vocab = corp.getTokenCount();
        int ndocs = corp.getDocCount();
        System.out.println();
        System.out.println("  -- Corpus -----------------------------------------");
        System.out.printf("  Vocabulary:      %d tokens%n", vocab);
        System.out.printf("  Documents:       %d%n", ndocs);

        /* ── 4. Query benchmarks ──────────────────────────────────── */

        System.out.println();
        System.out.println("  -- Query Benchmarks -------------------------------");

        String[] queries = {
            "gpu display drivers",
            "user mode scheduling",
            "pcie ethernet code",
            "memory allocation pages",
            "file system inode",
        };
        int[] topKs = {10, 15};

        for (int topK : topKs) {
            System.out.printf("%n  top_k = %d%n", topK);
            double totalFastMs = 0, totalTfidfMs = 0, totalBruteMs = 0;

            for (String qstr : queries) {
                /* Warm up */
                SearchResult[] fastResults = FastCodeEmbed.searchQuery(corp, qstr, topK);
                SearchResult[] tfidfResults = FastCodeEmbed.searchQueryTfidf(corp, qstr, topK);
                SearchResult[] bruteResults = FastCodeEmbed.searchQueryBruteforce(corp, qstr, topK);

                int ncand = FastCodeEmbed.searchCandidateCount(corp, qstr);

                int iters = 20;

                /* Benchmark fast path */
                long t1 = System.nanoTime();
                for (int i = 0; i < iters; i++)
                    fastResults = FastCodeEmbed.searchQuery(corp, qstr, topK);
                double fastMs = (System.nanoTime() - t1) / 1e6 / iters;
                totalFastMs += fastMs;

                /* Benchmark TF-IDF hybrid */
                t1 = System.nanoTime();
                for (int i = 0; i < iters; i++)
                    tfidfResults = FastCodeEmbed.searchQueryTfidf(corp, qstr, topK);
                double tfidfMs = (System.nanoTime() - t1) / 1e6 / iters;
                totalTfidfMs += tfidfMs;

                /* Benchmark brute-force */
                t1 = System.nanoTime();
                for (int i = 0; i < iters; i++)
                    bruteResults = FastCodeEmbed.searchQueryBruteforce(corp, qstr, topK);
                double bruteMs = (System.nanoTime() - t1) / 1e6 / iters;
                totalBruteMs += bruteMs;

                /* Count overlaps with brute-force */
                int overlapFast = 0, overlapTfidf = 0;
                for (int i = 0; i < fastResults.length && i < topK; i++) {
                    for (int j = 0; j < bruteResults.length; j++) {
                        if (fastResults[i].getIndex() == bruteResults[j].getIndex()) { overlapFast++; break; }
                    }
                }
                for (int i = 0; i < tfidfResults.length && i < topK; i++) {
                    for (int j = 0; j < bruteResults.length; j++) {
                        if (tfidfResults[i].getIndex() == bruteResults[j].getIndex()) { overlapTfidf++; break; }
                    }
                }

                System.out.printf("%n    %-30s  fast=%5.0fms tfidf=%5.0fms brute=%5.0fms  cands=%d%n",
                        qstr, fastMs, tfidfMs, bruteMs, ncand);
                System.out.printf("      overlap with brute: fast=%d/%d  tfidf=%d/%d%n",
                        overlapFast, topK, overlapTfidf, topK);

                /* Print side-by-side results (top 5) */
                int show = Math.min(topK, 5);
                System.out.printf("      %-4s %-28s %-4s %-28s %-4s %-28s%n",
                        "Fast", "", "TFIDF", "", "Brute", "");
                for (int r = 0; r < show; r++) {
                    String fName = "?", tName = "?", bName = "?";
                    float fScore = 0, tScore = 0, bScore = 0;

                    if (r < fastResults.length) {
                        int idx = fastResults[r].getIndex();
                        String p = (idx < docPaths.length) ? docPaths[idx] : "?";
                        int slash = p.lastIndexOf('/');
                        fName = (slash >= 0) ? p.substring(slash + 1) : p;
                        fScore = fastResults[r].getScore();
                    }
                    if (r < tfidfResults.length) {
                        int idx = tfidfResults[r].getIndex();
                        String p = (idx < docPaths.length) ? docPaths[idx] : "?";
                        int slash = p.lastIndexOf('/');
                        tName = (slash >= 0) ? p.substring(slash + 1) : p;
                        tScore = tfidfResults[r].getScore();
                    }
                    if (r < bruteResults.length) {
                        int idx = bruteResults[r].getIndex();
                        String p = (idx < docPaths.length) ? docPaths[idx] : "?";
                        int slash = p.lastIndexOf('/');
                        bName = (slash >= 0) ? p.substring(slash + 1) : p;
                        bScore = bruteResults[r].getScore();
                    }

                    boolean matchBf = (r < fastResults.length && r < bruteResults.length &&
                                       fastResults[r].getIndex() == bruteResults[r].getIndex());
                    boolean matchTf = (r < tfidfResults.length && r < bruteResults.length &&
                                       tfidfResults[r].getIndex() == bruteResults[r].getIndex());

                    System.out.printf("      [%d]%.3f %-22s [%d]%.3f %-22s [%d]%.3f %-22s%s%s%n",
                            r, fScore, fName, r, tScore, tName, r, bScore, bName,
                            matchBf ? " <<" : "", matchTf ? " <<" : "");
                }
            }
            System.out.printf("%n    Average:  fast=%5.0fms  tfidf=%5.0fms  brute=%5.0fms%n",
                    totalFastMs / queries.length, totalTfidfMs / queries.length, totalBruteMs / queries.length);
        }

        /* ── Summary ─────────────────────────────────────────────── */
        double totalMs = (System.nanoTime() - tTotal) / 1e6;
        System.out.println();
        System.out.println("  -- Final Summary -----------------------------------");
        System.out.printf("  Total time:      %.1f ms%n", totalMs);
        System.out.printf("  Peak RSS:        %.1f GB%n", peakRss / 1073741824.0);
        System.out.printf("  Post-build RSS:  %.1f GB%n", rssAfter / 1073741824.0);

        corp.close();
    }
}
