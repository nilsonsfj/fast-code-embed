package io.github.nilsonsfj.fastcodeembed;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Java equivalent of bench_mem_query.c — index directory, measure memory, benchmark queries.
 *
 * Build + run: cd java && ./build.sh memquery <directory> [chunk_size] [--save-load[=path]]
 *
 * With --save-load, the finalized corpus is written to a cache file and reloaded
 * via zero-copy mmap; save/load throughput is reported, query parity against the
 * in-memory corpus is verified (non-zero exit on mismatch), and the remaining
 * query benchmarks run on the mmap-loaded corpus.
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
            System.err.println("Usage: BenchMemQuery <directory> [chunk_size] [--save-load[=path]]");
            System.exit(1);
        }

        String rootDir = args[0];
        int chunkSize = DEFAULT_CHUNK_SIZE;
        boolean saveLoad = false;
        String saveLoadPath = null;
        for (int i = 1; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--save-load")) {
                saveLoad = true;
            } else if (a.startsWith("--save-load=")) {
                saveLoad = true;
                saveLoadPath = a.substring("--save-load=".length());
            } else {
                try {
                    int cs = Integer.parseInt(a);
                    if (cs > 0) chunkSize = cs;
                } catch (NumberFormatException ignored) {}
            }
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
        /* Free path-tracking memory before finalize — but only when not saving:
         * save() persists labels only while the internal path list still lines up
         * 1:1 with the documents, so keep it intact for the --save-load path. */
        if (!saveLoad) corp.clearDocPaths(); /* #3: free path tracking memory */

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

        /* ── 3b. Save / load (mmap cache) ─────────────────────────── */
        boolean parityFailed = false;
        if (saveLoad) {
            String cpath = saveLoadPath;
            if (cpath == null) {
                String dir = System.getenv("TMPDIR");
                if (dir == null || dir.isEmpty()) dir = "/tmp";
                if (!dir.endsWith("/")) dir = dir + "/";
                cpath = dir + "fce_bench_corpus.fce";
            }
            System.out.println();
            System.out.println("  -- Save / Load (mmap cache) -----------------------");
            System.out.printf("  Path: %s%n", cpath);

            long ts = System.nanoTime();
            corp.save(cpath);
            double saveMs = (System.nanoTime() - ts) / 1e6;
            double fsz = Files.size(Path.of(cpath));
            System.out.printf("  Save:                     %8.1f ms  (%.2f GB, %.0f MB/s)%n",
                    saveMs, fsz / 1073741824.0, fsz / 1e6 / (saveMs / 1000.0));

            ts = System.nanoTime();
            Corpus loaded = Corpus.load(cpath);
            double loadMs = (System.nanoTime() - ts) / 1e6;
            String[] loadedPaths = loaded.getDocPaths();
            System.out.printf("  Load:                     %8.1f ms  (%.0f MB/s, %.1fx faster than finalize)%n",
                    loadMs, fsz / 1e6 / (loadMs / 1000.0), finalizeMs / loadMs);
            System.out.printf("  Loaded: vocab=%d docs=%d labels=%d%n",
                    loaded.getTokenCount(), loaded.getDocCount(), loadedPaths.length);

            /* Confirm query results match the in-memory corpus exactly. */
            String[] pq = {"gpu display drivers", "memory allocation pages", "file system inode"};
            int identical = 0;
            for (String q : pq) {
                SearchResult[] a = FastCodeEmbed.searchQuery(corp, q, 10);
                SearchResult[] b = FastCodeEmbed.searchQuery(loaded, q, 10);
                boolean same = (a.length == b.length);
                for (int j = 0; same && j < a.length; j++) {
                    if (a[j].getIndex() != b[j].getIndex()
                        || Math.abs(a[j].getScore() - b[j].getScore()) > 1e-6f) same = false;
                }
                if (same) identical++;
            }
            System.out.printf("  Query parity vs in-memory corpus: %s (%d/%d identical)%n",
                    identical == pq.length ? "OK" : "MISMATCH", identical, pq.length);

            if (identical != pq.length) {
                /* A save/load that visibly fails parity must not have its later
                 * timings presented as if the loaded representation were
                 * trustworthy. Drop the loaded corpus, keep the in-memory one,
                 * and propagate a non-zero exit code. */
                parityFailed = true;
                loaded.close();
                System.out.println("  PARITY MISMATCH: discarding loaded corpus; benchmarks below "
                        + "run on the in-memory corpus; exit code will be non-zero.");
            } else {
                /* Continue the query benchmarks on the mmap-loaded corpus; its
                 * doc paths are restored from the cache's label table. */
                corp.close();
                corp = loaded;
                docPaths = loadedPaths;
                System.out.println("  (query benchmarks below run on the mmap-loaded corpus)");
            }
        }

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

        if (parityFailed) System.exit(1);
    }
}
