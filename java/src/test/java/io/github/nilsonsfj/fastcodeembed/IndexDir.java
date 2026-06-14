package io.github.nilsonsfj.fastcodeembed;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

/**
 * Recursively tokenize and index a directory of source files.
 * Chunks files at 2048 bytes, tokenizes each chunk, builds a corpus.
 * Streams files in batches to avoid holding all chunks in memory.
 *
 * Usage: java ...IndexDir <directory> [chunk_size] [batch_size]
 */
public class IndexDir {
    static final int DEFAULT_CHUNK_SIZE = 2048;
    static final int DEFAULT_BATCH_SIZE = 50000;
    static final String[] EXTENSIONS = {
        ".c", ".h", ".cpp", ".hpp", ".java", ".py", ".rs", ".go", ".js", ".ts"
    };

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: IndexDir <directory> [chunk_size] [batch_size]");
            System.exit(1);
        }
        String rootDir = args[0];
        int chunkSize = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_CHUNK_SIZE;
        int batchSize = args.length > 2 ? Integer.parseInt(args[2]) : DEFAULT_BATCH_SIZE;

        long tTotal = System.nanoTime();

        System.out.println("fast-code-embed directory indexer (Java)");
        System.out.println("============================================");
        System.out.println("Directory: " + rootDir);
        System.out.println("Chunk size: " + chunkSize + " bytes");
        System.out.println("Batch size: " + batchSize + " chunks\n");

        /* ── 1. Walk directory and collect source file paths ──── */
        long t0 = System.nanoTime();
        List<Path> sourceFiles = new ArrayList<>();
        Files.walkFileTree(Path.of(rootDir), new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                String name = file.getFileName().toString();
                for (String ext : EXTENSIONS) {
                    if (name.endsWith(ext)) {
                        sourceFiles.add(file);
                        break;
                    }
                }
                return FileVisitResult.CONTINUE;
            }
            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) {
                return FileVisitResult.CONTINUE;
            }
        });
        double walkMs = (System.nanoTime() - t0) / 1e6;
        System.out.printf("  Walk directory:           %8.1f ms  (%d files)%n", walkMs, sourceFiles.size());

        if (sourceFiles.isEmpty()) {
            System.out.println("No source files found.");
            return;
        }

        /* ── 2. Stream files to C in batches ──────────────────── */
        FastCodeEmbed.init();
        try (Corpus corp = new Corpus()) {
            long tRead = System.nanoTime();
            int totalChunks = 0;
            int filesProcessed = 0;
            int totalDocsAdded = 0;
            List<String> batch = new ArrayList<>(batchSize);

            for (Path filePath : sourceFiles) {
                try {
                    byte[] content = Files.readAllBytes(filePath);
                    filesProcessed++;

                    for (int offset = 0; offset < content.length; offset += chunkSize) {
                        int len = Math.min(chunkSize, content.length - offset);
                        batch.add(new String(content, offset, len));
                        totalChunks++;

                        if (batch.size() >= batchSize) {
                            String[] batchArray = batch.toArray(new String[0]);
                            corp.addDocsTokenized(batchArray);
                            totalDocsAdded += batchArray.length;
                            batch.clear();
                        }
                    }
                } catch (IOException e) {
                    // Skip unreadable files
                }
            }

            /* Flush remaining batch */
            if (!batch.isEmpty()) {
                String[] batchArray = batch.toArray(new String[0]);
                corp.addDocsTokenized(batchArray);
                totalDocsAdded += batchArray.length;
                batch.clear();
            }
            double readMs = (System.nanoTime() - tRead) / 1e6;
            System.out.printf("  Read + tokenize + add:   %8.1f ms  (%d chunks from %d files)%n",
                    readMs, totalChunks, filesProcessed);

            /* ── 3. Finalize corpus ─────────────────────────── */
            t0 = System.nanoTime();
            corp.complete();
            double buildMs = (System.nanoTime() - t0) / 1e6;

            /* ── 4. Stats ─────────────────────────────────────── */
            double totalMs = (System.nanoTime() - tTotal) / 1e6;
            System.out.printf("  Corpus finalize:          %8.1f ms%n", buildMs);
            System.out.println("\n  ── Summary ──────────────────────────────────");
            System.out.printf("  Total chunks:    %d%n", totalChunks);
            System.out.printf("  Documents added: %d%n", totalDocsAdded);
            System.out.printf("  Vocabulary:      %d tokens%n", corp.getTokenCount());
            System.out.printf("  Documents:       %d%n", corp.getDocCount());
            System.out.printf("  Total time:      %.1f ms%n", totalMs);
            System.out.printf("  Throughput:      %.0f chunks/sec%n", totalChunks / (totalMs / 1000.0));
        }
    }
}
