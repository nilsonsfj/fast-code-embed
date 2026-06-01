package com.github.nilsonsfj.fastcodeembed;

/**
 * Loads the native fast-code-embed shared library.
 *
 * <p>Tries {@code System.loadLibrary("fast_code_embed_jni")} first, then
 * falls back to extracting {@code libfast_code_embed_jni.dylib} (macOS),
 * {@code libfast_code_embed_jni.so} (Linux), or {@code fast_code_embed_jni.dll}
 * (Windows) from the JAR's {@code /native/} resource directory.</p>
 *
 * <p>Thread-safe. Calling {@link #load()} multiple times is a no-op after
 * the first successful load.</p>
 *
 * @since 0.0.1
 */
public final class NativeLibrary {
    private static boolean loaded = false;

    private NativeLibrary() {}

    /**
     * Load the native library. Safe to call multiple times.
     *
     * @throws UnsatisfiedLinkError if the library cannot be found or loaded
     */
    public static synchronized void load() {
        if (loaded) return;
        try {
            System.loadLibrary("fast_code_embed_jni");
            loaded = true;
        } catch (UnsatisfiedLinkError e) {
            loadFromJar();
        }
    }

    private static void loadFromJar() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String libName;
        if (os.contains("mac") || os.contains("darwin")) {
            libName = "libfast_code_embed_jni.dylib";
        } else if (os.contains("linux")) {
            libName = "libfast_code_embed_jni.so";
        } else {
            libName = "fast_code_embed_jni.dll";
        }

        /* M3 (review 0001-0001 §5): use Files.createTempDirectory for an
         * unpredictable directory name with 0700 permissions. The previous
         * PID-derived name was predictable, enabling a classic native-library
         * planting / symlink-race attack on shared-host /tmp. */
        java.nio.file.Path tmpDir;
        try {
            tmpDir = java.nio.file.Files.createTempDirectory("fce-jni-");
        } catch (java.io.IOException ex) {
            throw new UnsatisfiedLinkError("Failed to create temp directory: " + ex.getMessage());
        }
        tmpDir.toFile().deleteOnExit();

        /* Best-effort cleanup of legacy PID-based dirs from prior runs.
         * These use the old "fast-code-embed-jni-<pid>" naming convention. */
        String tmpRoot = System.getProperty("java.io.tmpdir");
        sweepStaleTempDirs(tmpRoot, "fast-code-embed-jni-");

        try {
            java.nio.file.Path tmpLib = tmpDir.resolve(libName);
            tmpLib.toFile().deleteOnExit();

            try (java.io.InputStream in = NativeLibrary.class.getResourceAsStream("/native/" + libName)) {
                if (in == null) {
                    throw new UnsatisfiedLinkError("Native library not found in JAR: /native/" + libName);
                }
                java.nio.file.Files.copy(in, tmpLib, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            try {
                System.load(tmpLib.toAbsolutePath().toString());
            } catch (UnsatisfiedLinkError loadErr) {
                /* M-3 (review 0002 §7.5): System.load can fail AFTER Files.copy
                 * succeeded (e.g. corrupt dylib, wrong architecture, missing
                 * dependency). deleteOnExit() doesn't fire on JVM abort or
                 * crash, so the extracted library would otherwise leak on
                 * disk. Best-effort delete and rethrow with the original
                 * cause so the caller still gets the real diagnostic. */
                try { java.nio.file.Files.deleteIfExists(tmpLib); } catch (java.io.IOException ignored) {}
                UnsatisfiedLinkError wrapped = new UnsatisfiedLinkError(
                        "Failed to load extracted native library at " + tmpLib
                                + ": " + loadErr.getMessage());
                wrapped.initCause(loadErr);
                throw wrapped;
            }
            loaded = true;
        } catch (java.io.IOException ex) {
            throw new UnsatisfiedLinkError("Failed to extract native library: " + ex.getMessage());
        }
    }

    private static void sweepStaleTempDirs(String root, String prefix) {
        try (java.util.stream.Stream<java.nio.file.Path> stream =
                 java.nio.file.Files.list(java.nio.file.Paths.get(root))) {
            stream.filter(p -> {
                        java.nio.file.Path fn = p.getFileName();
                        return fn != null && fn.toString().startsWith(prefix);
                    })
                  .filter(p -> java.nio.file.Files.isDirectory(p))
                  .forEach(p -> {
                      /* Best-effort recursive delete. We don't fail load() on
                       * cleanup errors — the worst case is some disk used. */
                      try {
                          try (java.util.stream.Stream<java.nio.file.Path> walk =
                                   java.nio.file.Files.walk(p)) {
                              walk.sorted(java.util.Comparator.reverseOrder())
                                  .forEach(child -> {
                                      try { java.nio.file.Files.deleteIfExists(child); }
                                      catch (java.io.IOException ignored) {}
                                  });
                          }
                      } catch (java.io.IOException ignored) {}
                  });
        } catch (java.io.IOException ignored) {}
    }
}