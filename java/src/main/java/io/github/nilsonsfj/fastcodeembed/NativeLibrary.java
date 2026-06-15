package io.github.nilsonsfj.fastcodeembed;

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
        String osKey;
        String libName;
        if (os.contains("mac") || os.contains("darwin")) {
            osKey = "darwin";
            libName = "libfast_code_embed_jni.dylib";
        } else if (os.contains("linux")) {
            osKey = "linux";
            libName = "libfast_code_embed_jni.so";
        } else {
            osKey = "windows";
            libName = "fast_code_embed_jni.dll";
        }
        /* Normalize JVM arch names: amd64 (x86-64 Linux/Windows) → x86_64 */
        String rawArch = System.getProperty("os.arch", "").toLowerCase();
        String archKey = rawArch.equals("amd64") ? "x86_64" : rawArch;

        /* C-2: sweep stale dirs BEFORE creating tmpDir.
         * The previous ordering swept AFTER createTempDirectory, which deleted
         * the freshly-created "fce-jni-*" dir (it matched the prefix) and then
         * caused Files.copy to throw NoSuchFileException — making the JAR
         * fallback path unconditionally fail on every deployment.
         * S-4: sweep both legacy prefix and new prefix
         * to clean up dirs left by prior JVM crashes (deleteOnExit doesn't fire
         * on abort). Age-gate to dirs older than 1 hour to avoid racing against
         * other JVM instances on the same host that are mid-extraction. */
        String tmpRoot = System.getProperty("java.io.tmpdir");
        sweepStaleTempDirs(tmpRoot, "fast-code-embed-jni-");
        sweepStaleTempDirs(tmpRoot, "fce-jni-");

        /* M3: use Files.createTempDirectory for an
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

        try {
            java.nio.file.Path tmpLib = tmpDir.resolve(libName);
            tmpLib.toFile().deleteOnExit();

            String resourcePath = "/native/" + osKey + "-" + archKey + "/" + libName;
            try (java.io.InputStream in = NativeLibrary.class.getResourceAsStream(resourcePath)) {
                if (in == null) {
                    throw new UnsatisfiedLinkError("Native library not found in JAR: " + resourcePath);
                }
                java.nio.file.Files.copy(in, tmpLib, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            try {
                System.load(tmpLib.toAbsolutePath().toString());
            } catch (UnsatisfiedLinkError loadErr) {
                /* M-3: System.load can fail AFTER Files.copy
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

    private static final long SWEEP_AGE_MS = 60L * 60 * 1000; /* 1 hour */

    private static void sweepStaleTempDirs(String root, String prefix) {
        long cutoff = System.currentTimeMillis() - SWEEP_AGE_MS;
        try (java.util.stream.Stream<java.nio.file.Path> stream =
                 java.nio.file.Files.list(java.nio.file.Paths.get(root))) {
            stream.filter(p -> {
                        java.nio.file.Path fn = p.getFileName();
                        return fn != null && fn.toString().startsWith(prefix);
                    })
                  .filter(p -> java.nio.file.Files.isDirectory(p))
                  /* C-2: only delete dirs older than
                   * SWEEP_AGE_MS to avoid racing a concurrent JVM that just
                   * created its extraction dir but hasn't called System.load
                   * yet. Dirs that young are almost certainly live. */
                  .filter(p -> {
                      try {
                          java.nio.file.attribute.BasicFileAttributes attrs =
                              java.nio.file.Files.readAttributes(
                                  p, java.nio.file.attribute.BasicFileAttributes.class);
                          return attrs.lastModifiedTime().toMillis() < cutoff;
                      } catch (java.io.IOException ignored) {
                          return false;
                      }
                  })
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