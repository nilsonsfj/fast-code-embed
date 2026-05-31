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

        try {
            java.nio.file.Path tmpDir = java.nio.file.Files.createTempDirectory("fast-code-embed-jni");
            tmpDir.toFile().deleteOnExit();
            java.nio.file.Path tmpLib = tmpDir.resolve(libName);
            tmpLib.toFile().deleteOnExit();

            try (java.io.InputStream in = NativeLibrary.class.getResourceAsStream("/native/" + libName)) {
                if (in == null) {
                    throw new UnsatisfiedLinkError("Native library not found in JAR: /native/" + libName);
                }
                java.nio.file.Files.copy(in, tmpLib, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            System.load(tmpLib.toAbsolutePath().toString());
            loaded = true;
        } catch (java.io.IOException ex) {
            throw new UnsatisfiedLinkError("Failed to extract native library: " + ex.getMessage());
        }
    }
}