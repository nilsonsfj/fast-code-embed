# fast-code-embed — Java JNI Binding

**Version: 0.0.1**

Java binding for the [fast-code-embed](../) C library. Provides
batch corpus operations and the zero-config simple scoring API via JNI.

## Requirements

- Java 11+ (tested with OpenJDK 25)
- GCC/Clang (to compile the JNI native code)
- The C library must be built first (`make` from the project root)

## Quick Start

```bash
# From the project root:
make                    # build C library
cd java && ./build.sh   # compile JNI + Java, run tests
```

## Usage

```java
import com.github.nilsonsfj.fastcodeembed.*;

public class Example {
    public static void main(String[] args) {
        // 1. Initialize (loads native lib + pretrained token table)
        FastCodeEmbed.init();

        // 2. Build corpus from tokenized code metadata
        try (Corpus corp = new Corpus()) {
            corp.addDocsBatch(new String[][]{
                {"handle", "request", "parse"},
                {"validate", "user", "check"},
                {"handle", "response", "send"}
            });
            corp.complete();

            // 3. Build function descriptors
            FuncDescriptor a = corp.buildFunc(
                "src/handler.c", new String[]{"handle", "request"});
            FuncDescriptor b = corp.buildFunc(
                "src/auth.c", new String[]{"validate", "user"});

            // 4. Score — returns [0.0, 1.0]
            float score = FastCodeEmbed.simpleScore(a, b);

            // 5. Or rank a corpus against a query
            FuncDescriptor[] corpus = {a, b};
            SearchResult[] results = FastCodeEmbed.simpleRank(a, corpus, 10);

            for (SearchResult r : results) {
                System.out.printf("  %s → %.4f%n",
                    r.getIndex() == 0 ? "handler" : "auth", r.getScore());
            }
        }
    }
}
```

## API Reference

### `FastCodeEmbed`

| Method | Returns | Description |
|--------|---------|-------------|
| `init()` | `void` | Initialize native library. Call once before scoring. |
| `tokenize(name)` | `String[]` | Split identifier into tokens (camelCase, snake_case, etc.) |
| `proximity(a, b)` | `float` | Module proximity [1.0, 1.10] |
| `simpleScore(a, b)` | `float` | TF-IDF + RI score, normalized to [0.0, 1.0] |
| `simpleRank(query, corpus, topK)` | `SearchResult[]` | Rank corpus by score descending |
| `simpleSearch(query, corpus, topK, minScore)` | `SearchResult[]` | Rank with minimum threshold |

### `Corpus`

| Method | Returns | Description |
|--------|---------|-------------|
| `new Corpus()` | | Create empty corpus |
| `addDoc(tokens)` | `void` | Add one document's tokens |
| `addDocsBatch(docs)` | `void` | Batch-add documents (more efficient) |
| `complete()` | `void` | Compute IDF + enriched vectors (required before querying) |
| `getIdf(token)` | `float` | IDF weight for a token |
| `getRiVec(token)` | `float[768]` | Enriched RI vector (null if unknown) |
| `getDocCount()` | `int` | Number of documents |
| `getTokenCount()` | `int` | Vocabulary size |
| `buildFunc(path, tokens)` | `FuncDescriptor` | Convenience: build a scored func from tokens |
| `close()` | `void` | Free native resources |

### `FuncDescriptor`

| Method | Returns | Description |
|--------|---------|-------------|
| `new FuncDescriptor(filePath)` | | Create with file path |
| `setTfidf(indices, weights)` | `void` | Set sparse TF-IDF vector |
| `setRiVec(vec)` | `void` | Set 768-dimensional RI vector |
| `getFilePath()` | `String` | Source file path |
| `getTfidfIndices()` | `int[]` | Token indices |
| `getTfidfWeights()` | `float[]` | IDF weights |
| `getRiVec()` | `float[]` | RI vector |

### `SearchResult`

| Method | Returns | Description |
|--------|---------|-------------|
| `getIndex()` | `int` | Index into the searched corpus |
| `getScore()` | `float` | Similarity score [0.0, 1.0] |

## Project Structure

```
java/
├── build.sh                               Build + test script
├── README.md                              This file
├── src/main/java/.../fastcodeembed/
│   ├── FastCodeEmbed.java                Main API (static methods)
│   ├── Corpus.java                       Corpus builder (AutoCloseable)
│   ├── FuncDescriptor.java               Function semantic data
│   ├── SearchResult.java                 Ranked result
│   └── NativeLibrary.java                Native library loader
├── src/main/native/
│   └── fast_code_embed_jni.c             JNI bridge implementation
└── src/test/java/.../fastcodeembed/
    ├── FastCodeEmbedTest.java            17 tests (no framework)
    ├── BenchJava.java                    Micro-benchmark
    ├── BenchMemQuery.java                Memory + query benchmark
    └── IndexDir.java                     Directory indexer
```

## How It Works

The JNI layer (`fast_code_embed_jni.c`) marshals Java objects into C structs
on each call:

1. `FuncDescriptor` fields → temporary `fce_sem_func_t` on the stack
2. Calls the C function (`fce_sem_simple_score`, etc.)
3. Copies results back to Java arrays
4. Frees the temporary C allocations

The `Corpus` class holds a native pointer (`long handle`) to an opaque
`fce_sem_corpus_t`. It is freed when `close()` is called (or via
try-with-resources).

## License

MIT — same as the C library. nomic-embed-code vectors are Apache 2.0.