# fast-code-embed — Java JNI Binding

**Version: 0.0.4**

Java binding for the [fast-code-embed](../) C library. Provides
batch corpus operations and the zero-config simple scoring API via JNI.

## Requirements

- Java 11+ (tested in CI with Temurin 11; also verified manually with OpenJDK 25)
- GCC/Clang (to compile the JNI native code)
- The C library must be built first (`make` from the project root)

## Quick Start

**Maven:**
```xml
<dependency>
    <groupId>io.github.nilsonsfj</groupId>
    <artifactId>fast-code-embed</artifactId>
    <version>0.0.3</version>
</dependency>
```

**From source:**

```bash
# From the project root:
make                    # build C library
cd java && ./build.sh   # compile JNI + Java, run tests
```

## Usage

```java
import io.github.nilsonsfj.fastcodeembed.*;

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
| `tokenizeBatch(names)` | `String[][]` | Batch-tokenize multiple names (one JNI call) |
| `proximity(a, b)` | `float` | Module proximity [1.0, 1.10] |
| `simpleScore(a, b)` | `float` | TF-IDF + RI score, normalized to [0.0, 1.0] |
| `simpleRank(query, corpus, topK)` | `SearchResult[]` | Rank corpus by score descending |
| `simpleSearch(query, corpus, topK, minScore)` | `SearchResult[]` | Rank with minimum threshold |
| `simpleRankBatch(...)` | `SearchResult[]` | Rank using pre-extracted flat arrays (fast path) |
| `searchQuery(corpus, query, topK)` | `SearchResult[]` | Tokenize query, retrieve via inverted index, rerank with RI |
| `searchQueryTfidf(corpus, query, topK)` | `SearchResult[]` | TF-IDF candidate retrieval + RI rerank |
| `searchQueryBruteforce(corpus, query, topK)` | `SearchResult[]` | Brute-force scan all documents |
| `searchCandidateCount(corpus, query)` | `int` | Number of inverted index candidates for a query |
| `getPeakRssBytes()` | `long` | Peak RSS via getrusage() |
| `getCurrentRssBytes()` | `long` | Current RSS (macOS: task_info, Linux: /proc/self/status) |

### `Corpus`

| Method | Returns | Description |
|--------|---------|-------------|
| `new Corpus()` | | Create empty corpus |
| `addDoc(tokens)` | `void` | Add one document's tokens |
| `addDoc(tokens, filePath)` | `void` | Add one document's tokens with file path |
| `addDocsBatch(docs)` | `void` | Batch-add documents (more efficient) |
| `addDocsBatch(docs, paths)` | `void` | Batch-add documents with file paths |
| `addDocsTokenized(names)` | `void` | Batch-add by raw names (tokenization in C) |
| `addDocsTokenized(names, paths)` | `void` | Batch-add by raw names with file paths |
| `addFiles(paths, chunkSize, maxTokens)` | `int` | Read source files, chunk at `}` boundaries, tokenize in C |
| `complete()` | `void` | Compute IDF + enriched vectors (required before querying) |
| `getIdf(token)` | `float` | IDF weight for a token |
| `getRiVec(token)` | `float[SEM_DIM]` | Enriched RI vector (null if unknown) |
| `getDocCount()` | `int` | Number of documents |
| `getTokenCount()` | `int` | Vocabulary size |
| `getDocPath(index)` | `String` | File path for a document by corpus index |
| `getDocPaths()` | `String[]` | All tracked file paths (one per document) |
| `clearDocPaths()` | `void` | Free memory held by tracked paths |
| `buildFunc(path, tokens)` | `FuncDescriptor` | Convenience: build a scored func from tokens (deprecated — use `extractFlat`) |
| `extractFlat(funcs)` | `FlatCorpus` | Extract flat arrays for `simpleRankBatch` |
| `close()` | `void` | Free native resources |

### `FuncDescriptor`

| Method | Returns | Description |
|--------|---------|-------------|
| `new FuncDescriptor(filePath)` | | Create with file path |
| `setTfidf(indices, weights)` | `void` | Set sparse TF-IDF vector |
| `setRiVec(vec)` | `void` | Set `FastCodeEmbed.SEM_DIM`-dimensional RI vector |
| `getFilePath()` | `String` | Source file path |
| `getTfidfIndices()` | `int[]` | Token indices |
| `getTfidfWeights()` | `float[]` | IDF weights |
| `getRiVec()` | `float[]` | RI vector |

### `SearchResult`

| Method | Returns | Description |
|--------|---------|-------------|
| `getIndex()` | `int` | Index into the searched corpus |
| `getScore()` | `float` | Similarity score [0.0, 1.0] |

### `FlatCorpus`

Returned by `Corpus.extractFlat()`. Reusable across multiple queries.

| Field | Type | Description |
|-------|------|-------------|
| `allWeights` | `float[]` | Flat IDF weights: [func × maxTokens + token] |
| `allIndices` | `int[]` | Flat token indices: [func × maxTokens + token] |
| `tfidfLens` | `int[]` | Per-function token count |
| `allRiVecs` | `float[]` | Flat RI vectors: [func × SEM_DIM + dim] |
| `filePaths` | `String[]` | Per-function file paths |
| `maxTokens` | `int` | Stride for flat arrays |
| `size()` | `int` | Number of functions |

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
    ├── FastCodeEmbedTest.java            21 tests (no framework)
    ├── BenchJava.java                    Micro-benchmark
    ├── BenchMemQuery.java                Memory + query benchmark
    └── IndexDir.java                     Directory indexer
```

## Notes

- `FastCodeEmbed.SEM_DIM` is the runtime embedding dimension (usually 768, or
  256 if the native library was built with `-DFCE_SEM_DIM_256`). All RI vectors
  passed to the JNI API must match this dimension.

- `FastCodeEmbed.init()` eagerly loads the pretrained token lookup map. If the
  JVM is under memory pressure, the map may be only partially populated; any
  token that fails to insert falls back to a deterministic sparse random vector.
  The library logs a warning in that case, but embedding quality may degrade
  slightly.

- Sparse vector storage (configured from the C side with
  `fce_sem_corpus_set_sparse`) saves memory but changes ranking: the query
  magnitude is computed over all dimensions while the document magnitude is
  computed only over the retained top-N dimensions per vector, producing a
  non-monotone cosine that can reorder results relative to dense mode. Use
  dense mode for faithful rank order; sparse mode is a memory/speed trade-off.

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