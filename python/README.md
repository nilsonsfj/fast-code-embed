# fast-code-embed (Python)

Pythonic interface to the [fast-code-embed](https://github.com/nilsonsfj/fast-code-embed)
C library: algorithmic code embeddings using TF-IDF + pretrained
nomic-embed-code token vectors (with optional Reflective Random Indexing).

- **No GPU, no API keys, no model download** — the 30 MB vector table is compiled
  into the wheel.
- **Sub-millisecond scoring** — lookup-based, no transformer inference.

> Status: **sketch / not yet published.** This package mirrors the Java binding's
> surface. The wheel links the prebuilt C static library, so `make lib` must run
> before building (cibuildwheel handles this in CI).

## Install (once published)

```bash
pip install fast-code-embed
```

## Usage

```python
import fast_code_embed as fce

# Global config (maps to the C library's process-state toggles; set once).
fce.set_dim(768)              # 256 or 768
fce.set_idf_weighting(True)
fce.set_abbrev_expansion(False)

with fce.Corpus() as corpus:
    corpus.add_doc("parse_config", "def parse_config(path): ...")
    corpus.add_doc("retry_request", "def retry_request(url, attempts): ...")
    # or: corpus.add_files(["a.py", "b.py"])  # chunk + tokenize in C
    corpus.finalize()

    for hit in corpus.search("load configuration from disk", k=10):
        print(f"{hit.score:.3f}  {hit.label}")

    # Persist while the corpus is still open (same-build cache, not portable
    # across dim / byte-order).
    corpus.save("index.fce")

# Reload later into a fresh corpus.
with fce.Corpus.load("index.fce") as reloaded:
    hits = reloaded.search("retry with backoff", k=5)
```

### Search modes

```python
from fast_code_embed import Mode
corpus.search("retry with backoff", k=5, mode=Mode.BRUTEFORCE)  # exhaustive
corpus.search("retry with backoff", k=5, mode=Mode.FAST)        # inverted index
corpus.search("retry with backoff", k=5, mode=Mode.TFIDF)       # tf-idf + rerank
# Mode.DEFAULT == AUTO (fast path, brute fallback)
```

`Mode.FAST`/`TFIDF`/`AUTO` only retrieve documents sharing a literal token with
the query; use `Mode.BRUTEFORCE` when exhaustive recall matters.

## Building locally

```bash
cd ..             # repo root
make lib          # produces build/libfast_code_embed.a (with embedded blob)
cd python
pip install -e .  # compiles the cffi extension against the archive
```

## Building wheels

```bash
pip install cibuildwheel
cibuildwheel --output-dir wheelhouse python
```
