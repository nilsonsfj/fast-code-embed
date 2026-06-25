#!/usr/bin/env python3
"""Benchmark the fast-code-embed Python binding.

Measures tokenize throughput, corpus build/finalize time, and per-mode search
latency. Uses the Linux-kernel comparison corpus at `_cmp/chunks.jsonl` when
present (the same 80k-chunk set the C benchmarks use); otherwise falls back to a
small synthetic corpus so the script runs anywhere.

Build the extension first (from the repo root):
    make lib && (cd python && pip install -e .)
then:
    python python/bench.py            # auto-detect corpus
    python python/bench.py path.jsonl # explicit {id,path,text} JSONL corpus
"""
from __future__ import annotations

import json
import os
import resource
import statistics
import sys
import time

# Allow running from a source checkout without installing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_code_embed as fce  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CORPUS = os.path.join(ROOT, "_cmp", "chunks.jsonl")
DEFAULT_QUERIES = os.path.join(ROOT, "_cmp", "queries.txt")

FALLBACK_QUERIES = [
    "load configuration from disk", "retry http request with backoff",
    "hash table insert and resize", "parse command line arguments",
    "allocate aligned memory buffer",
]


def rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def load_corpus(path):
    """Return (docs, queries). docs is a list of (label, text)."""
    if path and os.path.exists(path):
        docs = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                docs.append((d.get("path", d.get("id", "")), d["text"]))
        queries = FALLBACK_QUERIES
        if os.path.exists(DEFAULT_QUERIES):
            queries = [q.strip() for q in open(DEFAULT_QUERIES) if q.strip()]
        return docs, queries, os.path.basename(path)

    # Synthetic fallback: enough docs to make search timing meaningful.
    print("no corpus file found — using a synthetic fallback corpus")
    verbs = ["parse", "load", "encode", "retry", "allocate", "hash", "flush",
             "resize", "lookup", "schedule", "commit", "map", "lock", "scan"]
    nouns = ["config", "buffer", "request", "table", "page", "socket", "inode",
             "transaction", "cipher", "mutex", "queue", "cache", "driver"]
    docs = []
    for i in range(20000):
        v = verbs[i % len(verbs)]
        n = nouns[(i // len(verbs)) % len(nouns)]
        docs.append((f"{v}_{n}_{i}",
                     f"def {v}_{n}(x): {v} the {n} and return the {n} result"))
    return docs, FALLBACK_QUERIES, "synthetic"


def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS
    docs, queries, name = load_corpus(corpus_path)

    print(f"python {sys.version.split()[0]}  dim={fce.active_dim()}  "
          f"idf={fce.idf_weighting()}  abbrev={fce.abbrev_expansion()}")
    print(f"corpus: {name}  ({len(docs):,} docs)  rss {rss_mb():.0f} MB")

    # ── tokenize throughput ────────────────────────────────────────────
    sample = [label for label, _ in docs[:20000]]
    t = time.perf_counter()
    ntok = sum(len(fce.tokenize(s)) for s in sample)
    dt = time.perf_counter() - t
    print(f"\ntokenize: {len(sample):,} names -> {ntok:,} tokens in {dt:.2f}s  "
          f"({len(sample) / dt:,.0f} names/s)")

    # ── build ──────────────────────────────────────────────────────────
    c = fce.Corpus()
    t = time.perf_counter()
    for label, text in docs:
        c.add_doc(label, text)
    build = time.perf_counter() - t
    t = time.perf_counter()
    c.finalize()
    fin = time.perf_counter() - t
    print(f"\nbuild: add_doc x{len(docs):,} in {build:.2f}s "
          f"({len(docs) / build:,.0f} docs/s)")
    print(f"       finalize in {fin:.2f}s   doc_count={len(c):,}   "
          f"rss {rss_mb():.0f} MB")

    # ── search latency per mode ────────────────────────────────────────
    reps = max(1, 560 // len(queries))
    for mode in (fce.Mode.FAST, fce.Mode.TFIDF, fce.Mode.DEFAULT,
                 fce.Mode.BRUTEFORCE):
        for q in queries:  # warmup
            c.search(q, k=10, mode=mode)
        lat = []
        t0 = time.perf_counter()
        for _ in range(reps):
            for q in queries:
                t = time.perf_counter()
                c.search(q, k=10, mode=mode)
                lat.append((time.perf_counter() - t) * 1000)
        wall = time.perf_counter() - t0
        print(f"\n{mode.name:<11} {len(lat):>4} searches  "
              f"mean {statistics.mean(lat):6.2f}ms  p50 {pct(lat, 50):6.2f}ms  "
              f"p95 {pct(lat, 95):6.2f}ms  max {max(lat):6.2f}ms  "
              f"=> {len(lat) / wall:6.0f} qps")

    print(f"\nsample results (BRUTEFORCE, {queries[0]!r}):")
    for h in c.search(queries[0], k=5, mode=fce.Mode.BRUTEFORCE):
        print(f"  {h.score:.3f}  {h.label}")
    c.close()


if __name__ == "__main__":
    main()
