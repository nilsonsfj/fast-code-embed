# Code-Search Quality & Performance: fast-code-embed vs. potion-base-8M

A controlled comparison of **fast-code-embed (FCE)** — whose token vectors are
derived from [nomic-embed-code](https://huggingface.co/nomic-ai/nomic-embed-code)
— against **[minishlab/potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)**,
a general-purpose [model2vec](https://github.com/MinishLab/model2vec) static
embedding, on natural-language retrieval over Linux kernel source.

**Headline:** FCE delivers materially higher ranking quality (**nDCG@10 0.839
vs. 0.655**, **MRR 0.82 vs. 0.57**) while serving queries in **~1 ms** against an
80 K-chunk corpus. The two systems agree on only 0–4 of their top-10 results per
query, and FCE's advantage comes almost entirely from resolving polysemy that a
general text model gets wrong on code.

---

## 1. Methodology

The comparison is designed so that the only variable is the embedding/retrieval
model — both systems index the **exact same document units**.

### Corpus

- Source: a sparse checkout of `torvalds/linux` spanning 12 subsystems chosen to
  cover every query topic *and* provide realistic distractors: `drivers/gpu/drm/i915`,
  `drivers/pci`, `drivers/net/ethernet/intel`, `kernel/sched`, `kernel/irq`,
  `kernel/locking`, `mm`, `fs/ext4`, `fs/jbd2`, `net/ipv4`, `net/core`, `crypto`.
- 2,595 `.c`/`.h` files → **80,190 chunks** (split at brace boundaries, 60-line cap).
- Both systems consume the identical chunk list, so any quality difference is
  attributable to the model, not to chunking.

### Systems under test

| | fast-code-embed | potion-base-8M |
|---|---|---|
| Representation | nomic-embed-code-derived token vectors, **768-dim** | model2vec static embedding, **256-dim** |
| Document vector | enriched token aggregation (TF-IDF + Random-Indexing) | mean-pooled token embedding of raw chunk text |
| Query path | brute-force reference (exhaustive cosine) | dense cosine over all chunk vectors |
| Code-specialized | **Yes** (trained on code) | No (general English) |

Both systems were run at their strongest retrieval setting. potion was used as
intended off-the-shelf: embedding raw code as text.

### Queries

The 8 queries shipped in the FCE benchmark plus 6 added for subsystem coverage
(14 total):

```
gpu display drivers          spinlock mutex locking
user mode scheduling         tcp congestion control
pcie ethernet code           ext4 journaling commit transaction
memory allocation pages      rcu read lock grace period
file system inode            dma buffer mapping
network socket buffer        page fault handler
interrupt handler irq
crypto aes cipher
```

### Relevance judging

For each query the top-10 of both systems were **pooled** (~270 unique chunks)
and each `(query, chunk)` pair was graded on a 3-point scale by reading the
actual code (not the file path): `2` = highly relevant, `1` = partially relevant,
`0` = off-topic. Pooling means a chunk is judged once regardless of which system
surfaced it, removing per-system labeling bias. Metrics:

- **P@10** — fraction of top-10 with grade ≥ 1 (relevant or partial).
- **Strict P@10** — fraction with grade = 2 (highly relevant only).
- **nDCG@10** — rank-weighted graded gain, normalized to the ideal ordering.
- **MRR** — reciprocal rank of the first highly-relevant (grade 2) result.

---

## 2. Quality results

### Aggregate (mean over 14 queries)

| Metric | **fast-code-embed** | potion-base-8M | FCE advantage |
|---|---|---|---|
| **nDCG@10** | **0.839** | 0.655 | **+28%** |
| **MRR** | **0.82** | 0.57 | **+44%** |
| **Strict P@10** (grade 2 only) | **0.65** | 0.43 | **+51%** |
| P@10 (relevant-or-partial) | **0.86** | 0.81 | +6% |

The loose P@10 is close — both systems return *broadly on-topic* code — but on
every metric that rewards putting the *most* relevant code *first*
(nDCG, MRR, strict P@10) FCE leads substantially.

### Per query (FCE / potion)

| Query | P@10 | nDCG@10 | MRR |
|---|---|---|---|
| gpu display drivers | 1.00 / 1.00 | **1.000** / 0.426 | **1.00** / 0.50 |
| user mode scheduling | **0.60** / 0.00 | **0.667** / 0.000 | **0.17** / 0.00 |
| pcie ethernet code | 1.00 / 1.00 | 1.000 / 1.000 | 0.00 / 0.00 |
| memory allocation pages | 1.00 / 1.00 | 0.845 / 0.842 | 1.00 / 1.00 |
| file system inode | 0.90 / 1.00 | **0.861** / 0.753 | 1.00 / 1.00 |
| network socket buffer | **1.00** / 0.60 | **1.000** / 0.572 | **1.00** / 0.50 |
| interrupt handler irq | 1.00 / 1.00 | 1.000 / 1.000 | 1.00 / 1.00 |
| crypto aes cipher | 0.90 / 1.00 | 0.690 / **0.750** | 1.00 / 1.00 |
| spinlock mutex locking | 1.00 / 1.00 | **1.000** / 0.763 | **1.00** / 0.50 |
| tcp congestion control | **1.00** / 0.40 | **1.000** / 0.406 | 1.00 / 1.00 |
| ext4 journaling commit transaction | 1.00 / 1.00 | 1.000 / 1.000 | 1.00 / 1.00 |
| rcu read lock grace period | 0.10 / **1.00** | 0.192 / **0.801** | 0.33 / 0.33 |
| dma buffer mapping | 0.60 / **0.90** | 0.595 / 0.575 | **1.00** / 0.20 |
| page fault handler | **1.00** / 0.50 | **0.890** / 0.283 | **1.00** / 0.00 |

FCE wins or ties 12 of 14 queries on nDCG. (Bold marks the clearly stronger
system where the gap is material.)

---

## 3. Performance

Measured on Apple Silicon (single machine), 80,190 chunks, 14 queries.
FCE built in its default **768-dim** configuration; potion is **256-dim**.

### Indexing

| Stage | fast-code-embed | potion-base-8M |
|---|---|---|
| Model / table load (one-time) | included below | 0.5 s (warm) |
| Build index from 80 K chunks | **7.18 s** total¹ | 5.0 s encode |
| **Total cold index build** | **7.18 s** | 5.5 s |
| Throughput | ~11.2 K docs/s | ~16 K chunks/s |

¹ FCE's 7.18 s **includes** loading the nomic-embed-code token table; it is a
single end-to-end measurement from corpus creation through finalize.

potion indexes this corpus somewhat faster in wall-clock terms — expected, since
a static model is a table lookup + mean over 256 dims, while FCE loads a larger
768-dim code-trained table and computes TF-IDF/Random-Indexing enrichment. Both
are comfortably in the "index a real codebase in seconds" regime.

### Query latency (mean per query, top-15, full 80 K corpus)

| Path | Latency | Notes |
|---|---|---|
| **FCE — fast (inverted index + rerank)** | **1.05 ms** | production path |
| **FCE — brute-force reference** | **1.51 ms** | exhaustive; used for the quality results above |
| potion — dense cosine | 2.48 ms | numpy matmul over all chunks² |

² Excludes query-string encoding; FCE's figure includes query tokenization and
query-vector construction.

**FCE answers queries ~1.6–2.4× faster than the potion cosine scan despite using
3× higher-dimensional (768 vs. 256) vectors**, thanks to its hand-tuned SIMD
cosine kernels. FCE's production "fast" path widens the gap further and scales
sub-linearly via the inverted index, whereas the static-embedding baseline pays a
full dense scan on every query.

---

## 4. Analysis

### Where FCE wins: polysemy resolution

FCE's code-trained provenance shows up exactly where a general text model
struggles — words that mean different things in different subsystems:

- **"user mode scheduling"** — potion returned DRM **display "mode"** validation
  (`drm_modes.c`, `intel_panel.c`, `intel_ddi.c`), 0/10 relevant; FCE returned
  `kernel/sched/ext`, `cpufreq_schedutil`, `userfaultfd`, `seccomp`. Largest gap.
- **"network socket buffer"** — potion pulled GPU `drm_client_buffer_*` and gvt
  `RING_BUFFER` (matching "buffer"); FCE stayed in `net/socket.c`, `skbuff.c`.
- **"tcp congestion control"** — potion matched i915 **Type-C ("tc") PHY** port
  code; FCE returned the real `tcp_cong.c` / `tcp_*` congestion modules.
- **"page fault handler"** — potion returned signal/reboot/`sock_diag` *handlers*;
  FCE returned `mm/memory.c` `do_*_fault`.

potion only ties or edges ahead when the query terms are unambiguous,
tightly co-occurring tokens (`crypto aes cipher`, `memory allocation pages`,
`ext4 journaling`, `interrupt irq`).

### Where FCE loses (and why)

Two queries went the other way, with a shared root cause:

- **"rcu read lock grace period"** (FCE nDCG 0.19 vs. 0.80) — FCE's top-10
  filled with `kernel/sched/fair.c`/`rt.c` chunks containing the generic token
  **"period"** (scheduling period), not RCU grace periods.
- **"dma buffer mapping"** (FCE 0.60 vs. 0.90 P@10) — FCE pulled `address_space`
  **"mapping"** chunks (`shmem`, `page-writeback`, `nommu`) unrelated to DMA.

**Cause:** FCE builds the query vector by aggregating per-token enriched vectors,
so a single common-but-generic token (`period`, `mapping`) can dominate and pull
in lexically-matching but irrelevant documents, with no cross-token phrase
context. Likely mitigations: stronger IDF down-weighting of high-frequency tokens
*in the query vector*, or a phrase/bigram signal so that rarer, more
discriminative tokens (`rcu`, `grace`, `dma`) carry proportionally more weight.

---

## 5. Reproduction

The comparison follows the methodology in Section 1: chunk the corpus once at
brace boundaries, index the identical chunk list with each system, run the 14
queries, then pool both systems' top-10 and score against graded relevance
judgments (P@10, strict P@10, nDCG@10, MRR).

- **FCE** is driven through its public C API — `fce_sem_tokenize_batch` +
  `fce_sem_corpus_add_docs_batch`, then `fce_sem_corpus_finalize`, querying via
  `fce_sem_search_query_bruteforce` (and `fce_sem_search_query_fast` for the
  fast-path latency figure).
- **potion-base-8M** is loaded with [model2vec](https://github.com/MinishLab/model2vec)
  (`StaticModel.from_pretrained("minishlab/potion-base-8M")`), encoding each
  chunk's raw text, then ranked by dense cosine.

---

## 6. Caveats

- **Single judge.** Relevance grades are the author's, applied to code *content*
  and pooled so neither system was favored during labeling — but they are not
  multi-annotator gold labels.
- **Scope.** 14 queries, ~80 K chunks, one corpus. Results are directional, not a
  formal benchmark, and the absolute timings are machine-specific.
- **potion usage.** A general-English static model embedding raw code is its
  intended off-the-shelf mode; it has no code-specific pretraining — which is
  precisely the contrast being measured.
- **Configuration.** FCE ran at its default 768 dims; selecting 256 dims at
  runtime (`fce_sem_set_dim(256)`) would narrow the indexing-time and memory gap
  with potion while remaining the higher-quality retriever.
