"""fast-code-embed — Pythonic interface to the C semantic-search library.

Algorithmic code embeddings: TF-IDF + pretrained nomic-embed-code token vectors
(+ optional Reflective Random Indexing). No GPU, no API keys, no model download —
the 30 MB vector table is compiled into the extension.

Quick start
-----------
>>> import fast_code_embed as fce
>>> with fce.Corpus() as corpus:
...     corpus.add_doc("parse_config", "def parse_config(path): ...")
...     corpus.add_doc("retry_request", "def retry_request(url, attempts): ...")
...     corpus.finalize()
...     for hit in corpus.search("load configuration from disk", k=5):
...         print(f"{hit.score:.3f}  {hit.label}")

The module-level config functions (`set_dim`, `set_idf_weighting`,
`set_abbrev_expansion`) map to global process state in the C library — set them
ONCE at startup, before building or querying, and keep them consistent between
finalize and search (see the C header notes on idf_weighting).
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Sequence, Tuple

from ._fast_code_embed import ffi, lib

__all__ = [
    "Corpus",
    "SearchResult",
    "Mode",
    "tokenize",
    "set_dim",
    "active_dim",
    "set_idf_weighting",
    "idf_weighting",
    "set_abbrev_expansion",
    "abbrev_expansion",
]

# Mirrors FCE_SEM_MAX_TOKENS in the C header.
_MAX_TOKENS = 512


def _encode(s: str, what: str) -> bytes:
    """UTF-8 encode, rejecting embedded NUL bytes.

    The C side treats every string as NUL-terminated, so a NUL would silently
    truncate the value (dropping label/text/path content with no error). Reject
    it explicitly rather than let it corrupt data through tokenize/save/load.
    """
    if "\x00" in s:
        raise ValueError(f"{what} must not contain an embedded NUL byte")
    return s.encode("utf-8")


def _current_cfg() -> tuple:
    """Snapshot the global toggles that must stay consistent build->query."""
    return (lib.fce_sem_active_dim(),
            bool(lib.fce_sem_idf_weighting()),
            bool(lib.fce_sem_abbrev_expansion()))


class Mode(Enum):
    """Search strategy. Maps to the C `fce_sem_search_query*` family.

    DEFAULT (AUTO) uses the fast inverted-index path with a brute-force fallback.
    FAST / TFIDF / BRUTEFORCE select a specific strategy. Only BRUTEFORCE is
    guaranteed to find the global top-k (the index paths require a shared literal
    token); use it when exhaustive recall matters.
    """

    DEFAULT = "auto"
    FAST = "fast"
    TFIDF = "tfidf"
    BRUTEFORCE = "bruteforce"


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked hit. `index` is the corpus document index; `label` is the
    caller-supplied label (or loaded file path), empty if none was provided."""

    index: int
    label: str
    score: float


# ── Module-level config (global process state in C) ─────────────────────────
def set_dim(dim: int) -> None:
    """Select the embedding dimension (256 or 768). Set once at startup."""
    lib.fce_sem_set_dim(int(dim))


def active_dim() -> int:
    return lib.fce_sem_active_dim()


def set_idf_weighting(enabled: bool) -> None:
    lib.fce_sem_set_idf_weighting(bool(enabled))


def idf_weighting() -> bool:
    return bool(lib.fce_sem_idf_weighting())


def set_abbrev_expansion(enabled: bool) -> None:
    lib.fce_sem_set_abbrev_expansion(bool(enabled))


def abbrev_expansion() -> bool:
    return bool(lib.fce_sem_abbrev_expansion())


def tokenize(name: str) -> List[str]:
    """Split a name into lowercased tokens (camelCase / snake_case / dotted)."""
    out = ffi.new("char *[]", _MAX_TOKENS)
    n = lib.fce_sem_tokenize(name.encode("utf-8"), out, _MAX_TOKENS)
    try:
        return [ffi.string(out[i]).decode("utf-8", "replace") for i in range(n)]
    finally:
        for i in range(n):
            lib.free(out[i])


class Corpus:
    """A searchable corpus of documents.

    Build with `add_doc` / `add_docs` / `add_files`, call `finalize()`, then
    `search()`. Use as a context manager (or rely on GC) so the native handle is
    freed. The corpus must be BUILT from a single thread (the C build path is not
    thread-safe); once finalized, concurrent read-only queries are safe.

    The global toggles `set_dim` / `set_idf_weighting` / `set_abbrev_expansion`
    are process state in C and are read live at query time, so they must hold the
    same value at query time as when the corpus was finalized. This class
    snapshots them at finalize() and raises from search() if they have diverged,
    rather than returning silently wrong scores.

    Labels are tracked on the Python side in insertion order and surfaced on
    `SearchResult.label`; this assumes documents are added in order and not
    rejected (empty-token docs are skipped and do not consume an index).
    """

    def __init__(self) -> None:
        self._c = lib.fce_sem_corpus_new()
        if self._c == ffi.NULL:
            raise MemoryError("fce_sem_corpus_new failed")
        self._labels: List[str] = []
        self._finalized = False
        self._loaded = False  # loaded corpora carry labels in C, not Python
        self._closed = False
        self._finalize_failed = False
        self._cfg: tuple = ()        # toggles snapshot at finalize()/load()
        self._build_abbrev = None    # abbrev value fixed at first add_*
        self._owner = threading.get_ident()  # thread allowed to build

    # ── building ────────────────────────────────────────────────────────
    def add_doc(self, label: str, text: str) -> None:
        """Tokenize `text` and add it as one document tagged with `label`."""
        self._check_mutable()
        # Validate both before any C call so a bad label can't add a doc.
        c_text = _encode(text, "text")
        _ = _encode(label, "label")
        out = ffi.new("char *[]", _MAX_TOKENS)
        n = lib.fce_sem_tokenize(c_text, out, _MAX_TOKENS)
        try:
            if n > 0:
                lib.fce_sem_corpus_add_doc(
                    self._c, ffi.cast("const char **", out), n
                )
                self._labels.append(label)
        finally:
            for i in range(n):
                lib.free(out[i])

    def add_docs(self, items: Iterable[Tuple[str, str]]) -> None:
        """Add many (label, text) pairs."""
        for label, text in items:
            self.add_doc(label, text)

    def add_files(
        self,
        paths: Sequence[str],
        *,
        chunk_size: int = 2048,
        max_tokens_per_chunk: int = 0,
    ) -> List[int]:
        """Read source files, chunk by `}` boundaries, tokenize and add them.

        `chunk_size` is the target chunk size in BYTES (must be > 0; the C side
        rejects non-positive values and caps it at 16 MiB). `max_tokens_per_chunk`
        of 0 means the library default. Returns the number of document chunks
        produced per input path; each chunk's label is its source file path. All
        work happens in C.

        Under partial-batch rejection (only at the library's OOM / 1M-doc /
        5M-vocab limits) the per-file counts are approximate — the total is exact
        but a chunk may be attributed to the wrong file. When that happens this
        method warns and keeps the label list aligned to the true document count
        so search() and save() stay correct.
        """
        self._check_mutable()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive byte count")
        n = len(paths)
        if n == 0:
            return []
        c_paths = [ffi.new("char[]", _encode(p, "path")) for p in paths]
        arr = ffi.new("const char *[]", c_paths)
        counts = ffi.new("int[]", n)
        before = self.doc_count
        total = lib.fce_sem_corpus_add_files(
            self._c, arr, n, chunk_size, counts, max_tokens_per_chunk
        )
        if total < 0:
            raise RuntimeError("fce_sem_corpus_add_files failed")
        added = self.doc_count - before
        per_file = [counts[i] for i in range(n)]
        new_labels: List[str] = []
        for path, produced in zip(paths, per_file):
            new_labels.extend([path] * produced)
        if len(new_labels) != added:
            warnings.warn(
                "add_files: per-file chunk counts are approximate under partial "
                "batch rejection; labels kept aligned to the document count",
                RuntimeWarning, stacklevel=2,
            )
            if len(new_labels) > added:
                new_labels = new_labels[:added]
            else:
                new_labels.extend([""] * (added - len(new_labels)))
        self._labels.extend(new_labels)
        return per_file

    def set_ri_enrichment(self, enabled: bool) -> None:
        """Toggle Reflective Random Indexing enrichment. Call before finalize."""
        self._check_mutable()
        lib.fce_sem_corpus_set_ri_enrichment(self._c, bool(enabled))

    def finalize(self) -> None:
        """Compute IDF and build token/doc vectors. Required before search."""
        self._check_mutable()
        if lib.fce_sem_corpus_finalize(self._c) != 0:
            # The C side marks the corpus permanently unusable; it must not be
            # finalized again or added to. Record that so later calls fail loudly.
            self._finalize_failed = True
            raise RuntimeError(
                "fce_sem_corpus_finalize failed (out of memory); this corpus is "
                "now unusable — close() it and build a new Corpus"
            )
        self._finalized = True
        self._cfg = _current_cfg()

    # ── querying ────────────────────────────────────────────────────────
    def search(
        self, query: str, k: int = 10, mode: Mode = Mode.DEFAULT
    ) -> List[SearchResult]:
        """Return up to `k` results, ranked by descending score."""
        self._ensure_open()
        if not self._finalized and not self._loaded:
            raise RuntimeError("call finalize() before search()")
        if k < 1:
            raise ValueError("k must be a positive integer")
        self._check_query_cfg()
        results = ffi.new("fce_sem_ranked_t[]", k)
        count = ffi.new("uint32_t *")
        q = _encode(query, "query")
        if mode is Mode.FAST:
            lib.fce_sem_search_query_fast(self._c, q, k, results, count)
        elif mode is Mode.TFIDF:
            lib.fce_sem_search_query_tfidf(self._c, q, k, results, count)
        elif mode is Mode.BRUTEFORCE:
            lib.fce_sem_search_query_bruteforce(self._c, q, k, results, count)
        else:  # DEFAULT -> AUTO, via a config with query_mode = AUTO
            cfg = ffi.new("fce_sem_config_t *", lib.fce_sem_get_config())
            cfg.query_mode = lib.FCE_QUERY_AUTO
            lib.fce_sem_search_query(self._c, q, k, results, count, cfg)
        return [
            SearchResult(results[i].index, self._label_for(results[i].index),
                         results[i].score)
            for i in range(count[0])
        ]

    def candidate_count(self, query: str) -> int:
        """How many candidates the inverted index would retrieve for `query`."""
        self._ensure_open()
        return lib.fce_sem_search_candidate_count(self._c, _encode(query, "query"))

    # ── persistence ─────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Persist a finalized corpus (same-build cache, not portable)."""
        self._ensure_open()
        if not self._finalized:
            raise RuntimeError("only a finalized corpus can be saved")
        c_path = _encode(path, "path")
        labels = self._labels
        if labels:
            if len(labels) != self.doc_count:
                raise RuntimeError(
                    f"label count ({len(labels)}) does not match document count "
                    f"({self.doc_count}); cannot save a consistent label set"
                )
            keep = [ffi.new("char[]", _encode(s, "label")) for s in labels]
            arr = ffi.new("const char *[]", keep)
            rc = lib.fce_sem_corpus_save(
                self._c, c_path,
                ffi.cast("const char *const *", arr), len(labels),
            )
        else:
            rc = lib.fce_sem_corpus_save(self._c, c_path, ffi.NULL, 0)
        if rc != 0:
            raise RuntimeError("fce_sem_corpus_save failed")

    @classmethod
    def load(cls, path: str) -> "Corpus":
        """Load a corpus previously written by `save()` (same build only)."""
        c = lib.fce_sem_corpus_load(_encode(path, "path"))
        if c == ffi.NULL:
            raise RuntimeError(f"fce_sem_corpus_load failed for {path!r}")
        self = cls.__new__(cls)
        self._c = c
        self._labels = []
        self._finalized = True
        self._loaded = True
        self._closed = False
        self._finalize_failed = False
        # The dim is enforced by the loader; idf/abbrev are not stored, so adopt
        # the current global toggles as the corpus's assumed query-time config.
        self._cfg = _current_cfg()
        self._build_abbrev = None
        self._owner = threading.get_ident()
        return self

    # ── lifecycle / dunder ──────────────────────────────────────────────
    @property
    def doc_count(self) -> int:
        self._ensure_open()
        return lib.fce_sem_corpus_doc_count(self._c)

    def _label_for(self, index: int) -> str:
        if self._loaded:
            ptr = lib.fce_sem_corpus_doc_label(self._c, index)
            return "" if ptr == ffi.NULL else ffi.string(ptr).decode(
                "utf-8", "replace"
            )
        return self._labels[index] if 0 <= index < len(self._labels) else ""

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("corpus is closed")

    def _check_mutable(self) -> None:
        self._ensure_open()
        if self._finalize_failed:
            raise RuntimeError(
                "corpus is unusable after a failed finalize(); close() and "
                "build a new Corpus"
            )
        if self._finalized:
            raise RuntimeError("corpus is finalized; rebuild to add documents")
        if threading.get_ident() != self._owner:
            raise RuntimeError(
                "a Corpus must be built from the thread that created it"
            )
        # Tokenization depends on the global abbrev toggle; pin it for the whole
        # build so documents are not tokenized under inconsistent settings.
        cur = bool(lib.fce_sem_abbrev_expansion())
        if self._build_abbrev is None:
            self._build_abbrev = cur
        elif cur != self._build_abbrev:
            raise RuntimeError(
                "abbrev_expansion changed mid-build; keep set_abbrev_expansion "
                "constant while adding documents to a corpus"
            )

    def _check_query_cfg(self) -> None:
        now = _current_cfg()
        if self._cfg and now != self._cfg:
            names = ("dim", "idf_weighting", "abbrev_expansion")
            changed = [f"{n}: {a!r}->{b!r}"
                       for n, a, b in zip(names, self._cfg, now) if a != b]
            raise RuntimeError(
                "global config changed since finalize (" + ", ".join(changed)
                + "); restore it before querying or results will be wrong"
            )

    def close(self) -> None:
        c = getattr(self, "_c", ffi.NULL)
        if c != ffi.NULL:
            lib.fce_sem_corpus_free(c)
            self._c = ffi.NULL
        self._closed = True

    def __enter__(self) -> "Corpus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.doc_count

    def __repr__(self) -> str:
        if getattr(self, "_closed", True):
            return "<Corpus closed>"
        if self._finalize_failed:
            return "<Corpus failed>"
        state = "loaded" if self._loaded else (
            "finalized" if self._finalized else "building"
        )
        return f"<Corpus docs={lib.fce_sem_corpus_doc_count(self._c)} {state}>"
