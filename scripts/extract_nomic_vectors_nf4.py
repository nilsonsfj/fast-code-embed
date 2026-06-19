#!/usr/bin/env python3
"""
Extract token embeddings from nomic-embed-code (7B) — 4-bit NF4 GPU variant.

Identical to extract_nomic_vectors.py except the model is loaded in 4-bit
(bitsandbytes NF4) so the 7B model fits in ~4 GB of VRAM and runs on a small
GPU (e.g. an 8 GB RTX 4060) instead of crawling on CPU. The fp16 path OOMs on
an 8 GB card because the model needs ~14 GB at fp16.

The weights are still downloaded in whatever dtype the repo hosts (~30 GB for
fp32) — 4-bit quantization happens after load, in memory. The vectors are a
hair different from the fp16 path, but the whole vocabulary is re-distilled in
one consistent pass and ends up int8-quantized in the blob, so relative
geometry is preserved.

Usage (Windows/Linux with CUDA):
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate bitsandbytes sentence-transformers einops
    python scripts/extract_nomic_vectors_nf4.py [--output-dir src/embed]

Output:
    code_vectors.bin   — [int32 count][int32 dim] + count×dim int8
    code_tokens.txt    — one token per line
    code_tokens.h      — C header: static const char *FCE_PRETRAINED_TOKENS[N]
    code_vectors.h     — C header: defines + inline accessor
    code_vectors_blob.S — assembler .incbin (cross-platform)

Resumable: checkpoints to <output-dir>/checkpoint.npz every 500 tokens. Copy a
checkpoint.npz from a partial CPU run here to resume instead of restarting.
"""

import argparse
import json
import os
import re
import string
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Parallelize CPU inference across all cores BEFORE any torch ops
NUM_THREADS = min(os.cpu_count() * 2, 12)
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(max(NUM_THREADS // 2, 1))
os.environ.setdefault("OMP_NUM_THREADS", str(NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(NUM_THREADS))

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


# ── Configuration ──────────────────────────────────────────────────────

MODEL_NAME = "nomic-ai/nomic-embed-code"
OUTPUT_DIM = 768          # Target dimension (Matryoshka truncation if model outputs more)
SIM_ATTENTION_K = 32      # Top-K neighbors for simulated attention
SIM_ATTENTION_ITERS = 3   # Number of simulated attention iterations
SIM_ATTENTION_ALPHA = 0.3 # Blend ratio: (1-α)×original + α×neighbor_mean
BATCH_SIZE = 32           # Tokens per inference batch (sized for thread saturation)
CHECKPOINT_EVERY = 500    # Save checkpoint every N tokens


# ── Token filtering ───────────────────────────────────────────────────

def is_code_relevant(token_str: str) -> bool:
    """Filter vocabulary to code-relevant tokens.

    Goal: keep tokens that our runtime camelCase/snake_case splitter would
    produce from identifiers. Reject BPE noise, punctuation combos, and
    non-Latin scripts.
    """
    s = token_str.strip()
    if not s:
        return False

    # Remove BPE markers (Ġ = space prefix, ▁ = sentencepiece, Ċ/ċ = newline marker)
    clean = s.lstrip("\u0120\u2581")  # Ġ, ▁
    if not clean:
        return False

    # Skip special tokens
    if clean.startswith("<") and clean.endswith(">"):
        return False
    if clean.startswith("[") and clean.endswith("]"):
        return False

    # Strip leading/trailing underscores (common in BPE) but keep content
    inner = clean.strip("_")
    if not inner:
        return False

    # STRICT: must be purely alphanumeric + underscores (identifier-shaped)
    # This rejects BPE noise like "!");ċ", "!!!!ċċ", etc.
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', inner):
        return False

    # Must be at least 2 chars of actual content
    if len(inner) < 2:
        return False

    return True


def clean_token(token_str: str) -> str:
    """Normalize a BPE token to the form our runtime tokenizer produces."""
    s = token_str.strip()
    # Strip BPE space markers
    s = s.lstrip("\u0120\u2581")
    # Strip leading/trailing underscores
    s = s.strip("_")
    # Lowercase (our runtime tokenizer lowercases)
    s = s.lower()
    return s


# ── Simulated attention ──────────────────────────────────────────────

def simulated_attention(vectors: np.ndarray, k: int, iterations: int,
                        alpha: float) -> np.ndarray:
    """
    Apply simulated self-attention: for each vector, blend with mean of
    top-K nearest neighbors. This approximates contextual composition
    that real attention provides.

    vectors: (N, D) float32 unit-normalized
    Returns: (N, D) float32 unit-normalized
    """
    n, d = vectors.shape
    result = vectors.copy()

    for iteration in range(iterations):
        t0 = time.time()
        # Compute cosine similarity matrix in chunks to avoid OOM
        # For 40K vectors × 768d, full matrix = 40K² × 4 bytes = 6.4GB
        # Process in chunks of 2048
        chunk_size = 2048
        new_result = np.zeros_like(result)

        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk = result[i:end]  # (chunk, D)

            # Cosine similarity: chunk × all^T
            sims = chunk @ result.T  # (chunk, N)

            # For each vector in chunk, find top-K neighbors (excluding self)
            for j in range(end - i):
                global_idx = i + j
                sim_row = sims[j].copy()
                sim_row[global_idx] = -1.0  # Exclude self

                # Top-K indices
                if k < n - 1:
                    top_k_idx = np.argpartition(sim_row, -k)[-k:]
                else:
                    top_k_idx = np.arange(n)
                    top_k_idx = top_k_idx[top_k_idx != global_idx]

                neighbor_mean = result[top_k_idx].mean(axis=0)

                # Blend
                blended = (1 - alpha) * result[global_idx] + alpha * neighbor_mean
                # Re-normalize
                norm = np.linalg.norm(blended)
                if norm > 1e-8:
                    blended /= norm
                new_result[global_idx] = blended

        result = new_result
        elapsed = time.time() - t0
        print(f"  sim-attention iter {iteration + 1}/{iterations}: {elapsed:.1f}s")

    return result


# ── Extraction ───────────────────────────────────────────────────────

def extract_embeddings(model, tokenizer, tokens: list, device: str,
                       batch_size: int = 64,
                       checkpoint_path: str = None) -> np.ndarray:
    """Run full model inference on each token string. Returns (N, D) float32."""

    # Check for checkpoint
    start_idx = 0
    all_vecs = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path)
        all_vecs = list(data["vectors"])
        start_idx = len(all_vecs)
        print(f"  resuming from checkpoint: {start_idx}/{len(tokens)} tokens")

    model.eval()
    total = len(tokens)
    t0 = time.time()

    with torch.no_grad():
        for batch_start in range(start_idx, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_tokens = tokens[batch_start:batch_end]

            # nomic-embed-code requires search_query or search_document prefix
            # For single tokens, we use the token as-is (query mode)
            texts = [f"search_query: {t}" for t in batch_tokens]

            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded)

            # Mean pooling over non-padding tokens
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded, dim=1
            )
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # Truncate to OUTPUT_DIM if model outputs more (Matryoshka)
            if mean_pooled.shape[1] > OUTPUT_DIM:
                mean_pooled = mean_pooled[:, :OUTPUT_DIM]

            # L2 normalize
            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            vecs = mean_pooled.cpu().numpy()
            all_vecs.extend(vecs)

            # Progress
            done = batch_end
            elapsed = time.time() - t0
            rate = (done - start_idx) / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  [{done:>6}/{total}] "
                f"{rate:.1f} tok/s  "
                f"ETA {eta / 60:.0f}m",
                flush=True
            )

            # Checkpoint
            if checkpoint_path and (done % CHECKPOINT_EVERY < batch_size):
                np.savez_compressed(
                    checkpoint_path,
                    vectors=np.array(all_vecs, dtype=np.float32)
                )

    print()
    return np.array(all_vecs, dtype=np.float32)


# ── Output generation ────────────────────────────────────────────────

def write_bin(path: str, vectors: np.ndarray, dim: int):
    """Write binary blob: [int32 count][int32 dim] + count×dim int8."""
    n = vectors.shape[0]
    # Quantize: scale to [-127, 127], round to int8
    quantized = np.clip(np.round(vectors * 127.0), -127, 127).astype(np.int8)

    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, dim))
        f.write(quantized.tobytes())

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  {path}: {n} vectors × {dim}d = {size_mb:.1f} MB")


def write_tokens_txt(path: str, tokens: list):
    """Write plain text token list."""
    with open(path, "w") as f:
        for t in tokens:
            f.write(t + "\n")
    print(f"  {path}: {len(tokens)} tokens")


def write_tokens_h(path: str, tokens: list):
    """Write C header with token string array."""
    with open(path, "w") as f:
        f.write(f"/* nomic-embed-code token vocabulary — {len(tokens)} tokens. */\n")
        f.write("#ifndef FCE_EMBED_TOKENS_H\n")
        f.write("#define FCE_EMBED_TOKENS_H\n\n")
        f.write(f"static const char *FCE_PRETRAINED_TOKENS[{len(tokens)}] = {{\n")
        for t in tokens:
            escaped = t.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'"{escaped}",\n')
        f.write("};\n\n")
        f.write("#endif /* FCE_EMBED_TOKENS_H */\n")
    print(f"  {path}: written")


def write_vectors_h(path: str, token_count: int, dim: int, incbin_path: str):
    """Write C header with defines and inline accessor."""
    with open(path, "w") as f:
        f.write(f"""/* nomic-embed-code (nomic-ai/nomic-embed-code) token embeddings.
 * {token_count} tokens x {dim}d int8-quantized unit vectors.
 * Distilled from 7B model via full inference on filtered vocabulary.
 * Simulated attention: {SIM_ATTENTION_ITERS} iterations, K={SIM_ATTENTION_K}, alpha={SIM_ATTENTION_ALPHA}.
 *
 * Vector blob embedded via code_vectors_blob.S (assembler .incbin).
 * Token strings are in this header as a static array.
 *
 * Source: https://huggingface.co/nomic-ai/nomic-embed-code
 * License: Apache 2.0
 */
#ifndef FCE_EMBED_VECTORS_H
#define FCE_EMBED_VECTORS_H

#include <stdint.h>

#define FCE_PRETRAINED_TOKEN_COUNT {token_count}
#define FCE_PRETRAINED_DIM {dim}

/* Raw vector blob: first 8 bytes = [int32 count][int32 dim],
 * then count x dim int8 values (unit-normalized, x127 scaled). */
extern const unsigned char FCE_PRETRAINED_VECTOR_BLOB[];
extern const unsigned int FCE_PRETRAINED_VECTOR_BLOB_LEN;

/* Access the int8 vector for token index i. */
static inline const int8_t *fce_pretrained_vec_at(int i) {{
    return (const int8_t *)(FCE_PRETRAINED_VECTOR_BLOB + 8 + (size_t)i * FCE_PRETRAINED_DIM);
}}

/* Token strings (separate header to keep this file clean). */
#include "code_tokens.h"

#endif /* FCE_EMBED_VECTORS_H */
""")
    print(f"  {path}: written")


def write_blob_s(path: str, incbin_path: str):
    """Write the cross-platform assembler .incbin directive.

    Must match the symbol names used by src/embed/code_vectors.h and the
    section/prefix conventions of each object format (Mach-O underscores the
    symbol; ELF/COFF do not). The ELF branch also emits a .note.GNU-stack
    marker so the linker does not stamp the shared object executable-stack."""
    with open(path, "w") as f:
        f.write(f"""/* nomic-embed-code vector blob embedded via assembler.
 * Cross-platform: macOS (Mach-O) vs Linux (ELF) vs Windows (COFF). */

#if defined(__APPLE__)
    .section __DATA,__const
    .globl _FCE_PRETRAINED_VECTOR_BLOB
    .globl _FCE_PRETRAINED_VECTOR_BLOB_LEN
    .p2align 4
_FCE_PRETRAINED_VECTOR_BLOB:
    .incbin "{incbin_path}"
_FCE_PRETRAINED_VECTOR_BLOB_END:

    .section __DATA,__const
    .p2align 2
_FCE_PRETRAINED_VECTOR_BLOB_LEN:
    .long _FCE_PRETRAINED_VECTOR_BLOB_END - _FCE_PRETRAINED_VECTOR_BLOB

#elif defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
    .section .rdata,"dr"
    .globl FCE_PRETRAINED_VECTOR_BLOB
    .globl FCE_PRETRAINED_VECTOR_BLOB_LEN
    .p2align 4
FCE_PRETRAINED_VECTOR_BLOB:
    .incbin "{incbin_path}"
FCE_PRETRAINED_VECTOR_BLOB_END:

    .section .rdata,"dr"
    .p2align 2
FCE_PRETRAINED_VECTOR_BLOB_LEN:
    .long FCE_PRETRAINED_VECTOR_BLOB_END - FCE_PRETRAINED_VECTOR_BLOB

#else
    .section .rodata,"a",@progbits
    .globl FCE_PRETRAINED_VECTOR_BLOB
    .globl FCE_PRETRAINED_VECTOR_BLOB_LEN
    .p2align 4
FCE_PRETRAINED_VECTOR_BLOB:
    .incbin "{incbin_path}"
FCE_PRETRAINED_VECTOR_BLOB_END:

    .section .rodata,"a",@progbits
    .p2align 2
FCE_PRETRAINED_VECTOR_BLOB_LEN:
    .long FCE_PRETRAINED_VECTOR_BLOB_END - FCE_PRETRAINED_VECTOR_BLOB

    /* Mark the stack non-executable. Without this marker the linker stamps
     * PT_GNU_STACK RWE on the shared object, silently disabling NX for the
     * whole JVM process and causing dlopen failures on glibc >= 2.41. */
    .section .note.GNU-stack,"",@progbits
#endif
""")
    print(f"  {path}: written")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract nomic-embed-code token embeddings")
    parser.add_argument("--output-dir", default="src/embed",
                        help="Output directory (default: src/embed)")
    parser.add_argument("--device", default=None,
                        help="Device: cuda, mps, cpu (auto-detected)")
    parser.add_argument("--skip-attention", action="store_true",
                        help="Skip simulated attention (faster, lower quality)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint file path (auto: <output-dir>/checkpoint.npz)")
    args = parser.parse_args()

    batch_size = args.batch_size

    # Auto-detect device
    # Prefer CPU for 7B models on Apple Silicon — MPS shares unified memory
    # with the system and can cause OOM/crashes. CPU keeps allocation predictable.
    # Use --device mps to override if you have enough headroom (32GB+).
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Force line-buffered stdout so tee/log sees output immediately
    sys.stdout.reconfigure(line_buffering=True)

    print(f"device={device}")
    print(f"threads={torch.get_num_threads()}")
    print(f"model={MODEL_NAME}")
    print(f"output_dim={OUTPUT_DIM}")
    print()

    # Create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint or str(out_dir / "checkpoint.npz")

    # ── Step 1: Load model + tokenizer ──
    print("step 1: loading model + tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # extra ~0.4 GB saving — helps on 8 GB
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",               # places quantized weights on the GPU
        dtype=torch.float16,             # compute dtype for non-quantized modules
        low_cpu_mem_usage=True,
    )
    # NOTE: do NOT call model.to(device) here — bitsandbytes + device_map have
    # already placed the (quantized) weights; calling .to() raises an error.
    # Re-sync `device` to where the model actually landed so inputs match.
    device = str(model.device)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  hidden_size={model.config.hidden_size}")
    print(f"  vocab_size={tokenizer.vocab_size}")
    print()

    # ── Step 2: Filter vocabulary ──
    print("step 2: filtering vocabulary to code-relevant tokens...")
    vocab = tokenizer.get_vocab()
    print(f"  raw vocabulary: {len(vocab)} tokens")

    # Filter and deduplicate
    seen = set()
    filtered_tokens = []
    for tok_str, tok_id in sorted(vocab.items(), key=lambda x: x[1]):
        if not is_code_relevant(tok_str):
            continue
        clean = clean_token(tok_str)
        if not clean or clean in seen:
            continue
        if len(clean) < 2:
            continue
        seen.add(clean)
        filtered_tokens.append(clean)

    print(f"  code-relevant (deduplicated): {len(filtered_tokens)} tokens")

    # ── Single-character base layer (a-z, 0-9) ──
    # The runtime tokenizer emits only lowercase [a-z0-9] tokens and, for any
    # identifier missing from this vocabulary, decomposes it into the longest
    # in-vocab subwords. A single-character base layer guarantees that
    # decomposition always terminates on a known token, so there is no
    # out-of-vocabulary case and the random-vector fallback in semantic.c
    # (fce_sem_random_index / build_src_entry) is never reached. These are
    # normally removed by the len>=2 rule above, so add them explicitly.
    base_layer = list(string.ascii_lowercase + string.digits)
    added = 0
    for ch in base_layer:
        if ch not in seen:
            seen.add(ch)
            filtered_tokens.append(ch)
            added += 1
    print(f"  base layer: +{added} single-char tokens (a-z, 0-9)")

    filtered_tokens.sort()
    print(f"  total vocabulary: {len(filtered_tokens)} tokens")

    # Show sample
    sample = filtered_tokens[:20]
    print(f"  sample: {sample}")
    print()

    # ── Step 3: Extract embeddings (full inference) ──
    print(f"step 3: extracting embeddings ({len(filtered_tokens)} tokens, batch_size={batch_size})...")
    t0 = time.time()
    vectors = extract_embeddings(
        model, tokenizer, filtered_tokens, device,
        batch_size=batch_size, checkpoint_path=checkpoint_path
    )
    elapsed = time.time() - t0
    print(f"  extracted {vectors.shape[0]} vectors × {vectors.shape[1]}d in {elapsed:.0f}s")

    # Truncate to OUTPUT_DIM if needed
    if vectors.shape[1] > OUTPUT_DIM:
        print(f"  truncating {vectors.shape[1]}d -> {OUTPUT_DIM}d (Matryoshka)")
        vectors = vectors[:, :OUTPUT_DIM]
        # Re-normalize after truncation
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vectors = vectors / norms

    print(f"  final shape: {vectors.shape}")

    # Mean-center to fix anisotropy (transformer embeddings cluster tightly,
    # making all cosine similarities ~0.95+). Subtracting the corpus mean
    # spreads vectors apart, making cosine discriminative.
    mean_vec = vectors.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    print(f"  mean vector norm before centering: {mean_norm:.4f} (>0.5 = anisotropic)")
    vectors = vectors - mean_vec
    # Re-normalize after centering
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vectors = vectors / norms
    mean_after = np.linalg.norm(vectors.mean(axis=0))
    print(f"  mean vector norm after centering: {mean_after:.6f}")
    print()

    # ── Step 4: Simulated attention ──
    if not args.skip_attention:
        print(f"step 4: simulated attention (K={SIM_ATTENTION_K}, "
              f"iters={SIM_ATTENTION_ITERS}, alpha={SIM_ATTENTION_ALPHA})...")
        t0 = time.time()
        vectors = simulated_attention(
            vectors, SIM_ATTENTION_K, SIM_ATTENTION_ITERS, SIM_ATTENTION_ALPHA
        )
        print(f"  completed in {time.time() - t0:.1f}s")
        print()
    else:
        print("step 4: simulated attention SKIPPED")
        print()

    # ── Step 5: Write output files ──
    print("step 5: writing output files...")
    dim = vectors.shape[1]

    write_bin(str(out_dir / "code_vectors.bin"), vectors, dim)
    write_tokens_txt(str(out_dir / "code_tokens.txt"), filtered_tokens)
    write_tokens_h(str(out_dir / "code_tokens.h"), filtered_tokens)

    # Always use forward slashes: this path is embedded in the .incbin directive,
    # and backslashes (from a Windows run) break the macOS/Linux assembler.
    incbin_path = (out_dir / "code_vectors.bin").as_posix()
    write_vectors_h(str(out_dir / "code_vectors.h"), len(filtered_tokens), dim, incbin_path)
    write_blob_s(str(out_dir / "code_vectors_blob.S"), incbin_path)
    print()

    # Cleanup checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  removed checkpoint: {checkpoint_path}")

    # ── Summary ──
    bin_size = os.path.getsize(str(out_dir / "code_vectors.bin"))
    print()
    print("=" * 60)
    print(f"  model:      {MODEL_NAME}")
    print(f"  tokens:     {len(filtered_tokens)}")
    print(f"  dimensions: {dim}")
    print(f"  blob size:  {bin_size / (1024*1024):.1f} MB")
    print(f"  sim-attn:   {'yes' if not args.skip_attention else 'no'}")
    print(f"  output:     {out_dir}/")
    print("=" * 60)
    print()
    print("next steps:")
    print("  1. commit the generated files in src/embed/")
    print("  2. rebuild: make clean && make")


if __name__ == "__main__":
    main()
