#!/usr/bin/env python3
"""compute_pca_matrix.py — Compute PCA projection matrix for 768→256 dim reduction.

Reads the pretrained int8 vector blob, computes top-256 eigenvectors of the
covariance matrix, and writes a C header with the projection matrix.

Usage: python3 scripts/compute_pca_matrix.py src/embed/code_vectors.bin > src/embed/pca_projection.h
"""
import struct
import sys
import numpy as np

def load_vectors(path):
    with open(path, 'rb') as f:
        header = f.read(8)
        count, dim = struct.unpack('<ii', header)
        raw = f.read(count * dim)
        assert len(raw) == count * dim
        vecs = np.frombuffer(raw, dtype=np.int8).reshape(count, dim).astype(np.float32)
        return vecs / 127.0  # dequantize to [-1, 1]

def compute_pca(X, n_components):
    """Compute top-n_components PCA projection matrix."""
    # Center data
    mean = X.mean(axis=0)
    X_c = X - mean

    # Covariance matrix (d x d)
    C = (X_c.T @ X_c) / (X_c.shape[0] - 1)

    # Eigendecomposition via SVD (more numerically stable)
    # For symmetric C, SVD == eigendecomposition
    U, S, Vt = np.linalg.svd(C, full_matrices=False)

    # Top n_components eigenvectors (rows of Vt, or equivalently columns of V)
    P = Vt[:n_components].T  # (d, n_components)

    # Verify: explained variance ratio
    total_var = S.sum()
    explained = S[:n_components].sum() / total_var
    print(f"PCA: {n_components}/{X.shape[1]} dims, explained variance: {explained:.4f} ({explained*100:.1f}%)", file=sys.stderr)

    return P, mean

def write_header(P, mean, out_path):
    d, k = P.shape
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("/* pca_projection.h — Auto-generated PCA projection matrix.\n")
        f.write(f" * {d}-dim → {k}-dim projection for FCE_SEM_DIM_256 mode.\n")
        f.write(" * Computed from nomic-embed-code pretrained vectors (40856 tokens).\n")
        f.write(" * Run: python3 scripts/compute_pca_matrix.py src/embed/code_vectors.bin\n")
        f.write(" */\n")
        f.write("#ifndef FCE_PCA_PROJECTION_H\n")
        f.write("#define FCE_PCA_PROJECTION_H\n\n")

        # Projection matrix P (d x k) stored row-major
        f.write(f"/* {d} x {k} projection matrix, row-major. P[768][256]. */\n")
        f.write(f"static const float fce_pca_proj[{d}][{k}] = {{\n")
        for i in range(d):
            f.write("    {")
            for j in range(k):
                f.write(f"{P[i,j]:.8e}f")
                if j < k - 1:
                    f.write(", ")
            f.write("},\n")
        f.write("};\n\n")

        # Mean vector (used for centering before projection)
        f.write(f"/* Mean vector for centering ({d} floats). */\n")
        f.write(f"static const float fce_pca_mean[{d}] = {{\n")
        for i in range(0, d, 8):
            chunk = mean[i:i+8]
            f.write("    " + ", ".join(f"{v:.8e}f" for v in chunk) + ",\n")
        f.write("};\n\n")

        f.write("#endif /* FCE_PCA_PROJECTION_H */\n")

if __name__ == "__main__":
    blob_path = sys.argv[1] if len(sys.argv) > 1 else "src/embed/code_vectors.bin"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "src/embed/pca_projection.h"

    print(f"Loading vectors from {blob_path}...", file=sys.stderr)
    X = load_vectors(blob_path)
    print(f"Loaded {X.shape[0]} vectors x {X.shape[1]} dims", file=sys.stderr)

    print("Computing PCA...", file=sys.stderr)
    P, mean = compute_pca(X, 256)

    print(f"Writing {out_path}...", file=sys.stderr)
    write_header(P, mean, out_path)
    print("Done.", file=sys.stderr)
