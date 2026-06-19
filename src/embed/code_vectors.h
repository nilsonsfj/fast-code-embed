/* nomic-embed-code (nomic-ai/nomic-embed-code) token embeddings.
 * 40892 tokens x 768d int8-quantized unit vectors.
 * Distilled from 7B model via full inference on filtered vocabulary.
 * Simulated attention: 3 iterations, K=32, alpha=0.3.
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

#define FCE_PRETRAINED_TOKEN_COUNT 40892
#define FCE_PRETRAINED_DIM 768

/* Raw vector blob: first 8 bytes = [int32 count][int32 dim],
 * then count x dim int8 values (unit-normalized, x127 scaled). */
extern const unsigned char FCE_PRETRAINED_VECTOR_BLOB[];
extern const unsigned int FCE_PRETRAINED_VECTOR_BLOB_LEN;

/* Access the int8 vector for token index i. */
static inline const int8_t *fce_pretrained_vec_at(int i) {
    return (const int8_t *)(FCE_PRETRAINED_VECTOR_BLOB + 8 + (size_t)i * FCE_PRETRAINED_DIM);
}

/* Token strings (separate header to keep this file clean). */
#include "code_tokens.h"

#endif /* FCE_EMBED_VECTORS_H */
