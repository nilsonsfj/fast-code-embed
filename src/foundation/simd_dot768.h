/* simd_dot768.h — SIMD-accelerated dot products (768 or 256 dim).
 *
 * Variants:
 * fce_dot768(a, b) — pure dot product (float32)
 * fce_dot768_mags3(a, b, &dot, &ma, &mb) — dot + both magnitudes (for cosine)
 * fce_dot768_add_mag_b(a, b, &dot, &mb) — dot + b magnitude (for precomputed a mag)
 * fce_dot768_i8(a, b) — int8 dot product (for P0 quantized scan)
 * fce_quantize_f32_768(out, src) — float32 → int8 quantization
 *
 * AVX2 on x86_64 (runtime dispatch via __builtin_cpu_supports).
 * NEON on ARM64 (always available, compile-time #ifdef).
 * Scalar fallback for everything else.
 *
 * FCE_SIMD_DIM is 768 by default (nomic-embed-code). Compile with
 * -DFCE_SEM_DIM_256 to use 256 dims (saves ~640 MB memory).
 * Both 768 and 256 are divisible by 8 (AVX2) and 4 (NEON). */

#ifndef FCE_SIMD_DOT768_H
#define FCE_SIMD_DOT768_H

#include <stdint.h>
#include <math.h>
#include <fenv.h>

enum {
#ifdef FCE_SEM_DIM_256
    FCE_SIMD_DIM = 256
#else
    FCE_SIMD_DIM = 768
#endif
};

/* L-6: the AVX2 quantizer
 * (fce_quantize_f32_768_avx2) strides 32 floats per iteration. Both
 * supported dims (768, 256) satisfy this; a future dim that satisfies the
 * documented "divisible by 8" but not 32 would walk off the buffer. */
_Static_assert(FCE_SIMD_DIM % 32 == 0,
               "FCE_SIMD_DIM must be divisible by 32 for AVX2 quantizer stride");

/* ── AVX2 (x86_64) ─────────────────────────────────────────────── */

#if defined(__x86_64__) || defined(_M_X64)

#if defined(__AVX2__)
#define FCE_HAS_AVX2 1
#elif defined(__GNUC__) || defined(__clang__)
#define FCE_HAS_AVX2 1 /* runtime check via __builtin_cpu_supports */
#else
#define FCE_HAS_AVX2 0
#endif

/* Each AVX2 kernel carries a function-level target attribute so the TU
 * can be compiled at baseline x86-64 (no -mavx2) and the runtime
 * __builtin_cpu_supports("avx2") guard is genuinely effective. */

#if FCE_HAS_AVX2
#include <immintrin.h>

static inline int fce_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return __builtin_cpu_supports("avx2");
#endif
}

__attribute__((target("avx2,fma"))) static inline float fce_dot768_avx2(const float *restrict a,
                                                                        const float *restrict b) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    for (int i = 0; i < FCE_SIMD_DIM; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                               _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                               _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                               _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                               _mm256_loadu_ps(b + i + 24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma"))) static inline void fce_dot768_mags3_avx2(const float *restrict a,
                                                                             const float *restrict b,
                                                                             float *out_dot, float *out_ma,
                                                                             float *out_mb) {
    __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps();
    __m256 d2 = _mm256_setzero_ps(), d3 = _mm256_setzero_ps();
    __m256 ma0 = _mm256_setzero_ps(), ma1 = _mm256_setzero_ps();
    __m256 ma2 = _mm256_setzero_ps(), ma3 = _mm256_setzero_ps();
    __m256 mb0 = _mm256_setzero_ps(), mb1 = _mm256_setzero_ps();
    __m256 mb2 = _mm256_setzero_ps(), mb3 = _mm256_setzero_ps();
    for (int i = 0; i < FCE_SIMD_DIM; i += 32) {
        __m256 ai0 = _mm256_loadu_ps(a + i), bi0 = _mm256_loadu_ps(b + i);
        __m256 ai1 = _mm256_loadu_ps(a + i + 8), bi1 = _mm256_loadu_ps(b + i + 8);
        __m256 ai2 = _mm256_loadu_ps(a + i + 16), bi2 = _mm256_loadu_ps(b + i + 16);
        __m256 ai3 = _mm256_loadu_ps(a + i + 24), bi3 = _mm256_loadu_ps(b + i + 24);
        d0 = _mm256_fmadd_ps(ai0, bi0, d0);
        d1 = _mm256_fmadd_ps(ai1, bi1, d1);
        d2 = _mm256_fmadd_ps(ai2, bi2, d2);
        d3 = _mm256_fmadd_ps(ai3, bi3, d3);
        ma0 = _mm256_fmadd_ps(ai0, ai0, ma0);
        ma1 = _mm256_fmadd_ps(ai1, ai1, ma1);
        ma2 = _mm256_fmadd_ps(ai2, ai2, ma2);
        ma3 = _mm256_fmadd_ps(ai3, ai3, ma3);
        mb0 = _mm256_fmadd_ps(bi0, bi0, mb0);
        mb1 = _mm256_fmadd_ps(bi1, bi1, mb1);
        mb2 = _mm256_fmadd_ps(bi2, bi2, mb2);
        mb3 = _mm256_fmadd_ps(bi3, bi3, mb3);
    }
    d0 = _mm256_add_ps(_mm256_add_ps(d0, d1), _mm256_add_ps(d2, d3));
    ma0 = _mm256_add_ps(_mm256_add_ps(ma0, ma1), _mm256_add_ps(ma2, ma3));
    mb0 = _mm256_add_ps(_mm256_add_ps(mb0, mb1), _mm256_add_ps(mb2, mb3));
    __m128 dl = _mm256_castps256_ps128(d0), dh = _mm256_extractf128_ps(d0, 1);
    __m128 al = _mm256_castps256_ps128(ma0), ah = _mm256_extractf128_ps(ma0, 1);
    __m128 bl = _mm256_castps256_ps128(mb0), bh = _mm256_extractf128_ps(mb0, 1);
    __m128 ds = _mm_add_ps(dl, dh), as_ = _mm_add_ps(al, ah), bs = _mm_add_ps(bl, bh);
    ds = _mm_hadd_ps(ds, ds);
    ds = _mm_hadd_ps(ds, ds);
    as_ = _mm_hadd_ps(as_, as_);
    as_ = _mm_hadd_ps(as_, as_);
    bs = _mm_hadd_ps(bs, bs);
    bs = _mm_hadd_ps(bs, bs);
    *out_dot = _mm_cvtss_f32(ds);
    *out_ma = _mm_cvtss_f32(as_);
    *out_mb = _mm_cvtss_f32(bs);
}

__attribute__((target("avx2,fma"))) static inline void fce_dot768_add_mag_b_avx2(const float *restrict a,
                                                                                 const float *restrict b,
                                                                                 float *out_dot, float *out_mb) {
    __m256 d0 = _mm256_setzero_ps(), d1 = _mm256_setzero_ps();
    __m256 d2 = _mm256_setzero_ps(), d3 = _mm256_setzero_ps();
    __m256 m0 = _mm256_setzero_ps(), m1 = _mm256_setzero_ps();
    __m256 m2 = _mm256_setzero_ps(), m3 = _mm256_setzero_ps();
    for (int i = 0; i < FCE_SIMD_DIM; i += 32) {
        __m256 ai0 = _mm256_loadu_ps(a + i), bi0 = _mm256_loadu_ps(b + i);
        __m256 ai1 = _mm256_loadu_ps(a + i + 8), bi1 = _mm256_loadu_ps(b + i + 8);
        __m256 ai2 = _mm256_loadu_ps(a + i + 16), bi2 = _mm256_loadu_ps(b + i + 16);
        __m256 ai3 = _mm256_loadu_ps(a + i + 24), bi3 = _mm256_loadu_ps(b + i + 24);
        d0 = _mm256_fmadd_ps(ai0, bi0, d0);
        d1 = _mm256_fmadd_ps(ai1, bi1, d1);
        d2 = _mm256_fmadd_ps(ai2, bi2, d2);
        d3 = _mm256_fmadd_ps(ai3, bi3, d3);
        m0 = _mm256_fmadd_ps(bi0, bi0, m0);
        m1 = _mm256_fmadd_ps(bi1, bi1, m1);
        m2 = _mm256_fmadd_ps(bi2, bi2, m2);
        m3 = _mm256_fmadd_ps(bi3, bi3, m3);
    }
    d0 = _mm256_add_ps(_mm256_add_ps(d0, d1), _mm256_add_ps(d2, d3));
    m0 = _mm256_add_ps(_mm256_add_ps(m0, m1), _mm256_add_ps(m2, m3));
    __m128 dl = _mm256_castps256_ps128(d0), dh = _mm256_extractf128_ps(d0, 1);
    __m128 ml = _mm256_castps256_ps128(m0), mh = _mm256_extractf128_ps(m0, 1);
    __m128 ds = _mm_add_ps(dl, dh), ms = _mm_add_ps(ml, mh);
    ds = _mm_hadd_ps(ds, ds);
    ds = _mm_hadd_ps(ds, ds);
    ms = _mm_hadd_ps(ms, ms);
    ms = _mm_hadd_ps(ms, ms);
    *out_dot = _mm_cvtss_f32(ds);
    *out_mb = _mm_cvtss_f32(ms);
}

/* ── AXPY kernels (P6/P7): dst[i] += scale * src[i] ─────────────── */

__attribute__((target("avx2,fma"))) static inline void fce_axpy_f32_768_avx2(float *dst, const float *src, float scale) {
    __m256 s = _mm256_set1_ps(scale);
    for (int i = 0; i < FCE_SIMD_DIM; i += 32) {
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(s, _mm256_loadu_ps(src + i), _mm256_loadu_ps(dst + i)));
        _mm256_storeu_ps(dst + i + 8, _mm256_fmadd_ps(s, _mm256_loadu_ps(src + i + 8), _mm256_loadu_ps(dst + i + 8)));
        _mm256_storeu_ps(dst + i + 16, _mm256_fmadd_ps(s, _mm256_loadu_ps(src + i + 16), _mm256_loadu_ps(dst + i + 16)));
        _mm256_storeu_ps(dst + i + 24, _mm256_fmadd_ps(s, _mm256_loadu_ps(src + i + 24), _mm256_loadu_ps(dst + i + 24)));
    }
}

__attribute__((target("avx2,fma"))) static inline void fce_axpy_i8_768_avx2(float *dst, const int8_t *src, float scale) {
    __m256 s = _mm256_set1_ps(scale);
    for (int i = 0; i < FCE_SIMD_DIM; i += 32) {
        __m128i lo_i8 = _mm_loadl_epi64((const __m128i *)(src + i));
        __m128i hi_i8 = _mm_loadl_epi64((const __m128i *)(src + i + 8));
        __m256i lo_i32 = _mm256_cvtepi8_epi32(lo_i8);
        __m256i hi_i32 = _mm256_cvtepi8_epi32(hi_i8);
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(lo_i32), _mm256_loadu_ps(dst + i)));
        _mm256_storeu_ps(dst + i + 8, _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(hi_i32), _mm256_loadu_ps(dst + i + 8)));
        __m128i lo2_i8 = _mm_loadl_epi64((const __m128i *)(src + i + 16));
        __m128i hi2_i8 = _mm_loadl_epi64((const __m128i *)(src + i + 24));
        __m256i lo2_i32 = _mm256_cvtepi8_epi32(lo2_i8);
        __m256i hi2_i32 = _mm256_cvtepi8_epi32(hi2_i8);
        _mm256_storeu_ps(dst + i + 16, _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(lo2_i32), _mm256_loadu_ps(dst + i + 16)));
        _mm256_storeu_ps(dst + i + 24, _mm256_fmadd_ps(s, _mm256_cvtepi32_ps(hi2_i32), _mm256_loadu_ps(dst + i + 24)));
    }
}

__attribute__((target("avx2,fma"))) static inline void fce_init_f32_from_i8_768_avx2(float *dst, const int8_t *src, float scale) {
    __m256 s = _mm256_set1_ps(scale);
    for (int i = 0; i < FCE_SIMD_DIM; i += 32) {
        __m128i lo_i8 = _mm_loadl_epi64((const __m128i *)(src + i));
        __m128i hi_i8 = _mm_loadl_epi64((const __m128i *)(src + i + 8));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(s, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_i8))));
        _mm256_storeu_ps(dst + i + 8, _mm256_mul_ps(s, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_i8))));
        __m128i lo2_i8 = _mm_loadl_epi64((const __m128i *)(src + i + 16));
        __m128i hi2_i8 = _mm_loadl_epi64((const __m128i *)(src + i + 24));
        _mm256_storeu_ps(dst + i + 16, _mm256_mul_ps(s, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo2_i8))));
        _mm256_storeu_ps(dst + i + 24, _mm256_mul_ps(s, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi2_i8))));
    }
}

/* ── P0: int8 dot product + quantization ───────────────────────── */

__attribute__((target("avx2,fma"))) static inline int32_t fce_dot768_i8_avx2(const int8_t *restrict a,
                                                                             const int8_t *restrict b) {
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < FCE_SIMD_DIM; i += 32) {
        __m256i ai = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i bi = _mm256_loadu_si256((const __m256i *)(b + i));
        /* Sign-extend int8 → int16, multiply-accumulate adjacent pairs → int32. */
        __m128i ai_lo = _mm256_castsi256_si128(ai);
        __m128i ai_hi = _mm256_extracti128_si256(ai, 1);
        __m128i bi_lo = _mm256_castsi256_si128(bi);
        __m128i bi_hi = _mm256_extracti128_si256(bi, 1);
        __m256i a_lo16 = _mm256_cvtepi8_epi16(ai_lo);
        __m256i a_hi16 = _mm256_cvtepi8_epi16(ai_hi);
        __m256i b_lo16 = _mm256_cvtepi8_epi16(bi_lo);
        __m256i b_hi16 = _mm256_cvtepi8_epi16(bi_hi);
        __m256i p_lo = _mm256_madd_epi16(a_lo16, b_lo16);
        __m256i p_hi = _mm256_madd_epi16(a_hi16, b_hi16);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(p_lo, p_hi));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

__attribute__((target("avx2,fma"))) static inline void fce_quantize_f32_768_avx2(int8_t *restrict out,
                                                                                 const float *restrict src) {
    /* H-2: pin FP rounding mode to FE_TONEAREST so the AVX2
     * path (via _mm256_cvtps_epi32) agrees with the scalar path regardless of
     * the ambient MXCSR. fesetround sets both x87 and SSE rounding on x86.
     * FENV_ACCESS pragma ensures the compiler does not reorder FP ops across
     * the rounding-mode change. */
#pragma STDC FENV_ACCESS ON
    int saved_round = fegetround();
    if (saved_round != FE_TONEAREST) fesetround(FE_TONEAREST);
    __m256 scale = _mm256_set1_ps(127.0f);
    __m256 lo = _mm256_set1_ps(-127.0f);
    __m256 hi = _mm256_set1_ps(127.0f);
    __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < FCE_SIMD_DIM; i += 32) {
        __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(src + i), scale);
        __m256 v1 = _mm256_mul_ps(_mm256_loadu_ps(src + i + 8), scale);
        __m256 v2 = _mm256_mul_ps(_mm256_loadu_ps(src + i + 16), scale);
        __m256 v3 = _mm256_mul_ps(_mm256_loadu_ps(src + i + 24), scale);
        /* M1: sanitize NaN and ±Inf → 0 before clamp to match
         * the scalar path. NaN comparisons are always false, and Inf is ordered,
         * so neither the NaN-only mask nor clamp alone catches both. Use a
         * combined mask: NaN (v!=v) or Inf (|v| == Inf). */
        {
            __m256 nan_mask = _mm256_cmp_ps(v0, v0, _CMP_UNORD_Q);
            __m256 inf_mask = _mm256_or_ps(
                _mm256_cmp_ps(v0, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ),
                _mm256_cmp_ps(v0, _mm256_set1_ps(-INFINITY), _CMP_EQ_OQ));
            v0 = _mm256_blendv_ps(v0, zero, _mm256_or_ps(nan_mask, inf_mask));
        }
        {
            __m256 nan_mask = _mm256_cmp_ps(v1, v1, _CMP_UNORD_Q);
            __m256 inf_mask = _mm256_or_ps(
                _mm256_cmp_ps(v1, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ),
                _mm256_cmp_ps(v1, _mm256_set1_ps(-INFINITY), _CMP_EQ_OQ));
            v1 = _mm256_blendv_ps(v1, zero, _mm256_or_ps(nan_mask, inf_mask));
        }
        {
            __m256 nan_mask = _mm256_cmp_ps(v2, v2, _CMP_UNORD_Q);
            __m256 inf_mask = _mm256_or_ps(
                _mm256_cmp_ps(v2, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ),
                _mm256_cmp_ps(v2, _mm256_set1_ps(-INFINITY), _CMP_EQ_OQ));
            v2 = _mm256_blendv_ps(v2, zero, _mm256_or_ps(nan_mask, inf_mask));
        }
        {
            __m256 nan_mask = _mm256_cmp_ps(v3, v3, _CMP_UNORD_Q);
            __m256 inf_mask = _mm256_or_ps(
                _mm256_cmp_ps(v3, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ),
                _mm256_cmp_ps(v3, _mm256_set1_ps(-INFINITY), _CMP_EQ_OQ));
            v3 = _mm256_blendv_ps(v3, zero, _mm256_or_ps(nan_mask, inf_mask));
        }
        v0 = _mm256_max_ps(_mm256_min_ps(v0, hi), lo);
        v1 = _mm256_max_ps(_mm256_min_ps(v1, hi), lo);
        v2 = _mm256_max_ps(_mm256_min_ps(v2, hi), lo);
        v3 = _mm256_max_ps(_mm256_min_ps(v3, hi), lo);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        __m256i lo16_01 = _mm256_packs_epi32(i0, i1);
        __m256i lo16_23 = _mm256_packs_epi32(i2, i3);
        __m256i i8 = _mm256_packs_epi16(lo16_01, lo16_23);
        /* packs operates on 128-bit lanes — permute to correct order. */
        i8 = _mm256_permutevar8x32_epi32(i8,
                                         _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i *)(out + i), i8);
    }
    if (saved_round != FE_TONEAREST) fesetround(saved_round);
}
#pragma STDC FENV_ACCESS OFF

#endif /* FCE_HAS_AVX2 */
#endif /* x86_64 */

/* ── NEON (ARM64) ──────────────────────────────────────────────── */

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define FCE_HAS_NEON 1
#include <arm_neon.h>

static inline float fce_dot768_neon(const float *restrict a,
                                    const float *restrict b) {
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    float32x4_t acc2 = vdupq_n_f32(0), acc3 = vdupq_n_f32(0);
    for (int i = 0; i < FCE_SIMD_DIM; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    float32x2_t sum = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
}

static inline void fce_dot768_mags3_neon(const float *restrict a,
                                         const float *restrict b,
                                         float *out_dot, float *out_ma,
                                         float *out_mb) {
    float32x4_t d0 = vdupq_n_f32(0), d1 = vdupq_n_f32(0);
    float32x4_t d2 = vdupq_n_f32(0), d3 = vdupq_n_f32(0);
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0);
    float32x4_t b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);
    for (int i = 0; i < FCE_SIMD_DIM; i += 16) {
        float32x4_t ai0 = vld1q_f32(a + i), bi0 = vld1q_f32(b + i);
        float32x4_t ai1 = vld1q_f32(a + i + 4), bi1 = vld1q_f32(b + i + 4);
        float32x4_t ai2 = vld1q_f32(a + i + 8), bi2 = vld1q_f32(b + i + 8);
        float32x4_t ai3 = vld1q_f32(a + i + 12), bi3 = vld1q_f32(b + i + 12);
        d0 = vmlaq_f32(d0, ai0, bi0);
        d1 = vmlaq_f32(d1, ai1, bi1);
        d2 = vmlaq_f32(d2, ai2, bi2);
        d3 = vmlaq_f32(d3, ai3, bi3);
        a0 = vmlaq_f32(a0, ai0, ai0);
        a1 = vmlaq_f32(a1, ai1, ai1);
        a2 = vmlaq_f32(a2, ai2, ai2);
        a3 = vmlaq_f32(a3, ai3, ai3);
        b0 = vmlaq_f32(b0, bi0, bi0);
        b1 = vmlaq_f32(b1, bi1, bi1);
        b2 = vmlaq_f32(b2, bi2, bi2);
        b3 = vmlaq_f32(b3, bi3, bi3);
    }
    d0 = vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3));
    a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    b0 = vaddq_f32(vaddq_f32(b0, b1), vaddq_f32(b2, b3));
    float32x2_t ds = vadd_f32(vget_low_f32(d0), vget_high_f32(d0));
    float32x2_t as_ = vadd_f32(vget_low_f32(a0), vget_high_f32(a0));
    float32x2_t bs = vadd_f32(vget_low_f32(b0), vget_high_f32(b0));
    ds = vpadd_f32(ds, ds);
    as_ = vpadd_f32(as_, as_);
    bs = vpadd_f32(bs, bs);
    *out_dot = vget_lane_f32(ds, 0);
    *out_ma = vget_lane_f32(as_, 0);
    *out_mb = vget_lane_f32(bs, 0);
}

static inline void fce_dot768_add_mag_b_neon(const float *restrict a,
                                             const float *restrict b,
                                             float *out_dot, float *out_mb) {
    float32x4_t d0 = vdupq_n_f32(0), d1 = vdupq_n_f32(0);
    float32x4_t d2 = vdupq_n_f32(0), d3 = vdupq_n_f32(0);
    float32x4_t m0 = vdupq_n_f32(0), m1 = vdupq_n_f32(0);
    float32x4_t m2 = vdupq_n_f32(0), m3 = vdupq_n_f32(0);
    for (int i = 0; i < FCE_SIMD_DIM; i += 16) {
        float32x4_t ai0 = vld1q_f32(a + i), bi0 = vld1q_f32(b + i);
        float32x4_t ai1 = vld1q_f32(a + i + 4), bi1 = vld1q_f32(b + i + 4);
        float32x4_t ai2 = vld1q_f32(a + i + 8), bi2 = vld1q_f32(b + i + 8);
        float32x4_t ai3 = vld1q_f32(a + i + 12), bi3 = vld1q_f32(b + i + 12);
        d0 = vmlaq_f32(d0, ai0, bi0);
        d1 = vmlaq_f32(d1, ai1, bi1);
        d2 = vmlaq_f32(d2, ai2, bi2);
        d3 = vmlaq_f32(d3, ai3, bi3);
        m0 = vmlaq_f32(m0, bi0, bi0);
        m1 = vmlaq_f32(m1, bi1, bi1);
        m2 = vmlaq_f32(m2, bi2, bi2);
        m3 = vmlaq_f32(m3, bi3, bi3);
    }
    d0 = vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3));
    m0 = vaddq_f32(vaddq_f32(m0, m1), vaddq_f32(m2, m3));
    float32x2_t ds = vadd_f32(vget_low_f32(d0), vget_high_f32(d0));
    float32x2_t ms = vadd_f32(vget_low_f32(m0), vget_high_f32(m0));
    ds = vpadd_f32(ds, ds);
    ms = vpadd_f32(ms, ms);
    *out_dot = vget_lane_f32(ds, 0);
    *out_mb = vget_lane_f32(ms, 0);
}

/* ── AXPY kernels (P6/P7): dst[i] += scale * src[i] ─────────────── */

static inline void fce_axpy_f32_768_neon(float *dst, const float *src, float scale) {
    float32x4_t s = vdupq_n_f32(scale);
    for (int i = 0; i < FCE_SIMD_DIM; i += 16) {
        vst1q_f32(dst + i, vmlaq_f32(vld1q_f32(dst + i), s, vld1q_f32(src + i)));
        vst1q_f32(dst + i + 4, vmlaq_f32(vld1q_f32(dst + i + 4), s, vld1q_f32(src + i + 4)));
        vst1q_f32(dst + i + 8, vmlaq_f32(vld1q_f32(dst + i + 8), s, vld1q_f32(src + i + 8)));
        vst1q_f32(dst + i + 12, vmlaq_f32(vld1q_f32(dst + i + 12), s, vld1q_f32(src + i + 12)));
    }
}

static inline void fce_axpy_i8_768_neon(float *dst, const int8_t *src, float scale) {
    float32x4_t s = vdupq_n_f32(scale);
    for (int i = 0; i < FCE_SIMD_DIM; i += 16) {
        int8x8_t i8_lo = vld1_s8(src + i);
        int8x8_t i8_hi = vld1_s8(src + i + 8);
        vst1q_f32(dst + i, vmlaq_f32(vld1q_f32(dst + i), s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(i8_lo))))));
        vst1q_f32(dst + i + 4, vmlaq_f32(vld1q_f32(dst + i + 4), s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(i8_lo))))));
        vst1q_f32(dst + i + 8, vmlaq_f32(vld1q_f32(dst + i + 8), s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(i8_hi))))));
        vst1q_f32(dst + i + 12, vmlaq_f32(vld1q_f32(dst + i + 12), s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(i8_hi))))));
    }
}

/* ── P0: int8 dot product + quantization ───────────────────────── */

static inline int32_t fce_dot768_i8_neon(const int8_t *restrict a,
                                         const int8_t *restrict b) {
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < FCE_SIMD_DIM; i += 16) {
        int8x16_t ai = vld1q_s8(a + i);
        int8x16_t bi = vld1q_s8(b + i);
#ifdef __ARM_FEATURE_DOTPROD
        /* ARMv8.2+ DotProd: single instruction, fastest path. */
        acc = vdotq_s32(acc, ai, bi);
#else
        /* Portable fallback for ARMv8.0 without DotProd extension. */
        int16x8_t lo = vmull_s8(vget_low_s8(ai), vget_low_s8(bi));
        int16x8_t hi = vmull_s8(vget_high_s8(ai), vget_high_s8(bi));
        acc = vpadalq_s16(acc, lo);
        acc = vpadalq_s16(acc, hi);
#endif
    }
    return vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) +
           vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
}

static inline void fce_quantize_f32_768_neon(int8_t *restrict out,
                                             const float *restrict src) {
    float32x4_t scale = vdupq_n_f32(127.0f);
    float32x4_t lo = vdupq_n_f32(-127.0f);
    float32x4_t hi = vdupq_n_f32(127.0f);
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (int i = 0; i < FCE_SIMD_DIM; i += 16) {
        float32x4_t v0 = vmulq_f32(vld1q_f32(src + i), scale);
        float32x4_t v1 = vmulq_f32(vld1q_f32(src + i + 4), scale);
        float32x4_t v2 = vmulq_f32(vld1q_f32(src + i + 8), scale);
        float32x4_t v3 = vmulq_f32(vld1q_f32(src + i + 12), scale);
        /* M1: sanitize NaN and ±Inf → 0 before clamp to match
         * the scalar path. vceqq_f32(v,v) yields 0 for NaN (NaN != NaN) and
         * all-ones for finite/Inf. Then mask out ±Inf via comparison against
         * INFINITY / -INFINITY constants. */
        {
            uint32x4_t fin_mask = vceqq_f32(v0, v0);
            uint32x4_t inf_mask = vorrq_u32(
                vceqq_f32(v0, vdupq_n_f32(INFINITY)),
                vceqq_f32(v0, vdupq_n_f32(-INFINITY)));
            v0 = vbslq_f32(vandq_u32(vorrq_u32(fin_mask, inf_mask),
                                     vmvnq_u32(inf_mask)),
                           v0, zero);
        }
        {
            uint32x4_t fin_mask = vceqq_f32(v1, v1);
            uint32x4_t inf_mask = vorrq_u32(
                vceqq_f32(v1, vdupq_n_f32(INFINITY)),
                vceqq_f32(v1, vdupq_n_f32(-INFINITY)));
            v1 = vbslq_f32(vandq_u32(vorrq_u32(fin_mask, inf_mask),
                                     vmvnq_u32(inf_mask)),
                           v1, zero);
        }
        {
            uint32x4_t fin_mask = vceqq_f32(v2, v2);
            uint32x4_t inf_mask = vorrq_u32(
                vceqq_f32(v2, vdupq_n_f32(INFINITY)),
                vceqq_f32(v2, vdupq_n_f32(-INFINITY)));
            v2 = vbslq_f32(vandq_u32(vorrq_u32(fin_mask, inf_mask),
                                     vmvnq_u32(inf_mask)),
                           v2, zero);
        }
        {
            uint32x4_t fin_mask = vceqq_f32(v3, v3);
            uint32x4_t inf_mask = vorrq_u32(
                vceqq_f32(v3, vdupq_n_f32(INFINITY)),
                vceqq_f32(v3, vdupq_n_f32(-INFINITY)));
            v3 = vbslq_f32(vandq_u32(vorrq_u32(fin_mask, inf_mask),
                                     vmvnq_u32(inf_mask)),
                           v3, zero);
        }
        v0 = vmaxq_f32(vminq_f32(v0, hi), lo);
        v1 = vmaxq_f32(vminq_f32(v1, hi), lo);
        v2 = vmaxq_f32(vminq_f32(v2, hi), lo);
        v3 = vmaxq_f32(vminq_f32(v3, hi), lo);
        /* M2: use vcvtnq_s32_f32 (FCVTNS, round-to-nearest)
         * to match the scalar nearbyintf and AVX2 _mm256_cvtps_epi32 behavior.
         * The original vcvtq_s32_f32 (FCVTZS) truncates toward zero, producing
         * different int8 values on ARM vs x86 for the same float input. */
        int32x4_t i0 = vcvtnq_s32_f32(v0);
        int32x4_t i1 = vcvtnq_s32_f32(v1);
        int32x4_t i2 = vcvtnq_s32_f32(v2);
        int32x4_t i3 = vcvtnq_s32_f32(v3);
        int16x4_t n0 = vmovn_s32(i0);
        int16x4_t n1 = vmovn_s32(i1);
        int16x4_t n2 = vmovn_s32(i2);
        int16x4_t n3 = vmovn_s32(i3);
        int16x8_t n01 = vcombine_s16(n0, n1);
        int16x8_t n23 = vcombine_s16(n2, n3);
        int8x8_t b0 = vmovn_s16(n01);
        int8x8_t b1 = vmovn_s16(n23);
        vst1_s8(out + i, b0);
        vst1_s8(out + i + 8, b1);
    }
}

static inline void fce_init_f32_from_i8_768_neon(float *dst, const int8_t *src, float scale) {
    float32x4_t s = vdupq_n_f32(scale);
    for (int i = 0; i < FCE_SIMD_DIM; i += 16) {
        int8x8_t i8_lo = vld1_s8(src + i);
        int8x8_t i8_hi = vld1_s8(src + i + 8);
        vst1q_f32(dst + i, vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(i8_lo))))));
        vst1q_f32(dst + i + 4, vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(i8_lo))))));
        vst1q_f32(dst + i + 8, vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(i8_hi))))));
        vst1q_f32(dst + i + 12, vmulq_f32(s, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(i8_hi))))));
    }
}

#endif /* __ARM_NEON */
#endif /* aarch64 */

/* ── Scalar fallback ───────────────────────────────────────────── */

static inline float fce_dot768_scalar(const float *restrict a,
                                      const float *restrict b) {
    float acc = 0.0f;
    for (int i = 0; i < FCE_SIMD_DIM; i++) acc += a[i] * b[i];
    return acc;
}

static inline void fce_dot768_mags3_scalar(const float *restrict a,
                                           const float *restrict b,
                                           float *out_dot, float *out_ma,
                                           float *out_mb) {
    float dot = 0.0f, ma = 0.0f, mb = 0.0f;
    for (int i = 0; i < FCE_SIMD_DIM; i++) {
        dot += a[i] * b[i];
        ma += a[i] * a[i];
        mb += b[i] * b[i];
    }
    *out_dot = dot;
    *out_ma = ma;
    *out_mb = mb;
}

static inline void fce_dot768_add_mag_b_scalar(const float *restrict a,
                                               const float *restrict b,
                                               float *out_dot, float *out_mb) {
    float dot = 0.0f, mb = 0.0f;
    for (int i = 0; i < FCE_SIMD_DIM; i++) {
        dot += a[i] * b[i];
        mb += b[i] * b[i];
    }
    *out_dot = dot;
    *out_mb = mb;
}

static inline void fce_axpy_f32_768_scalar(float *dst, const float *src, float scale) {
    for (int i = 0; i < FCE_SIMD_DIM; i++) dst[i] += scale * src[i];
}

static inline void fce_axpy_i8_768_scalar(float *dst, const int8_t *src, float scale) {
    for (int i = 0; i < FCE_SIMD_DIM; i++) dst[i] += scale * (float)src[i];
}

static inline void fce_init_f32_from_i8_768_scalar(float *dst, const int8_t *src, float scale) {
    for (int i = 0; i < FCE_SIMD_DIM; i++) dst[i] = scale * (float)src[i];
}

static inline int32_t fce_dot768_i8_scalar(const int8_t *restrict a,
                                           const int8_t *restrict b) {
    int32_t acc = 0;
    for (int i = 0; i < FCE_SIMD_DIM; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

static inline void fce_quantize_f32_768_scalar(int8_t *restrict out,
                                               const float *restrict src) {
    /* H-2: pin rounding mode to FE_TONEAREST for the
     * duration of quantization so scalar and SIMD paths agree regardless
     * of the ambient fenv. FENV_ACCESS pragma ensures the compiler does not
     * reorder FP ops across the rounding-mode change. */
#pragma STDC FENV_ACCESS ON
    int saved_round = fegetround();
    if (saved_round != FE_TONEAREST) fesetround(FE_TONEAREST);
    for (int i = 0; i < FCE_SIMD_DIM; i++) {
        float v = src[i] * 127.0f;
        /* C2: sanitize NaN/Inf before clamp.
         * NaN comparisons are always false, so the clamp does not catch NaN,
         * and (int8_t)nearbyintf(NaN) is undefined behavior in C. This
         * guard matches the isfinite sweep in the float32 fallback worker. */
        if (!isfinite(v)) v = 0.0f;
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        /* Review 0001 §5.5: use nearbyintf (round-to-nearest-even per
         * IEEE 754 default rounding), matching NEON vcvtnq_s32_f32 (FCVTNS)
         * and AVX2 _mm256_cvtps_epi32 (round-to-nearest-even via MXCSR).
         * L4: with the mode pinned above, this matches SIMD even if the
         * host process changed the rounding mode. */
        out[i] = (int8_t)nearbyintf(v);
    }
    if (saved_round != FE_TONEAREST) fesetround(saved_round);
}
#pragma STDC FENV_ACCESS OFF

/* ── Dispatch ──────────────────────────────────────────────────── */

static inline float fce_dot768(const float *restrict a,
                               const float *restrict b) {
#if FCE_HAS_AVX2
    if (fce_has_avx2()) return fce_dot768_avx2(a, b);
#elif FCE_HAS_NEON
    return fce_dot768_neon(a, b);
#endif
    return fce_dot768_scalar(a, b);
}

static inline void fce_dot768_mags3(const float *restrict a,
                                    const float *restrict b,
                                    float *out_dot, float *out_ma,
                                    float *out_mb) {
#if FCE_HAS_AVX2
    if (fce_has_avx2()) {
        fce_dot768_mags3_avx2(a, b, out_dot, out_ma, out_mb);
        return;
    }
#elif FCE_HAS_NEON
    fce_dot768_mags3_neon(a, b, out_dot, out_ma, out_mb);
    return;
#endif
    fce_dot768_mags3_scalar(a, b, out_dot, out_ma, out_mb);
}

static inline void fce_dot768_add_mag_b(const float *restrict a,
                                        const float *restrict b,
                                        float *out_dot, float *out_mb) {
#if FCE_HAS_AVX2
    if (fce_has_avx2()) {
        fce_dot768_add_mag_b_avx2(a, b, out_dot, out_mb);
        return;
    }
#elif FCE_HAS_NEON
    fce_dot768_add_mag_b_neon(a, b, out_dot, out_mb);
    return;
#endif
    fce_dot768_add_mag_b_scalar(a, b, out_dot, out_mb);
}

static inline void fce_axpy_f32_768(float *dst, const float *src, float scale) {
#if FCE_HAS_AVX2
    if (fce_has_avx2()) {
        fce_axpy_f32_768_avx2(dst, src, scale);
        return;
    }
#elif FCE_HAS_NEON
    fce_axpy_f32_768_neon(dst, src, scale);
    return;
#endif
    fce_axpy_f32_768_scalar(dst, src, scale);
}

static inline void fce_axpy_i8_768(float *dst, const int8_t *src, float scale) {
#if FCE_HAS_AVX2
    if (fce_has_avx2()) {
        fce_axpy_i8_768_avx2(dst, src, scale);
        return;
    }
#elif FCE_HAS_NEON
    fce_axpy_i8_768_neon(dst, src, scale);
    return;
#endif
    fce_axpy_i8_768_scalar(dst, src, scale);
}

static inline void fce_init_f32_from_i8_768(float *dst, const int8_t *src, float scale) {
#if FCE_HAS_AVX2
    if (fce_has_avx2()) {
        fce_init_f32_from_i8_768_avx2(dst, src, scale);
        return;
    }
#elif FCE_HAS_NEON
    fce_init_f32_from_i8_768_neon(dst, src, scale);
    return;
#endif
    fce_init_f32_from_i8_768_scalar(dst, src, scale);
}

static inline int32_t fce_dot768_i8(const int8_t *restrict a,
                                    const int8_t *restrict b) {
#if FCE_HAS_AVX2
    if (fce_has_avx2()) return fce_dot768_i8_avx2(a, b);
#elif FCE_HAS_NEON
    return fce_dot768_i8_neon(a, b);
#endif
    return fce_dot768_i8_scalar(a, b);
}

static inline void fce_quantize_f32_768(int8_t *restrict out,
                                        const float *restrict src) {
#if FCE_HAS_AVX2
    if (fce_has_avx2()) {
        fce_quantize_f32_768_avx2(out, src);
        return;
    }
#elif FCE_HAS_NEON
    fce_quantize_f32_768_neon(out, src);
    return;
#endif
    fce_quantize_f32_768_scalar(out, src);
}

#endif /* FCE_SIMD_DOT768_H */
