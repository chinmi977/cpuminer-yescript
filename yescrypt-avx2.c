/*-
 * Copyright 2009 Colin Percival
 * Copyright 2012-2014 Alexander Peslyak
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 */

/*
 * On 64-bit, enabling SSE4.1 helps our pwxform code indirectly, via avoiding
 * gcc bug 54349 (fixed for gcc 4.9+).  On 32-bit, it's of direct help.  AVX
 * and XOP are of further help either way.
 */
#ifndef __AVX2__
#warning "Consider enabling SSE4.1, AVX, or XOP in the C compiler for significantly better performance"
#endif

#include <immintrin.h>

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.h"
#include "sysendian.h"

#include "yescrypt.h"

#include "yescrypt-platform.c"

#if __STDC_VERSION__ >= 199901L
/* have restrict */
#elif defined(__GNUC__)
#define restrict __restrict
#else
#define restrict
#endif

#define PREFETCH_AVX2(x, hint) _mm_prefetch((const char *)(x), (hint));
#define PREFETCH_AVX2_OUT(x, hint) /* disabled */

#define ARX_AVX2(out, in1, in2, s) \
	{ \
		__m128i T = _mm_add_epi32(in1, in2); \
		out = _mm_xor_si128(out, _mm_slli_epi32(T, s)); \
		out = _mm_xor_si128(out, _mm_srli_epi32(T, 32-s)); \
	}

#define SALSA20_2ROUNDS_AVX2 \
	/* Operate on "columns" */ \
	ARX_AVX2(X0.xmm[1], X0.xmm[0], X1.xmm[1], 7) \
	ARX_AVX2(X1.xmm[0], X0.xmm[1], X0.xmm[0], 9) \
	ARX_AVX2(X1.xmm[1], X1.xmm[0], X0.xmm[1], 13) \
	ARX_AVX2(X0.xmm[0], X1.xmm[1], X1.xmm[0], 18) \
\
	/* Rearrange data */ \
	X0.xmm[1] = _mm_shuffle_epi32(X1.xmm[1], 0x93); \
	X1.xmm[0] = _mm_shuffle_epi32(X1.xmm[0], 0x4E); \
	X1.xmm[1] = _mm_shuffle_epi32(X1.xmm[1], 0x39); \
\
	/* Operate on "rows" */ \
	ARX_AVX2(X1.xmm[1], X0.xmm[0], X0.xmm[1], 7) \
	ARX_AVX2(X1.xmm[0], X1.xmm[1], X0.xmm[0], 9) \
	ARX_AVX2(X0.xmm[1], X1.xmm[0], X1.xmm[1], 13) \
	ARX_AVX2(X0.xmm[0], X0.xmm[1], X1.xmm[0], 18) \
\
	/* Rearrange data */ \
	X0.xmm[1] = _mm_shuffle_epi32(X0.xmm[1], 0x39); \
	X1.xmm[0] = _mm_shuffle_epi32(X1.xmm[0], 0x4E); \
	X1.xmm[1] = _mm_shuffle_epi32(X1.xmm[1], 0x93);

#define SALSA20_8_BASE_AVX2(maybe_decl, out) \
	{ \
		maybe_decl Y0; \
		Y0.xmm[0] = X0.xmm[0]; \
		maybe_decl Y1; \
		Y1.xmm[0] = X0.xmm[1]; \
		maybe_decl Y2; \
		Y2.xmm[0] = X1.xmm[0]; \
		maybe_decl Y3; \
		Y3.xmm[0] = X1.xmm[1]; \
		SALSA20_2ROUNDS_AVX2 \
		SALSA20_2ROUNDS_AVX2 \
		SALSA20_2ROUNDS_AVX2 \
		SALSA20_2ROUNDS_AVX2 \
		Y0.xmm[1] = Y1.xmm[0]; \
		Y2.xmm[1] = Y3.xmm[0]; \
		(out)[0].ymm = X0.ymm = _mm256_add_epi64(X0.ymm, Y0.ymm); \
		(out)[1].ymm = X1.ymm = _mm256_add_epi64(X1.ymm, Y2.ymm); \
	}
#define SALSA20_8_AVX2(out) \
	SALSA20_8_BASE_AVX2(avx256i, out)

/**
 * Apply the salsa20/8 core to the block provided in (X0 ... X3) ^ (Z0 ... Z3).
 */
#define SALSA20_8_XOR_ANY_AVX2(maybe_decl, Z0, Z1, out) \
	X0.ymm = _mm256_xor_si256(X0.ymm, Z0.ymm); \
	X1.ymm = _mm256_xor_si256(X1.ymm, Z1.ymm); \
	SALSA20_8_BASE_AVX2(maybe_decl, out)

#define SALSA20_8_XOR_MEM_AVX2(in, out) \
	SALSA20_8_XOR_ANY_AVX2(avx256i, (in)[0], (in)[1], out)

#define SALSA20_8_XOR_REG_AVX2(out) \
	SALSA20_8_XOR_ANY_AVX2(avx256i, Y0, Y1, out)

typedef union {
	uint64_t u64[4];
	uint32_t u32[8];
	uint8_t u8[32];
	__m256i ymm;
	__m128i xmm[2];
} avx256i;

typedef union {
	uint32_t w[16];
	__m128i q[4];
	avx256i dq[2];
} salsa20_blk_t_avx2;

/**
 * blockmix_salsa8_avx2(Bin, Bout, r):
 * Compute Bout = BlockMix_{salsa20/8, r}(Bin).  The input Bin must be 128r
 * bytes in length; the output Bout must also be the same size.
 */
static inline void
blockmix_salsa8_avx2(const salsa20_blk_t_avx2 *restrict Bin,
    salsa20_blk_t_avx2 *restrict Bout, size_t r)
{
	avx256i X0, X1;
	size_t i;

	r--;
	PREFETCH_AVX2(&Bin[r * 2 + 1], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH_AVX2(&Bin[i * 2], _MM_HINT_T0)
		PREFETCH_AVX2_OUT(&Bout[i], _MM_HINT_T0)
		PREFETCH_AVX2(&Bin[i * 2 + 1], _MM_HINT_T0)
		PREFETCH_AVX2_OUT(&Bout[r + 1 + i], _MM_HINT_T0)
	}
	PREFETCH_AVX2(&Bin[r * 2], _MM_HINT_T0)
	PREFETCH_AVX2_OUT(&Bout[r], _MM_HINT_T0)
	PREFETCH_AVX2_OUT(&Bout[r * 2 + 1], _MM_HINT_T0)

	/* 1: X <-- B_{2r - 1} */
	X0 = Bin[r * 2 + 1].dq[0];
	X1 = Bin[r * 2 + 1].dq[1];
	

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	SALSA20_8_XOR_MEM_AVX2(Bin[0].dq, Bout[0].dq)

	/* 2: for i = 0 to 2r - 1 do */
	for (i = 0; i < r;) {
		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		SALSA20_8_XOR_MEM_AVX2(Bin[i * 2 + 1].dq, Bout[r + 1 + i].dq)

		i++;

		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		SALSA20_8_XOR_MEM_AVX2(Bin[i * 2].dq, Bout[i].dq)
	}

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	SALSA20_8_XOR_MEM_AVX2(Bin[r * 2 + 1].dq, Bout[r * 2 + 1].dq)
}

/*
 * (V)PSRLDQ and (V)PSHUFD have higher throughput than (V)PSRLQ on some CPUs
 * starting with Sandy Bridge.  Additionally, PSHUFD uses separate source and
 * destination registers, whereas the shifts would require an extra move
 * instruction for our code when building without AVX.  Unfortunately, PSHUFD
 * is much slower on Conroe (4 cycles latency vs. 1 cycle latency for PSRLQ)
 * and somewhat slower on some non-Intel CPUs (luckily not including AMD
 * Bulldozer and Piledriver).  Since for many other CPUs using (V)PSHUFD is a
 * win in terms of throughput or/and not needing a move instruction, we
 * currently use it despite of the higher latency on some older CPUs.  As an
 * alternative, the #if below may be patched to only enable use of (V)PSHUFD
 * when building with SSE4.1 or newer, which is not available on older CPUs
 * where this instruction has higher latency.
 */
#define HI32(X) \
	_mm256_shuffle_epi32((X), _MM_SHUFFLE(2,3,0,1))

/* This is tunable */
#define S_BITS_AVX2 8

/* Not tunable in this implementation, hard-coded in a few places */
#define S_SIMD_AVX2 2
#define S_P_AVX2 4

/* Number of S-boxes.  Not tunable by design, hard-coded in a few places. */
#define S_N_AVX2 2

/* Derived values.  Not tunable except via S_BITS_AVX2 above. */
#define S_SIZE1_AVX2 (1 << S_BITS_AVX2)
#define S_MASK_AVX2 ((S_SIZE1_AVX2 - 1) * S_SIMD_AVX2 * 8)
#define S_MASK2_AVX2 (((uint64_t)S_MASK_AVX2 << 32) | S_MASK_AVX2)
#define S_SIZE_ALL_AVX2 (S_N_AVX2 * S_SIZE1_AVX2 * S_SIMD_AVX2 * 8)
avx256i YMM_MASK;

#define PWXFORM_AVX2_T avx256i
#define PWXFORM_SIMD_AVX2(X, x, s0, s1) \
	x.ymm = _mm256_and_si256( X.ymm, YMM_MASK.ymm); \
	s0.ymm = _mm256_set_epi64x( \
		*(const uint64_t *)(S0 + x.u32[0]), \
		*(const uint64_t *)(S0 + x.u32[2]), \
		*(const uint64_t *)(S0 + x.u32[4]), \
		*(const uint64_t *)(S0 + x.u32[6])); \
	s1.ymm = _mm256_set_epi64x( \
		*(const uint64_t *)(S0 + x.u32[1]), \
		*(const uint64_t *)(S0 + x.u32[3]), \
		*(const uint64_t *)(S0 + x.u32[5]), \
		*(const uint64_t *)(S0 + x.u32[7])); \
	X.ymm = _mm256_mul_epu32(HI32(X.ymm), X.ymm); \
	X.ymm = _mm256_add_epi64(X.ymm, s0.ymm); \
	X.ymm = _mm256_xor_si256(X.ymm, s1.ymm);

#define PWXFORM_ROUND_AVX2 \
	PWXFORM_SIMD_AVX2(X0, x0, s00, s01) \
	PWXFORM_SIMD_AVX2(X1, x1, s10, s11)

#define PWXFORM_AVX2 \
	{ \
		PWXFORM_AVX2_T x0, x1; \
		PWXFORM_AVX2_T s00, s01, s10, s11; \
		PWXFORM_ROUND_AVX2 PWXFORM_ROUND_AVX2 \
		PWXFORM_ROUND_AVX2 PWXFORM_ROUND_AVX2 \
		PWXFORM_ROUND_AVX2 PWXFORM_ROUND_AVX2 \
	}

#define XOR4_AVX2(in) \
	X0.ymm = _mm256_xor_si256(X0.ymm, (in)[0].ymm); \
	X1.ymm = _mm256_xor_si256(X1.ymm, (in)[1].ymm);

#define OUTBLK(out) \
	(out)[0] = X0; \
	(out)[1] = X1;

/**
 * blockmix_pwxform(Bin, Bout, r, S):
 * Compute Bout = BlockMix_pwxform{salsa20/8, r, S}(Bin).  The input Bin must
 * be 128r bytes in length; the output Bout must also be the same size.
 */
static void
blockmix_avx2(const salsa20_blk_t_avx2 *restrict Bin, salsa20_blk_t_avx2 *restrict Bout,
    size_t r, const __m256i *restrict S)
{
	const uint8_t * S0, * S1;
	avx256i X0, X1;
	size_t i;
	//avx256i X_MASK;

	if (!S) {
		blockmix_salsa8_avx2(Bin, Bout, r);
		return;
	}

	S0 = (const uint8_t *)S;
	S1 = (const uint8_t *)S + S_SIZE_ALL_AVX2 / 2;

	/* Convert 128-byte blocks to 64-byte blocks */
	r *= 2;
	//r *= 4;

	r--;
	PREFETCH_AVX2(&Bin[r], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH_AVX2(&Bin[i], _MM_HINT_T0)
		PREFETCH_AVX2_OUT(&Bout[i], _MM_HINT_T0)
	}
	PREFETCH_AVX2_OUT(&Bout[r], _MM_HINT_T0)

	/* X <-- B_{r1 - 1} */
	X0 = Bin[r].dq[0];
	X1 = Bin[r].dq[1];

	/* for i = 0 to r1 - 1 do */
	for (i = 0; i < r; i++) {
		/* X <-- H'(X \xor B_i) */
		XOR4_AVX2(Bin[i].dq)
		PWXFORM_AVX2
		/* B'_i <-- X */
		OUTBLK(Bout[i].dq)
	}

	/* Last iteration of the loop above */
	XOR4_AVX2(Bin[i].dq)
	PWXFORM_AVX2

	/* B'_i <-- H(B'_i) */
	SALSA20_8_AVX2(Bout[i].dq)
}

#define XOR4_2_AVX2(in1, in2) \
	X0.ymm = _mm256_xor_si256((in1)[0].ymm, (in2)[0].ymm); \
	X1.ymm = _mm256_xor_si256((in1)[1].ymm, (in2)[1].ymm);

static inline uint32_t
blockmix_salsa8_xor_avx2(const salsa20_blk_t_avx2 *restrict Bin1,
    const salsa20_blk_t_avx2 *restrict Bin2, salsa20_blk_t_avx2 *restrict Bout,
    size_t r, int Bin2_in_ROM)
{
	avx256i X0, X1;
	size_t i;

	r--;
	if (Bin2_in_ROM) {
		PREFETCH_AVX2(&Bin2[r * 2 + 1], _MM_HINT_NTA)
		PREFETCH_AVX2(&Bin1[r * 2 + 1], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH_AVX2(&Bin2[i * 2], _MM_HINT_NTA)
			PREFETCH_AVX2(&Bin1[i * 2], _MM_HINT_T0)
			PREFETCH_AVX2(&Bin2[i * 2 + 1], _MM_HINT_NTA)
			PREFETCH_AVX2(&Bin1[i * 2 + 1], _MM_HINT_T0)
			PREFETCH_AVX2_OUT(&Bout[i], _MM_HINT_T0)
			PREFETCH_AVX2_OUT(&Bout[r + 1 + i], _MM_HINT_T0)
		}
		PREFETCH_AVX2(&Bin2[r * 2], _MM_HINT_T0)
	} else {
		PREFETCH_AVX2(&Bin2[r * 2 + 1], _MM_HINT_T0)
		PREFETCH_AVX2(&Bin1[r * 2 + 1], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH_AVX2(&Bin2[i * 2], _MM_HINT_T0)
			PREFETCH_AVX2(&Bin1[i * 2], _MM_HINT_T0)
			PREFETCH_AVX2(&Bin2[i * 2 + 1], _MM_HINT_T0)
			PREFETCH_AVX2(&Bin1[i * 2 + 1], _MM_HINT_T0)
			PREFETCH_AVX2_OUT(&Bout[i], _MM_HINT_T0)
			PREFETCH_AVX2_OUT(&Bout[r + 1 + i], _MM_HINT_T0)
		}
		PREFETCH_AVX2(&Bin2[r * 2], _MM_HINT_T0)
	}
	PREFETCH_AVX2(&Bin1[r * 2], _MM_HINT_T0)
	PREFETCH_AVX2_OUT(&Bout[r], _MM_HINT_T0)
	PREFETCH_AVX2_OUT(&Bout[r * 2 + 1], _MM_HINT_T0)

	/* 1: X <-- B_{2r - 1} */
	XOR4_2_AVX2(Bin1[r * 2 + 1].dq, Bin2[r * 2 + 1].dq)

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4_AVX2(Bin1[0].dq)
	SALSA20_8_XOR_MEM_AVX2(Bin2[0].dq, Bout[0].dq)

	/* 2: for i = 0 to 2r - 1 do */
	for (i = 0; i < r;) {
		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4_AVX2(Bin1[i * 2 + 1].dq)
		SALSA20_8_XOR_MEM_AVX2(Bin2[i * 2 + 1].dq, Bout[r + 1 + i].dq)

		i++;

		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4_AVX2(Bin1[i * 2].dq)
		SALSA20_8_XOR_MEM_AVX2(Bin2[i * 2].dq, Bout[i].dq)
	}

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4_AVX2(Bin1[r * 2 + 1].dq)
	SALSA20_8_XOR_MEM_AVX2(Bin2[r * 2 + 1].dq, Bout[r * 2 + 1].dq)

	return X0.u32[0];
}

static uint32_t
blockmix_xor_avx2(const salsa20_blk_t_avx2 *restrict Bin1,
    const salsa20_blk_t_avx2 *restrict Bin2, salsa20_blk_t_avx2 *restrict Bout,
    size_t r, int Bin2_in_ROM, const __m256i *restrict S)
{
	const uint8_t * S0, * S1;
	avx256i X0, X1;
	size_t i;
	//avx256i X_MASK;

	if (!S)
		return blockmix_salsa8_xor_avx2(Bin1, Bin2, Bout, r, Bin2_in_ROM);

	S0 = (const uint8_t *)S;
	S1 = (const uint8_t *)S + S_SIZE_ALL_AVX2 / 2;

	/* Convert 128-byte blocks to 64-byte blocks */
	r *= 2;

	r--;
	if (Bin2_in_ROM) {
		PREFETCH_AVX2(&Bin2[r], _MM_HINT_NTA)
		PREFETCH_AVX2(&Bin1[r], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH_AVX2(&Bin2[i], _MM_HINT_NTA)
			PREFETCH_AVX2(&Bin1[i], _MM_HINT_T0)
			PREFETCH_AVX2_OUT(&Bout[i], _MM_HINT_T0)
		}
	} else {
		PREFETCH_AVX2(&Bin2[r], _MM_HINT_T0)
		PREFETCH_AVX2(&Bin1[r], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH_AVX2(&Bin2[i], _MM_HINT_T0)
			PREFETCH_AVX2(&Bin1[i], _MM_HINT_T0)
			PREFETCH_AVX2_OUT(&Bout[i], _MM_HINT_T0)
		}
	}
	PREFETCH_AVX2_OUT(&Bout[r], _MM_HINT_T0);

	/* X <-- B_{r1 - 1} */
	XOR4_2_AVX2(Bin1[r].dq, Bin2[r].dq)

	/* for i = 0 to r1 - 1 do */
	for (i = 0; i < r; i++) {
		/* X <-- H'(X \xor B_i) */
		XOR4_AVX2(Bin1[i].dq)
		XOR4_AVX2(Bin2[i].dq)
		PWXFORM_AVX2
		/* B'_i <-- X */
		OUTBLK(Bout[i].dq)
	}

	/* Last iteration of the loop above */
	XOR4_AVX2(Bin1[i].dq)
	XOR4_AVX2(Bin2[i].dq)
	PWXFORM_AVX2

	/* B'_i <-- H(B'_i) */
	SALSA20_8_AVX2(Bout[i].dq)

	return X0.u32[0];
}

#undef XOR4_AVX2
#define XOR4_AVX2(in, out) \
	(out)[0].ymm = Y0.ymm = _mm256_xor_si256((in)[0].ymm, (out)[0].ymm); \
	(out)[1].ymm = Y1.ymm = _mm256_xor_si256((in)[1].ymm, (out)[1].ymm);

static inline uint32_t
blockmix_salsa8_xor_save_avx2(const salsa20_blk_t_avx2 *restrict Bin1,
    salsa20_blk_t_avx2 *restrict Bin2, salsa20_blk_t_avx2 *restrict Bout,
    size_t r)
{
	avx256i X0, X1, Y0, Y1;
	size_t i;

	r--;
	PREFETCH_AVX2(&Bin2[r * 2 + 1], _MM_HINT_T0)
	PREFETCH_AVX2(&Bin1[r * 2 + 1], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH_AVX2(&Bin2[i * 2], _MM_HINT_T0)
		PREFETCH_AVX2(&Bin1[i * 2], _MM_HINT_T0)
		PREFETCH_AVX2(&Bin2[i * 2 + 1], _MM_HINT_T0)
		PREFETCH_AVX2(&Bin1[i * 2 + 1], _MM_HINT_T0)
		PREFETCH_AVX2_OUT(&Bout[i], _MM_HINT_T0)
		PREFETCH_AVX2_OUT(&Bout[r + 1 + i], _MM_HINT_T0)
	}
	PREFETCH_AVX2(&Bin2[r * 2], _MM_HINT_T0)
	PREFETCH_AVX2(&Bin1[r * 2], _MM_HINT_T0)
	PREFETCH_AVX2_OUT(&Bout[r], _MM_HINT_T0)
	PREFETCH_AVX2_OUT(&Bout[r * 2 + 1], _MM_HINT_T0)

	/* 1: X <-- B_{2r - 1} */
	XOR4_2_AVX2(Bin1[r * 2 + 1].dq, Bin2[r * 2 + 1].dq)

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4_AVX2(Bin1[0].dq, Bin2[0].dq)
	SALSA20_8_XOR_REG_AVX2(Bout[0].dq)

	/* 2: for i = 0 to 2r - 1 do */
	for (i = 0; i < r;) {
		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4_AVX2(Bin1[i * 2 + 1].dq, Bin2[i * 2 + 1].dq)
		SALSA20_8_XOR_REG_AVX2(Bout[r + 1 + i].dq)

		i++;

		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4_AVX2(Bin1[i * 2].dq, Bin2[i * 2].dq)
		SALSA20_8_XOR_REG_AVX2(Bout[i].dq)
	}

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4_AVX2(Bin1[r * 2 + 1].dq, Bin2[r * 2 + 1].dq)
	SALSA20_8_XOR_REG_AVX2(Bout[r * 2 + 1].dq)

	return X0.u32[0];
}

#define XOR4_Y_AVX2 \
	X0.ymm = _mm256_xor_si256(X0.ymm, Y0.ymm); \
	X1.ymm = _mm256_xor_si256(X1.ymm, Y1.ymm);

static uint32_t
blockmix_xor_save_avx2(const salsa20_blk_t_avx2 *restrict Bin1,
    salsa20_blk_t_avx2 *restrict Bin2, salsa20_blk_t_avx2 *restrict Bout,
    size_t r, const __m256i *restrict S)
{
	const uint8_t * S0, * S1;
	avx256i X0, X1, Y0, Y1;
	size_t i;
	//avx256i X_MASK;

	if (!S)
		return blockmix_salsa8_xor_save_avx2(Bin1, Bin2, Bout, r);

	S0 = (const uint8_t *)S;
	S1 = (const uint8_t *)S + S_SIZE_ALL_AVX2 / 2;

	/* Convert 128-byte blocks to 64-byte blocks */
	r *= 2;

	r--;
	PREFETCH_AVX2(&Bin2[r], _MM_HINT_T0)
	PREFETCH_AVX2(&Bin1[r], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH_AVX2(&Bin2[i], _MM_HINT_T0)
		PREFETCH_AVX2(&Bin1[i], _MM_HINT_T0)
		PREFETCH_AVX2_OUT(&Bout[i], _MM_HINT_T0)
	}
	PREFETCH_AVX2_OUT(&Bout[r], _MM_HINT_T0);

	/* X <-- B_{r1 - 1} */
	XOR4_2_AVX2(Bin1[r].dq, Bin2[r].dq)

	/* for i = 0 to r1 - 1 do */
	for (i = 0; i < r; i++) {
		XOR4_AVX2(Bin1[i].dq, Bin2[i].dq)
		/* X <-- H'(X \xor B_i) */
		XOR4_Y_AVX2
		PWXFORM_AVX2
		/* B'_i <-- X */
		OUTBLK(Bout[i].dq)
	}

	/* Last iteration of the loop above */
	XOR4_AVX2(Bin1[i].dq, Bin2[i].dq)
	XOR4_Y_AVX2
	PWXFORM_AVX2

	/* B'_i <-- H(B'_i) */
	SALSA20_8_AVX2(Bout[i].dq)

	return X0.u32[0];
}

#undef ARX_AVX2
#undef SALSA20_2ROUNDS_AVX2
#undef SALSA20_8_AVX2
#undef SALSA20_8_XOR_ANY_AVX2
#undef SALSA20_8_XOR_MEM_AVX2
#undef SALSA20_8_XOR_REG_AVX2
#undef PWXFORM_ROUND
#undef PWXFORM_AVX2
#undef OUTBLK
#undef XOR4_AVX2
#undef XOR4_2_AVX2
#undef XOR4_Y_AVX2

/**
 * integerify(B, r):
 * Return the result of parsing B_{2r-1} as a little-endian integer.
 */
static inline uint32_t
integerify_avx2(const salsa20_blk_t_avx2 * B, size_t r)
{
	return B[2 * r - 1].w[0];
}

/**
 * smix1(B, r, N, flags, V, NROM, shared, XY, S):
 * Compute first loop of B = SMix_r(B, N).  The input B must be 128r bytes in
 * length; the temporary storage V must be 128rN bytes in length; the temporary
 * storage XY must be 128r bytes in length.  The value N must be even and no
 * smaller than 2.  The array V must be aligned to a multiple of 64 bytes, and
 * arrays B and XY to a multiple of at least 16 bytes (aligning them to 64
 * bytes as well saves cache lines, but might result in cache bank conflicts).
 */
static void
smix1_avx2(uint8_t * B, size_t r, uint32_t N, yescrypt_flags_t flags,
    salsa20_blk_t_avx2 * V, uint32_t NROM, const yescrypt_shared_t * shared,
    salsa20_blk_t_avx2 * XY, void * S)
{
	const salsa20_blk_t_avx2 * VROM = (salsa20_blk_t_avx2 *)shared->shared1.aligned;
	uint32_t VROM_mask = shared->mask1;
	size_t s = 2 * r;
	salsa20_blk_t_avx2 * X = V, * Y;
	uint32_t i, j;
	size_t k;

	/* 1: X <-- B */
	/* 3: V_i <-- X */
	for (k = 0; k < 2 * r; k++) {
		for (i = 0; i < 16; i++) {
			X[k].w[i] = le32dec(&B[(k * 16 + (i * 5 % 16)) * 4]);
		}
	}

	if (NROM && (VROM_mask & 1)) {
		uint32_t n;
		salsa20_blk_t_avx2 * V_n;
		const salsa20_blk_t_avx2 * V_j;

		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y = &V[s];
		blockmix_avx2(X, Y, r, S);

		X = &V[2 * s];
		if ((1 & VROM_mask) == 1) {
			/* j <-- Integerify(X) mod NROM */
			j = integerify_avx2(Y, r) & (NROM - 1);
			V_j = &VROM[j * s];

			/* X <-- H(X \xor VROM_j) */
			j = blockmix_xor_avx2(Y, V_j, X, r, 1, S);
		} else {
			/* X <-- H(X) */
			blockmix_avx2(Y, X, r, S);
			j = integerify_avx2(X, r);
		}

		for (n = 2; n < N; n <<= 1) {
			uint32_t m = (n < N / 2) ? n : (N - 1 - n);

			V_n = &V[n * s];

			/* 2: for i = 0 to N - 1 do */
			for (i = 1; i < m; i += 2) {
				/* j <-- Wrap(Integerify(X), i) */
				j &= n - 1;
				j += i - 1;
				V_j = &V[j * s];

				/* X <-- X \xor V_j */
				/* 4: X <-- H(X) */
				/* 3: V_i <-- X */
				Y = &V_n[i * s];
				j = blockmix_xor_avx2(X, V_j, Y, r, 0, S);

				if (((n + i) & VROM_mask) == 1) {
					/* j <-- Integerify(X) mod NROM */
					j &= NROM - 1;
					V_j = &VROM[j * s];
				} else {
					/* j <-- Wrap(Integerify(X), i) */
					j &= n - 1;
					j += i;
					V_j = &V[j * s];
				}

				/* X <-- H(X \xor VROM_j) */
				X = &V_n[(i + 1) * s];
				j = blockmix_xor_avx2(Y, V_j, X, r, 1, S);
			}
		}

		n >>= 1;

		/* j <-- Wrap(Integerify(X), i) */
		j &= n - 1;
		j += N - 2 - n;
		V_j = &V[j * s];

		/* X <-- X \xor V_j */
		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y = &V[(N - 1) * s];
		j = blockmix_xor_avx2(X, V_j, Y, r, 0, S);

		if (((N - 1) & VROM_mask) == 1) {
			/* j <-- Integerify(X) mod NROM */
			j &= NROM - 1;
			V_j = &VROM[j * s];
		} else {
			/* j <-- Wrap(Integerify(X), i) */
			j &= n - 1;
			j += N - 1 - n;
			V_j = &V[j * s];
		}

		/* X <-- X \xor V_j */
		/* 4: X <-- H(X) */
		X = XY;
		blockmix_xor_avx2(Y, V_j, X, r, 1, S);
	} else if (flags & YESCRYPT_RW) {
		uint32_t n;
		salsa20_blk_t_avx2 * V_n, * V_j;

		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y = &V[s];
		blockmix_avx2(X, Y, r, S);

		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		X = &V[2 * s];
		blockmix_avx2(Y, X, r, S);
		j = integerify_avx2(X, r);

		for (n = 2; n < N; n <<= 1) {
			uint32_t m = (n < N / 2) ? n : (N - 1 - n);

			V_n = &V[n * s];

			/* 2: for i = 0 to N - 1 do */
			for (i = 1; i < m; i += 2) {
				Y = &V_n[i * s];

				/* j <-- Wrap(Integerify(X), i) */
				j &= n - 1;
				j += i - 1;
				V_j = &V[j * s];

				/* X <-- X \xor V_j */
				/* 4: X <-- H(X) */
				/* 3: V_i <-- X */
				j = blockmix_xor_avx2(X, V_j, Y, r, 0, S);

				/* j <-- Wrap(Integerify(X), i) */
				j &= n - 1;
				j += i;
				V_j = &V[j * s];

				/* X <-- X \xor V_j */
				/* 4: X <-- H(X) */
				/* 3: V_i <-- X */
				X = &V_n[(i + 1) * s];
				j = blockmix_xor_avx2(Y, V_j, X, r, 0, S);
			}
		}

		n >>= 1;

		/* j <-- Wrap(Integerify(X), i) */
		j &= n - 1;
		j += N - 2 - n;
		V_j = &V[j * s];

		/* X <-- X \xor V_j */
		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y = &V[(N - 1) * s];
		j = blockmix_xor_avx2(X, V_j, Y, r, 0, S);

		/* j <-- Wrap(Integerify(X), i) */
		j &= n - 1;
		j += N - 1 - n;
		V_j = &V[j * s];

		/* X <-- X \xor V_j */
		/* 4: X <-- H(X) */
		X = XY;
		blockmix_xor_avx2(Y, V_j, X, r, 0, S);
	} else {
		/* 2: for i = 0 to N - 1 do */
		for (i = 1; i < N - 1; i += 2) {
			/* 4: X <-- H(X) */
			/* 3: V_i <-- X */
			Y = &V[i * s];
			blockmix_avx2(X, Y, r, S);

			/* 4: X <-- H(X) */
			/* 3: V_i <-- X */
			X = &V[(i + 1) * s];
			blockmix_avx2(Y, X, r, S);
		}

		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y = &V[i * s];
		blockmix_avx2(X, Y, r, S);

		/* 4: X <-- H(X) */
		X = XY;
		blockmix_avx2(Y, X, r, S);
	}

	/* B' <-- X */
	for (k = 0; k < 2 * r; k++) {
		for (i = 0; i < 16; i++) {
			le32enc(&B[(k * 16 + (i * 5 % 16)) * 4], X[k].w[i]);
		}
	}
}

/**
 * smix2(B, r, N, Nloop, flags, V, NROM, shared, XY, S):
 * Compute second loop of B = SMix_r(B, N).  The input B must be 128r bytes in
 * length; the temporary storage V must be 128rN bytes in length; the temporary
 * storage XY must be 256r bytes in length.  The value N must be a power of 2
 * greater than 1.  The value Nloop must be even.  The array V must be aligned
 * to a multiple of 64 bytes, and arrays B and XY to a multiple of at least 16
 * bytes (aligning them to 64 bytes as well saves cache lines, but might result
 * in cache bank conflicts).
 */
static void
smix2_avx2(uint8_t * B, size_t r, uint32_t N, uint64_t Nloop,
    yescrypt_flags_t flags, salsa20_blk_t_avx2 * V, uint32_t NROM,
    const yescrypt_shared_t * shared, salsa20_blk_t_avx2 * XY, void * S)
{
	const salsa20_blk_t_avx2 * VROM = (salsa20_blk_t_avx2 *)shared->shared1.aligned;
	uint32_t VROM_mask = shared->mask1;
	size_t s = 2 * r;
	salsa20_blk_t_avx2 * X = XY, * Y = &XY[s];
	uint64_t i;
	uint32_t j;
	size_t k;

	if (Nloop == 0)
		return;

	/* X <-- B' */
	/* 3: V_i <-- X */
	for (k = 0; k < 2 * r; k++) {
		for (i = 0; i < 16; i++) {
			X[k].w[i] = le32dec(&B[(k * 16 + (i * 5 % 16)) * 4]);
		}
	}

	i = Nloop / 2;

	/* 7: j <-- Integerify(X) mod N */
	j = integerify_avx2(X, r) & (N - 1);

/*
 * Normally, NROM implies YESCRYPT_RW, but we check for these separately
 * because YESCRYPT_PARALLEL_SMIX resets YESCRYPT_RW for the smix2() calls
 * operating on the entire V.
 */
	if (NROM && (flags & YESCRYPT_RW)) {
		/* 6: for i = 0 to N - 1 do */
		for (i = 0; i < Nloop; i += 2) {
			salsa20_blk_t_avx2 * V_j = &V[j * s];

			/* 8: X <-- H(X \xor V_j) */
			/* V_j <-- Xprev \xor V_j */
			/* j <-- Integerify(X) mod NROM */
			j = blockmix_xor_save_avx2(X, V_j, Y, r, S);

			if (((i + 1) & VROM_mask) == 1) {
				const salsa20_blk_t_avx2 * VROM_j;

				j &= NROM - 1;
				VROM_j = &VROM[j * s];

				/* X <-- H(X \xor VROM_j) */
				/* 7: j <-- Integerify(X) mod N */
				j = blockmix_xor_avx2(Y, VROM_j, X, r, 1, S);
			} else {
				j &= N - 1;
				V_j = &V[j * s];

				/* 8: X <-- H(X \xor V_j) */
				/* V_j <-- Xprev \xor V_j */
				/* j <-- Integerify(X) mod NROM */
				j = blockmix_xor_save_avx2(Y, V_j, X, r, S);
			}
			j &= N - 1;
			V_j = &V[j * s];
		}
	} else if (NROM) {
		/* 6: for i = 0 to N - 1 do */
		for (i = 0; i < Nloop; i += 2) {
			const salsa20_blk_t_avx2 * V_j = &V[j * s];

			/* 8: X <-- H(X \xor V_j) */
			/* V_j <-- Xprev \xor V_j */
			/* j <-- Integerify(X) mod NROM */
			j = blockmix_xor_avx2(X, V_j, Y, r, 0, S);

			if (((i + 1) & VROM_mask) == 1) {
				j &= NROM - 1;
				V_j = &VROM[j * s];
			} else {
				j &= N - 1;
				V_j = &V[j * s];
			}

			/* X <-- H(X \xor VROM_j) */
			/* 7: j <-- Integerify(X) mod N */
			j = blockmix_xor_avx2(Y, V_j, X, r, 1, S);
			j &= N - 1;
			V_j = &V[j * s];
		}
	} else if (flags & YESCRYPT_RW) {
		/* 6: for i = 0 to N - 1 do */
		do {
			salsa20_blk_t_avx2 * V_j = &V[j * s];

			/* 8: X <-- H(X \xor V_j) */
			/* V_j <-- Xprev \xor V_j */
			/* 7: j <-- Integerify(X) mod N */
			j = blockmix_xor_save_avx2(X, V_j, Y, r, S);
			j &= N - 1;
			V_j = &V[j * s];

			/* 8: X <-- H(X \xor V_j) */
			/* V_j <-- Xprev \xor V_j */
			/* 7: j <-- Integerify(X) mod N */
			j = blockmix_xor_save_avx2(Y, V_j, X, r, S);
			j &= N - 1;
		} while (--i);
	} else {
		/* 6: for i = 0 to N - 1 do */
		do {
			const salsa20_blk_t_avx2 * V_j = &V[j * s];

			/* 8: X <-- H(X \xor V_j) */
			/* 7: j <-- Integerify(X) mod N */
			j = blockmix_xor_avx2(X, V_j, Y, r, 0, S);
			j &= N - 1;
			V_j = &V[j * s];

			/* 8: X <-- H(X \xor V_j) */
			/* 7: j <-- Integerify(X) mod N */
			j = blockmix_xor_avx2(Y, V_j, X, r, 0, S);
			j &= N - 1;
		} while (--i);
	}

	/* 10: B' <-- X */
	for (k = 0; k < 2 * r; k++) {
		for (i = 0; i < 16; i++) {
			le32enc(&B[(k * 16 + (i * 5 % 16)) * 4], X[k].w[i]);
		}
	}
}

/**
 * p2floor(x):
 * Largest power of 2 not greater than argument.
 */
static uint64_t
p2floor_avx2(uint64_t x)
{
	uint64_t y;
	while ((y = x & (x - 1)))
		x = y;
	return x;
}

/**
 * smix(B, r, N, p, t, flags, V, NROM, shared, XY, S):
 * Compute B = SMix_r(B, N).  The input B must be 128rp bytes in length; the
 * temporary storage V must be 128rN bytes in length; the temporary storage XY
 * must be 256r or 256rp bytes in length (the larger size is required with
 * OpenMP-enabled builds).  The value N must be a power of 2 greater than 1.
 * The array V must be aligned to a multiple of 64 bytes, and arrays B and
 * XY to a multiple of at least 16 bytes (aligning them to 64 bytes as well
 * saves cache lines and helps avoid false sharing in OpenMP-enabled builds
 * when p > 1, but it might also result in cache bank conflicts).
 */
static void
smix_avx2(uint8_t * B, size_t r, uint32_t N, uint32_t p, uint32_t t,
    yescrypt_flags_t flags,
    salsa20_blk_t_avx2 * V, uint32_t NROM, const yescrypt_shared_t * shared,
    salsa20_blk_t_avx2 * XY, void * S)
{
	size_t s = 2 * r;
	uint32_t Nchunk = N / p;
	uint64_t Nloop_all, Nloop_rw;
	uint32_t i;

	Nloop_all = Nchunk;
	if (flags & YESCRYPT_RW) {
		if (t <= 1) {
			if (t)
				Nloop_all *= 2; /* 2/3 */
			Nloop_all = (Nloop_all + 2) / 3; /* 1/3, round up */
		} else {
			Nloop_all *= t - 1;
		}
	} else if (t) {
		if (t == 1)
			Nloop_all += (Nloop_all + 1) / 2; /* 1.5, round up */
		Nloop_all *= t;
	}

	Nloop_rw = 0;
	if (flags & __YESCRYPT_INIT_SHARED)
		Nloop_rw = Nloop_all;
	else if (flags & YESCRYPT_RW)
		Nloop_rw = Nloop_all / p;

	Nchunk &= ~(uint32_t)1; /* round down to even */
	Nloop_all++; Nloop_all &= ~(uint64_t)1; /* round up to even */
	Nloop_rw &= ~(uint64_t)1; /* round down to even */

#ifdef _OPENMP
#pragma omp parallel if (p > 1) default(none) private(i) shared(B, r, N, p, flags, V, NROM, shared, XY, S, s, Nchunk, Nloop_all, Nloop_rw)
	{
#pragma omp for
#endif
	for (i = 0; i < p; i++) {
		uint32_t Vchunk = i * Nchunk;
		uint8_t * Bp = &B[128 * r * i];
		salsa20_blk_t_avx2 * Vp = &V[Vchunk * s];
#ifdef _OPENMP
		salsa20_blk_t_avx2 * XYp = &XY[i * (2 * s)];
#else
		salsa20_blk_t_avx2 * XYp = XY;
#endif
		uint32_t Np = (i < p - 1) ? Nchunk : (N - Vchunk);
		void * Sp = S ? ((uint8_t *)S + i * S_SIZE_ALL_AVX2) : S;
		if (Sp)
			smix1_avx2(Bp, 1, S_SIZE_ALL_AVX2 / 128,
			    flags & ~YESCRYPT_PWXFORM,
			    Sp, NROM, shared, XYp, NULL);
		if (!(flags & __YESCRYPT_INIT_SHARED_2))
			smix1_avx2(Bp, r, Np, flags, Vp, NROM, shared, XYp, Sp);
		smix2_avx2(Bp, r, p2floor_avx2(Np), Nloop_rw, flags, Vp,
		    NROM, shared, XYp, Sp);
	}

	if (Nloop_all > Nloop_rw) {
#ifdef _OPENMP
#pragma omp for
#endif
		for (i = 0; i < p; i++) {
			uint8_t * Bp = &B[128 * r * i];
#ifdef _OPENMP
			salsa20_blk_t_avx2 * XYp = &XY[i * (2 * s)];
#else
			salsa20_blk_t_avx2 * XYp = XY;
#endif
			void * Sp = S ? ((uint8_t *)S + i * S_SIZE_ALL_AVX2) : S;
			smix2_avx2(Bp, r, N, Nloop_all - Nloop_rw,
			    flags & ~YESCRYPT_RW, V, NROM, shared, XYp, Sp);
		}
	}
#ifdef _OPENMP
	}
#endif
}

/**
 * yescrypt_kdf(shared, local, passwd, passwdlen, salt, saltlen,
 *     N, r, p, t, flags, buf, buflen):
 * Compute scrypt(passwd[0 .. passwdlen - 1], salt[0 .. saltlen - 1], N, r,
 * p, buflen), or a revision of scrypt as requested by flags and shared, and
 * write the result into buf.  The parameters r, p, and buflen must satisfy
 * r * p < 2^30 and buflen <= (2^32 - 1) * 32.  The parameter N must be a power
 * of 2 greater than 1.  (This optimized implementation currently additionally
 * limits N to the range from 8 to 2^31, but other implementation might not.)
 *
 * t controls computation time while not affecting peak memory usage.  shared
 * and flags may request special modes as described in yescrypt.h.  local is
 * the thread-local data structure, allowing to preserve and reuse a memory
 * allocation across calls, thereby reducing its overhead.
 *
 * Return 0 on success; or -1 on error.
 */
static int
yescrypt_kdf_avx2(const yescrypt_shared_t * shared, yescrypt_local_t * local,
    const uint8_t * passwd, size_t passwdlen,
    const uint8_t * salt, size_t saltlen,
    uint64_t N, uint32_t r, uint32_t p, uint32_t t, yescrypt_flags_t flags,
    uint8_t * buf, size_t buflen)
{
	yescrypt_region_t tmp;
	uint64_t NROM;
	size_t B_size, V_size, XY_size, need;
	uint8_t * B, * S;
	salsa20_blk_t_avx2 * V, * XY;
	uint8_t sha256[32];
	YMM_MASK.u64[0] = YMM_MASK.u64[1] = YMM_MASK.u64[2] = YMM_MASK.u64[3] = S_MASK_AVX2;

	/*
	 * YESCRYPT_PARALLEL_SMIX is a no-op at p = 1 for its intended purpose,
	 * so don't let it have side-effects.  Without this adjustment, it'd
	 * enable the SHA-256 password pre-hashing and output post-hashing,
	 * because any deviation from classic scrypt implies those.
	 */
	if (p == 1)
		flags &= ~YESCRYPT_PARALLEL_SMIX;

	/* Sanity-check parameters */
	if (flags & ~YESCRYPT_KNOWN_FLAGS) {
		errno = EINVAL;
		return -1;
	}
#if SIZE_MAX > UINT32_MAX
	if (buflen > (((uint64_t)(1) << 32) - 1) * 32) {
		errno = EFBIG;
		return -1;
	}
#endif
	if ((uint64_t)(r) * (uint64_t)(p) >= (1 << 30)) {
		errno = EFBIG;
		return -1;
	}
	if (N > UINT32_MAX) {
		errno = EFBIG;
		return -1;
	}
	if (((N & (N - 1)) != 0) || (N <= 7) || (r < 1) || (p < 1)) {
		errno = EINVAL;
		return -1;
	}
	if ((flags & YESCRYPT_PARALLEL_SMIX) && (N / p <= 7)) {
		errno = EINVAL;
		return -1;
	}
	if ((r > SIZE_MAX / 256 / p) ||
	    (N > SIZE_MAX / 128 / r)) {
		errno = ENOMEM;
		return -1;
	}
#ifdef _OPENMP
	if (!(flags & YESCRYPT_PARALLEL_SMIX) &&
	    (N > SIZE_MAX / 128 / (r * p))) {
		errno = ENOMEM;
		return -1;
	}
#endif
	if ((flags & YESCRYPT_PWXFORM) &&
#ifndef _OPENMP
	    (flags & YESCRYPT_PARALLEL_SMIX) &&
#endif
	    p > SIZE_MAX / S_SIZE_ALL_AVX2) {
		errno = ENOMEM;
		return -1;
	}

	NROM = 0;
	if (shared->shared1.aligned) {
		NROM = shared->shared1.aligned_size / ((size_t)128 * r);
		if (NROM > UINT32_MAX) {
			errno = EFBIG;
			return -1;
		}
		if (((NROM & (NROM - 1)) != 0) || (NROM <= 7) ||
		    !(flags & YESCRYPT_RW)) {
			errno = EINVAL;
			return -1;
		}
	}

	/* Allocate memory */
	V = NULL;
	V_size = (size_t)128 * r * N;
#ifdef _OPENMP
	if (!(flags & YESCRYPT_PARALLEL_SMIX))
		V_size *= p;
#endif
	need = V_size;
	if (flags & __YESCRYPT_INIT_SHARED) {
		if (local->aligned_size < need) {
			if (local->base || local->aligned ||
			    local->base_size || local->aligned_size) {
				errno = EINVAL;
				return -1;
			}
			if (!alloc_region(local, need))
				return -1;
		}
		V = (salsa20_blk_t_avx2 *)local->aligned;
		need = 0;
	}
	B_size = (size_t)128 * r * p;
	need += B_size;
	if (need < B_size) {
		errno = ENOMEM;
		return -1;
	}
	XY_size = (size_t)256 * r;
#ifdef _OPENMP
	XY_size *= p;
#endif
	need += XY_size;
	if (need < XY_size) {
		errno = ENOMEM;
		return -1;
	}
	if (flags & YESCRYPT_PWXFORM) {
		size_t S_size = S_SIZE_ALL_AVX2;
#ifdef _OPENMP
		S_size *= p;
#else
		if (flags & YESCRYPT_PARALLEL_SMIX)
			S_size *= p;
#endif
		need += S_size;
		if (need < S_size) {
			errno = ENOMEM;
			return -1;
		}
	}
	if (flags & __YESCRYPT_INIT_SHARED) {
		if (!alloc_region(&tmp, need))
			return -1;
		B = (uint8_t *)tmp.aligned;
		XY = (salsa20_blk_t_avx2 *)((uint8_t *)B + B_size);
	} else {
		init_region(&tmp);
		if (local->aligned_size < need) {
			if (free_region(local))
				return -1;
			if (!alloc_region(local, need))
				return -1;
		}
		B = (uint8_t *)local->aligned;
		V = (salsa20_blk_t_avx2 *)((uint8_t *)B + B_size);
		XY = (salsa20_blk_t_avx2 *)((uint8_t *)V + V_size);
	}
	S = NULL;
	if (flags & YESCRYPT_PWXFORM)
		S = (uint8_t *)XY + XY_size;

#ifdef __CGN__
	BCRYPT_ALG_HANDLE hAlgorithm;
	BCryptOpenAlgorithmProvider(&hAlgorithm, BCRYPT_SHA256_ALGORITHM, MS_PRIMITIVE_PROVIDER, 0);
#endif
	if (t || flags) {
		SHA256_CTX ctx;
#ifdef __CGN__
		ctx.hAlgorithm = hAlgorithm;
		ctx.hAlgorithm = hAlgorithm;
		ctx.hashWorkLength = HASH_WORK_LENGTH;
		ctx.hashWorkLength = HASH_WORK_LENGTH;
#endif
		SHA256_Init(&ctx);
		SHA256_Update(&ctx, passwd, passwdlen);
		SHA256_Final(sha256, &ctx);
		passwd = sha256;
		passwdlen = sizeof(sha256);
	}

	/* 1: (B_0 ... B_{p-1}) <-- PBKDF2(P, S, 1, p * MFLen) */
#ifndef __CGN__
	PBKDF2_SHA256(passwd, passwdlen, salt, saltlen, 1, B, B_size);
#else
	PBKDF2_SHA256(passwd, passwdlen, salt, saltlen, 1, B, B_size, hAlgorithm);
#endif

	if (t || flags)
		memcpy(sha256, B, sizeof(sha256));

	if (p == 1 || (flags & YESCRYPT_PARALLEL_SMIX)) {
		smix_avx2(B, r, N, p, t, flags, V, NROM, shared, XY, S);
	} else {
		uint32_t i;

		/* 2: for i = 0 to p - 1 do */
#ifdef _OPENMP
#pragma omp parallel for default(none) private(i) shared(B, r, N, p, t, flags, V, NROM, shared, XY, S)
#endif
		for (i = 0; i < p; i++) {
			/* 3: B_i <-- MF(B_i, N) */
#ifdef _OPENMP
			smix_avx2(&B[(size_t)128 * r * i], r, N, 1, t, flags,
			    &V[(size_t)2 * r * i * N],
			    NROM, shared,
			    &XY[(size_t)4 * r * i],
			    S ? &S[S_SIZE_ALL_AVX2 * i] : S);
#else
			smix_avx2(&B[(size_t)128 * r * i], r, N, 1, t, flags, V,
			    NROM, shared, XY, S);
#endif
		}
	}

	/* 5: DK <-- PBKDF2(P, B, 1, dkLen) */
#ifndef __CGN__
	PBKDF2_SHA256(passwd, passwdlen, B, B_size, 1, buf, buflen);
#else
	PBKDF2_SHA256(passwd, passwdlen, B, B_size, 1, buf, buflen, hAlgorithm);
#endif

	/*
	 * Except when computing classic scrypt, allow all computation so far
	 * to be performed on the client.  The final steps below match those of
	 * SCRAM (RFC 5802), so that an extension of SCRAM (with the steps so
	 * far in place of SCRAM's use of PBKDF2 and with SHA-256 in place of
	 * SCRAM's use of SHA-1) would be usable with yescrypt hashes.
	 */
	if ((t || flags) && buflen == sizeof(sha256)) {
		/* Compute ClientKey */
		{
			HMAC_SHA256_CTX ctx;
#ifdef __CGN__
			ctx.ictx.hAlgorithm = hAlgorithm;
			ctx.ictx.hAlgorithm = hAlgorithm;
			ctx.ictx.hashWorkLength = HASH_WORK_LENGTH;
			ctx.ictx.hashWorkLength = HASH_WORK_LENGTH;
			ctx.octx.hAlgorithm = hAlgorithm;
			ctx.octx.hAlgorithm = hAlgorithm;
			ctx.octx.hashWorkLength = HASH_WORK_LENGTH;
			ctx.octx.hashWorkLength = HASH_WORK_LENGTH;
#endif
			HMAC_SHA256_Init(&ctx, buf, buflen);
			HMAC_SHA256_Update(&ctx, "Client Key", 10);
			HMAC_SHA256_Final(sha256, &ctx);
		}
		/* Compute StoredKey */
		{
			SHA256_CTX ctx;
#ifdef __CGN__
			ctx.hAlgorithm = hAlgorithm;
			ctx.hAlgorithm = hAlgorithm;
			ctx.hashWorkLength = HASH_WORK_LENGTH;
			ctx.hashWorkLength = HASH_WORK_LENGTH;
#endif
			SHA256_Init(&ctx);
			SHA256_Update(&ctx, sha256, sizeof(sha256));
			SHA256_Final(buf, &ctx);
		}
	}

#ifdef __CGN__
	BCryptCloseAlgorithmProvider(hAlgorithm, 0);
#endif
	if (free_region(&tmp))
		return -1;

	/* Success! */
	return 0;
}
