/**
 * \file multiply.cc implementations for matrix multiply
 * \author cpapakonstantinou
 * \date 2025
 **/
// Copyright (c) 2025  Constantine Papakonstantinou
//
// Permission is hereby granted, free of charge, to any person obaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#include <macros.h>
#include <multiply.h>
#include <iostream>

namespace damm
{
	#define _MM_MMUL4_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	{\
		alignas(16)__m128 t0, t1, t2, t3; \
		\
		t0 = _mm_dp_ps(a0, b0, 0xF1); \
		t1 = _mm_dp_ps(a0, b1, 0xF2); \
		t2 = _mm_dp_ps(a0, b2, 0xF4); \
		t3 = _mm_dp_ps(a0, b3, 0xF8); \
		t0 = _mm_add_ps(t0, t1); \
		t0 = _mm_add_ps(t0, t2); \
		t0 = _mm_add_ps(t0, t3); \
		c0 = _mm_add_ps(c0, t0); \
		\
		t0 = _mm_dp_ps(a1, b0, 0xF1); \
		t1 = _mm_dp_ps(a1, b1, 0xF2); \
		t2 = _mm_dp_ps(a1, b2, 0xF4); \
		t3 = _mm_dp_ps(a1, b3, 0xF8); \
		t0 = _mm_add_ps(t0, t1); \
		t0 = _mm_add_ps(t0, t2); \
		t0 = _mm_add_ps(t0, t3); \
		c1 = _mm_add_ps(c1, t0); \
		\
		t0 = _mm_dp_ps(a2, b0, 0xF1); \
		t1 = _mm_dp_ps(a2, b1, 0xF2); \
		t2 = _mm_dp_ps(a2, b2, 0xF4); \
		t3 = _mm_dp_ps(a2, b3, 0xF8); \
		t0 = _mm_add_ps(t0, t1); \
		t0 = _mm_add_ps(t0, t2); \
		t0 = _mm_add_ps(t0, t3); \
		c2 = _mm_add_ps(c2, t0); \
		\
		t0 = _mm_dp_ps(a3, b0, 0xF1); \
		t1 = _mm_dp_ps(a3, b1, 0xF2); \
		t2 = _mm_dp_ps(a3, b2, 0xF4); \
		t3 = _mm_dp_ps(a3, b3, 0xF8); \
		t0 = _mm_add_ps(t0, t1); \
		t0 = _mm_add_ps(t0, t2); \
		t0 = _mm_add_ps(t0, t3); \
		c3 = _mm_add_ps(c3, t0); \
	}

	#define _MM_MMUL2_PD(a0, a1, b0, b1, c0, c1) \
	{ \
		alignas(16)__m128d t0, t1;\
		t0 = _mm_dp_pd(a0, b0, 0x31);\
		t1 = _mm_dp_pd(a0, b1, 0x31);\
		c0 = _mm_add_pd(c0, _mm_unpacklo_pd(t0, t1));\
		\
		t0 = _mm_dp_pd(a1, b0, 0x31);\
		t1 = _mm_dp_pd(a1, b1, 0x31);\
		c1 = _mm_add_pd(c1, _mm_unpacklo_pd(t0, t1));\
	}

	#define _MM256_MMUL4_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	{ \
		alignas(32)__m256d t0, t1, t2; \
		alignas(32)__m256d m0, m1, m2, m3; \
		\
		m0 = _mm256_mul_pd(a0, b0); \
		m1 = _mm256_mul_pd(a0, b1); \
		m2 = _mm256_mul_pd(a0, b2); \
		m3 = _mm256_mul_pd(a0, b3); \
		t0 = _mm256_hadd_pd(m0, m1); \
		t1 = _mm256_hadd_pd(m2, m3); \
		t0 = _mm256_permute4x64_pd(t0, _MM_SHUFFLE(3, 1, 2, 0)); \
		t1 = _mm256_permute4x64_pd(t1, _MM_SHUFFLE(3, 1, 2, 0)); \
		t2 = _mm256_hadd_pd(t0, t1); \
		t2 = _mm256_permute4x64_pd(t2, _MM_SHUFFLE(3, 1, 2, 0)); \
		c0 = _mm256_add_pd(c0, t2); \
		\
		m0 = _mm256_mul_pd(a1, b0); \
		m1 = _mm256_mul_pd(a1, b1); \
		m2 = _mm256_mul_pd(a1, b2); \
		m3 = _mm256_mul_pd(a1, b3); \
		t0 = _mm256_hadd_pd(m0, m1); \
		t1 = _mm256_hadd_pd(m2, m3); \
		t0 = _mm256_permute4x64_pd(t0, _MM_SHUFFLE(3, 1, 2, 0)); \
		t1 = _mm256_permute4x64_pd(t1, _MM_SHUFFLE(3, 1, 2, 0)); \
		t2 = _mm256_hadd_pd(t0, t1); \
		t2 = _mm256_permute4x64_pd(t2, _MM_SHUFFLE(3, 1, 2, 0)); \
		c1 = _mm256_add_pd(c1, t2); \
		\
		m0 = _mm256_mul_pd(a2, b0); \
		m1 = _mm256_mul_pd(a2, b1); \
		m2 = _mm256_mul_pd(a2, b2); \
		m3 = _mm256_mul_pd(a2, b3); \
		t0 = _mm256_hadd_pd(m0, m1); \
		t1 = _mm256_hadd_pd(m2, m3); \
		t0 = _mm256_permute4x64_pd(t0, _MM_SHUFFLE(3, 1, 2, 0)); \
		t1 = _mm256_permute4x64_pd(t1, _MM_SHUFFLE(3, 1, 2, 0)); \
		t2 = _mm256_hadd_pd(t0, t1); \
		t2 = _mm256_permute4x64_pd(t2, _MM_SHUFFLE(3, 1, 2, 0)); \
		c2 = _mm256_add_pd(c2, t2); \
		\
		m0 = _mm256_mul_pd(a3, b0); \
		m1 = _mm256_mul_pd(a3, b1); \
		m2 = _mm256_mul_pd(a3, b2); \
		m3 = _mm256_mul_pd(a3, b3); \
		t0 = _mm256_hadd_pd(m0, m1); \
		t1 = _mm256_hadd_pd(m2, m3); \
		t0 = _mm256_permute4x64_pd(t0, _MM_SHUFFLE(3, 1, 2, 0)); \
		t1 = _mm256_permute4x64_pd(t1, _MM_SHUFFLE(3, 1, 2, 0)); \
		t2 = _mm256_hadd_pd(t0, t1); \
		t2 = _mm256_permute4x64_pd(t2, _MM_SHUFFLE(3, 1, 2, 0)); \
		c3 = _mm256_add_pd(c3, t2); \
	}

	#define _MM256_MMUL8_PS(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5, c6, c7) \
	{ \
		alignas(32)__m256 t0, t1, t2, t3;\
		alignas(32)__m256 m0, m1, m2, m3, m4, m5, m6, m7;\
		const __m256i p = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);\
		\
		m0 = _mm256_mul_ps(a0, b0);\
		m1 = _mm256_mul_ps(a0, b1);\
		m2 = _mm256_mul_ps(a0, b2);\
		m3 = _mm256_mul_ps(a0, b3);\
		m4 = _mm256_mul_ps(a0, b4);\
		m5 = _mm256_mul_ps(a0, b5);\
		m6 = _mm256_mul_ps(a0, b6);\
		m7 = _mm256_mul_ps(a0, b7);\
		\
		t0 = _mm256_hadd_ps(m0, m1);\
		t1 = _mm256_hadd_ps(m2, m3);\
		t2 = _mm256_hadd_ps(m4, m5);\
		t3 = _mm256_hadd_ps(m6, m7);\
		\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		t2 = _mm256_permutevar8x32_ps(t2, p);\
		t3 = _mm256_permutevar8x32_ps(t3, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t1 = _mm256_hadd_ps(t2, t3);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		\
		c0 = _mm256_add_ps(c0, t0);\
		\
		m0 = _mm256_mul_ps(a1, b0);\
		m1 = _mm256_mul_ps(a1, b1);\
		m2 = _mm256_mul_ps(a1, b2);\
		m3 = _mm256_mul_ps(a1, b3);\
		m4 = _mm256_mul_ps(a1, b4);\
		m5 = _mm256_mul_ps(a1, b5);\
		m6 = _mm256_mul_ps(a1, b6);\
		m7 = _mm256_mul_ps(a1, b7);\
		\
		t0 = _mm256_hadd_ps(m0, m1);\
		t1 = _mm256_hadd_ps(m2, m3);\
		t2 = _mm256_hadd_ps(m4, m5);\
		t3 = _mm256_hadd_ps(m6, m7);\
		\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		t2 = _mm256_permutevar8x32_ps(t2, p);\
		t3 = _mm256_permutevar8x32_ps(t3, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t1 = _mm256_hadd_ps(t2, t3);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		\
		c1 = _mm256_add_ps(c1, t0);\
		\
		m0 = _mm256_mul_ps(a2, b0);\
		m1 = _mm256_mul_ps(a2, b1);\
		m2 = _mm256_mul_ps(a2, b2);\
		m3 = _mm256_mul_ps(a2, b3);\
		m4 = _mm256_mul_ps(a2, b4);\
		m5 = _mm256_mul_ps(a2, b5);\
		m6 = _mm256_mul_ps(a2, b6);\
		m7 = _mm256_mul_ps(a2, b7);\
		\
		t0 = _mm256_hadd_ps(m0, m1);\
		t1 = _mm256_hadd_ps(m2, m3);\
		t2 = _mm256_hadd_ps(m4, m5);\
		t3 = _mm256_hadd_ps(m6, m7);\
		\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		t2 = _mm256_permutevar8x32_ps(t2, p);\
		t3 = _mm256_permutevar8x32_ps(t3, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t1 = _mm256_hadd_ps(t2, t3);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		\
		c2 = _mm256_add_ps(c2, t0);\
		\
		m0 = _mm256_mul_ps(a3, b0);\
		m1 = _mm256_mul_ps(a3, b1);\
		m2 = _mm256_mul_ps(a3, b2);\
		m3 = _mm256_mul_ps(a3, b3);\
		m4 = _mm256_mul_ps(a3, b4);\
		m5 = _mm256_mul_ps(a3, b5);\
		m6 = _mm256_mul_ps(a3, b6);\
		m7 = _mm256_mul_ps(a3, b7);\
		\
		t0 = _mm256_hadd_ps(m0, m1);\
		t1 = _mm256_hadd_ps(m2, m3);\
		t2 = _mm256_hadd_ps(m4, m5);\
		t3 = _mm256_hadd_ps(m6, m7);\
		\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		t2 = _mm256_permutevar8x32_ps(t2, p);\
		t3 = _mm256_permutevar8x32_ps(t3, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t1 = _mm256_hadd_ps(t2, t3);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		\
		c3 = _mm256_add_ps(c3, t0);\
		\
		m0 = _mm256_mul_ps(a4, b0);\
		m1 = _mm256_mul_ps(a4, b1);\
		m2 = _mm256_mul_ps(a4, b2);\
		m3 = _mm256_mul_ps(a4, b3);\
		m4 = _mm256_mul_ps(a4, b4);\
		m5 = _mm256_mul_ps(a4, b5);\
		m6 = _mm256_mul_ps(a4, b6);\
		m7 = _mm256_mul_ps(a4, b7);\
		\
		t0 = _mm256_hadd_ps(m0, m1);\
		t1 = _mm256_hadd_ps(m2, m3);\
		t2 = _mm256_hadd_ps(m4, m5);\
		t3 = _mm256_hadd_ps(m6, m7);\
		\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		t2 = _mm256_permutevar8x32_ps(t2, p);\
		t3 = _mm256_permutevar8x32_ps(t3, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t1 = _mm256_hadd_ps(t2, t3);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		\
		c4 = _mm256_add_ps(c4, t0);\
		\
		m0 = _mm256_mul_ps(a5, b0);\
		m1 = _mm256_mul_ps(a5, b1);\
		m2 = _mm256_mul_ps(a5, b2);\
		m3 = _mm256_mul_ps(a5, b3);\
		m4 = _mm256_mul_ps(a5, b4);\
		m5 = _mm256_mul_ps(a5, b5);\
		m6 = _mm256_mul_ps(a5, b6);\
		m7 = _mm256_mul_ps(a5, b7);\
		\
		t0 = _mm256_hadd_ps(m0, m1);\
		t1 = _mm256_hadd_ps(m2, m3);\
		t2 = _mm256_hadd_ps(m4, m5);\
		t3 = _mm256_hadd_ps(m6, m7);\
		\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		t2 = _mm256_permutevar8x32_ps(t2, p);\
		t3 = _mm256_permutevar8x32_ps(t3, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t1 = _mm256_hadd_ps(t2, t3);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		\
		c5 = _mm256_add_ps(c5, t0);\
		\
		m0 = _mm256_mul_ps(a6, b0);\
		m1 = _mm256_mul_ps(a6, b1);\
		m2 = _mm256_mul_ps(a6, b2);\
		m3 = _mm256_mul_ps(a6, b3);\
		m4 = _mm256_mul_ps(a6, b4);\
		m5 = _mm256_mul_ps(a6, b5);\
		m6 = _mm256_mul_ps(a6, b6);\
		m7 = _mm256_mul_ps(a6, b7);\
		\
		t0 = _mm256_hadd_ps(m0, m1);\
		t1 = _mm256_hadd_ps(m2, m3);\
		t2 = _mm256_hadd_ps(m4, m5);\
		t3 = _mm256_hadd_ps(m6, m7);\
		\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		t2 = _mm256_permutevar8x32_ps(t2, p);\
		t3 = _mm256_permutevar8x32_ps(t3, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t1 = _mm256_hadd_ps(t2, t3);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		\
		c6 = _mm256_add_ps(c6, t0);\
		\
		m0 = _mm256_mul_ps(a7, b0);\
		m1 = _mm256_mul_ps(a7, b1);\
		m2 = _mm256_mul_ps(a7, b2);\
		m3 = _mm256_mul_ps(a7, b3);\
		m4 = _mm256_mul_ps(a7, b4);\
		m5 = _mm256_mul_ps(a7, b5);\
		m6 = _mm256_mul_ps(a7, b6);\
		m7 = _mm256_mul_ps(a7, b7);\
		\
		t0 = _mm256_hadd_ps(m0, m1);\
		t1 = _mm256_hadd_ps(m2, m3);\
		t2 = _mm256_hadd_ps(m4, m5);\
		t3 = _mm256_hadd_ps(m6, m7);\
		\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		t2 = _mm256_permutevar8x32_ps(t2, p);\
		t3 = _mm256_permutevar8x32_ps(t3, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t1 = _mm256_hadd_ps(t2, t3);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		t1 = _mm256_permutevar8x32_ps(t1, p);\
		\
		t0 = _mm256_hadd_ps(t0, t1);\
		t0 = _mm256_permutevar8x32_ps(t0, p);\
		\
		c7 = _mm256_add_ps(c7, t0); \
	}

	#define _MM512_MMUL8_PD(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5, c6, c7) \
	{\
		alignas(64)__m512d t0, t1, t2, t3, t4, t5, t6, t7, c; \
		\
		t0 = _mm512_mul_pd(a0, b0); \
		t1 = _mm512_mul_pd(a0, b1); \
		t2 = _mm512_mul_pd(a0, b2); \
		t3 = _mm512_mul_pd(a0, b3); \
		t4 = _mm512_mul_pd(a0, b4); \
		t5 = _mm512_mul_pd(a0, b5); \
		t6 = _mm512_mul_pd(a0, b6); \
		t7 = _mm512_mul_pd(a0, b7); \
		\
		c = _mm512_set_pd( \
			_mm512_reduce_add_pd(t7), \
			_mm512_reduce_add_pd(t6), \
			_mm512_reduce_add_pd(t5), \
			_mm512_reduce_add_pd(t4), \
			_mm512_reduce_add_pd(t3), \
			_mm512_reduce_add_pd(t2), \
			_mm512_reduce_add_pd(t1), \
			_mm512_reduce_add_pd(t0)); \
		c0 = _mm512_add_pd(c0, c); \
		\
		t0 = _mm512_mul_pd(a1, b0); \
		t1 = _mm512_mul_pd(a1, b1); \
		t2 = _mm512_mul_pd(a1, b2); \
		t3 = _mm512_mul_pd(a1, b3); \
		t4 = _mm512_mul_pd(a1, b4); \
		t5 = _mm512_mul_pd(a1, b5); \
		t6 = _mm512_mul_pd(a1, b6); \
		t7 = _mm512_mul_pd(a1, b7); \
		\
		c = _mm512_set_pd( \
			_mm512_reduce_add_pd(t7), \
			_mm512_reduce_add_pd(t6), \
			_mm512_reduce_add_pd(t5), \
			_mm512_reduce_add_pd(t4), \
			_mm512_reduce_add_pd(t3), \
			_mm512_reduce_add_pd(t2), \
			_mm512_reduce_add_pd(t1), \
			_mm512_reduce_add_pd(t0)); \
		c1 = _mm512_add_pd(c1, c); \
		\
		t0 = _mm512_mul_pd(a2, b0); \
		t1 = _mm512_mul_pd(a2, b1); \
		t2 = _mm512_mul_pd(a2, b2); \
		t3 = _mm512_mul_pd(a2, b3); \
		t4 = _mm512_mul_pd(a2, b4); \
		t5 = _mm512_mul_pd(a2, b5); \
		t6 = _mm512_mul_pd(a2, b6); \
		t7 = _mm512_mul_pd(a2, b7); \
		\
		c = _mm512_set_pd( \
			_mm512_reduce_add_pd(t7), \
			_mm512_reduce_add_pd(t6), \
			_mm512_reduce_add_pd(t5), \
			_mm512_reduce_add_pd(t4), \
			_mm512_reduce_add_pd(t3), \
			_mm512_reduce_add_pd(t2), \
			_mm512_reduce_add_pd(t1), \
			_mm512_reduce_add_pd(t0)); \
		c2 = _mm512_add_pd(c2, c); \
		\
		t0 = _mm512_mul_pd(a3, b0); \
		t1 = _mm512_mul_pd(a3, b1); \
		t2 = _mm512_mul_pd(a3, b2); \
		t3 = _mm512_mul_pd(a3, b3); \
		t4 = _mm512_mul_pd(a3, b4); \
		t5 = _mm512_mul_pd(a3, b5); \
		t6 = _mm512_mul_pd(a3, b6); \
		t7 = _mm512_mul_pd(a3, b7); \
		\
		c = _mm512_set_pd( \
			_mm512_reduce_add_pd(t7), \
			_mm512_reduce_add_pd(t6), \
			_mm512_reduce_add_pd(t5), \
			_mm512_reduce_add_pd(t4), \
			_mm512_reduce_add_pd(t3), \
			_mm512_reduce_add_pd(t2), \
			_mm512_reduce_add_pd(t1), \
			_mm512_reduce_add_pd(t0)); \
		c3 = _mm512_add_pd(c3, c); \
		\
		t0 = _mm512_mul_pd(a4, b0); \
		t1 = _mm512_mul_pd(a4, b1); \
		t2 = _mm512_mul_pd(a4, b2); \
		t3 = _mm512_mul_pd(a4, b3); \
		t4 = _mm512_mul_pd(a4, b4); \
		t5 = _mm512_mul_pd(a4, b5); \
		t6 = _mm512_mul_pd(a4, b6); \
		t7 = _mm512_mul_pd(a4, b7); \
		\
		c = _mm512_set_pd( \
			_mm512_reduce_add_pd(t7), \
			_mm512_reduce_add_pd(t6), \
			_mm512_reduce_add_pd(t5), \
			_mm512_reduce_add_pd(t4), \
			_mm512_reduce_add_pd(t3), \
			_mm512_reduce_add_pd(t2), \
			_mm512_reduce_add_pd(t1), \
			_mm512_reduce_add_pd(t0)); \
		c4 = _mm512_add_pd(c4, c); \
		\
		t0 = _mm512_mul_pd(a5, b0); \
		t1 = _mm512_mul_pd(a5, b1); \
		t2 = _mm512_mul_pd(a5, b2); \
		t3 = _mm512_mul_pd(a5, b3); \
		t4 = _mm512_mul_pd(a5, b4); \
		t5 = _mm512_mul_pd(a5, b5); \
		t6 = _mm512_mul_pd(a5, b6); \
		t7 = _mm512_mul_pd(a5, b7); \
		\
		c = _mm512_set_pd( \
			_mm512_reduce_add_pd(t7), \
			_mm512_reduce_add_pd(t6), \
			_mm512_reduce_add_pd(t5), \
			_mm512_reduce_add_pd(t4), \
			_mm512_reduce_add_pd(t3), \
			_mm512_reduce_add_pd(t2), \
			_mm512_reduce_add_pd(t1), \
			_mm512_reduce_add_pd(t0)); \
		c5 = _mm512_add_pd(c5, c); \
		\
		t0 = _mm512_mul_pd(a6, b0); \
		t1 = _mm512_mul_pd(a6, b1); \
		t2 = _mm512_mul_pd(a6, b2); \
		t3 = _mm512_mul_pd(a6, b3); \
		t4 = _mm512_mul_pd(a6, b4); \
		t5 = _mm512_mul_pd(a6, b5); \
		t6 = _mm512_mul_pd(a6, b6); \
		t7 = _mm512_mul_pd(a6, b7); \
		\
		c = _mm512_set_pd( \
			_mm512_reduce_add_pd(t7), \
			_mm512_reduce_add_pd(t6), \
			_mm512_reduce_add_pd(t5), \
			_mm512_reduce_add_pd(t4), \
			_mm512_reduce_add_pd(t3), \
			_mm512_reduce_add_pd(t2), \
			_mm512_reduce_add_pd(t1), \
			_mm512_reduce_add_pd(t0)); \
		c6 = _mm512_add_pd(c6, c); \
		\
		t0 = _mm512_mul_pd(a7, b0); \
		t1 = _mm512_mul_pd(a7, b1); \
		t2 = _mm512_mul_pd(a7, b2); \
		t3 = _mm512_mul_pd(a7, b3); \
		t4 = _mm512_mul_pd(a7, b4); \
		t5 = _mm512_mul_pd(a7, b5); \
		t6 = _mm512_mul_pd(a7, b6); \
		t7 = _mm512_mul_pd(a7, b7); \
		\
		c = _mm512_set_pd( \
			_mm512_reduce_add_pd(t7), \
			_mm512_reduce_add_pd(t6), \
			_mm512_reduce_add_pd(t5), \
			_mm512_reduce_add_pd(t4), \
			_mm512_reduce_add_pd(t3), \
			_mm512_reduce_add_pd(t2), \
			_mm512_reduce_add_pd(t1), \
			_mm512_reduce_add_pd(t0)); \
		c7 = _mm512_add_pd(c7, c);\
	}

	#define _MM512_MMUL16_PS(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb, cc, cd, ce, cf) \
	{\
		alignas(64) __m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, c; \
		\
		t0 = _mm512_mul_ps(a0, b0); \
		t1 = _mm512_mul_ps(a0, b1); \
		t2 = _mm512_mul_ps(a0, b2); \
		t3 = _mm512_mul_ps(a0, b3); \
		t4 = _mm512_mul_ps(a0, b4); \
		t5 = _mm512_mul_ps(a0, b5); \
		t6 = _mm512_mul_ps(a0, b6); \
		t7 = _mm512_mul_ps(a0, b7); \
		t8 = _mm512_mul_ps(a0, b8); \
		t9 = _mm512_mul_ps(a0, b9); \
		ta = _mm512_mul_ps(a0, ba); \
		tb = _mm512_mul_ps(a0, bb); \
		tc = _mm512_mul_ps(a0, bc); \
		td = _mm512_mul_ps(a0, bd); \
		te = _mm512_mul_ps(a0, be); \
		tf = _mm512_mul_ps(a0, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c0 = _mm512_add_ps(c0, c); \
		\
		t0 = _mm512_mul_ps(a1, b0); \
		t1 = _mm512_mul_ps(a1, b1); \
		t2 = _mm512_mul_ps(a1, b2); \
		t3 = _mm512_mul_ps(a1, b3); \
		t4 = _mm512_mul_ps(a1, b4); \
		t5 = _mm512_mul_ps(a1, b5); \
		t6 = _mm512_mul_ps(a1, b6); \
		t7 = _mm512_mul_ps(a1, b7); \
		t8 = _mm512_mul_ps(a1, b8); \
		t9 = _mm512_mul_ps(a1, b9); \
		ta = _mm512_mul_ps(a1, ba); \
		tb = _mm512_mul_ps(a1, bb); \
		tc = _mm512_mul_ps(a1, bc); \
		td = _mm512_mul_ps(a1, bd); \
		te = _mm512_mul_ps(a1, be); \
		tf = _mm512_mul_ps(a1, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c1 = _mm512_add_ps(c1, c); \
		\
		t0 = _mm512_mul_ps(a2, b0); \
		t1 = _mm512_mul_ps(a2, b1); \
		t2 = _mm512_mul_ps(a2, b2); \
		t3 = _mm512_mul_ps(a2, b3); \
		t4 = _mm512_mul_ps(a2, b4); \
		t5 = _mm512_mul_ps(a2, b5); \
		t6 = _mm512_mul_ps(a2, b6); \
		t7 = _mm512_mul_ps(a2, b7); \
		t8 = _mm512_mul_ps(a2, b8); \
		t9 = _mm512_mul_ps(a2, b9); \
		ta = _mm512_mul_ps(a2, ba); \
		tb = _mm512_mul_ps(a2, bb); \
		tc = _mm512_mul_ps(a2, bc); \
		td = _mm512_mul_ps(a2, bd); \
		te = _mm512_mul_ps(a2, be); \
		tf = _mm512_mul_ps(a2, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c2 = _mm512_add_ps(c2, c); \
		\
		t0 = _mm512_mul_ps(a3, b0); \
		t1 = _mm512_mul_ps(a3, b1); \
		t2 = _mm512_mul_ps(a3, b2); \
		t3 = _mm512_mul_ps(a3, b3); \
		t4 = _mm512_mul_ps(a3, b4); \
		t5 = _mm512_mul_ps(a3, b5); \
		t6 = _mm512_mul_ps(a3, b6); \
		t7 = _mm512_mul_ps(a3, b7); \
		t8 = _mm512_mul_ps(a3, b8); \
		t9 = _mm512_mul_ps(a3, b9); \
		ta = _mm512_mul_ps(a3, ba); \
		tb = _mm512_mul_ps(a3, bb); \
		tc = _mm512_mul_ps(a3, bc); \
		td = _mm512_mul_ps(a3, bd); \
		te = _mm512_mul_ps(a3, be); \
		tf = _mm512_mul_ps(a3, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c3 = _mm512_add_ps(c3, c); \
		\
		t0 = _mm512_mul_ps(a4, b0); \
		t1 = _mm512_mul_ps(a4, b1); \
		t2 = _mm512_mul_ps(a4, b2); \
		t3 = _mm512_mul_ps(a4, b3); \
		t4 = _mm512_mul_ps(a4, b4); \
		t5 = _mm512_mul_ps(a4, b5); \
		t6 = _mm512_mul_ps(a4, b6); \
		t7 = _mm512_mul_ps(a4, b7); \
		t8 = _mm512_mul_ps(a4, b8); \
		t9 = _mm512_mul_ps(a4, b9); \
		ta = _mm512_mul_ps(a4, ba); \
		tb = _mm512_mul_ps(a4, bb); \
		tc = _mm512_mul_ps(a4, bc); \
		td = _mm512_mul_ps(a4, bd); \
		te = _mm512_mul_ps(a4, be); \
		tf = _mm512_mul_ps(a4, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c4 = _mm512_add_ps(c4, c); \
		\
		t0 = _mm512_mul_ps(a5, b0); \
		t1 = _mm512_mul_ps(a5, b1); \
		t2 = _mm512_mul_ps(a5, b2); \
		t3 = _mm512_mul_ps(a5, b3); \
		t4 = _mm512_mul_ps(a5, b4); \
		t5 = _mm512_mul_ps(a5, b5); \
		t6 = _mm512_mul_ps(a5, b6); \
		t7 = _mm512_mul_ps(a5, b7); \
		t8 = _mm512_mul_ps(a5, b8); \
		t9 = _mm512_mul_ps(a5, b9); \
		ta = _mm512_mul_ps(a5, ba); \
		tb = _mm512_mul_ps(a5, bb); \
		tc = _mm512_mul_ps(a5, bc); \
		td = _mm512_mul_ps(a5, bd); \
		te = _mm512_mul_ps(a5, be); \
		tf = _mm512_mul_ps(a5, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c5 = _mm512_add_ps(c5, c); \
		\
		t0 = _mm512_mul_ps(a6, b0); \
		t1 = _mm512_mul_ps(a6, b1); \
		t2 = _mm512_mul_ps(a6, b2); \
		t3 = _mm512_mul_ps(a6, b3); \
		t4 = _mm512_mul_ps(a6, b4); \
		t5 = _mm512_mul_ps(a6, b5); \
		t6 = _mm512_mul_ps(a6, b6); \
		t7 = _mm512_mul_ps(a6, b7); \
		t8 = _mm512_mul_ps(a6, b8); \
		t9 = _mm512_mul_ps(a6, b9); \
		ta = _mm512_mul_ps(a6, ba); \
		tb = _mm512_mul_ps(a6, bb); \
		tc = _mm512_mul_ps(a6, bc); \
		td = _mm512_mul_ps(a6, bd); \
		te = _mm512_mul_ps(a6, be); \
		tf = _mm512_mul_ps(a6, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c6 = _mm512_add_ps(c6, c); \
		\
		t0 = _mm512_mul_ps(a7, b0); \
		t1 = _mm512_mul_ps(a7, b1); \
		t2 = _mm512_mul_ps(a7, b2); \
		t3 = _mm512_mul_ps(a7, b3); \
		t4 = _mm512_mul_ps(a7, b4); \
		t5 = _mm512_mul_ps(a7, b5); \
		t6 = _mm512_mul_ps(a7, b6); \
		t7 = _mm512_mul_ps(a7, b7); \
		t8 = _mm512_mul_ps(a7, b8); \
		t9 = _mm512_mul_ps(a7, b9); \
		ta = _mm512_mul_ps(a7, ba); \
		tb = _mm512_mul_ps(a7, bb); \
		tc = _mm512_mul_ps(a7, bc); \
		td = _mm512_mul_ps(a7, bd); \
		te = _mm512_mul_ps(a7, be); \
		tf = _mm512_mul_ps(a7, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c7 = _mm512_add_ps(c7, c); \
		\
		t0 = _mm512_mul_ps(a8, b0); \
		t1 = _mm512_mul_ps(a8, b1); \
		t2 = _mm512_mul_ps(a8, b2); \
		t3 = _mm512_mul_ps(a8, b3); \
		t4 = _mm512_mul_ps(a8, b4); \
		t5 = _mm512_mul_ps(a8, b5); \
		t6 = _mm512_mul_ps(a8, b6); \
		t7 = _mm512_mul_ps(a8, b7); \
		t8 = _mm512_mul_ps(a8, b8); \
		t9 = _mm512_mul_ps(a8, b9); \
		ta = _mm512_mul_ps(a8, ba); \
		tb = _mm512_mul_ps(a8, bb); \
		tc = _mm512_mul_ps(a8, bc); \
		td = _mm512_mul_ps(a8, bd); \
		te = _mm512_mul_ps(a8, be); \
		tf = _mm512_mul_ps(a8, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c8 = _mm512_add_ps(c8, c); \
		\
		t0 = _mm512_mul_ps(a9, b0); \
		t1 = _mm512_mul_ps(a9, b1); \
		t2 = _mm512_mul_ps(a9, b2); \
		t3 = _mm512_mul_ps(a9, b3); \
		t4 = _mm512_mul_ps(a9, b4); \
		t5 = _mm512_mul_ps(a9, b5); \
		t6 = _mm512_mul_ps(a9, b6); \
		t7 = _mm512_mul_ps(a9, b7); \
		t8 = _mm512_mul_ps(a9, b8); \
		t9 = _mm512_mul_ps(a9, b9); \
		ta = _mm512_mul_ps(a9, ba); \
		tb = _mm512_mul_ps(a9, bb); \
		tc = _mm512_mul_ps(a9, bc); \
		td = _mm512_mul_ps(a9, bd); \
		te = _mm512_mul_ps(a9, be); \
		tf = _mm512_mul_ps(a9, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		c9 = _mm512_add_ps(c9, c); \
		\
		t0 = _mm512_mul_ps(aa, b0); \
		t1 = _mm512_mul_ps(aa, b1); \
		t2 = _mm512_mul_ps(aa, b2); \
		t3 = _mm512_mul_ps(aa, b3); \
		t4 = _mm512_mul_ps(aa, b4); \
		t5 = _mm512_mul_ps(aa, b5); \
		t6 = _mm512_mul_ps(aa, b6); \
		t7 = _mm512_mul_ps(aa, b7); \
		t8 = _mm512_mul_ps(aa, b8); \
		t9 = _mm512_mul_ps(aa, b9); \
		ta = _mm512_mul_ps(aa, ba); \
		tb = _mm512_mul_ps(aa, bb); \
		tc = _mm512_mul_ps(aa, bc); \
		td = _mm512_mul_ps(aa, bd); \
		te = _mm512_mul_ps(aa, be); \
		tf = _mm512_mul_ps(aa, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		ca = _mm512_add_ps(ca, c); \
		\
		t0 = _mm512_mul_ps(ab, b0); \
		t1 = _mm512_mul_ps(ab, b1); \
		t2 = _mm512_mul_ps(ab, b2); \
		t3 = _mm512_mul_ps(ab, b3); \
		t4 = _mm512_mul_ps(ab, b4); \
		t5 = _mm512_mul_ps(ab, b5); \
		t6 = _mm512_mul_ps(ab, b6); \
		t7 = _mm512_mul_ps(ab, b7); \
		t8 = _mm512_mul_ps(ab, b8); \
		t9 = _mm512_mul_ps(ab, b9); \
		ta = _mm512_mul_ps(ab, ba); \
		tb = _mm512_mul_ps(ab, bb); \
		tc = _mm512_mul_ps(ab, bc); \
		td = _mm512_mul_ps(ab, bd); \
		te = _mm512_mul_ps(ab, be); \
		tf = _mm512_mul_ps(ab, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		cb = _mm512_add_ps(cb, c); \
		\
		t0 = _mm512_mul_ps(ac, b0); \
		t1 = _mm512_mul_ps(ac, b1); \
		t2 = _mm512_mul_ps(ac, b2); \
		t3 = _mm512_mul_ps(ac, b3); \
		t4 = _mm512_mul_ps(ac, b4); \
		t5 = _mm512_mul_ps(ac, b5); \
		t6 = _mm512_mul_ps(ac, b6); \
		t7 = _mm512_mul_ps(ac, b7); \
		t8 = _mm512_mul_ps(ac, b8); \
		t9 = _mm512_mul_ps(ac, b9); \
		ta = _mm512_mul_ps(ac, ba); \
		tb = _mm512_mul_ps(ac, bb); \
		tc = _mm512_mul_ps(ac, bc); \
		td = _mm512_mul_ps(ac, bd); \
		te = _mm512_mul_ps(ac, be); \
		tf = _mm512_mul_ps(ac, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		cc = _mm512_add_ps(cc, c); \
		\
		t0 = _mm512_mul_ps(ad, b0); \
		t1 = _mm512_mul_ps(ad, b1); \
		t2 = _mm512_mul_ps(ad, b2); \
		t3 = _mm512_mul_ps(ad, b3); \
		t4 = _mm512_mul_ps(ad, b4); \
		t5 = _mm512_mul_ps(ad, b5); \
		t6 = _mm512_mul_ps(ad, b6); \
		t7 = _mm512_mul_ps(ad, b7); \
		t8 = _mm512_mul_ps(ad, b8); \
		t9 = _mm512_mul_ps(ad, b9); \
		ta = _mm512_mul_ps(ad, ba); \
		tb = _mm512_mul_ps(ad, bb); \
		tc = _mm512_mul_ps(ad, bc); \
		td = _mm512_mul_ps(ad, bd); \
		te = _mm512_mul_ps(ad, be); \
		tf = _mm512_mul_ps(ad, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		cd = _mm512_add_ps(cd, c); \
		\
		t0 = _mm512_mul_ps(ae, b0); \
		t1 = _mm512_mul_ps(ae, b1); \
		t2 = _mm512_mul_ps(ae, b2); \
		t3 = _mm512_mul_ps(ae, b3); \
		t4 = _mm512_mul_ps(ae, b4); \
		t5 = _mm512_mul_ps(ae, b5); \
		t6 = _mm512_mul_ps(ae, b6); \
		t7 = _mm512_mul_ps(ae, b7); \
		t8 = _mm512_mul_ps(ae, b8); \
		t9 = _mm512_mul_ps(ae, b9); \
		ta = _mm512_mul_ps(ae, ba); \
		tb = _mm512_mul_ps(ae, bb); \
		tc = _mm512_mul_ps(ae, bc); \
		td = _mm512_mul_ps(ae, bd); \
		te = _mm512_mul_ps(ae, be); \
		tf = _mm512_mul_ps(ae, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		ce = _mm512_add_ps(ce, c); \
		\
		t0 = _mm512_mul_ps(af, b0); \
		t1 = _mm512_mul_ps(af, b1); \
		t2 = _mm512_mul_ps(af, b2); \
		t3 = _mm512_mul_ps(af, b3); \
		t4 = _mm512_mul_ps(af, b4); \
		t5 = _mm512_mul_ps(af, b5); \
		t6 = _mm512_mul_ps(af, b6); \
		t7 = _mm512_mul_ps(af, b7); \
		t8 = _mm512_mul_ps(af, b8); \
		t9 = _mm512_mul_ps(af, b9); \
		ta = _mm512_mul_ps(af, ba); \
		tb = _mm512_mul_ps(af, bb); \
		tc = _mm512_mul_ps(af, bc); \
		td = _mm512_mul_ps(af, bd); \
		te = _mm512_mul_ps(af, be); \
		tf = _mm512_mul_ps(af, bf); \
		\
		c = _mm512_set_ps( \
			_mm512_reduce_add_ps(tf), \
			_mm512_reduce_add_ps(te), \
			_mm512_reduce_add_ps(td), \
			_mm512_reduce_add_ps(tc), \
			_mm512_reduce_add_ps(tb), \
			_mm512_reduce_add_ps(ta), \
			_mm512_reduce_add_ps(t9), \
			_mm512_reduce_add_ps(t8), \
			_mm512_reduce_add_ps(t7), \
			_mm512_reduce_add_ps(t6), \
			_mm512_reduce_add_ps(t5), \
			_mm512_reduce_add_ps(t4), \
			_mm512_reduce_add_ps(t3), \
			_mm512_reduce_add_ps(t2), \
			_mm512_reduce_add_ps(t1), \
			_mm512_reduce_add_ps(t0)); \
		cf = _mm512_add_ps(cf, c); \
	}


	template <>
	void 
	_multiply_block_sse<float>(float* A, float* B, float* C, const size_t M, const size_t N, const size_t P)
	{
		alignas(16)__m128 a0, a1, a2, a3;
		if ((M * sizeof(float)) % 16 == 0)
		{
			a0 = _mm_load_ps(&A[0 * M]);
			a1 = _mm_load_ps(&A[1 * M]);
			a2 = _mm_load_ps(&A[2 * M]);
			a3 = _mm_load_ps(&A[3 * M]);
		}
		else
		{
			a0 = _mm_loadu_ps(&A[0 * M]);
			a1 = _mm_loadu_ps(&A[1 * M]);
			a2 = _mm_loadu_ps(&A[2 * M]);
			a3 = _mm_loadu_ps(&A[3 * M]);
		}
		
		alignas(16)__m128 b0, b1, b2, b3;
		if ((N * sizeof(float)) % 16 == 0)
		{
			b0 = _mm_load_ps(&B[0 * N]);
			b1 = _mm_load_ps(&B[1 * N]);
			b2 = _mm_load_ps(&B[2 * N]);
			b3 = _mm_load_ps(&B[3 * N]);
		}
		else
		{
			b0 = _mm_loadu_ps(&B[0 * N]);
			b1 = _mm_loadu_ps(&B[1 * N]);
			b2 = _mm_loadu_ps(&B[2 * N]);
			b3 = _mm_loadu_ps(&B[3 * N]);
		}
		
		alignas(16)__m128 c0, c1, c2, c3;
		if ((P * sizeof(float)) % 16 == 0)
		{
			c0 = _mm_load_ps(&C[0 * P]);
			c1 = _mm_load_ps(&C[1 * P]);
			c2 = _mm_load_ps(&C[2 * P]);
			c3 = _mm_load_ps(&C[3 * P]);
		}
		else
		{
			c0 = _mm_loadu_ps(&C[0 * P]);
			c1 = _mm_loadu_ps(&C[1 * P]);
			c2 = _mm_loadu_ps(&C[2 * P]);
			c3 = _mm_loadu_ps(&C[3 * P]);
		}

		_MM_MMUL4_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3)
		
		if ((P * sizeof(float)) % 16 == 0)
		{
			_mm_store_ps(&C[0 * P], c0);
			_mm_store_ps(&C[1 * P], c1);
			_mm_store_ps(&C[2 * P], c2);
			_mm_store_ps(&C[3 * P], c3);
		}
		else
		{
			_mm_storeu_ps(&C[0 * P], c0);
			_mm_storeu_ps(&C[1 * P], c1);
			_mm_storeu_ps(&C[2 * P], c2);
			_mm_storeu_ps(&C[3 * P], c3);
		}
	}

	template <>
	void 
	_multiply_block_sse<std::complex<float>>(std::complex<float>* A, std::complex<float>* B, std::complex<float>* C, const size_t M, const size_t N, const size_t P)
	{
		float* A_intlv = reinterpret_cast<float*>(A);
		float* B_intlv = reinterpret_cast<float*>(B);
		float* C_intlv = reinterpret_cast<float*>(C);

		alignas(16)__m128 a0, a1;
		alignas(16)__m128 b0, b1;
		alignas(16)__m128 c0, c1;

		if ((M * sizeof(std::complex<float>)) % 16 == 0)
		{
			a0 = _mm_load_ps(&A_intlv[0 * M * 2]);
			a1 = _mm_load_ps(&A_intlv[1 * M * 2]);
		}
		else
		{
			a0 = _mm_loadu_ps(&A_intlv[0 * M * 2]);
			a1 = _mm_loadu_ps(&A_intlv[1 * M * 2]);
		}
		
		if ((N * sizeof(std::complex<float>)) % 16 == 0)
		{
			b0 = _mm_load_ps(&B_intlv[0 * N * 2]);
			b1 = _mm_load_ps(&B_intlv[1 * N * 2]);
		}
		else
		{
			b0 = _mm_loadu_ps(&B_intlv[0 * N * 2]);
			b1 = _mm_loadu_ps(&B_intlv[1 * N * 2]);
		}
		
		if ((P * sizeof(std::complex<float>)) % 16 == 0)
		{
			c0 = _mm_load_ps(&C_intlv[0 * P * 2]);
			c1 = _mm_load_ps(&C_intlv[1 * P * 2]);
		}
		else
		{
			c0 = _mm_loadu_ps(&C_intlv[0 * P * 2]);
			c1 = _mm_loadu_ps(&C_intlv[1 * P * 2]);
		}

		_MM_MMUL2_C_PS(a0, a1, b0, b1, c0, c1)

		if ((P * sizeof(std::complex<float>)) % 16 == 0)
		{
			_mm_store_ps(&C_intlv[0 * P * 2], c0);
			_mm_store_ps(&C_intlv[1 * P * 2], c1);
		}
		else
		{
			_mm_storeu_ps(&C_intlv[0 * P * 2], c0);
			_mm_storeu_ps(&C_intlv[1 * P * 2], c1);
		}
	}

	template <>
	void 
	_multiply_block_sse<double>(double* A, double* B, double* C, const size_t M, const size_t N, const size_t P)
	{
		alignas(16)__m128d a0, a1;
		if ((M * sizeof(double)) % 16 == 0)
		{
			a0 = _mm_load_pd(&A[0 * M]);
			a1 = _mm_load_pd(&A[1 * M]);
		}
		else
		{
			a0 = _mm_loadu_pd(&A[0 * M]);
			a1 = _mm_loadu_pd(&A[1 * M]);
		}
		
		alignas(16)__m128d b0, b1;
		if ((N * sizeof(double)) % 16 == 0)
		{
			b0 = _mm_load_pd(&B[0 * N]);
			b1 = _mm_load_pd(&B[1 * N]);
		}
		else
		{
			b0 = _mm_loadu_pd(&B[0 * N]);
			b1 = _mm_loadu_pd(&B[1 * N]);
		}
		
		alignas(16)__m128d c0, c1;
		if ((P * sizeof(double)) % 16 == 0)
		{
			c0 = _mm_load_pd(&C[0 * P]);
			c1 = _mm_load_pd(&C[1 * P]);
		}
		else
		{
			c0 = _mm_loadu_pd(&C[0 * P]);
			c1 = _mm_loadu_pd(&C[1 * P]);
		}
		
		_MM_MMUL2_PD(a0, a1, b0, b1, c0, c1);
		
		if ((P * sizeof(double)) % 16 == 0)
		{
			_mm_store_pd(&C[0 * P], c0);
			_mm_store_pd(&C[1 * P], c1);
		}
		else
		{
			_mm_storeu_pd(&C[0 * P], c0);
			_mm_storeu_pd(&C[1 * P], c1);
		}
	}

	template <>
	void 
	_multiply_block_sse<std::complex<double>>(std::complex<double>* A, std::complex<double>* B, std::complex<double>* C, const size_t M, const size_t N, const size_t P)
	{
		double* A_intlv = reinterpret_cast<double*>(A);
		double* B_intlv = reinterpret_cast<double*>(B);
		double* C_intlv = reinterpret_cast<double*>(C);

		alignas(16)__m128d a0;
		alignas(16)__m128d b0;
		alignas(16)__m128d c0;

		if ((M * sizeof(std::complex<double>)) % 16 == 0)
		{
			a0 = _mm_load_pd(&A_intlv[0 * M * 2]);
		}
		else
		{
			a0 = _mm_loadu_pd(&A_intlv[0 * M * 2]);
		}
		
		if ((N * sizeof(std::complex<double>)) % 16 == 0)
		{
			b0 = _mm_load_pd(&B_intlv[0 * N * 2]);
		}
		else
		{
			b0 = _mm_loadu_pd(&B_intlv[0 * N * 2]);
		}
		
		if ((P * sizeof(std::complex<double>)) % 16 == 0)
		{
			c0 = _mm_load_pd(&C_intlv[0 * P * 2]);
		}
		else
		{
			c0 = _mm_loadu_pd(&C_intlv[0 * P * 2]);
		}

		_MM_MMUL_C_PD(a0, b0, c0)

		if ((P * sizeof(std::complex<double>)) % 16 == 0)
		{
			_mm_store_pd(&C_intlv[0 * P * 2], c0);
		}
		else
		{
			_mm_storeu_pd(&C_intlv[0 * P * 2], c0);
		}
	}


	template <>
	void 
	_multiply_block_avx<float>(float* A, float* B, float* C, const size_t M, const size_t N, const size_t P)
	{
		alignas(32)__m256 a0, a1, a2, a3, a4, a5, a6, a7;
		if ((M * sizeof(float)) % 32 == 0)
		{
			a0 = _mm256_load_ps(&A[0 * M]);
			a1 = _mm256_load_ps(&A[1 * M]);
			a2 = _mm256_load_ps(&A[2 * M]);
			a3 = _mm256_load_ps(&A[3 * M]);
			a4 = _mm256_load_ps(&A[4 * M]);
			a5 = _mm256_load_ps(&A[5 * M]);
			a6 = _mm256_load_ps(&A[6 * M]);
			a7 = _mm256_load_ps(&A[7 * M]);

		}
		else
		{
			a0 = _mm256_loadu_ps(&A[0 * M]);
			a1 = _mm256_loadu_ps(&A[1 * M]);
			a2 = _mm256_loadu_ps(&A[2 * M]);
			a3 = _mm256_loadu_ps(&A[3 * M]);
			a4 = _mm256_loadu_ps(&A[4 * M]);
			a5 = _mm256_loadu_ps(&A[5 * M]);
			a6 = _mm256_loadu_ps(&A[6 * M]);
			a7 = _mm256_loadu_ps(&A[7 * M]);
		}
		
		alignas(32)__m256 b0, b1, b2, b3, b4, b5, b6, b7;
		if ((N * sizeof(float)) % 32 == 0)
		{
			b0 = _mm256_load_ps(&B[0 * N]);
			b1 = _mm256_load_ps(&B[1 * N]);
			b2 = _mm256_load_ps(&B[2 * N]);
			b3 = _mm256_load_ps(&B[3 * N]);
			b4 = _mm256_load_ps(&B[4 * N]);
			b5 = _mm256_load_ps(&B[5 * N]);
			b6 = _mm256_load_ps(&B[6 * N]);
			b7 = _mm256_load_ps(&B[7 * N]);
		}
		else
		{
			b0 = _mm256_loadu_ps(&B[0 * N]);
			b1 = _mm256_loadu_ps(&B[1 * N]);
			b2 = _mm256_loadu_ps(&B[2 * N]);
			b3 = _mm256_loadu_ps(&B[3 * N]);
			b4 = _mm256_loadu_ps(&B[4 * N]);
			b5 = _mm256_loadu_ps(&B[5 * N]);
			b6 = _mm256_loadu_ps(&B[6 * N]);
			b7 = _mm256_loadu_ps(&B[7 * N]);
		}
		
		alignas(32)__m256 c0, c1, c2, c3, c4, c5, c6, c7;
		if ((P * sizeof(float)) % 32 == 0)
		{
			c0 = _mm256_load_ps(&C[0 * P]);
			c1 = _mm256_load_ps(&C[1 * P]);
			c2 = _mm256_load_ps(&C[2 * P]);
			c3 = _mm256_load_ps(&C[3 * P]);
			c4 = _mm256_load_ps(&C[4 * P]);
			c5 = _mm256_load_ps(&C[5 * P]);
			c6 = _mm256_load_ps(&C[6 * P]);
			c7 = _mm256_load_ps(&C[7 * P]);
		}
		else
		{
			c0 = _mm256_loadu_ps(&C[0 * P]);
			c1 = _mm256_loadu_ps(&C[1 * P]);
			c2 = _mm256_loadu_ps(&C[2 * P]);
			c3 = _mm256_loadu_ps(&C[3 * P]);
			c4 = _mm256_loadu_ps(&C[4 * P]);
			c5 = _mm256_loadu_ps(&C[5 * P]);
			c6 = _mm256_loadu_ps(&C[6 * P]);
			c7 = _mm256_loadu_ps(&C[7 * P]);
		}

		_MM256_MMUL8_PS(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5, c6, c7);
		
		if ((P * sizeof(float)) % 32 == 0)
		{
			_mm256_store_ps(&C[0 * P], c0);
			_mm256_store_ps(&C[1 * P], c1);
			_mm256_store_ps(&C[2 * P], c2);
			_mm256_store_ps(&C[3 * P], c3);
			_mm256_store_ps(&C[4 * P], c4);
			_mm256_store_ps(&C[5 * P], c5);
			_mm256_store_ps(&C[6 * P], c6);
			_mm256_store_ps(&C[7 * P], c7);
		}
		else
		{
			_mm256_storeu_ps(&C[0 * P], c0);
			_mm256_storeu_ps(&C[1 * P], c1);
			_mm256_storeu_ps(&C[2 * P], c2);
			_mm256_storeu_ps(&C[3 * P], c3);
			_mm256_storeu_ps(&C[4 * P], c4);
			_mm256_storeu_ps(&C[5 * P], c5);
			_mm256_storeu_ps(&C[6 * P], c6);
			_mm256_storeu_ps(&C[7 * P], c7);
		}
	}

	template <>
	void 
	_multiply_block_avx<std::complex<float>>(std::complex<float>* A, std::complex<float>* B, std::complex<float>* C, const size_t M, const size_t N, const size_t P)
	{
		float* A_intlv = reinterpret_cast<float*>(A);
		float* B_intlv = reinterpret_cast<float*>(B);
		float* C_intlv = reinterpret_cast<float*>(C);

		alignas(32)__m256 a0, a1, a2, a3;
		alignas(32)__m256 b0, b1, b2, b3;
		alignas(32)__m256 c0, c1, c2, c3;

		if ((M * sizeof(std::complex<float>)) % 32 == 0)
		{
			a0 = _mm256_load_ps(&A_intlv[0 * M * 2]);
			a1 = _mm256_load_ps(&A_intlv[1 * M * 2]);
			a2 = _mm256_load_ps(&A_intlv[2 * M * 2]);
			a3 = _mm256_load_ps(&A_intlv[3 * M * 2]);
		}
		else
		{
			a0 = _mm256_loadu_ps(&A_intlv[0 * M * 2]);
			a1 = _mm256_loadu_ps(&A_intlv[1 * M * 2]);
			a2 = _mm256_loadu_ps(&A_intlv[2 * M * 2]);
			a3 = _mm256_loadu_ps(&A_intlv[3 * M * 2]);
		}
		
		if ((N * sizeof(std::complex<float>)) % 32 == 0)
		{
			b0 = _mm256_load_ps(&B_intlv[0 * N * 2]);
			b1 = _mm256_load_ps(&B_intlv[1 * N * 2]);
			b2 = _mm256_load_ps(&B_intlv[2 * N * 2]);
			b3 = _mm256_load_ps(&B_intlv[3 * N * 2]);
		}
		else
		{
			b0 = _mm256_loadu_ps(&B_intlv[0 * N * 2]);
			b1 = _mm256_loadu_ps(&B_intlv[1 * N * 2]);
			b2 = _mm256_loadu_ps(&B_intlv[2 * N * 2]);
			b3 = _mm256_loadu_ps(&B_intlv[3 * N * 2]);
		}
		
		if ((P * sizeof(std::complex<float>)) % 32 == 0)
		{
			c0 = _mm256_load_ps(&C_intlv[0 * P * 2]);
			c1 = _mm256_load_ps(&C_intlv[1 * P * 2]);
			c2 = _mm256_load_ps(&C_intlv[2 * P * 2]);
			c3 = _mm256_load_ps(&C_intlv[3 * P * 2]);
		}
		else
		{
			c0 = _mm256_loadu_ps(&C_intlv[0 * P * 2]);
			c1 = _mm256_loadu_ps(&C_intlv[1 * P * 2]);
			c2 = _mm256_loadu_ps(&C_intlv[2 * P * 2]);
			c3 = _mm256_loadu_ps(&C_intlv[3 * P * 2]);
		}

		_MM256_MMUL4_C_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3);
		
		if ((P * sizeof(std::complex<float>)) % 32 == 0)
		{
			_mm256_store_ps(&C_intlv[0 * P * 2], c0);
			_mm256_store_ps(&C_intlv[1 * P * 2], c1);
			_mm256_store_ps(&C_intlv[2 * P * 2], c2);
			_mm256_store_ps(&C_intlv[3 * P * 2], c3);
		}
		else
		{
			_mm256_storeu_ps(&C_intlv[0 * P * 2], c0);
			_mm256_storeu_ps(&C_intlv[1 * P * 2], c1);
			_mm256_storeu_ps(&C_intlv[2 * P * 2], c2);
			_mm256_storeu_ps(&C_intlv[3 * P * 2], c3);
		}
	}

	template <>
	void 
	_multiply_block_avx<double>(double* A, double* B, double* C, const size_t M, const size_t N, const size_t P)
	{
		alignas(32)__m256d a0, a1, a2, a3;

		if ((M * sizeof(double)) % 32 == 0)
		{
			a0 = _mm256_load_pd(&A[0 * M]);
			a1 = _mm256_load_pd(&A[1 * M]);
			a2 = _mm256_load_pd(&A[2 * M]);
			a3 = _mm256_load_pd(&A[3 * M]);
		}
		else
		{
			a0 = _mm256_loadu_pd(&A[0 * M]);
			a1 = _mm256_loadu_pd(&A[1 * M]);
			a2 = _mm256_loadu_pd(&A[2 * M]);
			a3 = _mm256_loadu_pd(&A[3 * M]);
		}
		
		alignas(32)__m256d b0, b1, b2, b3;
		if ((N * sizeof(double)) % 32 == 0)
		{
			b0 = _mm256_load_pd(&B[0 * N]);
			b1 = _mm256_load_pd(&B[1 * N]);
			b2 = _mm256_load_pd(&B[2 * N]);
			b3 = _mm256_load_pd(&B[3 * N]);
		}
		else
		{
			b0 = _mm256_loadu_pd(&B[0 * N]);
			b1 = _mm256_loadu_pd(&B[1 * N]);
			b2 = _mm256_loadu_pd(&B[2 * N]);
			b3 = _mm256_loadu_pd(&B[3 * N]);
		}
		
		alignas(32)__m256d c0, c1, c2, c3;
		if ((P * sizeof(double)) % 32 == 0)
		{
			c0 = _mm256_load_pd(&C[0 * P]);
			c1 = _mm256_load_pd(&C[1 * P]);
			c2 = _mm256_load_pd(&C[2 * P]);
			c3 = _mm256_load_pd(&C[3 * P]);
		}
		else
		{
			c0 = _mm256_loadu_pd(&C[0 * P]);
			c1 = _mm256_loadu_pd(&C[1 * P]);
			c2 = _mm256_loadu_pd(&C[2 * P]);
			c3 = _mm256_loadu_pd(&C[3 * P]);
		}

		_MM256_MMUL4_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3);
		
		if ((P * sizeof(double)) % 32 == 0)
		{
			_mm256_store_pd(&C[0 * P], c0);
			_mm256_store_pd(&C[1 * P], c1);
			_mm256_store_pd(&C[2 * P], c2);
			_mm256_store_pd(&C[3 * P], c3);
		}
		else
		{
			_mm256_storeu_pd(&C[0 * P], c0);
			_mm256_storeu_pd(&C[1 * P], c1);
			_mm256_storeu_pd(&C[2 * P], c2);
			_mm256_storeu_pd(&C[3 * P], c3);
		}
	}

	template <>
	void 
	_multiply_block_avx<std::complex<double>>(std::complex<double>* A, std::complex<double>* B, std::complex<double>* C, const size_t M, const size_t N, const size_t P)
	{
		double* A_intlv = reinterpret_cast<double*>(A);
		double* B_intlv = reinterpret_cast<double*>(B);
		double* C_intlv = reinterpret_cast<double*>(C);

		alignas(32)__m256d a0, a1;
		alignas(32)__m256d b0, b1;
		alignas(32)__m256d c0, c1;

		if ((M * sizeof(std::complex<double>)) % 32 == 0)
		{
			a0 = _mm256_load_pd(&A_intlv[0 * M * 2]);
			a1 = _mm256_load_pd(&A_intlv[1 * M * 2]);
		}
		else
		{
			a0 = _mm256_loadu_pd(&A_intlv[0 * M * 2]);
			a1 = _mm256_loadu_pd(&A_intlv[1 * M * 2]);
		}
		
		if ((N * sizeof(std::complex<double>)) % 32 == 0)
		{
			b0 = _mm256_load_pd(&B_intlv[0 * N * 2]);
			b1 = _mm256_load_pd(&B_intlv[1 * N * 2]);
		}
		else
		{
			b0 = _mm256_loadu_pd(&B_intlv[0 * N * 2]);
			b1 = _mm256_loadu_pd(&B_intlv[1 * N * 2]);
		}
		
		if ((P * sizeof(std::complex<double>)) % 32 == 0)
		{
			c0 = _mm256_load_pd(&C_intlv[0 * P * 2]);
			c1 = _mm256_load_pd(&C_intlv[1 * P * 2]);
		}
		else
		{
			c0 = _mm256_loadu_pd(&C_intlv[0 * P * 2]);
			c1 = _mm256_loadu_pd(&C_intlv[1 * P * 2]);
		}

		_MM256_MMUL2_C_PD(a0, a1, b0, b1, c0, c1);
		
		if ((P * sizeof(std::complex<double>)) % 32 == 0)
		{
			_mm256_store_pd(&C_intlv[0 * P * 2], c0);
			_mm256_store_pd(&C_intlv[1 * P * 2], c1);
		}
		else
		{
			_mm256_storeu_pd(&C_intlv[0 * P * 2], c0);
			_mm256_storeu_pd(&C_intlv[1 * P * 2], c1);
		}
	}

	template<>
	void
	_multiply_block_avx512<float>(float* A, float* B, float* C, const size_t M, const size_t N, const size_t P)
	{
		alignas(64)__m512 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af;

		if (((M * sizeof(float)) % 64) == 0) 
		{
			a0 = _mm512_load_ps(&A[ 0*M]);
			a1 = _mm512_load_ps(&A[ 1*M]);
			a2 = _mm512_load_ps(&A[ 2*M]);
			a3 = _mm512_load_ps(&A[ 3*M]);
			a4 = _mm512_load_ps(&A[ 4*M]);
			a5 = _mm512_load_ps(&A[ 5*M]);
			a6 = _mm512_load_ps(&A[ 6*M]);
			a7 = _mm512_load_ps(&A[ 7*M]);
			a8 = _mm512_load_ps(&A[ 8*M]);
			a9 = _mm512_load_ps(&A[ 9*M]);
			aa = _mm512_load_ps(&A[10*M]);
			ab = _mm512_load_ps(&A[11*M]);
			ac = _mm512_load_ps(&A[12*M]);
			ad = _mm512_load_ps(&A[13*M]);
			ae = _mm512_load_ps(&A[14*M]);
			af = _mm512_load_ps(&A[15*M]);
		}
		else 
		{
			a0 = _mm512_loadu_ps(&A[ 0*M]);
			a1 = _mm512_loadu_ps(&A[ 1*M]);
			a2 = _mm512_loadu_ps(&A[ 2*M]);
			a3 = _mm512_loadu_ps(&A[ 3*M]);
			a4 = _mm512_loadu_ps(&A[ 4*M]);
			a5 = _mm512_loadu_ps(&A[ 5*M]);
			a6 = _mm512_loadu_ps(&A[ 6*M]);
			a7 = _mm512_loadu_ps(&A[ 7*M]);
			a8 = _mm512_loadu_ps(&A[ 8*M]);
			a9 = _mm512_loadu_ps(&A[ 9*M]);
			aa = _mm512_loadu_ps(&A[10*M]);
			ab = _mm512_loadu_ps(&A[11*M]);
			ac = _mm512_loadu_ps(&A[12*M]);
			ad = _mm512_loadu_ps(&A[13*M]);
			ae = _mm512_loadu_ps(&A[14*M]);
			af = _mm512_loadu_ps(&A[15*M]);
		}

		alignas(64)__m512 b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf;

		if (((N * sizeof(float)) % 64) == 0) 
		{
			b0 = _mm512_load_ps(&B[ 0*N]);
			b1 = _mm512_load_ps(&B[ 1*N]);
			b2 = _mm512_load_ps(&B[ 2*N]);
			b3 = _mm512_load_ps(&B[ 3*N]);
			b4 = _mm512_load_ps(&B[ 4*N]);
			b5 = _mm512_load_ps(&B[ 5*N]);
			b6 = _mm512_load_ps(&B[ 6*N]);
			b7 = _mm512_load_ps(&B[ 7*N]);
			b8 = _mm512_load_ps(&B[ 8*N]);
			b9 = _mm512_load_ps(&B[ 9*N]);
			ba = _mm512_load_ps(&B[10*N]);
			bb = _mm512_load_ps(&B[11*N]);
			bc = _mm512_load_ps(&B[12*N]);
			bd = _mm512_load_ps(&B[13*N]);
			be = _mm512_load_ps(&B[14*N]);
			bf = _mm512_load_ps(&B[15*N]);
		}
		else 
		{
			b0 = _mm512_loadu_ps(&B[ 0*N]);
			b1 = _mm512_loadu_ps(&B[ 1*N]);
			b2 = _mm512_loadu_ps(&B[ 2*N]);
			b3 = _mm512_loadu_ps(&B[ 3*N]);
			b4 = _mm512_loadu_ps(&B[ 4*N]);
			b5 = _mm512_loadu_ps(&B[ 5*N]);
			b6 = _mm512_loadu_ps(&B[ 6*N]);
			b7 = _mm512_loadu_ps(&B[ 7*N]);
			b8 = _mm512_loadu_ps(&B[ 8*N]);
			b9 = _mm512_loadu_ps(&B[ 9*N]);
			ba = _mm512_loadu_ps(&B[10*N]);
			bb = _mm512_loadu_ps(&B[11*N]);
			bc = _mm512_loadu_ps(&B[12*N]);
			bd = _mm512_loadu_ps(&B[13*N]);
			be = _mm512_loadu_ps(&B[14*N]);
			bf = _mm512_loadu_ps(&B[15*N]);
		}

		alignas(64)__m512 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb, cc, cd, ce, cf;

		if (((N * sizeof(float)) % 64) == 0) 
		{
			c0 = _mm512_load_ps(&C[ 0*P]);
			c1 = _mm512_load_ps(&C[ 1*P]);
			c2 = _mm512_load_ps(&C[ 2*P]);
			c3 = _mm512_load_ps(&C[ 3*P]);
			c4 = _mm512_load_ps(&C[ 4*P]);
			c5 = _mm512_load_ps(&C[ 5*P]);
			c6 = _mm512_load_ps(&C[ 6*P]);
			c7 = _mm512_load_ps(&C[ 7*P]);
			c8 = _mm512_load_ps(&C[ 8*P]);
			c9 = _mm512_load_ps(&C[ 9*P]);
			ca = _mm512_load_ps(&C[10*P]);
			cb = _mm512_load_ps(&C[11*P]);
			cc = _mm512_load_ps(&C[12*P]);
			cd = _mm512_load_ps(&C[13*P]);
			ce = _mm512_load_ps(&C[14*P]);
			cf = _mm512_load_ps(&C[15*P]);
		}
		else 
		{
			c0 = _mm512_loadu_ps(&C[ 0*P]);
			c1 = _mm512_loadu_ps(&C[ 1*P]);
			c2 = _mm512_loadu_ps(&C[ 2*P]);
			c3 = _mm512_loadu_ps(&C[ 3*P]);
			c4 = _mm512_loadu_ps(&C[ 4*P]);
			c5 = _mm512_loadu_ps(&C[ 5*P]);
			c6 = _mm512_loadu_ps(&C[ 6*P]);
			c7 = _mm512_loadu_ps(&C[ 7*P]);
			c8 = _mm512_loadu_ps(&C[ 8*P]);
			c9 = _mm512_loadu_ps(&C[ 9*P]);
			ca = _mm512_loadu_ps(&C[10*P]);
			cb = _mm512_loadu_ps(&C[11*P]);
			cc = _mm512_loadu_ps(&C[12*P]);
			cd = _mm512_loadu_ps(&C[13*P]);
			ce = _mm512_loadu_ps(&C[14*P]);
			cf = _mm512_loadu_ps(&C[15*P]);
		}
			_MM512_MMUL16_PS(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, \
				b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf, \
				c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb, cc, cd, ce, cf);
		if (((P * sizeof(float)) % 64) == 0) 
		{
			_mm512_store_ps(&C[ 0*P], c0);
			_mm512_store_ps(&C[ 1*P], c1);
			_mm512_store_ps(&C[ 2*P], c2);
			_mm512_store_ps(&C[ 3*P], c3);
			_mm512_store_ps(&C[ 4*P], c4);
			_mm512_store_ps(&C[ 5*P], c5);
			_mm512_store_ps(&C[ 6*P], c6);
			_mm512_store_ps(&C[ 7*P], c7);
			_mm512_store_ps(&C[ 8*P], c8);
			_mm512_store_ps(&C[ 9*P], c9);
			_mm512_store_ps(&C[10*P], ca);
			_mm512_store_ps(&C[11*P], cb);
			_mm512_store_ps(&C[12*P], cc);
			_mm512_store_ps(&C[13*P], cd);
			_mm512_store_ps(&C[14*P], ce);
			_mm512_store_ps(&C[15*P], cf);
		}
		else 
		{
			_mm512_storeu_ps(&C[ 0*P], c0);
			_mm512_storeu_ps(&C[ 1*P], c1);
			_mm512_storeu_ps(&C[ 2*P], c2);
			_mm512_storeu_ps(&C[ 3*P], c3);
			_mm512_storeu_ps(&C[ 4*P], c4);
			_mm512_storeu_ps(&C[ 5*P], c5);
			_mm512_storeu_ps(&C[ 6*P], c6);
			_mm512_storeu_ps(&C[ 7*P], c7);
			_mm512_storeu_ps(&C[ 8*P], c8);
			_mm512_storeu_ps(&C[ 9*P], c9);
			_mm512_storeu_ps(&C[10*P], ca);
			_mm512_storeu_ps(&C[11*P], cb);
			_mm512_storeu_ps(&C[12*P], cc);
			_mm512_storeu_ps(&C[13*P], cd);
			_mm512_storeu_ps(&C[14*P], ce);
			_mm512_storeu_ps(&C[15*P], cf);
		}
	}

	template<>
	void _multiply_block_avx512(std::complex<float>* A, std::complex<float>* B, std::complex<float>* C, const size_t M, const size_t N, const size_t P)
	{
		float* A_intlv = reinterpret_cast<float*>(A);
		float* B_intlv = reinterpret_cast<float*>(B);
		float* C_intlv = reinterpret_cast<float*>(C);
		
		alignas(64) __m512 a0, a1, a2, a3, a4, a5, a6, a7;
		
		if (((M * 2 * sizeof(float)) % 64) == 0) 
		{
			a0 = _mm512_load_ps(&A_intlv[0 * M * 2]);
			a1 = _mm512_load_ps(&A_intlv[1 * M * 2]);
			a2 = _mm512_load_ps(&A_intlv[2 * M * 2]);
			a3 = _mm512_load_ps(&A_intlv[3 * M * 2]);
			a4 = _mm512_load_ps(&A_intlv[4 * M * 2]);
			a5 = _mm512_load_ps(&A_intlv[5 * M * 2]);
			a6 = _mm512_load_ps(&A_intlv[6 * M * 2]);
			a7 = _mm512_load_ps(&A_intlv[7 * M * 2]);
		} 
		else 
		{
			a0 = _mm512_loadu_ps(&A_intlv[0 * M * 2]);
			a1 = _mm512_loadu_ps(&A_intlv[1 * M * 2]);
			a2 = _mm512_loadu_ps(&A_intlv[2 * M * 2]);
			a3 = _mm512_loadu_ps(&A_intlv[3 * M * 2]);
			a4 = _mm512_loadu_ps(&A_intlv[4 * M * 2]);
			a5 = _mm512_loadu_ps(&A_intlv[5 * M * 2]);
			a6 = _mm512_loadu_ps(&A_intlv[6 * M * 2]);
			a7 = _mm512_loadu_ps(&A_intlv[7 * M * 2]);
		}
		
		alignas(64) __m512 b0, b1, b2, b3, b4, b5, b6, b7;
		
		if (((N * 2 * sizeof(float)) % 64) == 0) 
		{
			b0 = _mm512_load_ps(&B_intlv[0 * N * 2]);
			b1 = _mm512_load_ps(&B_intlv[1 * N * 2]);
			b2 = _mm512_load_ps(&B_intlv[2 * N * 2]);
			b3 = _mm512_load_ps(&B_intlv[3 * N * 2]);
			b4 = _mm512_load_ps(&B_intlv[4 * N * 2]);
			b5 = _mm512_load_ps(&B_intlv[5 * N * 2]);
			b6 = _mm512_load_ps(&B_intlv[6 * N * 2]);
			b7 = _mm512_load_ps(&B_intlv[7 * N * 2]);
		} 
		else 
		{
			b0 = _mm512_loadu_ps(&B_intlv[0 * N * 2]);
			b1 = _mm512_loadu_ps(&B_intlv[1 * N * 2]);
			b2 = _mm512_loadu_ps(&B_intlv[2 * N * 2]);
			b3 = _mm512_loadu_ps(&B_intlv[3 * N * 2]);
			b4 = _mm512_loadu_ps(&B_intlv[4 * N * 2]);
			b5 = _mm512_loadu_ps(&B_intlv[5 * N * 2]);
			b6 = _mm512_loadu_ps(&B_intlv[6 * N * 2]);
			b7 = _mm512_loadu_ps(&B_intlv[7 * N * 2]);
		}
		
		alignas(64) __m512 c0, c1, c2, c3, c4, c5, c6, c7;
		
		if (((P * 2 * sizeof(float)) % 64) == 0) 
		{
			c0 = _mm512_load_ps(&C_intlv[0 * P * 2]);
			c1 = _mm512_load_ps(&C_intlv[1 * P * 2]);
			c2 = _mm512_load_ps(&C_intlv[2 * P * 2]);
			c3 = _mm512_load_ps(&C_intlv[3 * P * 2]);
			c4 = _mm512_load_ps(&C_intlv[4 * P * 2]);
			c5 = _mm512_load_ps(&C_intlv[5 * P * 2]);
			c6 = _mm512_load_ps(&C_intlv[6 * P * 2]);
			c7 = _mm512_load_ps(&C_intlv[7 * P * 2]);
		} 
		else 
		{
			c0 = _mm512_loadu_ps(&C_intlv[0 * P * 2]);
			c1 = _mm512_loadu_ps(&C_intlv[1 * P * 2]);
			c2 = _mm512_loadu_ps(&C_intlv[2 * P * 2]);
			c3 = _mm512_loadu_ps(&C_intlv[3 * P * 2]);
			c4 = _mm512_loadu_ps(&C_intlv[4 * P * 2]);
			c5 = _mm512_loadu_ps(&C_intlv[5 * P * 2]);
			c6 = _mm512_loadu_ps(&C_intlv[6 * P * 2]);
			c7 = _mm512_loadu_ps(&C_intlv[7 * P * 2]);
		}

		_MM512_MMUL8_C_PS(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5, c6, c7)
		
		if (((P * 2 * sizeof(float)) % 64) == 0) 
		{
			_mm512_store_ps(&C_intlv[0 * P * 2], c0);
			_mm512_store_ps(&C_intlv[1 * P * 2], c1);
			_mm512_store_ps(&C_intlv[2 * P * 2], c2);
			_mm512_store_ps(&C_intlv[3 * P * 2], c3);
			_mm512_store_ps(&C_intlv[4 * P * 2], c4);
			_mm512_store_ps(&C_intlv[5 * P * 2], c5);
			_mm512_store_ps(&C_intlv[6 * P * 2], c6);
			_mm512_store_ps(&C_intlv[7 * P * 2], c7);
		} 
		else 
		{
			_mm512_storeu_ps(&C_intlv[0 * P * 2], c0);
			_mm512_storeu_ps(&C_intlv[1 * P * 2], c1);
			_mm512_storeu_ps(&C_intlv[2 * P * 2], c2);
			_mm512_storeu_ps(&C_intlv[3 * P * 2], c3);
			_mm512_storeu_ps(&C_intlv[4 * P * 2], c4);
			_mm512_storeu_ps(&C_intlv[5 * P * 2], c5);
			_mm512_storeu_ps(&C_intlv[6 * P * 2], c6);
			_mm512_storeu_ps(&C_intlv[7 * P * 2], c7);
		}
	}

	template <>
	void 
	_multiply_block_avx512<double>(double* A, double* B, double* C, const size_t M, const size_t N, const size_t P)
	{
		alignas(64)__m512d a0, a1, a2, a3, a4, a5, a6, a7;
		
		if ((M * sizeof(float)) % 64 == 0)
		{
			a0 = _mm512_load_pd(&A[0 * M]);
			a1 = _mm512_load_pd(&A[1 * M]);
			a2 = _mm512_load_pd(&A[2 * M]);
			a3 = _mm512_load_pd(&A[3 * M]);
			a4 = _mm512_load_pd(&A[4 * M]);
			a5 = _mm512_load_pd(&A[5 * M]);
			a6 = _mm512_load_pd(&A[6 * M]);
			a7 = _mm512_load_pd(&A[7 * M]);

		}
		else
		{
			a0 = _mm512_loadu_pd(&A[0 * M]);
			a1 = _mm512_loadu_pd(&A[1 * M]);
			a2 = _mm512_loadu_pd(&A[2 * M]);
			a3 = _mm512_loadu_pd(&A[3 * M]);
			a4 = _mm512_loadu_pd(&A[4 * M]);
			a5 = _mm512_loadu_pd(&A[5 * M]);
			a6 = _mm512_loadu_pd(&A[6 * M]);
			a7 = _mm512_loadu_pd(&A[7 * M]);
		}
		
		alignas(64)__m512d b0, b1, b2, b3, b4, b5, b6, b7;
		if ((N * sizeof(float)) % 64 == 0)
		{
			b0 = _mm512_load_pd(&B[0 * N]);
			b1 = _mm512_load_pd(&B[1 * N]);
			b2 = _mm512_load_pd(&B[2 * N]);
			b3 = _mm512_load_pd(&B[3 * N]);
			b4 = _mm512_load_pd(&B[4 * N]);
			b5 = _mm512_load_pd(&B[5 * N]);
			b6 = _mm512_load_pd(&B[6 * N]);
			b7 = _mm512_load_pd(&B[7 * N]);
		}
		else
		{
			b0 = _mm512_loadu_pd(&B[0 * N]);
			b1 = _mm512_loadu_pd(&B[1 * N]);
			b2 = _mm512_loadu_pd(&B[2 * N]);
			b3 = _mm512_loadu_pd(&B[3 * N]);
			b4 = _mm512_loadu_pd(&B[4 * N]);
			b5 = _mm512_loadu_pd(&B[5 * N]);
			b6 = _mm512_loadu_pd(&B[6 * N]);
			b7 = _mm512_loadu_pd(&B[7 * N]);
		}
		
		alignas(64)__m512d c0, c1, c2, c3, c4, c5, c6, c7;
		if ((P * sizeof(float)) % 64 == 0)
		{
			c0 = _mm512_load_pd(&C[0 * P]);
			c1 = _mm512_load_pd(&C[1 * P]);
			c2 = _mm512_load_pd(&C[2 * P]);
			c3 = _mm512_load_pd(&C[3 * P]);
			c4 = _mm512_load_pd(&C[4 * P]);
			c5 = _mm512_load_pd(&C[5 * P]);
			c6 = _mm512_load_pd(&C[6 * P]);
			c7 = _mm512_load_pd(&C[7 * P]);
		}
		else
		{
			c0 = _mm512_loadu_pd(&C[0 * P]);
			c1 = _mm512_loadu_pd(&C[1 * P]);
			c2 = _mm512_loadu_pd(&C[2 * P]);
			c3 = _mm512_loadu_pd(&C[3 * P]);
			c4 = _mm512_loadu_pd(&C[4 * P]);
			c5 = _mm512_loadu_pd(&C[5 * P]);
			c6 = _mm512_loadu_pd(&C[6 * P]);
			c7 = _mm512_loadu_pd(&C[7 * P]);
		}

		_MM512_MMUL8_PD(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5, c6, c7);
		
		if ((P * sizeof(float)) % 64 == 0)
		{
			_mm512_store_pd(&C[0 * P], c0);
			_mm512_store_pd(&C[1 * P], c1);
			_mm512_store_pd(&C[2 * P], c2);
			_mm512_store_pd(&C[3 * P], c3);
			_mm512_store_pd(&C[4 * P], c4);
			_mm512_store_pd(&C[5 * P], c5);
			_mm512_store_pd(&C[6 * P], c6);
			_mm512_store_pd(&C[7 * P], c7);
		}
		else
		{
			_mm512_storeu_pd(&C[0 * P], c0);
			_mm512_storeu_pd(&C[1 * P], c1);
			_mm512_storeu_pd(&C[2 * P], c2);
			_mm512_storeu_pd(&C[3 * P], c3);
			_mm512_storeu_pd(&C[4 * P], c4);
			_mm512_storeu_pd(&C[5 * P], c5);
			_mm512_storeu_pd(&C[6 * P], c6);
			_mm512_storeu_pd(&C[7 * P], c7);
		}
	}

	template<>
	void _multiply_block_avx512(std::complex<double>* A, std::complex<double>* B, std::complex<double>* C, const size_t M, const size_t N, const size_t P)
	{
		double* A_intlv = reinterpret_cast<double*>(A);
		double* B_intlv = reinterpret_cast<double*>(B);
		double* C_intlv = reinterpret_cast<double*>(C);
		
		alignas(64) __m512d a0, a1, a2, a3;
		alignas(64) __m512d b0, b1, b2, b3;
		alignas(64) __m512d c0, c1, c2, c3;
		
		if (((M * 2 * sizeof(double)) % 64) == 0) 
		{
			a0 = _mm512_load_pd(&A_intlv[0 * M * 2]);
			a1 = _mm512_load_pd(&A_intlv[1 * M * 2]);
			a2 = _mm512_load_pd(&A_intlv[2 * M * 2]);
			a3 = _mm512_load_pd(&A_intlv[3 * M * 2]);
		} 
		else 
		{
			a0 = _mm512_loadu_pd(&A_intlv[0 * M * 2]);
			a1 = _mm512_loadu_pd(&A_intlv[1 * M * 2]);
			a2 = _mm512_loadu_pd(&A_intlv[2 * M * 2]);
			a3 = _mm512_loadu_pd(&A_intlv[3 * M * 2]);
		}
		
		if (((N * 2 * sizeof(double)) % 64) == 0) 
		{
			b0 = _mm512_load_pd(&B_intlv[0 * N * 2]);
			b1 = _mm512_load_pd(&B_intlv[1 * N * 2]);
			b2 = _mm512_load_pd(&B_intlv[2 * N * 2]);
			b3 = _mm512_load_pd(&B_intlv[3 * N * 2]);
		} 
		else 
		{
			b0 = _mm512_loadu_pd(&B_intlv[0 * N * 2]);
			b1 = _mm512_loadu_pd(&B_intlv[1 * N * 2]);
			b2 = _mm512_loadu_pd(&B_intlv[2 * N * 2]);
			b3 = _mm512_loadu_pd(&B_intlv[3 * N * 2]);
		}
		
		
		if (((P * 2 * sizeof(double)) % 64) == 0) 
		{
			c0 = _mm512_load_pd(&C_intlv[0 * P * 2]);
			c1 = _mm512_load_pd(&C_intlv[1 * P * 2]);
			c2 = _mm512_load_pd(&C_intlv[2 * P * 2]);
			c3 = _mm512_load_pd(&C_intlv[3 * P * 2]);
		} 
		else 
		{
			c0 = _mm512_loadu_pd(&C_intlv[0 * P * 2]);
			c1 = _mm512_loadu_pd(&C_intlv[1 * P * 2]);
			c2 = _mm512_loadu_pd(&C_intlv[2 * P * 2]);
			c3 = _mm512_loadu_pd(&C_intlv[3 * P * 2]);
		}

		_MM512_MMUL4_C_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3)
		
		if (((P * 2 * sizeof(double)) % 64) == 0) 
		{
			_mm512_store_pd(&C_intlv[0 * P * 2], c0);
			_mm512_store_pd(&C_intlv[1 * P * 2], c1);
			_mm512_store_pd(&C_intlv[2 * P * 2], c2);
			_mm512_store_pd(&C_intlv[3 * P * 2], c3);
		} 
		else 
		{
			_mm512_storeu_pd(&C_intlv[0 * P * 2], c0);
			_mm512_storeu_pd(&C_intlv[1 * P * 2], c1);
			_mm512_storeu_pd(&C_intlv[2 * P * 2], c2);
			_mm512_storeu_pd(&C_intlv[3 * P * 2], c3);
		}
	}
}//namespace damm