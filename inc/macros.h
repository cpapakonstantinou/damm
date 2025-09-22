#ifndef __MACROS_H__
#define __MACROS_H__
/**
 * \file macros.h
 * \brief macro includes for damm (Dense Arrayed Matrix Math) 
 * \author cpapakonstantinou
 * \date 2025
 * \note these macros are used within the src/ files. 
 * #defines are used to 'force' snippets inline inside kernels while maintaining modularity.
 **/
// Copyright (c) 2025  Constantine Papakonstantinou
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
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
#include <immintrin.h>

// Variations of this file can, and should exist for the target architecture
// this version was developed on System: Intel Tiger Lake i7-1165G7
// in practice we might have variants of this file
// #if cpu is ...
// #include macros_cpu_specific.h 


/* LOAD / STORE */
#define _MM_LOAD_LINE_PS(ptr, a0, a1, a2, a3) \
	if (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) \
	{ \
		a0 = _mm_load_ps(ptr); \
		a1 = _mm_load_ps(ptr + 4); \
		a2 = _mm_load_ps(ptr + 8); \
		a3 = _mm_load_ps(ptr + 12); \
	} \
	else \
	{ \
		a0 = _mm_loadu_ps(ptr); \
		a1 = _mm_loadu_ps(ptr + 4); \
		a2 = _mm_loadu_ps(ptr + 8); \
		a3 = _mm_loadu_ps(ptr + 12); \
	}

#define _MM_STORE_LINE_PS(ptr, c0, c1, c2, c3) \
	if (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) \
	{ \
		_mm_store_ps(ptr, c0); \
		_mm_store_ps(ptr + 4, c1); \
		_mm_store_ps(ptr + 8, c2); \
		_mm_store_ps(ptr + 12, c3); \
	} \
	else \
	{ \
		_mm_storeu_ps(ptr, c0); \
		_mm_storeu_ps(ptr + 4, c1); \
		_mm_storeu_ps(ptr + 8, c2); \
		_mm_storeu_ps(ptr + 12, c3); \
	}

#define _MM_LOAD_LINE_PD(ptr, a0, a1, a2, a3) \
	if (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) \
	{ \
		a0 = _mm_load_pd(ptr); \
		a1 = _mm_load_pd(ptr + 2); \
		a2 = _mm_load_pd(ptr + 4); \
		a3 = _mm_load_pd(ptr + 6); \
	} \
	else \
	{ \
		a0 = _mm_loadu_pd(ptr); \
		a1 = _mm_loadu_pd(ptr + 2); \
		a2 = _mm_loadu_pd(ptr + 4); \
		a3 = _mm_loadu_pd(ptr + 6); \
	}

#define _MM_STORE_LINE_PD(ptr, c0, c1, c2, c3) \
	if (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) \
	{ \
		_mm_store_pd(ptr, c0); \
		_mm_store_pd(ptr + 2, c1); \
		_mm_store_pd(ptr + 4, c2); \
		_mm_store_pd(ptr + 6, c3); \
	} \
	else \
	{ \
		_mm_storeu_pd(ptr, c0); \
		_mm_storeu_pd(ptr + 2, c1); \
		_mm_storeu_pd(ptr + 4, c2); \
		_mm_storeu_pd(ptr + 6, c3); \
	}

#define _MM256_LOAD_LINE_PS(ptr, a0, a1) \
	if (reinterpret_cast<uintptr_t>(ptr) % 32 == 0) \
	{ \
		a0 = _mm256_load_ps(ptr); \
		a1 = _mm256_load_ps(ptr + 8); \
	} \
	else \
	{ \
		a0 = _mm256_loadu_ps(ptr); \
		a1 = _mm256_loadu_ps(ptr + 8); \
	}

#define _MM256_STORE_LINE_PS(ptr, c0, c1) \
	if (reinterpret_cast<uintptr_t>(ptr) % 32 == 0) \
	{ \
		_mm256_store_ps(ptr, c0); \
		_mm256_store_ps(ptr + 8, c1); \
	} \
	else \
	{ \
		_mm256_storeu_ps(ptr, c0); \
		_mm256_storeu_ps(ptr + 8, c1); \
	}

#define _MM256_LOAD_LINE_PD(ptr, a0, a1) \
	if (reinterpret_cast<uintptr_t>(ptr) % 32 == 0) \
	{ \
		a0 = _mm256_load_pd(ptr); \
		a1 = _mm256_load_pd(ptr + 4); \
	} \
	else \
	{ \
		a0 = _mm256_loadu_pd(ptr); \
		a1 = _mm256_loadu_pd(ptr + 4); \
	}

#define _MM256_STORE_LINE_PD(ptr, c0, c1) \
	if (reinterpret_cast<uintptr_t>(ptr) % 32 == 0) \
	{ \
		_mm256_store_pd(ptr, c0); \
		_mm256_store_pd(ptr + 4, c1); \
	} \
	else \
	{ \
		_mm256_storeu_pd(ptr, c0); \
		_mm256_storeu_pd(ptr + 4, c1); \
	} \

#define _MM512_LOAD_LINE_PS(ptr, a) \
	if (reinterpret_cast<uintptr_t>(ptr) % 64 == 0) \
	{ \
		a = _mm512_load_ps(ptr); \
	} \
	else \
	{ \
		a = _mm512_loadu_ps(ptr); \
	} \

#define _MM512_STORE_LINE_PS(ptr, c) \
	if (reinterpret_cast<uintptr_t>(ptr) % 64 == 0) \
	{ \
		_mm512_store_ps(ptr, c); \
	} \
	else \
	{ \
		_mm512_storeu_ps(ptr, c); \
	}	

#define _MM512_LOAD_LINE_PD(ptr, a) \
	if (reinterpret_cast<uintptr_t>(ptr) % 64 == 0) \
	{ \
		a = _mm512_load_pd(ptr); \
	} \
	else \
	{ \
		a = _mm512_loadu_pd(ptr); \
	}

#define _MM512_STORE_LINE_PD(ptr, c) \
	if (reinterpret_cast<uintptr_t>(ptr) % 64 == 0) \
	{ \
		_mm512_store_pd(ptr, c); \
	} \
	else \
	{ \
		_mm512_storeu_pd(ptr, c); \
	}

/* BROADCASTS */

#define _MM_SET1_PS(B, b) \
	b = _mm_set1_ps(*B);

#define _MM_SET1_PD(B, b) \
	b = _mm_set1_pd(*B);

#define _MM256_SET1_PS(B, b) \
	b = _mm256_set1_ps(*B);

#define _MM256_SET1_PD(B, b) \
	b = _mm256_set1_pd(*B);

#define _MM512_SET1_PS(B, b) \
	b = _mm512_set1_ps(*B);

#define _MM512_SET1_PD(B, b) \
	b = _mm512_set1_pd(*B);

#define _MM_SETR_PS(B, b) \
	b = _mm_setr_ps(B[0], B[1], B[0], B[1]);

#define _MM_SETR_PD(B, b) \
	b = _mm_setr_pd(B[0], B[1]);

#define _MM256_SETR_PS(B, b) \
	b = _mm256_setr_ps(B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1]);

#define _MM256_SETR_PD(B, b) \
	b = _mm256_setr_pd(B[0], B[1], B[0], B[1]);

#define _MM512_SETR_PS(B, b) \
	b =_mm512_setr_ps( \
	B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1], \
	B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1]);

#define _MM512_SETR_PD(B, b) \
	b = _mm512_setr_pd( \
	B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1]);

/* ARITHMETIC OPS */

#define _MM_ADD_LINE_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	c0 = _mm_add_ps(a0, b0); \
	c1 = _mm_add_ps(a1, b1); \
	c2 = _mm_add_ps(a2, b2); \
	c3 = _mm_add_ps(a3, b3);

#define _MM_SUB_LINE_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	c0 = _mm_sub_ps(a0, b0); \
	c1 = _mm_sub_ps(a1, b1); \
	c2 = _mm_sub_ps(a2, b2); \
	c3 = _mm_sub_ps(a3, b3);

#define _MM_MUL_LINE_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	c0 = _mm_mul_ps(a0, b0); \
	c1 = _mm_mul_ps(a1, b1); \
	c2 = _mm_mul_ps(a2, b2); \
	c3 = _mm_mul_ps(a3, b3);

#define _MM_DIV_LINE_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	c0 = _mm_div_ps(a0, b0); \
	c1 = _mm_div_ps(a1, b1); \
	c2 = _mm_div_ps(a2, b2); \
	c3 = _mm_div_ps(a3, b3);

#define _MM_ADD_LINE_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	c0 = _mm_add_pd(a0, b0); \
	c1 = _mm_add_pd(a1, b1); \
	c2 = _mm_add_pd(a2, b2); \
	c3 = _mm_add_pd(a3, b3);

#define _MM_SUB_LINE_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	c0 = _mm_sub_pd(a0, b0); \
	c1 = _mm_sub_pd(a1, b1); \
	c2 = _mm_sub_pd(a2, b2); \
	c3 = _mm_sub_pd(a3, b3);

#define _MM_MUL_LINE_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	c0 = _mm_mul_pd(a0, b0); \
	c1 = _mm_mul_pd(a1, b1); \
	c2 = _mm_mul_pd(a2, b2); \
	c3 = _mm_mul_pd(a3, b3);

#define _MM_DIV_LINE_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
	c0 = _mm_div_pd(a0, b0); \
	c1 = _mm_div_pd(a1, b1); \
	c2 = _mm_div_pd(a2, b2); \
	c3 = _mm_div_pd(a3, b3);

#define _MM256_ADD_LINE_PS(a0, a1, b0, b1, c0, c1) \
	c0 = _mm256_add_ps(a0, b0); \
	c1 = _mm256_add_ps(a1, b1);

#define _MM256_SUB_LINE_PS(a0, a1, b0, b1, c0, c1) \
	c0 = _mm256_sub_ps(a0, b0); \
	c1 = _mm256_sub_ps(a1, b1);

#define _MM256_MUL_LINE_PS(a0, a1, b0, b1, c0, c1) \
	c0 = _mm256_mul_ps(a0, b0); \
	c1 = _mm256_mul_ps(a1, b1);

#define _MM256_DIV_LINE_PS(a0, a1, b0, b1, c0, c1) \
	c0 = _mm256_div_ps(a0, b0); \
	c1 = _mm256_div_ps(a1, b1);

#define _MM256_ADD_LINE_PD(a0, a1, b0, b1, c0, c1) \
	c0 = _mm256_add_pd(a0, b0); \
	c1 = _mm256_add_pd(a1, b1);

#define _MM256_SUB_LINE_PD(a0, a1, b0, b1, c0, c1) \
	c0 = _mm256_sub_pd(a0, b0); \
	c1 = _mm256_sub_pd(a1, b1);

#define _MM256_MUL_LINE_PD(a0, a1, b0, b1, c0, c1) \
	c0 = _mm256_mul_pd(a0, b0); \
	c1 = _mm256_mul_pd(a1, b1);

#define _MM256_DIV_LINE_PD(a0, a1, b0, b1, c0, c1) \
	c0 = _mm256_div_pd(a0, b0); \
	c1 = _mm256_div_pd(a1, b1);

#define _MM512_ADD_LINE_PS(a, b, c) \
	c = _mm512_add_ps(a, b);

#define _MM512_SUB_LINE_PS(a, b, c) \
	c = _mm512_sub_ps(a, b);

#define _MM512_MUL_LINE_PS(a, b, c) \
	c = _mm512_mul_ps(a, b);

#define _MM512_DIV_LINE_PS(a, b, c) \
	c = _mm512_div_ps(a, b);

#define _MM512_ADD_LINE_PD(a, b, c) \
	c = _mm512_add_pd(a, b);

#define _MM512_SUB_LINE_PD(a, b, c) \
	c = _mm512_sub_pd(a, b);

#define _MM512_MUL_LINE_PD(a, b, c) \
	c = _mm512_mul_pd(a, b);

#define _MM512_DIV_LINE_PD(a, b, c) \
	c = _mm512_div_pd(a, b);

/* SCALAR - MATRIX OPS */

#define _MM_ADD_LINE_S_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128 b = _mm_set1_ps(*B); \
	_MM_ADD_LINE_PS(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3)

#define _MM_SUB_LINE_S_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128 b = _mm_set1_ps(*B); \
	_MM_SUB_LINE_PS(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3)

#define _MM_LHS_SUB_LINE_S_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128 b = _mm_set1_ps(*B); \
	_MM_SUB_LINE_PS(b, b, b, b, a0, a1, a2, a3, c0, c1, c2, c3)

#define _MM_MUL_LINE_S_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128 b = _mm_set1_ps(*B); \
	_MM_MUL_LINE_PS(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3)

#define _MM_DIV_LINE_S_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128 b = _mm_set1_ps(*B); \
	_MM_DIV_LINE_PS(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3)

#define _MM_LHS_DIV_LINE_S_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128 b = _mm_set1_ps(*B); \
	_MM_DIV_LINE_PS(b, b, b, b, a0, a1, a2, a3, c0, c1, c2, c3)

#define _MM_ADD_LINE_S_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128d b = _mm_set1_pd(*B); \
	_MM_ADD_LINE_PD(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3)

#define _MM_SUB_LINE_S_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128d b = _mm_set1_pd(*B); \
	_MM_SUB_LINE_PD(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3)

#define _MM_LHS_SUB_LINE_S_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128d b = _mm_set1_pd(*B); \
	_MM_SUB_LINE_PD(b, b, b, b, a0, a1, a2, a3, c0, c1, c2, c3)

#define _MM_MUL_LINE_S_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128d b = _mm_set1_pd(*B); \
	_MM_MUL_LINE_PD(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3)

#define _MM_DIV_LINE_S_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128d b = _mm_set1_pd(*B); \
	_MM_DIV_LINE_PD(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3)

#define _MM_LHS_DIV_LINE_S_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
	__m128d b = _mm_set1_pd(*B); \
	_MM_DIV_LINE_PD(b, b, b, b, a0, a1, a2, a3, c0, c1, c2, c3)

#define _MM256_ADD_LINE_S_PS(a0, a1, B, c0, c1) \
	__m256 b = _mm256_set1_ps(*B); \
	_MM256_ADD_LINE_PS(a0, a1, b, b, c0, c1)

#define _MM256_SUB_LINE_S_PS(a0, a1, B, c0, c1) \
	__m256 b = _mm256_set1_ps(*B); \
	_MM256_SUB_LINE_PS(a0, a1, b, b, c0, c1)

#define _MM256_LHS_SUB_LINE_S_PS(a0, a1, B, c0, c1) \
	__m256 b = _mm256_set1_ps(*B); \
	_MM256_SUB_LINE_PS(b, b, a0, a1, c0, c1)

#define _MM256_MUL_LINE_S_PS(a0, a1, B, c0, c1) \
	__m256 b = _mm256_set1_ps(*B); \
	_MM256_MUL_LINE_PS(a0, a1, b, b, c0, c1)

#define _MM256_DIV_LINE_S_PS(a0, a1, B, c0, c1) \
	__m256 b = _mm256_set1_ps(*B); \
	_MM256_DIV_LINE_PS(a0, a1, b, b, c0, c1)

#define _MM256_LHS_DIV_LINE_S_PS(a0, a1, B, c0, c1) \
	__m256 b = _mm256_set1_ps(*B); \
	_MM256_DIV_LINE_PS(b, b, a0, a1, c0, c1)

#define _MM256_ADD_LINE_S_PD(a0, a1, B, c0, c1) \
	__m256d b = _mm256_set1_pd(*B); \
	_MM256_ADD_LINE_PD(a0, a1, b, b, c0, c1)

#define _MM256_SUB_LINE_S_PD(a0, a1, B, c0, c1) \
	__m256d b = _mm256_set1_pd(*B); \
	_MM256_SUB_LINE_PD(a0, a1, b, b, c0, c1)

#define _MM256_LHS_SUB_LINE_S_PD(a0, a1, B, c0, c1) \
	__m256d b = _mm256_set1_pd(*B); \
	_MM256_SUB_LINE_PD(b, b, a0, a1, c0, c1)

#define _MM256_MUL_LINE_S_PD(a0, a1, B, c0, c1) \
	__m256d b = _mm256_set1_pd(*B); \
	_MM256_MUL_LINE_PD(a0, a1, b, b, c0, c1)

#define _MM256_DIV_LINE_S_PD(a0, a1, B, c0, c1) \
	__m256d b = _mm256_set1_pd(*B); \
	_MM256_DIV_LINE_PD(a0, a1, b, b, c0, c1)

#define _MM256_LHS_DIV_LINE_S_PD(a0, a1, B, c0, c1) \
	__m256d b = _mm256_set1_pd(*B); \
	_MM256_DIV_LINE_PD(b, b, a0, a1, c0, c1)

#define _MM512_ADD_LINE_S_PS(a, B, c) \
	__m512 b = _mm512_set1_ps(*B); \
	_MM512_ADD_LINE_PS(a, b, c)

#define _MM512_SUB_LINE_S_PS(a, B, c) \
	__m512 b = _mm512_set1_ps(*B); \
	_MM512_SUB_LINE_PS(a, b, c)

#define _MM512_LHS_SUB_LINE_S_PS(a, B, c) \
	__m512 b = _mm512_set1_ps(*B); \
	_MM512_SUB_LINE_PS(b, a, c)

#define _MM512_MUL_LINE_S_PS(a, B, c) \
	__m512 b = _mm512_set1_ps(*B); \
	_MM512_MUL_LINE_PS(a, b, c)

#define _MM512_DIV_LINE_S_PS(a, B, c) \
	__m512 b = _mm512_set1_ps(*B); \
	_MM512_DIV_LINE_PS(a, b, c)

#define _MM512_LHS_DIV_LINE_S_PS(a, B, c) \
	__m512 b = _mm512_set1_ps(*B); \
	_MM512_DIV_LINE_PS(b, a, c)

#define _MM512_ADD_LINE_S_PD(a, B, c) \
	__m512d b = _mm512_set1_pd(*B); \
	_MM512_ADD_LINE_PD(a, b, c)

#define _MM512_SUB_LINE_S_PD(a, B, c) \
	__m512d b = _mm512_set1_pd(*B); \
	_MM512_SUB_LINE_PD(a, b, c)

#define _MM512_LHS_SUB_LINE_S_PD(a, B, c) \
	__m512d b = _mm512_set1_pd(*B); \
	_MM512_SUB_LINE_PD(b, a, c)

#define _MM512_MUL_LINE_S_PD(a, B, c) \
	__m512d b = _mm512_set1_pd(*B); \
	_MM512_MUL_LINE_PD(a, b, c)

#define _MM512_DIV_LINE_S_PD(a, B, c) \
	__m512d b = _mm512_set1_pd(*B); \
	_MM512_DIV_LINE_PD(a, b, c)

#define _MM512_LHS_DIV_LINE_S_PD(a, B, c) \
	__m512d b = _mm512_set1_pd(*B); \
	_MM512_DIV_LINE_PD(b, a, c)

/* COMPLEX SCALAR-MATRIX OPS */

#define _MM_ADD_LINE_SC_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128 b = _mm_setr_ps(B[0], B[1], B[0], B[1]); \
	_MM_ADD_LINE_PS(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3) \
}

#define _MM_SUB_LINE_SC_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128 b = _mm_setr_ps(B[0], B[1], B[0], B[1]); \
	_MM_SUB_LINE_PS(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3) \
}

#define _MM_LHS_SUB_LINE_SC_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128 b = _mm_setr_ps(B[0], B[1], B[0], B[1]); \
	_MM_SUB_LINE_PS(b, b, b, b, a0, a1, a2, a3, c0, c1, c2, c3) \
}

#define _MM_MUL_LINE_SC_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128 br = _mm_set1_ps(B[0]); \
	__m128 bi = _mm_set1_ps(B[1]); \
	__m128 a0_s = _mm_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m128 a1_s = _mm_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m128 a2_s = _mm_shuffle_ps(a2, a2, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m128 a3_s = _mm_shuffle_ps(a3, a3, _MM_SHUFFLE(2, 3, 0, 1)); \
	c0 = _mm_fmaddsub_ps(a0, br, _mm_mul_ps(a0_s, bi)); \
	c1 = _mm_fmaddsub_ps(a1, br, _mm_mul_ps(a1_s, bi)); \
	c2 = _mm_fmaddsub_ps(a2, br, _mm_mul_ps(a2_s, bi)); \
	c3 = _mm_fmaddsub_ps(a3, br, _mm_mul_ps(a3_s, bi)); \
}

#define _MM_DIV_LINE_SC_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128 br = _mm_set1_ps(B[0]); \
	__m128 bi = _mm_set1_ps(-B[1]); \
	__m128 norm_sq = _mm_set1_ps(B[0] * B[0] + B[1] * B[1]); \
	__m128 a0_s = _mm_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m128 a1_s = _mm_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m128 a2_s = _mm_shuffle_ps(a2, a2, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m128 a3_s = _mm_shuffle_ps(a3, a3, _MM_SHUFFLE(2, 3, 0, 1)); \
	c0 = _mm_div_ps(_mm_fmaddsub_ps(a0, br, _mm_mul_ps(a0_s, bi)), norm_sq); \
	c1 = _mm_div_ps(_mm_fmaddsub_ps(a1, br, _mm_mul_ps(a1_s, bi)), norm_sq); \
	c2 = _mm_div_ps(_mm_fmaddsub_ps(a2, br, _mm_mul_ps(a2_s, bi)), norm_sq); \
	c3 = _mm_div_ps(_mm_fmaddsub_ps(a3, br, _mm_mul_ps(a3_s, bi)), norm_sq); \
}

#define _MM_LHS_DIV_LINE_SC_PS(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128 br = _mm_set1_ps(B[0]); \
	__m128 bi = _mm_set1_ps(B[1]); \
	__m128 conj_sign = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f); \
 \
	__m128 a0_c = _mm_mul_ps(a0, conj_sign); \
	__m128 a1_c = _mm_mul_ps(a1, conj_sign); \
	__m128 a2_c = _mm_mul_ps(a2, conj_sign); \
	__m128 a3_c = _mm_mul_ps(a3, conj_sign); \
 \
	__m128 a0_s = _mm_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m128 a1_s = _mm_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m128 a2_s = _mm_shuffle_ps(a2, a2, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m128 a3_s = _mm_shuffle_ps(a3, a3, _MM_SHUFFLE(2, 3, 0, 1)); \
 \
	__m128 norm0 = _mm_add_ps(_mm_mul_ps(a0, a0), _mm_mul_ps(a0_s, a0_s)); \
	__m128 norm1 = _mm_add_ps(_mm_mul_ps(a1, a1), _mm_mul_ps(a1_s, a1_s)); \
	__m128 norm2 = _mm_add_ps(_mm_mul_ps(a2, a2), _mm_mul_ps(a2_s, a2_s)); \
	__m128 norm3 = _mm_add_ps(_mm_mul_ps(a3, a3), _mm_mul_ps(a3_s, a3_s)); \
 \
	c0 = _mm_div_ps(_mm_fmaddsub_ps(br, a0_c, _mm_mul_ps(_mm_shuffle_ps(a0_c, a0_c, _MM_SHUFFLE(2, 3, 0, 1)), bi)), norm0); \
	c1 = _mm_div_ps(_mm_fmaddsub_ps(br, a1_c, _mm_mul_ps(_mm_shuffle_ps(a1_c, a1_c, _MM_SHUFFLE(2, 3, 0, 1)), bi)), norm1); \
	c2 = _mm_div_ps(_mm_fmaddsub_ps(br, a2_c, _mm_mul_ps(_mm_shuffle_ps(a2_c, a2_c, _MM_SHUFFLE(2, 3, 0, 1)), bi)), norm2); \
	c3 = _mm_div_ps(_mm_fmaddsub_ps(br, a3_c, _mm_mul_ps(_mm_shuffle_ps(a3_c, a3_c, _MM_SHUFFLE(2, 3, 0, 1)), bi)), norm3); \
}

#define _MM_ADD_LINE_SC_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128d b = _mm_setr_pd(B[0], B[1]); \
	_MM_ADD_LINE_PD(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3) \
}

#define _MM_SUB_LINE_SC_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128d b = _mm_setr_pd(B[0], B[1]); \
	_MM_SUB_LINE_PD(a0, a1, a2, a3, b, b, b, b, c0, c1, c2, c3) \
}

#define _MM_LHS_SUB_LINE_SC_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128d b = _mm_setr_pd(B[0], B[1]); \
	_MM_SUB_LINE_PD(b, b, b, b, a0, a1, a2, a3, c0, c1, c2, c3) \
}

#define _MM_MUL_LINE_SC_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128d br = _mm_set1_pd(B[0]); \
	__m128d bi = _mm_set1_pd(B[1]); \
	__m128d a0_s = _mm_shuffle_pd(a0, a0, _MM_SHUFFLE2(0, 1)); \
	__m128d a1_s = _mm_shuffle_pd(a1, a1, _MM_SHUFFLE2(0, 1)); \
	__m128d a2_s = _mm_shuffle_pd(a2, a2, _MM_SHUFFLE2(0, 1)); \
	__m128d a3_s = _mm_shuffle_pd(a3, a3, _MM_SHUFFLE2(0, 1)); \
	c0 = _mm_fmaddsub_pd(a0, br, _mm_mul_pd(a0_s, bi)); \
	c1 = _mm_fmaddsub_pd(a1, br, _mm_mul_pd(a1_s, bi)); \
	c2 = _mm_fmaddsub_pd(a2, br, _mm_mul_pd(a2_s, bi)); \
	c3 = _mm_fmaddsub_pd(a3, br, _mm_mul_pd(a3_s, bi)); \
}

#define _MM_DIV_LINE_SC_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128d br = _mm_set1_pd(B[0]); \
	__m128d bi = _mm_set1_pd(-B[1]); \
	__m128d norm_sq = _mm_set1_pd(B[0] * B[0] + B[1] * B[1]); \
	__m128d a0_s = _mm_shuffle_pd(a0, a0, _MM_SHUFFLE2(0, 1)); \
	__m128d a1_s = _mm_shuffle_pd(a1, a1, _MM_SHUFFLE2(0, 1)); \
	__m128d a2_s = _mm_shuffle_pd(a2, a2, _MM_SHUFFLE2(0, 1)); \
	__m128d a3_s = _mm_shuffle_pd(a3, a3, _MM_SHUFFLE2(0, 1)); \
	c0 = _mm_div_pd(_mm_fmaddsub_pd(a0, br, _mm_mul_pd(a0_s, bi)), norm_sq); \
	c1 = _mm_div_pd(_mm_fmaddsub_pd(a1, br, _mm_mul_pd(a1_s, bi)), norm_sq); \
	c2 = _mm_div_pd(_mm_fmaddsub_pd(a2, br, _mm_mul_pd(a2_s, bi)), norm_sq); \
	c3 = _mm_div_pd(_mm_fmaddsub_pd(a3, br, _mm_mul_pd(a3_s, bi)), norm_sq); \
}

#define _MM_LHS_DIV_LINE_SC_PD(a0, a1, a2, a3, B, c0, c1, c2, c3) \
{ \
	__m128d br = _mm_set1_pd(B[0]); \
	__m128d bi = _mm_set1_pd(B[1]); \
	__m128d conj_sign = _mm_set_pd(-1.0, 1.0); \
 \
	__m128d a0_c = _mm_mul_pd(a0, conj_sign); \
	__m128d a1_c = _mm_mul_pd(a1, conj_sign); \
	__m128d a2_c = _mm_mul_pd(a2, conj_sign); \
	__m128d a3_c = _mm_mul_pd(a3, conj_sign); \
 \
	__m128d a0_s = _mm_shuffle_pd(a0, a0, _MM_SHUFFLE2(0, 1)); \
	__m128d a1_s = _mm_shuffle_pd(a1, a1, _MM_SHUFFLE2(0, 1)); \
	__m128d a2_s = _mm_shuffle_pd(a2, a2, _MM_SHUFFLE2(0, 1)); \
	__m128d a3_s = _mm_shuffle_pd(a3, a3, _MM_SHUFFLE2(0, 1)); \
 \
	__m128d norm0 = _mm_add_pd(_mm_mul_pd(a0, a0), _mm_mul_pd(a0_s, a0_s)); \
	__m128d norm1 = _mm_add_pd(_mm_mul_pd(a1, a1), _mm_mul_pd(a1_s, a1_s)); \
	__m128d norm2 = _mm_add_pd(_mm_mul_pd(a2, a2), _mm_mul_pd(a2_s, a2_s)); \
	__m128d norm3 = _mm_add_pd(_mm_mul_pd(a3, a3), _mm_mul_pd(a3_s, a3_s)); \
 \
	c0 = _mm_div_pd(_mm_fmaddsub_pd(br, a0_c, _mm_mul_pd(_mm_shuffle_pd(a0_c, a0_c, _MM_SHUFFLE2(0, 1)), bi)), norm0); \
	c1 = _mm_div_pd(_mm_fmaddsub_pd(br, a1_c, _mm_mul_pd(_mm_shuffle_pd(a1_c, a1_c, _MM_SHUFFLE2(0, 1)), bi)), norm1); \
	c2 = _mm_div_pd(_mm_fmaddsub_pd(br, a2_c, _mm_mul_pd(_mm_shuffle_pd(a2_c, a2_c, _MM_SHUFFLE2(0, 1)), bi)), norm2); \
	c3 = _mm_div_pd(_mm_fmaddsub_pd(br, a3_c, _mm_mul_pd(_mm_shuffle_pd(a3_c, a3_c, _MM_SHUFFLE2(0, 1)), bi)), norm3); \
}

#define _MM256_ADD_LINE_SC_PS(a0, a1, B, c0, c1) \
{ \
	__m256 b = _mm256_setr_ps(B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1]); \
	_MM256_ADD_LINE_PS(a0, a1, b, b, c0, c1) \
}

#define _MM256_SUB_LINE_SC_PS(a0, a1, B, c0, c1) \
{ \
	__m256 b = _mm256_setr_ps(B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1]); \
	_MM256_SUB_LINE_PS(a0, a1, b, b, c0, c1) \
}

#define _MM256_LHS_SUB_LINE_SC_PS(a0, a1, B, c0, c1) \
{ \
	__m256 b = _mm256_setr_ps(B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1]); \
	_MM256_SUB_LINE_PS(b, b, a0, a1, c0, c1) \
}

#define _MM256_MUL_LINE_SC_PS(a0, a1, B, c0, c1) \
{ \
	__m256 br = _mm256_set1_ps(B[0]); \
	__m256 bi = _mm256_set1_ps(B[1]); \
	__m256 a0_s = _mm256_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m256 a1_s = _mm256_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1)); \
	c0 = _mm256_fmaddsub_ps(a0, br, _mm256_mul_ps(a0_s, bi)); \
	c1 = _mm256_fmaddsub_ps(a1, br, _mm256_mul_ps(a1_s, bi)); \
}

#define _MM256_DIV_LINE_SC_PS(a0, a1, B, c0, c1) \
{ \
	__m256 br = _mm256_set1_ps(B[0]); \
	__m256 bi = _mm256_set1_ps(-B[1]); \
	__m256 norm_sq = _mm256_set1_ps(B[0] * B[0] + B[1] * B[1]); \
	__m256 a0_s = _mm256_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1)); \
	__m256 a1_s = _mm256_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1)); \
	c0 = _mm256_div_ps(_mm256_fmaddsub_ps(a0, br, _mm256_mul_ps(a0_s, bi)), norm_sq); \
	c1 = _mm256_div_ps(_mm256_fmaddsub_ps(a1, br, _mm256_mul_ps(a1_s, bi)), norm_sq); \
}

#define _MM256_LHS_DIV_LINE_SC_PS(a0, a1, B, c0, c1) \
{ \
	const __m256 br = _mm256_set1_ps(B[0]); \
	const __m256 bi = _mm256_set1_ps(B[1]); \
	const __m256 conj_mask = _mm256_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f); \
	\
	const __m256 a0_conj = _mm256_mul_ps(a0, conj_mask); \
	const __m256 a1_conj = _mm256_mul_ps(a1, conj_mask); \
	\
	const __m256 a0_swap = _mm256_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1)); \
	const __m256 a1_swap = _mm256_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1)); \
	\
	const __m256 norm0 = _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a0_swap, a0_swap)); \
	const __m256 norm1 = _mm256_add_ps(_mm256_mul_ps(a1, a1), _mm256_mul_ps(a1_swap, a1_swap)); \
	\
	const __m256 num0 = _mm256_fmaddsub_ps(a0_conj, br, _mm256_mul_ps(_mm256_shuffle_ps(a0_conj, a0_conj, _MM_SHUFFLE(2, 3, 0, 1)), bi)); \
	const __m256 num1 = _mm256_fmaddsub_ps(a1_conj, br, _mm256_mul_ps(_mm256_shuffle_ps(a1_conj, a1_conj, _MM_SHUFFLE(2, 3, 0, 1)), bi)); \
	\
	c0 = _mm256_div_ps(num0, norm0); \
	c1 = _mm256_div_ps(num1, norm1); \
}


#define _MM256_ADD_LINE_SC_PD(a0, a1, B, c0, c1) \
{ \
	__m256d b = _mm256_setr_pd(B[0], B[1], B[0], B[1]); \
	_MM256_ADD_LINE_PD(a0, a1, b, b, c0, c1) \
}

#define _MM256_SUB_LINE_SC_PD(a0, a1, B, c0, c1) \
{ \
	__m256d b = _mm256_setr_pd(B[0], B[1], B[0], B[1]); \
	_MM256_SUB_LINE_PD(a0, a1, b, b, c0, c1) \
}

#define _MM256_LHS_SUB_LINE_SC_PD(a0, a1, B, c0, c1) \
{ \
	__m256d b = _mm256_setr_pd(B[0], B[1], B[0], B[1]); \
	_MM256_SUB_LINE_PD(b, b, a0, a1, c0, c1) \
}

#define _MM256_MUL_LINE_SC_PD(a0, a1, B, c0, c1) \
{ \
	__m256d br = _mm256_set1_pd(B[0]); \
	__m256d bi = _mm256_set1_pd(B[1]); \
	__m256d a0_s = _mm256_shuffle_pd(a0, a0, 0x5); \
	__m256d a1_s = _mm256_shuffle_pd(a1, a1, 0x5); \
	c0 = _mm256_fmaddsub_pd(a0, br, _mm256_mul_pd(a0_s, bi)); \
	c1 = _mm256_fmaddsub_pd(a1, br, _mm256_mul_pd(a1_s, bi)); \
}

#define _MM256_DIV_LINE_SC_PD(a0, a1, B, c0, c1) \
{ \
	__m256d br = _mm256_set1_pd(B[0]); \
	__m256d bi = _mm256_set1_pd(-B[1]); \
	__m256d norm_sq = _mm256_set1_pd(B[0] * B[0] + B[1] * B[1]); \
	__m256d a0_s = _mm256_shuffle_pd(a0, a0, 0x5); \
	__m256d a1_s = _mm256_shuffle_pd(a1, a1, 0x5); \
	c0 = _mm256_div_pd(_mm256_fmaddsub_pd(a0, br, _mm256_mul_pd(a0_s, bi)), norm_sq); \
	c1 = _mm256_div_pd(_mm256_fmaddsub_pd(a1, br, _mm256_mul_pd(a1_s, bi)), norm_sq); \
}

#define _MM256_LHS_DIV_LINE_SC_PD(a0, a1, B, c0, c1) \
{  \
	const __m256d br = _mm256_set1_pd(B[0]); \
	const __m256d bi = _mm256_set1_pd(B[1]); \
	const __m256d conj_mask = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0); \
	\
	const __m256d a0_conj = _mm256_mul_pd(a0, conj_mask); \
	const __m256d a1_conj = _mm256_mul_pd(a1, conj_mask); \
	\
	const __m256d a0_swap = _mm256_shuffle_pd(a0, a0, 0x5); \
	const __m256d a1_swap = _mm256_shuffle_pd(a1, a1, 0x5); \
	\
	const __m256d norm0 = _mm256_add_pd(_mm256_mul_pd(a0, a0), _mm256_mul_pd(a0_swap, a0_swap)); \
	const __m256d norm1 = _mm256_add_pd(_mm256_mul_pd(a1, a1), _mm256_mul_pd(a1_swap, a1_swap)); \
	\
	const __m256d num0 = _mm256_fmaddsub_pd(a0_conj, br, _mm256_mul_pd(_mm256_shuffle_pd(a0_conj, a0_conj, 0x5), bi)); \
	const __m256d num1 = _mm256_fmaddsub_pd(a1_conj, br, _mm256_mul_pd(_mm256_shuffle_pd(a1_conj, a1_conj, 0x5), bi)); \
	\
	c0 = _mm256_div_pd(num0, norm0); \
	c1 = _mm256_div_pd(num1, norm1); \
}

#define _MM512_ADD_LINE_SC_PS(a, B, c) \
{ \
	__m512 b = _mm512_setr4_ps(B[0], B[1], B[0], B[1]); \
	_MM512_ADD_LINE_PS(a, b, c) \
}

#define _MM512_SUB_LINE_SC_PS(a, B, c) \
{\
	__m512 b = _mm512_setr4_ps(B[0], B[1], B[0], B[1]); \
	_MM512_SUB_LINE_PS(a, b, c) \
}

#define _MM512_LHS_SUB_LINE_SC_PS(a, B, c) \
{\
	__m512 b = _mm512_setr4_ps(B[0], B[1], B[0], B[1]); \
	_MM512_SUB_LINE_PS(b, a, c) \
}

#define _MM512_MUL_LINE_SC_PS(a, B, c) \
{\
	__m512 br = _mm512_set1_ps(B[0]); \
	__m512 bi = _mm512_set1_ps(B[1]); \
	__m512 a_s = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)); \
	c = _mm512_fmaddsub_ps(a, br, _mm512_mul_ps(a_s, bi)); \
}

#define _MM512_DIV_LINE_SC_PS(a, B, c) \
{\
	__m512 br = _mm512_set1_ps(B[0]); \
	__m512 bi = _mm512_set1_ps(-B[1]); \
	__m512 norm_sq = _mm512_set1_ps(B[0] * B[0] + B[1] * B[1]); \
	__m512 a_s = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)); \
	c = _mm512_div_ps(_mm512_fmaddsub_ps(a, br, _mm512_mul_ps(a_s, bi)), norm_sq); \
}

#define _MM512_LHS_DIV_LINE_SC_PS(a, B, c) \
{ \
	const __m512 br = _mm512_set1_ps(B[0]); \
	const __m512 bi = _mm512_set1_ps(B[1]); \
	const __m512 conj_mask = _mm512_setr4_ps(1.0f, -1.0f, 1.0f, -1.0f); \
	\
	const __m512 a_conj = _mm512_mul_ps(a, conj_mask); \
	const __m512 a_swap = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)); \
	\
	const __m512 norm = _mm512_add_ps(_mm512_mul_ps(a, a), _mm512_mul_ps(a_swap, a_swap)); \
	const __m512 numerator = _mm512_fmaddsub_ps(a_conj, br, _mm512_mul_ps(_mm512_shuffle_ps(a_conj, a_conj, _MM_SHUFFLE(2, 3, 0, 1)), bi)); \
	\
	c = _mm512_div_ps(numerator, norm); \
}

#define _MM512_ADD_LINE_SC_PD(a, B, c) \
{ \
	__m512d b = _mm512_setr4_pd(B[0], B[1], B[0], B[1]); \
	_MM512_ADD_LINE_PD(a, b, c) \
}

#define _MM512_SUB_LINE_SC_PD(a, B, c) \
{ \
	__m512d b = _mm512_setr4_pd(B[0], B[1], B[0], B[1]); \
	_MM512_SUB_LINE_PD(a, b, c) \
}

#define _MM512_LHS_SUB_LINE_SC_PD(a, B, c) \
{ \
	__m512d b = _mm512_setr4_pd(B[0], B[1], B[0], B[1]); \
	_MM512_SUB_LINE_PD(b, a, c) \
}

#define _MM512_MUL_LINE_SC_PD(a, B, c) \
{ \
	__m512d br = _mm512_set1_pd(B[0]); \
	__m512d bi = _mm512_set1_pd(B[1]); \
	__m512d a_s = _mm512_shuffle_pd(a, a, 0x55); \
	c = _mm512_fmaddsub_pd(a, br, _mm512_mul_pd(a_s, bi)); \
}

#define _MM512_DIV_LINE_SC_PD(a, B, c) \
{ \
	__m512d br = _mm512_set1_pd(B[0]); \
	__m512d bi = _mm512_set1_pd(-B[1]); \
	__m512d norm_sq = _mm512_set1_pd(B[0] * B[0] + B[1] * B[1]); \
	__m512d a_s = _mm512_shuffle_pd(a, a, 0x55); \
	c = _mm512_div_pd(_mm512_fmaddsub_pd(a, br, _mm512_mul_pd(a_s, bi)), norm_sq); \
}

#define _MM512_LHS_DIV_LINE_SC_PD(a, B, c) \
{ \
	const __m512d br = _mm512_set1_pd(B[0]); \
	const __m512d bi = _mm512_set1_pd(B[1]); \
	const __m512d conj_mask = _mm512_setr4_pd(1.0, -1.0, 1.0, -1.0); \
	\
	const __m512d a_conj = _mm512_mul_pd(a, conj_mask); \
	const __m512d a_swap = _mm512_shuffle_pd(a, a, 0x55); \
	\
	const __m512d norm = _mm512_add_pd(_mm512_mul_pd(a, a), _mm512_mul_pd(a_swap, a_swap)); \
	const __m512d numerator = _mm512_fmaddsub_pd(a_conj, br, _mm512_mul_pd(_mm512_shuffle_pd(a_conj, a_conj, 0x55), bi)); \
	\
	c = _mm512_div_pd(numerator, norm); \
}


/* COMPLEX ARITHMETIC OPS */

#define _MM_MUL_C_PS(a, b, c) \
{ \
	const __m128 a_perm = _mm_shuffle_ps(a, a, 0xB1); \
	const __m128 b_perm = _mm_shuffle_ps(b, b, 0xB1); \
	const __m128 real = _mm_fmsub_ps(a, b, _mm_mul_ps(a_perm, b_perm)); \
	const __m128 imag = _mm_fmadd_ps(a_perm, b, _mm_mul_ps(a, b_perm)); \
	c = _mm_blend_ps(real, imag, 0xA); \
}

#define _MM_MUL_C_PD(a, b, c) \
{ \
	const __m128d a_perm = _mm_shuffle_pd(a, a, 0x1); \
	const __m128d b_perm = _mm_shuffle_pd(b, b, 0x1); \
	const __m128d real = _mm_fmsub_pd(a, b, _mm_mul_pd(a_perm, b_perm)); \
	const __m128d imag = _mm_fmadd_pd(a_perm, b, _mm_mul_pd(a, b_perm)); \
	c = _mm_blend_pd(real, imag, 0x2); \
}

#define _MM256_MUL_C_PS(a, b, c) \
{ \
	const __m256 a_perm = _mm256_permute_ps(a, 0xB1); \
	const __m256 b_perm = _mm256_permute_ps(b, 0xB1); \
	const __m256 real = _mm256_fmsub_ps(a, b, _mm256_mul_ps(a_perm, b_perm)); \
	const __m256 imag = _mm256_fmadd_ps(a_perm, b, _mm256_mul_ps(a, b_perm)); \
	c = _mm256_blend_ps(real, imag, 0xAA); \
}

#define _MM256_MUL_C_PD(a, b, c) \
{ \
	const __m256d a_perm = _mm256_permute_pd(a, 0x5); \
	const __m256d b_perm = _mm256_permute_pd(b, 0x5); \
	const __m256d real = _mm256_fmsub_pd(a, b, _mm256_mul_pd(a_perm, b_perm)); \
	const __m256d imag = _mm256_fmadd_pd(a_perm, b, _mm256_mul_pd(a, b_perm)); \
	c = _mm256_blend_pd(real, imag, 0xA); \
}

#define _MM512_MUL_C_PS(a, b, c) \
{ \
	const __m512 a_perm = _mm512_permute_ps(a, 0xB1); \
	const __m512 b_perm = _mm512_permute_ps(b, 0xB1); \
	const __m512 real = _mm512_fmsub_ps(a, b, _mm512_mul_ps(a_perm, b_perm)); \
	const __m512 imag = _mm512_fmadd_ps(a_perm, b, _mm512_mul_ps(a, b_perm)); \
	c = _mm512_mask_blend_ps(0xAAAA, real, imag); \
}

#define _MM512_MUL_C_PD(a, b, c) \
{ \
	__m512d a_perm = _mm512_shuffle_pd(a, a, 0x55); \
	__m512d b_perm = _mm512_shuffle_pd(b, b, 0x55); \
	__m512d real = _mm512_fmsub_pd(a, b, _mm512_mul_pd(a_perm, b_perm)); \
	__m512d imag = _mm512_fmadd_pd(a, b_perm, _mm512_mul_pd(a_perm, b)); \
	c = _mm512_mask_blend_pd(0xAA, real, imag); \
}

#define _MM_MUL_LINE_C_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
{ \
	_MM_MUL_C_PS(a0, b0, c0) \
	_MM_MUL_C_PS(a1, b1, c1) \
	_MM_MUL_C_PS(a2, b2, c2) \
	_MM_MUL_C_PS(a3, b3, c3) \
}

#define _MM_MUL_LINE_C_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
{ \
	_MM_MUL_C_PD(a0, b0, c0) \
	_MM_MUL_C_PD(a1, b1, c1) \
	_MM_MUL_C_PD(a2, b2, c2) \
	_MM_MUL_C_PD(a3, b3, c3) \
}

#define _MM256_MUL_LINE_C_PS(a0, a1, b0, b1, c0, c1) \
{ \
	_MM256_MUL_C_PS(a0, b0, c0) \
	_MM256_MUL_C_PS(a1, b1, c1) \
}

#define _MM256_MUL_LINE_C_PD(a0, a1, b0, b1, c0, c1) \
{ \
	_MM256_MUL_C_PD(a0, b0, c0) \
	_MM256_MUL_C_PD(a1, b1, c1) \
}

#define _MM512_MUL_LINE_C_PS(a, b, c) \
{ \
	_MM512_MUL_C_PS(a, b, c) \
}

#define _MM512_MUL_LINE_C_PD(a, b, c) \
{ \
	_MM512_MUL_C_PD(a, b, c) \
}

#define _MM_DIV_C_PS(a, b, c) \
{ \
	const __m128 b_conj = _mm_mul_ps(b, _mm_setr_ps(1.0f, -1.0f, 1.0f, -1.0f)); \
	const __m128 b_r = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 0, 0)); \
	const __m128 b_i = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1)); \
	const __m128 b_norm = _mm_fmadd_ps(b_r, b_r, _mm_mul_ps(b_i, b_i)); \
	const __m128 a_perm = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)); \
	const __m128 b_conj_perm = _mm_shuffle_ps(b_conj, b_conj, _MM_SHUFFLE(2, 3, 0, 1)); \
	const __m128 real = _mm_fmsub_ps(a, b_conj, _mm_mul_ps(a_perm, b_conj_perm)); \
	const __m128 imag = _mm_fmadd_ps(a_perm, b_conj, _mm_mul_ps(a, b_conj_perm)); \
	const __m128 result = _mm_blend_ps(real, imag, 0xA); \
	c = _mm_div_ps(result, b_norm); \
}

#define _MM_DIV_C_PD(a, b, c) \
{ \
	const __m128d b_conj = _mm_mul_pd(b, _mm_setr_pd(1.0, -1.0)); \
	const __m128d b_r = _mm_unpacklo_pd(b, b); \
	const __m128d b_i = _mm_unpackhi_pd(b, b); \
	const __m128d b_norm = _mm_fmadd_pd(b_r, b_r, _mm_mul_pd(b_i, b_i)); \
	const __m128d a_perm = _mm_shuffle_pd(a, a, _MM_SHUFFLE2(0, 1)); \
	const __m128d b_conj_perm = _mm_shuffle_pd(b_conj, b_conj, _MM_SHUFFLE2(0, 1)); \
	const __m128d real = _mm_fmsub_pd(a, b_conj, _mm_mul_pd(a_perm, b_conj_perm)); \
	const __m128d imag = _mm_fmadd_pd(a_perm, b_conj, _mm_mul_pd(a, b_conj_perm)); \
	const __m128d result = _mm_blend_pd(real, imag, 0x2); \
	c = _mm_div_pd(result, b_norm); \
}

#define _MM256_DIV_C_PS(a, b, c) \
{ \
	const __m256 b_conj = _mm256_mul_ps(b, _mm256_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f)); \
	const __m256 b_r = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 0, 0)); \
	const __m256 b_i = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1)); \
	const __m256 b_norm = _mm256_fmadd_ps(b_r, b_r, _mm256_mul_ps(b_i, b_i)); \
	const __m256 a_perm = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)); \
	const __m256 b_conj_perm = _mm256_shuffle_ps(b_conj, b_conj, _MM_SHUFFLE(2, 3, 0, 1)); \
	const __m256 real = _mm256_fmsub_ps(a, b_conj, _mm256_mul_ps(a_perm, b_conj_perm)); \
	const __m256 imag = _mm256_fmadd_ps(a_perm, b_conj, _mm256_mul_ps(a, b_conj_perm)); \
	const __m256 result = _mm256_blend_ps(real, imag, 0xAA); \
	c = _mm256_div_ps(result, b_norm); \
}

#define _MM256_DIV_C_PD(a, b, c) \
{ \
	const __m256d b_conj = _mm256_mul_pd(b, _mm256_setr_pd(1.0, -1.0, 1.0, -1.0)); \
	const __m256d b_r = _mm256_unpacklo_pd(b, b); \
	const __m256d b_i = _mm256_unpackhi_pd(b, b); \
	const __m256d b_norm = _mm256_fmadd_pd(b_r, b_r, _mm256_mul_pd(b_i, b_i)); \
	const __m256d a_perm = _mm256_shuffle_pd(a, a, 0x5); \
	const __m256d b_conj_perm = _mm256_shuffle_pd(b_conj, b_conj, 0x5); \
	const __m256d real = _mm256_fmsub_pd(a, b_conj, _mm256_mul_pd(a_perm, b_conj_perm)); \
	const __m256d imag = _mm256_fmadd_pd(a_perm, b_conj, _mm256_mul_pd(a, b_conj_perm)); \
	const __m256d result = _mm256_blend_pd(real, imag, 0xA); \
	c = _mm256_div_pd(result, b_norm); \
}

#define _MM512_DIV_C_PS(a, b, c) \
{ \
	const __m512 b_conj = _mm512_mul_ps(b, _mm512_setr4_ps(1.0f, -1.0f, 1.0f, -1.0f)); \
	const __m512 b_r = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 0, 0)); \
	const __m512 b_i = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1)); \
	const __m512 b_norm = _mm512_fmadd_ps(b_r, b_r, _mm512_mul_ps(b_i, b_i)); \
	const __m512 a_perm = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)); \
	const __m512 b_conj_perm = _mm512_shuffle_ps(b_conj, b_conj, _MM_SHUFFLE(2, 3, 0, 1)); \
	const __m512 real = _mm512_fmsub_ps(a, b_conj, _mm512_mul_ps(a_perm, b_conj_perm)); \
	const __m512 imag = _mm512_fmadd_ps(a_perm, b_conj, _mm512_mul_ps(a, b_conj_perm)); \
	const __m512 result = _mm512_mask_blend_ps(0xAAAA, real, imag); \
	c = _mm512_div_ps(result, b_norm); \
}

#define _MM512_DIV_C_PD(a, b, c) \
{ \
	const __m512d b_conj = _mm512_mul_pd(b, _mm512_setr4_pd(1.0, -1.0, 1.0, -1.0)); \
	const __m512d b_r = _mm512_unpacklo_pd(b, b); \
	const __m512d b_i = _mm512_unpackhi_pd(b, b); \
	const __m512d b_norm = _mm512_fmadd_pd(b_r, b_r, _mm512_mul_pd(b_i, b_i)); \
	const __m512d a_perm = _mm512_shuffle_pd(a, a, 0x55); \
	const __m512d b_conj_perm = _mm512_shuffle_pd(b_conj, b_conj, 0x55); \
	const __m512d real = _mm512_fmsub_pd(a, b_conj, _mm512_mul_pd(a_perm, b_conj_perm)); \
	const __m512d imag = _mm512_fmadd_pd(a_perm, b_conj, _mm512_mul_pd(a, b_conj_perm)); \
	const __m512d result = _mm512_mask_blend_pd(0xAA, real, imag); \
	c = _mm512_div_pd(result, b_norm); \
}

#define _MM_DIV_LINE_C_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
{ \
	_MM_DIV_C_PS(a0, b0, c0) \
	_MM_DIV_C_PS(a1, b1, c1) \
	_MM_DIV_C_PS(a2, b2, c2) \
	_MM_DIV_C_PS(a3, b3, c3) \
}

#define _MM_DIV_LINE_C_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
{ \
	_MM_DIV_C_PD(a0, b0, c0) \
	_MM_DIV_C_PD(a1, b1, c1) \
	_MM_DIV_C_PD(a2, b2, c2) \
	_MM_DIV_C_PD(a3, b3, c3) \
}

#define _MM256_DIV_LINE_C_PS(a0, a1, b0, b1, c0, c1) \
{ \
	_MM256_DIV_C_PS(a0, b0, c0) \
	_MM256_DIV_C_PS(a1, b1, c1) \
}

#define _MM256_DIV_LINE_C_PD(a0, a1, b0, b1, c0, c1) \
{ \
	_MM256_DIV_C_PD(a0, b0, c0) \
	_MM256_DIV_C_PD(a1, b1, c1) \
}

#define _MM512_DIV_LINE_C_PS(a, b, c) \
{ \
	_MM512_DIV_C_PS(a, b, c) \
}

#define _MM512_DIV_LINE_C_PD(a, b, c) \
{ \
	_MM512_DIV_C_PD(a, b, c) \
}

/* REDUCTION OPS */

#define _MM_HADD_PS(acc, res) \
{ \
	const __m128 tmp1 = _mm_movehdup_ps(acc); \
	const __m128 sum1 = _mm_add_ps(acc, tmp1); \
	const __m128 tmp2 = _mm_movehl_ps(sum1, sum1); \
	const __m128 sum2 = _mm_add_ss(sum1, tmp2); \
	res = _mm_cvtss_f32(sum2); \
}

#define _MM_HMUL_PS(acc, res) \
{ \
	const __m128 tmp1 = _mm_movehdup_ps(acc); \
	const __m128 prod1 = _mm_mul_ps(acc, tmp1); \
	const __m128 tmp2 = _mm_movehl_ps(prod1, prod1); \
	const __m128 prod2 = _mm_mul_ss(prod1, tmp2); \
	res = _mm_cvtss_f32(prod2); \
}

#define _MM_HADD_PD(acc, res) \
{ \
	const __m128d sum0 = _mm_hadd_pd(acc, acc); \
	res = _mm_cvtsd_f64(sum0); \
}

#define _MM_HMUL_PD(acc, res) \
{ \
	const __m128d temp = _mm_unpackhi_pd(acc, acc); \
	const __m128d prod = _mm_mul_sd(acc, temp); \
	res = _mm_cvtsd_f64(prod); \
}

#define _MM256_HADD_PS(sum, res) \
{ \
	const __m128 lo = _mm256_castps256_ps128(sum); \
	const __m128 hi = _mm256_extractf128_ps(sum, 1); \
	const __m128 acc = _mm_add_ps(lo, hi); \
	_MM_HADD_PS(acc, res); \
}

#define _MM256_HMUL_PS(prod, res) \
{ \
	const __m128 lo = _mm256_castps256_ps128(prod); \
	const __m128 hi = _mm256_extractf128_ps(prod, 1); \
	const __m128 acc = _mm_mul_ps(lo, hi); \
	_MM_HMUL_PS(acc, res); \
}

#define _MM256_HADD_PD(sum, res) \
{ \
	const __m128d low = _mm256_castpd256_pd128(sum); \
	const __m128d high = _mm256_extractf128_pd(sum, 1); \
	const __m128d acc = _mm_add_pd(low, high); \
	const __m128d tmp = _mm_unpackhi_pd(acc, acc); \
	const __m128d final = _mm_add_sd(acc, tmp); \
	res = _mm_cvtsd_f64(final); \
}

#define _MM256_HMUL_PD(prod, res) \
{ \
	const __m128d low = _mm256_castpd256_pd128(prod); \
	const __m128d high = _mm256_extractf128_pd(prod, 1); \
	const __m128d acc = _mm_mul_pd(low, high); \
	const __m128d tmp = _mm_unpackhi_pd(acc, acc); \
	const __m128d final = _mm_mul_sd(acc, tmp); \
	res = _mm_cvtsd_f64(final); \
}

#define _MM_REDUCE_ADD_LINE_PS(r, a0, a1, a2, a3) \
{\
	__m128 acc = _mm_setzero_ps(); \
	acc = _mm_add_ps(acc, a0); \
	acc = _mm_add_ps(acc, a1); \
	acc = _mm_add_ps(acc, a2); \
	acc = _mm_add_ps(acc, a3); \
	_MM_HADD_PS(acc, *r) \
}

#define _MM_REDUCE_MUL_LINE_PS(r, a0, a1, a2, a3) \
{\
	__m128 acc = _mm_set1_ps(1.0f); \
	acc = _mm_mul_ps(acc, a0); \
	acc = _mm_mul_ps(acc, a1); \
	acc = _mm_mul_ps(acc, a2); \
	acc = _mm_mul_ps(acc, a3); \
	_MM_HMUL_PS(acc, *r) \
}

#define _MM256_REDUCE_ADD_LINE_PS(r, a0, a1) \
{\
	__m256 sum = _mm256_add_ps(a0, a1); \
	_MM256_HADD_PS(sum, *r) \
}

#define _MM256_REDUCE_MUL_LINE_PS(r, a0, a1) \
{\
	__m256 prod = _mm256_mul_ps(a0, a1); \
	_MM256_HMUL_PS(prod, *r) \
}

#define _MM_REDUCE_ADD_LINE_PD(r, a0, a1, a2, a3) \
{ \
	__m128d acc0 = _mm_add_pd(a0, a1); \
	__m128d acc1 = _mm_add_pd(a2, a3); \
	__m128d acc = _mm_add_pd(acc0, acc1); \
	_MM_HADD_PD(acc, *r) \
}

#define _MM_REDUCE_MUL_LINE_PD(r, a0, a1, a2, a3) \
{\
	__m128d acc = _mm_set1_pd(1.0); \
	acc = _mm_mul_pd(acc, a0); \
	acc = _mm_mul_pd(acc, a1); \
	acc = _mm_mul_pd(acc, a2); \
	acc = _mm_mul_pd(acc, a3); \
	_MM_HMUL_PD(acc, *r) \
}

#define _MM256_REDUCE_ADD_LINE_PD(r, a0, a1) \
{\
	__m256d sum = _mm256_add_pd(a0, a1); \
	_MM256_HADD_PD(sum, *r) \
}

#define _MM256_REDUCE_MUL_LINE_PD(r, a0, a1) \
{\
	__m256d prod = _mm256_mul_pd(a0, a1); \
	_MM256_HMUL_PD(prod, *r) \
}

#define _MM512_REDUCE_ADD_LINE_PS(r, a0) \
{ \
	*r = _mm512_reduce_add_ps(a0); \
}

#define _MM512_REDUCE_ADD_LINE_PD(r, a0) \
{ \
	*r = _mm512_reduce_add_pd(a0); \
}

#define _MM512_REDUCE_MUL_LINE_PS(r, a0) \
{ \
	*r = _mm512_reduce_mul_ps(a0); \
}

#define _MM512_REDUCE_MUL_LINE_PD(r, a0) \
{ \
	*r = _mm512_reduce_mul_pd(a0); \
}

/* COMPLEX REDUCTION OPS */

#define _MM_REDUCE_ADD_LINE_C_PS(r, a0, a1, a2, a3) \
{ \
	__m128 sum01 = _mm_add_ps(a0, a1); \
	__m128 sum23 = _mm_add_ps(a2, a3); \
	__m128 total = _mm_add_ps(sum01, sum23); \
	__m128 hi = _mm_movehl_ps(total, total); \
	__m128 final = _mm_add_ps(total, hi); \
	_mm_storel_pi((__m64*)r, final); \
}

#define _MM_REDUCE_ADD_LINE_C_PD(r, a0, a1, a2, a3) \
{ \
	__m128d sum01 = _mm_add_pd(a0, a1); \
	__m128d sum23 = _mm_add_pd(a2, a3); \
	__m128d total = _mm_add_pd(sum01, sum23); \
	_mm_store_pd(r, total); \
}

#define _MM256_REDUCE_ADD_LINE_C_PS(r, a0, a1) \
{ \
	__m256 sum = _mm256_add_ps(a0, a1); \
	__m128 lo = _mm256_castps256_ps128(sum); \
	__m128 hi = _mm256_extractf128_ps(sum, 1); \
	__m128 final = _mm_add_ps(lo, hi); \
	__m128 result = _mm_add_ps(final, _mm_movehl_ps(final, final)); \
	_mm_storel_pi((__m64*)r, result); \
}

#define _MM256_REDUCE_ADD_LINE_C_PD(r, a0, a1) \
{ \
	__m256d sum = _mm256_add_pd(a0, a1); \
	__m128d lo = _mm256_castpd256_pd128(sum); \
	__m128d hi = _mm256_extractf128_pd(sum, 1); \
	__m128d final = _mm_add_pd(lo, hi); \
	_mm_store_pd(r, final); \
}

#define _MM512_REDUCE_ADD_LINE_C_PS(r, a) \
{ \
	__m512 re = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 0, 2, 0)); \
	__m512 im = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 3, 1)); \
	r[0] = _mm512_reduce_add_ps(re) * 0.5000f; \
	r[1] = _mm512_reduce_add_ps(im) * 0.5000f; \
}

#define _MM512_REDUCE_ADD_LINE_C_PD(r, a) \
{ \
	__m512d re = _mm512_shuffle_pd(a, a, 0x00); \
	__m512d im = _mm512_shuffle_pd(a, a, 0xFF); \
	r[0] = _mm512_reduce_add_pd(re) * 0.5000; \
	r[1] = _mm512_reduce_add_pd(im) * 0.5000; \
}

#define _MM_REDUCE_MUL_LINE_C_PD(r, a0, a1, a2, a3) \
{ \
	__m128d acc = a0; \
	_MM_MUL_C_PD(acc, a1, acc); \
	_MM_MUL_C_PD(acc, a2, acc); \
	_MM_MUL_C_PD(acc, a3, acc); \
	_mm_store_pd(r, acc); \
}

#define _MM_REDUCE_MUL_LINE_C_PS(r, a0, a1, a2, a3) \
{ \
	__m128 acc = _mm_setr_ps(1.0f, 0.0f, 1.0f, 0.0f); \
	_MM_MUL_C_PS(acc, a0, acc); \
	_MM_MUL_C_PS(acc, a1, acc); \
	_MM_MUL_C_PS(acc, a2, acc); \
	_MM_MUL_C_PS(acc, a3, acc); \
	__m128 hi = _mm_movehl_ps(acc, acc); \
	__m128 result; \
	_MM_MUL_C_PS(acc, hi, result); \
	_mm_storel_pi((__m64*)r, result); \
}

#define _MM256_REDUCE_MUL_LINE_C_PD(r, a0, a1) \
{ \
	__m256d acc = a0; \
	_MM256_MUL_C_PD(acc, a1, acc); \
	__m128d lo = _mm256_castpd256_pd128(acc); \
	__m128d hi = _mm256_extractf128_pd(acc, 1); \
	__m128d result; \
	_MM_MUL_C_PD(lo, hi, result); \
	_mm_store_pd(r, result); \
}

#define _MM256_REDUCE_MUL_LINE_C_PS(r, a0, a1) \
{ \
	__m256 acc = a0; \
	_MM256_MUL_C_PS(acc, a1, acc); \
	__m128 lo = _mm256_castps256_ps128(acc); \
	__m128 hi = _mm256_extractf128_ps(acc, 1); \
	__m128 result; \
	_MM_MUL_C_PS(lo, hi, result); \
	__m128 hi_part = _mm_movehl_ps(result, result); \
	__m128 final_result; \
	_MM_MUL_C_PS(result, hi_part, final_result); \
	_mm_storel_pi((__m64*)r, final_result); \
}

#define _MM512_REDUCE_MUL_LINE_C_PS(r, a) \
{ \
	__m512 acc = a; \
	const __m256 lo = _mm512_castps512_ps256(acc); \
	const __m256 hi = _mm512_extractf32x8_ps(acc, 1); \
	__m256 prod256; \
	_MM256_MUL_C_PS(lo, hi, prod256); \
	const __m128 lo128 = _mm256_castps256_ps128(prod256); \
	const __m128 hi128 = _mm256_extractf128_ps(prod256, 1); \
	__m128 res; \
	_MM_MUL_C_PS(lo128, hi128, res); \
	__m128 hi_part = _mm_movehl_ps(res, res); \
	__m128 final_res; \
	_MM_MUL_C_PS(res, hi_part, final_res); \
	_mm_storel_pi((__m64*)r, final_res); \
}

#define _MM512_REDUCE_MUL_LINE_C_PD(r, a) \
{ \
	__m512d acc = a; \
	const __m256d lo = _mm512_castpd512_pd256(acc); \
	const __m256d hi = _mm512_extractf64x4_pd(acc, 1); \
	__m256d prod256; \
	_MM256_MUL_C_PD(lo, hi, prod256); \
	const __m128d lo128 = _mm256_castpd256_pd128(prod256); \
	const __m128d hi128 = _mm256_extractf128_pd(prod256, 1); \
	__m128d res; \
	_MM_MUL_C_PD(lo128, hi128, res); \
	_mm_store_pd(r, res); \
}

/* MATRIX-MATRIX MULTIPLY OPS */
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

#define _MM512_MREDUCE_PD(c0, c1, c2, c3, c4, c5, c6, c7, acc) \
{ \
	__m512d l1_01 = _mm512_add_pd( \
		_mm512_shuffle_f64x2(c0, c1, 0x44), \
		_mm512_shuffle_f64x2(c0, c1, 0xEE)); \
	__m512d l1_23 = _mm512_add_pd( \
		_mm512_shuffle_f64x2(c2, c3, 0x44), \
		_mm512_shuffle_f64x2(c2, c3, 0xEE)); \
	__m512d l1_45 = _mm512_add_pd( \
		_mm512_shuffle_f64x2(c4, c5, 0x44), \
		_mm512_shuffle_f64x2(c4, c5, 0xEE)); \
	__m512d l1_67 = _mm512_add_pd( \
		_mm512_shuffle_f64x2(c6, c7, 0x44), \
		_mm512_shuffle_f64x2(c6, c7, 0xEE)); \
	\
	__m512d l2_0123 = _mm512_add_pd( \
		_mm512_shuffle_f64x2(l1_01, l1_23, 0x88), \
		_mm512_shuffle_f64x2(l1_01, l1_23, 0xDD));\
	__m512d l2_4567 = _mm512_add_pd( \
		_mm512_shuffle_f64x2(l1_45, l1_67, 0x88), \
		_mm512_shuffle_f64x2(l1_45, l1_67, 0xDD)); \
	\
	__m512i idx_lo = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0); \
	__m512i idx_hi = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1); \
	\
	__m512d temp_lo = _mm512_permutex2var_pd(l2_0123, idx_lo, l2_4567); \
	__m512d temp_hi = _mm512_permutex2var_pd(l2_0123, idx_hi, l2_4567); \
	\
	acc = _mm512_add_pd(temp_lo, temp_hi); \
}

#define _MM512_MMUL8_PD(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5, c6, c7) \
{\
	__m512d t0, t1, t2, t3, t4, t5, t6, t7, acc; \
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
	_MM512_MREDUCE_PD(t0, t1, t2, t3, t4, t5, t6, t7, acc) \
	c0 = _mm512_add_pd(c0, acc); \
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
	_MM512_MREDUCE_PD(t0, t1, t2, t3, t4, t5, t6, t7, acc) \
	c1 = _mm512_add_pd(c1, acc); \
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
	_MM512_MREDUCE_PD(t0, t1, t2, t3, t4, t5, t6, t7, acc) \
	c2 = _mm512_add_pd(c2, acc); \
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
	_MM512_MREDUCE_PD(t0, t1, t2, t3, t4, t5, t6, t7, acc) \
	c3 = _mm512_add_pd(c3, acc); \
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
	_MM512_MREDUCE_PD(t0, t1, t2, t3, t4, t5, t6, t7, acc) \
	c4 = _mm512_add_pd(c4, acc); \
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
	_MM512_MREDUCE_PD(t0, t1, t2, t3, t4, t5, t6, t7, acc) \
	c5 = _mm512_add_pd(c5, acc); \
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
	_MM512_MREDUCE_PD(t0, t1, t2, t3, t4, t5, t6, t7, acc) \
	c6 = _mm512_add_pd(c6, acc); \
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
	_MM512_MREDUCE_PD(t0, t1, t2, t3, t4, t5, t6, t7, acc) \
	c7 = _mm512_add_pd(c7, acc); \
}

#define _MM512_MREDUCE_PS(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb, cc, cd, ce, cf, acc) \
{ \
	/* Extract 256-bit halves from all 16 registers */ \
	__m256 c0_lo = _mm512_castps512_ps256(c0), c0_hi = _mm512_extractf32x8_ps(c0, 1); \
	__m256 c1_lo = _mm512_castps512_ps256(c1), c1_hi = _mm512_extractf32x8_ps(c1, 1); \
	__m256 c2_lo = _mm512_castps512_ps256(c2), c2_hi = _mm512_extractf32x8_ps(c2, 1); \
	__m256 c3_lo = _mm512_castps512_ps256(c3), c3_hi = _mm512_extractf32x8_ps(c3, 1); \
	__m256 c4_lo = _mm512_castps512_ps256(c4), c4_hi = _mm512_extractf32x8_ps(c4, 1); \
	__m256 c5_lo = _mm512_castps512_ps256(c5), c5_hi = _mm512_extractf32x8_ps(c5, 1); \
	__m256 c6_lo = _mm512_castps512_ps256(c6), c6_hi = _mm512_extractf32x8_ps(c6, 1); \
	__m256 c7_lo = _mm512_castps512_ps256(c7), c7_hi = _mm512_extractf32x8_ps(c7, 1); \
	__m256 c8_lo = _mm512_castps512_ps256(c8), c8_hi = _mm512_extractf32x8_ps(c8, 1); \
	__m256 c9_lo = _mm512_castps512_ps256(c9), c9_hi = _mm512_extractf32x8_ps(c9, 1); \
	__m256 ca_lo = _mm512_castps512_ps256(ca), ca_hi = _mm512_extractf32x8_ps(ca, 1); \
	__m256 cb_lo = _mm512_castps512_ps256(cb), cb_hi = _mm512_extractf32x8_ps(cb, 1); \
	__m256 cc_lo = _mm512_castps512_ps256(cc), cc_hi = _mm512_extractf32x8_ps(cc, 1); \
	__m256 cd_lo = _mm512_castps512_ps256(cd), cd_hi = _mm512_extractf32x8_ps(cd, 1); \
	__m256 ce_lo = _mm512_castps512_ps256(ce), ce_hi = _mm512_extractf32x8_ps(ce, 1); \
	__m256 cf_lo = _mm512_castps512_ps256(cf), cf_hi = _mm512_extractf32x8_ps(cf, 1); \
	\
	/* Stack adjacent pairs for first level of tree reduction */ \
	__m512 stack01_lo = _mm512_insertf32x8(_mm512_castps256_ps512(c0_lo), c1_lo, 1); \
	__m512 stack01_hi = _mm512_insertf32x8(_mm512_castps256_ps512(c0_hi), c1_hi, 1); \
	__m512 stack23_lo = _mm512_insertf32x8(_mm512_castps256_ps512(c2_lo), c3_lo, 1); \
	__m512 stack23_hi = _mm512_insertf32x8(_mm512_castps256_ps512(c2_hi), c3_hi, 1); \
	__m512 stack45_lo = _mm512_insertf32x8(_mm512_castps256_ps512(c4_lo), c5_lo, 1); \
	__m512 stack45_hi = _mm512_insertf32x8(_mm512_castps256_ps512(c4_hi), c5_hi, 1); \
	__m512 stack67_lo = _mm512_insertf32x8(_mm512_castps256_ps512(c6_lo), c7_lo, 1); \
	__m512 stack67_hi = _mm512_insertf32x8(_mm512_castps256_ps512(c6_hi), c7_hi, 1); \
	__m512 stack89_lo = _mm512_insertf32x8(_mm512_castps256_ps512(c8_lo), c9_lo, 1); \
	__m512 stack89_hi = _mm512_insertf32x8(_mm512_castps256_ps512(c8_hi), c9_hi, 1); \
	__m512 stackab_lo = _mm512_insertf32x8(_mm512_castps256_ps512(ca_lo), cb_lo, 1); \
	__m512 stackab_hi = _mm512_insertf32x8(_mm512_castps256_ps512(ca_hi), cb_hi, 1); \
	__m512 stackcd_lo = _mm512_insertf32x8(_mm512_castps256_ps512(cc_lo), cd_lo, 1); \
	__m512 stackcd_hi = _mm512_insertf32x8(_mm512_castps256_ps512(cc_hi), cd_hi, 1); \
	__m512 stackef_lo = _mm512_insertf32x8(_mm512_castps256_ps512(ce_lo), cf_lo, 1); \
	__m512 stackef_hi = _mm512_insertf32x8(_mm512_castps256_ps512(ce_hi), cf_hi, 1); \
	\
	/* First level reduction: add hi and lo parts */ \
	__m512 level1_01 = _mm512_add_ps(stack01_lo, stack01_hi); \
	__m512 level1_23 = _mm512_add_ps(stack23_lo, stack23_hi); \
	__m512 level1_45 = _mm512_add_ps(stack45_lo, stack45_hi); \
	__m512 level1_67 = _mm512_add_ps(stack67_lo, stack67_hi); \
	__m512 level1_89 = _mm512_add_ps(stack89_lo, stack89_hi); \
	__m512 level1_ab = _mm512_add_ps(stackab_lo, stackab_hi); \
	__m512 level1_cd = _mm512_add_ps(stackcd_lo, stackcd_hi); \
	__m512 level1_ef = _mm512_add_ps(stackef_lo, stackef_hi); \
	\
	/* Second level: combine pairs of pairs */ \
	__m256 l1_01_lo = _mm512_castps512_ps256(level1_01), l1_01_hi = _mm512_extractf32x8_ps(level1_01, 1); \
	__m256 l1_23_lo = _mm512_castps512_ps256(level1_23), l1_23_hi = _mm512_extractf32x8_ps(level1_23, 1); \
	__m256 l1_45_lo = _mm512_castps512_ps256(level1_45), l1_45_hi = _mm512_extractf32x8_ps(level1_45, 1); \
	__m256 l1_67_lo = _mm512_castps512_ps256(level1_67), l1_67_hi = _mm512_extractf32x8_ps(level1_67, 1); \
	__m256 l1_89_lo = _mm512_castps512_ps256(level1_89), l1_89_hi = _mm512_extractf32x8_ps(level1_89, 1); \
	__m256 l1_ab_lo = _mm512_castps512_ps256(level1_ab), l1_ab_hi = _mm512_extractf32x8_ps(level1_ab, 1); \
	__m256 l1_cd_lo = _mm512_castps512_ps256(level1_cd), l1_cd_hi = _mm512_extractf32x8_ps(level1_cd, 1); \
	__m256 l1_ef_lo = _mm512_castps512_ps256(level1_ef), l1_ef_hi = _mm512_extractf32x8_ps(level1_ef, 1); \
	\
	__m512 level2_0123 = _mm512_add_ps( \
		_mm512_insertf32x8(_mm512_castps256_ps512(l1_01_lo), l1_23_lo, 1), \
		_mm512_insertf32x8(_mm512_castps256_ps512(l1_01_hi), l1_23_hi, 1)); \
	__m512 level2_4567 = _mm512_add_ps( \
		_mm512_insertf32x8(_mm512_castps256_ps512(l1_45_lo), l1_67_lo, 1), \
		_mm512_insertf32x8(_mm512_castps256_ps512(l1_45_hi), l1_67_hi, 1)); \
	__m512 level2_89ab = _mm512_add_ps( \
		_mm512_insertf32x8(_mm512_castps256_ps512(l1_89_lo), l1_ab_lo, 1), \
		_mm512_insertf32x8(_mm512_castps256_ps512(l1_89_hi), l1_ab_hi, 1)); \
	__m512 level2_cdef = _mm512_add_ps( \
		_mm512_insertf32x8(_mm512_castps256_ps512(l1_cd_lo), l1_ef_lo, 1), \
		_mm512_insertf32x8(_mm512_castps256_ps512(l1_cd_hi), l1_ef_hi, 1)); \
	\
	/* Third level: combine to get final 4 groups of 4 results each */ \
	__m256 l2_0123_lo = _mm512_castps512_ps256(level2_0123), l2_0123_hi = _mm512_extractf32x8_ps(level2_0123, 1); \
	__m256 l2_4567_lo = _mm512_castps512_ps256(level2_4567), l2_4567_hi = _mm512_extractf32x8_ps(level2_4567, 1); \
	__m256 l2_89ab_lo = _mm512_castps512_ps256(level2_89ab), l2_89ab_hi = _mm512_extractf32x8_ps(level2_89ab, 1); \
	__m256 l2_cdef_lo = _mm512_castps512_ps256(level2_cdef), l2_cdef_hi = _mm512_extractf32x8_ps(level2_cdef, 1); \
	\
	__m512 final_0to7 = _mm512_add_ps( \
		_mm512_insertf32x8(_mm512_castps256_ps512(l2_0123_lo), l2_4567_lo, 1), \
		_mm512_insertf32x8(_mm512_castps256_ps512(l2_0123_hi), l2_4567_hi, 1)); \
	__m512 final_8tof = _mm512_add_ps( \
		_mm512_insertf32x8(_mm512_castps256_ps512(l2_89ab_lo), l2_cdef_lo, 1), \
		_mm512_insertf32x8(_mm512_castps256_ps512(l2_89ab_hi), l2_cdef_hi, 1)); \
	\
	/* Final rearrangement and permutation to get all 16 results */ \
	__m512 result = _mm512_add_ps(final_0to7, final_8tof); \
	__m512i perm_idx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0); \
	result = _mm512_permutexvar_ps(perm_idx, result); \
	\
	acc = _mm512_add_ps(acc, result); \
}

#define _MM512_MMUL16_PS(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb, cc, cd, ce, cf) \
{\
	__m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc; \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c0 = _mm512_add_ps(c0, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c1 = _mm512_add_ps(c1, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c2 = _mm512_add_ps(c2, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c3 = _mm512_add_ps(c3, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c4 = _mm512_add_ps(c4, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c5 = _mm512_add_ps(c5, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c6 = _mm512_add_ps(c6, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c7 = _mm512_add_ps(c7, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c8 = _mm512_add_ps(c8, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	c9 = _mm512_add_ps(c9, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	ca = _mm512_add_ps(ca, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	cb = _mm512_add_ps(cb, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	cc = _mm512_add_ps(cc, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	cd = _mm512_add_ps(cd, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	ce = _mm512_add_ps(ce, acc); \
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
	_MM512_MREDUCE_PS(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, acc) \
	cf = _mm512_add_ps(cf, acc); \
}

// #define _MM512_MMUL16_PS(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb, cc, cd, ce, cf) \
// {\
// 	alignas(64) __m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf, c; \
// 	\
// 	t0 = _mm512_mul_ps(a0, b0); \
// 	t1 = _mm512_mul_ps(a0, b1); \
// 	t2 = _mm512_mul_ps(a0, b2); \
// 	t3 = _mm512_mul_ps(a0, b3); \
// 	t4 = _mm512_mul_ps(a0, b4); \
// 	t5 = _mm512_mul_ps(a0, b5); \
// 	t6 = _mm512_mul_ps(a0, b6); \
// 	t7 = _mm512_mul_ps(a0, b7); \
// 	t8 = _mm512_mul_ps(a0, b8); \
// 	t9 = _mm512_mul_ps(a0, b9); \
// 	ta = _mm512_mul_ps(a0, ba); \
// 	tb = _mm512_mul_ps(a0, bb); \
// 	tc = _mm512_mul_ps(a0, bc); \
// 	td = _mm512_mul_ps(a0, bd); \
// 	te = _mm512_mul_ps(a0, be); \
// 	tf = _mm512_mul_ps(a0, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c0 = _mm512_add_ps(c0, c); \
// 	\
// 	t0 = _mm512_mul_ps(a1, b0); \
// 	t1 = _mm512_mul_ps(a1, b1); \
// 	t2 = _mm512_mul_ps(a1, b2); \
// 	t3 = _mm512_mul_ps(a1, b3); \
// 	t4 = _mm512_mul_ps(a1, b4); \
// 	t5 = _mm512_mul_ps(a1, b5); \
// 	t6 = _mm512_mul_ps(a1, b6); \
// 	t7 = _mm512_mul_ps(a1, b7); \
// 	t8 = _mm512_mul_ps(a1, b8); \
// 	t9 = _mm512_mul_ps(a1, b9); \
// 	ta = _mm512_mul_ps(a1, ba); \
// 	tb = _mm512_mul_ps(a1, bb); \
// 	tc = _mm512_mul_ps(a1, bc); \
// 	td = _mm512_mul_ps(a1, bd); \
// 	te = _mm512_mul_ps(a1, be); \
// 	tf = _mm512_mul_ps(a1, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c1 = _mm512_add_ps(c1, c); \
// 	\
// 	t0 = _mm512_mul_ps(a2, b0); \
// 	t1 = _mm512_mul_ps(a2, b1); \
// 	t2 = _mm512_mul_ps(a2, b2); \
// 	t3 = _mm512_mul_ps(a2, b3); \
// 	t4 = _mm512_mul_ps(a2, b4); \
// 	t5 = _mm512_mul_ps(a2, b5); \
// 	t6 = _mm512_mul_ps(a2, b6); \
// 	t7 = _mm512_mul_ps(a2, b7); \
// 	t8 = _mm512_mul_ps(a2, b8); \
// 	t9 = _mm512_mul_ps(a2, b9); \
// 	ta = _mm512_mul_ps(a2, ba); \
// 	tb = _mm512_mul_ps(a2, bb); \
// 	tc = _mm512_mul_ps(a2, bc); \
// 	td = _mm512_mul_ps(a2, bd); \
// 	te = _mm512_mul_ps(a2, be); \
// 	tf = _mm512_mul_ps(a2, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c2 = _mm512_add_ps(c2, c); \
// 	\
// 	t0 = _mm512_mul_ps(a3, b0); \
// 	t1 = _mm512_mul_ps(a3, b1); \
// 	t2 = _mm512_mul_ps(a3, b2); \
// 	t3 = _mm512_mul_ps(a3, b3); \
// 	t4 = _mm512_mul_ps(a3, b4); \
// 	t5 = _mm512_mul_ps(a3, b5); \
// 	t6 = _mm512_mul_ps(a3, b6); \
// 	t7 = _mm512_mul_ps(a3, b7); \
// 	t8 = _mm512_mul_ps(a3, b8); \
// 	t9 = _mm512_mul_ps(a3, b9); \
// 	ta = _mm512_mul_ps(a3, ba); \
// 	tb = _mm512_mul_ps(a3, bb); \
// 	tc = _mm512_mul_ps(a3, bc); \
// 	td = _mm512_mul_ps(a3, bd); \
// 	te = _mm512_mul_ps(a3, be); \
// 	tf = _mm512_mul_ps(a3, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c3 = _mm512_add_ps(c3, c); \
// 	\
// 	t0 = _mm512_mul_ps(a4, b0); \
// 	t1 = _mm512_mul_ps(a4, b1); \
// 	t2 = _mm512_mul_ps(a4, b2); \
// 	t3 = _mm512_mul_ps(a4, b3); \
// 	t4 = _mm512_mul_ps(a4, b4); \
// 	t5 = _mm512_mul_ps(a4, b5); \
// 	t6 = _mm512_mul_ps(a4, b6); \
// 	t7 = _mm512_mul_ps(a4, b7); \
// 	t8 = _mm512_mul_ps(a4, b8); \
// 	t9 = _mm512_mul_ps(a4, b9); \
// 	ta = _mm512_mul_ps(a4, ba); \
// 	tb = _mm512_mul_ps(a4, bb); \
// 	tc = _mm512_mul_ps(a4, bc); \
// 	td = _mm512_mul_ps(a4, bd); \
// 	te = _mm512_mul_ps(a4, be); \
// 	tf = _mm512_mul_ps(a4, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c4 = _mm512_add_ps(c4, c); \
// 	\
// 	t0 = _mm512_mul_ps(a5, b0); \
// 	t1 = _mm512_mul_ps(a5, b1); \
// 	t2 = _mm512_mul_ps(a5, b2); \
// 	t3 = _mm512_mul_ps(a5, b3); \
// 	t4 = _mm512_mul_ps(a5, b4); \
// 	t5 = _mm512_mul_ps(a5, b5); \
// 	t6 = _mm512_mul_ps(a5, b6); \
// 	t7 = _mm512_mul_ps(a5, b7); \
// 	t8 = _mm512_mul_ps(a5, b8); \
// 	t9 = _mm512_mul_ps(a5, b9); \
// 	ta = _mm512_mul_ps(a5, ba); \
// 	tb = _mm512_mul_ps(a5, bb); \
// 	tc = _mm512_mul_ps(a5, bc); \
// 	td = _mm512_mul_ps(a5, bd); \
// 	te = _mm512_mul_ps(a5, be); \
// 	tf = _mm512_mul_ps(a5, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c5 = _mm512_add_ps(c5, c); \
// 	\
// 	t0 = _mm512_mul_ps(a6, b0); \
// 	t1 = _mm512_mul_ps(a6, b1); \
// 	t2 = _mm512_mul_ps(a6, b2); \
// 	t3 = _mm512_mul_ps(a6, b3); \
// 	t4 = _mm512_mul_ps(a6, b4); \
// 	t5 = _mm512_mul_ps(a6, b5); \
// 	t6 = _mm512_mul_ps(a6, b6); \
// 	t7 = _mm512_mul_ps(a6, b7); \
// 	t8 = _mm512_mul_ps(a6, b8); \
// 	t9 = _mm512_mul_ps(a6, b9); \
// 	ta = _mm512_mul_ps(a6, ba); \
// 	tb = _mm512_mul_ps(a6, bb); \
// 	tc = _mm512_mul_ps(a6, bc); \
// 	td = _mm512_mul_ps(a6, bd); \
// 	te = _mm512_mul_ps(a6, be); \
// 	tf = _mm512_mul_ps(a6, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c6 = _mm512_add_ps(c6, c); \
// 	\
// 	t0 = _mm512_mul_ps(a7, b0); \
// 	t1 = _mm512_mul_ps(a7, b1); \
// 	t2 = _mm512_mul_ps(a7, b2); \
// 	t3 = _mm512_mul_ps(a7, b3); \
// 	t4 = _mm512_mul_ps(a7, b4); \
// 	t5 = _mm512_mul_ps(a7, b5); \
// 	t6 = _mm512_mul_ps(a7, b6); \
// 	t7 = _mm512_mul_ps(a7, b7); \
// 	t8 = _mm512_mul_ps(a7, b8); \
// 	t9 = _mm512_mul_ps(a7, b9); \
// 	ta = _mm512_mul_ps(a7, ba); \
// 	tb = _mm512_mul_ps(a7, bb); \
// 	tc = _mm512_mul_ps(a7, bc); \
// 	td = _mm512_mul_ps(a7, bd); \
// 	te = _mm512_mul_ps(a7, be); \
// 	tf = _mm512_mul_ps(a7, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c7 = _mm512_add_ps(c7, c); \
// 	\
// 	t0 = _mm512_mul_ps(a8, b0); \
// 	t1 = _mm512_mul_ps(a8, b1); \
// 	t2 = _mm512_mul_ps(a8, b2); \
// 	t3 = _mm512_mul_ps(a8, b3); \
// 	t4 = _mm512_mul_ps(a8, b4); \
// 	t5 = _mm512_mul_ps(a8, b5); \
// 	t6 = _mm512_mul_ps(a8, b6); \
// 	t7 = _mm512_mul_ps(a8, b7); \
// 	t8 = _mm512_mul_ps(a8, b8); \
// 	t9 = _mm512_mul_ps(a8, b9); \
// 	ta = _mm512_mul_ps(a8, ba); \
// 	tb = _mm512_mul_ps(a8, bb); \
// 	tc = _mm512_mul_ps(a8, bc); \
// 	td = _mm512_mul_ps(a8, bd); \
// 	te = _mm512_mul_ps(a8, be); \
// 	tf = _mm512_mul_ps(a8, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c8 = _mm512_add_ps(c8, c); \
// 	\
// 	t0 = _mm512_mul_ps(a9, b0); \
// 	t1 = _mm512_mul_ps(a9, b1); \
// 	t2 = _mm512_mul_ps(a9, b2); \
// 	t3 = _mm512_mul_ps(a9, b3); \
// 	t4 = _mm512_mul_ps(a9, b4); \
// 	t5 = _mm512_mul_ps(a9, b5); \
// 	t6 = _mm512_mul_ps(a9, b6); \
// 	t7 = _mm512_mul_ps(a9, b7); \
// 	t8 = _mm512_mul_ps(a9, b8); \
// 	t9 = _mm512_mul_ps(a9, b9); \
// 	ta = _mm512_mul_ps(a9, ba); \
// 	tb = _mm512_mul_ps(a9, bb); \
// 	tc = _mm512_mul_ps(a9, bc); \
// 	td = _mm512_mul_ps(a9, bd); \
// 	te = _mm512_mul_ps(a9, be); \
// 	tf = _mm512_mul_ps(a9, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	c9 = _mm512_add_ps(c9, c); \
// 	\
// 	t0 = _mm512_mul_ps(aa, b0); \
// 	t1 = _mm512_mul_ps(aa, b1); \
// 	t2 = _mm512_mul_ps(aa, b2); \
// 	t3 = _mm512_mul_ps(aa, b3); \
// 	t4 = _mm512_mul_ps(aa, b4); \
// 	t5 = _mm512_mul_ps(aa, b5); \
// 	t6 = _mm512_mul_ps(aa, b6); \
// 	t7 = _mm512_mul_ps(aa, b7); \
// 	t8 = _mm512_mul_ps(aa, b8); \
// 	t9 = _mm512_mul_ps(aa, b9); \
// 	ta = _mm512_mul_ps(aa, ba); \
// 	tb = _mm512_mul_ps(aa, bb); \
// 	tc = _mm512_mul_ps(aa, bc); \
// 	td = _mm512_mul_ps(aa, bd); \
// 	te = _mm512_mul_ps(aa, be); \
// 	tf = _mm512_mul_ps(aa, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	ca = _mm512_add_ps(ca, c); \
// 	\
// 	t0 = _mm512_mul_ps(ab, b0); \
// 	t1 = _mm512_mul_ps(ab, b1); \
// 	t2 = _mm512_mul_ps(ab, b2); \
// 	t3 = _mm512_mul_ps(ab, b3); \
// 	t4 = _mm512_mul_ps(ab, b4); \
// 	t5 = _mm512_mul_ps(ab, b5); \
// 	t6 = _mm512_mul_ps(ab, b6); \
// 	t7 = _mm512_mul_ps(ab, b7); \
// 	t8 = _mm512_mul_ps(ab, b8); \
// 	t9 = _mm512_mul_ps(ab, b9); \
// 	ta = _mm512_mul_ps(ab, ba); \
// 	tb = _mm512_mul_ps(ab, bb); \
// 	tc = _mm512_mul_ps(ab, bc); \
// 	td = _mm512_mul_ps(ab, bd); \
// 	te = _mm512_mul_ps(ab, be); \
// 	tf = _mm512_mul_ps(ab, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	cb = _mm512_add_ps(cb, c); \
// 	\
// 	t0 = _mm512_mul_ps(ac, b0); \
// 	t1 = _mm512_mul_ps(ac, b1); \
// 	t2 = _mm512_mul_ps(ac, b2); \
// 	t3 = _mm512_mul_ps(ac, b3); \
// 	t4 = _mm512_mul_ps(ac, b4); \
// 	t5 = _mm512_mul_ps(ac, b5); \
// 	t6 = _mm512_mul_ps(ac, b6); \
// 	t7 = _mm512_mul_ps(ac, b7); \
// 	t8 = _mm512_mul_ps(ac, b8); \
// 	t9 = _mm512_mul_ps(ac, b9); \
// 	ta = _mm512_mul_ps(ac, ba); \
// 	tb = _mm512_mul_ps(ac, bb); \
// 	tc = _mm512_mul_ps(ac, bc); \
// 	td = _mm512_mul_ps(ac, bd); \
// 	te = _mm512_mul_ps(ac, be); \
// 	tf = _mm512_mul_ps(ac, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	cc = _mm512_add_ps(cc, c); \
// 	\
// 	t0 = _mm512_mul_ps(ad, b0); \
// 	t1 = _mm512_mul_ps(ad, b1); \
// 	t2 = _mm512_mul_ps(ad, b2); \
// 	t3 = _mm512_mul_ps(ad, b3); \
// 	t4 = _mm512_mul_ps(ad, b4); \
// 	t5 = _mm512_mul_ps(ad, b5); \
// 	t6 = _mm512_mul_ps(ad, b6); \
// 	t7 = _mm512_mul_ps(ad, b7); \
// 	t8 = _mm512_mul_ps(ad, b8); \
// 	t9 = _mm512_mul_ps(ad, b9); \
// 	ta = _mm512_mul_ps(ad, ba); \
// 	tb = _mm512_mul_ps(ad, bb); \
// 	tc = _mm512_mul_ps(ad, bc); \
// 	td = _mm512_mul_ps(ad, bd); \
// 	te = _mm512_mul_ps(ad, be); \
// 	tf = _mm512_mul_ps(ad, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	cd = _mm512_add_ps(cd, c); \
// 	\
// 	t0 = _mm512_mul_ps(ae, b0); \
// 	t1 = _mm512_mul_ps(ae, b1); \
// 	t2 = _mm512_mul_ps(ae, b2); \
// 	t3 = _mm512_mul_ps(ae, b3); \
// 	t4 = _mm512_mul_ps(ae, b4); \
// 	t5 = _mm512_mul_ps(ae, b5); \
// 	t6 = _mm512_mul_ps(ae, b6); \
// 	t7 = _mm512_mul_ps(ae, b7); \
// 	t8 = _mm512_mul_ps(ae, b8); \
// 	t9 = _mm512_mul_ps(ae, b9); \
// 	ta = _mm512_mul_ps(ae, ba); \
// 	tb = _mm512_mul_ps(ae, bb); \
// 	tc = _mm512_mul_ps(ae, bc); \
// 	td = _mm512_mul_ps(ae, bd); \
// 	te = _mm512_mul_ps(ae, be); \
// 	tf = _mm512_mul_ps(ae, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	ce = _mm512_add_ps(ce, c); \
// 	\
// 	t0 = _mm512_mul_ps(af, b0); \
// 	t1 = _mm512_mul_ps(af, b1); \
// 	t2 = _mm512_mul_ps(af, b2); \
// 	t3 = _mm512_mul_ps(af, b3); \
// 	t4 = _mm512_mul_ps(af, b4); \
// 	t5 = _mm512_mul_ps(af, b5); \
// 	t6 = _mm512_mul_ps(af, b6); \
// 	t7 = _mm512_mul_ps(af, b7); \
// 	t8 = _mm512_mul_ps(af, b8); \
// 	t9 = _mm512_mul_ps(af, b9); \
// 	ta = _mm512_mul_ps(af, ba); \
// 	tb = _mm512_mul_ps(af, bb); \
// 	tc = _mm512_mul_ps(af, bc); \
// 	td = _mm512_mul_ps(af, bd); \
// 	te = _mm512_mul_ps(af, be); \
// 	tf = _mm512_mul_ps(af, bf); \
// 	\
// 	c = _mm512_set_ps( \
// 		_mm512_reduce_add_ps(tf), \
// 		_mm512_reduce_add_ps(te), \
// 		_mm512_reduce_add_ps(td), \
// 		_mm512_reduce_add_ps(tc), \
// 		_mm512_reduce_add_ps(tb), \
// 		_mm512_reduce_add_ps(ta), \
// 		_mm512_reduce_add_ps(t9), \
// 		_mm512_reduce_add_ps(t8), \
// 		_mm512_reduce_add_ps(t7), \
// 		_mm512_reduce_add_ps(t6), \
// 		_mm512_reduce_add_ps(t5), \
// 		_mm512_reduce_add_ps(t4), \
// 		_mm512_reduce_add_ps(t3), \
// 		_mm512_reduce_add_ps(t2), \
// 		_mm512_reduce_add_ps(t1), \
// 		_mm512_reduce_add_ps(t0)); \
// 	cf = _mm512_add_ps(cf, c); \

/* COMPLEX MATRIX-MATRIX MULTIPLY OPS*/

#define _MM_MREDUCE_C_PS(c0, c1, acc) \
{ \
	__m128 lo = _mm_movelh_ps(c0, c1); \
	__m128 hi = _mm_movehl_ps(c1, c0); \
	acc = _mm_add_ps(acc, _mm_add_ps(lo, hi)); \
}

#define _MM256_MREDUCE_C_PS(c0, c1, c2, c3, acc) \
{ \
	__m128 c0_lo = _mm256_castps256_ps128(c0); \
	__m128 c0_hi = _mm256_extractf128_ps(c0, 1); \
	__m128 c1_lo = _mm256_castps256_ps128(c1); \
	__m128 c1_hi = _mm256_extractf128_ps(c1, 1); \
	__m128 c2_lo = _mm256_castps256_ps128(c2); \
	__m128 c2_hi = _mm256_extractf128_ps(c2, 1); \
	__m128 c3_lo = _mm256_castps256_ps128(c3); \
	__m128 c3_hi = _mm256_extractf128_ps(c3, 1); \
	\
	__m256 stack01_lh = _mm256_set_m128(c1_lo, c0_lo); \
	__m256 stack01_hh = _mm256_set_m128(c1_hi, c0_hi); \
	__m256 stack23_lh = _mm256_set_m128(c3_lo, c2_lo); \
	__m256 stack23_hh = _mm256_set_m128(c3_hi, c2_hi); \
	\
	__m256 partial01 = _mm256_add_ps(stack01_lh, stack01_hh); \
	__m256 partial23 = _mm256_add_ps(stack23_lh, stack23_hh); \
	\
	__m128 p01_lo = _mm256_castps256_ps128(partial01); \
	__m128 p01_hi = _mm256_extractf128_ps(partial01, 1); \
	__m128 p23_lo = _mm256_castps256_ps128(partial23);  \
	__m128 p23_hi = _mm256_extractf128_ps(partial23, 1); \
	\
	__m128 final_0 = _mm_movelh_ps(p01_lo, p23_lo); \
	__m128 final_1 = _mm_movehl_ps(p23_lo, p01_lo); \
	__m128 res_02 = _mm_add_ps(final_0, final_1); \
	\
	__m128 final_2 = _mm_movelh_ps(p01_hi, p23_hi); \
	__m128 final_3 = _mm_movehl_ps(p23_hi, p01_hi); \
	__m128 res_13 = _mm_add_ps(final_2, final_3);  \
	\
	__m128 final_01 = _mm_movelh_ps(res_02, res_13); \
	__m128 final_23 = _mm_movehl_ps(res_13, res_02); \
	__m256 result = _mm256_setr_m128(final_01, final_23); \
	acc = _mm256_add_ps(acc, result); \
}

#define _MM256_MREDUCE_C_PD(c0, c1, acc) \
{ \
	__m128d c0_lo = _mm256_castpd256_pd128(c0); \
	__m128d c0_hi = _mm256_extractf128_pd(c0, 1); \
	__m128d c1_lo = _mm256_castpd256_pd128(c1); \
	__m128d c1_hi = _mm256_extractf128_pd(c1, 1); \
	\
	__m256d stack_lh = _mm256_set_m128d(c1_lo, c0_lo); \
	__m256d stack_hh = _mm256_set_m128d(c1_hi, c0_hi); \
	\
	__m256d partial = _mm256_add_pd(stack_lh, stack_hh); \
	\
	__m128d final_lo = _mm256_castpd256_pd128(partial); \
	__m128d final_hi = _mm256_extractf128_pd(partial, 1); \
	\
	__m256d result = _mm256_set_m128d(final_hi, final_lo); \
	\
	acc = _mm256_add_pd(acc, result); \
}

#define _MM512_MREDUCE_C_PS(c0, c1, c2, c3, c4, c5, c6, c7, acc) \
{ \
	__m256 c0_lo = _mm512_castps512_ps256(c0); \
	__m256 c0_hi = _mm512_extractf32x8_ps(c0, 1); \
	__m256 sum0 = _mm256_add_ps(c0_lo, c0_hi); \
	\
	__m256 c1_lo = _mm512_castps512_ps256(c1); \
	__m256 c1_hi = _mm512_extractf32x8_ps(c1, 1); \
	__m256 sum1 = _mm256_add_ps(c1_lo, c1_hi); \
	\
	__m256 c2_lo = _mm512_castps512_ps256(c2); \
	__m256 c2_hi = _mm512_extractf32x8_ps(c2, 1); \
	__m256 sum2 = _mm256_add_ps(c2_lo, c2_hi); \
	\
	__m256 c3_lo = _mm512_castps512_ps256(c3); \
	__m256 c3_hi = _mm512_extractf32x8_ps(c3, 1); \
	__m256 sum3 = _mm256_add_ps(c3_lo, c3_hi); \
	\
	__m256 c4_lo = _mm512_castps512_ps256(c4); \
	__m256 c4_hi = _mm512_extractf32x8_ps(c4, 1); \
	__m256 sum4 = _mm256_add_ps(c4_lo, c4_hi); \
	\
	__m256 c5_lo = _mm512_castps512_ps256(c5); \
	__m256 c5_hi = _mm512_extractf32x8_ps(c5, 1); \
	__m256 sum5 = _mm256_add_ps(c5_lo, c5_hi); \
	\
	__m256 c6_lo = _mm512_castps512_ps256(c6); \
	__m256 c6_hi = _mm512_extractf32x8_ps(c6, 1); \
	__m256 sum6 = _mm256_add_ps(c6_lo, c6_hi); \
	\
	__m256 c7_lo = _mm512_castps512_ps256(c7); \
	__m256 c7_hi = _mm512_extractf32x8_ps(c7, 1); \
	__m256 sum7 = _mm256_add_ps(c7_lo, c7_hi); \
	\
	__m128 s0_lo = _mm256_castps256_ps128(sum0); \
	__m128 s0_hi = _mm256_extractf128_ps(sum0, 1); \
	__m128 red0 = _mm_add_ps(s0_lo, s0_hi); \
	\
	__m128 s1_lo = _mm256_castps256_ps128(sum1); \
	__m128 s1_hi = _mm256_extractf128_ps(sum1, 1); \
	__m128 red1 = _mm_add_ps(s1_lo, s1_hi); \
	\
	__m128 s2_lo = _mm256_castps256_ps128(sum2); \
	__m128 s2_hi = _mm256_extractf128_ps(sum2, 1); \
	__m128 red2 = _mm_add_ps(s2_lo, s2_hi); \
	\
	__m128 s3_lo = _mm256_castps256_ps128(sum3); \
	__m128 s3_hi = _mm256_extractf128_ps(sum3, 1); \
	__m128 red3 = _mm_add_ps(s3_lo, s3_hi); \
	\
	__m128 s4_lo = _mm256_castps256_ps128(sum4); \
	__m128 s4_hi = _mm256_extractf128_ps(sum4, 1); \
	__m128 red4 = _mm_add_ps(s4_lo, s4_hi); \
	\
	__m128 s5_lo = _mm256_castps256_ps128(sum5); \
	__m128 s5_hi = _mm256_extractf128_ps(sum5, 1); \
	__m128 red5 = _mm_add_ps(s5_lo, s5_hi); \
	\
	__m128 s6_lo = _mm256_castps256_ps128(sum6); \
	__m128 s6_hi = _mm256_extractf128_ps(sum6, 1); \
	__m128 red6 = _mm_add_ps(s6_lo, s6_hi); \
	\
	__m128 s7_lo = _mm256_castps256_ps128(sum7); \
	__m128 s7_hi = _mm256_extractf128_ps(sum7, 1); \
	__m128 red7 = _mm_add_ps(s7_lo, s7_hi); \
	\
	__m128 final0 = _mm_add_ps(red0, _mm_movehl_ps(red0, red0)); \
	__m128 final1 = _mm_add_ps(red1, _mm_movehl_ps(red1, red1)); \
	__m128 final2 = _mm_add_ps(red2, _mm_movehl_ps(red2, red2)); \
	__m128 final3 = _mm_add_ps(red3, _mm_movehl_ps(red3, red3)); \
	__m128 final4 = _mm_add_ps(red4, _mm_movehl_ps(red4, red4)); \
	__m128 final5 = _mm_add_ps(red5, _mm_movehl_ps(red5, red5)); \
	__m128 final6 = _mm_add_ps(red6, _mm_movehl_ps(red6, red6)); \
	__m128 final7 = _mm_add_ps(red7, _mm_movehl_ps(red7, red7)); \
	\
	__m128 res01 = _mm_movelh_ps(final0, final1); \
	__m128 res23 = _mm_movelh_ps(final2, final3); \
	__m128 res45 = _mm_movelh_ps(final4, final5); \
	__m128 res67 = _mm_movelh_ps(final6, final7); \
	\
	__m256 res0123 = _mm256_set_m128(res23, res01); \
	__m256 res4567 = _mm256_set_m128(res67, res45); \
	\
	__m512 result = _mm512_insertf32x8(_mm512_castps256_ps512(res0123), res4567, 1); \
	\
	acc = _mm512_add_ps(acc, result); \
}

#define _MM512_MREDUCE_C_PD(c0, c1, c2, c3, acc) \
{ \
	__m256d c0_lo = _mm512_castpd512_pd256(c0); \
	__m256d c0_hi = _mm512_extractf64x4_pd(c0, 1); \
	__m256d c1_lo = _mm512_castpd512_pd256(c1); \
	__m256d c1_hi = _mm512_extractf64x4_pd(c1, 1); \
	__m256d c2_lo = _mm512_castpd512_pd256(c2); \
	__m256d c2_hi = _mm512_extractf64x4_pd(c2, 1); \
	__m256d c3_lo = _mm512_castpd512_pd256(c3); \
	__m256d c3_hi = _mm512_extractf64x4_pd(c3, 1); \
	\
	__m512d stack01_lh = _mm512_insertf64x4(_mm512_castpd256_pd512(c0_lo), c1_lo, 1); \
	__m512d stack01_hh = _mm512_insertf64x4(_mm512_castpd256_pd512(c0_hi), c1_hi, 1); \
	__m512d stack23_lh = _mm512_insertf64x4(_mm512_castpd256_pd512(c2_lo), c3_lo, 1); \
	__m512d stack23_hh = _mm512_insertf64x4(_mm512_castpd256_pd512(c2_hi), c3_hi, 1); \
	\
	__m512d partial01 = _mm512_add_pd(stack01_lh, stack01_hh); \
	__m512d partial23 = _mm512_add_pd(stack23_lh, stack23_hh); \
	\
	__m512d partial01_rearr = _mm512_permutex_pd(partial01, 0xD8); \
	__m512d partial23_rearr = _mm512_permutex_pd(partial23, 0xD8); \
	\
	__m512d lo = _mm512_unpacklo_pd(partial01_rearr, partial23_rearr); \
	__m512d hi = _mm512_unpackhi_pd(partial01_rearr, partial23_rearr); \
	\
	__m512d result = _mm512_add_pd(lo, hi); \
	__m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0); \
	result = _mm512_permutexvar_pd(idx, result); \
	acc = _mm512_add_pd(acc, result); \
}

#define _MM_MMUL2_C_PS(a0, a1, b0, b1, c0, c1) \
{ \
	__m128 t0, t1; \
	_MM_MUL_C_PS(a0, b0, t0); \
	_MM_MUL_C_PS(a0, b1, t1); \
	_MM_MREDUCE_C_PS(t0, t1, c0); \
	\
	_MM_MUL_C_PS(a1, b0, t0); \
	_MM_MUL_C_PS(a1, b1, t1); \
	_MM_MREDUCE_C_PS(t0, t1, c1); \
}

#define _MM_MMUL_C_PD(a0, b0, c0) \
{ \
	__m128d acc; \
	_MM_MUL_C_PD(a0, b0, acc); \
	c0 = _mm_add_pd(c0, acc); \
}

#define _MM256_MMUL4_C_PS(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
{ \
	__m256 t0, t1, t2, t3; \
	\
	_MM256_MUL_C_PS(a0, b0, t0); \
	_MM256_MUL_C_PS(a0, b1, t1); \
	_MM256_MUL_C_PS(a0, b2, t2); \
	_MM256_MUL_C_PS(a0, b3, t3); \
	_MM256_MREDUCE_C_PS(t0, t1, t2, t3, c0) \
	\
	_MM256_MUL_C_PS(a1, b0, t0); \
	_MM256_MUL_C_PS(a1, b1, t1); \
	_MM256_MUL_C_PS(a1, b2, t2); \
	_MM256_MUL_C_PS(a1, b3, t3); \
	_MM256_MREDUCE_C_PS(t0, t1, t2, t3, c1) \
	\
	_MM256_MUL_C_PS(a2, b0, t0); \
	_MM256_MUL_C_PS(a2, b1, t1); \
	_MM256_MUL_C_PS(a2, b2, t2); \
	_MM256_MUL_C_PS(a2, b3, t3); \
	_MM256_MREDUCE_C_PS(t0, t1, t2, t3, c2) \
	\
	_MM256_MUL_C_PS(a3, b0, t0); \
	_MM256_MUL_C_PS(a3, b1, t1); \
	_MM256_MUL_C_PS(a3, b2, t2); \
	_MM256_MUL_C_PS(a3, b3, t3); \
	_MM256_MREDUCE_C_PS(t0, t1, t2, t3, c3) \
}

#define _MM256_MMUL2_C_PD(a0, a1, b0, b1, c0, c1) \
{ \
	__m256d t0, t1; \
	\
	_MM256_MUL_C_PD(a0, b0, t0); \
	_MM256_MUL_C_PD(a0, b1, t1); \
	_MM256_MREDUCE_C_PD(t0, t1, c0); \
	\
	_MM256_MUL_C_PD(a1, b0, t0); \
	_MM256_MUL_C_PD(a1, b1, t1); \
	_MM256_MREDUCE_C_PD(t0, t1, c1); \
}

#define _MM512_MMUL8_C_PS(a0, a1, a2, a3, a4, a5, a6, a7, \
						  b0, b1, b2, b3, b4, b5, b6, b7, \
						  c0, c1, c2, c3, c4, c5, c6, c7) \
{\
	__m512 t0, t1, t2, t3, t4, t5, t6, t7; \
	\
	_MM512_MUL_C_PS(a0, b0, t0); \
	_MM512_MUL_C_PS(a0, b1, t1); \
	_MM512_MUL_C_PS(a0, b2, t2); \
	_MM512_MUL_C_PS(a0, b3, t3); \
	_MM512_MUL_C_PS(a0, b4, t4); \
	_MM512_MUL_C_PS(a0, b5, t5); \
	_MM512_MUL_C_PS(a0, b6, t6); \
	_MM512_MUL_C_PS(a0, b7, t7); \
	_MM512_MREDUCE_C_PS(t0, t1, t2, t3, t4, t5, t6, t7, c0); \
	\
	_MM512_MUL_C_PS(a1, b0, t0); \
	_MM512_MUL_C_PS(a1, b1, t1); \
	_MM512_MUL_C_PS(a1, b2, t2); \
	_MM512_MUL_C_PS(a1, b3, t3); \
	_MM512_MUL_C_PS(a1, b4, t4); \
	_MM512_MUL_C_PS(a1, b5, t5); \
	_MM512_MUL_C_PS(a1, b6, t6); \
	_MM512_MUL_C_PS(a1, b7, t7); \
	_MM512_MREDUCE_C_PS(t0, t1, t2, t3, t4, t5, t6, t7, c1); \
	\
	_MM512_MUL_C_PS(a2, b0, t0); \
	_MM512_MUL_C_PS(a2, b1, t1); \
	_MM512_MUL_C_PS(a2, b2, t2); \
	_MM512_MUL_C_PS(a2, b3, t3); \
	_MM512_MUL_C_PS(a2, b4, t4); \
	_MM512_MUL_C_PS(a2, b5, t5); \
	_MM512_MUL_C_PS(a2, b6, t6); \
	_MM512_MUL_C_PS(a2, b7, t7); \
	_MM512_MREDUCE_C_PS(t0, t1, t2, t3, t4, t5, t6, t7, c2); \
	\
	_MM512_MUL_C_PS(a3, b0, t0); \
	_MM512_MUL_C_PS(a3, b1, t1); \
	_MM512_MUL_C_PS(a3, b2, t2); \
	_MM512_MUL_C_PS(a3, b3, t3); \
	_MM512_MUL_C_PS(a3, b4, t4); \
	_MM512_MUL_C_PS(a3, b5, t5); \
	_MM512_MUL_C_PS(a3, b6, t6); \
	_MM512_MUL_C_PS(a3, b7, t7); \
	_MM512_MREDUCE_C_PS(t0, t1, t2, t3, t4, t5, t6, t7, c3); \
	\
	_MM512_MUL_C_PS(a4, b0, t0); \
	_MM512_MUL_C_PS(a4, b1, t1); \
	_MM512_MUL_C_PS(a4, b2, t2); \
	_MM512_MUL_C_PS(a4, b3, t3); \
	_MM512_MUL_C_PS(a4, b4, t4); \
	_MM512_MUL_C_PS(a4, b5, t5); \
	_MM512_MUL_C_PS(a4, b6, t6); \
	_MM512_MUL_C_PS(a4, b7, t7); \
	_MM512_MREDUCE_C_PS(t0, t1, t2, t3, t4, t5, t6, t7, c4); \
	\
	_MM512_MUL_C_PS(a5, b0, t0); \
	_MM512_MUL_C_PS(a5, b1, t1); \
	_MM512_MUL_C_PS(a5, b2, t2); \
	_MM512_MUL_C_PS(a5, b3, t3); \
	_MM512_MUL_C_PS(a5, b4, t4); \
	_MM512_MUL_C_PS(a5, b5, t5); \
	_MM512_MUL_C_PS(a5, b6, t6); \
	_MM512_MUL_C_PS(a5, b7, t7); \
	_MM512_MREDUCE_C_PS(t0, t1, t2, t3, t4, t5, t6, t7, c5); \
	\
	_MM512_MUL_C_PS(a6, b0, t0); \
	_MM512_MUL_C_PS(a6, b1, t1); \
	_MM512_MUL_C_PS(a6, b2, t2); \
	_MM512_MUL_C_PS(a6, b3, t3); \
	_MM512_MUL_C_PS(a6, b4, t4); \
	_MM512_MUL_C_PS(a6, b5, t5); \
	_MM512_MUL_C_PS(a6, b6, t6); \
	_MM512_MUL_C_PS(a6, b7, t7); \
	_MM512_MREDUCE_C_PS(t0, t1, t2, t3, t4, t5, t6, t7, c6); \
	\
	_MM512_MUL_C_PS(a7, b0, t0); \
	_MM512_MUL_C_PS(a7, b1, t1); \
	_MM512_MUL_C_PS(a7, b2, t2); \
	_MM512_MUL_C_PS(a7, b3, t3); \
	_MM512_MUL_C_PS(a7, b4, t4); \
	_MM512_MUL_C_PS(a7, b5, t5); \
	_MM512_MUL_C_PS(a7, b6, t6); \
	_MM512_MUL_C_PS(a7, b7, t7); \
	_MM512_MREDUCE_C_PS(t0, t1, t2, t3, t4, t5, t6, t7, c7); \
}

#define _MM512_MMUL4_C_PD(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
{ \
	__m512d t0, t1, t2, t3; \
	\
	_MM512_MUL_C_PD(a0, b0, t0); \
	_MM512_MUL_C_PD(a0, b1, t1); \
	_MM512_MUL_C_PD(a0, b2, t2); \
	_MM512_MUL_C_PD(a0, b3, t3); \
	_MM512_MREDUCE_C_PD(t0, t1, t2, t3, c0); \
	\
	_MM512_MUL_C_PD(a1, b0, t0); \
	_MM512_MUL_C_PD(a1, b1, t1); \
	_MM512_MUL_C_PD(a1, b2, t2); \
	_MM512_MUL_C_PD(a1, b3, t3); \
	_MM512_MREDUCE_C_PD(t0, t1, t2, t3, c1); \
	\
	_MM512_MUL_C_PD(a2, b0, t0); \
	_MM512_MUL_C_PD(a2, b1, t1); \
	_MM512_MUL_C_PD(a2, b2, t2); \
	_MM512_MUL_C_PD(a2, b3, t3); \
	_MM512_MREDUCE_C_PD(t0, t1, t2, t3, c2); \
	\
	_MM512_MUL_C_PD(a3, b0, t0); \
	_MM512_MUL_C_PD(a3, b1, t1); \
	_MM512_MUL_C_PD(a3, b2, t2); \
	_MM512_MUL_C_PD(a3, b3, t3); \
	_MM512_MREDUCE_C_PD(t0, t1, t2, t3, c3); \
}

#endif //__MACROS_H__