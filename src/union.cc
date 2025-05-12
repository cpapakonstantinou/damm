/**
 * \file union.cc
 * \brief union utilities implementations
 * \author cpapakonstantinou
 * \date 2025
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

#include <union.h>

namespace damm
{
/*SSE*/
template <>
void 
_union_block_sse<float, std::plus<>>(float* A, float* B, float* C)
{
		alignas(16)__m128 a0, a1, a2, a3;
		alignas(16)__m128 b0, b1, b2, b3;
		__m128 c0, c1, c2, c3;

		if (reinterpret_cast<uintptr_t>(A) % 16 == 0)
		{
			a0 = _mm_load_ps(A);
			a1 = _mm_load_ps(A + 4);
			a2 = _mm_load_ps(A + 8);
			a3 = _mm_load_ps(A + 12);
		}
		else
		{
			a0 = _mm_loadu_ps(A);
			a1 = _mm_loadu_ps(A + 4);
			a2 = _mm_loadu_ps(A + 8);
			a3 = _mm_loadu_ps(A + 12);
		}

		if (reinterpret_cast<uintptr_t>(B) % 16 == 0)
		{
			b0 = _mm_load_ps(B);
			b1 = _mm_load_ps(B + 4);
			b2 = _mm_load_ps(B + 8);
			b3 = _mm_load_ps(B + 12);
		}
		else
		{
			b0 = _mm_loadu_ps(B);
			b1 = _mm_loadu_ps(B + 4);
			b2 = _mm_loadu_ps(B + 8);
			b3 = _mm_loadu_ps(B + 12);
		}

		c0 = _mm_add_ps(a0, b0);
		c1 = _mm_add_ps(a1, b1);
		c2 = _mm_add_ps(a2, b2);
		c3 = _mm_add_ps(a3, b3);

		if (reinterpret_cast<uintptr_t>(C) % 16 == 0)
		{
			_mm_store_ps(C, c0);
			_mm_store_ps(C + 4, c1);
			_mm_store_ps(C + 8, c2);
			_mm_store_ps(C + 12,c3);
		}
		else
		{
			_mm_storeu_ps(C, c0);
			_mm_storeu_ps(C + 4, c1);
			_mm_storeu_ps(C + 8, c2);
			_mm_storeu_ps(C + 12,c3);
		}
	}

	template <>
	void 
	_union_block_sse<float, std::minus<>>(float* A, float* B, float* C)
	{
		alignas(16)__m128 a0, a1, a2, a3;
		alignas(16)__m128 b0, b1, b2, b3;
		__m128 c0, c1, c2, c3;

		if (reinterpret_cast<uintptr_t>(A) % 16 == 0)
		{
			a0 = _mm_load_ps(A);
			a1 = _mm_load_ps(A + 4);
			a2 = _mm_load_ps(A + 8);
			a3 = _mm_load_ps(A + 12);
		}
		else
		{
			a0 = _mm_loadu_ps(A);
			a1 = _mm_loadu_ps(A + 4);
			a2 = _mm_loadu_ps(A + 8);
			a3 = _mm_loadu_ps(A + 12);
		}

		if (reinterpret_cast<uintptr_t>(B) % 16 == 0)
		{
			b0 = _mm_load_ps(B);
			b1 = _mm_load_ps(B + 4);
			b2 = _mm_load_ps(B + 8);
			b3 = _mm_load_ps(B + 12);
		}
		else
		{
			b0 = _mm_loadu_ps(B);
			b1 = _mm_loadu_ps(B + 4);
			b2 = _mm_loadu_ps(B + 8);
			b3 = _mm_loadu_ps(B + 12);
		}

		c0 = _mm_sub_ps(a0, b0);
		c1 = _mm_sub_ps(a1, b1);
		c2 = _mm_sub_ps(a2, b2);
		c3 = _mm_sub_ps(a3, b3);

		if (reinterpret_cast<uintptr_t>(C) % 16 == 0)
		{
			_mm_store_ps(C, c0);
			_mm_store_ps(C + 4, c1);
			_mm_store_ps(C + 8, c2);
			_mm_store_ps(C + 12,c3);
		}
		else
		{
			_mm_storeu_ps(C, c0);
			_mm_storeu_ps(C + 4, c1);
			_mm_storeu_ps(C + 8, c2);
			_mm_storeu_ps(C + 12,c3);
		}
	}

	template <>
	void 
	_union_block_sse<float, std::multiplies<>>(float* A, float* B, float* C)
	{
		alignas(16)__m128 a0, a1, a2, a3;
		alignas(16)__m128 b0, b1, b2, b3;
		__m128 c0, c1, c2, c3;

		if (reinterpret_cast<uintptr_t>(A) % 16 == 0)
		{
			a0 = _mm_load_ps(A);
			a1 = _mm_load_ps(A + 4);
			a2 = _mm_load_ps(A + 8);
			a3 = _mm_load_ps(A + 12);
		}
		else
		{
			a0 = _mm_loadu_ps(A);
			a1 = _mm_loadu_ps(A + 4);
			a2 = _mm_loadu_ps(A + 8);
			a3 = _mm_loadu_ps(A + 12);
		}

		if (reinterpret_cast<uintptr_t>(B) % 16 == 0)
		{
			b0 = _mm_load_ps(B);
			b1 = _mm_load_ps(B + 4);
			b2 = _mm_load_ps(B + 8);
			b3 = _mm_load_ps(B + 12);
		}
		else
		{
			b0 = _mm_loadu_ps(B);
			b1 = _mm_loadu_ps(B + 4);
			b2 = _mm_loadu_ps(B + 8);
			b3 = _mm_loadu_ps(B + 12);
		}

		c0 = _mm_mul_ps(a0, b0);
		c1 = _mm_mul_ps(a1, b1);
		c2 = _mm_mul_ps(a2, b2);
		c3 = _mm_mul_ps(a3, b3);

		if (reinterpret_cast<uintptr_t>(C) % 16 == 0)
		{
			_mm_store_ps(C, c0);
			_mm_store_ps(C + 4, c1);
			_mm_store_ps(C + 8, c2);
			_mm_store_ps(C + 12,c3);
		}
		else
		{
			_mm_storeu_ps(C, c0);
			_mm_storeu_ps(C + 4, c1);
			_mm_storeu_ps(C + 8, c2);
			_mm_storeu_ps(C + 12,c3);
		}	
	}

	template <>
	void 
	_union_block_sse<float, std::divides<>>(float* A, float* B, float* C)
	{
		alignas(16)__m128 a0, a1, a2, a3;
		alignas(16)__m128 b0, b1, b2, b3;
		__m128 c0, c1, c2, c3;

		if (reinterpret_cast<uintptr_t>(A) % 16 == 0)
		{
			a0 = _mm_load_ps(A);
			a1 = _mm_load_ps(A + 4);
			a2 = _mm_load_ps(A + 8);
			a3 = _mm_load_ps(A + 12);
		}
		else
		{
			a0 = _mm_loadu_ps(A);
			a1 = _mm_loadu_ps(A + 4);
			a2 = _mm_loadu_ps(A + 8);
			a3 = _mm_loadu_ps(A + 12);
		}

		if (reinterpret_cast<uintptr_t>(B) % 16 == 0)
		{
			b0 = _mm_load_ps(B);
			b1 = _mm_load_ps(B + 4);
			b2 = _mm_load_ps(B + 8);
			b3 = _mm_load_ps(B + 12);
		}
		else
		{
			b0 = _mm_loadu_ps(B);
			b1 = _mm_loadu_ps(B + 4);
			b2 = _mm_loadu_ps(B + 8);
			b3 = _mm_loadu_ps(B + 12);
		}

		c0 = _mm_div_ps(a0, b0);
		c1 = _mm_div_ps(a1, b1);
		c2 = _mm_div_ps(a2, b2);
		c3 = _mm_div_ps(a3, b3);

		if (reinterpret_cast<uintptr_t>(C) % 16 == 0)
		{
			_mm_store_ps(C, c0);
			_mm_store_ps(C + 4, c1);
			_mm_store_ps(C + 8, c2);
			_mm_store_ps(C + 12,c3);
		}
		else
		{
			_mm_storeu_ps(C, c0);
			_mm_storeu_ps(C + 4, c1);
			_mm_storeu_ps(C + 8, c2);
			_mm_storeu_ps(C + 12,c3);
		}
	}


	template <>
	void 
	_union_block_sse<double, std::plus<>>(double* A, double* B, double* C)
	{
		alignas(16) __m128d a0, a1, a2, a3;
		alignas(16) __m128d b0, b1, b2, b3;
		__m128d c0, c1, c2, c3;

		if (reinterpret_cast<uintptr_t>(A) % 16 == 0)
		{
			a0 = _mm_load_pd(A);
			a1 = _mm_load_pd(A + 2);
			a2 = _mm_load_pd(A + 4);
			a3 = _mm_load_pd(A + 6);
		}
		else
		{
			a0 = _mm_loadu_pd(A);
			a1 = _mm_loadu_pd(A + 2);
			a2 = _mm_loadu_pd(A + 4);
			a3 = _mm_loadu_pd(A + 6);
		}

		if (reinterpret_cast<uintptr_t>(B) % 16 == 0)
		{
			b0 = _mm_load_pd(B);
			b1 = _mm_load_pd(B + 2);
			b2 = _mm_load_pd(B + 4);
			b3 = _mm_load_pd(B + 6);
		}
		else
		{
			b0 = _mm_loadu_pd(B);
			b1 = _mm_loadu_pd(B + 2);
			b2 = _mm_loadu_pd(B + 4);
			b3 = _mm_loadu_pd(B + 6);
		}

		c0 = _mm_add_pd(a0, b0);
		c1 = _mm_add_pd(a1, b1);
		c2 = _mm_add_pd(a2, b2);
		c3 = _mm_add_pd(a3, b3);

		if (reinterpret_cast<uintptr_t>(C) % 16 == 0)
		{
			_mm_store_pd(C, c0);
			_mm_store_pd(C + 2, c1);
			_mm_store_pd(C + 4, c2);
			_mm_store_pd(C + 6, c3);
		}
		else
		{
			_mm_storeu_pd(C, c0);
			_mm_storeu_pd(C + 2, c1);
			_mm_storeu_pd(C + 4, c2);
			_mm_storeu_pd(C + 6, c3);
		}
	}

	template <>
	void 
	_union_block_sse<double, std::minus<>>(double* A, double* B, double* C)
	{
		alignas(16) __m128d a0, a1, a2, a3;
		alignas(16) __m128d b0, b1, b2, b3;
		__m128d c0, c1, c2, c3;

		if (reinterpret_cast<uintptr_t>(A) % 16 == 0)
		{
			a0 = _mm_load_pd(A);
			a1 = _mm_load_pd(A + 2);
			a2 = _mm_load_pd(A + 4);
			a3 = _mm_load_pd(A + 6);
		}
		else
		{
			a0 = _mm_loadu_pd(A);
			a1 = _mm_loadu_pd(A + 2);
			a2 = _mm_loadu_pd(A + 4);
			a3 = _mm_loadu_pd(A + 6);
		}

		if (reinterpret_cast<uintptr_t>(B) % 16 == 0)
		{
			b0 = _mm_load_pd(B);
			b1 = _mm_load_pd(B + 2);
			b2 = _mm_load_pd(B + 4);
			b3 = _mm_load_pd(B + 6);
		}
		else
		{
			b0 = _mm_loadu_pd(B);
			b1 = _mm_loadu_pd(B + 2);
			b2 = _mm_loadu_pd(B + 4);
			b3 = _mm_loadu_pd(B + 6);
		}

		c0 = _mm_sub_pd(a0, b0);
		c1 = _mm_sub_pd(a1, b1);
		c2 = _mm_sub_pd(a2, b2);
		c3 = _mm_sub_pd(a3, b3);

		if (reinterpret_cast<uintptr_t>(C) % 16 == 0)
		{
			_mm_store_pd(C, c0);
			_mm_store_pd(C + 2, c1);
			_mm_store_pd(C + 4, c2);
			_mm_store_pd(C + 6, c3);
		}
		else
		{
			_mm_storeu_pd(C, c0);
			_mm_storeu_pd(C + 2, c1);
			_mm_storeu_pd(C + 4, c2);
			_mm_storeu_pd(C + 6, c3);
		}	
	}

	template <>
	void 
	_union_block_sse<double, std::multiplies<>>(double* A, double* B, double* C)
	{
		alignas(16) __m128d a0, a1, a2, a3;
		alignas(16) __m128d b0, b1, b2, b3;
		__m128d c0, c1, c2, c3;

		if (reinterpret_cast<uintptr_t>(A) % 16 == 0)
		{
			a0 = _mm_load_pd(A);
			a1 = _mm_load_pd(A + 2);
			a2 = _mm_load_pd(A + 4);
			a3 = _mm_load_pd(A + 6);
		}
		else
		{
			a0 = _mm_loadu_pd(A);
			a1 = _mm_loadu_pd(A + 2);
			a2 = _mm_loadu_pd(A + 4);
			a3 = _mm_loadu_pd(A + 6);
		}

		if (reinterpret_cast<uintptr_t>(B) % 16 == 0)
		{
			b0 = _mm_load_pd(B);
			b1 = _mm_load_pd(B + 2);
			b2 = _mm_load_pd(B + 4);
			b3 = _mm_load_pd(B + 6);
		}
		else
		{
			b0 = _mm_loadu_pd(B);
			b1 = _mm_loadu_pd(B + 2);
			b2 = _mm_loadu_pd(B + 4);
			b3 = _mm_loadu_pd(B + 6);
		}

		c0 = _mm_mul_pd(a0, b0);
		c1 = _mm_mul_pd(a1, b1);
		c2 = _mm_mul_pd(a2, b2);
		c3 = _mm_mul_pd(a3, b3);

		if (reinterpret_cast<uintptr_t>(C) % 16 == 0)
		{
			_mm_store_pd(C, c0);
			_mm_store_pd(C + 2, c1);
			_mm_store_pd(C + 4, c2);
			_mm_store_pd(C + 6, c3);
		}
		else
		{
			_mm_storeu_pd(C, c0);
			_mm_storeu_pd(C + 2, c1);
			_mm_storeu_pd(C + 4, c2);
			_mm_storeu_pd(C + 6, c3);
		}
	}

	template <>
	void 
	_union_block_sse<double, std::divides<>>(double* A, double* B, double* C)
	{
		alignas(16) __m128d a0, a1, a2, a3;
		alignas(16) __m128d b0, b1, b2, b3;
		__m128d c0, c1, c2, c3;

		if (reinterpret_cast<uintptr_t>(A) % 16 == 0)
		{
			a0 = _mm_load_pd(A);
			a1 = _mm_load_pd(A + 2);
			a2 = _mm_load_pd(A + 4);
			a3 = _mm_load_pd(A + 6);
		}
		else
		{
			a0 = _mm_loadu_pd(A);
			a1 = _mm_loadu_pd(A + 2);
			a2 = _mm_loadu_pd(A + 4);
			a3 = _mm_loadu_pd(A + 6);
		}

		if (reinterpret_cast<uintptr_t>(B) % 16 == 0)
		{
			b0 = _mm_load_pd(B);
			b1 = _mm_load_pd(B + 2);
			b2 = _mm_load_pd(B + 4);
			b3 = _mm_load_pd(B + 6);
		}
		else
		{
			b0 = _mm_loadu_pd(B);
			b1 = _mm_loadu_pd(B + 2);
			b2 = _mm_loadu_pd(B + 4);
			b3 = _mm_loadu_pd(B + 6);
		}

		c0 = _mm_div_pd(a0, b0);
		c1 = _mm_div_pd(a1, b1);
		c2 = _mm_div_pd(a2, b2);
		c3 = _mm_div_pd(a3, b3);

		if (reinterpret_cast<uintptr_t>(C) % 16 == 0)
		{
			_mm_store_pd(C, c0);
			_mm_store_pd(C + 2, c1);
			_mm_store_pd(C + 4, c2);
			_mm_store_pd(C + 6, c3);
		}
		else
		{
			_mm_storeu_pd(C, c0);
			_mm_storeu_pd(C + 2, c1);
			_mm_storeu_pd(C + 4, c2);
			_mm_storeu_pd(C + 6, c3);
		}
	}

/*AVX256*/

	template <>
	void 
	_union_block_avx<float, std::plus<>>(float* A, float* B, float* C)
	{
		alignas(32) __m256 a0, a1;
		alignas(32) __m256 b0, b1;
		__m256 c0, c1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			a0 = _mm256_load_ps(A);
			a1 = _mm256_load_ps(A + 8);
		}
		else
		{
			a0 = _mm256_loadu_ps(A);
			a1 = _mm256_loadu_ps(A + 8);
		}

		if (reinterpret_cast<uintptr_t>(B) % 32 == 0)
		{
			b0 = _mm256_load_ps(B);
			b1 = _mm256_load_ps(B + 8);
		}
		else
		{
			b0 = _mm256_loadu_ps(B);
			b1 = _mm256_loadu_ps(B + 8);
		}

		c0 = _mm256_add_ps(a0, b0);
		c1 = _mm256_add_ps(a1, b1);

		if (reinterpret_cast<uintptr_t>(C) % 32 == 0)
		{
			_mm256_store_ps(C, c0);
			_mm256_store_ps(C + 8, c1);
		}
		else
		{
			_mm256_storeu_ps(C, c0);
			_mm256_storeu_ps(C + 8, c1);
		}
	}

	template <>
	void 
	_union_block_avx<float, std::minus<>>(float* A, float* B, float* C)
	{
		alignas(32) __m256 a0, a1;
		alignas(32) __m256 b0, b1;
		__m256 c0, c1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			a0 = _mm256_load_ps(A);
			a1 = _mm256_load_ps(A + 8);
		}
		else
		{
			a0 = _mm256_loadu_ps(A);
			a1 = _mm256_loadu_ps(A + 8);
		}

		if (reinterpret_cast<uintptr_t>(B) % 32 == 0)
		{
			b0 = _mm256_load_ps(B);
			b1 = _mm256_load_ps(B + 8);
		}
		else
		{
			b0 = _mm256_loadu_ps(B);
			b1 = _mm256_loadu_ps(B + 8);
		}

		c0 = _mm256_sub_ps(a0, b0);
		c1 = _mm256_sub_ps(a1, b1);

		if (reinterpret_cast<uintptr_t>(C) % 32 == 0)
		{
			_mm256_store_ps(C, c0);
			_mm256_store_ps(C + 8, c1);
		}
		else
		{
			_mm256_storeu_ps(C, c0);
			_mm256_storeu_ps(C + 8, c1);
		}
	}

	template <>
	void 
	_union_block_avx<float, std::multiplies<>>(float* A, float* B, float* C)
	{
		alignas(32) __m256 a0, a1;
		alignas(32) __m256 b0, b1;
		__m256 c0, c1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			a0 = _mm256_load_ps(A);
			a1 = _mm256_load_ps(A + 8);
		}
		else
		{
			a0 = _mm256_loadu_ps(A);
			a1 = _mm256_loadu_ps(A + 8);
		}

		if (reinterpret_cast<uintptr_t>(B) % 32 == 0)
		{
			b0 = _mm256_load_ps(B);
			b1 = _mm256_load_ps(B + 8);
		}
		else
		{
			b0 = _mm256_loadu_ps(B);
			b1 = _mm256_loadu_ps(B + 8);
		}

		c0 = _mm256_mul_ps(a0, b0);
		c1 = _mm256_mul_ps(a1, b1);

		if (reinterpret_cast<uintptr_t>(C) % 32 == 0)
		{
			_mm256_store_ps(C, c0);
			_mm256_store_ps(C + 8, c1);
		}
		else
		{
			_mm256_storeu_ps(C, c0);
			_mm256_storeu_ps(C + 8, c1);
		}
	}

	template <>
	void 
	_union_block_avx<float, std::divides<>>(float* A, float* B, float* C)
	{
		alignas(32) __m256 a0, a1;
		alignas(32) __m256 b0, b1;
		__m256 c0, c1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			a0 = _mm256_load_ps(A);
			a1 = _mm256_load_ps(A + 8);
		}
		else
		{
			a0 = _mm256_loadu_ps(A);
			a1 = _mm256_loadu_ps(A + 8);
		}

		if (reinterpret_cast<uintptr_t>(B) % 32 == 0)
		{
			b0 = _mm256_load_ps(B);
			b1 = _mm256_load_ps(B + 8);
		}
		else
		{
			b0 = _mm256_loadu_ps(B);
			b1 = _mm256_loadu_ps(B + 8);
		}

		c0 = _mm256_div_ps(a0, b0);
		c1 = _mm256_div_ps(a1, b1);

		if (reinterpret_cast<uintptr_t>(C) % 32 == 0)
		{
			_mm256_store_ps(C, c0);
			_mm256_store_ps(C + 8, c1);
		}
		else
		{
			_mm256_storeu_ps(C, c0);
			_mm256_storeu_ps(C + 8, c1);
		}
	}

	template <>
	void 
	_union_block_avx<double, std::plus<>>(double* A, double* B, double* C)
	{
		alignas(32) __m256d a0, a1;
		alignas(32) __m256d b0, b1;
		__m256d c0, c1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			a0 = _mm256_load_pd(A);
			a1 = _mm256_load_pd(A + 4);
		}
		else
		{
			a0 = _mm256_loadu_pd(A);
			a1 = _mm256_loadu_pd(A + 4);
		}

		if (reinterpret_cast<uintptr_t>(B) % 32 == 0)
		{
			b0 = _mm256_load_pd(B);
			b1 = _mm256_load_pd(B + 4);
		}
		else
		{
			b0 = _mm256_loadu_pd(B);
			b1 = _mm256_loadu_pd(B + 4);
		}

		c0 = _mm256_add_pd(a0, b0);
		c1 = _mm256_add_pd(a1, b1);

		if (reinterpret_cast<uintptr_t>(C) % 32 == 0)
		{
			_mm256_store_pd(C, c0);
			_mm256_store_pd(C + 4, c1);
		}
		else
		{
			_mm256_storeu_pd(C, c0);
			_mm256_storeu_pd(C + 4, c1);
		}
	}

	template <>
	void 
	_union_block_avx<double, std::minus<>>(double* A, double* B, double* C)
	{
		alignas(32) __m256d a0, a1;
		alignas(32) __m256d b0, b1;
		__m256d c0, c1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			a0 = _mm256_load_pd(A);
			a1 = _mm256_load_pd(A + 4);
		}
		else
		{
			a0 = _mm256_loadu_pd(A);
			a1 = _mm256_loadu_pd(A + 4);
		}

		if (reinterpret_cast<uintptr_t>(B) % 32 == 0)
		{
			b0 = _mm256_load_pd(B);
			b1 = _mm256_load_pd(B + 4);
		}
		else
		{
			b0 = _mm256_loadu_pd(B);
			b1 = _mm256_loadu_pd(B + 4);
		}

		c0 = _mm256_sub_pd(a0, b0);
		c1 = _mm256_sub_pd(a1, b1);

		if (reinterpret_cast<uintptr_t>(C) % 32 == 0)
		{
			_mm256_store_pd(C, c0);
			_mm256_store_pd(C + 4, c1);
		}
		else
		{
			_mm256_storeu_pd(C, c0);
			_mm256_storeu_pd(C + 4, c1);
		}
	}

	template <>
	void 
	_union_block_avx<double, std::multiplies<>>(double* A, double* B, double* C)
	{
		alignas(32) __m256d a0, a1;
		alignas(32) __m256d b0, b1;
		__m256d c0, c1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			a0 = _mm256_load_pd(A);
			a1 = _mm256_load_pd(A + 4);
		}
		else
		{
			a0 = _mm256_loadu_pd(A);
			a1 = _mm256_loadu_pd(A + 4);
		}

		if (reinterpret_cast<uintptr_t>(B) % 32 == 0)
		{
			b0 = _mm256_load_pd(B);
			b1 = _mm256_load_pd(B + 4);
		}
		else
		{
			b0 = _mm256_loadu_pd(B);
			b1 = _mm256_loadu_pd(B + 4);
		}

		c0 = _mm256_mul_pd(a0, b0);
		c1 = _mm256_mul_pd(a1, b1);

		if (reinterpret_cast<uintptr_t>(C) % 32 == 0)
		{
			_mm256_store_pd(C, c0);
			_mm256_store_pd(C + 4, c1);
		}
		else
		{
			_mm256_storeu_pd(C, c0);
			_mm256_storeu_pd(C + 4, c1);
		}
	}

	template <>
	void 
	_union_block_avx<double, std::divides<>>(double* A, double* B, double* C)
	{
		alignas(32) __m256d a0, a1;
		alignas(32) __m256d b0, b1;
		__m256d c0, c1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			a0 = _mm256_load_pd(A);
			a1 = _mm256_load_pd(A + 4);
		}
		else
		{
			a0 = _mm256_loadu_pd(A);
			a1 = _mm256_loadu_pd(A + 4);
		}

		if (reinterpret_cast<uintptr_t>(B) % 32 == 0)
		{
			b0 = _mm256_load_pd(B);
			b1 = _mm256_load_pd(B + 4);
		}
		else
		{
			b0 = _mm256_loadu_pd(B);
			b1 = _mm256_loadu_pd(B + 4);
		}

		c0 = _mm256_div_pd(a0, b0);
		c1 = _mm256_div_pd(a1, b1);

		if (reinterpret_cast<uintptr_t>(C) % 32 == 0)
		{
			_mm256_store_pd(C, c0);
			_mm256_store_pd(C + 4, c1);
		}
		else
		{
			_mm256_storeu_pd(C, c0);
			_mm256_storeu_pd(C + 4, c1);
		}
	}

/*AVX512*/
	template <>
	void 
	_union_block_avx512<float, std::plus<>>(float* A, float* B, float* C)
	{
		alignas(64) __m512 a, b, c;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_ps(A);
		}
		else
		{
			a = _mm512_loadu_ps(A);
		}

		if (reinterpret_cast<uintptr_t>(B) % 64 == 0)
		{
			b = _mm512_load_ps(B);
		}
		else
		{
			b = _mm512_loadu_ps(B);
		}

		c = _mm512_add_ps(a, b);

		if (reinterpret_cast<uintptr_t>(C) % 64 == 0)
		{
			_mm512_store_ps(C, c);
		}
		else
		{
			_mm512_storeu_ps(C, c);
		}
	}

	template <>
	void 
	_union_block_avx512<float, std::minus<>>(float* A, float* B, float* C)
	{
		alignas(64) __m512 a, b, c;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_ps(A);
		}
		else
		{
			a = _mm512_loadu_ps(A);
		}

		if (reinterpret_cast<uintptr_t>(B) % 64 == 0)
		{
			b = _mm512_load_ps(B);
		}
		else
		{
			b = _mm512_loadu_ps(B);
		}

		c = _mm512_sub_ps(a, b);

		if (reinterpret_cast<uintptr_t>(C) % 64 == 0)
		{
			_mm512_store_ps(C, c);
		}
		else
		{
			_mm512_storeu_ps(C, c);
		}
	}

	template <>
	void 
	_union_block_avx512<float, std::multiplies<>>(float* A, float* B, float* C)
	{
		alignas(64) __m512 a, b, c;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_ps(A);
		}
		else
		{
			a = _mm512_loadu_ps(A);
		}

		if (reinterpret_cast<uintptr_t>(B) % 64 == 0)
		{
			b = _mm512_load_ps(B);
		}
		else
		{
			b = _mm512_loadu_ps(B);
		}

		c = _mm512_mul_ps(a, b);

		if (reinterpret_cast<uintptr_t>(C) % 64 == 0)
		{
			_mm512_store_ps(C, c);
		}
		else
		{
			_mm512_storeu_ps(C, c);
		}
	}

	template <>
	void 
	_union_block_avx512<float, std::divides<>>(float* A, float* B, float* C)
	{
		alignas(64) __m512 a, b, c;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_ps(A);
		}
		else
		{
			a = _mm512_loadu_ps(A);
		}

		if (reinterpret_cast<uintptr_t>(B) % 64 == 0)
		{
			b = _mm512_load_ps(B);
		}
		else
		{
			b = _mm512_loadu_ps(B);
		}

		c = _mm512_div_ps(a, b);

		if (reinterpret_cast<uintptr_t>(C) % 64 == 0)
		{
			_mm512_store_ps(C, c);
		}
		else
		{
			_mm512_storeu_ps(C, c);
		}
	}

	template <>
	void 
	_union_block_avx512<double, std::plus<>>(double* A, double* B, double* C)
	{
		alignas(64) __m512d a, b, c;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_pd(A);
		}
		else
		{
			a = _mm512_loadu_pd(A);
		}

		if (reinterpret_cast<uintptr_t>(B) % 64 == 0)
		{
			b = _mm512_load_pd(B);
		}
		else
		{
			b = _mm512_loadu_pd(B);
		}

		c = _mm512_add_pd(a, b);

		if (reinterpret_cast<uintptr_t>(C) % 64 == 0)
		{
			_mm512_store_pd(C, c);
		}
		else
		{
			_mm512_storeu_pd(C, c);
		}
	}

	template <>
	void 
	_union_block_avx512<double, std::minus<>>(double* A, double* B, double* C)
	{
		alignas(64) __m512d a, b, c;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_pd(A);
		}
		else
		{
			a = _mm512_loadu_pd(A);
		}

		if (reinterpret_cast<uintptr_t>(B) % 64 == 0)
		{
			b = _mm512_load_pd(B);
		}
		else
		{
			b = _mm512_loadu_pd(B);
		}

		c = _mm512_sub_pd(a, b);

		if (reinterpret_cast<uintptr_t>(C) % 64 == 0)
		{
			_mm512_store_pd(C, c);
		}
		else
		{
			_mm512_storeu_pd(C, c);
		}
	}

	template <>
	void 
	_union_block_avx512<double, std::multiplies<>>(double* A, double* B, double* C)
	{
		alignas(64) __m512d a, b, c;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_pd(A);
		}
		else
		{
			a = _mm512_loadu_pd(A);
		}

		if (reinterpret_cast<uintptr_t>(B) % 64 == 0)
		{
			b = _mm512_load_pd(B);
		}
		else
		{
			b = _mm512_loadu_pd(B);
		}

		c = _mm512_mul_pd(a, b);

		if (reinterpret_cast<uintptr_t>(C) % 64 == 0)
		{
			_mm512_store_pd(C, c);
		}
		else
		{
			_mm512_storeu_pd(C, c);
		}
	}

	template <>
	void 
	_union_block_avx512<double, std::divides<>>(double* A, double* B, double* C)
	{
				alignas(64) __m512d a, b, c;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_pd(A);
		}
		else
		{
			a = _mm512_loadu_pd(A);
		}

		if (reinterpret_cast<uintptr_t>(B) % 64 == 0)
		{
			b = _mm512_load_pd(B);
		}
		else
		{
			b = _mm512_loadu_pd(B);
		}

		c = _mm512_div_pd(a, b);

		if (reinterpret_cast<uintptr_t>(C) % 64 == 0)
		{
			_mm512_store_pd(C, c);
		}
		else
		{
			_mm512_storeu_pd(C, c);
		}
	}

} //namespace damm