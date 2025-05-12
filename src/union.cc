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
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRAMTY OF AMY KIMD, EXPRESS OR
// IMPLIED, IMCLUDIMG BUT MOT LIMITED TO THE WARRAMTIES OF MERCHAMTABILITY,
// FITMESS FOR A PARTICULAR PURPOSE AMD MOMIMFRIMGEMEMT. IM MO EVEMT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR AMY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IM AM ACTIOM OF COMTRACT, TORT OR OTHERWISE, ARISIMG FROM,
// OUT OF OR IM COMMECTIOM WITH THE SOFTWARE OR THE USE OR OTHER DEALIMGS IM
// THE SOFTWARE.

#include <union.h>

namespace damm
{
/*SSE*/
	template <>
	void 
	_union_block_sse<float, std::plus<>>(float* A, float* B, float* C,  const size_t M)
	{
		alignas(16)__m128 a, b, c;

		if ((M * sizeof(float)) % 16 == 0)
		{
			a = _mm_load_ps(A);
		}
		else
		{
			a = _mm_loadu_ps(A);
		}

		if ((M * sizeof(float)) % 16 == 0)
		{
			b = _mm_load_ps(B);
		}
		else
		{
			b = _mm_loadu_ps(B);
		}

		c = _mm_add_ps(a, b);

		if ((M * sizeof(float)) % 16 == 0)
		{
			_mm_store_ps(C, c);
		}
		else
		{
			_mm_storeu_ps(C, c);
		}
	}

	template <>
	void 
	_union_block_sse<float, std::minus<>>(float* A, float* B, float* C, const size_t M)
	{
		alignas(16)__m128 a, b, c;

		if ((M * sizeof(float)) % 16 == 0)
		{
			a = _mm_load_ps(A);
		}
		else
		{
			a = _mm_loadu_ps(A);
		}

		if ((M * sizeof(float)) % 16 == 0)
		{
			b = _mm_load_ps(B);
		}
		else
		{
			b = _mm_loadu_ps(B);
		}

		c = _mm_sub_ps(a, b);

		if ((M * sizeof(float)) % 16 == 0)
		{
			_mm_store_ps(C, c);
		}
		else
		{
			_mm_storeu_ps(C, c);
		}
	}

	template <>
	void 
	_union_block_sse<float, std::multiplies<>>(float* A, float* B, float* C, const size_t M)
	{
		alignas(16)__m128 a, b, c;

		if ((M * sizeof(float)) % 16 == 0)
		{
			a = _mm_load_ps(A);
		}
		else
		{
			a = _mm_loadu_ps(A);
		}

		if ((M * sizeof(float)) % 16 == 0)
		{
			b = _mm_load_ps(B);
		}
		else
		{
			b = _mm_loadu_ps(B);
		}

		c = _mm_mul_ps(a, b);

		if ((M * sizeof(float)) % 16 == 0)
		{
			_mm_store_ps(C, c);
		}
		else
		{
			_mm_storeu_ps(C, c);
		}
	}


	template <>
	void 
	_union_block_sse<double, std::plus<>>(double* A, double* B, double* C, const size_t M)
	{
		alignas(16)__m128d a, b, c;

		if ((M * sizeof(double)) % 16 == 0)
		{
			a = _mm_load_pd(A);
		}
		else
		{
			a = _mm_loadu_pd(A);
		}

		if ((M * sizeof(double)) % 16 == 0)
		{
			b = _mm_load_pd(B);
		}
		else
		{
			b = _mm_loadu_pd(B);
		}

		c = _mm_add_pd(a, b);

		if ((M * sizeof(double)) % 16 == 0)
		{
			_mm_store_pd(C, c);
		}
		else
		{
			_mm_storeu_pd(C, c);
		}
	}

	template <>
	void 
	_union_block_sse<double, std::minus<>>(double* A, double* B, double* C, const size_t M)
	{
		alignas(16)__m128d a, b, c;

		if ((M * sizeof(double)) % 16 == 0)
		{
			a = _mm_load_pd(A);
		}
		else
		{
			a = _mm_loadu_pd(A);
		}

		if ((M * sizeof(double)) % 16 == 0)
		{
			b = _mm_load_pd(B);
		}
		else
		{
			b = _mm_loadu_pd(B);
		}

		c = _mm_sub_pd(a, b);

		if ((M * sizeof(double)) % 16 == 0)
		{
			_mm_store_pd(C, c);
		}
		else
		{
			_mm_storeu_pd(C, c);
		}
	}

	template <>
	void 
	_union_block_sse<double, std::multiplies<>>(double* A, double* B, double* C, const size_t M)
	{
		alignas(16)__m128d a, b, c;

		if ((M * sizeof(double)) % 16 == 0)
		{
			a = _mm_load_pd(A);
		}
		else
		{
			a = _mm_loadu_pd(A);
		}

		if ((M * sizeof(double)) % 16 == 0)
		{
			b = _mm_load_pd(B);
		}
		else
		{
			b = _mm_loadu_pd(B);
		}

		c = _mm_mul_pd(a, b);

		if ((M * sizeof(double)) % 16 == 0)
		{
			_mm_store_pd(C, c);
		}
		else
		{
			_mm_storeu_pd(C, c);
		}
	}

/*AVX256*/

	template <>
	void 
	_union_block_avx<float, std::plus<>>(float* A, float* B, float* C, const size_t M)
	{
		alignas(32)__m256 a, b, c;

		if ((M * sizeof(float)) % 32 == 0)
		{
			a = _mm256_load_ps(A);
		}
		else
		{
			a = _mm256_loadu_ps(A);
		}

		if ((M * sizeof(float)) % 32 == 0)
		{
			b = _mm256_load_ps(B);
		}
		else
		{
			b = _mm256_loadu_ps(B);
		}

		c = _mm256_add_ps(a, b);

		if ((M * sizeof(float)) % 32 == 0)
		{
			_mm256_store_ps(C, c);
		}
		else
		{
			_mm256_storeu_ps(C, c);
		}
	}

	template <>
	void 
	_union_block_avx<float, std::minus<>>(float* A, float* B, float* C, const size_t M)
	{
		alignas(32)__m256 a, b, c;

		if ((M * sizeof(float)) % 32 == 0)
		{
			a = _mm256_load_ps(A);
		}
		else
		{
			a = _mm256_loadu_ps(A);
		}

		if ((M * sizeof(float)) % 32 == 0)
		{
			b = _mm256_load_ps(B);
		}
		else
		{
			b = _mm256_loadu_ps(B);
		}

		c = _mm256_sub_ps(a, b);

		if ((M * sizeof(float)) % 32 == 0)
		{
			_mm256_store_ps(C, c);
		}
		else
		{
			_mm256_storeu_ps(C, c);
		}
	}

	template <>
	void 
	_union_block_avx<float, std::multiplies<>>(float* A, float* B, float* C, const size_t M)
	{
		alignas(32)__m256 a, b, c;

		if ((M * sizeof(float)) % 32 == 0)
		{
			a = _mm256_load_ps(A);
		}
		else
		{
			a = _mm256_loadu_ps(A);
		}

		if ((M * sizeof(float)) % 32 == 0)
		{
			b = _mm256_load_ps(B);
		}
		else
		{
			b = _mm256_loadu_ps(B);
		}

		c = _mm256_mul_ps(a, b);

		if ((M * sizeof(float)) % 32 == 0)
		{
			_mm256_store_ps(C, c);
		}
		else
		{
			_mm256_storeu_ps(C, c);
		}
	}

	template <>
	void 
	_union_block_avx<double, std::plus<>>(double* A, double* B, double* C, const size_t M)
	{
		alignas(32)__m256d a, b, c;

		if ((M * sizeof(double)) % 32 == 0)
		{
			a = _mm256_load_pd(A);
		}
		else
		{
			a = _mm256_loadu_pd(A);
		}

		if ((M * sizeof(double)) % 32 == 0)
		{
			b = _mm256_load_pd(B);
		}
		else
		{
			b = _mm256_loadu_pd(B);
		}

		c = _mm256_add_pd(a, b);

		if ((M * sizeof(double)) % 32 == 0)
		{
			_mm256_store_pd(C, c);
		}
		else
		{
			_mm256_storeu_pd(C, c);
		}
	}

	template <>
	void 
	_union_block_avx<double, std::minus<>>(double* A, double* B, double* C, const size_t M)
	{
		alignas(32)__m256d a, b, c;

		if ((M * sizeof(double)) % 32 == 0)
		{
			a = _mm256_load_pd(A);
		}
		else
		{
			a = _mm256_loadu_pd(A);
		}

		if ((M * sizeof(double)) % 32 == 0)
		{
			b = _mm256_load_pd(B);
		}
		else
		{
			b = _mm256_loadu_pd(B);
		}

		c = _mm256_sub_pd(a, b);

		if ((M * sizeof(double)) % 32 == 0)
		{
			_mm256_store_pd(C, c);
		}
		else
		{
			_mm256_storeu_pd(C, c);
		}
	}

	template <>
	void 
	_union_block_avx<double, std::multiplies<>>(double* A, double* B, double* C, const size_t M)
	{
		alignas(32)__m256d a, b, c;

		if ((M * sizeof(double)) % 32 == 0)
		{
			a = _mm256_load_pd(A);
		}
		else
		{
			a = _mm256_loadu_pd(A);
		}

		if ((M * sizeof(double)) % 32 == 0)
		{
			b = _mm256_load_pd(B);
		}
		else
		{
			b = _mm256_loadu_pd(B);
		}

		c = _mm256_mul_pd(a, b);

		if ((M * sizeof(double)) % 32 == 0)
		{
			_mm256_store_pd(C, c);
		}
		else
		{
			_mm256_storeu_pd(C, c);
		}
	}

/*AVX512*/
	template <>
	void 
	_union_block_avx512<float, std::plus<>>(float* A, float* B, float* C, const size_t M)
	{
		alignas(64)__m512 a, b, c;

		if ((M * sizeof(float)) % 64 == 0)
		{
			a = _mm512_load_ps(A);
		}
		else
		{
			a = _mm512_loadu_ps(A);
		}

		if ((M * sizeof(float)) % 64 == 0)
		{
			b = _mm512_load_ps(B);
		}
		else
		{
			b = _mm512_loadu_ps(B);
		}

		c = _mm512_add_ps(a, b);

		if ((M * sizeof(float)) % 64 == 0)
		{
			c = _mm512_load_ps(C);
		}
		else
		{
			c = _mm512_loadu_ps(C);
		}
	}

	template <>
	void 
	_union_block_avx512<float, std::minus<>>(float* A, float* B, float* C, const size_t M)
	{
		alignas(64)__m512 a, b, c;

		if ((M * sizeof(float)) % 64 == 0)
		{
			a = _mm512_load_ps(A);
		}
		else
		{
			a = _mm512_loadu_ps(A);
		}

		if ((M * sizeof(float)) % 64 == 0)
		{
			b = _mm512_load_ps(B);
		}
		else
		{
			b = _mm512_loadu_ps(B);
		}

		c = _mm512_sub_ps(a, b);

		if ((M * sizeof(float)) % 64 == 0)
		{
			c = _mm512_load_ps(C);
		}
		else
		{
			c = _mm512_loadu_ps(C);
		}
	}

	template <>
	void 
	_union_block_avx512<float, std::multiplies<>>(float* A, float* B, float* C, const size_t M)
	{
		alignas(64)__m512 a, b, c;

		if ((M * sizeof(float)) % 64 == 0)
		{
			a = _mm512_load_ps(A);
		}
		else
		{
			a = _mm512_loadu_ps(A);
		}

		if ((M * sizeof(float)) % 64 == 0)
		{
			b = _mm512_load_ps(B);
		}
		else
		{
			b = _mm512_loadu_ps(B);
		}

		c = _mm512_mul_ps(a, b);

		if ((M * sizeof(float)) % 64 == 0)
		{
			c = _mm512_load_ps(C);
		}
		else
		{
			c = _mm512_loadu_ps(C);
		}
	}

	template <>
	void 
	_union_block_avx512<double, std::plus<>>(double* A, double* B, double* C, const size_t M)
	{
		alignas(64)__m512d a, b, c;

		if ((M * sizeof(double)) % 64 == 0)
		{
			a = _mm512_load_pd(A);
		}
		else
		{
			a = _mm512_loadu_pd(A);

		}

		if ((M * sizeof(double)) % 64 == 0)
		{
			b = _mm512_load_pd(B);
		}
		else
		{
			b = _mm512_loadu_pd(B);
		}

		c = _mm512_add_pd(a, b);

		if ((M * sizeof(double)) % 64 == 0)
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
	_union_block_avx512<double, std::minus<>>(double* A, double* B, double* C, const size_t M)
	{
		alignas(64)__m512d a, b, c;

		if ((M * sizeof(double)) % 64 == 0)
		{
			a = _mm512_load_pd(A);
		}
		else
		{
			a = _mm512_loadu_pd(A);

		}

		if ((M * sizeof(double)) % 64 == 0)
		{
			b = _mm512_load_pd(B);
		}
		else
		{
			b = _mm512_loadu_pd(B);
		}

		c = _mm512_sub_pd(a, b);

		if ((M * sizeof(double)) % 64 == 0)
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
	_union_block_avx512<double, std::multiplies<>>(double* A, double* B, double* C, const size_t M)
	{
		alignas(64)__m512d a, b, c;

		if ((M * sizeof(double)) % 64 == 0)
		{
			a = _mm512_load_pd(A);
		}
		else
		{
			a = _mm512_loadu_pd(A);

		}

		if ((M * sizeof(double)) % 64 == 0)
		{
			b = _mm512_load_pd(B);
		}
		else
		{
			b = _mm512_loadu_pd(B);
		}

		c = _mm512_mul_pd(a, b);

		if ((M * sizeof(double)) % 64 == 0)
		{
			_mm512_store_pd(C, c);
		}
		else
		{
			_mm512_storeu_pd(C, c);
		}
	}

} //namespace damm