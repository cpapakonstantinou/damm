/**
 * \file reduce.cc
 * \brief reduce utilities implementations
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

#include <reduce.h>

namespace damm
{
/*SSE*/
	template <>
	void 
	_reduce_block_sse<float, std::plus<>>(float* A, float& r)
	{

		alignas(16)__m128 a0, a1, a2, a3; 

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

		__m128 acc = _mm_setzero_ps();

		acc = _mm_add_ps(acc, a0);
		acc = _mm_add_ps(acc, a1);
		acc = _mm_add_ps(acc, a2);
		acc = _mm_add_ps(acc, a3);

		__m128 tmp1 = _mm_movehdup_ps(acc);
		__m128 sum1 = _mm_add_ps(acc, tmp1);
		__m128 tmp2 = _mm_movehl_ps(tmp1, sum1);
		__m128 sum2 = _mm_add_ss(sum1, tmp2);

		r = _mm_cvtss_f32(sum2);
	}

	template <>
	void 
	_reduce_block_sse<float, std::multiplies<>>(float* A, float& r)
	{
		alignas(16)__m128 a0, a1, a2, a3;

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

		__m128 acc = _mm_set1_ps(1.0f);

		acc = _mm_mul_ps(acc, a0);
		acc = _mm_mul_ps(acc, a1);
		acc = _mm_mul_ps(acc, a2);
		acc = _mm_mul_ps(acc, a3);

		__m128 tmp1 = _mm_movehdup_ps(acc);
		__m128 prod1 = _mm_mul_ps(acc, tmp1);

		__m128 tmp2 = _mm_movehl_ps(tmp1, prod1);
		__m128 prod2 = _mm_mul_ss(prod1, tmp2);

		r = _mm_cvtss_f32(prod2);
	}

	// template <>
	// void 
	// _reduce_block_sse<float, std::divides<>>(float* A, float& r)
	// {
	// 	static_assert(false, "divide reduce not implemented for SSE");
	// }

	// template <>
	// void 
	// _reduce_block_sse<float, std::minus<>>(float* A, float& r)
	// {
	// 	static_assert(false, "minus reduce not implemented for SSE");
	// }


	template <>
	void 
	_reduce_block_sse<double, std::plus<>>(double* A, double& r)
	{
		alignas(16)__m128d a0, a1, a2, a3;

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

		__m128d acc0 = _mm_add_pd(a0, a1);
		__m128d acc1 = _mm_add_pd(a2, a3);
		__m128d acc = _mm_add_pd(acc0, acc1);

		__m128d sum0 = _mm_hadd_pd(acc, acc);
		r = _mm_cvtsd_f64(sum0);

	}

	template <>
	void 
	_reduce_block_sse<double, std::multiplies<>>(double* A, double& r)
	{
		alignas(16)__m128d a0, a1, a2, a3;

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

		__m128d acc = _mm_set1_pd(1.0);

		acc = _mm_mul_pd(acc, a0);
		acc = _mm_mul_pd(acc, a1);
		acc = _mm_mul_pd(acc, a2);
		acc = _mm_mul_pd(acc, a3);

		__m128d temp = _mm_unpackhi_pd(acc, acc);
		__m128d result = _mm_mul_sd(acc, temp);

		r = _mm_cvtsd_f64(result);
	}

	// template <>
	// void 
	// _reduce_block_sse<double, std::minus<>>(double* A, double& r)
	// {
	// 	//The cost to load and unload registers + perform the reduction operation is worse than the sequential reduction
	// 	static_assert(false, "minus reduce not implemented for SSE");
	// }


	// template <>
	// void 
	// _reduce_block_sse<double, std::divides<>>(double* A, double& r)
	// {
	// 	//The cost to load and unload registers + perform the reduction operation is worse than the sequential reduction
	// 	static_assert(false, "divide reduce not implemented for SSE");
	// }

/*AVX256*/

	template <>
	void 
	_reduce_block_avx<float, std::plus<>>(float* A, float& r)
	{
		alignas(32)__m256 v0, v1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			v0 = _mm256_load_ps(A);       // First 8 floats (32 bytes)
			v1 = _mm256_load_ps(A + 8);   // Next 8 floats (32 bytes)
		}
		else
		{
			v0 = _mm256_loadu_ps(A);
			v1 = _mm256_loadu_ps(A + 8);
		}

		__m256 sum = _mm256_add_ps(v0, v1);

		__m128 lo = _mm256_castps256_ps128(sum);
		__m128 hi = _mm256_extractf128_ps(sum, 1);
		__m128 acc = _mm_add_ps(lo, hi);

		__m128 tmp = _mm_movehdup_ps(acc);
		__m128 sum1 = _mm_add_ps(acc, tmp);
		__m128 tmp2 = _mm_movehl_ps(tmp, sum1);
		__m128 sum2 = _mm_add_ss(sum1, tmp2);

		r = _mm_cvtss_f32(sum2);
	}

	template <>
	void 
	_reduce_block_avx<float, std::multiplies<>>(float* A, float& r)
	{
		alignas(32)__m256 v0, v1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			v0 = _mm256_load_ps(A);       // First 8 floats
			v1 = _mm256_load_ps(A + 8);   // Next 8 floats
		}
		else
		{
			v0 = _mm256_loadu_ps(A);
			v1 = _mm256_loadu_ps(A + 8);
		}

		// Multiply element-wise
		__m256 prod = _mm256_mul_ps(v0, v1);

		// Horizontal multiply reduction
		__m128 lo = _mm256_castps256_ps128(prod);         // Lower 128 bits (4 floats)
		__m128 hi = _mm256_extractf128_ps(prod, 1);       // Upper 128 bits

		__m128 acc = _mm_mul_ps(lo, hi);  // 4 element-wise multiplies

		// Reduce 4 floats in acc
		__m128 shuf = _mm_movehdup_ps(acc);  // [acc1, acc1, acc3, acc3]
		__m128 prod1 = _mm_mul_ps(acc, shuf);
		__m128 shuf2 = _mm_movehl_ps(shuf, prod1);
		__m128 prod2 = _mm_mul_ss(prod1, shuf2);

		r = _mm_cvtss_f32(prod2);
	}

	// template <>
	// void 
	// _reduce_block_avx<float, std::minus<>>(float* A, float& r)
	// {
	// 	static_assert(false, "minus reduce not implemented for AVX");
	// }

	// template <>
	// void 
	// _reduce_block_avx<float, std::divides<>>(float* A, float& r)
	// {
	// 	static_assert(false, "divide reduce not implemented for AVX");
	// }

	template <>
	void 
	_reduce_block_avx<double, std::plus<>>(double* A, double& r)
	{
		alignas(32)__m256d v0, v1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			v0 = _mm256_load_pd(A);       // First 4 doubles
			v1 = _mm256_load_pd(A + 4);   // Next 4 doubles
		}
		else
		{
			v0 = _mm256_loadu_pd(A);
			v1 = _mm256_loadu_pd(A + 4);
		}

		__m256d sum = _mm256_add_pd(v0, v1);

		// Horizontal add 4 doubles
		__m128d low = _mm256_castpd256_pd128(sum);
		__m128d high = _mm256_extractf128_pd(sum, 1);
		__m128d acc = _mm_add_pd(low, high);            // [a0+a4, a1+a5]

		__m128d tmp = _mm_unpackhi_pd(acc, acc);        // [a1+a5, a1+a5]
		__m128d final = _mm_add_sd(acc, tmp);           // (a0+a4) + (a1+a5)

		r = _mm_cvtsd_f64(final);
	}

	template <>
	void 
	_reduce_block_avx<double, std::multiplies<>>(double* A, double& r)
	{
		alignas(32)__m256d v0, v1;

		if (reinterpret_cast<uintptr_t>(A) % 32 == 0)
		{
			v0 = _mm256_load_pd(A);
			v1 = _mm256_load_pd(A + 4);
		}
		else
		{
			v0 = _mm256_loadu_pd(A);
			v1 = _mm256_loadu_pd(A + 4);
		}

		__m256d prod = _mm256_mul_pd(v0, v1);

		// Horizontal multiply of 4 doubles
		__m128d low = _mm256_castpd256_pd128(prod);
		__m128d high = _mm256_extractf128_pd(prod, 1);
		__m128d acc = _mm_mul_pd(low, high);           // [a0*a4, a1*a5]

		__m128d tmp = _mm_unpackhi_pd(acc, acc);       // [a1*a5, a1*a5]
		__m128d final = _mm_mul_sd(acc, tmp);          // (a0*a4) * (a1*a5)

		r = _mm_cvtsd_f64(final);
	}

	// template <>
	// void 
	// _reduce_block_avx<double, std::minus<>>(float* A, float& r)
	// {
	// 	static_assert(false, "minus reduce not implemented for AVX");
	// }

	// template <>
	// void 
	// _reduce_block_avx<double, std::divides<>>(float* A, float& r)
	// {
	// 	static_assert(false, "divide reduce not implemented for AVX");
	// }

/*AVX512*/
	template <>
	void 
	_reduce_block_avx512<float, std::plus<>>(float* A, float& r)
	{
		alignas(64)__m512 a;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_ps(A);
		}
		else
		{
			a = _mm512_loadu_ps(A);
		}

		r = _mm512_reduce_add_ps(a);
	}

	// template <>
	// void 
	// _reduce_block_avx512<float, std::minus<>>(float* A, float& r)
	// {
	// 	alignas(64)__m512 a;

	// 	if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
	// 	{
	// 		a = _mm512_load_ps(A);
	// 	}
	// 	else
	// 	{
	// 		a = _mm512_loadu_ps(A);
	// 	}

	// 	//force an aligned store to improve memory bandwidth
	// 	alignas(64) float b[16];

	// 	_mm512_store_ps(b, a);

	// 	r =	b[0] - b[1] - b[2] - b[3]
	// 		- b[4] - b[5] - b[6] - b[7]
	// 		- b[8] - b[9] - b[10] - b[11]
	// 		- b[12] - b[13] - b[14] - b[15];

	// }

	template <>
	void 
	_reduce_block_avx512<float, std::multiplies<>>(float* A, float& r)
	{
		alignas(64)__m512 a;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_ps(A);
		}
		else
		{
			a = _mm512_loadu_ps(A);
		}

		r = _mm512_reduce_mul_ps(a);
	}

	// template <>
	// void 
	// _reduce_block_avx512<float, std::divides<>>(float* A, float& r)
	// {
	// 	alignas(64)__m512 a;

	// 	if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
	// 	{
	// 		a = _mm512_load_ps(A);
	// 	}
	// 	else
	// 	{
	// 		a = _mm512_loadu_ps(A);
	// 	}

	// 	//force an aligned store to improve memory bandwidth
	// 	alignas(64) float b[16];

	// 	_mm512_store_ps(b, a);

	// 	r =	b[0] / b[1] / b[2] / b[3]
	// 		/ b[4] / b[5] / b[6] / b[7]
	// 		/ b[8] / b[9] / b[10] / b[11]
	// 		/ b[12] / b[13] / b[14] / b[15];
	// }

	template <>
	void 
	_reduce_block_avx512<double, std::plus<>>(double* A, double& r)
	{
		alignas(64)__m512d a;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_pd(A);
		}
		else
		{
			a = _mm512_loadu_pd(A);
		}

		r = _mm512_reduce_add_pd(a);
	}

	// template <>
	// void 
	// _reduce_block_avx512<double, std::minus<>>(double* A, double& r)
	// {
	// 	alignas(64)__m512d a;

	// 	if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
	// 	{
	// 		a = _mm512_load_pd(A);
	// 	}
	// 	else
	// 	{
	// 		a = _mm512_loadu_pd(A);
	// 	}

	// 	//force an aligned store to improve memory bandwidth
	// 	alignas(64) double b[8];

	// 	_mm512_store_pd(b, a);

	// 	r =	b[0] - b[1] - b[2] - b[3]
	// 		- b[4] - b[5] - b[6] - b[7];
	// }

	template <>
	void 
	_reduce_block_avx512<double, std::multiplies<>>(double* A, double& r)
	{
		alignas(64)__m512d a;

		if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
		{
			a = _mm512_load_pd(A);
		}
		else
		{
			a = _mm512_loadu_pd(A);
		}

		r = _mm512_reduce_mul_pd(a);
	}

	// template <>
	// void 
	// _reduce_block_avx512<double, std::divides<>>(double* A, double& r)
	// {
	// 	alignas(64)__m512d a;

	// 	if (reinterpret_cast<uintptr_t>(A) % 64 == 0)
	// 	{
	// 		a = _mm512_load_pd(A);
	// 	}
	// 	else
	// 	{
	// 		a = _mm512_loadu_pd(A);
	// 	}

	// 	//force an aligned store to improve memory bandwidth
	// 	alignas(64) double b[8];

	// 	_mm512_store_pd(b, a);

	// 	r =	b[0] / b[1] / b[2] / b[3]
	// 		/ b[4] / b[5] / b[6] / b[7];
	// }

} //namespace damm