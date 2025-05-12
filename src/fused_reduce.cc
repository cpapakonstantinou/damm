/**
 * \file fused_reduce.cc
 * \brief fused reduce utilities implementations
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

#include <macros.h>
#include <fused_reduce.h>

#define SSE_KERNEL(T, REG_T, MAC_T, UNION_OP, REDUCE_OP, UNION_INTR, REDUCE_INTR) \
	template <> \
	void _fused_reduce_block_sse<T, std::UNION_OP<>, std::REDUCE_OP<>>(T* A, T* B, T& r) \
	{ \
		alignas(16) REG_T a0, a1, a2, a3; \
		alignas(16) REG_T b0, b1, b2, b3; \
		alignas(16) REG_T u0, u1, u2, u3; \
		_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1, a2, a3); \
		_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0, b1, b2, b3); \
		UNION_INTR(a0, a1, a2, a3, b0, b1, b2, b3, u0, u1, u2, u3); \
		REDUCE_INTR(reinterpret_cast<value<T>::type*>(&r), u0, u1, u2, u3); \
	}

#define AVX_KERNEL(T, REG_T, MAC_T, UNION_OP, REDUCE_OP, UNION_INTR, REDUCE_INTR) \
	template <> \
	void _fused_reduce_block_avx<T, std::UNION_OP<>, std::REDUCE_OP<>>(T* A, T* B, T& r) \
	{ \
		alignas(32) REG_T a0, a1; \
		alignas(32) REG_T b0, b1; \
		alignas(32) REG_T u0, u1; \
		_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1); \
		_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0, b1); \
		UNION_INTR(a0, a1, b0, b1, u0, u1); \
		REDUCE_INTR(reinterpret_cast<value<T>::type*>(&r), u0, u1); \
	}

#define AVX512_KERNEL(T, REG_T, MAC_T, UNION_OP, REDUCE_OP, UNION_INTR, REDUCE_INTR) \
	template <> \
	void _fused_reduce_block_avx512<T, std::UNION_OP<>, std::REDUCE_OP<>>(T* A, T* B, T& r) \
	{ \
		alignas(64) REG_T a, b, u; \
		_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a); \
		_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b); \
		UNION_INTR(a, b, u); \
		REDUCE_INTR(reinterpret_cast<value<T>::type*>(&r), u); \
	}

namespace damm
{


/*SSE*/

	SSE_KERNEL(float, __m128, PS, plus, plus, _MM_ADD_LINE_PS, _MM_REDUCE_ADD_LINE_PS)
	SSE_KERNEL(float, __m128, PS, minus, plus, _MM_SUB_LINE_PS, _MM_REDUCE_ADD_LINE_PS)
	SSE_KERNEL(float, __m128, PS, multiplies, plus, _MM_MUL_LINE_PS, _MM_REDUCE_ADD_LINE_PS)
	SSE_KERNEL(float, __m128, PS, divides, plus, _MM_DIV_LINE_PS, _MM_REDUCE_ADD_LINE_PS)

	SSE_KERNEL(float, __m128, PS, plus, multiplies, _MM_ADD_LINE_PS, _MM_REDUCE_MUL_LINE_PS)
	SSE_KERNEL(float, __m128, PS, minus, multiplies, _MM_SUB_LINE_PS, _MM_REDUCE_MUL_LINE_PS)
	SSE_KERNEL(float, __m128, PS, multiplies, multiplies, _MM_MUL_LINE_PS, _MM_REDUCE_MUL_LINE_PS)
	SSE_KERNEL(float, __m128, PS, divides, multiplies, _MM_DIV_LINE_PS, _MM_REDUCE_MUL_LINE_PS)

	SSE_KERNEL(double, __m128d, PD, plus, plus, _MM_ADD_LINE_PD, _MM_REDUCE_ADD_LINE_PD)
	SSE_KERNEL(double, __m128d, PD, minus, plus, _MM_SUB_LINE_PD, _MM_REDUCE_ADD_LINE_PD)
	SSE_KERNEL(double, __m128d, PD, multiplies, plus, _MM_MUL_LINE_PD, _MM_REDUCE_ADD_LINE_PD)
	SSE_KERNEL(double, __m128d, PD, divides, plus, _MM_DIV_LINE_PD, _MM_REDUCE_ADD_LINE_PD)

	SSE_KERNEL(double, __m128d, PD, plus, multiplies, _MM_ADD_LINE_PD, _MM_REDUCE_MUL_LINE_PD)
	SSE_KERNEL(double, __m128d, PD, minus, multiplies, _MM_SUB_LINE_PD, _MM_REDUCE_MUL_LINE_PD)
	SSE_KERNEL(double, __m128d, PD, multiplies, multiplies, _MM_MUL_LINE_PD, _MM_REDUCE_MUL_LINE_PD)
	SSE_KERNEL(double, __m128d, PD, divides, multiplies, _MM_DIV_LINE_PD, _MM_REDUCE_MUL_LINE_PD)

	SSE_KERNEL(std::complex<float>, __m128, PS, plus, plus, _MM_ADD_LINE_PS, _MM_REDUCE_ADD_LINE_C_PS)
	SSE_KERNEL(std::complex<float>, __m128, PS, minus, plus, _MM_SUB_LINE_PS, _MM_REDUCE_ADD_LINE_C_PS)
	SSE_KERNEL(std::complex<float>, __m128, PS, multiplies, plus, _MM_MUL_LINE_C_PS, _MM_REDUCE_ADD_LINE_C_PS)
	SSE_KERNEL(std::complex<float>, __m128, PS, divides, plus, _MM_DIV_LINE_C_PS, _MM_REDUCE_ADD_LINE_C_PS)

	SSE_KERNEL(std::complex<float>, __m128, PS, plus, multiplies, _MM_ADD_LINE_PS, _MM_REDUCE_MUL_LINE_C_PS)
	SSE_KERNEL(std::complex<float>, __m128, PS, minus, multiplies, _MM_SUB_LINE_PS, _MM_REDUCE_MUL_LINE_C_PS)
	SSE_KERNEL(std::complex<float>, __m128, PS, multiplies, multiplies, _MM_MUL_LINE_C_PS, _MM_REDUCE_MUL_LINE_C_PS)
	SSE_KERNEL(std::complex<float>, __m128, PS, divides, multiplies, _MM_DIV_LINE_C_PS, _MM_REDUCE_MUL_LINE_C_PS)

	SSE_KERNEL(std::complex<double>, __m128d, PD, plus, plus, _MM_ADD_LINE_PD, _MM_REDUCE_ADD_LINE_C_PD)
	SSE_KERNEL(std::complex<double>, __m128d, PD, minus, plus, _MM_SUB_LINE_PD, _MM_REDUCE_ADD_LINE_C_PD)
	SSE_KERNEL(std::complex<double>, __m128d, PD, multiplies, plus, _MM_MUL_LINE_C_PD, _MM_REDUCE_ADD_LINE_C_PD)
	SSE_KERNEL(std::complex<double>, __m128d, PD, divides, plus, _MM_DIV_LINE_C_PD, _MM_REDUCE_ADD_LINE_C_PD)

	SSE_KERNEL(std::complex<double>, __m128d, PD, plus, multiplies, _MM_ADD_LINE_PD, _MM_REDUCE_MUL_LINE_C_PD)
	SSE_KERNEL(std::complex<double>, __m128d, PD, minus, multiplies, _MM_SUB_LINE_PD, _MM_REDUCE_MUL_LINE_C_PD)
	SSE_KERNEL(std::complex<double>, __m128d, PD, multiplies, multiplies, _MM_MUL_LINE_C_PD, _MM_REDUCE_MUL_LINE_C_PD)
	SSE_KERNEL(std::complex<double>, __m128d, PD, divides, multiplies, _MM_DIV_LINE_C_PD, _MM_REDUCE_MUL_LINE_C_PD)

/*AVX*/

	AVX_KERNEL(float, __m256, PS, plus, plus, _MM256_ADD_LINE_PS, _MM256_REDUCE_ADD_LINE_PS)
	AVX_KERNEL(float, __m256, PS, minus, plus, _MM256_SUB_LINE_PS, _MM256_REDUCE_ADD_LINE_PS)
	AVX_KERNEL(float, __m256, PS, multiplies, plus, _MM256_MUL_LINE_PS, _MM256_REDUCE_ADD_LINE_PS)
	AVX_KERNEL(float, __m256, PS, divides, plus, _MM256_DIV_LINE_PS, _MM256_REDUCE_ADD_LINE_PS)

	AVX_KERNEL(float, __m256, PS, plus, multiplies, _MM256_ADD_LINE_PS, _MM256_REDUCE_MUL_LINE_PS)
	AVX_KERNEL(float, __m256, PS, minus, multiplies, _MM256_SUB_LINE_PS, _MM256_REDUCE_MUL_LINE_PS)
	AVX_KERNEL(float, __m256, PS, multiplies, multiplies, _MM256_MUL_LINE_PS, _MM256_REDUCE_MUL_LINE_PS)
	AVX_KERNEL(float, __m256, PS, divides, multiplies, _MM256_DIV_LINE_PS, _MM256_REDUCE_MUL_LINE_PS)

	AVX_KERNEL(double, __m256d, PD, plus, plus, _MM256_ADD_LINE_PD, _MM256_REDUCE_ADD_LINE_PD)
	AVX_KERNEL(double, __m256d, PD, minus, plus, _MM256_SUB_LINE_PD, _MM256_REDUCE_ADD_LINE_PD)
	AVX_KERNEL(double, __m256d, PD, multiplies, plus, _MM256_MUL_LINE_PD, _MM256_REDUCE_ADD_LINE_PD)
	AVX_KERNEL(double, __m256d, PD, divides, plus, _MM256_DIV_LINE_PD, _MM256_REDUCE_ADD_LINE_PD)

	AVX_KERNEL(double, __m256d, PD, plus, multiplies, _MM256_ADD_LINE_PD, _MM256_REDUCE_MUL_LINE_PD)
	AVX_KERNEL(double, __m256d, PD, minus, multiplies, _MM256_SUB_LINE_PD, _MM256_REDUCE_MUL_LINE_PD)
	AVX_KERNEL(double, __m256d, PD, multiplies, multiplies, _MM256_MUL_LINE_PD, _MM256_REDUCE_MUL_LINE_PD)
	AVX_KERNEL(double, __m256d, PD, divides, multiplies, _MM256_DIV_LINE_PD, _MM256_REDUCE_MUL_LINE_PD)

	AVX_KERNEL(std::complex<float>, __m256, PS, plus, plus, _MM256_ADD_LINE_PS, _MM256_REDUCE_ADD_LINE_C_PS)
	AVX_KERNEL(std::complex<float>, __m256, PS, minus, plus, _MM256_SUB_LINE_PS, _MM256_REDUCE_ADD_LINE_C_PS)
	AVX_KERNEL(std::complex<float>, __m256, PS, multiplies, plus, _MM256_MUL_LINE_C_PS, _MM256_REDUCE_ADD_LINE_C_PS)
	AVX_KERNEL(std::complex<float>, __m256, PS, divides, plus, _MM256_DIV_LINE_C_PS, _MM256_REDUCE_ADD_LINE_C_PS)

	AVX_KERNEL(std::complex<float>, __m256, PS, plus, multiplies, _MM256_ADD_LINE_PS, _MM256_REDUCE_MUL_LINE_C_PS)
	AVX_KERNEL(std::complex<float>, __m256, PS, minus, multiplies, _MM256_SUB_LINE_PS, _MM256_REDUCE_MUL_LINE_C_PS)
	AVX_KERNEL(std::complex<float>, __m256, PS, multiplies, multiplies, _MM256_MUL_LINE_C_PS, _MM256_REDUCE_MUL_LINE_C_PS)
	AVX_KERNEL(std::complex<float>, __m256, PS, divides, multiplies, _MM256_DIV_LINE_C_PS, _MM256_REDUCE_MUL_LINE_C_PS)

	AVX_KERNEL(std::complex<double>, __m256d, PD, plus, plus, _MM256_ADD_LINE_PD, _MM256_REDUCE_ADD_LINE_C_PD)
	AVX_KERNEL(std::complex<double>, __m256d, PD, minus, plus, _MM256_SUB_LINE_PD, _MM256_REDUCE_ADD_LINE_C_PD)
	AVX_KERNEL(std::complex<double>, __m256d, PD, multiplies, plus, _MM256_MUL_LINE_C_PD, _MM256_REDUCE_ADD_LINE_C_PD)
	AVX_KERNEL(std::complex<double>, __m256d, PD, divides, plus, _MM256_DIV_LINE_C_PD, _MM256_REDUCE_ADD_LINE_C_PD)

	AVX_KERNEL(std::complex<double>, __m256d, PD, plus, multiplies, _MM256_ADD_LINE_PD, _MM256_REDUCE_MUL_LINE_C_PD)
	AVX_KERNEL(std::complex<double>, __m256d, PD, minus, multiplies, _MM256_SUB_LINE_PD, _MM256_REDUCE_MUL_LINE_C_PD)
	AVX_KERNEL(std::complex<double>, __m256d, PD, multiplies, multiplies, _MM256_MUL_LINE_C_PD, _MM256_REDUCE_MUL_LINE_C_PD)
	AVX_KERNEL(std::complex<double>, __m256d, PD, divides, multiplies, _MM256_DIV_LINE_C_PD, _MM256_REDUCE_MUL_LINE_C_PD)

/*AVX512*/

	AVX512_KERNEL(float, __m512, PS, plus, plus, _MM512_ADD_LINE_PS, _MM512_REDUCE_ADD_LINE_PS)
	AVX512_KERNEL(float, __m512, PS, minus, plus, _MM512_SUB_LINE_PS, _MM512_REDUCE_ADD_LINE_PS)
	AVX512_KERNEL(float, __m512, PS, multiplies, plus, _MM512_MUL_LINE_PS, _MM512_REDUCE_ADD_LINE_PS)
	AVX512_KERNEL(float, __m512, PS, divides, plus, _MM512_DIV_LINE_PS, _MM512_REDUCE_ADD_LINE_PS)

	AVX512_KERNEL(float, __m512, PS, plus, multiplies, _MM512_ADD_LINE_PS, _MM512_REDUCE_MUL_LINE_PS)
	AVX512_KERNEL(float, __m512, PS, minus, multiplies, _MM512_SUB_LINE_PS, _MM512_REDUCE_MUL_LINE_PS)
	AVX512_KERNEL(float, __m512, PS, multiplies, multiplies, _MM512_MUL_LINE_PS, _MM512_REDUCE_MUL_LINE_PS)
	AVX512_KERNEL(float, __m512, PS, divides, multiplies, _MM512_DIV_LINE_PS, _MM512_REDUCE_MUL_LINE_PS)

	AVX512_KERNEL(double, __m512d, PD, plus, plus, _MM512_ADD_LINE_PD, _MM512_REDUCE_ADD_LINE_PD)
	AVX512_KERNEL(double, __m512d, PD, minus, plus, _MM512_SUB_LINE_PD, _MM512_REDUCE_ADD_LINE_PD)
	AVX512_KERNEL(double, __m512d, PD, multiplies, plus, _MM512_MUL_LINE_PD, _MM512_REDUCE_ADD_LINE_PD)
	AVX512_KERNEL(double, __m512d, PD, divides, plus, _MM512_DIV_LINE_PD, _MM512_REDUCE_ADD_LINE_PD)

	AVX512_KERNEL(double, __m512d, PD, plus, multiplies, _MM512_ADD_LINE_PD, _MM512_REDUCE_MUL_LINE_PD)
	AVX512_KERNEL(double, __m512d, PD, minus, multiplies, _MM512_SUB_LINE_PD, _MM512_REDUCE_MUL_LINE_PD)
	AVX512_KERNEL(double, __m512d, PD, multiplies, multiplies, _MM512_MUL_LINE_PD, _MM512_REDUCE_MUL_LINE_PD)
	AVX512_KERNEL(double, __m512d, PD, divides, multiplies, _MM512_DIV_LINE_PD, _MM512_REDUCE_MUL_LINE_PD)

	AVX512_KERNEL(std::complex<float>, __m512, PS, plus, plus, _MM512_ADD_LINE_PS, _MM512_REDUCE_ADD_LINE_C_PS)
	AVX512_KERNEL(std::complex<float>, __m512, PS, minus, plus, _MM512_SUB_LINE_PS, _MM512_REDUCE_ADD_LINE_C_PS)
	AVX512_KERNEL(std::complex<float>, __m512, PS, multiplies, plus, _MM512_MUL_LINE_C_PS, _MM512_REDUCE_ADD_LINE_C_PS)
	AVX512_KERNEL(std::complex<float>, __m512, PS, divides, plus, _MM512_DIV_LINE_C_PS, _MM512_REDUCE_ADD_LINE_C_PS)

	AVX512_KERNEL(std::complex<float>, __m512, PS, plus, multiplies, _MM512_ADD_LINE_PS, _MM512_REDUCE_MUL_LINE_C_PS)
	AVX512_KERNEL(std::complex<float>, __m512, PS, minus, multiplies, _MM512_SUB_LINE_PS, _MM512_REDUCE_MUL_LINE_C_PS)
	AVX512_KERNEL(std::complex<float>, __m512, PS, multiplies, multiplies, _MM512_MUL_LINE_C_PS, _MM512_REDUCE_MUL_LINE_C_PS)
	AVX512_KERNEL(std::complex<float>, __m512, PS, divides, multiplies, _MM512_DIV_LINE_C_PS, _MM512_REDUCE_MUL_LINE_C_PS)

	AVX512_KERNEL(std::complex<double>, __m512d, PD, plus, plus, _MM512_ADD_LINE_PD, _MM512_REDUCE_ADD_LINE_C_PD)
	AVX512_KERNEL(std::complex<double>, __m512d, PD, minus, plus, _MM512_SUB_LINE_PD, _MM512_REDUCE_ADD_LINE_C_PD)
	AVX512_KERNEL(std::complex<double>, __m512d, PD, multiplies, plus, _MM512_MUL_LINE_C_PD, _MM512_REDUCE_ADD_LINE_C_PD)
	AVX512_KERNEL(std::complex<double>, __m512d, PD, divides, plus, _MM512_DIV_LINE_C_PD, _MM512_REDUCE_ADD_LINE_C_PD)

	AVX512_KERNEL(std::complex<double>, __m512d, PD, plus, multiplies, _MM512_ADD_LINE_PD, _MM512_REDUCE_MUL_LINE_C_PD)
	AVX512_KERNEL(std::complex<double>, __m512d, PD, minus, multiplies, _MM512_SUB_LINE_PD, _MM512_REDUCE_MUL_LINE_C_PD)
	AVX512_KERNEL(std::complex<double>, __m512d, PD, multiplies, multiplies, _MM512_MUL_LINE_C_PD, _MM512_REDUCE_MUL_LINE_C_PD)
	AVX512_KERNEL(std::complex<double>, __m512d, PD, divides, multiplies, _MM512_DIV_LINE_C_PD, _MM512_REDUCE_MUL_LINE_C_PD)

} //namespace damm

#undef SSE_KERNEL
#undef AVX_KERNEL
#undef AVX512_KERNEL