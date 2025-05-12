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
#include <macros.h>
#include <complex>

namespace damm
{
	#define SCALAR_SSE_KERNEL(T, REG_T, MAC_T, INTR_OP, STD_OP) \
		template <> \
		void _union_block_sse<T, std::STD_OP<>>(T* A, const T B, T* C) \
		{ \
			alignas(16) REG_T a0, a1, a2, a3; \
			REG_T c0, c1, c2, c3; \
			_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1, a2, a3); \
			INTR_OP(a0, a1, a2, a3, reinterpret_cast<const value<T>::type*>(&B), c0, c1, c2, c3) \
			_MM_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(C), c0, c1, c2, c3); \
		}

	#define SCALAR_AVX_KERNEL(T, REG_T, MAC_T, INTR_OP, STD_OP) \
		template <> \
		void _union_block_avx<T, std::STD_OP<>>(T* A, const T B, T* C) \
		{ \
			alignas(32) REG_T a0, a1; \
			REG_T c0, c1; \
			_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1); \
			INTR_OP(a0, a1, reinterpret_cast<const value<T>::type*>(&B), c0, c1) \
			_MM256_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(C), c0, c1); \
		}

	#define SCALAR_AVX512_KERNEL(T, REG_T, MAC_T, INTR_OP, STD_OP) \
		template <> \
		void _union_block_avx512<T, std::STD_OP<>>(T* A, const T B, T* C) \
		{ \
			alignas(64) REG_T a0; \
			REG_T c0; \
			_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0); \
			INTR_OP(a0, reinterpret_cast<const value<T>::type*>(&B), c0) \
			_MM512_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(C), c0); \
		}

	#define MATRIX_SSE_KERNEL(T, REG_T, MAC_T, INTR_OP, STD_OP) \
		template <> \
		void _union_block_sse<T, std::STD_OP<>>(T* A, T* B, T* C) \
		{ \
			alignas(16) REG_T b0, b1, b2, b3; \
			alignas(16) REG_T a0, a1, a2, a3; \
			REG_T c0, c1, c2, c3; \
			_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1, a2, a3); \
			_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0, b1, b2, b3); \
			INTR_OP(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) \
			_MM_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(C), c0, c1, c2, c3); \
		}

	#define MATRIX_AVX_KERNEL(T, REG_T, MAC_T, INTR_OP, STD_OP) \
		template <> \
		void _union_block_avx<T, std::STD_OP<>>(T* A, T* B, T* C) \
		{ \
			alignas(32) REG_T a0, a1; \
			alignas(32) REG_T b0, b1; \
			REG_T c0, c1; \
			_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1); \
			_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0, b1); \
			INTR_OP(a0, a1, b0, b1, c0, c1) \
			_MM256_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(C), c0, c1); \
		}

	#define MATRIX_AVX512_KERNEL(T, REG_T, MAC_T, INTR_OP, STD_OP) \
		template <> \
		void _union_block_avx512<T, std::STD_OP<>>(T* A, T* B, T* C) \
		{ \
			alignas(64) REG_T a0; \
			alignas(64) REG_T b0; \
			REG_T c0; \
			_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0); \
			_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0); \
			INTR_OP(a0, b0, c0) \
			_MM512_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(C), c0); \
		}

	namespace scalar 
	{
		SCALAR_SSE_KERNEL(float, __m128, PS, _MM_ADD_LINE_S_PS, plus)
		SCALAR_SSE_KERNEL(float, __m128, PS, _MM_SUB_LINE_S_PS, minus)
		SCALAR_SSE_KERNEL(float, __m128, PS, _MM_MUL_LINE_S_PS, multiplies)
		SCALAR_SSE_KERNEL(float, __m128, PS, _MM_DIV_LINE_S_PS, divides)

		SCALAR_AVX_KERNEL(float, __m256, PS, _MM256_ADD_LINE_S_PS, plus)
		SCALAR_AVX_KERNEL(float, __m256, PS, _MM256_SUB_LINE_S_PS, minus)
		SCALAR_AVX_KERNEL(float, __m256, PS, _MM256_MUL_LINE_S_PS, multiplies)
		SCALAR_AVX_KERNEL(float, __m256, PS, _MM256_DIV_LINE_S_PS, divides)

		SCALAR_AVX512_KERNEL(float, __m512, PS, _MM512_ADD_LINE_S_PS, plus)
		SCALAR_AVX512_KERNEL(float, __m512, PS, _MM512_SUB_LINE_S_PS, minus)
		SCALAR_AVX512_KERNEL(float, __m512, PS, _MM512_MUL_LINE_S_PS, multiplies)
		SCALAR_AVX512_KERNEL(float, __m512, PS, _MM512_DIV_LINE_S_PS, divides)

		SCALAR_SSE_KERNEL(double, __m128d, PD, _MM_ADD_LINE_S_PD, plus)
		SCALAR_SSE_KERNEL(double, __m128d, PD, _MM_SUB_LINE_S_PD, minus)
		SCALAR_SSE_KERNEL(double, __m128d, PD, _MM_MUL_LINE_S_PD, multiplies)
		SCALAR_SSE_KERNEL(double, __m128d, PD, _MM_DIV_LINE_S_PD, divides)

		SCALAR_AVX_KERNEL(double, __m256d, PD, _MM256_ADD_LINE_S_PD, plus)
		SCALAR_AVX_KERNEL(double, __m256d, PD, _MM256_SUB_LINE_S_PD, minus)
		SCALAR_AVX_KERNEL(double, __m256d, PD, _MM256_MUL_LINE_S_PD, multiplies)
		SCALAR_AVX_KERNEL(double, __m256d, PD, _MM256_DIV_LINE_S_PD, divides)

		SCALAR_AVX512_KERNEL(double, __m512d, PD, _MM512_ADD_LINE_S_PD, plus)
		SCALAR_AVX512_KERNEL(double, __m512d, PD, _MM512_SUB_LINE_S_PD, minus)
		SCALAR_AVX512_KERNEL(double, __m512d, PD, _MM512_MUL_LINE_S_PD, multiplies)
		SCALAR_AVX512_KERNEL(double, __m512d, PD, _MM512_DIV_LINE_S_PD, divides)

		SCALAR_SSE_KERNEL(std::complex<float>, __m128, PS, _MM_ADD_LINE_SC_PS, plus)
		SCALAR_SSE_KERNEL(std::complex<float>, __m128, PS, _MM_SUB_LINE_SC_PS, minus)
		SCALAR_SSE_KERNEL(std::complex<float>, __m128, PS, _MM_MUL_LINE_SC_PS, multiplies)
		SCALAR_SSE_KERNEL(std::complex<float>, __m128, PS, _MM_DIV_LINE_SC_PS, divides)

		SCALAR_AVX_KERNEL(std::complex<float>, __m256, PS, _MM256_ADD_LINE_SC_PS, plus)
		SCALAR_AVX_KERNEL(std::complex<float>, __m256, PS, _MM256_SUB_LINE_SC_PS, minus)
		SCALAR_AVX_KERNEL(std::complex<float>, __m256, PS, _MM256_MUL_LINE_SC_PS, multiplies)
		SCALAR_AVX_KERNEL(std::complex<float>, __m256, PS, _MM256_DIV_LINE_SC_PS, divides)

		SCALAR_AVX512_KERNEL(std::complex<float>, __m512, PS, _MM512_ADD_LINE_SC_PS, plus)
		SCALAR_AVX512_KERNEL(std::complex<float>, __m512, PS, _MM512_SUB_LINE_SC_PS, minus)
		SCALAR_AVX512_KERNEL(std::complex<float>, __m512, PS, _MM512_MUL_LINE_SC_PS, multiplies)
		SCALAR_AVX512_KERNEL(std::complex<float>, __m512, PS, _MM512_DIV_LINE_SC_PS, divides)

		SCALAR_SSE_KERNEL(std::complex<double>, __m128d, PD, _MM_ADD_LINE_SC_PD, plus)
		SCALAR_SSE_KERNEL(std::complex<double>, __m128d, PD, _MM_SUB_LINE_SC_PD, minus)
		SCALAR_SSE_KERNEL(std::complex<double>, __m128d, PD, _MM_MUL_LINE_SC_PD, multiplies)
		SCALAR_SSE_KERNEL(std::complex<double>, __m128d, PD, _MM_DIV_LINE_SC_PD, divides)

		SCALAR_AVX_KERNEL(std::complex<double>, __m256d, PD, _MM256_ADD_LINE_SC_PD, plus)
		SCALAR_AVX_KERNEL(std::complex<double>, __m256d, PD, _MM256_SUB_LINE_SC_PD, minus)
		SCALAR_AVX_KERNEL(std::complex<double>, __m256d, PD, _MM256_MUL_LINE_SC_PD, multiplies)
		SCALAR_AVX_KERNEL(std::complex<double>, __m256d, PD, _MM256_DIV_LINE_SC_PD, divides)

		SCALAR_AVX512_KERNEL(std::complex<double>, __m512d, PD, _MM512_ADD_LINE_SC_PD, plus)
		SCALAR_AVX512_KERNEL(std::complex<double>, __m512d, PD, _MM512_SUB_LINE_SC_PD, minus)
		SCALAR_AVX512_KERNEL(std::complex<double>, __m512d, PD, _MM512_MUL_LINE_SC_PD, multiplies)
		SCALAR_AVX512_KERNEL(std::complex<double>, __m512d, PD, _MM512_DIV_LINE_SC_PD, divides)

	} // namespace scalar

	namespace matrix 
	{
		MATRIX_SSE_KERNEL(float, __m128, PS, _MM_ADD_LINE_PS, plus)
		MATRIX_SSE_KERNEL(float, __m128, PS, _MM_SUB_LINE_PS, minus)
		MATRIX_SSE_KERNEL(float, __m128, PS, _MM_MUL_LINE_PS, multiplies)
		MATRIX_SSE_KERNEL(float, __m128, PS, _MM_DIV_LINE_PS, divides)

		MATRIX_AVX_KERNEL(float, __m256, PS, _MM256_ADD_LINE_PS, plus)
		MATRIX_AVX_KERNEL(float, __m256, PS, _MM256_SUB_LINE_PS, minus)
		MATRIX_AVX_KERNEL(float, __m256, PS, _MM256_MUL_LINE_PS, multiplies)
		MATRIX_AVX_KERNEL(float, __m256, PS, _MM256_DIV_LINE_PS, divides)

		MATRIX_AVX512_KERNEL(float, __m512, PS, _MM512_ADD_LINE_PS, plus)
		MATRIX_AVX512_KERNEL(float, __m512, PS, _MM512_SUB_LINE_PS, minus)
		MATRIX_AVX512_KERNEL(float, __m512, PS, _MM512_MUL_LINE_PS, multiplies)
		MATRIX_AVX512_KERNEL(float, __m512, PS, _MM512_DIV_LINE_PS, divides)

		MATRIX_SSE_KERNEL(double, __m128d, PD, _MM_ADD_LINE_PD, plus)
		MATRIX_SSE_KERNEL(double, __m128d, PD, _MM_SUB_LINE_PD, minus)
		MATRIX_SSE_KERNEL(double, __m128d, PD, _MM_MUL_LINE_PD, multiplies)
		MATRIX_SSE_KERNEL(double, __m128d, PD, _MM_DIV_LINE_PD, divides)

		MATRIX_AVX_KERNEL(double, __m256d, PD, _MM256_ADD_LINE_PD, plus)
		MATRIX_AVX_KERNEL(double, __m256d, PD, _MM256_SUB_LINE_PD, minus)
		MATRIX_AVX_KERNEL(double, __m256d, PD, _MM256_MUL_LINE_PD, multiplies)
		MATRIX_AVX_KERNEL(double, __m256d, PD, _MM256_DIV_LINE_PD, divides)

		MATRIX_AVX512_KERNEL(double, __m512d, PD, _MM512_ADD_LINE_PD, plus)
		MATRIX_AVX512_KERNEL(double, __m512d, PD, _MM512_SUB_LINE_PD, minus)
		MATRIX_AVX512_KERNEL(double, __m512d, PD, _MM512_MUL_LINE_PD, multiplies)
		MATRIX_AVX512_KERNEL(double, __m512d, PD, _MM512_DIV_LINE_PD, divides)

		MATRIX_SSE_KERNEL(std::complex<float>, __m128, PS, _MM_ADD_LINE_PS, plus)
		MATRIX_SSE_KERNEL(std::complex<float>, __m128, PS, _MM_SUB_LINE_PS, minus)
		MATRIX_SSE_KERNEL(std::complex<float>, __m128, PS, _MM_MUL_LINE_C_PS, multiplies)
		MATRIX_SSE_KERNEL(std::complex<float>, __m128, PS, _MM_DIV_LINE_C_PS, divides)

		MATRIX_AVX_KERNEL(std::complex<float>, __m256, PS, _MM256_ADD_LINE_PS, plus)
		MATRIX_AVX_KERNEL(std::complex<float>, __m256, PS, _MM256_SUB_LINE_PS, minus)
		MATRIX_AVX_KERNEL(std::complex<float>, __m256, PS, _MM256_MUL_LINE_C_PS, multiplies)
		MATRIX_AVX_KERNEL(std::complex<float>, __m256, PS, _MM256_DIV_LINE_C_PS, divides)

		MATRIX_AVX512_KERNEL(std::complex<float>, __m512, PS, _MM512_ADD_LINE_PS, plus)
		MATRIX_AVX512_KERNEL(std::complex<float>, __m512, PS, _MM512_SUB_LINE_PS, minus)
		MATRIX_AVX512_KERNEL(std::complex<float>, __m512, PS, _MM512_MUL_LINE_C_PS, multiplies)
		MATRIX_AVX512_KERNEL(std::complex<float>, __m512, PS, _MM512_DIV_LINE_C_PS, divides)

		MATRIX_SSE_KERNEL(std::complex<double>, __m128d, PD, _MM_ADD_LINE_PD, plus)
		MATRIX_SSE_KERNEL(std::complex<double>, __m128d, PD, _MM_SUB_LINE_PD, minus)
		MATRIX_SSE_KERNEL(std::complex<double>, __m128d, PD, _MM_MUL_LINE_C_PD, multiplies)
		MATRIX_SSE_KERNEL(std::complex<double>, __m128d, PD, _MM_DIV_LINE_C_PD, divides)

		MATRIX_AVX_KERNEL(std::complex<double>, __m256d, PD, _MM256_ADD_LINE_PD, plus)
		MATRIX_AVX_KERNEL(std::complex<double>, __m256d, PD, _MM256_SUB_LINE_PD, minus)
		MATRIX_AVX_KERNEL(std::complex<double>, __m256d, PD, _MM256_MUL_LINE_C_PD, multiplies)
		MATRIX_AVX_KERNEL(std::complex<double>, __m256d, PD, _MM256_DIV_LINE_C_PD, divides)

		MATRIX_AVX512_KERNEL(std::complex<double>, __m512d, PD, _MM512_ADD_LINE_PD, plus)
		MATRIX_AVX512_KERNEL(std::complex<double>, __m512d, PD, _MM512_SUB_LINE_PD, minus)
		MATRIX_AVX512_KERNEL(std::complex<double>, __m512d, PD, _MM512_MUL_LINE_C_PD, multiplies)
		MATRIX_AVX512_KERNEL(std::complex<double>, __m512d, PD, _MM512_DIV_LINE_C_PD, divides)

	} // namespace matrix

	#undef SCALAR_SSE_KERNEL
	#undef SCALAR_AVX_KERNEL
	#undef SCALAR_AVX512_KERNEL
	#undef MATRIX_SSE_KERNEL
	#undef MATRIX_AVX_KERNEL
	#undef MATRIX_AVX512_KERNEL
} // namespace damm