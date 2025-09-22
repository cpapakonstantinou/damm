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

#include <macros.h>
#include <reduce.h>

#define SSE_KERNEL(T, REG_T, MAC_T, STD_OP, REDUCE_OP) \
	template <> \
	void _reduce_block_sse<T, std::STD_OP<>>(T* A, T& r) \
	{ \
		alignas(16) REG_T a0, a1, a2, a3; \
		_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1, a2, a3); \
		REDUCE_OP(reinterpret_cast<value<T>::type*>(&r), a0, a1, a2, a3) \
	}

#define AVX_KERNEL(T, REG_T, MAC_T, STD_OP, REDUCE_OP) \
	template <> \
	void _reduce_block_avx<T, std::STD_OP<>>(T* A, T& r) \
	{ \
		alignas(32) REG_T a0, a1; \
		_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1); \
		REDUCE_OP(reinterpret_cast<value<T>::type*>(&r), a0, a1) \
	}

#define AVX512_KERNEL(T, REG_T, MAC_T, STD_OP, REDUCE_OP) \
	template <> \
	void _reduce_block_avx512<T, std::STD_OP<>>(T* A, T& r) \
	{ \
		alignas(64) REG_T a0; \
		_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0); \
		REDUCE_OP(reinterpret_cast<value<T>::type*>(&r), a0) \
	}

//Alternative version that operates on NxN tiles
// #define AVX512_KERNEL(T, REG_T, MAC_T, STD_OP, REDUCE_OP) \
// 	template <> \
// 	void _reduce_block_avx512<T, std::STD_OP<>>(T* A, T& r) \
// 	{ \
// 		constexpr size_t N = static_cast<size_t>(S)/sizeof(T);
// 		alignas(64) REG_T[N] a; \
// 		static_for<N>([&]<auto i>()\
// 		{
// 			_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A)+i*sizeof(T), a[i]); \
// 		});
// 		REDUCE_OP(reinterpret_cast<value<T>::type*>(&r), a) \
// 	}

namespace damm
{
	SSE_KERNEL(float, __m128, PS, plus, _MM_REDUCE_ADD_LINE_PS)
	SSE_KERNEL(float, __m128, PS, multiplies, _MM_REDUCE_MUL_LINE_PS)
	SSE_KERNEL(double, __m128d, PD, plus, _MM_REDUCE_ADD_LINE_PD)
	SSE_KERNEL(double, __m128d, PD, multiplies, _MM_REDUCE_MUL_LINE_PD)
	SSE_KERNEL(std::complex<float>, __m128, PS, plus, _MM_REDUCE_ADD_LINE_C_PS)
	SSE_KERNEL(std::complex<float>, __m128, PS, multiplies, _MM_REDUCE_MUL_LINE_C_PS)
	SSE_KERNEL(std::complex<double>, __m128d, PD, plus, _MM_REDUCE_ADD_LINE_C_PD)
	SSE_KERNEL(std::complex<double>, __m128d, PD, multiplies, _MM_REDUCE_MUL_LINE_C_PD)

	AVX_KERNEL(float, __m256, PS, plus, _MM256_REDUCE_ADD_LINE_PS)
	AVX_KERNEL(float, __m256, PS, multiplies, _MM256_REDUCE_MUL_LINE_PS)
	AVX_KERNEL(double, __m256d, PD, plus, _MM256_REDUCE_ADD_LINE_PD)
	AVX_KERNEL(double, __m256d, PD, multiplies, _MM256_REDUCE_MUL_LINE_PD)
	AVX_KERNEL(std::complex<float>, __m256, PS, plus, _MM256_REDUCE_ADD_LINE_C_PS)
	AVX_KERNEL(std::complex<float>, __m256, PS, multiplies, _MM256_REDUCE_MUL_LINE_C_PS)
	AVX_KERNEL(std::complex<double>, __m256d, PD, plus, _MM256_REDUCE_ADD_LINE_C_PD)
	AVX_KERNEL(std::complex<double>, __m256d, PD, multiplies, _MM256_REDUCE_MUL_LINE_C_PD)

	AVX512_KERNEL(float, __m512, PS, plus, _MM512_REDUCE_ADD_LINE_PS)
	AVX512_KERNEL(float, __m512, PS, multiplies, _MM512_REDUCE_MUL_LINE_PS)
	AVX512_KERNEL(double, __m512d, PD, plus, _MM512_REDUCE_ADD_LINE_PD)
	AVX512_KERNEL(double, __m512d, PD, multiplies, _MM512_REDUCE_MUL_LINE_PD)
	AVX512_KERNEL(std::complex<float>, __m512, PS, plus, _MM512_REDUCE_ADD_LINE_C_PS)
	AVX512_KERNEL(std::complex<float>, __m512, PS, multiplies, _MM512_REDUCE_MUL_LINE_C_PS)
	AVX512_KERNEL(std::complex<double>, __m512d, PD, plus, _MM512_REDUCE_ADD_LINE_C_PD)
	AVX512_KERNEL(std::complex<double>, __m512d, PD, multiplies, _MM512_REDUCE_MUL_LINE_C_PD)
} //namespace damm

#undef SSE_KERNEL
#undef AVX_KERNEL
#undef AVX512_KERNEL