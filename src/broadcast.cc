/**
 * \file broadcast.cc
 * \brief broadcast utilities implementations
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
#include <broadcast.h>
#include <common.h>

namespace damm
{
	#define SSE_KERNEL(T, REG_T, SET_REGISTER, STORE_LINE) \
		template <> \
		void _broadcast_block_sse<T>(T* A, const T B) \
		{ \
			alignas(16) REG_T b; \
			SET_REGISTER(reinterpret_cast<const value<T>::type*>(&B), b) \
			STORE_LINE(reinterpret_cast<value<T>::type*>(A), b, b, b, b); \
		}

	#define AVX_KERNEL(T, REG_T, SET_REGISTER, STORE_LINE) \
		template <> \
		void _broadcast_block_avx<T>(T* A, const T B) \
		{ \
			alignas(32) REG_T b; \
			SET_REGISTER(reinterpret_cast<const value<T>::type*>(&B), b) \
			STORE_LINE(reinterpret_cast<value<T>::type*>(A), b, b); \
		}

	#define AVX512_KERNEL(T, REG_T, SET_REGISTER, STORE_LINE) \
		template <> \
		void _broadcast_block_avx512<T>(T* A, const T B) \
		{ \
			alignas(64) REG_T b; \
			SET_REGISTER(reinterpret_cast<const value<T>::type*>(&B), b); \
			STORE_LINE(reinterpret_cast<value<T>::type*>(A), b); \
		}
	
	SSE_KERNEL(float, __m128, _MM_SET1_PS, _MM_STORE_LINE_PS)
	AVX_KERNEL(float, __m256 , _MM256_SET1_PS, _MM256_STORE_LINE_PS)
	AVX512_KERNEL(float, __m512, _MM512_SET1_PS, _MM512_STORE_LINE_PS)
	
	SSE_KERNEL(double, __m128d, _MM_SET1_PD, _MM_STORE_LINE_PD)
	AVX_KERNEL(double, __m256d, _MM256_SET1_PD, _MM256_STORE_LINE_PD)
	AVX512_KERNEL(double, __m512d, _MM512_SET1_PD, _MM512_STORE_LINE_PD)

	SSE_KERNEL(std::complex<float>, __m128, _MM_SETR_PS, _MM_STORE_LINE_PS)
	AVX_KERNEL(std::complex<float>, __m256, _MM256_SETR_PS, _MM256_STORE_LINE_PS)
	AVX512_KERNEL(std::complex<float>, __m512, _MM512_SETR_PS, _MM512_STORE_LINE_PS)
	
	SSE_KERNEL(std::complex<double>, __m128d, _MM_SETR_PD, _MM_STORE_LINE_PD)
	AVX_KERNEL(std::complex<double>, __m256d, _MM256_SETR_PD, _MM256_STORE_LINE_PD)
	AVX512_KERNEL(std::complex<double>, __m512d, _MM512_SETR_PD, _MM512_STORE_LINE_PD)

	#undef SSE_KERNEL
	#undef AVX_KERNEL
	#undef AVX512_KERNEL

} // namespace damm