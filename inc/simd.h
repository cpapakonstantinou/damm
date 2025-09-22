#ifndef __SIMD_H__
#define __SIMD_H__
/**
 * \file common.h
 * \brief common definitions for libmm 
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

#include <immintrin.h>
#include <common.h>

namespace damm
{
	/**
	 * \brief The register size in bytes for a given SIMD architecture.
	 * 
	 *  This type is used to instantiate / branch templates for the SIMD architecture.
	 **/ 
	#pragma GCC diagnostic push	
	#pragma GCC diagnostic ignored "-Wignored-attributes"
	template<size_t N>
	struct simd {
		static constexpr size_t bytes = N;
		template<typename T>
		static constexpr size_t elements() { return N / sizeof(T); }
		template<typename T>
		using register_t = std::conditional_t< \
			std::is_same_v<typename base<T>::type, float>,
			// float:
			std::conditional_t<N == 16, __m128,
			std::conditional_t<N == 32, __m256, __m512>>,
			// double:
			std::conditional_t<N == 16, __m128d,
			std::conditional_t<N == 32, __m256d, __m512d>>
		>;
	};
	#pragma GCC diagnostic pop

	using NONE = simd<8>;
	using SSE = simd<16>;
	using AVX = simd<32>;
	using AVX512 = simd<64>;

	// Compile-time SIMD detection
	consteval auto detect_simd() 
	{
		#ifdef __AVX512F__
			return AVX512{};
		#elif defined(__AVX__)
			return AVX{};
		#elif defined(__SSE__)
			return SSE{};
		#else
			return NONE{};
		#endif
	}

	template<typename T, typename S> 
	void load(T* ptr, typename S::template register_t<T>* rows, const size_t stride);

	template<typename T, typename S> 
	void store(T* ptr, typename S::template register_t<T>* rows, const size_t stride);
}
#endif //__SIMD_H__