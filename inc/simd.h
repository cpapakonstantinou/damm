#ifndef __SIMD_H__
#define __SIMD_H__
/**
 * \file simd.h
 * \brief definitions for simd  
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
	 * \brief Representation of the information associated with a SIMD architecture.
	 * 
	 * \tparam	N	The register size in bytes for a given SIMD architecture.
	 * 
	 *  This type is used to inform algorithms working with a specific the SIMD architecture.
	 **/ 
	#pragma GCC diagnostic push	
	#pragma GCC diagnostic ignored "-Wignored-attributes"
	template<size_t N>
	struct simd 
	{
		static constexpr size_t bytes = N; ///< the size of the register in bytes
		template<typename T>
		static consteval size_t elements() { return N / sizeof(T); } ///< the number of elements of type T that can occupy the register
		template<typename T>
		using register_t = std::conditional_t< \
			std::is_same_v<typename base<T>::type, float>,
			// float:
			std::conditional_t<N == 16, __m128,
			std::conditional_t<N == 32, __m256, __m512>>,
			// double:
			std::conditional_t<N == 16, __m128d,
			std::conditional_t<N == 32, __m256d, __m512d>>
		>; ///< The variable type of the register
		template<typename T>
		static consteval size_t registers() ///< The number of registers available for the given architecture 
		{
			if constexpr (std::is_same_v<register_t<T>, __m128> || std::is_same_v<register_t<T>, __m128d>)
				return 16; // SSE
			else if constexpr (std::is_same_v<register_t<T>, __m256> || std::is_same_v<register_t<T>, __m256d>)
				return 16; // AVX
			else if constexpr (std::is_same_v<register_t<T>, __m512> || std::is_same_v<register_t<T>, __m512d>)
				return 32; // AVX-512
			else
				return 0; // Unknown
		}
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

	/* LOAD */
	
	template<typename T, typename S>
	inline constexpr auto _load = nullptr;

	template<typename T, typename S>
	inline constexpr auto _loadu = nullptr;

	template<> inline constexpr auto _load<float, SSE> = _mm_load_ps;
	template<> inline constexpr auto _loadu<float, SSE> = _mm_loadu_ps;
	template<> inline constexpr auto _load<double, SSE> = _mm_load_pd;
	template<> inline constexpr auto _loadu<double, SSE> = _mm_loadu_pd;
	template<> inline constexpr auto _load<std::complex<float>, SSE> = _mm_load_ps;
	template<> inline constexpr auto _loadu<std::complex<float>, SSE> = _mm_loadu_ps;
	template<> inline constexpr auto _load<std::complex<double>, SSE> = _mm_load_pd;
	template<> inline constexpr auto _loadu<std::complex<double>, SSE> = _mm_loadu_pd;

	template<> inline constexpr auto _load<float, AVX> = _mm256_load_ps;
	template<> inline constexpr auto _loadu<float, AVX> = _mm256_loadu_ps;
	template<> inline constexpr auto _load<double, AVX> = _mm256_load_pd;
	template<> inline constexpr auto _loadu<double, AVX> = _mm256_loadu_pd;
	template<> inline constexpr auto _load<std::complex<float>, AVX> = _mm256_load_ps;
	template<> inline constexpr auto _loadu<std::complex<float>, AVX> = _mm256_loadu_ps;
	template<> inline constexpr auto _load<std::complex<double>, AVX> = _mm256_load_pd;
	template<> inline constexpr auto _loadu<std::complex<double>, AVX> = _mm256_loadu_pd;

	template<> inline constexpr auto _load<float, AVX512> = _mm512_load_ps;
	template<> inline constexpr auto _loadu<float, AVX512> = _mm512_loadu_ps;
	template<> inline constexpr auto _load<double, AVX512> = _mm512_load_pd;
	template<> inline constexpr auto _loadu<double, AVX512> = _mm512_loadu_pd;
	template<> inline constexpr auto _load<std::complex<float>, AVX512> = _mm512_load_ps;
	template<> inline constexpr auto _loadu<std::complex<float>, AVX512> = _mm512_loadu_ps;
	template<> inline constexpr auto _load<std::complex<double>, AVX512> = _mm512_load_pd;
	template<> inline constexpr auto _loadu<std::complex<double>, AVX512> = _mm512_loadu_pd;

	template<typename T, typename S, template<typename, typename> class K> 
	void load(T** ptr, typename S::template register_t<T>** registers, const size_t row_offset, const size_t col_offset)
	{
		using kernel = K<T, S>;
		constexpr size_t rows = kernel::row_registers;
		constexpr size_t cols = kernel::col_registers;
		constexpr size_t elems = kernel::register_elements();
			
		// Check alignment based on the column offset
		if ((col_offset * sizeof(T)) % S::bytes == 0)
		{
			// Aligned load
			static_for<rows>([&]<auto i>()
			{
				static_for<cols>([&]<auto j>()
				{
					if constexpr (!std::is_same_v<T, std::complex<typename base<T>::type>>)
					{
						registers[i][j] = _load<T, S>(&ptr[row_offset + i][col_offset + j * elems]);
					}
					else 
					{
						auto* base_ptr = reinterpret_cast<typename base<T>::type*>(&ptr[row_offset + i][col_offset]);
						registers[i][j] = _load<T, S>(&base_ptr[j * elems * 2]);
					}
				});
			});
		}
		else
		{
			// Unaligned load
			static_for<rows>([&]<auto i>()
			{
				static_for<cols>([&]<auto j>()
				{
					if constexpr (!std::is_same_v<T, std::complex<typename base<T>::type>>)
					{
						registers[i][j] = _loadu<T, S>(&ptr[row_offset + i][col_offset + j * elems]);
					}
					else 
					{
						auto* base_ptr = reinterpret_cast<typename base<T>::type*>(&ptr[row_offset + i][col_offset]);
						registers[i][j] = _loadu<T, S>(&base_ptr[j * elems * 2]);
					}
				});
			});
		}
	}

	/* STORE */

	template<typename T, typename S>
	inline constexpr auto _store = nullptr;

	template<typename T, typename S>
	inline constexpr auto _storeu = nullptr;

	template<> inline constexpr auto _store<float, SSE> = _mm_store_ps;
	template<> inline constexpr auto _storeu<float, SSE> = _mm_storeu_ps;
	template<> inline constexpr auto _store<double, SSE> = _mm_store_pd;
	template<> inline constexpr auto _storeu<double, SSE> = _mm_storeu_pd;
	template<> inline constexpr auto _store<std::complex<float>, SSE> = _mm_store_ps;
	template<> inline constexpr auto _storeu<std::complex<float>, SSE> = _mm_storeu_ps;
	template<> inline constexpr auto _store<std::complex<double>, SSE> = _mm_store_pd;
	template<> inline constexpr auto _storeu<std::complex<double>, SSE> = _mm_storeu_pd;

	template<> inline constexpr auto _store<float, AVX> = _mm256_store_ps;
	template<> inline constexpr auto _storeu<float, AVX> = _mm256_storeu_ps;
	template<> inline constexpr auto _store<double, AVX> = _mm256_store_pd;
	template<> inline constexpr auto _storeu<double, AVX> = _mm256_storeu_pd;
	template<> inline constexpr auto _store<std::complex<float>, AVX> = _mm256_store_ps;
	template<> inline constexpr auto _storeu<std::complex<float>, AVX> = _mm256_storeu_ps;
	template<> inline constexpr auto _store<std::complex<double>, AVX> = _mm256_store_pd;
	template<> inline constexpr auto _storeu<std::complex<double>, AVX> = _mm256_storeu_pd;

	template<> inline constexpr auto _store<float, AVX512> = _mm512_store_ps;
	template<> inline constexpr auto _storeu<float, AVX512> = _mm512_storeu_ps;
	template<> inline constexpr auto _store<double, AVX512> = _mm512_store_pd;
	template<> inline constexpr auto _storeu<double, AVX512> = _mm512_storeu_pd;
	template<> inline constexpr auto _store<std::complex<float>, AVX512> = _mm512_store_ps;
	template<> inline constexpr auto _storeu<std::complex<float>, AVX512> = _mm512_storeu_ps;
	template<> inline constexpr auto _store<std::complex<double>, AVX512> = _mm512_store_pd;
	template<> inline constexpr auto _storeu<std::complex<double>, AVX512> = _mm512_storeu_pd;
	
	template<typename T, typename S, template<typename, typename> class K> 
	void store(T** ptr, typename S::template register_t<T>** registers, const size_t row_offset, const size_t col_offset)
	{
		using kernel = K<T, S>;
		constexpr size_t rows = kernel::row_registers;
		constexpr size_t cols = kernel::col_registers;
		constexpr size_t elems = kernel::register_elements();
				
		// Check alignment based on the column offset
		if ((col_offset * sizeof(T)) % S::bytes == 0)
		{
			// Aligned store
			static_for<rows>([&]<auto i>()
			{
				static_for<cols>([&]<auto j>()
				{
					if constexpr (!std::is_same_v<T, std::complex<typename base<T>::type>>)
					{
						_store<T, S>(&ptr[row_offset + i][col_offset + j * elems], registers[i][j]);
					}
					else 
					{
						auto* base_ptr = reinterpret_cast<typename base<T>::type*>(&ptr[row_offset + i][col_offset]);
						_store<T, S>(&base_ptr[j * elems * 2], registers[i][j]);
					}
				});
			});
		}
		else
		{
			// Unaligned store
			static_for<rows>([&]<auto i>()
			{
				static_for<cols>([&]<auto j>()
				{
					if constexpr (!std::is_same_v<T, std::complex<typename base<T>::type>>)
					{
						_storeu<T, S>(&ptr[row_offset + i][col_offset + j * elems], registers[i][j]);
					}
					else 
					{
						auto* base_ptr = reinterpret_cast<typename base<T>::type*>(&ptr[row_offset + i][col_offset]);
						_storeu<T, S>(&base_ptr[j * elems * 2], registers[i][j]);
					}
				});
			});
		}
	}

	/* BROADCAST */

	inline __m128 
	_mm_set1c_ps(const std::complex<float>& val) 
	{
		const float* B = reinterpret_cast<const float*>(&val);
		return _mm_setr_ps(B[0], B[1], B[0], B[1]);
	}

	inline __m128d 
	_mm_set1c_pd(const std::complex<double>& val) 
	{
		const double* B = reinterpret_cast<const double*>(&val);
		return _mm_setr_pd(B[0], B[1]);
	}

	inline __m256 
	_mm256_set1c_ps(const std::complex<float>& val) 
	{
		const float* B = reinterpret_cast<const float*>(&val);
		return _mm256_setr_ps(B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1]);
	}

	inline __m256d 
	_mm256_set1c_pd(const std::complex<double>& val) 
	{
		const double* B = reinterpret_cast<const double*>(&val);
		return _mm256_setr_pd(B[0], B[1], B[0], B[1]);
	}

	inline __m512 
	_mm512_set1c_ps(const std::complex<float>& val) 
	{
		const float* B = reinterpret_cast<const float*>(&val);
		return _mm512_setr_ps(
			B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1],
			B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1]);
	}

	inline __m512d 
	_mm512_set1c_pd(const std::complex<double>& val) 
	{
		const double* B = reinterpret_cast<const double*>(&val);
		return _mm512_setr_pd(B[0], B[1], B[0], B[1], B[0], B[1], B[0], B[1]);
	}


	template<typename T, typename S>
	inline constexpr auto _set1 = nullptr;

	template<> inline constexpr auto _set1<float, SSE> = _mm_set1_ps;
	template<> inline constexpr auto _set1<double, SSE> = _mm_set1_pd;
	template<> inline constexpr auto _set1<std::complex<float>, SSE> = _mm_set1c_ps;
	template<> inline constexpr auto _set1<std::complex<double>, SSE> = _mm_set1c_pd;

	template<> inline constexpr auto _set1<float, AVX> = _mm256_set1_ps;
	template<> inline constexpr auto _set1<double, AVX> = _mm256_set1_pd;
	template<> inline constexpr auto _set1<std::complex<float>, AVX> = _mm256_set1c_ps;
	template<> inline constexpr auto _set1<std::complex<double>, AVX> = _mm256_set1c_pd;

	template<> inline constexpr auto _set1<float, AVX512> = _mm512_set1_ps;
	template<> inline constexpr auto _set1<double, AVX512> = _mm512_set1_pd;
	template<> inline constexpr auto _set1<std::complex<float>, AVX512> = _mm512_set1c_ps;
	template<> inline constexpr auto _set1<std::complex<double>, AVX512> = _mm512_set1c_pd;

	/* CAST */

	template<typename T, typename S>
	inline constexpr auto _castps_pd = nullptr;

	template<> inline constexpr auto _castps_pd<float, SSE> = _mm_castps_pd;
	template<> inline constexpr auto _castps_pd<double, SSE> = _mm_castps_pd;
	template<> inline constexpr auto _castps_pd<std::complex<float>, SSE> = _mm_castps_pd;
	template<> inline constexpr auto _castps_pd<std::complex<double>, SSE> = _mm_castps_pd;

	template<> inline constexpr auto _castps_pd<float, AVX> = _mm256_castps_pd;
	template<> inline constexpr auto _castps_pd<double, AVX> = _mm256_castps_pd;
	template<> inline constexpr auto _castps_pd<std::complex<float>, AVX> = _mm256_castps_pd;
	template<> inline constexpr auto _castps_pd<std::complex<double>, AVX> = _mm256_castps_pd;

	template<> inline constexpr auto _castps_pd<float, AVX512> = _mm512_castps_pd;
	template<> inline constexpr auto _castps_pd<double, AVX512> = _mm512_castps_pd;
	template<> inline constexpr auto _castps_pd<std::complex<float>, AVX512> = _mm512_castps_pd;
	template<> inline constexpr auto _castps_pd<std::complex<double>, AVX512> = _mm512_castps_pd;

	template<typename T, typename S>
	inline constexpr auto _castpd_ps = nullptr;

	template<> inline constexpr auto _castpd_ps<float, SSE> = _mm_castpd_ps;
	template<> inline constexpr auto _castpd_ps<double, SSE> = _mm_castpd_ps;
	template<> inline constexpr auto _castpd_ps<std::complex<float>, SSE> = _mm_castpd_ps;
	template<> inline constexpr auto _castpd_ps<std::complex<double>, SSE> = _mm_castpd_ps;

	template<> inline constexpr auto _castpd_ps<float, AVX> = _mm256_castpd_ps;
	template<> inline constexpr auto _castpd_ps<double, AVX> = _mm256_castpd_ps;
	template<> inline constexpr auto _castpd_ps<std::complex<float>, AVX> = _mm256_castpd_ps;
	template<> inline constexpr auto _castpd_ps<std::complex<double>, AVX> = _mm256_castpd_ps;

	template<> inline constexpr auto _castpd_ps<float, AVX512> = _mm512_castpd_ps;
	template<> inline constexpr auto _castpd_ps<double, AVX512> = _mm512_castpd_ps;
	template<> inline constexpr auto _castpd_ps<std::complex<float>, AVX512> = _mm512_castpd_ps;
	template<> inline constexpr auto _castpd_ps<std::complex<double>, AVX512> = _mm512_castpd_ps;
	
	/* ADD */ 

	template<typename T, typename S>
	inline constexpr auto _add = nullptr;

	template<> inline constexpr auto _add<float, SSE> = _mm_add_ps;
	template<> inline constexpr auto _add<double, SSE> = _mm_add_pd;
	template<> inline constexpr auto _add<std::complex<float>, SSE> = _mm_add_ps;
	template<> inline constexpr auto _add<std::complex<double>, SSE> = _mm_add_pd;

	template<> inline constexpr auto _add<float, AVX> = _mm256_add_ps;
	template<> inline constexpr auto _add<double, AVX> = _mm256_add_pd;
	template<> inline constexpr auto _add<std::complex<float>, AVX> = _mm256_add_ps;
	template<> inline constexpr auto _add<std::complex<double>, AVX> = _mm256_add_pd;

	template<> inline constexpr auto _add<float, AVX512> = _mm512_add_ps;
	template<> inline constexpr auto _add<double, AVX512> = _mm512_add_pd;
	template<> inline constexpr auto _add<std::complex<float>, AVX512> = _mm512_add_ps;
	template<> inline constexpr auto _add<std::complex<double>, AVX512> = _mm512_add_pd;

	/* SUB */ 

	template<typename T, typename S>
	inline constexpr auto _sub = nullptr;

	template<> inline constexpr auto _sub<float, SSE> = _mm_sub_ps;
	template<> inline constexpr auto _sub<double, SSE> = _mm_sub_pd;
	template<> inline constexpr auto _sub<std::complex<float>, SSE> = _mm_sub_ps;
	template<> inline constexpr auto _sub<std::complex<double>, SSE> = _mm_sub_pd;

	template<> inline constexpr auto _sub<float, AVX> = _mm256_sub_ps;
	template<> inline constexpr auto _sub<double, AVX> = _mm256_sub_pd;
	template<> inline constexpr auto _sub<std::complex<float>, AVX> = _mm256_sub_ps;
	template<> inline constexpr auto _sub<std::complex<double>, AVX> = _mm256_sub_pd;

	template<> inline constexpr auto _sub<float, AVX512> = _mm512_sub_ps;
	template<> inline constexpr auto _sub<double, AVX512> = _mm512_sub_pd;
	template<> inline constexpr auto _sub<std::complex<float>, AVX512> = _mm512_sub_ps;
	template<> inline constexpr auto _sub<std::complex<double>, AVX512> = _mm512_sub_pd;

	/* MULTIPLY */

	inline __m128 
	_mm_mulc_ps(const __m128& a, const __m128& b) 
	{ 
		const __m128 re_a = _mm_moveldup_ps(a);
		const __m128 im_a = _mm_movehdup_ps(a);
		const __m128 b_flip = _mm_shuffle_ps(b, b, 0xB1);
		const __m128 t1 = _mm_mul_ps(re_a, b);
		const __m128 t2 = _mm_mul_ps(im_a, b_flip);
		return _mm_addsub_ps(t1, t2);
	}

	inline __m128d 
	_mm_mulc_pd(const __m128d& a, const __m128d& b)
	{
		const __m128d re_a = _mm_unpacklo_pd(a, a);
		const __m128d im_a = _mm_unpackhi_pd(a, a);
		const __m128d b_flip = _mm_shuffle_pd(b, b, 0x1);
		const __m128d t1 = _mm_mul_pd(re_a, b);
		const __m128d t2 = _mm_mul_pd(im_a, b_flip);
		return _mm_addsub_pd(t1, t2);
	}

	inline __m256 
	_mm256_mulc_ps(const __m256& a, const __m256& b)
	{
		const __m256 re_a = _mm256_moveldup_ps(a);
		const __m256 im_a = _mm256_movehdup_ps(a);
		const __m256 b_flip = _mm256_permute_ps(b, 0xB1);
		const __m256 t1 = _mm256_mul_ps(re_a, b);
		const __m256 t2 = _mm256_mul_ps(im_a, b_flip);
		return _mm256_addsub_ps(t1, t2);
	}

	inline __m256d 
	_mm256_mulc_pd(const __m256d& a, const __m256d& b)
	{
		const __m256d re_a = _mm256_unpacklo_pd(a, a);
		const __m256d im_a = _mm256_unpackhi_pd(a, a);
		const __m256d b_flip = _mm256_permute_pd(b, 0x5);
		const __m256d t1 = _mm256_mul_pd(re_a, b);
		const __m256d t2 = _mm256_mul_pd(im_a, b_flip);
		return _mm256_addsub_pd(t1, t2);
	}

	inline __m512 
	_mm512_mulc_ps(const __m512& a, const __m512& b)
	{
		const __m512 a_perm = _mm512_permute_ps(a, 0xB1);
		const __m512 b_perm = _mm512_permute_ps(b, 0xB1);
		const __m512 real = _mm512_fmsub_ps(a, b, _mm512_mul_ps(a_perm, b_perm));
		const __m512 imag = _mm512_fmadd_ps(a_perm, b, _mm512_mul_ps(a, b_perm));
		return _mm512_mask_blend_ps(0xAAAA, real, imag);  
	}

	inline __m512d 
	_mm512_mulc_pd(const __m512d& a, const __m512d& b)
	{
		const __m512d a_perm = _mm512_shuffle_pd(a, a, 0x55);
		const __m512d b_perm = _mm512_shuffle_pd(b, b, 0x55);
		const __m512d real = _mm512_fmsub_pd(a, b, _mm512_mul_pd(a_perm, b_perm));
		const __m512d imag = _mm512_fmadd_pd(a_perm, b, _mm512_mul_pd(a, b_perm));
		return _mm512_mask_blend_pd(0xAA, real, imag);
	}

	template<typename T, typename S>
	inline constexpr auto _mul = nullptr;

	template<> inline constexpr auto _mul<float, SSE> = _mm_mul_ps;
	template<> inline constexpr auto _mul<double, SSE> = _mm_mul_pd;
	template<> inline constexpr auto _mul<std::complex<float>, SSE> = _mm_mulc_ps;
	template<> inline constexpr auto _mul<std::complex<double>, SSE> = _mm_mulc_pd;

	template<> inline constexpr auto _mul<float, AVX> = _mm256_mul_ps;
	template<> inline constexpr auto _mul<double, AVX> = _mm256_mul_pd;
	template<> inline constexpr auto _mul<std::complex<float>, AVX> = _mm256_mulc_ps;
	template<> inline constexpr auto _mul<std::complex<double>, AVX> = _mm256_mulc_pd;

	template<> inline constexpr auto _mul<float, AVX512> = _mm512_mul_ps;
	template<> inline constexpr auto _mul<double, AVX512> = _mm512_mul_pd;
	template<> inline constexpr auto _mul<std::complex<float>, AVX512> = _mm512_mulc_ps;
	template<> inline constexpr auto _mul<std::complex<double>, AVX512> = _mm512_mulc_pd;


	/* DIVIDE */

	inline __m128 
	_mm_divc_ps(const __m128& a, const __m128& b) 
	{ 
		const __m128 b_conj = _mm_mul_ps(b, _mm_setr_ps(1.0, -1.0, 1.0, -1.0));
		const __m128 b_r = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 0, 0));
		const __m128 b_i = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1));
		const __m128 b_norm = _mm_fmadd_ps(b_r, b_r, _mm_mul_ps(b_i, b_i));
		const __m128 a_perm = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
		const __m128 b_conj_perm = _mm_shuffle_ps(b_conj, b_conj, _MM_SHUFFLE(2, 3, 0, 1));
		const __m128 real = _mm_fmsub_ps(a, b_conj, _mm_mul_ps(a_perm, b_conj_perm));
		const __m128 imag = _mm_fmadd_ps(a_perm, b_conj, _mm_mul_ps(a, b_conj_perm));
		const __m128 result = _mm_blend_ps(real, imag, 0xA);
		return _mm_div_ps(result, b_norm);	
	}

	inline __m128d 
	_mm_divc_pd(const __m128d& a, const __m128d& b)
	{
		const __m128d b_conj = _mm_mul_pd(b, _mm_setr_pd(1.0, -1.0));
		const __m128d b_r = _mm_unpacklo_pd(b, b);
		const __m128d b_i = _mm_unpackhi_pd(b, b);
		const __m128d b_norm = _mm_fmadd_pd(b_r, b_r, _mm_mul_pd(b_i, b_i));
		const __m128d a_perm = _mm_shuffle_pd(a, a, _MM_SHUFFLE2(0, 1));
		const __m128d b_conj_perm = _mm_shuffle_pd(b_conj, b_conj, _MM_SHUFFLE2(0, 1));
		const __m128d real = _mm_fmsub_pd(a, b_conj, _mm_mul_pd(a_perm, b_conj_perm));
		const __m128d imag = _mm_fmadd_pd(a_perm, b_conj, _mm_mul_pd(a, b_conj_perm));
		const __m128d result = _mm_blend_pd(real, imag, 0x2);
		return _mm_div_pd(result, b_norm);
	}

	inline __m256 
	_mm256_divc_ps(const __m256& a, const __m256& b)
	{
		const __m256 b_conj = _mm256_mul_ps(b, _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0));
		const __m256 b_r = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 0, 0));
		const __m256 b_i = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1));
		const __m256 b_norm = _mm256_fmadd_ps(b_r, b_r, _mm256_mul_ps(b_i, b_i));
		const __m256 a_perm = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
		const __m256 b_conj_perm = _mm256_shuffle_ps(b_conj, b_conj, _MM_SHUFFLE(2, 3, 0, 1));
		const __m256 real = _mm256_fmsub_ps(a, b_conj, _mm256_mul_ps(a_perm, b_conj_perm));
		const __m256 imag = _mm256_fmadd_ps(a_perm, b_conj, _mm256_mul_ps(a, b_conj_perm));
		const __m256 result = _mm256_blend_ps(real, imag, 0xAA);
		return _mm256_div_ps(result, b_norm);
	}

	inline __m256d 
	_mm256_divc_pd(const __m256d& a, const __m256d& b)
	{
		const __m256d b_conj = _mm256_mul_pd(b, _mm256_setr_pd(1.0, -1.0, 1.0, -1.0));
		const __m256d b_r = _mm256_unpacklo_pd(b, b);
		const __m256d b_i = _mm256_unpackhi_pd(b, b);
		const __m256d b_norm = _mm256_fmadd_pd(b_r, b_r, _mm256_mul_pd(b_i, b_i));
		const __m256d a_perm = _mm256_shuffle_pd(a, a, 0x5);
		const __m256d b_conj_perm = _mm256_shuffle_pd(b_conj, b_conj, 0x5);
		const __m256d real = _mm256_fmsub_pd(a, b_conj, _mm256_mul_pd(a_perm, b_conj_perm));
		const __m256d imag = _mm256_fmadd_pd(a_perm, b_conj, _mm256_mul_pd(a, b_conj_perm));
		const __m256d result = _mm256_blend_pd(real, imag, 0xA);
		return _mm256_div_pd(result, b_norm);
	}

	inline __m512 
	_mm512_divc_ps(const __m512& a, const __m512& b)
	{
		const __m512 b_conj = _mm512_mul_ps(b, _mm512_setr4_ps(1.0, -1.0, 1.0, -1.0));
		const __m512 b_r = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 0, 0));
		const __m512 b_i = _mm512_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1));
		const __m512 b_norm = _mm512_fmadd_ps(b_r, b_r, _mm512_mul_ps(b_i, b_i));
		const __m512 a_perm = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
		const __m512 b_conj_perm = _mm512_shuffle_ps(b_conj, b_conj, _MM_SHUFFLE(2, 3, 0, 1));
		const __m512 real = _mm512_fmsub_ps(a, b_conj, _mm512_mul_ps(a_perm, b_conj_perm));
		const __m512 imag = _mm512_fmadd_ps(a_perm, b_conj, _mm512_mul_ps(a, b_conj_perm));
		const __m512 result = _mm512_mask_blend_ps(0xAAAA, real, imag);
		return _mm512_div_ps(result, b_norm);
	}

	inline __m512d 
	_mm512_divc_pd(const __m512d& a, const __m512d& b)
	{
		const __m512d b_conj = _mm512_mul_pd(b, _mm512_setr4_pd(1.0, -1.0, 1.0, -1.0));
		const __m512d b_r = _mm512_unpacklo_pd(b, b);
		const __m512d b_i = _mm512_unpackhi_pd(b, b);
		const __m512d b_norm = _mm512_fmadd_pd(b_r, b_r, _mm512_mul_pd(b_i, b_i));
		const __m512d a_perm = _mm512_shuffle_pd(a, a, 0x55);
		const __m512d b_conj_perm = _mm512_shuffle_pd(b_conj, b_conj, 0x55);
		const __m512d real = _mm512_fmsub_pd(a, b_conj, _mm512_mul_pd(a_perm, b_conj_perm));
		const __m512d imag = _mm512_fmadd_pd(a_perm, b_conj, _mm512_mul_pd(a, b_conj_perm));
		const __m512d result = _mm512_mask_blend_pd(0xAA, real, imag);
		return _mm512_div_pd(result, b_norm); \
	}

	template<typename T, typename S>
	inline constexpr auto _div = nullptr;

	template<> inline constexpr auto _div<float, SSE> = _mm_div_ps;
	template<> inline constexpr auto _div<double, SSE> = _mm_div_pd;
	template<> inline constexpr auto _div<std::complex<float>, SSE> = _mm_divc_ps;
	template<> inline constexpr auto _div<std::complex<double>, SSE> = _mm_divc_pd;

	template<> inline constexpr auto _div<float, AVX> = _mm256_div_ps;
	template<> inline constexpr auto _div<double, AVX> = _mm256_div_pd;
	template<> inline constexpr auto _div<std::complex<float>, AVX> = _mm256_divc_ps;
	template<> inline constexpr auto _div<std::complex<double>, AVX> = _mm256_divc_pd;

	template<> inline constexpr auto _div<float, AVX512> = _mm512_div_ps;
	template<> inline constexpr auto _div<double, AVX512> = _mm512_div_pd;
	template<> inline constexpr auto _div<std::complex<float>, AVX512> = _mm512_divc_ps;
	template<> inline constexpr auto _div<std::complex<double>, AVX512> = _mm512_divc_pd;

	/* REDUCE ADD (horizontal) */

	inline float 
	_mm_reduce_add_ps(const __m128& a) 
	{ 
		__m128 shuf = _mm_movehdup_ps(a);
		__m128 sum1 = _mm_add_ps(a, shuf);
		shuf = _mm_movehl_ps(shuf, sum1);
		__m128 sum2 = _mm_add_ss(sum1, shuf);
		return _mm_cvtss_f32(sum2);
	}

	inline double 
	_mm_reduce_add_pd(const __m128d& a)
	{
		__m128d shuf = _mm_unpackhi_pd(a, a);
		__m128d sum = _mm_add_sd(a, shuf);
		return _mm_cvtsd_f64(sum);
	}

	inline std::complex<float> 
	_mm_reduce_addc_ps(const __m128& a) 
	{ 
		__m128 hi = _mm_movehl_ps(a, a);
		__m128 sum = _mm_add_ps(a, hi);
		alignas(16) float result[2];
		_mm_storel_pi((__m64*)result, sum);
		return std::complex<float>(result[0], result[1]);
	}

	inline std::complex<double> 
	_mm_reduce_addc_pd(const __m128d& a)
	{
		alignas(16) double result[2];
		_mm_store_pd(result, a);
		return std::complex<double>(result[0], result[1]);
	}

	inline float 
	_mm256_reduce_add_ps(const __m256& a)
	{
		__m128 lo = _mm256_castps256_ps128(a);
		__m128 hi = _mm256_extractf128_ps(a, 1);
		__m128 sum = _mm_add_ps(lo, hi);
		return _mm_reduce_add_ps(sum);
	}

	inline double 
	_mm256_reduce_add_pd(const __m256d& a)
	{
		__m128d lo = _mm256_castpd256_pd128(a);
		__m128d hi = _mm256_extractf128_pd(a, 1);
		__m128d sum = _mm_add_pd(lo, hi);
		return _mm_reduce_add_pd(sum);
	}

	inline std::complex<float>
	_mm256_reduce_addc_ps(const __m256& a)
	{
		__m128 lo = _mm256_castps256_ps128(a);
		__m128 hi = _mm256_extractf128_ps(a, 1);
		__m128 sum = _mm_add_ps(lo, hi);
		return _mm_reduce_addc_ps(sum);
	}

	inline std::complex<double>
	_mm256_reduce_addc_pd(const __m256d& a)
	{
		__m128d lo = _mm256_castpd256_pd128(a);
		__m128d hi = _mm256_extractf128_pd(a, 1);
		__m128d sum = _mm_add_pd(lo, hi);
		alignas(16) double result[2];
		_mm_store_pd(result, sum);
		return std::complex<double>(result[0], result[1]);
	}

	inline std::complex<float>
	_mm512_reduce_addc_ps(const __m512& a)
	{
		// Reduce 512 -> 256
		__m256 lo = _mm512_castps512_ps256(a);
		__m256 hi = _mm512_extractf32x8_ps(a, 1);
		__m256 sum256 = _mm256_add_ps(lo, hi);
		
		// Reduce 256 -> 128
		__m128 lo128 = _mm256_castps256_ps128(sum256);
		__m128 hi128 = _mm256_extractf128_ps(sum256, 1);
		__m128 sum128 = _mm_add_ps(lo128, hi128);
		
		// Reduce 128 -> complex<float>
		return _mm_reduce_addc_ps(sum128);
	}

	inline std::complex<double>
	_mm512_reduce_addc_pd(const __m512d& a)
	{
		// Reduce 512 -> 256
		__m256d lo = _mm512_castpd512_pd256(a);
		__m256d hi = _mm512_extractf64x4_pd(a, 1);
		__m256d sum256 = _mm256_add_pd(lo, hi);
		
		// Reduce 256 -> 128
		__m128d lo128 = _mm256_castpd256_pd128(sum256);
		__m128d hi128 = _mm256_extractf128_pd(sum256, 1);
		__m128d sum128 = _mm_add_pd(lo128, hi128);
		
		// Reduce 128 -> complex<double>
		return _mm_reduce_addc_pd(sum128);
	}


	template<typename T, typename S>
	inline constexpr auto _reduce_add = nullptr;

	template<> inline constexpr auto _reduce_add<float, SSE> = _mm_reduce_add_ps;
	template<> inline constexpr auto _reduce_add<double, SSE> = _mm_reduce_add_pd;
	template<> inline constexpr auto _reduce_add<std::complex<float>, SSE> = _mm_reduce_addc_ps;
	template<> inline constexpr auto _reduce_add<std::complex<double>, SSE> = _mm_reduce_addc_pd;

	template<> inline constexpr auto _reduce_add<float, AVX> = _mm256_reduce_add_ps;
	template<> inline constexpr auto _reduce_add<double, AVX> = _mm256_reduce_add_pd;
	template<> inline constexpr auto _reduce_add<std::complex<float>, AVX> = _mm256_reduce_addc_ps;
	template<> inline constexpr auto _reduce_add<std::complex<double>, AVX> = _mm256_reduce_addc_pd;

	template<> inline constexpr auto _reduce_add<float, AVX512> = _mm512_reduce_add_ps;
	template<> inline constexpr auto _reduce_add<double, AVX512> = _mm512_reduce_add_pd;
	template<> inline constexpr auto _reduce_add<std::complex<float>, AVX512> = _mm512_reduce_addc_ps;
	template<> inline constexpr auto _reduce_add<std::complex<double>, AVX512> = _mm512_reduce_addc_pd;

	/* REDUCE MUL */

	inline float 
	_mm_reduce_mul_ps(const __m128& acc) 
	{ 
		const __m128 tmp1 = _mm_movehdup_ps(acc);
		const __m128 prod1 = _mm_mul_ps(acc, tmp1);
		const __m128 tmp2 = _mm_movehl_ps(prod1, prod1);
		const __m128 prod2 = _mm_mul_ss(prod1, tmp2);
		return _mm_cvtss_f32(prod2);
	}

	inline double 
	_mm_reduce_mul_pd(const __m128d& acc)
	{
		const __m128d temp = _mm_unpackhi_pd(acc, acc);
		const __m128d prod = _mm_mul_sd(acc, temp);
		return _mm_cvtsd_f64(prod);
	}

	inline std::complex<float> 
	_mm_reduce_mulc_ps(const __m128& a) 
	{ 
		__m128 acc = _mm_setr_ps(1.0, 0.0, 1.0, 0.0);
		acc = _mm_mulc_ps(acc, a);
		__m128 hi = _mm_movehl_ps(acc, acc);
		__m128 result = _mm_mulc_ps(acc, hi);
		alignas(16) float res[2];
		_mm_storel_pi((__m64*)res, result);
		return std::complex<float>(res[0], res[1]);
	}

	inline std::complex<double> 
	_mm_reduce_mulc_pd(const __m128d& a)
	{
		alignas(16) double result[2];
		_mm_store_pd(result, a);
		return std::complex<double>(result[0], result[1]);
	}

	inline float 
	_mm256_reduce_mul_ps(const __m256& prod)
	{
		const __m128 lo = _mm256_castps256_ps128(prod);
		const __m128 hi = _mm256_extractf128_ps(prod, 1);
		const __m128 acc = _mm_mul_ps(lo, hi);
		return _mm_reduce_mul_ps(acc);
	}

	inline double 
	_mm256_reduce_mul_pd(const __m256d& prod)
	{
		const __m128d low = _mm256_castpd256_pd128(prod);
		const __m128d high = _mm256_extractf128_pd(prod, 1);
		const __m128d acc = _mm_mul_pd(low, high);
		const __m128d tmp = _mm_unpackhi_pd(acc, acc);
		const __m128d final = _mm_mul_sd(acc, tmp);
		return _mm_cvtsd_f64(final);
	}

	inline std::complex<float>
	_mm256_reduce_mulc_ps(const __m256& a)
	{
		__m128 lo = _mm256_castps256_ps128(a);
		__m128 hi = _mm256_extractf128_ps(a, 1);
		__m128 result = _mm_mulc_ps(lo, hi);
		__m128 hi_part = _mm_movehl_ps(result, result);
		__m128 final_result = _mm_mulc_ps(result, hi_part);
		alignas(16) float res[2];
		_mm_storel_pi((__m64*)res, final_result);
		return std::complex<float>(res[0], res[1]);
	}

	inline std::complex<double>
	_mm256_reduce_mulc_pd(const __m256d& a)
	{
		__m128d lo = _mm256_castpd256_pd128(a);
		__m128d hi = _mm256_extractf128_pd(a, 1);
		__m128d res = _mm_mulc_pd(lo, hi);
		alignas(16) double result[2];
		_mm_store_pd(result, res);
		return std::complex<double>(result[0], result[1]);
	}

	inline std::complex<float>
	_mm512_reduce_mulc_ps(const __m512& a)
	{
		const __m256 lo = _mm512_castps512_ps256(a);
		const __m256 hi = _mm512_extractf32x8_ps(a, 1);
		__m256 prod256 = _mm256_mulc_ps(lo, hi);
		const __m128 lo128 = _mm256_castps256_ps128(prod256);
		const __m128 hi128 = _mm256_extractf128_ps(prod256, 1);
		__m128 res = _mm_mulc_ps(lo128, hi128);
		__m128 hi_part = _mm_movehl_ps(res, res);
		__m128 final_res = _mm_mulc_ps(res, hi_part);
		alignas(16) float result[2];
		_mm_storel_pi((__m64*)result, final_res);
		return std::complex<float>(result[0], result[1]);
	}

	inline std::complex<double>
	_mm512_reduce_mulc_pd(const __m512d& a)
	{
		const __m256d lo = _mm512_castpd512_pd256(a);
		const __m256d hi = _mm512_extractf64x4_pd(a, 1);
		__m256d prod256 = _mm256_mulc_pd(lo, hi);
		const __m128d lo128 = _mm256_castpd256_pd128(prod256);
		const __m128d hi128 = _mm256_extractf128_pd(prod256, 1);
		__m128d res = _mm_mulc_pd(lo128, hi128);
		alignas(16) double result[2];
		_mm_store_pd(result, res);
		return std::complex<double>(result[0], result[1]);
	}


	template<typename T, typename S>
	inline constexpr auto _reduce_mul = nullptr;

	template<> inline constexpr auto _reduce_mul<float, SSE> = _mm_reduce_mul_ps;
	template<> inline constexpr auto _reduce_mul<double, SSE> = _mm_reduce_mul_pd;
	template<> inline constexpr auto _reduce_mul<std::complex<float>, SSE> = _mm_reduce_mulc_ps;
	template<> inline constexpr auto _reduce_mul<std::complex<double>, SSE> = _mm_reduce_mulc_pd;

	template<> inline constexpr auto _reduce_mul<float, AVX> = _mm256_reduce_mul_ps;
	template<> inline constexpr auto _reduce_mul<double, AVX> = _mm256_reduce_mul_pd;
	template<> inline constexpr auto _reduce_mul<std::complex<float>, AVX> = _mm256_reduce_mulc_ps;
	template<> inline constexpr auto _reduce_mul<std::complex<double>, AVX> = _mm256_reduce_mulc_pd;

	template<> inline constexpr auto _reduce_mul<float, AVX512> = _mm512_reduce_mul_ps;
	template<> inline constexpr auto _reduce_mul<double, AVX512> = _mm512_reduce_mul_pd;
	template<> inline constexpr auto _reduce_mul<std::complex<float>, AVX512> = _mm512_reduce_mulc_ps;
	template<> inline constexpr auto _reduce_mul<std::complex<double>, AVX512> = _mm512_reduce_mulc_pd;



	/* FMA */

	inline __m128 
	_mm_fmaddc_ps(const __m128& a, const __m128& b, const __m128& c) 
	{ 
		return _mm_add_ps(_mm_mulc_ps(a, b), c);
	}

	inline __m128d 
	_mm_fmaddc_pd(const __m128d& a, const __m128d& b, const __m128d& c)
	{
		return _mm_add_pd(_mm_mulc_pd(a, b), c);
	}

	inline __m256 
	_mm256_fmaddc_ps(const __m256& a, const __m256& b, const __m256& c)
	{
		return _mm256_add_ps(_mm256_mulc_ps(a, b), c);
	}

	inline __m256d 
	_mm256_fmaddc_pd(const __m256d& a, const __m256d& b, const __m256d& c)
	{
		return _mm256_add_pd(_mm256_mulc_pd(a, b), c);
	}

	inline __m512 
	_mm512_fmaddc_ps(const __m512& a, const __m512& b, const __m512& c)
	{
		return _mm512_add_ps(_mm512_mulc_ps(a, b), c);
	}

	inline __m512d 
	_mm512_fmaddc_pd(const __m512d& a, const __m512d& b, const __m512d& c)
	{
		return _mm512_add_pd(_mm512_mulc_pd(a, b), c);
	}

	template<typename T, typename S>
	inline constexpr auto _fmadd = nullptr;

	template<> inline constexpr auto _fmadd<float, SSE> = _mm_fmadd_ps;
	template<> inline constexpr auto _fmadd<double, SSE> = _mm_fmadd_pd;
	template<> inline constexpr auto _fmadd<std::complex<float>, SSE> = _mm_fmaddc_ps;
	template<> inline constexpr auto _fmadd<std::complex<double>, SSE> = _mm_fmaddc_pd;

	template<> inline constexpr auto _fmadd<float, AVX> = _mm256_fmadd_ps;
	template<> inline constexpr auto _fmadd<double, AVX> = _mm256_fmadd_pd;
	template<> inline constexpr auto _fmadd<std::complex<float>, AVX> = _mm256_fmaddc_ps;
	template<> inline constexpr auto _fmadd<std::complex<double>, AVX> = _mm256_fmaddc_pd;

	template<> inline constexpr auto _fmadd<float, AVX512> = _mm512_fmadd_ps;
	template<> inline constexpr auto _fmadd<double, AVX512> = _mm512_fmadd_pd;
	template<> inline constexpr auto _fmadd<std::complex<float>, AVX512> = _mm512_fmaddc_ps;
	template<> inline constexpr auto _fmadd<std::complex<double>, AVX512> = _mm512_fmaddc_pd;


	/* FMS - Fused Multiply-Subtract*/

	template<typename T, typename S>
	inline constexpr auto _fmsub = nullptr;

	template<> inline constexpr auto _fmsub<float, SSE> = _mm_fmsub_ps;
	template<> inline constexpr auto _fmsub<double, SSE> = _mm_fmsub_pd;

	template<> inline constexpr auto _fmsub<float, AVX> = _mm256_fmsub_ps;
	template<> inline constexpr auto _fmsub<double, AVX> = _mm256_fmsub_pd;

	template<> inline constexpr auto _fmsub<float, AVX512> = _mm512_fmsub_ps;
	template<> inline constexpr auto _fmsub<double, AVX512> = _mm512_fmsub_pd;

	/* FMADDSUB */

	template<typename T, typename S>
	inline constexpr auto _fmaddsub = nullptr;

	template<> inline constexpr auto _fmaddsub<float, SSE> = _mm_fmaddsub_ps;
	template<> inline constexpr auto _fmaddsub<double, SSE> = _mm_fmaddsub_pd;

	template<> inline constexpr auto _fmaddsub<float, AVX> = _mm256_fmaddsub_ps;
	template<> inline constexpr auto _fmaddsub<double, AVX> = _mm256_fmaddsub_pd;

	template<> inline constexpr auto _fmaddsub<float, AVX512> = _mm512_fmaddsub_ps;
	template<> inline constexpr auto _fmaddsub<double, AVX512> = _mm512_fmaddsub_pd;

	/* FMADDSUB */

	template<typename T, typename S>
	inline constexpr auto _fmsubadd = nullptr;

	template<> inline constexpr auto _fmsubadd<float, SSE> = _mm_fmsubadd_ps;
	template<> inline constexpr auto _fmsubadd<double, SSE> = _mm_fmsubadd_pd;

	template<> inline constexpr auto _fmsubadd<float, AVX> = _mm256_fmsubadd_ps;
	template<> inline constexpr auto _fmsubadd<double, AVX> = _mm256_fmsubadd_pd;

	template<> inline constexpr auto _fmsubadd<float, AVX512> = _mm512_fmsubadd_ps;
	template<> inline constexpr auto _fmsubadd<double, AVX512> = _mm512_fmsubadd_pd;


	/* FNMADD - Fused Negated Multiply-Add: -(a*b) + c = c - a*b */

	template<typename T, typename S>
	inline constexpr auto _fnmadd = nullptr;

	template<> inline constexpr auto _fnmadd<float, SSE> = _mm_fnmadd_ps;
	template<> inline constexpr auto _fnmadd<double, SSE> = _mm_fnmadd_pd;
	
	template<> inline constexpr auto _fnmadd<float, AVX> = _mm256_fnmadd_ps;
	template<> inline constexpr auto _fnmadd<double, AVX> = _mm256_fnmadd_pd;
	
	template<> inline constexpr auto _fnmadd<float, AVX512> = _mm512_fnmadd_ps;
	template<> inline constexpr auto _fnmadd<double, AVX512> = _mm512_fnmadd_pd;
	
	/* FNMSUB - Fused Negated Multiply-Subtract: -(a*b) - c = -(a*b + c) */

	template<typename T, typename S>
	inline constexpr auto _fnmsub = nullptr;

	template<> inline constexpr auto _fnmsub<float, SSE> = _mm_fnmsub_ps;
	template<> inline constexpr auto _fnmsub<double, SSE> = _mm_fnmsub_pd;

	template<> inline constexpr auto _fnmsub<float, AVX> = _mm256_fnmsub_ps;
	template<> inline constexpr auto _fnmsub<double, AVX> = _mm256_fnmsub_pd;

	template<> inline constexpr auto _fnmsub<float, AVX512> = _mm512_fnmsub_ps;
	template<> inline constexpr auto _fnmsub<double, AVX512> = _mm512_fnmsub_pd;


	/* Unpack / Shuffle */

	template<typename T, typename S>
	inline constexpr auto _unpacklo = nullptr;

	template<> inline constexpr auto _unpacklo<float, SSE> = _mm_unpacklo_ps;
	template<> inline constexpr auto _unpacklo<double, SSE> = _mm_unpacklo_pd;
	template<> inline constexpr auto _unpacklo<std::complex<float>, SSE> = _mm_unpacklo_ps;
	template<> inline constexpr auto _unpacklo<std::complex<double>, SSE> = _mm_unpacklo_pd;

	template<> inline constexpr auto _unpacklo<float, AVX> = _mm256_unpacklo_ps;
	template<> inline constexpr auto _unpacklo<double, AVX> = _mm256_unpacklo_pd;
	template<> inline constexpr auto _unpacklo<std::complex<float>, AVX> = _mm256_unpacklo_ps;
	template<> inline constexpr auto _unpacklo<std::complex<double>, AVX> = _mm256_unpacklo_pd;

	template<> inline constexpr auto _unpacklo<float, AVX512> = _mm512_unpacklo_ps;
	template<> inline constexpr auto _unpacklo<double, AVX512> = _mm512_unpacklo_pd;
	template<> inline constexpr auto _unpacklo<std::complex<float>, AVX512> = _mm512_unpacklo_ps;
	template<> inline constexpr auto _unpacklo<std::complex<double>, AVX512> = _mm512_unpacklo_pd;

	template<typename T, typename S>
	inline constexpr auto _unpackhi = nullptr;

	template<> inline constexpr auto _unpackhi<float, SSE> = _mm_unpackhi_ps;
	template<> inline constexpr auto _unpackhi<double, SSE> = _mm_unpackhi_pd;
	template<> inline constexpr auto _unpackhi<std::complex<float>, SSE> = _mm_unpackhi_ps;
	template<> inline constexpr auto _unpackhi<std::complex<double>, SSE> = _mm_unpackhi_pd;

	template<> inline constexpr auto _unpackhi<float, AVX> = _mm256_unpackhi_ps;
	template<> inline constexpr auto _unpackhi<double, AVX> = _mm256_unpackhi_pd;
	template<> inline constexpr auto _unpackhi<std::complex<float>, AVX> = _mm256_unpackhi_ps;
	template<> inline constexpr auto _unpackhi<std::complex<double>, AVX> = _mm256_unpackhi_pd;

	template<> inline constexpr auto _unpackhi<float, AVX512> = _mm512_unpackhi_ps;
	template<> inline constexpr auto _unpackhi<double, AVX512> = _mm512_unpackhi_pd;
	template<> inline constexpr auto _unpackhi<std::complex<float>, AVX512> = _mm512_unpackhi_ps;
	template<> inline constexpr auto _unpackhi<std::complex<double>, AVX512> = _mm512_unpackhi_pd;

	template<typename T, typename S>
	inline constexpr auto _shuffle = nullptr;

	template<> inline constexpr auto _shuffle<float, SSE> = _mm_shuffle_ps;
	template<> inline constexpr auto _shuffle<double, SSE> = _mm_shuffle_pd;
	template<> inline constexpr auto _shuffle<std::complex<float>, SSE> = _mm_shuffle_ps;
	template<> inline constexpr auto _shuffle<std::complex<double>, SSE> = _mm_shuffle_pd;

	template<> inline constexpr auto _shuffle<float, AVX> = _mm256_shuffle_ps;
	template<> inline constexpr auto _shuffle<double, AVX> = _mm256_shuffle_pd;
	template<> inline constexpr auto _shuffle<std::complex<float>, AVX> = _mm256_shuffle_ps;
	template<> inline constexpr auto _shuffle<std::complex<double>, AVX> = _mm256_shuffle_pd;

	template<> inline constexpr auto _shuffle<float, AVX512> = _mm512_shuffle_ps;
	template<> inline constexpr auto _shuffle<double, AVX512> = _mm512_shuffle_pd;
	template<> inline constexpr auto _shuffle<std::complex<float>, AVX512> = _mm512_shuffle_ps;
	template<> inline constexpr auto _shuffle<std::complex<double>, AVX512> = _mm512_shuffle_pd;


	inline __m128 _mm_shuffle_lanes_ps(__m128 a, __m128 b, int mask)
	{
		// mask 0x44 (0100 0100) means take low from both -> movelh
		// mask 0xEE (1110 1110) means take high from both -> movehl
		return (mask == 0x44) ? _mm_movelh_ps(a, b) : _mm_movehl_ps(b, a);
	}

	inline __m128d _mm_shuffle_lanes_pd(__m128d a, __m128d b, int mask)
	{
		return (mask == 0x44) ? _mm_unpacklo_pd(a, b) : _mm_unpackhi_pd(a, b);
	}
	
	template<typename T, typename S>
	inline constexpr auto _shuffle_lanes = nullptr;

	template<> inline constexpr auto _shuffle_lanes<float, SSE> = _mm_shuffle_lanes_ps;
	template<> inline constexpr auto _shuffle_lanes<double, SSE> = _mm_shuffle_lanes_pd;
	template<> inline constexpr auto _shuffle_lanes<std::complex<float>, SSE> = _mm_shuffle_lanes_ps;
	template<> inline constexpr auto _shuffle_lanes<std::complex<double>, SSE> = _mm_shuffle_lanes_pd;

	template<> inline constexpr auto _shuffle_lanes<float, AVX> = _mm256_permute2f128_ps;
	template<> inline constexpr auto _shuffle_lanes<double, AVX> = _mm256_permute2f128_pd;
	template<> inline constexpr auto _shuffle_lanes<std::complex<float>, AVX> = _mm256_permute2f128_ps;
	template<> inline constexpr auto _shuffle_lanes<std::complex<double>, AVX> = _mm256_permute2f128_pd;

	template<> inline constexpr auto _shuffle_lanes<float, AVX512> = _mm512_shuffle_f32x4;
	template<> inline constexpr auto _shuffle_lanes<double, AVX512> = _mm512_shuffle_f64x2;
	template<> inline constexpr auto _shuffle_lanes<std::complex<float>, AVX512> = _mm512_shuffle_f32x4;
	template<> inline constexpr auto _shuffle_lanes<std::complex<double>, AVX512> = _mm512_shuffle_f64x2;

/* XOR */

	template<typename T, typename S>
	inline constexpr auto _xor = nullptr;

	template<> inline constexpr auto _xor<float, SSE> = _mm_xor_ps;
	template<> inline constexpr auto _xor<double, SSE> = _mm_xor_pd;

	template<> inline constexpr auto _xor<float, AVX> = _mm256_xor_ps;
	template<> inline constexpr auto _xor<double, AVX> = _mm256_xor_pd;

	template<> inline constexpr auto _xor<float, AVX512> = _mm512_xor_ps;
	template<> inline constexpr auto _xor<double, AVX512> = _mm512_xor_pd;

/* ETC */

	/**
	 * \brief Duplicate even-indexed elements
	 */
	template<typename T, typename S>
	inline constexpr auto _duplicate_even = nullptr;

	template<> inline constexpr auto _duplicate_even<float, SSE> = _mm_moveldup_ps;
	template<> inline constexpr auto _duplicate_even<float, AVX> = _mm256_moveldup_ps;
	template<> inline constexpr auto _duplicate_even<float, AVX512> = _mm512_moveldup_ps;

	// For double, use unpacklo which duplicates lower element of each pair
	inline __m128d _mm_duplicate_even_pd(__m128d a) { return _mm_unpacklo_pd(a, a); }
	inline __m256d _mm256_duplicate_even_pd(__m256d a) { return _mm256_unpacklo_pd(a, a); }
	inline __m512d _mm512_duplicate_even_pd(__m512d a) { return _mm512_unpacklo_pd(a, a); }

	template<> inline constexpr auto _duplicate_even<double, SSE> = _mm_duplicate_even_pd;
	template<> inline constexpr auto _duplicate_even<double, AVX> = _mm256_duplicate_even_pd;
	template<> inline constexpr auto _duplicate_even<double, AVX512> = _mm512_duplicate_even_pd;

	/**
	 * \brief Duplicate odd-indexed elements
	 */
	template<typename T, typename S>
	inline constexpr auto _duplicate_odd = nullptr;

	template<> inline constexpr auto _duplicate_odd<float, SSE> = _mm_movehdup_ps;
	template<> inline constexpr auto _duplicate_odd<float, AVX> = _mm256_movehdup_ps;
	template<> inline constexpr auto _duplicate_odd<float, AVX512> = _mm512_movehdup_ps;

	// For double, use unpackhi which duplicates higher element of each pair
	inline __m128d _mm_duplicate_odd_pd(__m128d a) { return _mm_unpackhi_pd(a, a); }
	inline __m256d _mm256_duplicate_odd_pd(__m256d a) { return _mm256_unpackhi_pd(a, a); }
	inline __m512d _mm512_duplicate_odd_pd(__m512d a) { return _mm512_unpackhi_pd(a, a); }

	template<> inline constexpr auto _duplicate_odd<double, SSE> = _mm_duplicate_odd_pd;
	template<> inline constexpr auto _duplicate_odd<double, AVX> = _mm256_duplicate_odd_pd;
	template<> inline constexpr auto _duplicate_odd<double, AVX512> = _mm512_duplicate_odd_pd;
	
	/**
	 * \brief Shuffle/permute to swap adjacent pairs [a,b,c,d,...] -> [b,a,d,c,...]
	 */
	template<typename T, typename S>
	inline __attribute__((always_inline))
	typename S::template register_t<T> swap_adjacent_pairs(typename S::template register_t<T> vec)
	{
		if constexpr (std::is_same_v<T, float>)
		{
			if constexpr (std::is_same_v<S, SSE>)
			{
				return _mm_shuffle_ps(vec, vec, 0xB1);
			}
			else if constexpr (std::is_same_v<S, AVX>)
			{
				return _mm256_permute_ps(vec, 0xB1);
			}
			else if constexpr (std::is_same_v<S, AVX512>)
			{
				return _mm512_permute_ps(vec, 0xB1);
			}
		}
		else // double
		{
			if constexpr (std::is_same_v<S, SSE>)
			{
				return _mm_shuffle_pd(vec, vec, 0x1);
			}
			else if constexpr (std::is_same_v<S, AVX>)
			{
				return _mm256_permute_pd(vec, 0x5);
			}
			else if constexpr (std::is_same_v<S, AVX512>)
			{
				return _mm512_permute_pd(vec, 0x55);
			}
		}
	}

	/**
	 * \brief Create a register with alternating sign bits on even register indices [1.0, -1.0, 1.0, -1.0]
	 */
	template<typename T, typename S>
	inline __attribute__((always_inline))
	typename S::template register_t<T> alternating_sign_mask_odd()
	{
		if constexpr (std::is_same_v<T, float>)
		{
			if constexpr (std::is_same_v<S, AVX512>)
				return _mm512_setr_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
			else if constexpr (std::is_same_v<S, AVX>)
				return _mm256_setr_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
			else // SSE
				return _mm_setr_ps(-1.0, 1.0, -1.0, 1.0);
		}
		else // double
		{
			if constexpr (std::is_same_v<S, AVX512>)
				return _mm512_setr_pd(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
			else if constexpr (std::is_same_v<S, AVX>)
				return _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);
			else // SSE
				return _mm_setr_pd(-1.0, 1.0);
		}
	}

	/**
	 * \brief Create a register with alternating sign bits on odd register indices [-1.0, 1.0, -1.0, 1.0]
	 */
	template<typename T, typename S>
	inline __attribute__((always_inline))
	typename S::template register_t<T> alternating_sign_mask_even()
	{
		if constexpr (std::is_same_v<T, float>)
		{
			if constexpr (std::is_same_v<S, AVX512>)
				return _mm512_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
			else if constexpr (std::is_same_v<S, AVX>)
				return _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
			else
				return _mm_setr_ps(1.0, -1.0, 1.0, -1.0);
		}
		else
		{
			if constexpr (std::is_same_v<S, AVX512>)
				return _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
			else if constexpr (std::is_same_v<S, AVX>)
				return _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
			else
				return _mm_setr_pd(1.0, -1.0);
		}
	}

}
#endif //__SIMD_H__