/**
 * \file multiply.cc 
 * \brief implementations for matrix multiply
 * \author cpapakonstantinou
 * \date 2025
 **/
// Copyright (c) 2025  Constantine Papakonstantinou
//
// Permission is hereby granted, free of charge, to any person obaining a copy
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
#include <multiply.h>
#include <transpose.h>
#include <macros.h>

namespace damm
{
	template <>
	void 
	__multiply_block_simd<float, SSE>(__m128* a, __m128* b, __m128* c)
	{	
		constexpr size_t N = SSE::elements<float>();

		alignas(SSE::bytes)__m128 t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				t[j] = _mm_mul_ps(a[i], b[j]);
			});

			__transpose_block_simd<float, SSE>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm_add_ps(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm_add_ps(c[i], t[0]);
		});
	}

	template <>
	void 
	__multiply_block_simd<double, SSE>(__m128d* a, __m128d* b, __m128d* c)
	{	
		constexpr size_t N = SSE::elements<double>();

		alignas(SSE::bytes)__m128d t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				t[j] = _mm_mul_pd(a[i], b[j]);
			});

			__transpose_block_simd<double, SSE>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm_add_pd(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm_add_pd(c[i], t[0]);
		});
	}

	template <>
	void 
	__multiply_block_simd<std::complex<float>, SSE>(__m128* a, __m128* b, __m128* c)
	{	
		constexpr size_t N = SSE::elements<std::complex<float>>();

		alignas(SSE::bytes)__m128 t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				_MM_MUL_C_PS(a[i], b[j], t[j]);
			});

			__transpose_block_simd<std::complex<float>, SSE>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm_add_ps(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm_add_ps(c[i], t[0]);
		});
	}

	template <>
	void 
	__multiply_block_simd<std::complex<double>, SSE>(__m128d* a, __m128d* b, __m128d* c)
	{	
		constexpr size_t N = SSE::elements<std::complex<double>>();

		alignas(SSE::bytes)__m128d t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				_MM_MUL_C_PD(a[i], b[j], t[j]);
			});

			__transpose_block_simd<std::complex<double>, SSE>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm_add_pd(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm_add_pd(c[i], t[0]);
		});
	}

	template <>
	void 
	__multiply_block_simd<float, AVX>(__m256* a, __m256* b, __m256* c)
	{	
		constexpr size_t N = AVX::elements<float>();

		alignas(AVX::bytes)__m256 t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				t[j] = _mm256_mul_ps(a[i], b[j]);
			});

			__transpose_block_simd<float, AVX>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm256_add_ps(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm256_add_ps(c[i], t[0]);
		});
	}

	template <>
	void 
	__multiply_block_simd<double, AVX>(__m256d* a, __m256d* b, __m256d* c)
	{	
		constexpr size_t N = AVX::elements<double>();

		alignas(AVX::bytes)__m256d t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				t[j] = _mm256_mul_pd(a[i], b[j]);
			});

			__transpose_block_simd<double, AVX>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm256_add_pd(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm256_add_pd(c[i], t[0]);
		});
	}

	template <>
	void 
	__multiply_block_simd<std::complex<float>, AVX>(__m256* a, __m256* b, __m256* c)
	{	
		constexpr size_t N = AVX::elements<std::complex<float>>();

		alignas(AVX::bytes)__m256 t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				_MM256_MUL_C_PS(a[i], b[j], t[j]);
			});

			__transpose_block_simd<std::complex<float>, AVX>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm256_add_ps(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm256_add_ps(c[i], t[0]);
		});
	}

	template <>
	void 
	__multiply_block_simd<std::complex<double>, AVX>(__m256d* a, __m256d* b, __m256d* c)
	{	
		constexpr size_t N = AVX::elements<std::complex<double>>();

		alignas(AVX::bytes)__m256d t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				_MM256_MUL_C_PD(a[i], b[j], t[j]);
			});

			__transpose_block_simd<std::complex<double>, AVX>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm256_add_pd(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm256_add_pd(c[i], t[0]);
		});
	}


	template <>
	void 
	__multiply_block_simd<float, AVX512>(__m512* a, __m512* b, __m512* c)
	{	
		constexpr size_t N = AVX512::elements<float>();

		alignas(AVX512::bytes)__m512 t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				t[j] = _mm512_mul_ps(a[i], b[j]);
			});

			__transpose_block_simd<float, AVX512>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm512_add_ps(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm512_add_ps(c[i], t[0]);
		});

		// _MM512_MMUL16_PS(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],a[13],a[14],a[15],
		//  b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15],
		//  c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],c[13],c[14],c[15])
	}

	template <>
	void 
	__multiply_block_simd<double, AVX512>(__m512d* a, __m512d* b, __m512d* c)
	{	
		constexpr size_t N = AVX512::elements<double>();

		alignas(AVX512::bytes)__m512d t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				t[j] = _mm512_mul_pd(a[i], b[j]);
			});

			__transpose_block_simd<double, AVX512>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm512_add_pd(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm512_add_pd(c[i], t[0]);
		});

		// _MM512_MMUL8_PD(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],
		//  b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],
		//  c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7])
	}

	template <>
	void 
	__multiply_block_simd<std::complex<float>, AVX512>(__m512* a, __m512* b, __m512* c)
	{	
		constexpr size_t N = AVX512::elements<std::complex<float>>();

		alignas(AVX512::bytes)__m512 t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				_MM512_MUL_C_PS(a[i], b[j], t[j]);
			});

			__transpose_block_simd<std::complex<float>, AVX512>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm512_add_ps(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm512_add_ps(c[i], t[0]);
		});
	}

	template <>
	void 
	__multiply_block_simd<std::complex<double>, AVX512>(__m512d* a, __m512d* b, __m512d* c)
	{	
		constexpr size_t N = AVX512::elements<std::complex<double>>();

		alignas(AVX512::bytes)__m512d t[N];

		static_for<N>([&]<auto i>
		{
			static_for<N>([&]<auto j>
			{
				_MM512_MUL_C_PD(a[i], b[j], t[j]);
			});

			__transpose_block_simd<std::complex<double>, AVX512>(t);
			
			static_for<__builtin_ctz(N)>([&]<auto level>() 
			{
				constexpr size_t step = 1 << level;
				constexpr size_t pairs = N >> (level + 1);
				
				static_for<pairs>([&]<auto p>() 
				{
					constexpr size_t lo = p * (step << 1);
					constexpr size_t hi = lo + step;		
					t[lo] = _mm512_add_pd(t[lo], t[hi]);
				
				});
			});

			c[i] = _mm512_add_pd(c[i], t[0]);
		});
	}
	
}//namespace damm