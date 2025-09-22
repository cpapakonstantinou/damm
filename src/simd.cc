/**
 * \file simd.cc
 * \brief implementations for simd.h 
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

#include <simd.h>

namespace damm
{
/* LOAD */
	template<> 
	void load<float, SSE>(float* ptr, __m128* rows, const size_t stride) 
	{
		if ((stride * sizeof(float)) % SSE::bytes == 0)
		{
			static_for<SSE::elements<float>()>([&]<auto i>()
			{
				rows[i] = _mm_load_ps(&ptr[i*stride]);
			});
		}
		else
		{
			static_for<SSE::elements<float>()>([&]<auto i>()
			{
				rows[i] = _mm_loadu_ps(&ptr[i*stride]);
			});
		}
	}

	template<> 
	void load<double, SSE>(double* ptr, __m128d* rows, const size_t stride) 
	{
		if ((stride * sizeof(double)) % SSE::bytes == 0)
		{
			static_for<SSE::elements<double>()>([&]<auto i>()
			{
				rows[i] = _mm_load_pd(&ptr[i*stride]);
			});
		}
		else
		{
			static_for<SSE::elements<double>()>([&]<auto i>()
			{
				rows[i] = _mm_loadu_pd(&ptr[i*stride]);
			});
		}
	}

	template<> 
	void load<std::complex<float>, SSE>(std::complex<float>* ptr, __m128* rows, const size_t stride) 
	{
		float* ptr_intlv = reinterpret_cast<float*>(ptr);

		if ((stride * sizeof(std::complex<float>)) % SSE::bytes == 0)
		{
			static_for<SSE::elements<std::complex<float>>()>([&]<auto i>()
			{
				rows[i] = _mm_load_ps(&ptr_intlv[i*stride*2]);
			});
		}
		else
		{
			static_for<SSE::elements<std::complex<float>>()>([&]<auto i>()
			{
				rows[i] = _mm_loadu_ps(&ptr_intlv[i*stride*2]);
			});
		}
	}

	template<> 
	void load<std::complex<double>, SSE>(std::complex<double>* ptr, __m128d* rows, const size_t stride) 
	{
		double* ptr_intlv = reinterpret_cast<double*>(ptr);

		if ((stride * sizeof(std::complex<double>)) % SSE::bytes == 0)
		{
			static_for<SSE::elements<std::complex<double>>()>([&]<auto i>()
			{
				rows[i] = _mm_load_pd(&ptr_intlv[i*stride*2]);
			});
		}
		else
		{
			static_for<SSE::elements<std::complex<double>>()>([&]<auto i>()
			{
				rows[i] = _mm_loadu_pd(&ptr_intlv[i*stride*2]);
			});
		}
	}

	template<> 
	void load<float, AVX>(float* ptr, __m256* rows, const size_t stride) 
	{
		if ((stride * sizeof(float)) % AVX::bytes == 0)
		{
			static_for<AVX::elements<float>()>([&]<auto i>()
			{
				rows[i] = _mm256_load_ps(&ptr[i*stride]);
			});
		}
		else
		{
			static_for<AVX::elements<float>()>([&]<auto i>()
			{
				rows[i] = _mm256_loadu_ps(&ptr[i*stride]);
			});
		}
	}

	template<> 
	void load<double, AVX>(double* ptr, __m256d* rows, const size_t stride) 
	{
		if ((stride * sizeof(double)) % AVX::bytes == 0)
		{
			static_for<AVX::elements<double>()>([&]<auto i>()
			{
				rows[i] = _mm256_load_pd(&ptr[i*stride]);
			});
		}
		else
		{
			static_for<AVX::elements<double>()>([&]<auto i>()
			{
				rows[i] = _mm256_loadu_pd(&ptr[i*stride]);
			});
		}
	}

	template<> 
	void load<std::complex<float>, AVX>(std::complex<float>* ptr, __m256* rows, const size_t stride) 
	{
		float* ptr_intlv = reinterpret_cast<float*>(ptr);

		if ((stride * sizeof(std::complex<float>)) % AVX::bytes == 0)
		{
			static_for<AVX::elements<std::complex<float>>()>([&]<auto i>()
			{
				rows[i] = _mm256_load_ps(&ptr_intlv[i*stride*2]);
			});
		}
		else
		{
			static_for<AVX::elements<std::complex<float>>()>([&]<auto i>()
			{
				rows[i] = _mm256_loadu_ps(&ptr_intlv[i*stride*2]);
			});
		}
	}

	template<> 
	void load<std::complex<double>, AVX>(std::complex<double>* ptr, __m256d* rows, const size_t stride) 
	{
		double* ptr_intlv = reinterpret_cast<double*>(ptr);

		if ((stride * sizeof(std::complex<double>)) % AVX::bytes == 0)
		{
			static_for<AVX::elements<std::complex<double>>()>([&]<auto i>()
			{
				rows[i] = _mm256_load_pd(&ptr_intlv[i*stride*2]);
			});
		}
		else
		{
			static_for<AVX::elements<std::complex<double>>()>([&]<auto i>()
			{
				rows[i] = _mm256_loadu_pd(&ptr_intlv[i*stride*2]);
			});
		}
	}

	template<> 
	void load<float, AVX512>(float* ptr, __m512* rows, const size_t stride) 
	{
		if ((stride * sizeof(float)) % AVX512::bytes == 0)
		{
			static_for<AVX512::elements<float>()>([&]<auto i>()
			{
				rows[i] = _mm512_load_ps(&ptr[i*stride]);
			});
		}
		else
		{
			static_for<AVX512::elements<float>()>([&]<auto i>()
			{
				rows[i] = _mm512_loadu_ps(&ptr[i*stride]);
			});
		}
	}

	template<> 
	void load<double, AVX512>(double* ptr, __m512d* rows, const size_t stride) 
	{
		if ((stride * sizeof(double)) % AVX512::bytes == 0)
		{
			static_for<AVX512::elements<double>()>([&]<auto i>()
			{
				rows[i] = _mm512_load_pd(&ptr[i*stride]);
			});
		}
		else
		{
			static_for<AVX512::elements<double>()>([&]<auto i>()
			{
				rows[i] = _mm512_loadu_pd(&ptr[i*stride]);
			});
		}
	}

	template<> 
	void load<std::complex<float>, AVX512>(std::complex<float>* ptr, __m512* rows, const size_t stride) 
	{
		float* ptr_intlv = reinterpret_cast<float*>(ptr);

		if ((stride * sizeof(std::complex<float>)) % AVX512::bytes == 0)
		{
			static_for<AVX512::elements<std::complex<float>>()>([&]<auto i>()
			{
				rows[i] = _mm512_load_ps(&ptr_intlv[i*stride*2]);
			});
		}
		else
		{
			static_for<AVX512::elements<std::complex<float>>()>([&]<auto i>()
			{
				rows[i] = _mm512_loadu_ps(&ptr_intlv[i*stride*2]);
			});
		}
	}

	template<> 
	void load<std::complex<double>, AVX512>(std::complex<double>* ptr, __m512d* rows, const size_t stride) 
	{
		double* ptr_intlv = reinterpret_cast<double*>(ptr);

		if ((stride * sizeof(std::complex<double>)) % AVX512::bytes == 0)
		{
			static_for<AVX512::elements<std::complex<double>>()>([&]<auto i>()
			{
				rows[i] = _mm512_load_pd(&ptr_intlv[i*stride*2]);
			});
		}
		else
		{
			static_for<AVX512::elements<std::complex<double>>()>([&]<auto i>()
			{
				rows[i] = _mm512_loadu_pd(&ptr_intlv[i*stride*2]);
			});
		}
	}

/* STORE */
	template<> 
	void store<float, SSE>(float* ptr, __m128* rows, const size_t stride) 
	{
		if ((stride * sizeof(float)) % SSE::bytes == 0)
		{
			static_for<SSE::elements<float>()>([&]<auto i>()
			{
				_mm_store_ps(&ptr[i*stride], rows[i]);
			});
		}
		else
		{
			static_for<SSE::elements<float>()>([&]<auto i>()
			{
				_mm_storeu_ps(&ptr[i*stride], rows[i]);
			});
		}
	}

	template<> 
	void store<double, SSE>(double* ptr, __m128d* rows, const size_t stride) 
	{
		if ((stride * sizeof(double)) % SSE::bytes == 0)
		{
			static_for<SSE::elements<double>()>([&]<auto i>()
			{
				_mm_store_pd(&ptr[i*stride], rows[i]);
			});
		}
		else
		{
			static_for<SSE::elements<double>()>([&]<auto i>()
			{
				_mm_storeu_pd(&ptr[i*stride], rows[i]);
			});
		}
	}

	template<> 
	void store<std::complex<float>, SSE>(std::complex<float>* ptr, __m128* rows, const size_t stride) 
	{
		float* ptr_intlv = reinterpret_cast<float*>(ptr);

		if ((stride * sizeof(std::complex<float>)) % SSE::bytes == 0)
		{
			static_for<SSE::elements<std::complex<float>>()>([&]<auto i>()
			{
				_mm_store_ps(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
		else
		{
			static_for<SSE::elements<std::complex<float>>()>([&]<auto i>()
			{
				_mm_storeu_ps(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
	}

	template<> 
	void store<std::complex<double>, SSE>(std::complex<double>* ptr, __m128d* rows, const size_t stride) 
	{
		double* ptr_intlv = reinterpret_cast<double*>(ptr);

		if ((stride * sizeof(std::complex<double>)) % SSE::bytes == 0)
		{
			static_for<SSE::elements<std::complex<double>>()>([&]<auto i>()
			{
				_mm_store_pd(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
		else
		{
			static_for<SSE::elements<std::complex<double>>()>([&]<auto i>()
			{
				_mm_storeu_pd(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
	}

	template<> 
	void store<float, AVX>(float* ptr, __m256* rows, const size_t stride) 
	{
		if ((stride * sizeof(float)) % AVX::bytes == 0)
		{
			static_for<AVX::elements<float>()>([&]<auto i>()
			{
				_mm256_store_ps(&ptr[i*stride], rows[i]);
			});
		}
		else
		{
			static_for<AVX::elements<float>()>([&]<auto i>()
			{
				_mm256_storeu_ps(&ptr[i*stride], rows[i]);
			});
		}
	}

	template<> 
	void store<double, AVX>(double* ptr, __m256d* rows, const size_t stride) 
	{
		if ((stride * sizeof(double)) % AVX::bytes == 0)
		{
			static_for<AVX::elements<double>()>([&]<auto i>()
			{
				_mm256_store_pd(&ptr[i*stride], rows[i]);
			});
		}
		else
		{
			static_for<AVX::elements<double>()>([&]<auto i>()
			{
				_mm256_storeu_pd(&ptr[i*stride], rows[i]);
			});
		}
	}

	template<> 
	void store<std::complex<float>, AVX>(std::complex<float>* ptr, __m256* rows, const size_t stride) 
	{
		float* ptr_intlv = reinterpret_cast<float*>(ptr);

		if ((stride * sizeof(std::complex<float>)) % AVX::bytes == 0)
		{
			static_for<AVX::elements<std::complex<float>>()>([&]<auto i>()
			{
				_mm256_store_ps(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
		else
		{
			static_for<AVX::elements<std::complex<float>>()>([&]<auto i>()
			{
				_mm256_storeu_ps(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
	}

	template<> 
	void store<std::complex<double>, AVX>(std::complex<double>* ptr, __m256d* rows, const size_t stride) 
	{
		double* ptr_intlv = reinterpret_cast<double*>(ptr);

		if ((stride * sizeof(std::complex<double>)) % AVX::bytes == 0)
		{
			static_for<AVX::elements<std::complex<double>>()>([&]<auto i>()
			{
				_mm256_store_pd(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
		else
		{
			static_for<AVX::elements<std::complex<double>>()>([&]<auto i>()
			{
				_mm256_storeu_pd(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
	}

	template<> 
	void store<float, AVX512>(float* ptr, __m512* rows, const size_t stride) 
	{
		if ((stride * sizeof(float)) % AVX512::bytes == 0)
		{
			static_for<AVX512::elements<float>()>([&]<auto i>()
			{
				_mm512_store_ps(&ptr[i*stride], rows[i]);
			});
		}
		else
		{
			static_for<AVX512::elements<float>()>([&]<auto i>()
			{
				_mm512_storeu_ps(&ptr[i*stride], rows[i]);
			});
		}
	}

	template<> 
	void store<double, AVX512>(double* ptr, __m512d* rows, const size_t stride) 
	{
		if ((stride * sizeof(double)) % AVX512::bytes == 0)
		{
			static_for<AVX512::elements<double>()>([&]<auto i>()
			{
				_mm512_store_pd(&ptr[i*stride], rows[i]);
			});
		}
		else
		{
			static_for<AVX512::elements<double>()>([&]<auto i>()
			{
				_mm512_storeu_pd(&ptr[i*stride], rows[i]);
			});
		}
	}

	template<> 
	void store<std::complex<float>, AVX512>(std::complex<float>* ptr, __m512* rows, const size_t stride) 
	{
		float* ptr_intlv = reinterpret_cast<float*>(ptr);

		if ((stride * sizeof(std::complex<float>)) % AVX512::bytes == 0)
		{
			static_for<AVX512::elements<std::complex<float>>()>([&]<auto i>()
			{
				_mm512_store_ps(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
		else
		{
			static_for<AVX512::elements<std::complex<float>>()>([&]<auto i>()
			{
				_mm512_storeu_ps(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
	}

	template<> 
	void store<std::complex<double>, AVX512>(std::complex<double>* ptr, __m512d* rows, const size_t stride) 
	{
		double* ptr_intlv = reinterpret_cast<double*>(ptr);

		if ((stride * sizeof(std::complex<double>)) % AVX512::bytes == 0)
		{
			static_for<AVX512::elements<std::complex<double>>()>([&]<auto i>()
			{
				_mm512_store_pd(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
		else
		{
			static_for<AVX512::elements<std::complex<double>>()>([&]<auto i>()
			{
				_mm512_storeu_pd(&ptr_intlv[i*stride*2], rows[i]);
			});
		}
	}
} //namespace damm

