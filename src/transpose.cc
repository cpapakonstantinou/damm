/**
 * \file transpose.cc
 * \brief transpose utilities implementations
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
// IMPLIED, INCLUDING BUT  MOT LIMITED TO THE WARRANTIES OF NERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND  MONINFRINGEMENT. IN  MO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <transpose.h>

namespace damm
{

	#define _MM_TRANSPOSE2_PD(a, b) \
		{\
			__m128d t0 = _mm_unpacklo_pd(a, b);\
			__m128d t1 = _mm_unpackhi_pd(a, b);\
			a = t0;\
			b = t1;\
		}

	#define _MM256_TRANSPOSE4_PD(row0, row1, row2, row3) \
		{\
			__m256d t0 = _mm256_unpacklo_pd(row0, row1);\
			__m256d t1 = _mm256_unpackhi_pd(row0, row1);\
			__m256d t2 = _mm256_unpacklo_pd(row2, row3);\
			__m256d t3 = _mm256_unpackhi_pd(row2, row3);\
			\
			row0 = _mm256_permute2f128_pd(t0, t2, 0x20);\
			row1 = _mm256_permute2f128_pd(t1, t3, 0x20);\
			row2 = _mm256_permute2f128_pd(t0, t2, 0x31);\
			row3 = _mm256_permute2f128_pd(t1, t3, 0x31);\
		}
	

	#define _MM256_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
		{\
			__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;\
			__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;\
			__t0 = _mm256_unpacklo_ps(row0, row1);\
			__t1 = _mm256_unpackhi_ps(row0, row1);\
			__t2 = _mm256_unpacklo_ps(row2, row3);\
			__t3 = _mm256_unpackhi_ps(row2, row3);\
			__t4 = _mm256_unpacklo_ps(row4, row5);\
			__t5 = _mm256_unpackhi_ps(row4, row5);\
			__t6 = _mm256_unpacklo_ps(row6, row7);\
			__t7 = _mm256_unpackhi_ps(row6, row7);\
			__tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));\
			__tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));\
			__tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));\
			__tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));\
			__tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));\
			__tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));\
			__tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));\
			__tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));\
			row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);\
			row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);\
			row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);\
			row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);\
			row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);\
			row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);\
			row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);\
			row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);\
		}
	#define _MM512_TRANSPOSE_PS(row0, row1, row2, row3, row4, row5, row6, row7, row8, row9, rowa, rowb, rowc, rowd, rowe, rowf) \
		{\
			__m512 t0  = _mm512_unpacklo_ps(row0, row1);\
			__m512 t1  = _mm512_unpackhi_ps(row0, row1);\
			__m512 t2  = _mm512_unpacklo_ps(row2, row3);\
			__m512 t3  = _mm512_unpackhi_ps(row2, row3);\
			__m512 t4  = _mm512_unpacklo_ps(row4, row5);\
			__m512 t5  = _mm512_unpackhi_ps(row4, row5);\
			__m512 t6  = _mm512_unpacklo_ps(row6, row7);\
			__m512 t7  = _mm512_unpackhi_ps(row6, row7);\
			__m512 t8  = _mm512_unpacklo_ps(row8, row9);\
			__m512 t9  = _mm512_unpackhi_ps(row8, row9);\
			__m512 t10 = _mm512_unpacklo_ps(rowa, rowb);\
			__m512 t11 = _mm512_unpackhi_ps(rowa, rowb);\
			__m512 t12 = _mm512_unpacklo_ps(rowc, rowd);\
			__m512 t13 = _mm512_unpackhi_ps(rowc, rowd);\
			__m512 t14 = _mm512_unpacklo_ps(rowe, rowf);\
			__m512 t15 = _mm512_unpackhi_ps(rowe, rowf);\
			__m512 s0  = _mm512_shuffle_ps(t0,  t2,  _MM_SHUFFLE(1,0,1,0));\
			__m512 s1  = _mm512_shuffle_ps(t0,  t2,  _MM_SHUFFLE(3,2,3,2));\
			__m512 s2  = _mm512_shuffle_ps(t1,  t3,  _MM_SHUFFLE(1,0,1,0));\
			__m512 s3  = _mm512_shuffle_ps(t1,  t3,  _MM_SHUFFLE(3,2,3,2));\
			__m512 s4  = _mm512_shuffle_ps(t4,  t6,  _MM_SHUFFLE(1,0,1,0));\
			__m512 s5  = _mm512_shuffle_ps(t4,  t6,  _MM_SHUFFLE(3,2,3,2));\
			__m512 s6  = _mm512_shuffle_ps(t5,  t7,  _MM_SHUFFLE(1,0,1,0));\
			__m512 s7  = _mm512_shuffle_ps(t5,  t7,  _MM_SHUFFLE(3,2,3,2));\
			__m512 s8  = _mm512_shuffle_ps(t8,  t10, _MM_SHUFFLE(1,0,1,0));\
			__m512 s9  = _mm512_shuffle_ps(t8,  t10, _MM_SHUFFLE(3,2,3,2));\
			__m512 s10 = _mm512_shuffle_ps(t9,  t11, _MM_SHUFFLE(1,0,1,0));\
			__m512 s11 = _mm512_shuffle_ps(t9,  t11, _MM_SHUFFLE(3,2,3,2));\
			__m512 s12 = _mm512_shuffle_ps(t12, t14, _MM_SHUFFLE(1,0,1,0));\
			__m512 s13 = _mm512_shuffle_ps(t12, t14, _MM_SHUFFLE(3,2,3,2));\
			__m512 s14 = _mm512_shuffle_ps(t13, t15, _MM_SHUFFLE(1,0,1,0));\
			__m512 s15 = _mm512_shuffle_ps(t13, t15, _MM_SHUFFLE(3,2,3,2));\
			__m512 v0 = _mm512_shuffle_f32x4(s0, s4, 0x44);\
			__m512 v1 = _mm512_shuffle_f32x4(s1, s5, 0x44);\
			__m512 v2 = _mm512_shuffle_f32x4(s2, s6, 0x44);\
			__m512 v3 = _mm512_shuffle_f32x4(s3, s7, 0x44);\
			__m512 v4 = _mm512_shuffle_f32x4(s0, s4, 0xEE);\
			__m512 v5 = _mm512_shuffle_f32x4(s1, s5, 0xEE);\
			__m512 v6 = _mm512_shuffle_f32x4(s2, s6, 0xEE);\
			__m512 v7 = _mm512_shuffle_f32x4(s3, s7, 0xEE);\
			__m512 v8 = _mm512_shuffle_f32x4(s8, s12, 0x44);\
			__m512 v9 = _mm512_shuffle_f32x4(s9, s13, 0x44);\
			__m512 v10 = _mm512_shuffle_f32x4(s10, s14, 0x44);\
			__m512 v11 = _mm512_shuffle_f32x4(s11, s15, 0x44);\
			__m512 v12 = _mm512_shuffle_f32x4(s8, s12, 0xEE);\
			__m512 v13 = _mm512_shuffle_f32x4(s9, s13, 0xEE);\
			__m512 v14 = _mm512_shuffle_f32x4(s10, s14, 0xEE);\
			__m512 v15 = _mm512_shuffle_f32x4(s11, s15, 0xEE);\
			row0 = _mm512_shuffle_f32x4(v0, v8, 0x88);\
			row1 = _mm512_shuffle_f32x4(v1, v9, 0x88);\
			row2 = _mm512_shuffle_f32x4(v2, v10, 0x88);\
			row3 = _mm512_shuffle_f32x4(v3, v11, 0x88);\
			row8 = _mm512_shuffle_f32x4(v4, v12, 0x88);\
			row9 = _mm512_shuffle_f32x4(v5, v13, 0x88);\
			rowa = _mm512_shuffle_f32x4(v6, v14, 0x88);\
			rowb = _mm512_shuffle_f32x4(v7, v15, 0x88);\
			row4 = _mm512_shuffle_f32x4(v0, v8, 0xDD);\
			row5 = _mm512_shuffle_f32x4(v1, v9, 0xDD);\
			row6 = _mm512_shuffle_f32x4(v2, v10, 0xDD);\
			row7 = _mm512_shuffle_f32x4(v3, v11, 0xDD);\
			rowc = _mm512_shuffle_f32x4(v4, v12, 0xDD);\
			rowd = _mm512_shuffle_f32x4(v5, v13, 0xDD);\
			rowe = _mm512_shuffle_f32x4(v6, v14, 0xDD);\
			rowf = _mm512_shuffle_f32x4(v7, v15, 0xDD);\
		}

	#define _MM512_TRANSPOSE_PD(row0, row1, row2, row3, row4, row5, row6, row7) \
		{\
		__m512d t0 = _mm512_unpacklo_pd(row0, row1);\
		__m512d t1 = _mm512_unpackhi_pd(row0, row1);\
		__m512d t2 = _mm512_unpacklo_pd(row2, row3);\
		__m512d t3 = _mm512_unpackhi_pd(row2, row3);\
		__m512d t4 = _mm512_unpacklo_pd(row4, row5);\
		__m512d t5 = _mm512_unpackhi_pd(row4, row5);\
		__m512d t6 = _mm512_unpacklo_pd(row6, row7);\
		__m512d t7 = _mm512_unpackhi_pd(row6, row7);\
		__m512d v0 = _mm512_shuffle_f64x2(t0, t2, 0x44);\
		__m512d v1 = _mm512_shuffle_f64x2(t1, t3, 0x44);\
		__m512d v2 = _mm512_shuffle_f64x2(t0, t2, 0xEE);\
		__m512d v3 = _mm512_shuffle_f64x2(t1, t3, 0xEE);\
		__m512d v4 = _mm512_shuffle_f64x2(t4, t6, 0x44);\
		__m512d v5 = _mm512_shuffle_f64x2(t5, t7, 0x44);\
		__m512d v6 = _mm512_shuffle_f64x2(t4, t6, 0xEE);\
		__m512d v7 = _mm512_shuffle_f64x2(t5, t7, 0xEE);\
		row0 = _mm512_shuffle_f64x2(v0, v4, 0x88);\
		row1 = _mm512_shuffle_f64x2(v1, v5, 0x88);\
		row4 = _mm512_shuffle_f64x2(v2, v6, 0x88);\
		row5 = _mm512_shuffle_f64x2(v3, v7, 0x88);\
		row2 = _mm512_shuffle_f64x2(v0, v4, 0xDD);\
		row3 = _mm512_shuffle_f64x2(v1, v5, 0xDD);\
		row6 = _mm512_shuffle_f64x2(v2, v6, 0xDD);\
		row7 = _mm512_shuffle_f64x2(v3, v7, 0xDD);\
	}

	#define _MM512_TRANSPOSE4_COMPLEX_PD(row0, row1, row2, row3) \
		{ \
			__m512d t0 = _mm512_shuffle_f64x2(row0, row1, 0x44); \
			__m512d t1 = _mm512_shuffle_f64x2(row0, row1, 0xEE); \
			__m512d t2 = _mm512_shuffle_f64x2(row2, row3, 0x44); \
			__m512d t3 = _mm512_shuffle_f64x2(row2, row3, 0xEE); \
			row0 = _mm512_shuffle_f64x2(t0, t2, 0x88); \
			row1 = _mm512_shuffle_f64x2(t0, t2, 0xDD); \
			row2 = _mm512_shuffle_f64x2(t1, t3, 0x88); \
			row3 = _mm512_shuffle_f64x2(t1, t3, 0xDD); \
		}
	
	template <>
	void 
	_transpose_block_sse<float>(float* A, float* B, const size_t  M, const size_t N)
	{
		alignas(16)__m128 row0, row1, row2, row3;

		if ((N * sizeof(float)) % 16 == 0)
		{
			row0 = _mm_load_ps(&A[0*N]);
			row1 = _mm_load_ps(&A[1*N]);
			row2 = _mm_load_ps(&A[2*N]);
			row3 = _mm_load_ps(&A[3*N]);
		}
		else
		{
			row0 = _mm_loadu_ps(&A[0*N]);
			row1 = _mm_loadu_ps(&A[1*N]);
			row2 = _mm_loadu_ps(&A[2*N]);
			row3 = _mm_loadu_ps(&A[3*N]);
		}

		_MM_TRANSPOSE4_PS(row0, row1, row2, row3);

		if ((M * sizeof(float)) % 16 == 0)
		{
			_mm_store_ps(&B[0*M], row0);
			_mm_store_ps(&B[1*M], row1);
			_mm_store_ps(&B[2*M], row2);
			_mm_store_ps(&B[3*M], row3);
		}
		else
		{
			_mm_storeu_ps(&B[0*M], row0);
			_mm_storeu_ps(&B[1*M], row1);
			_mm_storeu_ps(&B[2*M], row2);
			_mm_storeu_ps(&B[3*M], row3);
		}
	}

	template <>
	void 
	_transpose_block_sse<std::complex<float>>(std::complex<float>* A, std::complex<float>* B, const size_t M, const size_t N)
	{
		// Process 2x2 block of complex numbers
		alignas(16) __m128 row0, row1;
		
		float* A_intlv = reinterpret_cast<float*>(A);
		float* B_intlv = reinterpret_cast<float*>(B);
		
		// Load 2 rows of 2 complex numbers each
		if ((N * sizeof(std::complex<float>)) % 16 == 0)
		{
			row0 = _mm_load_ps(&A_intlv[0 * N * 2]);
			row1 = _mm_load_ps(&A_intlv[1 * N * 2]);
		}
		else
		{
			row0 = _mm_loadu_ps(&A_intlv[0 * N * 2]);
			row1 = _mm_loadu_ps(&A_intlv[1 * N * 2]);
		}
		
		__m128d row0_d = _mm_castps_pd(row0);
		__m128d row1_d = _mm_castps_pd(row1);
		
		_MM_TRANSPOSE2_PD(row0_d, row1_d);
		
		row0 = _mm_castpd_ps(row0_d);
		row1 = _mm_castpd_ps(row1_d);
		
		if ((M * sizeof(std::complex<float>)) % 16 == 0)
		{
			_mm_store_ps(&B_intlv[0 * M * 2], row0);
			_mm_store_ps(&B_intlv[1 * M * 2], row1);
		}
		else
		{
			_mm_storeu_ps(&B_intlv[0 * M * 2], row0);
			_mm_storeu_ps(&B_intlv[1 * M * 2], row1);
		}
	}

	template <>
	void 
	_transpose_block_sse<double>(double* A, double* B, const size_t  M, const size_t N)
	{
		alignas(16)__m128d row0, row1;

		if ((N * sizeof(double)) % 16 == 0)
		{
			row0 = _mm_load_pd(&A[0*N]);
			row1 = _mm_load_pd(&A[1*N]);
		}
		else
		{
			row0 = _mm_loadu_pd(&A[0*N]);
			row1 = _mm_loadu_pd(&A[1*N]);
		}

		 _MM_TRANSPOSE2_PD(row0, row1);

		if ((M * sizeof(double)) % 16 == 0)
		{
			_mm_store_pd(&B[0*M], row0);
			_mm_store_pd(&B[1*M], row1);
		}
		else
		{
			_mm_storeu_pd(&B[0*M], row0);
			_mm_storeu_pd(&B[1*M], row1);
		}
	}

	template <>
	void 
	_transpose_block_sse<std::complex<double>>(std::complex<double>* A, std::complex<double>* B, const size_t M, const size_t N)
	{
		alignas(16) __m128d cz;
		
		double* A_intlv = reinterpret_cast<double*>(A);
		double* B_intlv = reinterpret_cast<double*>(B);
		
		if ((N * sizeof(std::complex<double>)) % 16 == 0)
		{
			cz = _mm_load_pd(&A_intlv[0 * N * 2]);
		}
		else
		{
			cz = _mm_loadu_pd(&A_intlv[0 * N * 2]);
		}
		
		
		if ((M * sizeof(std::complex<double>)) % 16 == 0)
		{
			_mm_store_pd(&B_intlv[0 * M * 2], cz);
		}
		else
		{
			_mm_storeu_pd(&B_intlv[0 * M * 2], cz);
		}
	}

	template<>
	void
	_transpose_block_avx<float>(float* A, float* B, const size_t  M, const size_t N)
	{
		alignas(32)__m256 row0, row1, row2, row3, row4, row5, row6, row7;

		if ((N * sizeof(float)) % 32 == 0)
		{
			row0 = _mm256_load_ps(&A[0*N]);
			row1 = _mm256_load_ps(&A[1*N]);
			row2 = _mm256_load_ps(&A[2*N]);
			row3 = _mm256_load_ps(&A[3*N]);
			row4 = _mm256_load_ps(&A[4*N]);
			row5 = _mm256_load_ps(&A[5*N]);
			row6 = _mm256_load_ps(&A[6*N]);
			row7 = _mm256_load_ps(&A[7*N]);
		}
		else
		{
			row0 = _mm256_loadu_ps(&A[0*N]);
			row1 = _mm256_loadu_ps(&A[1*N]);
			row2 = _mm256_loadu_ps(&A[2*N]);
			row3 = _mm256_loadu_ps(&A[3*N]);
			row4 = _mm256_loadu_ps(&A[4*N]);
			row5 = _mm256_loadu_ps(&A[5*N]);
			row6 = _mm256_loadu_ps(&A[6*N]);
			row7 = _mm256_loadu_ps(&A[7*N]);
		}

		_MM256_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7);

		if ((M * sizeof(float)) % 32 == 0)
		{
			_mm256_store_ps(&B[0*M], row0);
			_mm256_store_ps(&B[1*M], row1);
			_mm256_store_ps(&B[2*M], row2);
			_mm256_store_ps(&B[3*M], row3);
			_mm256_store_ps(&B[4*M], row4);
			_mm256_store_ps(&B[5*M], row5);
			_mm256_store_ps(&B[6*M], row6);
			_mm256_store_ps(&B[7*M], row7);
		}
		else
		{
			_mm256_storeu_ps(&B[0*M], row0);
			_mm256_storeu_ps(&B[1*M], row1);
			_mm256_storeu_ps(&B[2*M], row2);
			_mm256_storeu_ps(&B[3*M], row3);
			_mm256_storeu_ps(&B[4*M], row4);
			_mm256_storeu_ps(&B[5*M], row5);
			_mm256_storeu_ps(&B[6*M], row6);
			_mm256_storeu_ps(&B[7*M], row7);
		}
	}

	template<>
	void
	_transpose_block_avx<std::complex<float>>(std::complex<float>* A, std::complex<float>* B, const size_t M, const size_t N)
	{
		alignas(32) __m256 row0, row1, row2, row3;
		
		float* A_intlv = reinterpret_cast<float*>(A);
		float* B_intlv = reinterpret_cast<float*>(B);
		
		if ((N * sizeof(std::complex<float>)) % 32 == 0)
		{
			row0 = _mm256_load_ps(&A_intlv[0 * N * 2]);  
			row1 = _mm256_load_ps(&A_intlv[1 * N * 2]);  
			row2 = _mm256_load_ps(&A_intlv[2 * N * 2]);  
			row3 = _mm256_load_ps(&A_intlv[3 * N * 2]);  
		}
		else
		{
			row0 = _mm256_loadu_ps(&A_intlv[0 * N * 2]);
			row1 = _mm256_loadu_ps(&A_intlv[1 * N * 2]);
			row2 = _mm256_loadu_ps(&A_intlv[2 * N * 2]);
			row3 = _mm256_loadu_ps(&A_intlv[3 * N * 2]);
		}
		
		__m256d row0_d = _mm256_castps_pd(row0);
		__m256d row1_d = _mm256_castps_pd(row1);
		__m256d row2_d = _mm256_castps_pd(row2);
		__m256d row3_d = _mm256_castps_pd(row3);
		
		_MM256_TRANSPOSE4_PD(row0_d, row1_d, row2_d, row3_d);
		
		row0 = _mm256_castpd_ps(row0_d);
		row1 = _mm256_castpd_ps(row1_d);
		row2 = _mm256_castpd_ps(row2_d);
		row3 = _mm256_castpd_ps(row3_d);
		
		if ((M * sizeof(std::complex<float>)) % 32 == 0)
		{
			_mm256_store_ps(&B_intlv[0 * M * 2], row0);
			_mm256_store_ps(&B_intlv[1 * M * 2], row1);
			_mm256_store_ps(&B_intlv[2 * M * 2], row2);
			_mm256_store_ps(&B_intlv[3 * M * 2], row3);
		}
		else
		{
			_mm256_storeu_ps(&B_intlv[0 * M * 2], row0);
			_mm256_storeu_ps(&B_intlv[1 * M * 2], row1);
			_mm256_storeu_ps(&B_intlv[2 * M * 2], row2);
			_mm256_storeu_ps(&B_intlv[3 * M * 2], row3);
		}
	}

	template<>
	void
	_transpose_block_avx<double>(double* A, double* B, const size_t  M, const size_t N)
	{
		alignas(32)__m256d row0, row1, row2, row3;

		if ((N * sizeof(double)) % 32 == 0)
		{
			row0 = _mm256_load_pd(&A[0*N]);
			row1 = _mm256_load_pd(&A[1*N]);
			row2 = _mm256_load_pd(&A[2*N]);
			row3 = _mm256_load_pd(&A[3*N]);
		}
		else
		{
			row0 = _mm256_loadu_pd(&A[0*N]);
			row1 = _mm256_loadu_pd(&A[1*N]);
			row2 = _mm256_loadu_pd(&A[2*N]);
			row3 = _mm256_loadu_pd(&A[3*N]);
		}

		_MM256_TRANSPOSE4_PD(row0, row1, row2, row3);

		if ((M * sizeof(double)) % 32 == 0)
		{
			_mm256_store_pd(&B[0*M], row0);
			_mm256_store_pd(&B[1*M], row1);
			_mm256_store_pd(&B[2*M], row2);
			_mm256_store_pd(&B[3*M], row3);
		}
		else
		{
			_mm256_storeu_pd(&B[0*M], row0);
			_mm256_storeu_pd(&B[1*M], row1);
			_mm256_storeu_pd(&B[2*M], row2);
			_mm256_storeu_pd(&B[3*M], row3);
		}
	}
	
	template<>
	void
	_transpose_block_avx<std::complex<double>>(std::complex<double>* A, std::complex<double>* B, const size_t M, const size_t N)
	{
		alignas(32) __m256d row0, row1, row2, row3;
		
		double* A_intlv = reinterpret_cast<double*>(A);
		double* B_intlv = reinterpret_cast<double*>(B);
		
		if ((N * sizeof(std::complex<double>)) % 32 == 0)
		{
			row0 = _mm256_load_pd(&A_intlv[0 * N * 2]);      // Row 0: [c00_re, c00_im, c01_re, c01_im]
			row1 = _mm256_load_pd(&A_intlv[1 * N * 2]);      // Row 1: [c10_re, c10_im, c11_re, c11_im]
		}
		else
		{
			row0 = _mm256_loadu_pd(&A_intlv[0 * N * 2]);
			row1 = _mm256_loadu_pd(&A_intlv[1 * N * 2]);
		}
		
		__m128d row0_lo = _mm256_castpd256_pd128(row0);        // [c00_re, c00_im]
		__m128d row0_hi = _mm256_extractf128_pd(row0, 1);      // [c01_re, c01_im]
		__m128d row1_lo = _mm256_castpd256_pd128(row1);        // [c10_re, c10_im]  
		__m128d row1_hi = _mm256_extractf128_pd(row1, 1);      // [c11_re, c11_im]
		
		row2 = _mm256_set_m128d(row1_lo, row0_lo);             // [c00_re, c00_im, c10_re, c10_im]
		row3 = _mm256_set_m128d(row1_hi, row0_hi);             // [c01_re, c01_im, c11_re, c11_im]
		
		if ((M * sizeof(std::complex<double>)) % 32 == 0)
		{
			_mm256_store_pd(&B_intlv[0 * M * 2], row2);
			_mm256_store_pd(&B_intlv[1 * M * 2], row3);
		}
		else
		{
			_mm256_storeu_pd(&B_intlv[0 * M * 2], row2);
			_mm256_storeu_pd(&B_intlv[1 * M * 2], row3);
		}
	}
	template<>
	void
	_transpose_block_avx512<float>(float* A, float* B, const size_t  M, const size_t N)
	{
		alignas(64)__m512 row0, row1, row2, row3, row4, row5, row6, row7;
		alignas(64)__m512 row8, row9, rowa, rowb, rowc, rowd, rowe, rowf;

		if (((N * sizeof(float)) % 64) == 0) 
		{
			row0 = _mm512_load_ps(&A[ 0*N]);
			row1 = _mm512_load_ps(&A[ 1*N]);
			row2 = _mm512_load_ps(&A[ 2*N]);
			row3 = _mm512_load_ps(&A[ 3*N]);
			row4 = _mm512_load_ps(&A[ 4*N]);
			row5 = _mm512_load_ps(&A[ 5*N]);
			row6 = _mm512_load_ps(&A[ 6*N]);
			row7 = _mm512_load_ps(&A[ 7*N]);
			row8 = _mm512_load_ps(&A[ 8*N]);
			row9 = _mm512_load_ps(&A[ 9*N]);
			rowa = _mm512_load_ps(&A[10*N]);
			rowb = _mm512_load_ps(&A[11*N]);
			rowc = _mm512_load_ps(&A[12*N]);
			rowd = _mm512_load_ps(&A[13*N]);
			rowe = _mm512_load_ps(&A[14*N]);
			rowf = _mm512_load_ps(&A[15*N]);
		}
		else 
		{
			row0 = _mm512_loadu_ps(&A[ 0*N]);
			row1 = _mm512_loadu_ps(&A[ 1*N]);
			row2 = _mm512_loadu_ps(&A[ 2*N]);
			row3 = _mm512_loadu_ps(&A[ 3*N]);
			row4 = _mm512_loadu_ps(&A[ 4*N]);
			row5 = _mm512_loadu_ps(&A[ 5*N]);
			row6 = _mm512_loadu_ps(&A[ 6*N]);
			row7 = _mm512_loadu_ps(&A[ 7*N]);
			row8 = _mm512_loadu_ps(&A[ 8*N]);
			row9 = _mm512_loadu_ps(&A[ 9*N]);
			rowa = _mm512_loadu_ps(&A[10*N]);
			rowb = _mm512_loadu_ps(&A[11*N]);
			rowc = _mm512_loadu_ps(&A[12*N]);
			rowd = _mm512_loadu_ps(&A[13*N]);
			rowe = _mm512_loadu_ps(&A[14*N]);
			rowf = _mm512_loadu_ps(&A[15*N]);
		}

		_MM512_TRANSPOSE_PS(row0, row1, row2, row3, row4, row5, row6, row7, row8, row9, rowa, rowb, rowc, rowd, rowe, rowf);

		if (((M * sizeof(float)) % 64) == 0) 
		{
			_mm512_store_ps(&B[ 0*M], row0);
			_mm512_store_ps(&B[ 1*M], row1);
			_mm512_store_ps(&B[ 2*M], row2);
			_mm512_store_ps(&B[ 3*M], row3);
			_mm512_store_ps(&B[ 4*M], row4);
			_mm512_store_ps(&B[ 5*M], row5);
			_mm512_store_ps(&B[ 6*M], row6);
			_mm512_store_ps(&B[ 7*M], row7);
			_mm512_store_ps(&B[ 8*M], row8);
			_mm512_store_ps(&B[ 9*M], row9);
			_mm512_store_ps(&B[10*M], rowa);
			_mm512_store_ps(&B[11*M], rowb);
			_mm512_store_ps(&B[12*M], rowc);
			_mm512_store_ps(&B[13*M], rowd);
			_mm512_store_ps(&B[14*M], rowe);
			_mm512_store_ps(&B[15*M], rowf);
		}
		else 
		{
			_mm512_storeu_ps(&B[ 0*M], row0);
			_mm512_storeu_ps(&B[ 1*M], row1);
			_mm512_storeu_ps(&B[ 2*M], row2);
			_mm512_storeu_ps(&B[ 3*M], row3);
			_mm512_storeu_ps(&B[ 4*M], row4);
			_mm512_storeu_ps(&B[ 5*M], row5);
			_mm512_storeu_ps(&B[ 6*M], row6);
			_mm512_storeu_ps(&B[ 7*M], row7);
			_mm512_storeu_ps(&B[ 8*M], row8);
			_mm512_storeu_ps(&B[ 9*M], row9);
			_mm512_storeu_ps(&B[10*M], rowa);
			_mm512_storeu_ps(&B[11*M], rowb);
			_mm512_storeu_ps(&B[12*M], rowc);
			_mm512_storeu_ps(&B[13*M], rowd);
			_mm512_storeu_ps(&B[14*M], rowe);
			_mm512_storeu_ps(&B[15*M], rowf);
		}
	}

	template<>
	void
	_transpose_block_avx512<std::complex<float>>(std::complex<float>* A, std::complex<float>* B, const size_t M, const size_t N)
	{
		alignas(64) __m512 row0, row1, row2, row3, row4, row5, row6, row7;
		
		float* A_intlv = reinterpret_cast<float*>(A);
		float* B_intlv = reinterpret_cast<float*>(B);
		
		if ((N * sizeof(std::complex<float>)) % 64 == 0)
		{
			row0 = _mm512_load_ps(&A_intlv[0 * N * 2]);  
			row1 = _mm512_load_ps(&A_intlv[1 * N * 2]);  
			row2 = _mm512_load_ps(&A_intlv[2 * N * 2]);  
			row3 = _mm512_load_ps(&A_intlv[3 * N * 2]);  
			row4 = _mm512_load_ps(&A_intlv[4 * N * 2]);  
			row5 = _mm512_load_ps(&A_intlv[5 * N * 2]);  
			row6 = _mm512_load_ps(&A_intlv[6 * N * 2]);  
			row7 = _mm512_load_ps(&A_intlv[7 * N * 2]);  
		}
		else
		{
			row0 = _mm512_loadu_ps(&A_intlv[0 * N * 2]);
			row1 = _mm512_loadu_ps(&A_intlv[1 * N * 2]);
			row2 = _mm512_loadu_ps(&A_intlv[2 * N * 2]);
			row3 = _mm512_loadu_ps(&A_intlv[3 * N * 2]);
			row4 = _mm512_loadu_ps(&A_intlv[4 * N * 2]);
			row5 = _mm512_loadu_ps(&A_intlv[5 * N * 2]);
			row6 = _mm512_loadu_ps(&A_intlv[6 * N * 2]);
			row7 = _mm512_loadu_ps(&A_intlv[7 * N * 2]);
		}
		
		__m512d row0_d = _mm512_castps_pd(row0);
		__m512d row1_d = _mm512_castps_pd(row1);
		__m512d row2_d = _mm512_castps_pd(row2);
		__m512d row3_d = _mm512_castps_pd(row3);
		__m512d row4_d = _mm512_castps_pd(row4);
		__m512d row5_d = _mm512_castps_pd(row5);
		__m512d row6_d = _mm512_castps_pd(row6);
		__m512d row7_d = _mm512_castps_pd(row7);
		
		_MM512_TRANSPOSE_PD(row0_d, row1_d, row2_d, row3_d, row4_d, row5_d, row6_d, row7_d);
		
		row0 = _mm512_castpd_ps(row0_d);
		row1 = _mm512_castpd_ps(row1_d);
		row2 = _mm512_castpd_ps(row2_d);
		row3 = _mm512_castpd_ps(row3_d);
		row4 = _mm512_castpd_ps(row4_d);
		row5 = _mm512_castpd_ps(row5_d);
		row6 = _mm512_castpd_ps(row6_d);
		row7 = _mm512_castpd_ps(row7_d);
		
		if ((M * sizeof(std::complex<float>)) % 64 == 0)
		{
			_mm512_store_ps(&B_intlv[0 * M * 2], row0);
			_mm512_store_ps(&B_intlv[1 * M * 2], row1);
			_mm512_store_ps(&B_intlv[2 * M * 2], row2);
			_mm512_store_ps(&B_intlv[3 * M * 2], row3);
			_mm512_store_ps(&B_intlv[4 * M * 2], row4);
			_mm512_store_ps(&B_intlv[5 * M * 2], row5);
			_mm512_store_ps(&B_intlv[6 * M * 2], row6);
			_mm512_store_ps(&B_intlv[7 * M * 2], row7);
		}
		else
		{
			_mm512_storeu_ps(&B_intlv[0 * M * 2], row0);
			_mm512_storeu_ps(&B_intlv[1 * M * 2], row1);
			_mm512_storeu_ps(&B_intlv[2 * M * 2], row2);
			_mm512_storeu_ps(&B_intlv[3 * M * 2], row3);
			_mm512_storeu_ps(&B_intlv[4 * M * 2], row4);
			_mm512_storeu_ps(&B_intlv[5 * M * 2], row5);
			_mm512_storeu_ps(&B_intlv[6 * M * 2], row6);
			_mm512_storeu_ps(&B_intlv[7 * M * 2], row7);
		}
	}

	template<>
	void
	_transpose_block_avx512<std::complex<double>>(std::complex<double>* A, std::complex<double>* B, const size_t M, const size_t N)
	{
		alignas(64) __m512d row0, row1, row2, row3;
		
		double* A_intlv = reinterpret_cast<double*>(A);
		double* B_intlv = reinterpret_cast<double*>(B);
		   
		if (((N * sizeof(std::complex<double>)) % 64) == 0)
		{
			row0 = _mm512_load_pd(&A_intlv[0 * N * 2]);
			row1 = _mm512_load_pd(&A_intlv[1 * N * 2]);
			row2 = _mm512_load_pd(&A_intlv[2 * N * 2]);
			row3 = _mm512_load_pd(&A_intlv[3 * N * 2]);
		}
		else
		{
			row0 = _mm512_loadu_pd(&A_intlv[0 * N * 2]);
			row1 = _mm512_loadu_pd(&A_intlv[1 * N * 2]);
			row2 = _mm512_loadu_pd(&A_intlv[2 * N * 2]);
			row3 = _mm512_loadu_pd(&A_intlv[3 * N * 2]);
		}

		_MM512_TRANSPOSE4_COMPLEX_PD(row0, row1, row2, row3);
		
		if (((M * sizeof(std::complex<double>)) % 64) == 0)
		{
			_mm512_store_pd(&B_intlv[0 * M * 2], row0);
			_mm512_store_pd(&B_intlv[1 * M * 2], row1);
			_mm512_store_pd(&B_intlv[2 * M * 2], row2);
			_mm512_store_pd(&B_intlv[3 * M * 2], row3);
		}
		else
		{
			_mm512_storeu_pd(&B_intlv[0 * M * 2], row0);
			_mm512_storeu_pd(&B_intlv[1 * M * 2], row1);
			_mm512_storeu_pd(&B_intlv[2 * M * 2], row2);
			_mm512_storeu_pd(&B_intlv[3 * M * 2], row3);
		}
	}

	template<>
	void
	_transpose_block_avx512<double>(double* A, double* B, const size_t  M, const size_t N)
	{
		alignas(64)__m512d row0, row1, row2, row3, row4, row5, row6, row7;

		if (((N * sizeof(double)) % 64) == 0)
		{
			row0 = _mm512_load_pd(&A[ 0*N]);
			row1 = _mm512_load_pd(&A[ 1*N]);
			row2 = _mm512_load_pd(&A[ 2*N]);
			row3 = _mm512_load_pd(&A[ 3*N]);
			row4 = _mm512_load_pd(&A[ 4*N]);
			row5 = _mm512_load_pd(&A[ 5*N]);
			row6 = _mm512_load_pd(&A[ 6*N]);
			row7 = _mm512_load_pd(&A[ 7*N]);
		}
		else
		{
			row0 = _mm512_loadu_pd(&A[ 0*N]);
			row1 = _mm512_loadu_pd(&A[ 1*N]);
			row2 = _mm512_loadu_pd(&A[ 2*N]);
			row3 = _mm512_loadu_pd(&A[ 3*N]);
			row4 = _mm512_loadu_pd(&A[ 4*N]);
			row5 = _mm512_loadu_pd(&A[ 5*N]);
			row6 = _mm512_loadu_pd(&A[ 6*N]);
			row7 = _mm512_loadu_pd(&A[ 7*N]);
		}

		_MM512_TRANSPOSE_PD(row0, row1, row2, row3, row4, row5, row6, row7);
		
		if (((M * sizeof(double)) % 64) == 0)
		{
			_mm512_store_pd(&B[ 0*M], row0);
			_mm512_store_pd(&B[ 1*M], row1);
			_mm512_store_pd(&B[ 2*M], row2);
			_mm512_store_pd(&B[ 3*M], row3);
			_mm512_store_pd(&B[ 4*M], row4);
			_mm512_store_pd(&B[ 5*M], row5);
			_mm512_store_pd(&B[ 6*M], row6);
			_mm512_store_pd(&B[ 7*M], row7);
		}
		else
		{
			_mm512_storeu_pd(&B[ 0*M], row0);
			_mm512_storeu_pd(&B[ 1*M], row1);
			_mm512_storeu_pd(&B[ 2*M], row2);
			_mm512_storeu_pd(&B[ 3*M], row3);
			_mm512_storeu_pd(&B[ 4*M], row4);
			_mm512_storeu_pd(&B[ 5*M], row5);
			_mm512_storeu_pd(&B[ 6*M], row6);
			_mm512_storeu_pd(&B[ 7*M], row7);
		}
	}

} //namespace damm