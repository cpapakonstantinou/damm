/**
 * \file __transpose_block_simd.cc
 * \brief __transpose_block_simd utilities implementations
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

	template<> 
	void __transpose_block_simd<float, SSE>(__m128* r) 
	{
		_MM_TRANSPOSE4_PS(r[0], r[1], r[2], r[3]);
	}

	template<> 
	void __transpose_block_simd<double, SSE>(__m128d* r) 
	{
		__m128d t0 = _mm_unpacklo_pd(r[0], r[1]);
		__m128d t1 = _mm_unpackhi_pd(r[0], r[1]);
		r[0] = t0;
		r[1] = t1;
	}

	template<> 
	void __transpose_block_simd<std::complex<float>, SSE>(__m128* r) 
	{
		__m128d r_d[2];
		r_d[0] = _mm_castps_pd(r[0]);
		r_d[1] = _mm_castps_pd(r[1]);

		__transpose_block_simd<double, SSE>(r_d);

		r[0] = _mm_castpd_ps(r_d[0]);
		r[1] = _mm_castpd_ps(r_d[1]);
	}

	template<> 
	void __transpose_block_simd<std::complex<double>, SSE>(__m128d* r) 
	{
		//nothing to do here 1x1 __transpose_block_simd...
	}

	template<> 
	void __transpose_block_simd<float, AVX>(__m256* r) 
	{
		__m256 t0, t1, t2, t3, t4, t5, t6, t7;
		__m256 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;
		t0 = _mm256_unpacklo_ps(r[0], r[1]);
		t1 = _mm256_unpackhi_ps(r[0], r[1]);
		t2 = _mm256_unpacklo_ps(r[2], r[3]);
		t3 = _mm256_unpackhi_ps(r[2], r[3]);
		t4 = _mm256_unpacklo_ps(r[4], r[5]);
		t5 = _mm256_unpackhi_ps(r[4], r[5]);
		t6 = _mm256_unpacklo_ps(r[6], r[7]);
		t7 = _mm256_unpackhi_ps(r[6], r[7]);
		tt0 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0));
		tt1 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2));
		tt2 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(1,0,1,0));
		tt3 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(3,2,3,2));
		tt4 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(1,0,1,0));
		tt5 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(3,2,3,2));
		tt6 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(1,0,1,0));
		tt7 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(3,2,3,2));
		r[0] = _mm256_permute2f128_ps(tt0, tt4, 0x20);
		r[1] = _mm256_permute2f128_ps(tt1, tt5, 0x20);
		r[2] = _mm256_permute2f128_ps(tt2, tt6, 0x20);
		r[3] = _mm256_permute2f128_ps(tt3, tt7, 0x20);
		r[4] = _mm256_permute2f128_ps(tt0, tt4, 0x31);
		r[5] = _mm256_permute2f128_ps(tt1, tt5, 0x31);
		r[6] = _mm256_permute2f128_ps(tt2, tt6, 0x31);
		r[7] = _mm256_permute2f128_ps(tt3, tt7, 0x31);
	}

	template<> 
	void __transpose_block_simd<double, AVX>(__m256d* r) 
	{
		__m256d t0 = _mm256_unpacklo_pd(r[0], r[1]);
		__m256d t1 = _mm256_unpackhi_pd(r[0], r[1]);
		__m256d t2 = _mm256_unpacklo_pd(r[2], r[3]);
		__m256d t3 = _mm256_unpackhi_pd(r[2], r[3]);
		r[0] = _mm256_permute2f128_pd(t0, t2, 0x20);
		r[1] = _mm256_permute2f128_pd(t1, t3, 0x20);
		r[2] = _mm256_permute2f128_pd(t0, t2, 0x31);
		r[3] = _mm256_permute2f128_pd(t1, t3, 0x31);
	}

	template<> 
	void __transpose_block_simd<std::complex<float>, AVX>(__m256* r) 
	{
		__m256d r_d[4];
		r_d[0] = _mm256_castps_pd(r[0]);
		r_d[1] = _mm256_castps_pd(r[1]);
		r_d[2] = _mm256_castps_pd(r[2]);
		r_d[3] = _mm256_castps_pd(r[3]);
		
		__transpose_block_simd<double, AVX>(r_d);
		
		r[0] = _mm256_castpd_ps(r_d[0]);
		r[1] = _mm256_castpd_ps(r_d[1]);
		r[2] = _mm256_castpd_ps(r_d[2]);
		r[3] = _mm256_castpd_ps(r_d[3]);
	}


	template<> 
	void __transpose_block_simd<std::complex<double>, AVX>(__m256d* r) 
	{
		__m128d row0_lo = _mm256_castpd256_pd128(r[0]);
		__m128d row0_hi = _mm256_extractf128_pd(r[0], 1);
		__m128d row1_lo = _mm256_castpd256_pd128(r[1]);  
		__m128d row1_hi = _mm256_extractf128_pd(r[1], 1);
		r[0] = _mm256_set_m128d(row1_lo, row0_lo);
		r[1] = _mm256_set_m128d(row1_hi, row0_hi);
	}

	template<> 
	void __transpose_block_simd<float, AVX512>(__m512* r) 
	{
		__m512 t0  = _mm512_unpacklo_ps(r[0], r[1]);
		__m512 t1  = _mm512_unpackhi_ps(r[0], r[1]);
		__m512 t2  = _mm512_unpacklo_ps(r[2], r[3]);
		__m512 t3  = _mm512_unpackhi_ps(r[2], r[3]);
		__m512 t4  = _mm512_unpacklo_ps(r[4], r[5]);
		__m512 t5  = _mm512_unpackhi_ps(r[4], r[5]);
		__m512 t6  = _mm512_unpacklo_ps(r[6], r[7]);
		__m512 t7  = _mm512_unpackhi_ps(r[6], r[7]);
		__m512 t8  = _mm512_unpacklo_ps(r[8], r[9]);
		__m512 t9  = _mm512_unpackhi_ps(r[8], r[9]);
		__m512 t10 = _mm512_unpacklo_ps(r[10], r[11]);
		__m512 t11 = _mm512_unpackhi_ps(r[10], r[11]);
		__m512 t12 = _mm512_unpacklo_ps(r[12], r[13]);
		__m512 t13 = _mm512_unpackhi_ps(r[12], r[13]);
		__m512 t14 = _mm512_unpacklo_ps(r[14], r[15]);
		__m512 t15 = _mm512_unpackhi_ps(r[14], r[15]);
		__m512 s0  = _mm512_shuffle_ps(t0,  t2,  _MM_SHUFFLE(1,0,1,0));
		__m512 s1  = _mm512_shuffle_ps(t0,  t2,  _MM_SHUFFLE(3,2,3,2));
		__m512 s2  = _mm512_shuffle_ps(t1,  t3,  _MM_SHUFFLE(1,0,1,0));
		__m512 s3  = _mm512_shuffle_ps(t1,  t3,  _MM_SHUFFLE(3,2,3,2));
		__m512 s4  = _mm512_shuffle_ps(t4,  t6,  _MM_SHUFFLE(1,0,1,0));
		__m512 s5  = _mm512_shuffle_ps(t4,  t6,  _MM_SHUFFLE(3,2,3,2));
		__m512 s6  = _mm512_shuffle_ps(t5,  t7,  _MM_SHUFFLE(1,0,1,0));
		__m512 s7  = _mm512_shuffle_ps(t5,  t7,  _MM_SHUFFLE(3,2,3,2));
		__m512 s8  = _mm512_shuffle_ps(t8,  t10, _MM_SHUFFLE(1,0,1,0));
		__m512 s9  = _mm512_shuffle_ps(t8,  t10, _MM_SHUFFLE(3,2,3,2));
		__m512 s10 = _mm512_shuffle_ps(t9,  t11, _MM_SHUFFLE(1,0,1,0));
		__m512 s11 = _mm512_shuffle_ps(t9,  t11, _MM_SHUFFLE(3,2,3,2));
		__m512 s12 = _mm512_shuffle_ps(t12, t14, _MM_SHUFFLE(1,0,1,0));
		__m512 s13 = _mm512_shuffle_ps(t12, t14, _MM_SHUFFLE(3,2,3,2));
		__m512 s14 = _mm512_shuffle_ps(t13, t15, _MM_SHUFFLE(1,0,1,0));
		__m512 s15 = _mm512_shuffle_ps(t13, t15, _MM_SHUFFLE(3,2,3,2));
		__m512 v0 = _mm512_shuffle_f32x4(s0, s4, 0x44);
		__m512 v1 = _mm512_shuffle_f32x4(s1, s5, 0x44);
		__m512 v2 = _mm512_shuffle_f32x4(s2, s6, 0x44);
		__m512 v3 = _mm512_shuffle_f32x4(s3, s7, 0x44);
		__m512 v4 = _mm512_shuffle_f32x4(s0, s4, 0xEE);
		__m512 v5 = _mm512_shuffle_f32x4(s1, s5, 0xEE);
		__m512 v6 = _mm512_shuffle_f32x4(s2, s6, 0xEE);
		__m512 v7 = _mm512_shuffle_f32x4(s3, s7, 0xEE);
		__m512 v8 = _mm512_shuffle_f32x4(s8, s12, 0x44);
		__m512 v9 = _mm512_shuffle_f32x4(s9, s13, 0x44);
		__m512 v10 = _mm512_shuffle_f32x4(s10, s14, 0x44);
		__m512 v11 = _mm512_shuffle_f32x4(s11, s15, 0x44);
		__m512 v12 = _mm512_shuffle_f32x4(s8, s12, 0xEE);
		__m512 v13 = _mm512_shuffle_f32x4(s9, s13, 0xEE);
		__m512 v14 = _mm512_shuffle_f32x4(s10, s14, 0xEE);
		__m512 v15 = _mm512_shuffle_f32x4(s11, s15, 0xEE);
		r[0] = _mm512_shuffle_f32x4(v0, v8, 0x88);
		r[1] = _mm512_shuffle_f32x4(v1, v9, 0x88);
		r[2] = _mm512_shuffle_f32x4(v2, v10, 0x88);
		r[3] = _mm512_shuffle_f32x4(v3, v11, 0x88);
		r[8] = _mm512_shuffle_f32x4(v4, v12, 0x88);
		r[9] = _mm512_shuffle_f32x4(v5, v13, 0x88);
		r[10] = _mm512_shuffle_f32x4(v6, v14, 0x88);
		r[11] = _mm512_shuffle_f32x4(v7, v15, 0x88);
		r[4] = _mm512_shuffle_f32x4(v0, v8, 0xDD);
		r[5] = _mm512_shuffle_f32x4(v1, v9, 0xDD);
		r[6] = _mm512_shuffle_f32x4(v2, v10, 0xDD);
		r[7] = _mm512_shuffle_f32x4(v3, v11, 0xDD);
		r[12] = _mm512_shuffle_f32x4(v4, v12, 0xDD);
		r[13] = _mm512_shuffle_f32x4(v5, v13, 0xDD);
		r[14] = _mm512_shuffle_f32x4(v6, v14, 0xDD);
		r[15] = _mm512_shuffle_f32x4(v7, v15, 0xDD);
	}

	template<> 
	void __transpose_block_simd<double, AVX512>(__m512d* r) 
	{
		__m512d t0 = _mm512_unpacklo_pd(r[0], r[1]);
		__m512d t1 = _mm512_unpackhi_pd(r[0], r[1]);
		__m512d t2 = _mm512_unpacklo_pd(r[2], r[3]);
		__m512d t3 = _mm512_unpackhi_pd(r[2], r[3]);
		__m512d t4 = _mm512_unpacklo_pd(r[4], r[5]);
		__m512d t5 = _mm512_unpackhi_pd(r[4], r[5]);
		__m512d t6 = _mm512_unpacklo_pd(r[6], r[7]);
		__m512d t7 = _mm512_unpackhi_pd(r[6], r[7]);
		__m512d v0 = _mm512_shuffle_f64x2(t0, t2, 0x44);
		__m512d v1 = _mm512_shuffle_f64x2(t1, t3, 0x44);
		__m512d v2 = _mm512_shuffle_f64x2(t0, t2, 0xEE);
		__m512d v3 = _mm512_shuffle_f64x2(t1, t3, 0xEE);
		__m512d v4 = _mm512_shuffle_f64x2(t4, t6, 0x44);
		__m512d v5 = _mm512_shuffle_f64x2(t5, t7, 0x44);
		__m512d v6 = _mm512_shuffle_f64x2(t4, t6, 0xEE);
		__m512d v7 = _mm512_shuffle_f64x2(t5, t7, 0xEE);
		r[0] = _mm512_shuffle_f64x2(v0, v4, 0x88);
		r[1] = _mm512_shuffle_f64x2(v1, v5, 0x88);
		r[2] = _mm512_shuffle_f64x2(v0, v4, 0xDD);
		r[3] = _mm512_shuffle_f64x2(v1, v5, 0xDD);
		r[4] = _mm512_shuffle_f64x2(v2, v6, 0x88);
		r[5] = _mm512_shuffle_f64x2(v3, v7, 0x88);
		r[6] = _mm512_shuffle_f64x2(v2, v6, 0xDD);
		r[7] = _mm512_shuffle_f64x2(v3, v7, 0xDD);
	}

	template<> 
	void __transpose_block_simd<std::complex<float>, AVX512>(__m512* r) 
	{
		__m512d r_d[8];
		r_d[0] = _mm512_castps_pd(r[0]);
		r_d[1] = _mm512_castps_pd(r[1]);
		r_d[2] = _mm512_castps_pd(r[2]);
		r_d[3] = _mm512_castps_pd(r[3]);
		r_d[4] = _mm512_castps_pd(r[4]);
		r_d[5] = _mm512_castps_pd(r[5]);
		r_d[6] = _mm512_castps_pd(r[6]);
		r_d[7] = _mm512_castps_pd(r[7]);
		
		__transpose_block_simd<double, AVX512>(r_d);

		r[0] = _mm512_castpd_ps(r_d[0]);
		r[1] = _mm512_castpd_ps(r_d[1]);
		r[2] = _mm512_castpd_ps(r_d[2]);
		r[3] = _mm512_castpd_ps(r_d[3]);
		r[4] = _mm512_castpd_ps(r_d[4]);
		r[5] = _mm512_castpd_ps(r_d[5]);
		r[6] = _mm512_castpd_ps(r_d[6]);
		r[7] = _mm512_castpd_ps(r_d[7]);
	}

	template<> 
	void __transpose_block_simd<std::complex<double>, AVX512>(__m512d* r) 
	{
		__m512d t0 = _mm512_shuffle_f64x2(r[0], r[1], 0x44);
		__m512d t1 = _mm512_shuffle_f64x2(r[0], r[1], 0xEE);
		__m512d t2 = _mm512_shuffle_f64x2(r[2], r[3], 0x44);
		__m512d t3 = _mm512_shuffle_f64x2(r[2], r[3], 0xEE);
		r[0] = _mm512_shuffle_f64x2(t0, t2, 0x88);
		r[1] = _mm512_shuffle_f64x2(t0, t2, 0xDD);
		r[2] = _mm512_shuffle_f64x2(t1, t3, 0x88);
		r[3] = _mm512_shuffle_f64x2(t1, t3, 0xDD);
	}

} //namespace damm