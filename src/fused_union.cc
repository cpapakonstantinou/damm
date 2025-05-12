/**
 * \file fused_union.cc
 * \brief fused union utilities implementations
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
#include <fused_union.h>

namespace damm
{
	#define SCALAR_SSE_KERNEL(SIGNATURE, POLICY, T, REG_T, MAC_T, O1, O2, UNION, FUSION) \
	template <> \
	void _fused_union_block_sse<FusionPolicy::POLICY, T, std::O1<>, std::O2<>>	 \
		SIGNATURE \
	{ \
		auto constexpr policy = FusionPolicy::POLICY; \
		alignas(16) REG_T a0, a1, a2, a3; \
		alignas(16) REG_T b0, b1, b2, b3; \
		REG_T t0, t1, t2, t3; \
		REG_T d0, d1, d2, d3; \
		_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1, a2, a3); \
		_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0, b1, b2, b3); \
		if constexpr (policy == FusionPolicy::UNION_FIRST) \
		{ \
			UNION(a0, a1, a2, a3, b0, b1, b2, b3, t0, t1, t2, t3) \
			FUSION(t0, t1, t2, t3, reinterpret_cast<const value<T>::type*>(&C), d0, d1, d2, d3) \
		} \
		else \
		{ \
			FUSION(b0, b1, b2, b3, reinterpret_cast<const value<T>::type*>(&C), t0, t1, t2, t3) \
			UNION(a0, a1, a2, a3, t0, t1, t2, t3, d0, d1, d2, d3) \
		} \
		_MM_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(D), d0, d1, d2, d3); \
	}

	#define SCALAR_AVX_KERNEL(SIGNATURE, POLICY, T, REG_T, MAC_T, O1, O2, UNION, FUSION) \
	template <> \
	void _fused_union_block_avx<FusionPolicy::POLICY, T, std::O1<>, std::O2<>>	 \
		 SIGNATURE \
	{ \
		auto constexpr policy = FusionPolicy::POLICY; \
		alignas(32) REG_T a0, a1; \
		alignas(32) REG_T b0, b1; \
		REG_T d0, d1; \
		REG_T t0, t1; \
		_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1); \
		_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0, b1); \
		if constexpr (policy == FusionPolicy::UNION_FIRST) \
		{ \
			UNION(a0, a1, b0, b1, t0, t1) \
			FUSION(t0, t1, reinterpret_cast<const value<T>::type*>(&C), d0, d1) \
		} \
		else \
		{ \
			FUSION(b0, b1, reinterpret_cast<const value<T>::type*>(&C), t0, t1) \
			UNION(a0, a1, t0, t1, d0, d1) \
		} \
		_MM256_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(D), d0, d1); \
	}

	#define SCALAR_AVX512_KERNEL(SIGNATURE, POLICY, T, REG_T, MAC_T, O1, O2, UNION, FUSION) \
	template <> \
	void _fused_union_block_avx512<FusionPolicy::POLICY, T, std::O1<>, std::O2<>>	 \
		 SIGNATURE \
	{ \
		auto constexpr policy = FusionPolicy::POLICY; \
		alignas(64) REG_T a0; \
		alignas(64) REG_T b0; \
		REG_T d0; \
		REG_T t0; \
		_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0); \
		_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0); \
		if constexpr (policy == FusionPolicy::UNION_FIRST) \
		{ \
			UNION(a0, b0, t0)\
			FUSION(t0, reinterpret_cast<const value<T>::type*>(&C), d0) \
		} \
		else \
		{ \
			FUSION(b0, reinterpret_cast<const value<T>::type*>(&C), t0) \
			UNION(a0, t0, d0) \
		} \
		_MM512_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(D), d0); \
	}

	#define SCALAR_KERNELS_RHS(T, t, P, O1, O2, S1, S2) \
		SCALAR_SSE_KERNEL((T* A, T* B, const T C, T* D), UNION_FIRST, T, __m128##t, P, O1, O2, _MM_##S1, _MM_##S2) \
		SCALAR_AVX_KERNEL((T* A, T* B, const T C, T* D), UNION_FIRST, T, __m256##t, P, O1, O2, _MM256_##S1, _MM256_##S2) \
		SCALAR_AVX512_KERNEL((T* A, T* B, const T C, T* D), UNION_FIRST, T, __m512##t, P, O1, O2, _MM512_##S1, _MM512_##S2) \
		SCALAR_SSE_KERNEL((T* A, T* B, const T C, T* D), FUSION_FIRST, T, __m128##t, P, O1, O2, _MM_##S1, _MM_##S2) \
		SCALAR_AVX_KERNEL((T* A, T* B, const T C, T* D), FUSION_FIRST, T, __m256##t, P, O1, O2, _MM256_##S1, _MM256_##S2) \
		SCALAR_AVX512_KERNEL((T* A, T* B, const T C, T* D), FUSION_FIRST, T, __m512##t, P, O1, O2, _MM512_##S1, _MM512_##S2)
	
	#define SCALAR_KERNELS_LHS(T, t, P, O1, O2, S1, S2) \
		SCALAR_SSE_KERNEL((T* A, const T C, T* B, T* D), UNION_FIRST, T, __m128##t, P, O1, O2, _MM_##S1, _MM_##S2) \
		SCALAR_AVX_KERNEL((T* A, const T C, T* B, T* D), UNION_FIRST, T, __m256##t, P, O1, O2, _MM256_##S1, _MM256_##S2) \
		SCALAR_AVX512_KERNEL((T* A, const T C, T* B, T* D), UNION_FIRST, T, __m512##t, P, O1, O2, _MM512_##S1, _MM512_##S2) \
		SCALAR_SSE_KERNEL((T* A, const T C, T* B, T* D), FUSION_FIRST, T, __m128##t, P, O1, O2, _MM_##S1, _MM_##S2) \
		SCALAR_AVX_KERNEL((T* A, const T C, T* B, T* D), FUSION_FIRST, T, __m256##t, P, O1, O2, _MM256_##S1, _MM256_##S2) \
		SCALAR_AVX512_KERNEL((T* A, const T C, T* B, T* D), FUSION_FIRST, T, __m512##t, P, O1, O2, _MM512_##S1, _MM512_##S2) \
	

	namespace scalar 
	{
		SCALAR_KERNELS_RHS(float, , PS, plus, plus, ADD_LINE_PS, ADD_LINE_S_PS) // D = (A + B) + C
		SCALAR_KERNELS_RHS(float, , PS, plus, minus, ADD_LINE_PS, SUB_LINE_S_PS) // D = (A + B) - C
		SCALAR_KERNELS_RHS(float, , PS, plus, multiplies, ADD_LINE_PS, MUL_LINE_S_PS) // D = (A + B) * D - Scaled element-wise sum
		SCALAR_KERNELS_RHS(float, , PS, plus, divides, ADD_LINE_PS, DIV_LINE_S_PS) // D = (A + B) / C
		SCALAR_KERNELS_RHS(float, , PS, minus, plus, SUB_LINE_PS, ADD_LINE_S_PS) // D = (A - B) + C
		SCALAR_KERNELS_RHS(float, , PS, minus, minus, SUB_LINE_PS, SUB_LINE_S_PS) // D = (A - B) - C
		SCALAR_KERNELS_RHS(float, , PS, minus, multiplies, SUB_LINE_PS, MUL_LINE_S_PS) // D = (A - B) * C - Scaled element-wise diff
		SCALAR_KERNELS_RHS(float, , PS, minus, divides, SUB_LINE_PS, DIV_LINE_S_PS) // D = (A - B) / C
		SCALAR_KERNELS_RHS(float, , PS, multiplies, plus, MUL_LINE_PS, ADD_LINE_S_PS) // D = (A * B) + C - Element-wise FMA
		SCALAR_KERNELS_RHS(float, , PS, multiplies, minus, MUL_LINE_PS, SUB_LINE_S_PS) // D = (A * B) - C
		SCALAR_KERNELS_RHS(float, , PS, multiplies, multiplies, MUL_LINE_PS, MUL_LINE_S_PS)	// D = (A * B) * C
		SCALAR_KERNELS_RHS(float, , PS, multiplies, divides, MUL_LINE_PS, DIV_LINE_S_PS) // D = (A * B) / C
		SCALAR_KERNELS_RHS(float, , PS, divides, plus, DIV_LINE_PS, ADD_LINE_S_PS) // D = (A / B) + C
		SCALAR_KERNELS_RHS(float, , PS, divides, minus, DIV_LINE_PS, SUB_LINE_S_PS) // D = (A / B) - C
		SCALAR_KERNELS_RHS(float, , PS, divides, multiplies, DIV_LINE_PS, MUL_LINE_S_PS) // D = (A / B) * C
		SCALAR_KERNELS_RHS(float, , PS, divides, divides, DIV_LINE_PS, DIV_LINE_S_PS) // D = (A / B) / C

		SCALAR_KERNELS_RHS(std::complex<float>, , PS, plus, plus, ADD_LINE_PS, ADD_LINE_SC_PS) // D = (A + B) + C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, plus, minus, ADD_LINE_PS, SUB_LINE_SC_PS) // D = (A + B) - C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, plus, multiplies, ADD_LINE_PS, MUL_LINE_SC_PS) // D = (A + B) * D - Scaled element-wise sum
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, plus, divides, ADD_LINE_PS, DIV_LINE_SC_PS) // D = (A + B) / C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, minus, plus, SUB_LINE_PS, ADD_LINE_SC_PS) // D = (A - B) + C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, minus, minus, SUB_LINE_PS, SUB_LINE_SC_PS) // D = (A - B) - C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, minus, multiplies, SUB_LINE_PS, MUL_LINE_SC_PS) // D = (A - B) * C - Scaled element-wise diff
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, minus, divides, SUB_LINE_PS, DIV_LINE_SC_PS) // D = (A - B) / C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, multiplies, plus, MUL_LINE_C_PS, ADD_LINE_SC_PS) // D = (A * B) + C - Element-wise FMA
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, multiplies, minus, MUL_LINE_C_PS, SUB_LINE_SC_PS) // D = (A * B) - C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, multiplies, multiplies, MUL_LINE_C_PS, MUL_LINE_SC_PS)	// D = (A * B) * C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, multiplies, divides, MUL_LINE_C_PS, DIV_LINE_SC_PS) // D = (A * B) / C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, divides, plus, DIV_LINE_C_PS, ADD_LINE_SC_PS) // D = (A / B) + C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, divides, minus, DIV_LINE_C_PS, SUB_LINE_SC_PS) // D = (A / B) - C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, divides, multiplies, DIV_LINE_C_PS, MUL_LINE_SC_PS) // D = (A / B) * C
		SCALAR_KERNELS_RHS(std::complex<float>, , PS, divides, divides, DIV_LINE_C_PS, DIV_LINE_SC_PS) // D = (A / B) / C

		SCALAR_KERNELS_RHS(double, d, PD, plus, plus, ADD_LINE_PD, ADD_LINE_S_PD) // D = (A + B) + C
		SCALAR_KERNELS_RHS(double, d, PD, plus, minus, ADD_LINE_PD, SUB_LINE_S_PD) // D = (A + B) - C
		SCALAR_KERNELS_RHS(double, d, PD, plus, multiplies, ADD_LINE_PD, MUL_LINE_S_PD) // D = (A + B) * D - Scaled element-wise sum
		SCALAR_KERNELS_RHS(double, d, PD, plus, divides, ADD_LINE_PD, DIV_LINE_S_PD) // D = (A + B) / C
		SCALAR_KERNELS_RHS(double, d, PD, minus, plus, SUB_LINE_PD, ADD_LINE_S_PD) // D = (A - B) + C
		SCALAR_KERNELS_RHS(double, d, PD, minus, minus, SUB_LINE_PD, SUB_LINE_S_PD) // D = (A - B) - C
		SCALAR_KERNELS_RHS(double, d, PD, minus, multiplies, SUB_LINE_PD, MUL_LINE_S_PD) // D = (A - B) * C - Scaled element-wise diff
		SCALAR_KERNELS_RHS(double, d, PD, minus, divides, SUB_LINE_PD, DIV_LINE_S_PD) // D = (A - B) / C
		SCALAR_KERNELS_RHS(double, d, PD, multiplies, plus, MUL_LINE_PD, ADD_LINE_S_PD) // D = (A * B) + C - Element-wise FMA
		SCALAR_KERNELS_RHS(double, d, PD, multiplies, minus, MUL_LINE_PD, SUB_LINE_S_PD) // D = (A * B) - C
		SCALAR_KERNELS_RHS(double, d, PD, multiplies, multiplies, MUL_LINE_PD, MUL_LINE_S_PD)// D = (A * B) * C
		SCALAR_KERNELS_RHS(double, d, PD, multiplies, divides, MUL_LINE_PD, DIV_LINE_S_PD)	// D = (A * B) / C
		SCALAR_KERNELS_RHS(double, d, PD, divides, plus, DIV_LINE_PD, ADD_LINE_S_PD) // D = (A / B) + C
		SCALAR_KERNELS_RHS(double, d, PD, divides, minus, DIV_LINE_PD, SUB_LINE_S_PD) // D = (A / B) - C
		SCALAR_KERNELS_RHS(double, d, PD, divides, multiplies, DIV_LINE_PD, MUL_LINE_S_PD)	// D = (A / B) * C
		SCALAR_KERNELS_RHS(double, d, PD, divides, divides, DIV_LINE_PD, DIV_LINE_S_PD) // D = (A / B) / C

		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, plus, plus, ADD_LINE_PD, ADD_LINE_SC_PD) // D = (A + B) + C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, plus, minus, ADD_LINE_PD, SUB_LINE_SC_PD) // D = (A + B) - C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, plus, multiplies, ADD_LINE_PD, MUL_LINE_SC_PD) // D = (A + B) * D - Scaled element-wise sum
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, plus, divides, ADD_LINE_PD, DIV_LINE_SC_PD) // D = (A + B) / C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, minus, plus, SUB_LINE_PD, ADD_LINE_SC_PD) // D = (A - B) + C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, minus, minus, SUB_LINE_PD, SUB_LINE_SC_PD) // D = (A - B) - C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, minus, multiplies, SUB_LINE_PD, MUL_LINE_SC_PD) // D = (A - B) * C - Scaled element-wise diff
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, minus, divides, SUB_LINE_PD, DIV_LINE_SC_PD) // D = (A - B) / C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, multiplies, plus, MUL_LINE_C_PD, ADD_LINE_SC_PD) // D = (A * B) + C - Element-wise FMA
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, multiplies, minus, MUL_LINE_C_PD, SUB_LINE_SC_PD) // D = (A * B) - C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, multiplies, multiplies, MUL_LINE_C_PD, MUL_LINE_SC_PD)// D = (A * B) * C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, multiplies, divides, MUL_LINE_C_PD, DIV_LINE_SC_PD)	// D = (A * B) / C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, divides, plus, DIV_LINE_C_PD, ADD_LINE_SC_PD) // D = (A / B) + C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, divides, minus, DIV_LINE_C_PD, SUB_LINE_SC_PD) // D = (A / B) - C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, divides, multiplies, DIV_LINE_C_PD, MUL_LINE_SC_PD)	// D = (A / B) * C
		SCALAR_KERNELS_RHS(std::complex<double>, d, PD, divides, divides, DIV_LINE_C_PD, DIV_LINE_SC_PD) // D = (A / B) / C


		SCALAR_KERNELS_LHS(float, , PS, plus, plus, ADD_LINE_PS, ADD_LINE_S_PS) // D = (A + B) + C
		SCALAR_KERNELS_LHS(float, , PS, plus, minus, ADD_LINE_PS, LHS_SUB_LINE_S_PS) // D = (A + B) - C
		SCALAR_KERNELS_LHS(float, , PS, plus, multiplies, ADD_LINE_PS, MUL_LINE_S_PS) // D = (A + B) * D - Scaled element-wise sum
		SCALAR_KERNELS_LHS(float, , PS, plus, divides, ADD_LINE_PS, LHS_DIV_LINE_S_PS) // D = (A + B) / C
		SCALAR_KERNELS_LHS(float, , PS, minus, plus, SUB_LINE_PS, ADD_LINE_S_PS) // D = (A - B) + C
		SCALAR_KERNELS_LHS(float, , PS, minus, minus, SUB_LINE_PS, LHS_SUB_LINE_S_PS) // D = (A - B) - C
		SCALAR_KERNELS_LHS(float, , PS, minus, multiplies, SUB_LINE_PS, MUL_LINE_S_PS) // D = (A - B) * C - Scaled element-wise diff
		SCALAR_KERNELS_LHS(float, , PS, minus, divides, SUB_LINE_PS, LHS_DIV_LINE_S_PS) // D = (A - B) / C
		SCALAR_KERNELS_LHS(float, , PS, multiplies, plus, MUL_LINE_PS, ADD_LINE_S_PS) // D = (A * B) + C - Element-wise FMA
		SCALAR_KERNELS_LHS(float, , PS, multiplies, minus, MUL_LINE_PS, LHS_SUB_LINE_S_PS) // D = (A * B) - C
		SCALAR_KERNELS_LHS(float, , PS, multiplies, multiplies, MUL_LINE_PS, MUL_LINE_S_PS)	// D = (A * B) * C
		SCALAR_KERNELS_LHS(float, , PS, multiplies, divides, MUL_LINE_PS, LHS_DIV_LINE_S_PS) // D = (A * B) / C
		SCALAR_KERNELS_LHS(float, , PS, divides, plus, DIV_LINE_PS, ADD_LINE_S_PS) // D = (A / B) + C
		SCALAR_KERNELS_LHS(float, , PS, divides, minus, DIV_LINE_PS, LHS_SUB_LINE_S_PS) // D = (A / B) - C
		SCALAR_KERNELS_LHS(float, , PS, divides, multiplies, DIV_LINE_PS, MUL_LINE_S_PS) // D = (A / B) * C
		SCALAR_KERNELS_LHS(float, , PS, divides, divides, DIV_LINE_PS, LHS_DIV_LINE_S_PS) // D = (A / B) / C

		SCALAR_KERNELS_LHS(std::complex<float>, , PS, plus, plus, ADD_LINE_PS, ADD_LINE_SC_PS) // D = (A + B) + C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, plus, minus, ADD_LINE_PS, LHS_SUB_LINE_SC_PS) // D = (A + B) - C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, plus, multiplies, ADD_LINE_PS, MUL_LINE_SC_PS) // D = (A + B) * D - Scaled element-wise sum
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, plus, divides, ADD_LINE_PS, LHS_DIV_LINE_SC_PS) // D = (A + B) / C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, minus, plus, SUB_LINE_PS, ADD_LINE_SC_PS) // D = (A - B) + C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, minus, minus, SUB_LINE_PS, LHS_SUB_LINE_SC_PS) // D = (A - B) - C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, minus, multiplies, SUB_LINE_PS, MUL_LINE_SC_PS) // D = (A - B) * C - Scaled element-wise diff
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, minus, divides, SUB_LINE_PS, LHS_DIV_LINE_SC_PS) // D = (A - B) / C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, multiplies, plus, MUL_LINE_C_PS, ADD_LINE_SC_PS) // D = (A * B) + C - Element-wise FMA
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, multiplies, minus, MUL_LINE_C_PS, LHS_SUB_LINE_SC_PS) // D = (A * B) - C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, multiplies, multiplies, MUL_LINE_C_PS, MUL_LINE_SC_PS)	// D = (A * B) * C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, multiplies, divides, MUL_LINE_C_PS, LHS_DIV_LINE_SC_PS) // D = (A * B) / C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, divides, plus, DIV_LINE_C_PS, ADD_LINE_SC_PS) // D = (A / B) + C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, divides, minus, DIV_LINE_C_PS, LHS_SUB_LINE_SC_PS) // D = (A / B) - C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, divides, multiplies, DIV_LINE_C_PS, MUL_LINE_SC_PS) // D = (A / B) * C
		SCALAR_KERNELS_LHS(std::complex<float>, , PS, divides, divides, DIV_LINE_C_PS, LHS_DIV_LINE_SC_PS) // D = (A / B) / C

		SCALAR_KERNELS_LHS(double, d, PD, plus, plus, ADD_LINE_PD, ADD_LINE_S_PD) // D = (A + B) + C
		SCALAR_KERNELS_LHS(double, d, PD, plus, minus, ADD_LINE_PD, LHS_SUB_LINE_S_PD) // D = (A + B) - C
		SCALAR_KERNELS_LHS(double, d, PD, plus, multiplies, ADD_LINE_PD, MUL_LINE_S_PD) // D = (A + B) * D - Scaled element-wise sum
		SCALAR_KERNELS_LHS(double, d, PD, plus, divides, ADD_LINE_PD, LHS_DIV_LINE_S_PD) // D = (A + B) / C
		SCALAR_KERNELS_LHS(double, d, PD, minus, plus, SUB_LINE_PD, ADD_LINE_S_PD) // D = (A - B) + C
		SCALAR_KERNELS_LHS(double, d, PD, minus, minus, SUB_LINE_PD, LHS_SUB_LINE_S_PD) // D = (A - B) - C
		SCALAR_KERNELS_LHS(double, d, PD, minus, multiplies, SUB_LINE_PD, MUL_LINE_S_PD) // D = (A - B) * C - Scaled element-wise diff
		SCALAR_KERNELS_LHS(double, d, PD, minus, divides, SUB_LINE_PD, LHS_DIV_LINE_S_PD) // D = (A - B) / C
		SCALAR_KERNELS_LHS(double, d, PD, multiplies, plus, MUL_LINE_PD, ADD_LINE_S_PD) // D = (A * B) + C - Element-wise FMA
		SCALAR_KERNELS_LHS(double, d, PD, multiplies, minus, MUL_LINE_PD, LHS_SUB_LINE_S_PD) // D = (A * B) - C
		SCALAR_KERNELS_LHS(double, d, PD, multiplies, multiplies, MUL_LINE_PD, MUL_LINE_S_PD)// D = (A * B) * C
		SCALAR_KERNELS_LHS(double, d, PD, multiplies, divides, MUL_LINE_PD, LHS_DIV_LINE_S_PD)	// D = (A * B) / C
		SCALAR_KERNELS_LHS(double, d, PD, divides, plus, DIV_LINE_PD, ADD_LINE_S_PD) // D = (A / B) + C
		SCALAR_KERNELS_LHS(double, d, PD, divides, minus, DIV_LINE_PD, LHS_SUB_LINE_S_PD) // D = (A / B) - C
		SCALAR_KERNELS_LHS(double, d, PD, divides, multiplies, DIV_LINE_PD, MUL_LINE_S_PD)	// D = (A / B) * C
		SCALAR_KERNELS_LHS(double, d, PD, divides, divides, DIV_LINE_PD, LHS_DIV_LINE_S_PD) // D = (A / B) / C

		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, plus, plus, ADD_LINE_PD, ADD_LINE_SC_PD) // D = (A + B) + C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, plus, minus, ADD_LINE_PD, LHS_SUB_LINE_SC_PD) // D = (A + B) - C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, plus, multiplies, ADD_LINE_PD, MUL_LINE_SC_PD) // D = (A + B) * D - Scaled element-wise sum
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, plus, divides, ADD_LINE_PD, LHS_DIV_LINE_SC_PD) // D = (A + B) / C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, minus, plus, SUB_LINE_PD, ADD_LINE_SC_PD) // D = (A - B) + C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, minus, minus, SUB_LINE_PD, LHS_SUB_LINE_SC_PD) // D = (A - B) - C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, minus, multiplies, SUB_LINE_PD, MUL_LINE_SC_PD) // D = (A - B) * C - Scaled element-wise diff
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, minus, divides, SUB_LINE_PD, LHS_DIV_LINE_SC_PD) // D = (A - B) / C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, multiplies, plus, MUL_LINE_C_PD, ADD_LINE_SC_PD) // D = (A * B) + C - Element-wise FMA
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, multiplies, minus, MUL_LINE_C_PD, LHS_SUB_LINE_SC_PD) // D = (A * B) - C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, multiplies, multiplies, MUL_LINE_C_PD, MUL_LINE_SC_PD)// D = (A * B) * C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, multiplies, divides, MUL_LINE_C_PD, LHS_DIV_LINE_SC_PD)	// D = (A * B) / C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, divides, plus, DIV_LINE_C_PD, ADD_LINE_SC_PD) // D = (A / B) + C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, divides, minus, DIV_LINE_C_PD, LHS_SUB_LINE_SC_PD) // D = (A / B) - C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, divides, multiplies, DIV_LINE_C_PD, MUL_LINE_SC_PD)	// D = (A / B) * C
		SCALAR_KERNELS_LHS(std::complex<double>, d, PD, divides, divides, DIV_LINE_C_PD, LHS_DIV_LINE_SC_PD) // D = (A / B) / C		
	} // namespace scalar


	#define MATRIX_SSE_KERNEL(POLICY, T, REG_T, MAC_T, O1, O2, UNION, FUSION) \
	template <> \
	void _fused_union_block_sse<FusionPolicy::POLICY, T, std::O1<>, std::O2<>>	 \
		(T* A, T* B, T* C, T* D) \
	{ \
		auto constexpr policy = FusionPolicy::POLICY; \
		alignas(16) REG_T a0, a1, a2, a3; \
		alignas(16) REG_T b0, b1, b2, b3; \
		alignas(16) REG_T c0, c1, c2, c3; \
		alignas(16) REG_T t0, t1, t2, t3; \
		REG_T d0, d1, d2, d3; \
		_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1, a2, a3); \
		_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0, b1, b2, b3); \
		_MM_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(C), c0, c1, c2, c3); \
		if constexpr (policy == FusionPolicy::UNION_FIRST) \
		{ \
			UNION(a0, a1, a2, a3, b0, b1, b2, b3, t0, t1, t2, t3) \
			FUSION(t0, t1, t2, t3, c0, c1, c2, c3, d0, d1, d2, d3) \
		}\
		else \
		{ \
			FUSION(b0, b1, b2, b3, c0, c1, c2, c3, t0, t1, t2, t3) \
			UNION(a0, a1, a2, a3, t0, t1, t2, t3, d0, d1, d2, d3) \
		} \
		_MM_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(D), d0, d1, d2, d3); \
	}

	#define MATRIX_AVX_KERNEL(POLICY, T, REG_T, MAC_T, O1, O2, UNION, FUSION) \
	template <> \
	void _fused_union_block_avx<FusionPolicy::POLICY, T, std::O1<>, std::O2<>>	 \
		(T* A, T* B, T* C, T* D) \
	{ \
		auto constexpr policy = FusionPolicy::POLICY; \
		alignas(32) REG_T a0, a1; \
		alignas(32) REG_T b0, b1; \
		alignas(32) REG_T c0, c1; \
		alignas(32) REG_T t0, t1; \
		REG_T d0, d1; \
		_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0, a1); \
		_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0, b1); \
		_MM256_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(C), c0, c1); \
		if constexpr (policy == FusionPolicy::UNION_FIRST) \
		{ \
			UNION(a0, a1, b0, b1, t0, t1) \
			FUSION(t0, t1, c0, c1, d0, d1) \
		} \
		else \
		{ \
			FUSION(b0, b1, c0, c1, t0, t1) \
			UNION(a0, a1, t0, t1, d0, d1) \
		} \
		_MM256_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(D), d0, d1); \
	}

	#define MATRIX_AVX512_KERNEL(POLICY, T, REG_T, MAC_T, O1, O2, UNION, FUSION) \
	template <> \
	void _fused_union_block_avx512<FusionPolicy::POLICY, T, std::O1<>, std::O2<>> \
		(T* A, T* B, T* C, T* D) \
	{ \
		auto constexpr policy = FusionPolicy::POLICY; \
		alignas(64) REG_T a0; \
		alignas(64) REG_T b0; \
		alignas(64) REG_T c0; \
		alignas(64) REG_T t0; \
		REG_T d0; \
		_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(A), a0); \
		_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(B), b0); \
		_MM512_LOAD_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(C), c0); \
		if constexpr (policy == FusionPolicy::UNION_FIRST) \
		{ \
			UNION(a0, b0, t0) \
			FUSION(t0, c0, d0) \
		} \
		else \
		{ \
			FUSION(b0, c0, t0) \
			UNION(a0, t0, d0) \
		} \
		_MM512_STORE_LINE_##MAC_T(reinterpret_cast<value<T>::type*>(D), d0); \
	}

	#define MATRIX_KERNELS(T, t, P, O1, O2, S1, S2) \
		MATRIX_SSE_KERNEL(UNION_FIRST, T, __m128##t, P, O1, O2, _MM_##S1, _MM_##S2) \
		MATRIX_AVX_KERNEL(UNION_FIRST, T, __m256##t, P, O1, O2, _MM256_##S1, _MM256_##S2) \
		MATRIX_AVX512_KERNEL(UNION_FIRST, T, __m512##t, P, O1, O2, _MM512_##S1, _MM512_##S2) \
		MATRIX_SSE_KERNEL(FUSION_FIRST, T, __m128##t, P, O1, O2, _MM_##S1, _MM_##S2) \
		MATRIX_AVX_KERNEL(FUSION_FIRST, T, __m256##t, P, O1, O2, _MM256_##S1, _MM256_##S2) \
		MATRIX_AVX512_KERNEL(FUSION_FIRST, T, __m512##t, P, O1, O2, _MM512_##S1, _MM512_##S2)

	namespace matrix 
	{
		MATRIX_KERNELS(float, , PS, plus, plus, ADD_LINE_PS, ADD_LINE_PS) // D = (A + B) + C
		MATRIX_KERNELS(float, , PS, plus, minus, ADD_LINE_PS, SUB_LINE_PS) // D = (A + B) - C
		MATRIX_KERNELS(float, , PS, plus, multiplies, ADD_LINE_PS, MUL_LINE_PS) // D = (A + B) * D - Scaled element-wise sum
		MATRIX_KERNELS(float, , PS, plus, divides, ADD_LINE_PS, DIV_LINE_PS) // D = (A + B) / C
		MATRIX_KERNELS(float, , PS, minus, plus, SUB_LINE_PS, ADD_LINE_PS) // D = (A - B) + C
		MATRIX_KERNELS(float, , PS, minus, minus, SUB_LINE_PS, SUB_LINE_PS) // D = (A - B) - C
		MATRIX_KERNELS(float, , PS, minus, multiplies, SUB_LINE_PS, MUL_LINE_PS) // D = (A - B) * C - Scaled element-wise diff
		MATRIX_KERNELS(float, , PS, minus, divides, SUB_LINE_PS, DIV_LINE_PS) // D = (A - B) / C
		MATRIX_KERNELS(float, , PS, multiplies, plus, MUL_LINE_PS, ADD_LINE_PS) // D = (A * B) + C - Element-wise FMA
		MATRIX_KERNELS(float, , PS, multiplies, minus, MUL_LINE_PS, SUB_LINE_PS) // D = (A * B) - C
		MATRIX_KERNELS(float, , PS, multiplies, multiplies, MUL_LINE_PS, MUL_LINE_PS)	// D = (A * B) * C
		MATRIX_KERNELS(float, , PS, multiplies, divides, MUL_LINE_PS, DIV_LINE_PS) // D = (A * B) / C
		MATRIX_KERNELS(float, , PS, divides, plus, DIV_LINE_PS, ADD_LINE_PS) // D = (A / B) + C
		MATRIX_KERNELS(float, , PS, divides, minus, DIV_LINE_PS, SUB_LINE_PS) // D = (A / B) - C
		MATRIX_KERNELS(float, , PS, divides, multiplies, DIV_LINE_PS, MUL_LINE_PS) // D = (A / B) * C
		MATRIX_KERNELS(float, , PS, divides, divides, DIV_LINE_PS, DIV_LINE_PS) // D = (A / B) / C

		MATRIX_KERNELS(double, d, PD, plus, plus, ADD_LINE_PD, ADD_LINE_PD) // D = (A + B) + C
		MATRIX_KERNELS(double, d, PD, plus, minus, ADD_LINE_PD, SUB_LINE_PD) // D = (A + B) - C
		MATRIX_KERNELS(double, d, PD, plus, multiplies, ADD_LINE_PD, MUL_LINE_PD) // D = (A + B) * D - Scaled element-wise sum
		MATRIX_KERNELS(double, d, PD, plus, divides, ADD_LINE_PD, DIV_LINE_PD) // D = (A + B) / C
		MATRIX_KERNELS(double, d, PD, minus, plus, SUB_LINE_PD, ADD_LINE_PD) // D = (A - B) + C
		MATRIX_KERNELS(double, d, PD, minus, minus, SUB_LINE_PD, SUB_LINE_PD) // D = (A - B) - C
		MATRIX_KERNELS(double, d, PD, minus, multiplies, SUB_LINE_PD, MUL_LINE_PD) // D = (A - B) * C - Scaled element-wise diff
		MATRIX_KERNELS(double, d, PD, minus, divides, SUB_LINE_PD, DIV_LINE_PD) // D = (A - B) / C
		MATRIX_KERNELS(double, d, PD, multiplies, plus, MUL_LINE_PD, ADD_LINE_PD) // D = (A * B) + C - Element-wise FMA
		MATRIX_KERNELS(double, d, PD, multiplies, minus, MUL_LINE_PD, SUB_LINE_PD) // D = (A * B) - C
		MATRIX_KERNELS(double, d, PD, multiplies, multiplies, MUL_LINE_PD, MUL_LINE_PD)	// D = (A * B) * C
		MATRIX_KERNELS(double, d, PD, multiplies, divides, MUL_LINE_PD, DIV_LINE_PD) // D = (A * B) / C
		MATRIX_KERNELS(double, d, PD, divides, plus, DIV_LINE_PD, ADD_LINE_PD) // D = (A / B) + C
		MATRIX_KERNELS(double, d, PD, divides, minus, DIV_LINE_PD, SUB_LINE_PD) // D = (A / B) - C
		MATRIX_KERNELS(double, d, PD, divides, multiplies, DIV_LINE_PD, MUL_LINE_PD) // D = (A / B) * C
		MATRIX_KERNELS(double, d, PD, divides, divides, DIV_LINE_PD, DIV_LINE_PD) // D = (A / B) / C

		MATRIX_KERNELS(std::complex<float>, , PS, plus, plus, ADD_LINE_PS, ADD_LINE_PS) // D = (A + B) + C
		MATRIX_KERNELS(std::complex<float>, , PS, plus, minus, ADD_LINE_PS, SUB_LINE_PS) // D = (A + B) - C
		MATRIX_KERNELS(std::complex<float>, , PS, plus, multiplies, ADD_LINE_PS, MUL_LINE_C_PS) // D = (A + B) * D - Scaled element-wise sum
		MATRIX_KERNELS(std::complex<float>, , PS, plus, divides, ADD_LINE_PS, DIV_LINE_C_PS) // D = (A + B) / C
		MATRIX_KERNELS(std::complex<float>, , PS, minus, plus, SUB_LINE_PS, ADD_LINE_PS) // D = (A - B) + C
		MATRIX_KERNELS(std::complex<float>, , PS, minus, minus, SUB_LINE_PS, SUB_LINE_PS) // D = (A - B) - C
		MATRIX_KERNELS(std::complex<float>, , PS, minus, multiplies, SUB_LINE_PS, MUL_LINE_C_PS) // D = (A - B) * C - Scaled element-wise diff
		MATRIX_KERNELS(std::complex<float>, , PS, minus, divides, SUB_LINE_PS, DIV_LINE_C_PS) // D = (A - B) / C
		MATRIX_KERNELS(std::complex<float>, , PS, multiplies, plus, MUL_LINE_C_PS, ADD_LINE_PS) // D = (A * B) + C - Element-wise FMA
		MATRIX_KERNELS(std::complex<float>, , PS, multiplies, minus, MUL_LINE_C_PS, SUB_LINE_PS) // D = (A * B) - C
		MATRIX_KERNELS(std::complex<float>, , PS, multiplies, multiplies, MUL_LINE_C_PS, MUL_LINE_C_PS)	// D = (A * B) * C
		MATRIX_KERNELS(std::complex<float>, , PS, multiplies, divides, MUL_LINE_C_PS, DIV_LINE_C_PS) // D = (A * B) / C
		MATRIX_KERNELS(std::complex<float>, , PS, divides, plus, DIV_LINE_C_PS, ADD_LINE_PS) // D = (A / B) + C
		MATRIX_KERNELS(std::complex<float>, , PS, divides, minus, DIV_LINE_C_PS, SUB_LINE_PS) // D = (A / B) - C
		MATRIX_KERNELS(std::complex<float>, , PS, divides, multiplies, DIV_LINE_C_PS, MUL_LINE_C_PS) // D = (A / B) * C
		MATRIX_KERNELS(std::complex<float>, , PS, divides, divides, DIV_LINE_C_PS, DIV_LINE_C_PS) // D = (A / B) / C

		MATRIX_KERNELS(std::complex<double>, d, PD, plus, plus, ADD_LINE_PD, ADD_LINE_PD) // D = (A + B) + C
		MATRIX_KERNELS(std::complex<double>, d, PD, plus, minus, ADD_LINE_PD, SUB_LINE_PD) // D = (A + B) - C
		MATRIX_KERNELS(std::complex<double>, d, PD, plus, multiplies, ADD_LINE_PD, MUL_LINE_C_PD) // D = (A + B) * D - Scaled element-wise sum
		MATRIX_KERNELS(std::complex<double>, d, PD, plus, divides, ADD_LINE_PD, DIV_LINE_C_PD) // D = (A + B) / C
		MATRIX_KERNELS(std::complex<double>, d, PD, minus, plus, SUB_LINE_PD, ADD_LINE_PD) // D = (A - B) + C
		MATRIX_KERNELS(std::complex<double>, d, PD, minus, minus, SUB_LINE_PD, SUB_LINE_PD) // D = (A - B) - C
		MATRIX_KERNELS(std::complex<double>, d, PD, minus, multiplies, SUB_LINE_PD, MUL_LINE_C_PD) // D = (A - B) * C - Scaled element-wise diff
		MATRIX_KERNELS(std::complex<double>, d, PD, minus, divides, SUB_LINE_PD, DIV_LINE_C_PD) // D = (A - B) / C
		MATRIX_KERNELS(std::complex<double>, d, PD, multiplies, plus, MUL_LINE_C_PD, ADD_LINE_PD) // D = (A * B) + C - Element-wise FMA
		MATRIX_KERNELS(std::complex<double>, d, PD, multiplies, minus, MUL_LINE_C_PD, SUB_LINE_PD) // D = (A * B) - C
		MATRIX_KERNELS(std::complex<double>, d, PD, multiplies, multiplies, MUL_LINE_C_PD, MUL_LINE_C_PD)	// D = (A * B) * C
		MATRIX_KERNELS(std::complex<double>, d, PD, multiplies, divides, MUL_LINE_C_PD, DIV_LINE_C_PD) // D = (A * B) / C
		MATRIX_KERNELS(std::complex<double>, d, PD, divides, plus, DIV_LINE_C_PD, ADD_LINE_PD) // D = (A / B) + C
		MATRIX_KERNELS(std::complex<double>, d, PD, divides, minus, DIV_LINE_C_PD, SUB_LINE_PD) // D = (A / B) - C
		MATRIX_KERNELS(std::complex<double>, d, PD, divides, multiplies, DIV_LINE_C_PD, MUL_LINE_C_PD) // D = (A / B) * C
		MATRIX_KERNELS(std::complex<double>, d, PD, divides, divides, DIV_LINE_C_PD, DIV_LINE_C_PD) // D = (A / B) / C
	} // namespace matrix

	#undef SCALAR_SSE_KERNEL
	#undef SCALAR_AVX_KERNEL
	#undef SCALAR_AVX512_KERNEL

	#undef SCALAR_KERNELS_RHS
	#undef SCALAR_KERNELS_LHS

	#undef MATRIX_SSE_KERNEL
	#undef MATRIX_AVX_KERNEL
	#undef MATRIX_AVX512_KERNEL
	#undef MATRIX_KERNELS

} // namespace damm