#ifndef __MULTIPLY_H__
#define __MULTIPLY_H__
/**
 * \file multiply.h
 * \brief definitions for multiplication utilities 
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

#include <common.h>
#include <carray.h>
#include <transpose.h>

namespace damm
{
	/** 
	 * \brief kernal for multiply_block. 
	 * Low level function not intended for the public API.
	 * This function can be compile time switched for cache efficiency
	 * TR = true implies the transpose of B is provided
	 * In other words, the transpose of the B matrix being multiplied with A is provided instead of B.
	 * Providing the transpose of B in lieu of B preserves cache coherence with a more efficient memory access pattern.    
	 * */
	template <typename T, bool TR=false>
	inline void
	_multiply_block(T** A, T** B, T** C, 
					const size_t I, const size_t J, const size_t K,
					const size_t M, const size_t N, const size_t P)
	{
		for(size_t i = 0; i < M; i++) 
			for(size_t j = 0; j < P; j++)
				for(size_t k = 0; k < N; ++k)
				{
					if constexpr (TR)
						C[I + i][J + j] += A[I + i][K + k] * B[J + j][K + k];
					else 
						C[I + i][J + j] += A[I + i][K + k] * B[K + k][J + j];
				}
	}
	/**
	 * \brief Multiplication of a matrix using blocks.
	 *  Block width can be specified at instantiation. 
	 *  Block width should be selected for fitting in L1 cache.
	 *  Leave space in the L1 cache for both A, and B.
	 * 
	 * \param A the outer matrix to multiply (MxN)
	 * \param B the inner matrix to multiply (NxP)
	 * \param C storage for the product A*B (MxP)
	 * \param M the stride of A
	 * \param N the columns of a, the stride of B
	 * \param P the columns of B
	 * 
	 * \note internally 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template <typename T, bool TR=false, const size_t block_size=32>
	inline void
	multiply_block(T** A, T** B, T** C, const size_t M, const size_t N, const size_t P)
	{

		for (size_t i = 0; i < M; i += block_size)
			for (size_t j = 0; j < P; j += block_size) 
				for (size_t k = 0; k < N; k += block_size)
				{
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - k);
					size_t p = std::min(block_size, P - j);
					_multiply_block<T, TR>(A, B, C, i, j, k, m, n, p);
				}
	}

	/**
	 * \brief Multiplication of a matrix using SSE intrinsics.
	 * 
	 *  float multiplication corresponds to a 4x4 parallel block multiplication
	 *  double multiplication corresponds to a 2x2 parallel block multiplication
	 * 
	 * \param A the outer matrix to multiply (MxN)
	 * \param B the inner matrix to multiply (NxP)
	 * \param C storage for the product A*B (MxP)
	 * \param M the stride of A
	 * \param N the columns of a, the stride of B
	 * \param P the columns of B
	 * 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template <typename T>
	inline void 
	_multiply_block_sse(T* A, T* B, T* C, const size_t M, const size_t N, const size_t P);

	/**
	 * \brief Multiplication of a matrix using SSE intrinsics.
	 * 
	 *  float multiplication corresponds to a 8x8 parallel block multiplication
	 *  double multiplication corresponds to a 4x4 parallel block multiplication
	 * 
	 * \param A the outer matrix to multiply (MxN)
	 * \param B the inner matrix to multiply (NxP)
	 * \param C storage for the product A*B (MxP)
	 * \param M the stride of A
	 * \param N the columns of a, the stride of B
	 * \param P the columns of B
	 * 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template <typename T>
	inline void 
	_multiply_block_avx(T* A, T* B, T* C, const size_t M, const size_t N, const size_t P);

	/**
	 * \brief Multiplication of a matrix using SSE intrinsics.
	 * 
	 *  float multiplication corresponds to a 16x16 parallel block multiplication
	 *  double multiplication corresponds to a 8x8 parallel block multiplication
	 * 
	 * \param A the outer matrix to multiply (MxN)
	 * \param B the inner matrix to multiply (NxP)
	 * \param C storage for the product A*B (MxP)
	 * \param M the stride of A
	 * \param N the columns of a, the stride of B
	 * \param P the columns of B
	 * 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template <typename T>
	inline void 
	_multiply_block_avx512(T* A, T* B, T* C, const size_t M, const size_t N, const size_t P);


	/**
	 * \brief Multiply matrix using SIMD intrinsics.
	 *  Specify the SIMD type using the template parameter.
	 *  This should always be used as the entry point to the _multiply_block_ simd functions.  
	 * 
	 * \param A the outer matrix to multiply (MxN)
	 * \param B the inner matrix to multiply (NxP)
	 * \param C storage for the product A*B (MxP)
	 * \param M the stride of A
	 * \param N the columns of a, the stride of B
	 * \param P the columns of B
	 * 
	 * \note Assymetric matrices are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template<typename T, SIMD S> 
	inline void
	multiply_block_simd(T** A, T** B, T** C, const size_t M, const size_t N, const size_t P)
	{
		constexpr size_t block_size = static_cast<size_t>(S/sizeof(T));
		
		//here we transpose B to improve cache locality and reduce the instruction set of the inner simd operations
		//I should remove the implicit carray dependence here. 
		//the sse kernel is not branched -> define multiply macros for both B and Bt 
		//just find another way to transpose within this function and give the user the option to transpose out of this function. 
		carray<T, 2, S> Bt(P, N);

		transpose_block_simd<T, S>(&B[0], &Bt[0], N, P);
		T* B0 = Bt.begin(); 
		T* A0 = A[0];
		T* C0 = C[0];

		const size_t simd_rows_M = M - (M % block_size);
		const size_t simd_cols_P = P - (P % block_size);
		const size_t simd_inner_N = N - (N % block_size);
			
		for (size_t i = 0; i + block_size <= M; i += block_size) 
			for (size_t j = 0; j + block_size <= P; j += block_size) 
				for (size_t k = 0; k + block_size <= N; k += block_size) 
				{
					if constexpr (S == SSE)
						_multiply_block_sse<T>(&A0[i * N + k], &B0[j * N + k], &C0[i * P + j], N, N, P);
					if constexpr (S == AVX)
						_multiply_block_avx<T>(&A0[i * N + k], &B0[j * N + k], &C0[i * P + j], N, N, P);
					if constexpr (S == AVX512)
						_multiply_block_avx512<T>(&A0[i * N + k], &B0[j * N + k], &C0[i * P + j], N, N, P);
				}

		// remainder rows in A
		if (M % block_size != 0) {
			const size_t rem_rows = M % block_size;
			_multiply_block<T, true>(A, &Bt[0], C, simd_rows_M, 0, 0, rem_rows, N, P);
		}

		// remainder columns in B
		if (P % block_size != 0) {
			const size_t rem_cols = P % block_size;
			_multiply_block<T, true>(A, &Bt[0], C, 0, simd_cols_P, 0, simd_rows_M, N, rem_cols);
		}

		// remainder columns of A and rows of B
		if (N % block_size != 0) {
			const size_t rem_inner = N % block_size;
			_multiply_block<T, true>(A, &Bt[0], C, 0, 0, simd_inner_N, simd_rows_M, rem_inner, simd_cols_P);
		}
	}

}//namespace damm

#endif //__MULTIPLY_H__