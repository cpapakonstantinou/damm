#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

/**
 * \file transpose.h
 * \brief transpose utilities definitions
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

namespace damm
{
	/** \brief kernal for transpose_block. Low level function not intended for the public API*/
	template <typename T>
	inline void
	_transpose_block(T** A, T** B, const size_t I, const size_t J, const size_t N, const size_t M)
	{
		for(size_t i=0; i < N; i++) 
			for(size_t j=0; j < M; j++) 
				 B[J + j][I + i] = A[I + i][J + j];
	}
	/**
	 * \brief Transpose of a matrix using blocks.
	 *  Block width can be specified at instantiation. 
	 *  Block width should be selected for fitting in L1 cache.
	 *  Leave space in the L1 cache for both A, and B.
	 * 
	 * \param A the matrix to transpose
	 * \param B storage for the transpose of A 
	 * \param N the stride of A where A is row-major allocated
	 * \param M the stride of B where B is row-major allocated
	 * 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template <typename T, const size_t block_size=32>
	inline void
	transpose_block(T** A, T** B, const size_t N, const size_t M)
	{
		for (size_t i = 0; i < N; i += block_size)
			for (size_t j = 0; j < M; j += block_size) 
			{
				size_t n = std::min(block_size, N - i);
				size_t m = std::min(block_size, M - j);
				_transpose_block(A, B, i, j, n, m);
			}
	}

	/**
	 * \brief Transpose of a matrix using SSE intrinsics.
	 *  SSE registers are 128 bit wide. 
	 *  float transpose corresponds to a 4x4 parallel block transpose
	 *  double tranpose corresponds to a 2x2 parallel block transpose
	 * 
	 * \param A the matrix to transpose
	 * \param B storage for the transpose of A 
	 * \param N the stride of A where A is row-major allocated
	 * \param M the stride of B where B is row-major allocated
	 * 
	 * \note Strides which are not multiples of the register size are allowed, but lead to unaligned loads and stores and performance penalties. 
	 */
	template <typename T>
	inline void 
	_transpose_block_sse(T* A, T* B, const size_t N, const size_t M);


	/**
	 * \brief Transpose of a matrix using AVX intrinsics.
	 *  AVX registers are 256 bit wide. 
	 *  float transpose corresponds to a 4x4 parallel block transpose
	 *  double tranpose corresponds to a 2x2 parallel block transpose
	 * 
	 * \param A the matrix to transpose
	 * \param B storage for the transpose of A 
	 * \param N the stride of A where A is row-major allocated
	 * \param M the stride of B where B is row-major allocated
	 * 
	 * \note Strides which are not multiples of the register size are allowed, but lead to unaligned loads and stores and performance penalties. 
	 */
	template <typename T>
	inline void
	_transpose_block_avx(T* A, T* B, const size_t N, const size_t M);


	/**
	 * \brief Transpose of a matrix using AVX-512 intrinsics.
	 *  AVX-512 registers are 512 bit wide. 
	 *  float transpose corresponds to a 8x8 parallel block transpose
	 *  double tranpose corresponds to a 16x16 parallel block transpose
	 * 
	 * \param A the matrix to transpose
	 * \param B storage for the transpose of A 
	 * \param N the stride of A where A is row-major allocated
	 * \param M the stride of B where B is row-major allocated
	 * 
	 * \note Strides which are not multiples of the register size are allowed, but lead to unaligned loads and stores and performance penalties. 
	 */
	template <typename T>
	inline void
	_transpose_block_avx512(T* A, T* B, const size_t N, const size_t M);

	/**
	 * \brief Transpose of a matrix using SIMD intrinsics.
	 *  Specify the SIMD type using the template parameter.
	 *  This should always be used as the entry point to the _transpose_block_ simd functions.  
	 * 
	 * \param A the matrix to transpose
	 * \param B storage for the transpose of A 
	 * \param N the stride of A where A is row-major allocated
	 * \param M the stride of B where B is row-major allocated
	 * 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the register size are allowed, but lead to sequential handling of the edge cases.
	 * \note Strides which are not multiples of the register size are allowed, but lead to unaligned loads and stores and performance penalties. 
	 */
	template<typename T, SIMD S> 
	inline void
	transpose_block_simd(T** A, T** B, const size_t N, const size_t M)
	{

		constexpr size_t block_size = static_cast<size_t>(S/sizeof(T));

		T* A0 = &A[0][0];
		T* B0 = &B[0][0];

		const size_t simd_rows = N - (N % block_size);
		const size_t simd_cols = M - (M % block_size);

		for (size_t i = 0; i + block_size <= N; i += block_size)
			for (size_t j = 0; j + block_size <= M; j += block_size)
			{
				if constexpr (S == SSE)
					_transpose_block_sse<T>(&A0[i * M + j], &B0[j * N + i], M, N);
				if constexpr (S == AVX)
					_transpose_block_avx<T>(&A0[i * M + j], &B0[j * N + i], M, N);
				if constexpr (S == AVX512)
					_transpose_block_avx512<T>(&A0[i * M + j], &B0[j * N + i], M, N);
			}


		// Bottom edge (partial rows)
		if (N % block_size != 0) 
		{
			size_t rem = N % block_size;
			for (size_t j = 0; j < simd_cols; j += block_size)
				_transpose_block<T>(A, B, simd_rows, j, rem, block_size);
		}

		// Right edge (partial cols)
		if (M % block_size != 0)
		{
			size_t rem = M % block_size;
			for (size_t i = 0; i < simd_rows; i += block_size)
				_transpose_block<T>(A, B, i, simd_cols, block_size, rem);
		}

		// Bottom-right corner (partial rows + cols)
		if ((N % block_size != 0) && (M % block_size != 0))
			_transpose_block<T>(A, B, simd_rows, simd_cols, N % block_size, M % block_size);
	}
}//namespace damm
#endif //__TRANSPOSE_H__