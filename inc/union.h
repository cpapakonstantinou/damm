#ifndef __UNION_H__
#define __UNION_H__
/**
 * \file union.h
 * \brief definitions for union utilities 
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
#include <functional>

// notes to write out in more detail later...
// union can be add / subtract // or hadamard product for pointwise multiplication... 
// Unions are not defined in the strict set-theory sense but in the computational sense
// It's a merge over an index domain where the union is performed by an arbitrary arithmetic operator
namespace damm
{
	/** 
	 * \brief kernal for union_block. 
	 * Low level function not intended for the public API.
	 * */
	template <typename T, typename O>
	inline __attribute__((always_inline)) 
	void
	_union_block(T** A, T** B, T** C, 
					const size_t I, const size_t J,
					const size_t M, const size_t N)
	{
		for(size_t i = 0; i < M; i++) 
			for(size_t j = 0; j < N; j++)
				C[I+i][J+j] = O()(A[I+i][J+j], B[I+i][J+j]);
	}

	/**
	 * \brief Union of a matrix using blocks.
	 *  Block width can be specified at instantiation. 
	 *  Block width should be selected for fitting in L1 cache.
	 *  Leave space in the L1 cache for both A, and B.
	 * 
	 *  Union operation must be one of std::plus, std::minus, std::multiplies
	 * 
	 * \param A the outer matrix to union 
	 * \param B the inner matrix to union
	 * \param C storage for the union
	 * \param M the rows of the domain
	 * \param N the columns of the domain
	 * 
	 * \note Matrices must have the same dimension
	 */
	template <typename T, typename O, const size_t block_size=32>	
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::minus<>> ||
		std::same_as<O, std::multiplies<>>
	)
	inline void
	union_block(T** A, T** B, T** C, const size_t M, const size_t N)
	{
		for (size_t i = 0; i < M; i += block_size)
			for (size_t j = 0; j < N; j += block_size)
			{ 
				size_t m = std::min(block_size, M - i);
				size_t n = std::min(block_size, N - j);
				_union_block<T, O>(A, B, C, i, j, m, n);
			}
	}

	/**
	 * \brief Union of a matrix using SSE intrinsics.
	 * 
	 *  float corresponds to a 4x4 parallel block union
	 *  double corresponds to a 2x2 parallel block union
	 * 
	 *  Union operation must be one of std::plus, std::minus, std::multiplies
	 * 
	 * \param A the outer matrix to union 
	 * \param B the inner matrix to union
	 * \param C storage for the union
	 * \param M the rows of the domain
	 * \param N the columns of the domain
	 * 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template <typename T, typename O>
	inline void 
	_union_block_sse(T* A, T* B, T* C, const size_t M);

	/**
	 * \brief Multiplication of a matrix using SSE intrinsics.
	 * 
	 *  float corresponds to a 8x8 parallel block union
	 *  double corresponds to a 4x4 parallel block union
	 * 
	 *  Union operation must be one of std::plus, std::minus, std::multiplies
	 * 
	 * \param A the outer matrix to union 
	 * \param B the inner matrix to union
	 * \param C storage for the union
	 * \param M the rows of the domain
	 * \param N the columns of the domain
	 * 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template <typename T, typename O>
	inline void 
	_union_block_avx(T* A, T* B, T* C, const size_t M);

	/**
	 * \brief Union of a matrix using SSE intrinsics.
	 * 
	 *  float multiplication corresponds to a 16x16 parallel block union
	 *  double multiplication corresponds to a 8x8 parallel block union
	 * 
	 *  Union operation must be one of std::plus, std::minus, std::multiplies
	 * 
	 * \param A the outer matrix to union 
	 * \param B the inner matrix to union
	 * \param C storage for the union
	 * \param M the rows of the domain
	 * \param N the columns of the domain
	 * 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template <typename T, typename O>
	inline void 
	_union_block_avx512(T* A, T* B, T* C, const size_t M);

	/**
	 * \brief Union matrix using SIMD intrinsics.
	 *  Specify the SIMD type using the template parameter.
	 *  This should always be used as the entry point to the *multiply*block_ simd functions.  
	 * 
	 *  Union operation must be one of std::plus, std::minus, std::multiplies
	 * 
	 * \param A the outer matrix to union 
	 * \param B the inner matrix to union
	 * \param C storage for the union
	 * \param M the rows of the domain
	 * \param N the columns of the domain
	 * 
	 * \note Asymmetric matrices are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 */
	template<typename T, typename O, SIMD S>
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::minus<>> ||
		std::same_as<O, std::multiplies<>>
	) 
	inline void
	union_block_simd(T** A, T** B, T** C, const size_t M, const size_t N)
	{
		constexpr const size_t block_size = static_cast<size_t>(S/sizeof(T));
		
		T* A0 = &A[0][0];
		T* B0 = &B[0][0];
		T* C0 = &C[0][0];
		
		const size_t total_elements = M * N;
		const size_t full_blocks = total_elements / block_size; 
		
		for (size_t block = 0; block < full_blocks; block++) 
		{
			size_t offset = block * block_size;
			if constexpr (S == SSE)
				_union_block_sse<T, O>(&A0[offset], &B0[offset], &C0[offset], block_size);
			else if constexpr (S == AVX)
				_union_block_avx<T, O>(&A0[offset], &B0[offset], &C0[offset], block_size);
			else if constexpr (S == AVX512)
				_union_block_avx512<T, O>(&A0[offset], &B0[offset], &C0[offset], block_size);
		}
		
		// Handle remainder
		for (size_t i = full_blocks * block_size; i < total_elements; i++)
			C0[i] = O()(A0[i], B0[i]);
	}

}//namespace damm

#endif //__UNION_H__