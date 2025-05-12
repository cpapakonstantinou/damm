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

/**
 * Computational union.
 *
 * \note 
 * A union in this context refers to a merge over an index domain, where 
 * the merge operation is defined by an arbitrary arithmetic operator 
 * (e.g., addition, subtraction, or pointwise multiplication such as 
 * the Hadamard product).
 *
 * \note
 * This is not a union in the strict set-theoretic sense. Instead, it 
 * reflects a computational interpretation: elements are combined using 
 * a defined arithmetic rule rather than simply accumulated as distinct 
 * members of a set.
 */
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
	 * \brief kernal for union_block. 
	 * Low level function not intended for the public API.
	 * */
	template <typename T, typename O>
	inline __attribute__((always_inline)) 
	void
	_union_block_flat(T* A, T* B, T* C, 
					const size_t I, const size_t J,
					const size_t M, const size_t N)
	{
		for(size_t i = 0; i < M; i++) 
			for(size_t j = 0; j < N; j++)
				C[(I+i)*M + (J+j)] = O()(A[(I+i)*M + (J+j)], B[(I+i)*M + (J+j)]);
	}

	/**
	 * \brief	Element-wise union of two 2D pointer matrices using a binary operation.
	 *
	 * Performs a blocked element-wise operation (union) on two matrices A and B,
	 * writing the result into matrix C. The operation is defined by the template
	 * parameter `O`, and must be one of `std::plus<>`, `std::minus<>`,
	 * `std::multiplies<>`, or `std::divides<>`.
	 *
	 * \tparam T			Scalar type (e.g., float or double)
	 * \tparam O			Binary operator (e.g., std::plus<>)
	 * \tparam block_size	Tile width for cache-friendly blocking (default: _block_size)
	 * \tparam threads		Number of threads for parallel execution
	 *
	 * \param A				Matrix A as array of M row pointers, each of size N
	 * \param B				Matrix B as array of M row pointers, each of size N
	 * \param C				Output matrix C as array of M row pointers, each of size N
	 * \param M				Number of rows in A, B, and C
	 * \param N				Number of columns in A, B, and C
	 *
	 * \note	Matrices must have the same shape: M × N.
	 * \note	Blocked traversal ensures improved cache locality.
	 * \note	Requires that O is one of the standard arithmetic function objects.
	 * \note	This function operates on a 2D array-of-pointers representation.
	 */
	template <typename T, typename O, const size_t block_size = _block_size, const size_t threads = _threads>	
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::minus<>> ||
		std::same_as<O, std::multiplies<>> ||
		std::same_as<O, std::divides<>>
	)
	inline void
	union_block(T** A, T** B, T** C, const size_t M, const size_t N)
	{
		parallel_for(0, M, block_size,
		[&](size_t i)
		{
			for (size_t j = 0; j < N; j += block_size)
			{ 
				size_t m = std::min(block_size, M - i);
				size_t n = std::min(block_size, N - j);
				_union_block<T, O>(A, B, C, i, j, m, n);
			}
		}, threads);
	}

	/**
	 * \brief	Element-wise union of two flat matrices using a binary operation.
	 *
	 * Performs a blocked element-wise operation (union) on flat 1D arrays A and B,
	 * storing the result in array C. The operation is defined by the template
	 * parameter `O`, and must be one of `std::plus<>`, `std::minus<>`,
	 * `std::multiplies<>`, or `std::divides<>`.
	 *
	 * \tparam T			Scalar type (e.g., float or double)
	 * \tparam O			Binary operator (e.g., std::plus<>)
	 * \tparam block_size	Tile width for cache-friendly blocking (default: _block_size)
	 * \tparam threads		Number of threads for parallel execution
	 *
	 * \param A				Pointer to matrix A, flattened in row-major order (size M × N)
	 * \param B				Pointer to matrix B, flattened in row-major order (size M × N)
	 * \param C				Pointer to output matrix C, flattened in row-major order (size M × N)
	 * \param M				Number of rows in A, B, and C
	 * \param N				Number of columns in A, B, and C
	 *
	 * \note	Input and output arrays must be dense and laid out in row-major order.
	 * \note	Matrices must have identical dimensions: M × N.
	 * \note	Blocked traversal improves L1 cache locality.
	 * \note	Requires that O is one of the standard arithmetic function objects.
	 */
	template <typename T, typename O, const size_t block_size = _block_size, const size_t threads = _threads>	
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::minus<>> ||
		std::same_as<O, std::multiplies<>> ||
		std::same_as<O, std::divides<>>
	)
	inline void
	union_block_flat(T* A, T* B, T* C, const size_t M, const size_t N)
	{
		parallel_for(0, M, block_size,
		[&](size_t i)
		{
			for (size_t j = 0; j < N; j += block_size)
			{ 
				size_t m = std::min(block_size, M - i);
				size_t n = std::min(block_size, N - j);
				_union_block_flat<T, O>(A, B, C, i, j, m, n);
			}
		}, threads);
	}

	/**
	 * \brief Element-wise union of two matrices using SSE intrinsics.
	 *
	 * Performs an element-wise binary operation on flat matrices A and B,
	 * storing the result in matrix C. SIMD vectorization is used for efficiency.
	 *
	 * \tparam T Scalar type (float or double)
	 * \tparam O Binary operator: must be one of std::plus<>, std::minus<>, or std::multiplies<>
	 *
	 * \param A Pointer to matrix A (flattened, row-major)
	 * \param B Pointer to matrix B (flattened, row-major)
	 * \param C Pointer to matrix C (flattened, row-major) for storing the result
	 *
	 * \note For float: processes 4×4 blocks in parallel (SSE 128-bit registers)
	 * \note For double: processes 2×2 blocks in parallel
	 * \note Asymmetric matrix shapes (M ≠ N) are supported.
	 * \note Matrix strides that are not multiples of the block size are allowed.
	 */
	template <typename T, typename O>
	void 
	_union_block_sse(T* A, T* B, T* C);

	/**
	 * \brief Element-wise union of two matrices using AVX intrinsics.
	 *
	 * Performs an element-wise binary operation on flat matrices A and B,
	 * storing the result in matrix C. Uses AVX 256-bit SIMD operations.
	 *
	 * \tparam T Scalar type (float or double)
	 * \tparam O Binary operator: must be one of std::plus<>, std::minus<>, or std::multiplies<>
	 *
	 * \param A Pointer to matrix A (flattened, row-major)
	 * \param B Pointer to matrix B (flattened, row-major)
	 * \param C Pointer to matrix C (flattened, row-major) for storing the result
	 *
	 * \note For float: processes 8×8 blocks in parallel (AVX 256-bit registers)
	 * \note For double: processes 4×4 blocks in parallel
	 * \note Asymmetric matrix shapes (M ≠ N) are supported.
	 * \note Matrix strides that are not multiples of the block size are allowed.
	 */
	template <typename T, typename O>
	void 
	_union_block_avx(T* A, T* B, T* C);

	/**
 	* \brief Element-wise union of two matrices using AVX-512 intrinsics.
	 *
	 * Performs an element-wise binary operation on flat matrices A and B,
	 * storing the result in matrix C. Uses AVX-512 512-bit SIMD operations.
	 *
	 * \tparam T Scalar type (float or double)
	 * \tparam O Binary operator: must be one of std::plus<>, std::minus<>, or std::multiplies<>
	 *
	 * \param A Pointer to matrix A (flattened, row-major)
	 * \param B Pointer to matrix B (flattened, row-major)
	 * \param C Pointer to matrix C (flattened, row-major) for storing the result
	 *
	 * \note For float: processes 16×16 blocks in parallel (AVX-512 512-bit registers)
	 * \note For double: processes 8×8 blocks in parallel
	 * \note Asymmetric matrix shapes (M ≠ N) are supported.
	 * \note Matrix strides that are not multiples of the block size are allowed.
	 */
	template <typename T, typename O>
	void 
	_union_block_avx512(T* A, T* B, T* C);

	/**
	 * \brief Element-wise union of two flattened matrices using SIMD intrinsics.
	 *
	 * Performs an element-wise binary operation on 1D (flattened row-major) matrices A and B,
	 * storing the result in matrix C. SIMD vectorization is selected via the
	 * template parameter `S`. This variant accepts flat memory layout directly.
	 *
	 * \tparam T Scalar type (float or double)
	 * \tparam O Binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
	 * \tparam S SIMD instruction set (SSE, AVX, AVX512)
	 * \tparam threads Number of threads used in parallel execution
	 *
	 * \param A Pointer to flattened matrix A (row-major order)
	 * \param B Pointer to flattened matrix B (row-major order)
	 * \param C Pointer to flattened matrix C for storing the result
	 * \param M Number of rows
	 * \param N Number of columns
	 *
	 * \note Matrices must have the same dimensions (M×N).
	 * \note Asymmetric matrices (M ≠ N) are supported.
	 * \note Strides not aligned with SIMD block sizes are safely handled.
	 */
	template<typename T, typename O, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::minus<>> ||
		std::same_as<O, std::multiplies<>> ||
		std::same_as<O, std::divides<>>
	) 
	inline void
	union_block_simd_flat(T* A, T* B, T* C, const size_t M, const size_t N)
	{
		constexpr size_t simd_width = static_cast<size_t>(S / sizeof(T));
		constexpr size_t cache_line_bytes = 64;
		constexpr size_t kernel_stride = cache_line_bytes / sizeof(T);

		const size_t total_elements = M * N;
		const size_t full_blocks = total_elements / kernel_stride;
		
		parallel_for(0, full_blocks, 1, [&](size_t block) 
		{
			size_t offset = block * kernel_stride;

			if constexpr (S == SSE)
				_union_block_sse<T, O>(A + offset, B + offset, C + offset);
			else if constexpr (S == AVX)
				_union_block_avx<T, O>(A + offset, B + offset, C + offset);
			else if constexpr (S == AVX512)
				_union_block_avx512<T, O>(A + offset, B + offset, C + offset);
		});

		// Remainder
		for (size_t i = full_blocks * kernel_stride; i < total_elements; ++i)
			C[i] = O{}(A[i], B[i]);
	}

	/**
	 * \brief Element-wise union of two matrices using SIMD intrinsics.
	 *
	 * Performs an element-wise binary operation on 2D matrices A and B, storing
	 * the result in matrix C. SIMD vectorization is selected at compile time via the
	 * template parameter `S`. This is the main entry point for SIMD-based union operations.
	 *
	 * \tparam T Scalar type (float or double)
	 * \tparam O Binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
	 * \tparam S SIMD instruction set (SSE, AVX, AVX512)
	 * \tparam threads Number of threads used in parallel execution
	 *
	 * \param A Pointer to a 2D matrix A (outer matrix)
	 * \param B Pointer to a 2D matrix B (inner matrix)
	 * \param C Pointer to a 2D matrix C for storing the result
	 * \param M Number of rows
	 * \param N Number of columns
	 *
	 * \note Matrices must have the same dimensions (M×N).
	 * \note Asymmetric matrices (M ≠ N) are supported.
	 * \note Strides not aligned with SIMD block sizes are safely handled.
	 */
	template<typename T, typename O, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::minus<>> ||
		std::same_as<O, std::multiplies<>> ||
		std::same_as<O, std::divides<>>
	) 
	inline void
	union_block_simd(T** A, T** B, T** C, const size_t M, const size_t N)
	{
		T* A0 = &A[0][0];
		T* B0 = &B[0][0];
		T* C0 = &C[0][0];

		union_block_simd_flat<T, O, S, block_size, threads>(A0, B0, C0, M, N);
	}


}//namespace damm

#endif //__UNION_H__