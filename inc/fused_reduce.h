#ifndef __FUSED_REDUCE_H__
#define __FUSED_REDUCE_H__
/**
 * \file fused_reduce.h
 * \brief definitions for fused reduce utilities 
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
 * \brief Fused Union-Reduce Operations
 *
 * \note
 * Fused reduce operations combine element-wise union operations with reduction
 * to eliminate intermediate storage and improve cache efficiency through loop fusion.
 * This is particularly effective for operations like dot products, sum of squares,
 * and other compound matrix operations.
 *
 * \note
 * The union operation is applied element-wise between matrices A and B,
 * followed immediately by reduction using the specified reduction operator.
 * This eliminates the need for intermediate arrays and reduces memory bandwidth.
 *
 * \note
 * Only associative reduction operators are supported for safe parallelization.
 * Union operators can be any of the standard arithmetic operations.
 */
namespace damm
{
	/** 
	 * \brief kernel for fused_reduce_block. 
	 * Low level function not intended for the public API.
	 * */
	template <typename T, typename U, typename R>
	inline __attribute__((always_inline))
	void
	_fused_reduce_block(T** A, T** B, T& r,
						const size_t I, const size_t J,
						const size_t M, const size_t N)
	{
		for (size_t i = 0; i < M; ++i)
			for (size_t j = 0; j < N; ++j)
			{
				T u = U{}(A[I + i][J + j], B[I + i][J + j]);
				r = R{}(r, u);
			}
	}
	
	/** 
	 * \brief kernel for fused_reduce_block. 
	 * Low level function not intended for the public API.
	 * */
	template <typename T, typename U, typename R>
	inline __attribute__((always_inline)) 
	void
	_fused_reduce_block(T* A, T* B, T& r, 
						const size_t I, const size_t J,
						const size_t M, const size_t N)
	{
		for(size_t i = 0; i < M; i++) 
			for(size_t j = 0; j < N; j++)
			{
				T u = U{}(A[(I+i)*N + (J+j)], B[(I+i)*N + (J+j)]);
				r = R{}(r, u);
			}
	}

	/**
	 * \brief Fused reduce of two matrices using blocked traversal.
	 * 
	 * Combines element-wise union operations with reduction in a single pass
	 * to optimize cache usage and eliminate intermediate storage. The matrices
	 * are processed in tiles of `block_size × block_size`.
	 * 
	 * The operation computes: reduce_op(union_op(A[i][j], B[i][j])) for all elements.
	 * 
	 * Supported union operations:
	 * | Operation | Supported |
	 * |-----------|-----------|
	 * | Add       | Yes       |
	 * | Subtract  | Yes       |
	 * | Multiply  | Yes       |
	 * | Divide    | Yes       |
	 * 
	 * Supported reduction operations (must be associative):
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T          Element type (e.g., float, double)
	 * \tparam U    Union operator type (e.g., std::plus<>, std::multiplies<>)
	 * \tparam R   Reduction operator type (must be std::plus<> or std::multiplies<>)
	 * \tparam block_size Block size for cache tiling (default: _block_size)
	 * \tparam threads    Number of parallel threads to use (default: _threads)
	 * 
	 * \param A     Input matrix A (M×N) as 2D pointer array
	 * \param B     Input matrix B (M×N) as 2D pointer array
	 * \param seed  Initial seed value for the reduction
	 * \param M     Number of rows in matrices A and B
	 * \param N     Number of columns in matrices A and B
	 * 
	 * \return The scalar result of the fused union-reduce operation
	 * 
	 * \note Only `std::plus<>` and `std::multiplies<>` are supported for reduction operations.
	 * \note Matrices may be asymmetric (M != N).
	 * \note Strides not multiples of block_size are allowed.
	 */
	template <typename T, typename U, typename R, const size_t block_size = _block_size, const size_t threads = _threads>	
	requires 
	(
		(std::same_as<U, std::plus<>> ||
		 std::same_as<U, std::minus<>> ||
		 std::same_as<U, std::multiplies<>> ||
		 std::same_as<U, std::divides<>>) &&
		(std::same_as<R, std::plus<>> ||
		 std::same_as<R, std::multiplies<>>)
	)
	inline __attribute__((always_inline))
	T 
	_fused_reduce(T** A, T** B, T seed, size_t M, size_t N)
	{
		std::array<T, threads> partials;
		partials.fill(seed);

		parallel_for(0, M, block_size,
			[&](size_t i, size_t index, size_t thread_id)
			{
				T r = seed;

				for (size_t j = 0; j < N; j += block_size)
				{
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - j);
					_fused_reduce_block<T, U, R>(A, B, r, i, j, m, n);
				}

				partials[thread_id] = R{}(partials[thread_id], r);
			},threads);

		T result = seed;
		for (const auto& val : partials)
			result = R{}(result, val);

		return result;
	}

	template <typename T, typename U, typename R, const size_t block_size = _block_size, const size_t threads = _threads>
	requires (
		(std::same_as<U, std::plus<>> ||
		 std::same_as<U, std::minus<>> ||
		 std::same_as<U, std::multiplies<>> ||
		 std::same_as<U, std::divides<>>) &&
		(std::same_as<R, std::plus<>> ||
		 std::same_as<R, std::multiplies<>>)
	)
	inline __attribute__((always_inline))
	T
	_fused_reduce(T* A, T* B, T seed, const size_t M, const size_t N)
	{
		std::array<T, threads> partials;
		partials.fill(seed);

		parallel_for(0, M, block_size,
			[&](size_t i, size_t index, size_t thread_id)
			{
				T r = seed;

				for (size_t j = 0; j < N; j += block_size)
				{
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - j);
					_fused_reduce_block<T, U, R>(A, B, r, i, j, m, n);
				}

				partials[thread_id] = R{}(partials[thread_id], r);
			},threads);

		T result = seed;
		for (const auto& val : partials)
			result = R{}(result, val);

		return result;
	}

	/**
	 * \brief Fused reduce of two matrix blocks using SSE intrinsics.
	 * 
	 * Utilizes 128-bit vector instructions for parallel fused union-reduce operations
	 * on small matrix blocks.
	 * 
	 * Supported union operations:
	 * | Operation | Supported |
	 * |-----------|-----------|
	 * | Add       | Yes       |
	 * | Subtract  | Yes       |
	 * | Multiply  | Yes       |
	 * | Divide    | Yes       |
	 * 
	 * Supported reduction operations:
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T Element type (float or double)
	 * \tparam U Union operation (any of std::plus<>, std::minus<>, std::multiplies<>, std::divides<>)
	 * \tparam R Reduction operation (must be std::plus<> or std::multiplies<>)
	 * 
	 * \param A Pointer to the first matrix block
	 * \param B Pointer to the second matrix block
	 * \param r Reference to scalar output for storing the partial result
	 * 
	 * \note Asymmetric matrix shapes (M ≠ N) are allowed.
	 * \note Block stride does not need to be a multiple of the vector width.
	 * \note Subtract and Divide are not implemented for reduction due to lack of associativity.
	 */
	template <typename T, typename U, typename R>
	void 
	_fused_reduce_block_sse(T* A, T* B, T& r);

	/**
	 * \brief Fused reduce of two matrix blocks using AVX intrinsics.
	 * 
	 * Utilizes 256-bit vector instructions for high-throughput fused union-reduce operations.
	 * 
	 * Supported union operations:
	 * | Operation | Supported |
	 * |-----------|-----------|
	 * | Add       | Yes       |
	 * | Subtract  | Yes       |
	 * | Multiply  | Yes       |
	 * | Divide    | Yes       |
	 * 
	 * Supported reduction operations:
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T Element type (float or double)
	 * \tparam U Union operation (any of std::plus<>, std::minus<>, std::multiplies<>, std::divides<>)
	 * \tparam R Reduction operation (must be std::plus<> or std::multiplies<>)
	 * 
	 * \param A Pointer to the first matrix block
	 * \param B Pointer to the second matrix block
	 * \param r Reference to scalar output for storing the partial result
	 * 
	 * \note Asymmetric matrix shapes (M ≠ N) are allowed.
	 * \note Strides which are not multiples of the vector length are supported.
	 * \note Subtract and Divide are not implemented for reduction due to ordering constraints.
	 */
	template <typename T, typename U, typename R>
	void 
	_fused_reduce_block_avx(T* A, T* B, T& r);

	/**
	 * \brief Fused reduce of two matrix blocks using AVX-512 intrinsics.
	 * 
	 * Utilizes 512-bit vector instructions for maximum throughput fused union-reduce operations.
	 * 
	 * Supported union operations:
	 * | Operation | Supported |
	 * |-----------|-----------|
	 * | Add       | Yes       |
	 * | Subtract  | Yes       |
	 * | Multiply  | Yes       |
	 * | Divide    | Yes       |
	 * 
	 * Supported reduction operations:
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T Element type (float or double)
	 * \tparam U Union operation (any of std::plus<>, std::minus<>, std::multiplies<>, std::divides<>)
	 * \tparam R Reduction operation (must be std::plus<> or std::multiplies<>)
	 * 
	 * \param A Pointer to the first matrix block
	 * \param B Pointer to the second matrix block
	 * \param r Reference to scalar output for storing the partial result
	 * 
	 * \note Asymmetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 * \note Subtract and Divide are not implemented for reduction
	 */
	template <typename T, typename U, typename R>
	void 
	_fused_reduce_block_avx512(T* A, T* B, T& r);

	/**
	 * \brief Parallel fused union-reduce using SIMD intrinsics.
	 * 
	 * This is the high-level SIMD entry point for blockwise fused union-reduce operations.
	 * Internally partitions the matrices into cache-line–sized chunks and performs
	 * vectorized union followed by reduction in parallel using `SSE`, `AVX`, or `AVX512` instructions.
	 * 
	 * The operation is performed across all matrix elements in row-major order.
	 * Threads process blocks in a round-robin fashion for balanced load.
	 * 
	 * Supported Union Operations:
	 * | Operation | Supported |
	 * |-----------|-----------|
	 * | Add       | Yes       |
	 * | Subtract  | Yes       |
	 * | Multiply  | Yes       |
	 * | Divide    | Yes       |
	 * 
	 * Supported Reduction Operations:
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T       		Element type (float or double)
	 * \tparam U     	Binary union operator (e.g. std::plus<> or std::multiplies<>)
	 * \tparam R    	Binary reduction operator (e.g. std::plus<> or std::multiplies<>)
	 * \tparam S       		SIMD instruction set to use (SSE, AVX, or AVX512)
	 * \tparam threads		Number of parallel threads
	 * 
	 * \param A     First matrix of size M×N (flattened to 1D array)
	 * \param B     Second matrix of size M×N (flattened to 1D array)
	 * \param seed  Initial value for reduction
	 * \param M     Number of matrix rows
	 * \param N     Number of matrix columns
	 * 
	 * \return The reduced scalar result of applying the fused union-reduce across the matrices
	 * 
	 * \note Subtract and Divide are not implemented for reduction due to lack of associativity/commutativity.
	 * \note Matrix dimensions may be asymmetric (M ≠ N).
	 * \note Block-aligned memory is not required.
	 */
	template<typename T, typename U, typename R, SIMD S, const size_t threads = _threads>	
	requires 
	(
		(std::same_as<U, std::plus<>> ||
		 std::same_as<U, std::minus<>> ||
		 std::same_as<U, std::multiplies<>> ||
		 std::same_as<U, std::divides<>>) &&
		(std::same_as<R, std::plus<>> ||
		 std::same_as<R, std::multiplies<>>)
	)
	inline __attribute__((always_inline))
	T 
	_fused_reduce_simd(T* A, T* B, T seed, const size_t M, const size_t N)
	{
		constexpr size_t cache_line_bytes = 64;
		constexpr size_t cache_line_elems = cache_line_bytes / sizeof(T);

		const size_t total_elements = M * N;
		const size_t full_blocks = total_elements / cache_line_elems;
		const size_t remainder = total_elements % cache_line_elems;

		std::array<T, threads> partials;
		partials.fill(seed_left_fold<T, R>());

		parallel_for( 0, full_blocks, 1,
			[&](size_t block, size_t index, size_t thread_id)
			{
				const size_t offset = block * cache_line_elems;
				T partial = seed_left_fold<T, R>();

				if constexpr (S == SSE)
					_fused_reduce_block_sse<T, U, R>(A + offset, B + offset, partial);
				if constexpr (S == AVX)
					_fused_reduce_block_avx<T, U, R>(A + offset, B + offset, partial);
				if constexpr (S == AVX512)
					_fused_reduce_block_avx512<T, U, R>(A + offset, B + offset, partial);

				partials[thread_id] = R{}(partials[thread_id], partial);
			}, threads);

		T result = seed;
		for (const auto& val : partials)
			result = R{}(result, val);

		// Handle remainder elements
		for (size_t i = total_elements - remainder; i < total_elements; ++i)
		{
			T fused_val = U{}(A[i], B[i]);
			result = R{}(result, fused_val);
		}

		return result;	
	}

	/**
	 * \brief Parallel fused union-reduce using SIMD intrinsics.
	 * 
	 * This is the high-level SIMD entry point for blockwise fused union-reduce operations.
	 * Internally partitions the matrices into cache-line–sized chunks and performs
	 * vectorized union followed by reduction in parallel using `SSE`, `AVX`, or `AVX512` instructions.
	 * 
	 * The operation is performed across all matrix elements in row-major order.
	 * Threads process blocks in a round-robin fashion for balanced load.
	 * 
	 * Supported Union Operations:
	 * | Operation | Supported |
	 * |-----------|-----------|
	 * | Add       | Yes       |
	 * | Subtract  | Yes       |
	 * | Multiply  | Yes       |
	 * | Divide    | Yes       |
	 * 
	 * Supported Reduction Operations:
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T       		Element type (float or double)
	 * \tparam U     	Binary union operator (e.g. std::plus<> or std::multiplies<>)
	 * \tparam R    	Binary reduction operator (e.g. std::plus<> or std::multiplies<>)
	 * \tparam S       		SIMD instruction set to use (SSE, AVX, or AVX512)
	 * \tparam threads		Number of parallel threads
	 * 
	 * \param A     First matrix of size M×N (2D pointer to row-major blocks)
	 * \param B     Second matrix of size M×N (2D pointer to row-major blocks)
	 * \param seed  Initial value for reduction
	 * \param M     Number of matrix rows
	 * \param N     Number of matrix columns
	 * 
	 * \return The reduced scalar result of applying the fused union-reduce across the matrices
	 * 
	 * \note Subtract and Divide are not implemented for reduction due to lack of associativity/commutativity.
	 * \note Matrix dimensions may be asymmetric (M ≠ N).
	 * \note Block-aligned memory is not required.
	 */
	template<typename T, typename U, typename R, SIMD S, const size_t threads = _threads>
	requires (
		(std::same_as<U, std::plus<>> ||
		 std::same_as<U, std::minus<>> ||
		 std::same_as<U, std::multiplies<>> ||
		 std::same_as<U, std::divides<>>) &&
		(std::same_as<R, std::plus<>> ||
		 std::same_as<R, std::multiplies<>>)
	)
	inline __attribute__((always_inline))
	T 
	_fused_reduce_simd(T** A, T** B, T seed, const size_t M, const size_t N)
	{
		T* A0 = &A[0][0];
		T* B0 = &B[0][0];

		return _fused_reduce_simd<T, U, R, S, threads>(A0, B0, seed, M, N);
	}

	template<typename T, typename U, typename R, SIMD S = detect_simd(), 
		const size_t block_size = _block_size, const size_t threads = _threads>
	requires (
		(std::same_as<U, std::plus<>> ||
		 std::same_as<U, std::minus<>> ||
		 std::same_as<U, std::multiplies<>> ||
		 std::same_as<U, std::divides<>>) &&
		(std::same_as<R, std::plus<>> ||
		 std::same_as<R, std::multiplies<>>)
	)
	inline __attribute__((always_inline))
	T 
	fused_reduce(T** A, T** B, T seed, const size_t M, const size_t N)
	{
		right<T>("fused reduce:", 
			std::make_tuple(A, M, N),
			std::make_tuple(B, M, N));

		if constexpr (S == SIMD::NONE)
			return _fused_reduce<T, U, R, block_size, threads>(A, B, seed, M, N);
		else
			return _fused_reduce_simd<T, U, R, S, threads>(A, B, seed, M, N);
	}

	template<typename T, typename U, typename R, SIMD S = detect_simd(), 
		const size_t block_size = _block_size, const size_t threads = _threads>
	requires (
		(std::same_as<U, std::plus<>> ||
		 std::same_as<U, std::minus<>> ||
		 std::same_as<U, std::multiplies<>> ||
		 std::same_as<U, std::divides<>>) &&
		(std::same_as<R, std::plus<>> ||
		 std::same_as<R, std::multiplies<>>)
	)
	inline __attribute__((always_inline))
	T 
	fused_reduce(T* A, T* B, T seed, const size_t M, const size_t N)
	{
		right<T>("fused reduce:", 
			std::make_tuple(A, M, N),
			std::make_tuple(B, M, N));

		if constexpr (S == SIMD::NONE)
			return _fused_reduce<T, U, R, block_size, threads>(A, B, seed, M, N);
		else
			return _fused_reduce_simd<T, U, R, S, threads>(A, B, seed, M, N);
	}

}//namespace damm

#endif //__FUSED_REDUCE_H__