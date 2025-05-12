#ifndef __REDUCE_H__
#define __REDUCE_H__
/**
 * \file reduce.h
 * \brief definitions for reduce utilities 
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
 * Reduction operation.
 *
 * \note
 * A reduction combines elements over an index domain using a specified 
 * binary arithmetic operator (e.g., addition, subtraction, multiplication, 
 * or division) to produce a single aggregated result.
 *
 * \note
 * Reduction operations typically assume associative operators for safe 
 * and efficient vectorization and parallelization, but may be adapted 
 * to other operators depending on numerical requirements.
 *
 * \note
 * This operation is fundamental in parallel computing and SIMD 
 * optimizations, where partial results are combined efficiently 
 * using hardware-supported instructions.
 */
namespace damm
{
	/** 
	 * \brief kernal for reduce_block. 
	 * Low level function not intended for the public API.
	 * */
	template <typename T, typename O>
	inline __attribute__((always_inline))
	void 
	_reduce_block(T** A, T& r,
					   const size_t I, const size_t J,
					   const size_t M, const size_t N)
	{
		for (size_t i = 0; i < M; ++i)
			for (size_t j = 0; j < N; ++j)
				r = O{}(r, A[I + i][J + j]);
	}
	
	/** 
	 * \brief kernal for reduce_block. 
	 * Low level function not intended for the public API.
	 * */
	template <typename T, typename O>
	inline __attribute__((always_inline)) 
	void
	_reduce_block(T* A, T& r, 
					const size_t I, const size_t J,
					const size_t M, const size_t N)
	{
		for(size_t i = 0; i < M; i++) 
			for(size_t j = 0; j < N; j++)
				r = O()(r, A[(I+i)*M + (J+j)]);
	}

	/**
	 * \brief Reduce a matrix using blocked traversal and a binary reduction operator.
	 * 
	 * The matrix is processed in tiles of `block_size × block_size` to optimize cache usage.
	 * The reduction operation must be associative and commutative for correct parallel execution.
	 * 
	 * Supported operations and their properties:
	 * 
	 * | Operation | Associative | Commutative | Parallel   |
	 * |-----------|-------------|-------------|------------|
	 * | Add       | Yes         | Yes         | Yes        |
	 * | Multiply  | Yes         | Yes         | Yes        |
	 * | Subtract  | No          | No          | No         |
	 * | Divide    | No          | No          | No         |
	 * 
	 * \tparam T          Element type (e.g., float, double)
	 * \tparam O          Reduction operator type (must be std::plus<> or std::multiplies<>)
	 * \tparam block_size Block size for cache tiling (default: _block_size)
	 * \tparam threads    Number of parallel threads to use (default: _threads)
	 * 
	 * \param A     Input matrix (M×N) as 2D pointer array
	 * \param seed  Initial seed value for the reduction
	 * \param M     Number of rows in matrix A
	 * \param N     Number of columns in matrix A
	 * 
	 * \return The scalar result of the reduction operation over all elements in A
	 * 
	 * \note Only `std::plus<>` and `std::multiplies<>` are currently implemented and supported.
	 * \note Matrices may be asymmetric (M != N).
	 * \note Strides not multiples of block_size are allowed.
	 */
	template <typename T, typename O, const size_t block_size = _block_size, const size_t threads = _threads>	
	requires 
	(
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::multiplies<>>
		// ||
		// std::same_as<O, std::minus<>> ||
		// std::same_as<O, std::divides<>>
	)
	inline __attribute__((always_inline))
	T 
	_reduce(T** A, T seed, size_t M, size_t N)
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
					_reduce_block<T, O>(A, r, i, j, m, n);
				}

				partials[thread_id] = O{}(partials[thread_id], r);
			},threads);

		T result = seed;
		for (const auto& val : partials)
			result = O{}(result, val);

		return result;
	}

	template <typename T, typename O, const size_t block_size = _block_size, const size_t threads = _threads>
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::multiplies<>>
		// ||
		// std::same_as<O, std::minus<>> ||
		// std::same_as<O, std::divides<>>
	)
	inline __attribute__((always_inline))
	T
	_reduce(T* A, T seed, const size_t M, const size_t N)
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
					_reduce_block<T, O>(A, r, i, j, m, n);
				}

				partials[thread_id] = O{}(partials[thread_id], r);
			}, threads);

		T result = seed;
		for (const auto& val : partials)
			result = O{}(result, val);

		return result;
	}

	/**
	 * \brief Reduce a matrix block using SSE intrinsics.
	 * 
	 * Utilizes 128-bit vector instructions for parallel reduction of a small matrix block.
	 * 
	 * Supported operations and their properties:
	 * 
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T Element type (float or double)
	 * \tparam O Reduction operation (must be std::plus<> or std::multiplies<>)
	 * 
	 * \param A Pointer to the matrix block to reduce
	 * \param r Reference to scalar output for storing the result
	 * 
	 * \note Asymmetric matrix shapes (M ≠ N) are allowed.
	 * \note Block stride does not need to be a multiple of the vector width.
	 * \note Subtract and Divide are not implemented due to lack of associativity.
	 */
	template <typename T, typename O>
	void 
	_reduce_block_sse(T* A, T& r);

	/**
	 * \brief Reduce a matrix block using AVX-512 intrinsics.
	 * 
	 * Utilizes 512-bit vector instructions for high-throughput reduction.
	 * 
	 * Supported operations and their properties:
	 * 
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T Element type (float or double)
	 * \tparam O Reduction operation (must be std::plus<> or std::multiplies<>)
	 * 
	 * \param A Pointer to the matrix block to reduce
	 * \param r Reference to scalar output for storing the result
	 * 
	 * \note Asymmetric matrix shapes (M ≠ N) are allowed.
	 * \note Strides which are not multiples of the vector length are supported.
	 * \note Subtract and Divide are not implemented due to ordering constraints.
	 */
	template <typename T, typename O>
	void 
	_reduce_block_avx(T* A, T& r);

	/**
	 * \brief Reduce of a matrix using SSE intrinsics.
	 * 
	 *	Supported operations and their properties:
	 * 
	 *	| Operation | Associative | Commutative | Parallel      |
	 *	|-----------|-------------|-------------|---------------|
	 *	| Add       | Yes         | Yes         | Yes           |
	 *	| Multiply  | Yes         | Yes         | Yes           |
	 *	| Subtract  | No          | No          | No            |
	 *	| Divide    | No          | No          | No            |
	 *  
	 * \param A the matrix to reduce 
	 * \param B storage for the reduction
	 * \param M the stride of the matrix
	 * 
	 * \note Assymetric matrices (N != M) are allowed.
	 * \note Strides which are not multiples of the block size are allowed.
	 * \note Subtract and Divide are not implemented
	 */
	template <typename T, typename O>
	void 
	_reduce_block_avx512(T* A, T& r);

	 /* \brief Parallel matrix reduction using SIMD intrinsics.
	 * 
	 * This is the high-level SIMD entry point for blockwise reduction.
	 * Internally partitions the matrix into cache-line–sized chunks and performs
	 * vectorized reduction in parallel using `SSE`, `AVX`, or `AVX512` instructions.
	 * 
	 * The reduction is performed across all matrix elements in row-major order.
	 * Threads process blocks in a round-robin fashion for balanced load.
	 * 
	 * Supported Operations
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T       		Element type (float or double)
	 * \tparam O       		Binary reduction operator (e.g. std::plus<> or std::multiplies<>)
	 * \tparam S       		SIMD instruction set to use (SSE, AVX, or AVX512)
	 * \tparam block_size	line size of L1 cache
	 * \tparam threads		Number of parallel threads
	 * 
	 * \param A     Matrix of size M×N (2D pointer to row-major blocks)
	 * \param seed  Initial value for reduction
	 * \param M     Number of matrix rows
	 * \param N     Number of matrix columns
	 * 
	 * \return The reduced scalar result of applying the binary operator across the matrix
	 * 
	 * \note Subtract and Divide are not implemented due to lack of associativity/commutativity.
	 * \note Matrix dimensions may be asymmetric (M ≠ N).
	 * \note Block-aligned memory is not required.
	 */
	template<typename T, typename O, SIMD S, const size_t threads = _threads>	
	requires 
	(
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::multiplies<>>
		// ||
		// std::same_as<O, std::minus<>> ||
		// std::same_as<O, std::divides<>>
	)
	inline __attribute__((always_inline))
	T 
	_reduce_simd(T* A, T seed, const size_t M, const size_t N)
	{
		constexpr size_t cache_line_bytes = 64;
		constexpr size_t cache_line_elems = cache_line_bytes / sizeof(T);

		const size_t total_elements = M * N;
		const size_t full_blocks = total_elements / cache_line_elems;
		const size_t remainder = total_elements % cache_line_elems;

		alignas(S) std::array<T, threads> partials;
		partials.fill(seed_left_fold<T, O>());

		parallel_for( 0, full_blocks, 1,
			[&](size_t block, size_t index, size_t thread_id)
			{
				const size_t offset = block * cache_line_elems;
				T partial = seed_left_fold<T, O>();

				if constexpr (S == SSE)
					_reduce_block_sse<T, O>(A + offset, partial);
				if constexpr (S == AVX)
					_reduce_block_avx<T, O>(A + offset, partial);
				if constexpr (S == AVX512)
					_reduce_block_avx512<T, O>(A + offset, partial);

				partials[thread_id] = O{}(partials[thread_id], partial);
			}, threads);

		T result = seed;
		for (const auto& val : partials)
			result = O{}(result, val);

		for (size_t i = total_elements - remainder; i < total_elements; ++i)
			result = O{}(result, A[i]);

		return result;	
	}

	/**
	 * \brief Parallel matrix reduction using SIMD intrinsics.
	 * 
	 * This is the high-level SIMD entry point for blockwise reduction.
	 * Internally partitions the matrix into cache-line–sized chunks and performs
	 * vectorized reduction in parallel using `SSE`, `AVX`, or `AVX512` instructions.
	 * 
	 * The reduction is performed across all matrix elements in row-major order.
	 * Threads process blocks in a round-robin fashion for balanced load.
	 * 
	 * Supported Operations
	 * | Operation | Associative | Commutative | Parallel |
	 * |-----------|-------------|-------------|----------|
	 * | Add       | Yes         | Yes         | Yes      |
	 * | Multiply  | Yes         | Yes         | Yes      |
	 * | Subtract  | No          | No          | No       |
	 * | Divide    | No          | No          | No       |
	 * 
	 * \tparam T       		Element type (float or double)
	 * \tparam O       		Binary reduction operator (e.g. std::plus<> or std::multiplies<>)
	 * \tparam S       		SIMD instruction set to use (SSE, AVX, or AVX512)
	 * \tparam block_size	line size of L1 cache
	 * \tparam threads		Number of parallel threads
	 * 
	 * \param A     Matrix of size M×N (2D pointer to row-major blocks)
	 * \param seed  Initial value for reduction
	 * \param M     Number of matrix rows
	 * \param N     Number of matrix columns
	 * 
	 * \return The reduced scalar result of applying the binary operator across the matrix
	 * 
	 * \note Subtract and Divide are not implemented due to lack of associativity/commutativity.
	 * \note Matrix dimensions may be asymmetric (M ≠ N).
	 * \note Block-aligned memory is not required.
	 */
	template<typename T, typename O, SIMD S, const size_t threads = _threads>
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::multiplies<>>
		// ||
		// std::same_as<O, std::minus<>> ||
		// std::same_as<O, std::divides<>>
	)
	inline __attribute__((always_inline))
	T 
	_reduce_simd(T** A, T seed, const size_t M, const size_t N)
	{
		T* A0 = &A[0][0];

		return _reduce_simd<T, O, S, threads>(A0, seed, M, N);
	}

	template<typename T, typename O, SIMD S = detect_simd(), 
		const size_t block_size = _block_size, const size_t threads = _threads>
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::multiplies<>>
		// ||
		// std::same_as<O, std::minus<>> ||
		// std::same_as<O, std::divides<>>
	)
	inline __attribute__((always_inline))
	T 
	reduce(T** A, T seed, const size_t M, const size_t N)
	{

		right<T>("reduce:", std::make_tuple(A, M, N));

		if constexpr (S == SIMD::NONE)
			return _reduce<T, O, block_size, threads>(A, seed, M, N);
		else
			return _reduce_simd<T, O, S, threads>(A, seed, M, N);
	}

	template<typename T, typename O, SIMD S = detect_simd(), 
		const size_t block_size = _block_size, const size_t threads = _threads>
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::multiplies<>>
		// ||
		// std::same_as<O, std::minus<>> ||
		// std::same_as<O, std::divides<>>
	)
	inline __attribute__((always_inline))
	T 
	reduce(T* A, T seed, const size_t M, const size_t N)
	{

		right<T>("reduce:", std::make_tuple(A, M, N));

		if constexpr (S == SIMD::NONE)
			return _reduce<T, O, block_size, threads>(A, seed, M, N);
		else
			return _reduce_simd<T, O, S, threads>(A, seed, M, N);
	}


}//namespace damm

#endif //__REDUCE_H__