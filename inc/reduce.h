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
#include <simd.h>
#include <damm_kernels.h>
#include <omp.h>

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
	 * \brief Reduce a matrix using blocked traversal and a binary reduction operator.
	 * 
	 * The matrix is processed using cache-aware blocking determined by the kernel policy.
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
	 * \tparam T	Element type (e.g., float, double)
	 * \tparam O	Reduction operator type (must be std::plus<> or std::multiplies<>)
	 * \tparam K	Kernel policy defining cache blocking sizes
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
	template <typename T, typename O, template<typename, typename> class K>	
	requires 
	(
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::multiplies<>>
	)
	inline __attribute__((always_inline))
	T 
	_reduce(T** A, T seed, size_t M, size_t N)
	{
		using kernel = K<T, NONE>;
		using blocking = typename kernel::blocking;
		
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;
		
		T result = seed;
		
		if constexpr (std::same_as<O, std::plus<>>)
		{
			#pragma omp parallel for schedule(static, l2_block) reduction(+:result)
			for (size_t i = 0; i < M; i += l2_block)
			{
				for (size_t j = 0; j < N; j += l3_block)
				{
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l3_block, N - j);
					_reduce_block<T, O>(A, result, i, j, m, n);
				}
			}
		}
		else if constexpr (std::same_as<O, std::multiplies<>>)
		{
			#pragma omp parallel for schedule(static, l2_block) reduction(*:result)
			for (size_t i = 0; i < M; i += l2_block)
			{
				for (size_t j = 0; j < N; j += l3_block)
				{
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l3_block, N - j);
					_reduce_block<T, O>(A, result, i, j, m, n);
				}
			}
		}

		return result;
	}

	/**
	 * \brief Reduce a block of matrix data using SIMD registers.
	 * 
	 * Loads a tile of the matrix into SIMD registers, performs vertical reduction
	 * across columns, then horizontal reduction across lanes to produce partial
	 * scalar results.
	 * 
	 * \tparam T   Scalar type (float, double, or complex variants)
	 * \tparam S   SIMD instruction set (SSE, AVX, AVX512)
	 * \tparam O  Operation type (std::plus<> or std::multiplies<>)
	 * 
	 * \param A      Input matrix
	 * \param row    Starting row offset
	 * \param col    Starting column offset
	 * \return Scalar result of reducing this block
	 */
	template <typename T, typename O, typename S, template<typename, typename> class K>
	inline __attribute__((always_inline))
	void _reduce_simd_block(T** A, T& partial, const size_t row, const size_t col)
	{
		using kernel = K<T, S>;
		using register_t = typename S::template register_t<T>;
		
		constexpr size_t M = kernel::row_registers;
		constexpr size_t N = kernel::col_registers;

		// Load data into registers
		alignas(S::bytes) register_t a[M][N];
		register_t* a_ptrs[M];
		
		for (size_t i = 0; i < M; ++i)
			a_ptrs[i] = a[i];
		
		load<T, S, K>(A, a_ptrs, row, col);

		// Reduce each row across columns (simple pairwise reduction)
		alignas(S::bytes) register_t row_results[M];
		static_for<M>([&]<auto i>() 
		{
			register_t acc = a[i][0];
			static_for<N - 1>([&]<auto j>() 
			{
				if constexpr (std::same_as<O, std::plus<>>)
					acc = _add<T, S>(acc, a[i][j + 1]);
				else if constexpr (std::same_as<O, std::multiplies<>>)
					acc = _mul<T, S>(acc, a[i][j + 1]);
			});
			row_results[i] = acc;
		});
		
		// Reduce across rows (simple pairwise reduction)
		register_t final_reg = row_results[0];
		static_for<M - 1>([&]<auto i>() 
		{
			if constexpr (std::same_as<O, std::plus<>>)
				final_reg = _add<T, S>(final_reg, row_results[i + 1]);
			else if constexpr (std::same_as<O, std::multiplies<>>)
				final_reg = _mul<T, S>(final_reg, row_results[i + 1]);
		});
		
		// Reduce horizontally within the register to get scalar
		T block_result;
		if constexpr (std::same_as<O, std::plus<>>)
			block_result = _reduce_add<T, S>(final_reg);
		else if constexpr (std::same_as<O, std::multiplies<>>)
			block_result = _reduce_mul<T, S>(final_reg);
		
		// Accumulate into partial
		partial = O{}(partial, block_result);
	}

	/**
	 * \brief Matrix reduction using SIMD intrinsics.
	 * 
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
	 * \tparam K			Kernel policy defining cache blocking sizes
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
	template<typename T, typename O, typename S, template<typename, typename> class K>	
	requires 
	(
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::multiplies<>>
	)
	inline __attribute__((always_inline))
	T 
	_reduce_simd(T** A, T seed, const size_t M, const size_t N)
	{
		using kernel = K<T, S>;
		using blocking = typename kernel::blocking;
		
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;
		
		constexpr size_t tile_rows = kernel::kernel_rows();
		constexpr size_t tile_cols = kernel::kernel_cols();

		const size_t simd_rows = M - (M % tile_rows);
		const size_t simd_cols = N - (N % tile_cols);

		T result = seed;

		if constexpr (std::same_as<O, std::plus<>>)
		{
			#pragma omp parallel for schedule(static, l2_block) reduction(+:result)
			for (size_t i_block = 0; i_block < simd_rows; i_block += l2_block)
			{
				size_t i_end = std::min(i_block + l2_block, simd_rows);
				
				// L3 blocking over columns
				for (size_t j_block = 0; j_block < simd_cols; j_block += l3_block)
				{
					size_t j_end = std::min(j_block + l3_block, simd_cols);
					
					// Micro-kernel tiles within cache blocks
					for (size_t i = i_block; i < i_end; i += tile_rows)
					{
						for (size_t j = j_block; j < j_end; j += tile_cols)
						{
							_reduce_simd_block<T, O, S, K>(A, result, i, j);
						}
					}
				}
			}
		}
		else if constexpr (std::same_as<O, std::multiplies<>>)
		{
			#pragma omp parallel for schedule(static, l2_block) reduction(*:result)
			for (size_t i_block = 0; i_block < simd_rows; i_block += l2_block)
			{
				size_t i_end = std::min(i_block + l2_block, simd_rows);
				
				// L3 blocking over columns
				for (size_t j_block = 0; j_block < simd_cols; j_block += l3_block)
				{
					size_t j_end = std::min(j_block + l3_block, simd_cols);
					
					// Micro-kernel tiles within cache blocks
					for (size_t i = i_block; i < i_end; i += tile_rows)
					{
						for (size_t j = j_block; j < j_end; j += tile_cols)
						{
							_reduce_simd_block<T, O, S, K>(A, result, i, j);
						}
					}
				}
			}
		}

		const size_t rem_rows = M % tile_rows;
		const size_t rem_cols = N % tile_cols;

		if (rem_rows != 0)
		{
			for (size_t i = simd_rows; i < M; ++i)
				for (size_t j = 0; j < simd_cols; j += tile_cols)
					_reduce_block<T, O>(A, result, i, j, 1, tile_cols);
		}

		if (rem_cols != 0)
		{
			for (size_t i = 0; i < simd_rows; ++i)
				for (size_t j = simd_cols; j < N; ++j)
					result = O{}(result, A[i][j]);
		}

		if (rem_rows != 0 && rem_cols != 0)
		{
			for (size_t i = simd_rows; i < M; ++i)
				for (size_t j = simd_cols; j < N; ++j)
					result = O{}(result, A[i][j]);
		}

		return result;
	}

	/**
	 * \brief Matrix reduction using SIMD intrinsics.
	 * 
	 * This is the high-level SIMD entry point for blockwise reduction.
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
	 * \tparam K			Kernel policy defining cache blocking sizes
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

	template<typename T, typename O, typename S = decltype(detect_simd()),
		template<typename, typename> class K = reduce_kernel>
	requires (
		std::same_as<O, std::plus<>> ||
		std::same_as<O, std::multiplies<>>
	)
	inline __attribute__((always_inline))
	T 
	reduce(T** A, T seed, const size_t M, const size_t N)
	{

		right<T>("reduce:", std::make_tuple(A, M, N));

		if constexpr (std::is_same_v<S, NONE>)
			return _reduce<T, O, K>(A, seed, M, N);
		else
			return _reduce_simd<T, O, S, K>(A, seed, M, N);
	}


}//namespace damm

#endif //__REDUCE_H__