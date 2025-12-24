#ifndef __FUSED_REDUCE_H__
#define __FUSED_REDUCE_H__
/**
 * \file fused_reduce.h
 * \brief definitions for fused reduce utilities (OpenMP version)
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
#include <simd.h>
#include <functional>
#include <damm_kernels.h>
#include <omp.h>

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
	 * \brief Fused reduce of two matrices using blocked traversal with OpenMP.
	 * 
	 * Combines element-wise union operations with reduction in a single pass
	 * to optimize cache usage and eliminate intermediate storage. The matrices
	 * are processed using cache-aware blocking determined by the kernel policy.
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
	 * \tparam T	Element type (e.g., float, double)
	 * \tparam U	Union operator type (e.g., std::plus<>, std::multiplies<>)
	 * \tparam R	Reduction operator type (must be std::plus<> or std::multiplies<>)
	 * \tparam K	Kernel policy defining cache blocking sizes
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
	 * \note Uses OpenMP for parallelization.
	 */
	template <typename T, typename U, typename R, template<typename, typename> class K>	
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
		using kernel = K<T, NONE>;
		using blocking = typename kernel::blocking;
		
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;
		
		T result = seed;
		
		#pragma omp parallel
		{
			T local_result = seed_left_fold<T, R>();
			
			#pragma omp for schedule(static) nowait
			for (size_t i = 0; i < M; i += l2_block)
			{
				T r = seed;
				
				for (size_t j = 0; j < N; j += l3_block)
				{
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l3_block, N - j);
					_fused_reduce_block<T, U, R>(A, B, r, i, j, m, n);
				}
				
				local_result = R{}(local_result, r);
			}
			
			#pragma omp critical
			{
				result = R{}(result, local_result);
			}
		}
		
		return result;
	}

	/**
	 * \brief Reduce a block of matrix data using SIMD registers with fused union operation.
	 * 
	 * Loads tiles of matrices A and B into SIMD registers, performs element-wise union
	 * operation, then reduces the result. Optimized patterns detect common operations
	 * like dot products (multiply+add) to use FMA instructions.
	 * 
	 * \tparam T   Scalar type (float, double, or complex variants)
	 * \tparam U   Union operator (std::plus<>, std::minus<>, std::multiplies<>, std::divides<>)
	 * \tparam R   Reduction operator (std::plus<> or std::multiplies<>)
	 * \tparam S   SIMD instruction set (SSE, AVX, AVX512)
	 * \tparam K   Kernel policy
	 * 
	 * \param A       Input matrix A
	 * \param B       Input matrix B
	 * \param partial Accumulator for block result
	 * \param row     Starting row offset
	 * \param col     Starting column offset
	 */
	template <typename T, typename U, typename R, typename S, template<typename, typename> class K>
	inline __attribute__((always_inline))
	void 
	_fused_reduce_simd_block(T** A, T** B, T& partial, const size_t row, const size_t col)
	{
		using kernel = K<T, S>;
		using register_t = typename S::template register_t<T>;
		
		constexpr size_t M = kernel::row_registers;
		constexpr size_t N = kernel::col_registers;

		// Load data into registers
		alignas(S::bytes) register_t a[M][N];
		alignas(S::bytes) register_t b[M][N];
		register_t* a_ptrs[M];
		register_t* b_ptrs[M];
		
		for (size_t i = 0; i < M; ++i)
		{
			a_ptrs[i] = a[i];
			b_ptrs[i] = b[i];
		}
		
		load<T, S, K>(A, a_ptrs, row, col);
		load<T, S, K>(B, b_ptrs, row, col);

		constexpr bool use_fmadd = std::same_as<U, std::multiplies<>> && std::same_as<R, std::plus<>>;

		// Reduce each row across columns
		alignas(S::bytes) register_t row_results[M];
		
		if constexpr (use_fmadd)
		{
			static_for<M>([&]<auto i>() 
			{
				register_t acc = _mul<T, S>(a[i][0], b[i][0]);
				
				static_for<N - 1>([&]<auto j>() 
				{
					acc = _fmadd<T, S>(a[i][j + 1], b[i][j + 1], acc);
				});
				row_results[i] = acc;
			});
		}
		else
		{
			static_for<M>([&]<auto i>() 
			{
				// Apply union operation
				alignas(S::bytes) register_t union_results[N];
				static_for<N>([&]<auto j>() 
				{
					if constexpr (std::same_as<U, std::plus<>>)
						union_results[j] = _add<T, S>(a[i][j], b[i][j]);
					else if constexpr (std::same_as<U, std::minus<>>)
						union_results[j] = _sub<T, S>(a[i][j], b[i][j]);
					else if constexpr (std::same_as<U, std::multiplies<>>)
						union_results[j] = _mul<T, S>(a[i][j], b[i][j]);
					else if constexpr (std::same_as<U, std::divides<>>)
						union_results[j] = _div<T, S>(a[i][j], b[i][j]);
				});
				
				// Reduce across columns
				register_t acc = union_results[0];
				static_for<N - 1>([&]<auto j>() 
				{
					if constexpr (std::same_as<R, std::plus<>>)
						acc = _add<T, S>(acc, union_results[j + 1]);
					else if constexpr (std::same_as<R, std::multiplies<>>)
						acc = _mul<T, S>(acc, union_results[j + 1]);
				});
				row_results[i] = acc;
			});
		}
		
		register_t final_reg = row_results[0];
		static_for<M - 1>([&]<auto i>() 
		{
			if constexpr (std::same_as<R, std::plus<>>)
				final_reg = _add<T, S>(final_reg, row_results[i + 1]);
			else if constexpr (std::same_as<R, std::multiplies<>>)
				final_reg = _mul<T, S>(final_reg, row_results[i + 1]);
		});
		
		// Horizontal reduction to scalar
		T block_result;
		if constexpr (std::same_as<R, std::plus<>>)
			block_result = _reduce_add<T, S>(final_reg);
		else if constexpr (std::same_as<R, std::multiplies<>>)
			block_result = _reduce_mul<T, S>(final_reg);
		
		// Accumulate into partial
		partial = R{}(partial, block_result);
	}

	/**
	 * \brief Parallel fused union-reduce using SIMD intrinsics with OpenMP.
	 * 
	 * This is the high-level SIMD entry point for blockwise fused union-reduce operations.
	 * Internally partitions the matrices into cache-line–sized chunks and performs
	 * vectorized union followed by reduction in parallel using SSE, AVX, or AVX512 instructions.
	 * 
	 * The operation is performed across all matrix elements in row-major order.
	 * OpenMP threads process blocks with static scheduling for balanced load.
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
	 * \tparam T  Element type (float or double)
	 * \tparam U  Binary union operator (e.g. std::plus<> or std::multiplies<>)
	 * \tparam R  Binary reduction operator (e.g. std::plus<> or std::multiplies<>)
	 * \tparam S  SIMD instruction set to use (SSE, AVX, or AVX512)
	 * \tparam K  Kernel policy
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
	 * \note Uses OpenMP for parallelization.
	 */
	template<typename T, typename U, typename R, 
		typename S, template<typename, typename> class K>
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
		using kernel = K<T, S>;
		using blocking = typename kernel::blocking;
		constexpr size_t tile_rows = kernel::kernel_rows();
		constexpr size_t tile_cols = kernel::kernel_cols();
		
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;

		const size_t simd_rows = M - (M % tile_rows);
		const size_t simd_cols = N - (N % tile_cols);

		T result = seed;
		
		#pragma omp parallel
		{
			T local_result = seed_left_fold<T, R>();
			
			#pragma omp for schedule(static) nowait
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
								T partial = seed_left_fold<T, R>();
								_fused_reduce_simd_block<T, U, R, S, K>(A, B, partial, i, j);
								local_result = R{}(local_result, partial);
							}
						}
					}
				}
			
			#pragma omp critical
			{
				result = R{}(result, local_result);
			}
		}

		const size_t rem_rows = M % tile_rows;
		const size_t rem_cols = N % tile_cols;

		// Bottom strip: remainder rows × SIMD-aligned columns
		if (rem_rows != 0)
		{
			for (size_t i = simd_rows; i < M; ++i)
				for (size_t j = 0; j < simd_cols; j += tile_cols)
					_fused_reduce_block<T, U, R>(A, B, result, i, j, 1, tile_cols);
		}

		// Right strip: SIMD-aligned rows × remainder columns
		if (rem_cols != 0)
		{
			for (size_t i = 0; i < simd_rows; ++i)
				for (size_t j = simd_cols; j < N; ++j)
				{
					T fused_val = U{}(A[i][j], B[i][j]);
					result = R{}(result, fused_val);
				}
		}

		// Corner: remainder rows × remainder columns
		if (rem_rows != 0 && rem_cols != 0)
		{
			for (size_t i = simd_rows; i < M; ++i)
				for (size_t j = simd_cols; j < N; ++j)
				{
					T fused_val = U{}(A[i][j], B[i][j]);
					result = R{}(result, fused_val);
				}
		}

		return result;	
	}

	/**
	 * \brief Perform fused union-reduce operation with automatic SIMD selection and OpenMP parallelization.
	 * 
	 * Main public interface for fused union-reduce operations. Automatically selects
	 * between SIMD-optimized and standard blocked implementations based on the
	 * specified SIMD instruction set.
	 * 
	 * \tparam T  Element type (e.g., float, double)
	 * \tparam U  Union operator (std::plus<>, std::minus<>, std::multiplies<>, std::divides<>)
	 * \tparam R  Reduction operator (std::plus<> or std::multiplies<>)
	 * \tparam S  SIMD instruction set (SSE, AVX, AVX512, or NONE)
	 * \tparam K  Kernel policy defining blocking strategy
	 * 
	 * \param A     Input matrix A (M×N)
	 * \param B     Input matrix B (M×N)
	 * \param seed  Initial value for reduction
	 * \param M     Number of rows
	 * \param N     Number of columns
	 * 
	 * \return Scalar result of fused union-reduce operation
	 * 
	 * \note Uses OpenMP for parallelization.
	 * \note Compile with -fopenmp flag.
	 */
	template<typename T, typename U, typename R, typename S = decltype(detect_simd()), 
		template<typename, typename> class K = fused_reduce_kernel>
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

		if constexpr (std::is_same_v<S, NONE>)
			return _fused_reduce<T, U, R, K>(A, B, seed, M, N);
		else
			return _fused_reduce_simd<T, U, R, S, K>(A, B, seed, M, N);
	}

}//namespace damm

#endif //__FUSED_REDUCE_H__