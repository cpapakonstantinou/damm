#ifndef __BROADCAST_H__
#define __BROADCAST_H__

/**
 * \file broadcast.h
 * \brief broadcast utilities definitions
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
#include <damm_kernels.h>
#include <omp.h>

/**
 * \brief Scalar Broadcasting Operations
 */
namespace damm
{
	/** 
	 * \brief kernel for broadcast_block. 
	 * Low level function not intended for the public API.
	 * */
	template <typename T>
	inline __attribute__((always_inline)) 
	void
	_broadcast_block(T** A, const T B, 
		const size_t I, const size_t J,
		const size_t M, const size_t N)
	{
		for(size_t i = 0; i < M; i++) 
			for(size_t j = 0; j < N; j++)
				A[I+i][J+j] = B;
	}

	/**
	 * \brief	Element-wise broadcast of scalar to 2D pointer matrix using blocked traversal.
	 *
	 * Performs a blocked scalar broadcast operation, filling matrix A with scalar value B.
	 * The matrix is processed in tiles of `block_size x block_size` to improve cache locality.
	 * This function operates on matrix views that provide 2D access (e.g., row-major 
	 * pointer-to-pointer layout), typically created from contiguous memory.
	 *
	 * \tparam T	Element type of the matrix (e.g., float, double).
	 *
	 * \param A		Target matrix with dimensions M x N
	 * \param B		Scalar value to broadcast to all elements of A
	 * \param M		Number of rows in matrix A
	 * \param N		Number of columns in matrix A
	 *
	 * \note Matrix A must be a valid 2D view over contiguous row-major memory.
	 * \note The broadcast supports asymmetric matrices (M != N) and dimensions
	 *       that are not multiples of the block size.
	 * \note Uses block tiling to optimize cache performance and reduce TLB pressure.
	 */
	template <typename T, template<typename, typename> class K>
	inline __attribute__((always_inline))
	void
	_broadcast(T** A, const T B, const size_t M, const size_t N)
	{
		using kernel = K<T, NONE>;
		using blocking = typename kernel::blocking;
		
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;
		
		#pragma omp parallel for schedule(static, l2_block)
		for (size_t i = 0; i < M; i += l2_block)
		{
			for (size_t j = 0; j < N; j += l3_block)
			{ 
				size_t m = std::min(l2_block, M - i);
				size_t n = std::min(l3_block, N - j);
				_broadcast_block<T>(A, B, i, j, m, n);
			}
		}
	}

	/**
	 * \brief	Broadcast scalar to block using SIMD intrinsics.
	 *
	 * Low-level SIMD kernel that fills a cache-line sized block with a scalar value.
	 * This is the fundamental building block for SIMD-accelerated broadcast operations.
	 * Each invocation fills exactly 64 bytes (one cache line).
	 *
	 * \tparam T	Scalar type (float, double, or complex variants)
	 * \tparam S	SIMD instruction set to use (SSE, AVX, AVX512)
	 *
	 * \param A		Pointer to memory block to fill
	 * \param B		Scalar value to broadcast
	 * \param row	The starting row offset
	 * \param col	The starting col offset
	 *
	 */
	template<typename T, typename S, 
		template<typename, typename> class K = broadcast_kernel>
	inline __attribute__((always_inline))
	void
	_broadcast_block_simd(T** A, const T B, const size_t row, const size_t col)
	{
		using kernel = K<T, S>;
		using register_t = typename S::template register_t<T>;
		constexpr size_t rows = kernel::row_registers;
		constexpr size_t cols = kernel::col_registers;
		constexpr size_t N = rows;
		
		alignas(S::bytes) register_t registers[rows][cols];
		register_t* reg_ptrs[rows];

		for (size_t i = 0; i < rows; ++i)
			reg_ptrs[i] = registers[i];
		
		load<T, S, K>(A, reg_ptrs, row, col);
		
		auto b = _set1<T, S>(B);
		static_for<rows>([&]<auto i>()
		{
			static_for<cols>([&]<auto j>()
			{
				reg_ptrs[i][j] = b;
			});
		});

		store<T, S, K>(A, reg_ptrs, row, col);
	}

	/**
	 * \brief	Broadcast scalar to matrix using SIMD intrinsics.
	 *
	 * Performs SIMD-accelerated scalar broadcasting to fill 2D matrix A with scalar value B.
	 * SIMD vectorization is selected at compile time via the template parameter `S`. 
	 * This is the main entry point for SIMD-based broadcast operations on 2D pointer arrays.
	 *
	 * \tparam T	Scalar type (float, double, or complex variants)
	 * \tparam S	SIMD instruction set to use (SSE, AVX, AVX512)
	 *
	 * \param A		Pointer to a 2D matrix A
	 * \param B		Scalar value to broadcast to all elements
	 * \param M		Number of rows
	 * \param N		Number of columns
	 *
	 * \note Uses broadcast semantics: scalar B is assigned to each element of A.
	 * \note Internally flattens 2D view to invoke flat SIMD routines for optimal performance.
	 * \note Asymmetric matrices (M != N) are supported.
	 * \note Strides not aligned with SIMD block sizes are safely handled with scalar fallback.
	 */
	template<typename T, typename S, template<typename, typename> class K = broadcast_kernel>
	inline __attribute__((always_inline))
	void
	_broadcast_simd(T** A, const T B, const size_t M, const size_t N)
	{		
		using kernel = K<T, S>;
		using blocking = typename kernel::blocking;
		
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;
		
		constexpr size_t kernel_rows = kernel::kernel_rows();
		constexpr size_t kernel_cols = kernel::kernel_cols();
	
		const size_t simd_rows = M - (M % kernel_rows);
		const size_t simd_cols = N - (N % kernel_cols);
			
		#pragma omp parallel for schedule(static, l2_block)
		for (size_t i_block = 0; i_block < simd_rows; i_block += l2_block)
		{
			size_t i_end = std::min(i_block + l2_block, simd_rows);
			
			for (size_t j_block = 0; j_block < simd_cols; j_block += l3_block)
			{
				size_t j_end = std::min(j_block + l3_block, simd_cols);
				
				for (size_t i = i_block; i < i_end; i += kernel_rows)
				{
					for (size_t j = j_block; j < j_end; j += kernel_cols)
					{
						_broadcast_block_simd<T, S, K>(A, B, i, j);
					}
				}
			}
		}
			
		const size_t rem_rows = M % kernel_rows;
		const size_t rem_cols = N % kernel_cols;

		if (rem_rows != 0) 
		{
			for (size_t j = 0; j < simd_cols; j += kernel_cols)
				_broadcast_block(A, B, simd_rows, j, rem_rows, kernel_cols);
		}
		
		if (rem_cols != 0)
		{
			for (size_t i = 0; i < simd_rows; i += kernel_rows)
				_broadcast_block(A, B, i, simd_cols, kernel_rows, rem_cols);
		}
		
		if (rem_rows != 0 && rem_cols != 0)
		{
			_broadcast_block(A, B, simd_rows, simd_cols, rem_rows, rem_cols);
		}
	}

	/**
	 * \brief Broadcast scalar to matrix.
	 *
	 * This is the main public interface for scalar broadcasting.
	 * Select between SIMD-optimized and standard blocked implementations based on the
	 * specified SIMD instruction set. The function operates on 2D matrix views with row-major layout.
	 *
	 * The broadcast operation assigns scalar value B to all elements of matrix A, where 
	 * A[i][j] = B for all valid indices.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 *
	 * \param A		Target matrix of dimensions MxN in row-major layout.
	 * \param B		Scalar value to broadcast to all matrix elements.
	 * \param M		Number of rows in matrix A.
	 * \param N		Number of columns in matrix A.
	 *
	 * \note Matrix must be allocated as contiguous memory block accessible through the 2D pointer interface.
	 * \note Non-square matrices and asymmetric dimensions (M != N) are fully supported.
	 * 
	 */
	template<typename T, typename S = decltype(detect_simd()), 
		template<typename, typename> class K = broadcast_kernel>
	inline 
	void
	broadcast(T** A, const T B, const size_t M, const size_t N)
	{
		right<T>("broadcast:", std::make_tuple(A, M, N));

		if constexpr (std::is_same_v<S, NONE>) 
			_broadcast<T, K>(A, B, M, N);
		else
			_broadcast_simd<T, S, K>(A, B, M, N);
	}

	/**
	 * \brief Initialize matrix with unit values using optimized algorithms (2D arrays).
	 *
	 * Fills 2D matrix A with unit values. This is a specialized version of broadcast 
	 * optimized for the common unit initialization pattern. Equivalent to broadcast(A, T(1), M, N)
	 * but may have additional optimizations.
	 *
	 * \tparam T	Element type of the matrix (e.g., float, double).
	 * \tparam S	SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 *
	 * \param A		Target matrix of dimensions MxN in row-major layout.
	 * \param M		Number of rows in matrix A.
	 * \param N		Number of columns in matrix A.
	 *
	 * \note All elements will be set to T(1), where T(1) represents the unit value for type T.
	 * \note Uses the same optimized algorithms as broadcast with automatic SIMD selection.
	 * \note Preferred for initialization patterns where all elements should be 1.
	 */
	template<typename T, typename S = decltype(detect_simd()), 
		template<typename, typename> class K = broadcast_kernel>
	inline 
	void
	ones(T** A, const size_t M, const size_t N)
	{
		broadcast<T, S, K>(A, T(1), M, N);
	}
	
	/**
	 * \brief Initialize matrix with zero values using optimized algorithms (2D arrays).
	 *
	 * Fills 2D matrix A with zero values. This is useful when the target platform 
	 * does not zero-initialize memory automatically. This is a specialized version 
	 * of broadcast optimized for the zero initialization pattern.
	 *
	 * \tparam T	Element type of the matrix (e.g., float, double).
	 * \tparam S	SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 *
	 * \param A		Target matrix of dimensions MxN in row-major layout.
	 * \param M		Number of rows in matrix A.
	 * \param N		Number of columns in matrix A.
	 *
	 * \note All elements will be set to T(0), where T(0) represents the zero value for type T.
	 * \note Equivalent to broadcast(A, T(0), M, N) but may have additional optimizations.
	 * \note Can be useful if target platform does not zero-initialize memory.
	 */
	template<typename T, typename S = decltype(detect_simd()), 
		template<typename, typename> class K = broadcast_kernel>
	inline 
	void
	zeros(T** A, const size_t M, const size_t N)
	{
		broadcast<T, S, K>(A, T(0), M, N);
	}

	/** \brief set identity matrix.
	*
	* the naive approach would be to implement a dirac-delta using an (if i == j) branch
	* since the memory layout is known a priori, take advantage of it to skip indices
	* \note requires a zero filled matrix 
	*/
	template<typename T>
	void
	set_identity(T** I, const size_t M, const size_t N)
	{
		const size_t diag_len = std::min(M, N);
		for (size_t i = 0; i < diag_len; ++i)
			I[i][i] = T(1);
	}

	/**
	 * \brief Create identity matrix .
	 *
	 * Initializes 2D matrix A as an identity matrix. The matrix is first filled with zeros,
	 * then diagonal elements are set to unit values.
	 *
	 * \tparam T	Element type of the matrix (e.g., float, double).
	 * \tparam S	SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 *
	 * \param A		Target matrix of dimensions MxN in row-major layout.
	 * \param M		Number of rows in matrix A.
	 * \param N		Number of columns in matrix A.
	 *
	 * \note For square matrices (M=N): creates standard identity matrix with 1s on diagonal.
	 * \note For rectangular matrices: creates identity pattern up to min(M,N) diagonal elements.
	 * \note All off-diagonal elements are set to T(0), diagonal elements to T(1).
	 */
	template<typename T, typename S = decltype(detect_simd()), 
		template<typename, typename> class K = broadcast_kernel>
	inline 
	void
	identity(T** A, const size_t M, const size_t N)
	{
		zeros<T, S, K>(A, M, N);
		set_identity(A, M, N);
	}

}//namespace damm
#endif //__BROADCAST_H__