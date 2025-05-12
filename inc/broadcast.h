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

/**
 * \brief Scalar Broadcasting Operations
 *
 * \note Using broadcast<NONE> is recommended for most cases, as it performs comparably 
 * or better than std::fill in benchmarks.
 *
 * \note If the matrix allocation is already zero-initialized, common.h:set_identity 
 * is faster than identity<T, NONE>.
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
	 * \brief kernel for broadcast_block_flat. 
	 * Low level function not intended for the public API.
	 * */
	template <typename T>
	inline __attribute__((always_inline)) 
	void
	_broadcast_block(T* A, const T B,
		const size_t I, const size_t J,
		const size_t M, const size_t N)
	{
		for(size_t i = 0; i < M; i++)
			for(size_t j = 0; j < N; j++)
				A[(I+i)*N + (J+j)] = B;
	}

	/**
	 * \brief	Element-wise broadcast of scalar to 2D pointer matrix using blocked traversal.
	 *
	 * Performs a blocked scalar broadcast operation, filling matrix A with scalar value B.
	 * The matrix is processed in tiles of `block_size × block_size` to improve cache locality.
	 * This function operates on matrix views that provide 2D access (e.g., row-major 
	 * pointer-to-pointer layout), typically created from contiguous memory.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam block_size	Tile size for cache-friendly blocking (default: _block_size)
	 * \tparam threads		Number of threads to use for parallel execution (default: _threads)
	 *
	 * \param A		Target matrix view with dimensions M × N
	 * \param B		Scalar value to broadcast to all elements of A
	 * \param M		Number of rows in matrix A
	 * \param N		Number of columns in matrix A
	 *
	 * \note Matrix A must be a valid 2D view over contiguous row-major memory.
	 * \note The broadcast supports asymmetric matrices (M ≠ N) and dimensions
	 *       that are not multiples of the block size.
	 * \note Uses block tiling to optimize cache performance and reduce TLB pressure.
	 */
	template <typename T, const size_t block_size = _block_size, const size_t threads = _threads>	
	inline __attribute__((always_inline))
	void
	_broadcast(T** A, const T B, const size_t M, const size_t N)
	{
		parallel_for(0, M, block_size,
		[&](size_t i)
		{
			for (size_t j = 0; j < N; j += block_size)
			{ 
				size_t m = std::min(block_size, M - i);
				size_t n = std::min(block_size, N - j);
				_broadcast_block<T>(A, B, i, j, m, n);
			}
		}, threads);
	}

	/**
	 * \brief	Element-wise broadcast of scalar to flat matrix using blocked traversal.
	 *
	 * Performs a blocked scalar broadcast operation, filling flat array A with scalar value B.
	 * The matrix is processed in tiles of `block_size × block_size` to optimize cache usage.
	 * Operates on flat row-major 1D arrays. Suitable for use in performance-sensitive contexts 
	 * where control over memory layout is critical.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam block_size	Tile size for cache-friendly blocking (default: _block_size)
	 * \tparam threads		Number of threads to use for parallel execution (default: _threads)
	 *
	 * \param A		Pointer to row-major matrix A, of shape M×N
	 * \param B		Scalar value to broadcast to all elements of A
	 * \param M		Number of rows in A
	 * \param N		Number of columns in A
	 *
	 * \note A must be allocated as contiguous 1D array in row-major layout.
	 * \note This method uses blocked (tiled) iteration for optimal cache performance.
	 * \note Handles asymmetric matrices (M ≠ N) and non-divisible block edges safely.
	 * \note Block size should be selected to fit into L1 data cache optimally.
	 */
	template <typename T, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	_broadcast(T* A, const T B, const size_t M, const size_t N)
	{
		parallel_for(0, M, block_size,
		[&](size_t i)
		{
			for (size_t j = 0; j < N; j += block_size)
			{ 
				size_t m = std::min(block_size, M - i);
				size_t n = std::min(block_size, N - j);
				_broadcast_block<T>(A, B, i, j, m, n);
			}
		}, threads);
	}

	// SIMD kernel declarations for broadcast operations
	template <typename T>
	void _broadcast_block_sse(T* A, const T B);

	template <typename T>
	void _broadcast_block_avx(T* A, const T B);

	template <typename T>
	void _broadcast_block_avx512(T* A, const T B);

	/**
	 * \brief	Broadcast scalar to flattened matrix using SIMD intrinsics.
	 *
	 * Performs SIMD-accelerated scalar broadcasting to fill 1D (flattened row-major) matrix A
	 * with scalar value B. SIMD vectorization is selected via the template parameter `S`. 
	 * This variant accepts flat memory layout directly and processes data in cache-line sized chunks.
	 *
	 * \tparam T			Scalar type (float or double)
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512)
	 * \tparam threads		Number of threads for parallel execution (default: _threads)
	 *
	 * \param A		Pointer to flattened matrix A (row-major order)
	 * \param B		Scalar value to broadcast to all elements
	 * \param M		Number of rows
	 * \param N		Number of columns
	 *
	 * \note Uses broadcast semantics: scalar B is assigned to each element of A.
	 * \note Processes data in cache-line aligned chunks for optimal memory bandwidth.
	 * \note Asymmetric matrices (M ≠ N) are supported.
	 * \note Strides not aligned with SIMD block sizes are safely handled with scalar fallback.
	 */
	template<typename T, SIMD S, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	_broadcast_simd(T* A, const T B, const size_t M, const size_t N)
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
				_broadcast_block_sse<T>(A + offset, B);
			if constexpr (S == AVX)
				_broadcast_block_avx<T>(A + offset, B);
			if constexpr (S == AVX512)
				_broadcast_block_avx512<T>(A + offset, B);
		});

		// Remainder
		for (size_t i = full_blocks * kernel_stride; i < total_elements; ++i)
			A[i] = B;
	}

	/**
	 * \brief	Broadcast scalar to matrix using SIMD intrinsics.
	 *
	 * Performs SIMD-accelerated scalar broadcasting to fill 2D matrix A with scalar value B.
	 * SIMD vectorization is selected at compile time via the template parameter `S`. 
	 * This is the main entry point for SIMD-based broadcast operations on 2D pointer arrays.
	 *
	 * \tparam T			Scalar type (float or double)
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512)
	 * \tparam threads		Number of threads for parallel execution (default: _threads)
	 *
	 * \param A		Pointer to a 2D matrix A
	 * \param B		Scalar value to broadcast to all elements
	 * \param M		Number of rows
	 * \param N		Number of columns
	 *
	 * \note Uses broadcast semantics: scalar B is assigned to each element of A.
	 * \note Internally flattens 2D view to invoke flat SIMD routines for optimal performance.
	 * \note Asymmetric matrices (M ≠ N) are supported.
	 * \note Strides not aligned with SIMD block sizes are safely handled with scalar fallback.
	 */
	template<typename T, SIMD S, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	_broadcast_simd(T** A, const T B, const size_t M, const size_t N)
	{
		T* A0 = &A[0][0];
		_broadcast_simd<T, S, threads>(A0, B, M, N);
	}

	/**
	 * \brief Broadcast scalar to matrix using optimized SIMD and blocking algorithms.
	 *
	 * This is the main public interface for scalar broadcasting. It automatically
	 * selects between SIMD-optimized and standard blocked implementations based on the
	 * specified SIMD instruction set. The function operates on 2D matrix views (pointer-to-pointer 
	 * arrays) with row-major layout.
	 *
	 * The broadcast operation assigns scalar value B to all elements of matrix A, where 
	 * A[i][j] = B for all valid indices. The implementation uses cache-aware
	 * blocking and optional SIMD acceleration for optimal performance.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Target matrix of dimensions M×N in row-major layout.
	 * \param B		Scalar value to broadcast to all matrix elements.
	 * \param M		Number of rows in matrix A.
	 * \param N		Number of columns in matrix A.
	 *
	 * \note Matrix must be allocated as contiguous memory block accessible
	 *       through the 2D pointer interface. Submatrix views are not supported.
	 * \note Non-square matrices and asymmetric dimensions (M ≠ N) are fully supported.
	 * \note When S=NONE, the function uses standard blocked broadcast without SIMD.
	 * \note SIMD implementations automatically handle non-aligned dimensions with scalar fallback.
	 * \note This operation is equivalent to: for(i=0; i<M; i++) for(j=0; j<N; j++) A[i][j] = B;
	 *
	 * \throws std::invalid_argument if matrix pointer is null.
	 * \throws std::runtime_error if memory layout validation fails.
	 */
	template<typename T, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline)) 
	void
	broadcast(T** A, const T B, const size_t M, const size_t N)
	{
		right<T>("broadcast: ", std::make_tuple(A, M, N));

		if constexpr (S == SIMD::NONE) 
			_broadcast<T, block_size, threads>(A, B, M, N);
		else
			_broadcast_simd<T, S, threads>(A, B, M, N);
	}

	/**
	 * \brief Broadcast scalar to matrix using optimized SIMD and blocking algorithms (flat arrays).
	 *
	 * This is the main public interface for scalar broadcasting using flat 1D arrays
	 * in row-major layout. It automatically selects between SIMD-optimized and standard
	 * blocked implementations based on the specified SIMD instruction set.
	 *
	 * The broadcast operation assigns scalar value B to all elements of matrix A, where 
	 * A[i*N + j] = B for all valid indices. The implementation uses cache-aware
	 * blocking and optional SIMD acceleration for optimal performance.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Target matrix stored as flat array of size M×N in row-major layout.
	 * \param B		Scalar value to broadcast to all matrix elements.
	 * \param M		Number of rows.
	 * \param N		Number of columns.
	 *
	 * \note Array must be allocated as contiguous memory block of appropriate size.
	 * \note Non-square matrices and asymmetric dimensions (M ≠ N) are fully supported.
	 * \note When S=NONE, the function uses standard blocked broadcast without SIMD.
	 * \note SIMD implementations automatically handle non-aligned dimensions with scalar fallback.
	 * \note This flat array interface is often preferred for interoperability with other
	 *       libraries or when working with pre-allocated buffers.
	 * \note This operation is equivalent to: for(i=0; i<M*N; i++) A[i] = B;
	 *
	 * \throws std::invalid_argument if matrix pointer is null.
	 * \throws std::runtime_error if memory layout validation fails.
	 */
	template<typename T, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	broadcast(T* A, const T B, const size_t M, const size_t N)
	{
		right<T>("broadcast:", std::make_tuple(A, M, N));

		if constexpr (S == SIMD::NONE)
			_broadcast<T, block_size, threads>(A, B, M, N);
		else
			_broadcast_simd<T, S, threads>(A, B, M, N);
	}
	
	/**
	 * \brief Initialize matrix with unit values using optimized algorithms (flat arrays).
	 *
	 * Fills flat 1D array A (representing an M×N matrix in row-major layout) with unit values.
	 * This is a specialized version of broadcast optimized for the common unit initialization pattern.
	 * Equivalent to broadcast(A, T(1), M, N) but may have additional optimizations.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Target matrix stored as flat array of size M×N in row-major layout.
	 * \param M		Number of rows.
	 * \param N		Number of columns.
	 *
	 * \note All elements will be set to T(1), where T(1) represents the unit value for type T.
	 * \note Uses the same optimized algorithms as broadcast with automatic SIMD selection.
	 * \note Preferred for initialization patterns where all elements should be 1.
	 */
	template<typename T, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	ones(T* A, const size_t M, const size_t N)
	{
		right<T>("ones:", std::make_tuple(A, M, N));

		if constexpr (S == SIMD::NONE)
			_broadcast<T, block_size, threads>(A, T(1), M, N);
		else
			_broadcast_simd<T, S, threads>(A, T(1), M, N);
	}

	/**
	 * \brief Initialize matrix with unit values using optimized algorithms (2D arrays).
	 *
	 * Fills 2D matrix A with unit values. This is a specialized version of broadcast 
	 * optimized for the common unit initialization pattern. Equivalent to broadcast(A, T(1), M, N)
	 * but may have additional optimizations.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Target matrix of dimensions M×N in row-major layout.
	 * \param M		Number of rows in matrix A.
	 * \param N		Number of columns in matrix A.
	 *
	 * \note All elements will be set to T(1), where T(1) represents the unit value for type T.
	 * \note Uses the same optimized algorithms as broadcast with automatic SIMD selection.
	 * \note Preferred for initialization patterns where all elements should be 1.
	 */
	template<typename T, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	ones(T** A, const size_t M, const size_t N)
	{
		right<T>("ones:", std::make_tuple(A, M, N));

		if constexpr (S == SIMD::NONE)
			_broadcast<T, block_size, threads>(A, T(1), M, N);
		else
			_broadcast_simd<T, S, threads>(A, T(1), M, N);
	}

	/**
	 * \brief Initialize matrix with zero values using optimized algorithms (flat arrays).
	 *
	 * Fills flat 1D array A (representing an M×N matrix in row-major layout) with zero values.
	 * This is useful when the target platform does not zero-initialize memory automatically.
	 * This is a specialized version of broadcast optimized for the zero initialization pattern.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Target matrix stored as flat array of size M×N in row-major layout.
	 * \param M		Number of rows.
	 * \param N		Number of columns.
	 *
	 * \note All elements will be set to T(0), where T(0) represents the zero value for type T.
	 * \note Equivalent to broadcast(A, T(0), M, N) but may have additional optimizations.
	 * \note Can be useful if target platform does not zero-initialize memory.
	 */
	template<typename T, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	zeros(T* A, const size_t M, const size_t N)
	{
		right<T>("zeros:", std::make_tuple(A, M, N));

		if constexpr (S == SIMD::NONE)
			_broadcast<T, block_size, threads>(A, T(0), M, N);
		else
			_broadcast_simd<T, S, threads>(A, T(0), M, N);
	}
	
	/**
	 * \brief Initialize matrix with zero values using optimized algorithms (2D arrays).
	 *
	 * Fills 2D matrix A with zero values. This is useful when the target platform 
	 * does not zero-initialize memory automatically. This is a specialized version 
	 * of broadcast optimized for the zero initialization pattern.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Target matrix of dimensions M×N in row-major layout.
	 * \param M		Number of rows in matrix A.
	 * \param N		Number of columns in matrix A.
	 *
	 * \note All elements will be set to T(0), where T(0) represents the zero value for type T.
	 * \note Equivalent to broadcast(A, T(0), M, N) but may have additional optimizations.
	 * \note Can be useful if target platform does not zero-initialize memory.
	 */
	template<typename T, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	zeros(T** A, const size_t M, const size_t N)
	{
		right<T>("zeros:", std::make_tuple(A, M, N));

		if constexpr (S == SIMD::NONE)
			_broadcast<T, block_size, threads>(A, T(0), M, N);
		else
			_broadcast_simd<T, S, threads>(A, T(0), M, N);
	}

	/**
	 * \brief Create identity matrix using optimized algorithms (flat arrays).
	 *
	 * Initializes flat 1D array A (representing an M×N matrix in row-major layout) as an identity matrix.
	 * The matrix is first filled with zeros, then diagonal elements are set to unit values.
	 * This function is useful when the target platform does not zero-initialize memory automatically.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Target matrix stored as flat array of size M×N in row-major layout.
	 * \param M		Number of rows.
	 * \param N		Number of columns.
	 *
	 * \note For square matrices (M=N): creates standard identity matrix with 1s on diagonal.
	 * \note For rectangular matrices: creates identity pattern up to min(M,N) diagonal elements.
	 * \note All off-diagonal elements are set to T(0), diagonal elements to T(1).
	 * \note If memory is already zero-initialized, consider using set_identity() from common.h directly.
	 */
	template<typename T, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	identity(T* A, const size_t M, const size_t N)
	{
		zeros<T, S>(A, M, N);
		set_identity(A, M, N);
	}

	/**
	 * \brief Create identity matrix using optimized algorithms (2D arrays).
	 *
	 * Initializes 2D matrix A as an identity matrix. The matrix is first filled with zeros,
	 * then diagonal elements are set to unit values. This function is useful when the target
	 * platform does not zero-initialize memory automatically.
	 *
	 * \tparam T			Element type of the matrix (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Target matrix of dimensions M×N in row-major layout.
	 * \param M		Number of rows in matrix A.
	 * \param N		Number of columns in matrix A.
	 *
	 * \note For square matrices (M=N): creates standard identity matrix with 1s on diagonal.
	 * \note For rectangular matrices: creates identity pattern up to min(M,N) diagonal elements.
	 * \note All off-diagonal elements are set to T(0), diagonal elements to T(1).
	 * \note If memory is already zero-initialized, consider using set_identity() from common.h directly.
	 */
	template<typename T, SIMD S, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	identity(T** A, const size_t M, const size_t N)
	{
		zeros<T, S>(A, M, N);
		set_identity(A, M, N);
	}

}//namespace damm
#endif //__BROADCAST_H__