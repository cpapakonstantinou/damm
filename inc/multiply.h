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
#include <transpose.h>

#include <iostream>
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
	inline __attribute__((always_inline))
	void
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
	 * \brief kernal for multiply_block_flat. 
	 * Low level function not intended for the public API.
	 * */
	template <typename T, bool TR=false>
	inline __attribute__((always_inline))
	void 
	_multiply_block(T* A, T* B, T* C,
		const size_t I, const size_t J, const size_t K,
		const size_t M, const size_t N, const size_t P)
	{
		for(size_t i = 0; i < M; i++)
			for(size_t j = 0; j < P; j++)
				for(size_t k = 0; k < N; ++k) 
				{
					if constexpr (TR)
						C[(I + i) * P + (J + j)] += A[(I + i) * N + (K + k)] * B[(J + j) * N + (K + k)];
					else
						C[(I + i) * P + (J + j)] += A[(I + i) * N + (K + k)] * B[(K + k) * P + (J + j)];
				}
	}

	/**
	 * \brief	Matrix multiplication using cache-blocked flat (1D) arrays.
	 *
	 * Performs blocked matrix multiplication of A × B = C using tiling for improved
	 * cache locality. Operates on flat row-major 1D arrays. Suitable for use in
	 * performance-sensitive contexts where control over memory layout is critical.
	 *
	 * \tparam T			Scalar type (e.g., float or double)
	 * \tparam TR			If true, matrix B is assumed to be transposed (i.e., Bᵀ)
	 * \tparam block_size	Tile/block width (default: _block_size)
	 * \tparam threads		Number of threads to use in parallel_for
	 *
	 * \param A		Pointer to row-major matrix A, of shape M×N
	 * \param B		Pointer to row-major matrix B, of shape N×P (or P×N if TR=true)
	 * \param C		Pointer to row-major output matrix C, of shape M×P
	 * \param M		Number of rows in A and C
	 * \param N		Number of columns in A and rows in B
	 * \param P		Number of columns in B and C
	 *
	 * \note	This method uses blocked (tiled) iteration over i/j/k loops.
	 * \note	TR=true enables pre-transposed inner matrix B for faster column access.
	 * \note	Block size should be selected to fit into L1 data cache for A, B, and C tiles.
	 * \note	Handles asymmetric matrices (M ≠ N ≠ P) and non-divisible block edges safely.
	 */
	template <typename T, bool TR=false, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	_multiply(T* A, T* B, T* C, const size_t M, const size_t N, const size_t P)
	{
		parallel_for(0, M, block_size,
		[&](size_t i)
		{
			for (size_t j = 0; j < P; j += block_size) 
				for (size_t k = 0; k < N; k += block_size)
				{
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - k);
					size_t p = std::min(block_size, P - j);
					_multiply_block<T, TR>(A, B, C, i, j, k, m, n, p);
				}
		}, threads);
	}

	/**
	 * \brief	Matrix multiplication using cache-blocked 2D pointer-to-pointer matrices.
	 *
	 * Performs blocked matrix multiplication of A × B = C using tiling, with inputs
	 * represented as T** (array of pointers to rows). Suitable for dynamically allocated
	 * or jagged row-major matrices. Enables optional use of transposed B for performance.
	 *
	 * \tparam T			Scalar type (e.g., float or double)
	 * \tparam TR			If true, matrix B is assumed to be transposed (i.e., Bᵀ)
	 * \tparam block_size	Tile/block width (default: _block_size)
	 * \tparam threads		Number of threads to use in parallel_for
	 *
	 * \param A		Pointer to rows of matrix A, of shape M×N
	 * \param B		Pointer to rows of matrix B, of shape N×P (or P×N if TR=true)
	 * \param C		Pointer to rows of output matrix C, of shape M×P
	 * \param M		Number of rows in A and C
	 * \param N		Number of columns in A and rows in B
	 * \param P		Number of columns in B and C
	 *
	 * \note	Uses block tiling to optimize cache performance and reduce TLB pressure.
	 * \note	TR=true enables multiplication with a transposed matrix B for improved memory access patterns.
	 * \note	Supports asymmetric dimensions and non-multiple block sizes.
	 */
	template <typename T, bool TR=false, const size_t block_size = _block_size, const size_t threads = _threads>
	inline __attribute__((always_inline))
	void
	_multiply(T** A, T** B, T** C, const size_t M, const size_t N, const size_t P)
	{
		parallel_for(0, M, block_size,
		[&](size_t i)
		{
			for (size_t j = 0; j < P; j += block_size) 
				for (size_t k = 0; k < N; k += block_size)
				{
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - k);
					size_t p = std::min(block_size, P - j);
					_multiply_block<T, TR>(A, B, C, i, j, k, m, n, p);
				}
		}, threads);
	}

	/**
	 * \brief	Multiplication of a matrix using SSE intrinsics.
	 *
	 * Uses 128-bit SSE registers for performing blocked matrix multiplication.
	 * - For \c float: operates on 4×4 blocks in parallel
	 * - For \c double: operates on 2×2 blocks in parallel
	 *
	 * \tparam T	The scalar type (e.g., float or double)
	 *
	 * \param A		The left-hand matrix (MxN), in row-major order
	 * \param B		The right-hand matrix (NxP), in row-major order
	 * \param C		Output matrix to store the result A*B (MxP), in row-major order
	 * \param M		Number of rows in A and C
	 * \param N		Number of columns in A, and rows in B
	 * \param P		Number of columns in B and C
	 *
	 * \note		Asymmetric matrices (e.g., M ≠ P) are fully supported.
	 * \note		Strides not divisible by the SIMD width are handled, but may result in unaligned access and reduced performance.
	 */
	template <typename T>
	void 
	_multiply_block_sse(T* A, T* B, T* C, const size_t M, const size_t N, const size_t P);


	/**
	 * \brief	Multiplication of a matrix using AVX intrinsics.
	 *
	 * Uses 256-bit AVX registers for performing blocked matrix multiplication.
	 * - For \c float: operates on 8×8 blocks in parallel
	 * - For \c double: operates on 4×4 blocks in parallel
	 *
	 * \tparam T	The scalar type (e.g., float or double)
	 *
	 * \param A		The left-hand matrix (MxN), in row-major order
	 * \param B		The right-hand matrix (NxP), in row-major order
	 * \param C		Output matrix to store the result A*B (MxP), in row-major order
	 * \param M		Number of rows in A and C
	 * \param N		Number of columns in A, and rows in B
	 * \param P		Number of columns in B and C
	 *
	 * \note		Asymmetric matrices are supported.
	 * \note		Strides not divisible by the SIMD width are handled with fallback code paths.
	 */
	template <typename T>
	void 
	_multiply_block_avx(T* A, T* B, T* C, const size_t M, const size_t N, const size_t P);


	/**
	 * \brief	Multiplication of a matrix using AVX-512 intrinsics.
	 *
	 * Uses 512-bit AVX-512 registers for performing blocked matrix multiplication.
	 * - For \c float: operates on 16×16 blocks in parallel
	 * - For \c double: operates on 8×8 blocks in parallel
	 *
	 * \tparam T	The scalar type (e.g., float or double)
	 *
	 * \param A		The left-hand matrix (MxN), in row-major order
	 * \param B		The right-hand matrix (NxP), in row-major order
	 * \param C		Output matrix to store the result A*B (MxP), in row-major order
	 * \param M		Number of rows in A and C
	 * \param N		Number of columns in A, and rows in B
	 * \param P		Number of columns in B and C
	 *
	 * \note		Supports non-square and asymmetric dimensions.
	 * \note		Non-aligned strides are handled at a performance cost due to fallback scalar paths.
	 */
	template <typename T>
	void 
	_multiply_block_avx512(T* A, T* B, T* C, const size_t M, const size_t N, const size_t P);


	/**
	 * \brief	Multiply matrices using SIMD intrinsics (flat arrays).
	 *
	 * Performs SIMD-accelerated matrix multiplication A × B = C using tiled blocking
	 * and architecture-specific intrinsics (SSE, AVX, AVX512). Accepts 1D flat arrays
	 * in row-major layout. This is the main SIMD entry point for flat arrays.
	 *
	 * \tparam T			Scalar type (e.g., float or double)
	 * \tparam S			SIMD type to use (SSE, AVX, or AVX512)
	 * \tparam threads		Number of threads for parallel_for
	 *
	 * \param A				Pointer to matrix A, shape M×N (row-major)
	 * \param B				Pointer to matrix B, shape N×P (row-major)
	 * \param C				Pointer to output matrix C, shape M×P (row-major)
	 * \param M				Number of rows in A and C
	 * \param N				Number of columns in A and rows in B
	 * \param P				Number of columns in B and C
	 *
	 * \note	Performs internal transposition of B into SIMD-aligned buffer.
	 * \note	Handles edge tiles using fallback scalar implementation.
	 * \note	Input pointers are validated for contiguity and size correctness.
	 * \note	Supports asymmetric matrices and non-divisible tile dimensions.
	 * \note	This should always be used as the entry point for flat SIMD block multiply.
	 */
	template<typename T, SIMD S, const size_t threads = _threads> 
	inline __attribute__((always_inline))
	void
	_multiply_simd(T* A, T* B, T* C, const size_t M, const size_t N, const size_t P)
	{
		constexpr size_t block_size = static_cast<size_t>(S/sizeof(T));
		
		auto Bt = aligned_alloc_1D<T, S>(P, N);
		transpose<T, S>(B, Bt.get(), N, P);
		T* A0 = A;
		T* B0 = Bt.get();
		T* C0 = C;
	
		const size_t simd_rows_M = M - (M % block_size);
		const size_t simd_cols_P = P - (P % block_size);
		const size_t simd_inner_N = N - (N % block_size);
			
		parallel_for(0, M / block_size, 1, [&](size_t bi) 
		{
			size_t i = bi * block_size;			
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
		}, threads);
		// remainder rows in A
		if (M % block_size != 0) 
		{
			const size_t rem_rows = M % block_size;
			_multiply_block<T, true>(A, B0, C, simd_rows_M, 0, 0, rem_rows, N, P);
		}

		// remainder columns in B
		if (P % block_size != 0) 
		{
			const size_t rem_cols = P % block_size;
			_multiply_block<T, true>(A, B0, C, 0, simd_cols_P, 0, simd_rows_M, N, rem_cols);
		}

		// remainder columns of A and rows of B
		if (N % block_size != 0) 
		{
			const size_t rem_inner = N % block_size;
			_multiply_block<T, true>(A, B0, C, 0, 0, simd_inner_N, simd_rows_M, rem_inner, simd_cols_P);
		}
	}

	/**
	 * \brief	Multiply matrices using SIMD intrinsics (2D arrays).
	 *
	 * Performs SIMD-accelerated matrix multiplication A × B = C using blocking and
	 * SIMD intrinsics. Accepts T** input matrices (array-of-rows) and internally
	 * flattens them to invoke the flat SIMD routines. Handles all edge cases safely.
	 *
	 * \tparam T			Scalar type (e.g., float or double)
	 * \tparam S			SIMD type to use (SSE, AVX, or AVX512)
	 * \tparam threads		Number of threads
	 *
	 * \param A				Matrix A as array of M row pointers, each of size N
	 * \param B				Matrix B as array of N row pointers, each of size P
	 * \param C				Matrix C as array of M row pointers, each of size P
	 * \param M				Number of rows in A and C
	 * \param N				Number of columns in A and rows in B
	 * \param P				Number of columns in B and C
	 *
	 * \note	Matrices are validated for size and alignment.
	 * \note	B is internally transposed into SIMD-aligned layout.
	 * \note	Handles asymmetric and misaligned matrix sizes.
	 * \note	Fallback scalar multiply is used for remainder tiles.
	 * \note	This should always be used as the entry point for 2D SIMD block multiply.
	 */
	template<typename T, SIMD S, const size_t threads = _threads> 
	inline __attribute__((always_inline))
	void
	_multiply_simd(T** A, T** B, T** C, const size_t M, const size_t N, const size_t P)
	{
		constexpr size_t block_size = static_cast<size_t>(S/sizeof(T));
		
		//currently the simd functions only interface to flat pointers
		//the variable alias could be removed if implementing a dense interface function to access the simd routines
		auto Bt = aligned_alloc_2D<T, S>(P, N);
		transpose<T, S>(&B[0], &Bt[0], N, P);
		T* A0 = &A[0][0];
		T* B0 = &Bt[0][0];
		T* C0 = &C[0][0];

		const size_t simd_rows_M = M - (M % block_size);
		const size_t simd_cols_P = P - (P % block_size);
		const size_t simd_inner_N = N - (N % block_size);
			
		parallel_for(0, M / block_size, 1, [&](size_t bi) 
		{
			size_t i = bi * block_size; 
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
		}, threads);
		// remainder rows in A
		if (M % block_size != 0) 
		{
			const size_t rem_rows = M % block_size;
			_multiply_block<T, true>(A, Bt.get(), C, simd_rows_M, 0, 0, rem_rows, N, P);	
		}

		// remainder columns in B
		if (P % block_size != 0) 
		{
			const size_t rem_cols = P % block_size;
			_multiply_block<T, true>(A, Bt.get(), C, 0, simd_cols_P, 0, simd_rows_M, N, rem_cols);
		}

		// remainder columns of A and rows of B
		if (N % block_size != 0) 
		{
			const size_t rem_inner = N % block_size;
			_multiply_block<T, true>(A, Bt.get(), C, 0, 0, simd_inner_N, simd_rows_M, rem_inner, simd_cols_P);
		}

	}

	/**
	 * \brief Perform optimized matrix multiplication using SIMD and blocking algorithms.
	 *
	 * This is the main public interface for matrix multiplication A × B = C. It automatically
	 * selects between SIMD-optimized and standard blocked implementations based on the
	 * specified SIMD instruction set. The function operates on 2D matrix views (pointer-to-pointer 
	 * arrays) with row-major layout.
	 *
	 * The multiplication operation computes C[i][j] = Σ(A[i][k] * B[k][j]) for all valid
	 * indices. Matrix A must have dimensions M×N, matrix B must have dimensions N×P, and
	 * the result matrix C will have dimensions M×P.
	 *
	 * \tparam T			Element type of the matrices (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Left operand matrix of dimensions M×N in row-major layout.
	 * \param B		Right operand matrix of dimensions N×P in row-major layout.
	 * \param C		Result matrix of dimensions M×P in row-major layout.
	 * \param M		Number of rows in matrix A (and result matrix C).
	 * \param N		Number of columns in matrix A and rows in matrix B.
	 * \param P		Number of columns in matrix B (and result matrix C).
	 *
	 * \note All matrices must be allocated as contiguous memory blocks accessible
	 *       through the 2D pointer interface. Submatrix views are not supported.
	 * \note Matrix C should be zero-initialized before calling this function if accumulation
	 *       is not desired, as the implementation uses += operations internally.
	 * \note Non-square matrices and asymmetric dimensions (M ≠ N ≠ P) are fully supported.
	 * \note When S=NONE, the function uses standard blocked multiplication without SIMD.
	 * \note SIMD implementations automatically handle non-aligned dimensions with scalar fallback.
	 *
	 * \throws std::invalid_argument if any matrix pointer is null.
	 * \throws std::runtime_error if memory layout validation fails.
	 */
	template<typename T, SIMD S = detect_simd(), const size_t block_size = _block_size, const size_t threads = _threads>
	inline 
	void multiply(T** A, T** B, T** C, const size_t M, const size_t N, const size_t P)
	{
		right<T>("multiply:", 
			std::make_tuple(A, M, N), 
			std::make_tuple(B, N, P), 
			std::make_tuple(C, M, P));

		if constexpr (S == SIMD::NONE) 
			_multiply<T, false, block_size, threads>(A, B, C, M, N, P);
		else
			_multiply_simd<T, S, threads>(A, B, C, M, N, P);
	}

	/**
	 * \brief Perform optimized matrix multiplication using SIMD and blocking algorithms (flat arrays).
	 *
	 * This is the main public interface for matrix multiplication using flat 1D arrays
	 * in row-major layout. It automatically selects between SIMD-optimized and standard
	 * blocked implementations based on the specified SIMD instruction set.
	 *
	 * The multiplication operation computes C[i*P + j] = Σ(A[i*N + k] * B[k*P + j])
	 * for all valid indices. Matrix A must be of size M×N, matrix B must be of size N×P,
	 * and the result matrix C will be of size M×P.
	 *
	 * \tparam T			Element type of the matrices (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size).
	 * \tparam threads		Number of threads for parallel execution (default: _threads).
	 *
	 * \param A		Left operand matrix stored as flat array of size M×N in row-major layout.
	 * \param B		Right operand matrix stored as flat array of size N×P in row-major layout.
	 * \param C		Result matrix stored as flat array of size M×P in row-major layout.
	 * \param M		Number of rows in matrix A (and result matrix C).
	 * \param N		Number of columns in matrix A and rows in matrix B.
	 * \param P		Number of columns in matrix B (and result matrix C).
	 *
	 * \note All arrays must be allocated as contiguous memory blocks of appropriate size.
	 * \note Matrix C should be zero-initialized before calling this function if accumulation
	 *       is not desired, as the implementation uses += operations internally.
	 * \note Non-square matrices and asymmetric dimensions (M ≠ N ≠ P) are fully supported.
	 * \note When S=NONE, the function uses standard blocked multiplication without SIMD.
	 * \note SIMD implementations automatically handle non-aligned dimensions with scalar fallback.
	 * \note This flat array interface is often preferred for interoperability with other
	 *       libraries or when working with pre-allocated buffers.
	 *
	 * \throws std::invalid_argument if any matrix pointer is null.
	 * \throws std::runtime_error if memory layout validation fails
	 *
	 */
	template<typename T, SIMD S = detect_simd(), const size_t block_size = _block_size, const size_t threads = _threads>
	inline 
	void multiply(T* A, T* B, T* C, const size_t M, const size_t N, const size_t P)
	{
		right<T>("multiply:", 
			std::make_tuple(A, M, N), 
			std::make_tuple(B, N, P), 
			std::make_tuple(C, M, P));

		if constexpr (S == SIMD::NONE)
			_multiply<T, false, block_size, threads>(A, B, C, M, N, P);
		else
			_multiply_simd<T, S, threads>(A, B, C, M, N, P);
	}

}//namespace damm

#endif //__MULTIPLY_H__