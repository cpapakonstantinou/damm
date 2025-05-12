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
	inline __attribute__((always_inline))
	void
	_transpose_block(T** A, T** B, const size_t I, const size_t J, const size_t N, const size_t M)
	{
		for(size_t i=0; i < N; i++) 
			for(size_t j=0; j < M; j++) 
				 B[J + j][I + i] = A[I + i][J + j];
	}
	/** \brief kernal for transpose_block_flat. Low level function not intended for the public API*/
	template <typename T>
	inline __attribute__((always_inline))
	void 
	_transpose_block_flat(T* A, T* B, const size_t I, const size_t J, const size_t N, const size_t M)
	{
		for(size_t i = 0; i < N; i++) 
			for(size_t j = 0; j < M; j++) 
				B[(J + j) * N + (I + i)] = A[(I + i) * M + (J + j)];
	}

	/**
	 * \brief Transpose a matrix using a blocked and parallel strategy.
	 *
	 * This function performs a blocked (tiled) transpose of a row-major matrix \c A 
	 * into a row-major matrix \c B. The matrix is processed in blocks of size 
	 * \c block_size × block_size to improve cache locality, particularly fitting 
	 * blocks into L1 cache. The function supports multithreaded execution via \c parallel_for.
	 *
	 * \tparam T			Element type of the matrices (e.g., float, double).
	 * \tparam block_size	Size of the square block used in the tiling strategy.
	 * \tparam threads		Number of threads to use for parallel execution.
	 *
	 * \param A Pointer to the source matrix (row-major layout) of size N × M.
	 * \param B Pointer to the destination matrix (row-major layout) of size M × N.
	 * \param N Number of rows in matrix A (and columns in matrix B).
	 * \param M Number of columns in matrix A (and rows in matrix B).
	 *
	 * \note Both \c A and \c B must be allocated as contiguous 1D arrays in row-major layout.
	 * \note This function assumes both matrices are fully allocated (not submatrices).
	 * \note Asymmetric matrices (\c N ≠ \c M) and matrices whose dimensions are not multiples
	 *       of \c block_size are fully supported; partial edge blocks are handled correctly.
	 *
	 * \throws std::invalid_argument if either A or B is a null pointer.
	 * \throws std::runtime_error if either A or B is not allocated as a contiguous row-major matrix.
	 *
	 * \warning This function validates the contiguity of both A and B at runtime. If your
	 *			matrices are subviews or derived from non-contiguous allocations, validation will fail.
	 *
	 * \example
	 * \code
	 * constexpr size_t N = 512, M = 768;
	 * auto A = aligned_alloc_1D<float, 64>(N, M);
	 * auto B = aligned_alloc_1D<float, 64>(M, N);
	 * 
	 * transpose_block_flat<float, 64, 4>(A.get(), B.get(), N, M);
	 * \endcode
	 */
	template <typename T, const size_t block_size = _block_size, const size_t threads = _threads>
	inline void
	transpose_block_flat(T* A, T* B, const size_t N, const size_t M)
	{
		if (!A || !B)
			throw std::invalid_argument("transpose_block_simd: null pointer");

		// Validate A is a properly laid out N×M block
		const T* A_base = A[0];
		const T* A_end  = A[N - 1] + M;
		if (reinterpret_cast<uintptr_t>(A_end) - reinterpret_cast<uintptr_t>(A_base) < N * M * sizeof(T))
			throw std::runtime_error("transpose_block_simd: A is not a contiguous N×M block");

		// Validate B is a properly laid out M×N block
		const T* B_base = B[0];
		const T* B_end  = B[M - 1] + N;
		if (reinterpret_cast<uintptr_t>(B_end) - reinterpret_cast<uintptr_t>(B_base) < M * N * sizeof(T))
			throw std::runtime_error("transpose_block_simd: B is not a contiguous M×N block");

		parallel_for(0, N, block_size,
		[&](size_t i)
		{
			for (size_t j = 0; j < M; j += block_size) 
			{
				size_t n = std::min(block_size, N - i);
				size_t m = std::min(block_size, M - j);
				_transpose_block_flat(A, B, i, j, n, m);
			}
		}, threads);
	}

	/**
	 * \brief Transpose a matrix using a blocked and parallel strategy.
	 *
	 * This function performs a block-wise transpose of a matrix \c A into \c B,
	 * optimizing for cache locality and enabling multithreaded execution. It operates
	 * on matrix views that provide 2D access (e.g., row-major pointer-to-pointer layout),
	 * typically created from contiguous memory.
	 *
	 * \tparam T			Element type of the matrices (e.g., float, double).
	 * \tparam block_size	Size of the square block used in the tiling strategy.
	 * \tparam threads		Number of threads to use for parallel execution.
	 *
	 * \param A Source matrix view with dimensions N × M.
	 * \param B Destination matrix view with dimensions M × N.
	 * \param N Number of rows in matrix A.
	 * \param M Number of columns in matrix A.
	 *
	 * \note Both \c A and \c B must be valid 2D views over contiguous row-major memory.
	 *       This function assumes full allocation, not submatrices.
	 * \note The transpose supports asymmetric matrices (N ≠ M) and dimensions
	 *       that are not multiples of the block size.
	 *
	 * \throws std::invalid_argument if either A or B is null.
	 * \throws std::runtime_error if the underlying memory layout is not contiguous.
	 *
	 * \example
	 * \code
	 * constexpr size_t N = 512, M = 256;
	 * auto A = aligned_alloc_2D<float, 64>(N, M);
	 * auto B = aligned_alloc_2D<float, 64>(M, N);
	 *
	 * transpose_block<float, 64, 4>(A.get(), B.get(), N, M);
	 * \endcode
	 */
	template <typename T, const size_t block_size = _block_size, const size_t threads = _threads>
	inline void
	transpose_block(T** A, T** B, const size_t N, const size_t M)
	{
		if (!A || !B)
			throw std::invalid_argument("transpose_block_simd: null pointer");

		// Validate A is a properly laid out N×M block
		const T* A_base = A[0];
		const T* A_end  = A[N - 1] + M;
		if (reinterpret_cast<uintptr_t>(A_end) - reinterpret_cast<uintptr_t>(A_base) < N * M * sizeof(T))
			throw std::runtime_error("transpose_block_simd: A is not a contiguous N×M block");

		// Validate B is a properly laid out M×N block
		const T* B_base = B[0];
		const T* B_end  = B[M - 1] + N;
		if (reinterpret_cast<uintptr_t>(B_end) - reinterpret_cast<uintptr_t>(B_base) < M * N * sizeof(T))
			throw std::runtime_error("transpose_block_simd: B is not a contiguous M×N block");

		parallel_for(0, N, block_size,
		[&](size_t i)
		{
			for (size_t j = 0; j < M; j += block_size) 
			{
				size_t n = std::min(block_size, N - i);
				size_t m = std::min(block_size, M - j);
				_transpose_block(A, B, i, j, n, m);
			}
		}, threads);
	}

	/**
	 * \brief Transpose a matrix block using SSE intrinsics.
	 *
	 * This function uses 128-bit SSE registers to perform a block transpose.
	 * For \c float types, it effectively transposes 4x4 blocks in parallel.
	 * For \c double types, it transposes 2x2 blocks in parallel.
	 *
	 * \tparam T		The scalar type, typically float or double.
	 * \param A			Pointer to the source matrix block (row-major layout).
	 * \param B			Pointer to the destination matrix block (row-major layout).
	 * \param N			The leading dimension (stride) of \c A.
	 * \param M			The leading dimension (stride) of \c B.
	 *
	 * \note Strides \c N and \c M not divisible by the SIMD block size may cause
	 *       unaligned loads/stores, leading to performance penalties.
	 */
	template <typename T>
	void 
	_transpose_block_sse(T* A, T* B, const size_t N, const size_t M);


	/**
	 * \brief Transpose a matrix block using AVX intrinsics.
	 *
	 * This function uses 256-bit AVX registers to perform a block transpose.
	 * For \c float types, it transposes 8x8 blocks in parallel.
	 * For \c double types, it transposes 4x4 blocks in parallel.
	 *
	 * \tparam T		The scalar type, typically float or double.
	 * \param A			Pointer to the source matrix block (row-major layout).
	 * \param B			Pointer to the destination matrix block (row-major layout).
	 * \param N			The leading dimension (stride) of \c A.
	 * \param M			The leading dimension (stride) of \c B.
	 *
	 * \note Strides \c N and \c M not divisible by the SIMD block size may cause
	 *       unaligned loads/stores, leading to performance penalties.
	 */
	template <typename T>
	void
	_transpose_block_avx(T* A, T* B, const size_t N, const size_t M);


	/**
	 * \brief Transpose a matrix block using AVX-512 intrinsics.
	 *
	 * This function uses 512-bit AVX-512 registers to perform a block transpose.
	 * For \c float types, it transposes 16x16 blocks in parallel.
	 * For \c double types, it transposes 8x8 blocks in parallel.
	 *
	 * \tparam T		The scalar type, typically float or double.
	 * \param A			Pointer to the source matrix block (row-major layout).
	 * \param B			Pointer to the destination matrix block (row-major layout).
	 * \param N			The leading dimension (stride) of \c A.
	 * \param M			The leading dimension (stride) of \c B.
	 *
	 * \note Strides \c N and \c M not divisible by the SIMD block size may cause
	 *       unaligned loads/stores, leading to performance penalties.
	 */
	template <typename T>
	void
	_transpose_block_avx512(T* A, T* B, const size_t N, const size_t M);

	/**
	 * \brief Transpose a matrix block using SIMD intrinsics.
	 *
	 * This function serves as the main entry point for SIMD-based block transposition.
	 * The SIMD instruction set is specified via the template parameter \c S, allowing
	 * selection between SSE, AVX, AVX-512, or other SIMD types as implemented.
	 * 
	 * It efficiently transposes the matrix \c A (with row-major layout) into \c B, using
	 * blocked operations optimized for the selected SIMD register width.
	 * 
	 * \tparam T		The scalar data type of the matrix elements (e.g., float, double).
	 * \tparam S		The SIMD instruction set tag (e.g., SSE, AVX, AVX512).
	 * \tparam thread	The number of threads to use for parallel execution (default: \c _threads).
	 * 
	 * \param A_endPointer to the input matrix \c A stored in row-major order.
	 * \param B			 Pointer to the output matrix \c B stored in row-major order.
	 * \param N			 The leading dimension (stride) of \c A.
	 * \param M			 The leading dimension (stride) of \c B.
	 * 
	 * \note Asymmetric matrices (\c N != \c M) are supported without restrictions.
	 * \note If \c N or \c M are not multiples of the SIMD register width, the implementation
	 *       handles the remaining edge elements sequentially to ensure correctness.
	 * \note Using strides not aligned to SIMD register sizes may cause unaligned memory
	 *       accesses, potentially incurring performance penalties.
	 */
	template<typename T, SIMD S, const size_t threads = _threads> 
	inline void
	transpose_block_simd_flat(T* A, T* B, const size_t N, const size_t M)
	{
		if (!A || !B)
			throw std::invalid_argument("transpose_block_simd: null pointer");

		// Validate A is a properly laid out N×M block
		const T* A_base = A[0];
		const T* A_end  = A[N - 1] + M;
		if (reinterpret_cast<uintptr_t>(A_end) - reinterpret_cast<uintptr_t>(A_base) < N * M * sizeof(T))
			throw std::runtime_error("transpose_block_simd: A is not a contiguous N×M block");

		// Validate B is a properly laid out M×N block
		const T* B_base = B[0];
		const T* B_end  = B[M - 1] + N;
		if (reinterpret_cast<uintptr_t>(B_end) - reinterpret_cast<uintptr_t>(B_base) < M * N * sizeof(T))
			throw std::runtime_error("transpose_block_simd: B is not a contiguous M×N block");

		constexpr size_t block_size = static_cast<size_t>(S/sizeof(T));

		const size_t simd_rows = N - (N % block_size);
		const size_t simd_cols = M - (M % block_size);

		parallel_for(0, N / block_size, 1, 
			[&](size_t bi) 
			{
				size_t i = bi * block_size;
				for (size_t j = 0; j + block_size <= M; j += block_size) 
				{
					if constexpr (S == SSE)
						_transpose_block_sse<T>(&A[i * M + j], &B[j * N + i], M, N);
					else if constexpr (S == AVX)
						_transpose_block_avx<T>(&A[i * M + j], &B[j * N + i], M, N);
					else if constexpr (S == AVX512)
						_transpose_block_avx512<T>(&A[i * M + j], &B[j * N + i], M, N);
				}
			}, threads);
		// Handle remaining rows
		if (N % block_size != 0) 
		{
			size_t rem = N % block_size;
			for (size_t j = 0; j < simd_cols; j += block_size)
				_transpose_block_flat<T>(A, B, simd_rows, j, rem, block_size);
		}

		// Handle remaining columns
		if (M % block_size != 0)
		{
			size_t rem = M % block_size;
			for (size_t i = 0; i < simd_rows; i += block_size)
				_transpose_block_flat<T>(A, B, i, simd_cols, block_size, rem);
		}

		// Handle remaining rows and columns
		if ((N % block_size != 0) && (M % block_size != 0))
		{
			_transpose_block_flat<T>(A, B, simd_rows, simd_cols, N % block_size, M % block_size);
		}	
	}
	/**
	 * \brief Transpose a matrix block using SIMD intrinsics.
	 *
	 * This function serves as the main entry point for SIMD-based block transposition.
	 * The SIMD instruction set is specified via the template parameter \c S, allowing
	 * selection between SSE, AVX, AVX-512, or other SIMD types as implemented.
	 * 
	 * It efficiently transposes the matrix \c A (with row-major layout) into \c B, using
	 * blocked operations optimized for the selected SIMD register width.
	 * 
	 * \tparam T		The scalar data type of the matrix elements (e.g., float, double).
	 * \tparam S		The SIMD instruction set tag (e.g., SSE, AVX, AVX512).
	 * \tparam thread	The number of threads to use for parallel execution (default: \c _threads).
	 * 
	 * \param A			 Pointer to the input matrix \c A stored in row-major order.
	 * \param B			 Pointer to the output matrix \c B stored in row-major order.
	 * \param N			 The leading dimension (stride) of \c A.
	 * \param M			 The leading dimension (stride) of \c B.
	 * 
	 * \note Asymmetric matrices (\c N != \c M) are supported without restrictions.
	 * \note If \c N or \c M are not multiples of the SIMD register width, the implementation
	 *       handles the remaining edge elements sequentially to ensure correctness.
	 * \note Using strides not aligned to SIMD register sizes may cause unaligned memory
	 *       accesses, potentially incurring performance penalties.
	 */
	template<typename T, SIMD S, const size_t threads = _threads> 
	inline void
	transpose_block_simd(T** A, T** B, const size_t N, const size_t M)
	{
		if (!A || !B)
			throw std::invalid_argument("transpose_block_simd: null pointer");

		// Validate A is a properly laid out N×M block
		const T* A_base = A[0];
		const T* A_end  = A[N - 1] + M;
		if (reinterpret_cast<uintptr_t>(A_end) - reinterpret_cast<uintptr_t>(A_base) < N * M * sizeof(T))
			throw std::runtime_error("transpose_block_simd: A is not a contiguous N×M block");

		// Validate B is a properly laid out M×N block
		const T* B_base = B[0];
		const T* B_end  = B[M - 1] + N;
		if (reinterpret_cast<uintptr_t>(B_end) - reinterpret_cast<uintptr_t>(B_base) < M * N * sizeof(T))
			throw std::runtime_error("transpose_block_simd: B is not a contiguous M×N block");

		constexpr size_t block_size = static_cast<size_t>(S/sizeof(T));

		//currently the simd functions only interface to flat pointers
		//the variable alias could be removed if implementing a dense interface function to access the simd routines
		T* A0 = &A[0][0];
		T* B0 = &B[0][0];

		const size_t simd_rows = N - (N % block_size);
		const size_t simd_cols = M - (M % block_size);

		parallel_for(0, N / block_size, 1, 
			[&](size_t bi) 
			{
				size_t i = bi * block_size;
				for (size_t j = 0; j + block_size <= M; j += block_size) 
				{
					if constexpr (S == SSE)
						_transpose_block_sse<T>(&A0[i * M + j], &B0[j * N + i], M, N);
					if constexpr (S == AVX)
						_transpose_block_avx<T>(&A0[i * M + j], &B0[j * N + i], M, N);
					if constexpr (S == AVX512)
						_transpose_block_avx512<T>(&A0[i * M + j], &B0[j * N + i], M, N);
				}
			}, threads);

		// Handle remaining rows
		if (N % block_size != 0) 
		{
			size_t rem = N % block_size;
			for (size_t j = 0; j < simd_cols; j += block_size)
				_transpose_block<T>(A, B, simd_rows, j, rem, block_size);
		}

		// Handle remaining columns
		if (M % block_size != 0)
		{
			size_t rem = M % block_size;
			for (size_t i = 0; i < simd_rows; i += block_size)
				_transpose_block<T>(A, B, i, simd_cols, block_size, rem);
		}

		// Handle remaining rows and columns
		if ((N % block_size != 0) && (M % block_size != 0))
		{
			_transpose_block<T>(A, B, simd_rows, simd_cols, N % block_size, M % block_size);
		}	
	}
}//namespace damm
#endif //__TRANSPOSE_H__