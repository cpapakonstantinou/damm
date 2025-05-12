#ifndef __FUSED_UNION_H__
#define __FUSED_UNION_H__
/**
 * \file fused_union.h
 * \brief definitions for fused union utilities 
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
 * Fused Union Operations - Efficient Ternary Matrix Operations.
 *
 * \note
 * Fused union operations combine two binary operators to perform efficient ternary operations
 * with configurable operation order via FusionPolicy. These operations eliminate intermediate
 * storage and improve cache efficiency through loop fusion, unlocking comprehensive BLAS-style
 * patterns including AXPY families, scalar variants, and Hadamard fusion operations.
 *
 * \note
 * The scalar namespace provides signature-controlled operation ordering, enabling flexible
 * BLAS patterns like A + α*B, (A+B)*α, and α/(A+B) through strategic operand positioning.
 * The matrix namespace supports full element-wise fusion of three matrices with configurable
 * operation precedence.
 *
 * \note
 * All operations support asymmetric matrices (M ≠ N) and provide both 2D pointer and flat
 * array interfaces. SIMD acceleration (SSE/AVX/AVX512) is automatically selected based on
 * compile-time detection or explicit template parameters.
 */
namespace damm
{
	/**
	 * \brief Fusion Policy to define Order of Operations.
	 * UNION_FIRST:  First combine primary operands (A and B), then apply operation with third operand
	 * FUSION_FIRST: First apply operation to one primary operand with third operand, then combine with remaining primary operand
	 *
	 * SCALAR NAMESPACE COVERAGE (signature-controlled):
	 * fused_union(A, B, C, D): UNION_FIRST  -> D = OP2((A OP1 B), C)  // (A + B) * C, (A - B) / C
	 * fused_union(A, C, B, D): UNION_FIRST  -> D = OP2(A, (C OP1 B))  // A + (C * B), A - (C * B)
	 * fused_union(A, B, C, D): FUSION_FIRST -> D = OP1(A, (B OP2 C))  // A + (B * C), A - (B / C)
	 * fused_union(C, A, B, D): FUSION_FIRST -> D = OP1((C OP2 A), B)  // (C * A) + B, (C / A) - B
	 *
	 * MATRIX NAMESPACE COVERAGE:
	 * UNION_FIRST:  (A + B) ⊙ D, (A - B) + D, (A ⊙ B) / D
	 * FUSION_FIRST: A + (B ⊙ D), A - (B ⊙ D), A ⊙ (B / D)
	 *
	 * KEY BLAS PATTERNS:
	 * AXPY family:     A + D*B, A - D*B (via signature control)
	 * SCAL variants:   D*(A+B), D*(A-B), (A+B)*D, (A-B)*D  
	 * Hadamard fusion: A + B⊙D, A ⊙ (B+D)
	 * Reciprocal ops:  A + D/B, A + B/D, D/(A+B), (A+B)/D
	 */
	enum class FusionPolicy 
	{
		UNION_FIRST,    ///< OP2(OP1(A, B), C) - combine A and B first, then apply OP2 with C
		FUSION_FIRST    ///< OP1(A, OP2(B, C)) - apply OP2 to B and C first, then apply OP1 with A
	};

	namespace scalar 
	{
		/** 
		 * \brief kernel for scalar fused_union_block with scalar operand. 
		 * Low level function not intended for the public API.
		 * */
		template <FusionPolicy P, typename T, typename O1, typename O2>
		inline __attribute__((always_inline)) 
		void
		_fused_union_block(T** A, T** B, const T C, T** D, 
			const size_t I, const size_t J,
			const size_t M, const size_t N)
		{
			for(size_t i = 0; i < M; i++) {
				for(size_t j = 0; j < N; j++) {
					if constexpr (P == FusionPolicy::UNION_FIRST) 
						D[I+i][J+j] = O2{}(O1{}(A[I+i][J+j], B[I+i][J+j]), C);
					else // FUSION_FIRST
						D[I+i][J+j] = O1{}(A[I+i][J+j], O2{}(B[I+i][J+j], C));
				}
			}
		}

		/** 
		 * \brief kernel for scalar fused_union_block with scalar operand. 
		 * Low level function not intended for the public API.
		 * */
		template <FusionPolicy P, typename T, typename O1, typename O2>
		inline __attribute__((always_inline)) 
		void
		_fused_union_block(T* A, T* B, const T C, T* D, 
			const size_t I, const size_t J,
			const size_t M, const size_t N)
		{
			for(size_t i = 0; i < M; i++) {
				for(size_t j = 0; j < N; j++) {
					if constexpr (P == FusionPolicy::UNION_FIRST)
						D[(I+i)*N + (J+j)] = O2{}(O1{}(A[(I+i)*N + (J+j)], B[(I+i)*N + (J+j)]), C);
					else // FUSION_FIRST
						D[(I+i)*N + (J+j)] = O1{}(A[(I+i)*N + (J+j)], O2{}(B[(I+i)*N + (J+j)], C));
				}
			}
		}

		/** 
		 * \brief kernel for scalar fused_union_block with scalar in different position. 
		 * Low level function not intended for the public API.
		 * */
		template <FusionPolicy P, typename T, typename O1, typename O2>
		inline __attribute__((always_inline)) 
		void
		_fused_union_block(T** A, const T B, T** C, T** D, 
			const size_t I, const size_t J,
			const size_t M, const size_t N)
		{
			for(size_t i = 0; i < M; i++) {
				for(size_t j = 0; j < N; j++) {
					if constexpr (P == FusionPolicy::UNION_FIRST) 
						D[I+i][J+j] = O2{}(B, O1{}(A[I+i][J+j], C[I+i][J+j]));
					else // FUSION_FIRST
						D[I+i][J+j] = O1{}(A[I+i][J+j], O2{}(B, C[I+i][J+j]));
				}
			}
		}
		
		/** 
		 * \brief kernel for scalar fused_union_block with scalar in different position. 
		 * Low level function not intended for the public API.
		 * */
		template <FusionPolicy P, typename T, typename O1, typename O2>
		inline __attribute__((always_inline)) 
		void
		_fused_union_block(T* A, const T B, T* C, T* D, 
			const size_t I, const size_t J,
			const size_t M, const size_t N)
		{
			for(size_t i = 0; i < M; i++) {
				for(size_t j = 0; j < N; j++) {
					if constexpr (P == FusionPolicy::UNION_FIRST)
						D[(I+i)*N + (J+j)] = O2{}(B, O1{}(A[(I+i)*N + (J+j)], C[(I+i)*N + (J+j)]));
					else // FUSION_FIRST
						D[(I+i)*N + (J+j)] = O1{}(A[(I+i)*N + (J+j)], O2{}(B, C[(I+i)*N + (J+j)]));
				}
			}
		}

		/**
		 * \brief	Element-wise fused union of two 2D pointer matrices with scalar using cache-blocked traversal.
		 *
		 * Performs blocked element-wise fused operations combining two binary operators on matrices A, B
		 * and scalar C, writing the result into matrix D. The operation order is controlled by FusionPolicy:
		 * UNION_FIRST computes OP2(OP1(A,B), C), while FUSION_FIRST computes OP1(A, OP2(B,C)).
		 * Uses tiling for improved cache locality.
		 *
		 * \tparam P			Fusion policy controlling operation order
		 * \tparam T			Scalar type (e.g., float or double)
		 * \tparam O1			First binary operator (e.g., std::plus<>)
		 * \tparam O2			Second binary operator (e.g., std::multiplies<>)
		 * \tparam block_size	Tile width for cache-friendly blocking (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Matrix B as array of M row pointers, each of size N
		 * \param C		Scalar operand C
		 * \param D		Output matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows in matrices A, B, and D
		 * \param N		Number of columns in matrices A, B, and D
		 *
		 * \note	Matrices must have the same shape: M × N.
		 * \note	Blocked traversal ensures improved cache locality.
		 * \note	Supports all combinations of standard arithmetic operators.
		 * \note	Enables BLAS-style patterns like AXPY variants and scalar multiplication.
		 */
		template <FusionPolicy P, typename T, typename O1, typename O2, const size_t block_size = _block_size, const size_t threads = _threads>	
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union(T** A, T** B, const T C, T** D, const size_t M, const size_t N)
		{
			parallel_for(0, M, block_size,
			[&](size_t i)
			{
				for (size_t j = 0; j < N; j += block_size)
				{ 
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - j);
					_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, m, n);
				}
			}, threads);
		}

		/**
		 * \brief	Element-wise fused union with different scalar positioning using cache-blocked traversal.
		 *
		 * Performs blocked element-wise fused operations with scalar B positioned between matrices A and C.
		 * The operation order is controlled by FusionPolicy: UNION_FIRST computes OP2(B, OP1(A,C)), 
		 * while FUSION_FIRST computes OP1(A, OP2(B,C)). Enables flexible BLAS patterns through
		 * signature-controlled operand positioning.
		 *
		 * \tparam P			Fusion policy controlling operation order
		 * \tparam T			Scalar type (e.g., float or double)
		 * \tparam O1			First binary operator (e.g., std::plus<>)
		 * \tparam O2			Second binary operator (e.g., std::multiplies<>)
		 * \tparam block_size	Tile width for cache-friendly blocking (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Scalar operand B
		 * \param C		Matrix C as array of M row pointers, each of size N
		 * \param D		Output matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows in matrices A, C, and D
		 * \param N		Number of columns in matrices A, C, and D
		 *
		 * \note	Alternative signature enables different BLAS patterns via operand positioning.
		 * \note	Particularly useful for patterns like A + α*C or A*α + C.
		 * \note	Cache-blocked implementation optimizes memory access patterns.
		 */
		template <FusionPolicy P, typename T, typename O1, typename O2, const size_t block_size = _block_size, const size_t threads = _threads>	
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union(T** A, const T B, T** C, T** D, const size_t M, const size_t N)
		{
			parallel_for(0, M, block_size,
			[&](size_t i)
			{
				for (size_t j = 0; j < N; j += block_size)
				{ 
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - j);
					_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, m, n);
				}
			}, threads);
		}

		/**
		 * \brief	Element-wise fused union of flat matrices with scalar using cache-blocked traversal.
		 *
		 * Performs blocked element-wise fused operations on flat 1D arrays A, B with scalar C,
		 * storing the result in array D. Uses row-major layout and cache-optimized blocking.
		 * The operation order is controlled by FusionPolicy for flexible computation patterns.
		 *
		 * \tparam P			Fusion policy controlling operation order
		 * \tparam T			Scalar type (e.g., float or double)
		 * \tparam O1			First binary operator (e.g., std::plus<>)
		 * \tparam O2			Second binary operator (e.g., std::multiplies<>)
		 * \tparam block_size	Tile width for cache-friendly blocking (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution
		 *
		 * \param A		Matrix A stored as a 1D array in row-major order
		 * \param B		Matrix B stored as a 1D array in row-major order
		 * \param C		Scalar operand C
		 * \param D		Output matrix D stored as a 1D array in row-major order
		 * \param M		Number of rows
		 * \param N		Number of columns
		 *
		 * \note	All matrices must have the same logical shape: M × N.
		 * \note	More cache-friendly for large matrices than 2D pointer versions.
		 * \note	Input arrays must contain at least M×N elements each.
		 * \note	Flat array interface preferred for external library interoperability.
		 */
		template <FusionPolicy P, typename T, typename O1, typename O2, const size_t block_size = _block_size, const size_t threads = _threads>	
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union(T* A, T* B, const T C, T* D, const size_t M, const size_t N)
		{
			parallel_for(0, M, block_size,
			[&](size_t i)
			{
				for (size_t j = 0; j < N; j += block_size)
				{ 
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - j);
					_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, m, n);
				}
			}, threads);
		}

		/**
		 * \brief Element-wise fused union of flattened matrices using SIMD intrinsics.
		 *
		 * Performs SIMD-accelerated fused operations on 1D (flattened row-major) matrices A, B, and C,
		 * storing the result in matrix D. SIMD vectorization is selected via the template parameter S
		 * and processes cache-line sized blocks for optimal throughput and memory bandwidth utilization.
		 *
		 * \tparam P		Fusion policy controlling operation order
		 * \tparam T		Scalar type (float or double)
		 * \tparam O1		First binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam O2		Second binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam S		SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam threads	Number of threads used in parallel execution
		 *
		 * \param A		Pointer to flattened matrix A (row-major order)
		 * \param B		Pointer to flattened matrix B (row-major order)
		 * \param C		Pointer to flattened matrix C (row-major order)
		 * \param D		Pointer to flattened matrix D for storing the result
		 * \param M		Number of rows
		 * \param N		Number of columns
		 *
		 * \note All matrices must have the same dimensions (M×N).
		 * \note Asymmetric matrices (M ≠ N) are supported.
		 * \note Strides not aligned with SIMD block sizes are safely handled.
		 * \note Enables comprehensive element-wise ternary operations with SIMD acceleration.
		 */
		template <FusionPolicy P, typename T, typename O1, typename O2, const size_t block_size = _block_size, const size_t threads = _threads>	
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union(T* A, const T B, T* C, T* D, const size_t M, const size_t N)
		{
			parallel_for(0, M, block_size,
			[&](size_t i)
			{
				for (size_t j = 0; j < N; j += block_size)
				{ 
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - j);
					_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, m, n);
				}
			}, threads);
		}

		template <FusionPolicy P, typename T, typename O1, typename O2>
		void _fused_union_block_sse(T* A, T* B, const T C, T* D);

		template <FusionPolicy P, typename T, typename O1, typename O2>
		void _fused_union_block_sse(T* A, const T B, T* C, T* D);

		template <FusionPolicy P, typename T, typename O1, typename O2>
		void _fused_union_block_avx(T* A, T* B, const T C, T* D);

		template <FusionPolicy P, typename T, typename O1, typename O2>
		void _fused_union_block_avx(T* A, const T B, T* C, T* D);

		template <FusionPolicy P, typename T, typename O1, typename O2>
		void _fused_union_block_avx512(T* A, T* B, const T C, T* D);

		template <FusionPolicy P, typename T, typename O1, typename O2>
		void _fused_union_block_avx512(T* A, const T B, T* C, T* D);

		/**
		 * \brief Element-wise fused union using SIMD intrinsics for flattened matrices with scalar.
		 *
		 * Performs SIMD-accelerated fused operations on 1D (flattened row-major) matrices A, B
		 * with scalar C, storing the result in matrix D. SIMD vectorization is selected via the
		 * template parameter S and processes cache-line sized blocks for optimal throughput.
		 *
		 * \tparam P		Fusion policy controlling operation order
		 * \tparam T		Scalar type (float or double)
		 * \tparam O1		First binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam O2		Second binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam S		SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam threads	Number of threads used in parallel execution
		 *
		 * \param A		Pointer to flattened matrix A (row-major order)
		 * \param B		Pointer to flattened matrix B (row-major order)
		 * \param C		Scalar operand C
		 * \param D		Pointer to flattened matrix D for storing the result
		 * \param M		Number of rows
		 * \param N		Number of columns
		 *
		 * \note Signature: (A, B, scalar, D) enables specific BLAS patterns.
		 * \note Remainder elements not divisible by SIMD width are handled with scalar fallback.
		 * \note Cache-line alignment is preferred but not required for correctness.
		 */		
		 template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union_simd(T* A, T* B, const T C, T* D, const size_t M, const size_t N)
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
					_fused_union_block_sse<P, T, O1, O2>(A + offset, B + offset, C, D + offset);
				if constexpr (S == AVX)
					_fused_union_block_avx<P, T, O1, O2>(A + offset, B + offset, C, D + offset);
				if constexpr (S == AVX512)
					_fused_union_block_avx512<P, T, O1, O2>(A + offset, B + offset, C, D + offset);
			});

			// remainder
			if (const size_t remainder_start = full_blocks * kernel_stride; remainder_start < total_elements) 
			{
				const size_t remainder_size = total_elements - remainder_start;
				_fused_union_block<P, T, O1, O2>(A + remainder_start, B + remainder_start, C, D + remainder_start, 0, 0, remainder_size, 1);
			}
		}

		/**
		 * \brief Element-wise fused union using SIMD intrinsics with alternative scalar positioning.
		 *
		 * Performs SIMD-accelerated fused operations on matrices A, C with scalar B positioned
		 * between them. This signature enables different BLAS patterns through operand positioning
		 * while maintaining SIMD optimization for high performance.
		 *
		 * \tparam P		Fusion policy controlling operation order
		 * \tparam T		Scalar type (float or double)
		 * \tparam O1		First binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam O2		Second binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam S		SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam threads	Number of threads used in parallel execution
		 *
		 * \param A		Pointer to flattened matrix A (row-major order)
		 * \param B		Scalar operand B
		 * \param C		Pointer to flattened matrix C (row-major order)
		 * \param D		Pointer to flattened matrix D for storing the result
		 * \param M		Number of rows
		 * \param N		Number of columns
		 *
		 * \note Signature: (A, scalar, C, D) enables alternative BLAS patterns.
		 * \note Scalar broadcast is optimized within SIMD kernels for efficient computation.
		 * \note Handles non-aligned strides gracefully with scalar remainder processing.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union_simd(T* A, const T B, T* C, T* D, const size_t M, const size_t N)
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
					_fused_union_block_sse<P, T, O1, O2>(A + offset, B, C + offset, D + offset);
				if constexpr (S == AVX)
					_fused_union_block_avx<P, T, O1, O2>(A + offset, B, C + offset, D + offset);
				if constexpr (S == AVX512)
					_fused_union_block_avx512<P, T, O1, O2>(A + offset, B, C + offset, D + offset);
			});

			// remainder
			if (const size_t remainder_start = full_blocks * kernel_stride; remainder_start < total_elements) 
			{
				const size_t remainder_size = total_elements - remainder_start;
				_fused_union_block<P, T, O1, O2>(A + remainder_start, B, C + remainder_start, D + remainder_start, 0, 0, remainder_size, 1);
			}
		}

		/**
		 * \brief Element-wise fused union using SIMD intrinsics for 2D pointer matrices with scalar.
		 *
		 * Performs SIMD-accelerated fused operations on 2D matrix A, B with scalar C, storing
		 * the result in matrix D. Internally flattens matrices to invoke SIMD routines while
		 * maintaining the 2D pointer interface for compatibility.
		 *
		 * \tparam P		Fusion policy controlling operation order
		 * \tparam T		Scalar type (float or double)
		 * \tparam O1		First binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam O2		Second binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam S		SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam threads	Number of threads used in parallel execution
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Matrix B as array of M row pointers, each of size N
		 * \param C		Scalar operand C
		 * \param D		Output matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows
		 * \param N		Number of columns
		 *
		 * \note Matrices are validated for contiguity and proper alignment.
		 * \note 2D pointer interface maintained while leveraging SIMD optimizations.
		 * \note Signature: (A, B, scalar, D) pattern for specific BLAS operations.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union_simd(T** A, T** B, const T C, T** D, const size_t M, const size_t N)
		{
			T* A0 = &A[0][0];
			T* B0 = &B[0][0];
			T* D0 = &D[0][0];

			_fused_union_simd<P, T, O1, O2, S, threads>(A0, B0, C, D0, M, N);
		}

		/**
		 * \brief Element-wise fused union using SIMD intrinsics for 2D pointer matrices with alternative scalar positioning.
		 *
		 * Performs SIMD-accelerated fused operations on 2D matrices A, C with scalar B positioned
		 * between them. Maintains 2D pointer interface while enabling signature-controlled BLAS
		 * patterns through strategic operand positioning.
		 *
		 * \tparam P		Fusion policy controlling operation order
		 * \tparam T		Scalar type (float or double)
		 * \tparam O1		First binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam O2		Second binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam S		SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam threads	Number of threads used in parallel execution
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Scalar operand B
		 * \param C		Matrix C as array of M row pointers, each of size N
		 * \param D		Output matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows
		 * \param N		Number of columns
		 *
		 * \note Alternative signature enables different BLAS patterns via operand positioning.
		 * \note 2D interface with SIMD acceleration for optimal performance and usability.
		 * \note Signature: (A, scalar, C, D) pattern for flexible operation control.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union_simd(T** A, const T B, T** C, T** D, const size_t M, const size_t N)
		{
			T* A0 = &A[0][0];
			T* C0 = &C[0][0];
			T* D0 = &D[0][0];

			_fused_union_simd<P, T, O1, O2, S, threads>(A0, B, C0, D0, M, N);
		}

		/**
		 * \brief Perform optimized element-wise fused union with signature (A, B, scalar, D) using SIMD and blocking algorithms.
		 *
		 * This is the main public interface for scalar-matrix fused union operations using flat 1D arrays
		 * in row-major layout. It automatically selects between SIMD-optimized and standard blocked
		 * implementations based on the specified SIMD instruction set. The operation order is controlled
		 * by FusionPolicy template parameter.
		 *
		 * The fused operation computes either OP2(OP1(A[i*N + j], B[i*N + j]), C) for UNION_FIRST
		 * or OP1(A[i*N + j], OP2(B[i*N + j], C)) for FUSION_FIRST, enabling efficient BLAS-style
		 * patterns like AXPY variants and compound scalar operations.
		 *
		 * \tparam P			Fusion policy controlling operation precedence
		 * \tparam T			Element type of the matrices (e.g., float, double)
		 * \tparam O1			First binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam O2			Second binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE)
		 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution (default: _threads)
		 *
		 * \param A		Matrix A stored as flat array of size M×N in row-major layout
		 * \param B		Matrix B stored as flat array of size M×N in row-major layout
		 * \param C		Scalar operand C
		 * \param D		Result matrix D stored as flat array of size M×N in row-major layout
		 * \param M		Number of rows in matrices A, B, and D
		 * \param N		Number of columns in matrices A, B, and D
		 *
		 * \note All arrays must be allocated as contiguous memory blocks of appropriate size.
		 * \note When S=NONE, the function uses standard blocked operations without SIMD.
		 * \note SIMD implementations automatically handle non-aligned dimensions with scalar fallback.
		 * \note This flat array interface is often preferred for interoperability with other
		 *       libraries or when working with pre-allocated buffers.
		 * \note Enables comprehensive BLAS patterns through fusion policy and signature control.
		 *
		 * \throws std::invalid_argument if any matrix pointer is null.
		 * \throws std::runtime_error if memory layout validation fails.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S = detect_simd(),
			const size_t block_size = _block_size, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		fused_union(T* A, T* B, const T C, T* D, const size_t M, const size_t N)
		{
			right<T>("fused_union:", 
				std::make_tuple(A, M, N),
				std::make_tuple(B, M, N),
				std::make_tuple(D, M, N));
			
			if constexpr (S == SIMD::NONE)
				_fused_union<P, T, O1, O2, block_size, threads>(A, B, C, D, M, N);
			else
				_fused_union_simd<P, T, O1, O2, S, threads>(A, B, C, D, M, N);
		}

		/**
		 * \brief Perform optimized element-wise fused union with signature (A, B, scalar, D) using SIMD and blocking algorithms.
		 *
		 * This is the main public interface for scalar-matrix fused union operations using 2D pointer arrays
		 * with row-major layout. It automatically selects between SIMD-optimized and standard blocked
		 * implementations based on the specified SIMD instruction set. The operation order is controlled
		 * by FusionPolicy template parameter.
		 *
		 * The fused operation computes either OP2(OP1(A[i][j], B[i][j]), C) for UNION_FIRST
		 * or OP1(A[i][j], OP2(B[i][j], C)) for FUSION_FIRST, enabling efficient BLAS-style
		 * patterns like compound operations and scalar broadcasting.
		 *
		 * \tparam P			Fusion policy controlling operation precedence
		 * \tparam T			Element type of the matrices (e.g., float, double)
		 * \tparam O1			First binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam O2			Second binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE)
		 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution (default: _threads)
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Matrix B as array of M row pointers, each of size N
		 * \param C		Scalar operand C
		 * \param D		Result matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows in matrices A, B, and D
		 * \param N		Number of columns in matrices A, B, and D
		 *
		 * \note Matrices must be allocated as contiguous memory blocks accessible
		 *       through the 2D pointer interface. Submatrix views are not supported.
		 * \note When S=NONE, the function uses standard blocked operations without SIMD.
		 * \note SIMD implementations automatically handle non-aligned dimensions with scalar fallback.
		 * \note Signature enables specific BLAS patterns: (A op1 B) op2 C or A op1 (B op2 C).
		 *
		 * \throws std::invalid_argument if any matrix pointer is null.
		 * \throws std::runtime_error if memory layout validation fails.
		 */

		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S = detect_simd(),
			const size_t block_size = _block_size, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		fused_union(T** A, T** B, const T C, T** D, const size_t M, const size_t N)
		{
			right<T>("fused_union:", 
				std::make_tuple(A, M, N),
				std::make_tuple(B, M, N),
				std::make_tuple(D, M, N));
			
			if constexpr (S == SIMD::NONE)
				_fused_union<P, T, O1, O2, block_size, threads>(A, B, C, D, M, N);
			else
				_fused_union_simd<P, T, O1, O2, S, threads>(A, B, C, D, M, N);
		}

		/**
		 * \brief Perform optimized element-wise fused union with signature (A, scalar, B, D) using SIMD and blocking algorithms.
		 *
		 * This is the main public interface for scalar-matrix fused union operations with alternative
		 * operand positioning using 2D pointer arrays. The scalar operand is positioned between the
		 * two matrices, enabling different BLAS patterns through signature-controlled operation ordering.
		 *
		 * The fused operation computes either OP2(B, OP1(A[i][j], C[i][j])) for UNION_FIRST
		 * or OP1(A[i][j], OP2(B, C[i][j])) for FUSION_FIRST, unlocking alternative BLAS-style
		 * patterns like A + α*C or A*α + C through strategic operand positioning.
		 *
		 * \tparam P			Fusion policy controlling operation precedence
		 * \tparam T			Element type of the matrices (e.g., float, double)
		 * \tparam O1			First binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam O2			Second binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE)
		 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution (default: _threads)
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Scalar operand B
		 * \param C		Matrix C as array of M row pointers, each of size N
		 * \param D		Result matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows in matrices A, C, and D
		 * \param N		Number of columns in matrices A, C, and D
		 *
		 * \note Alternative signature enables different BLAS patterns via strategic operand positioning.
		 * \note Matrices must be allocated as contiguous memory blocks accessible through 2D interface.
		 * \note Particularly effective for patterns requiring scalar multiplication or division with matrices.
		 * \note SIMD optimizations apply automatically based on template parameter selection.
		 *
		 * \throws std::invalid_argument if any matrix pointer is null.
		 * \throws std::runtime_error if memory layout validation fails.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S = detect_simd(),
			const size_t block_size = _block_size, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		fused_union(T** A, const T B, T** C, T** D, const size_t M, const size_t N)
		{
			right<T>("fused_union:", 
				std::make_tuple(A, M, N),
				std::make_tuple(C, M, N),
				std::make_tuple(D, M, N));

			if constexpr (S == SIMD::NONE)
				_fused_union<P, T, O1, O2, block_size, threads>(A, B, C, D, M, N);
			else
				_fused_union_simd<P, T, O1, O2, S, threads>(A, B, C, D, M, N);
		}

		/**
		 * \brief Perform optimized element-wise fused union with signature (A, scalar, B, D) using SIMD and blocking algorithms.
		 *
		 * This is the main public interface for scalar-matrix fused union operations with alternative
		 * operand positioning using flat 1D arrays. The scalar operand is positioned between the
		 * two matrices, enabling different BLAS patterns through signature-controlled operation ordering.
		 *
		 * The fused operation computes either OP2(B, OP1(A[i*N + j], C[i*N + j])) for UNION_FIRST
		 * or OP1(A[i*N + j], OP2(B, C[i*N + j])) for FUSION_FIRST, enabling flexible BLAS-style
		 * operations through strategic scalar positioning and policy control.
		 *
		 * \tparam P			Fusion policy controlling operation precedence
		 * \tparam T			Element type of the matrices (e.g., float, double)
		 * \tparam O1			First binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam O2			Second binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE)
		 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution (default: _threads)
		 *
		 * \param A		Matrix A stored as flat array of size M×N in row-major layout
		 * \param B		Scalar operand B
		 * \param C		Matrix C stored as flat array of size M×N in row-major layout
		 * \param D		Result matrix D stored as flat array of size M×N in row-major layout
		 * \param M		Number of rows in matrices A, C, and D
		 * \param N		Number of columns in matrices A, C, and D
		 *
		 * \note Flat array interface with alternative operand ordering for maximum flexibility.
		 * \note Preferred interface for external library integration and pre-allocated buffers.
		 * \note Enables comprehensive BLAS coverage through signature control and fusion policies.
		 * \note SIMD acceleration automatically applied based on hardware capabilities and template selection.
		 *
		 * \throws std::invalid_argument if any matrix pointer is null.
		 * \throws std::runtime_error if memory layout validation fails.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S = detect_simd(),
			const size_t block_size = _block_size, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		fused_union(T* A, const T B, T* C, T* D, const size_t M, const size_t N)
		{
			right<T>("fused_union:", 
				std::make_tuple(A, M, N),
				std::make_tuple(C, M, N),
				std::make_tuple(D, M, N));

			if constexpr (S == SIMD::NONE)
				_fused_union<P, T, O1, O2, block_size, threads>(A, B, C, D, M, N);
			else
				_fused_union_simd<P, T, O1, O2, S, threads>(A, B, C, D, M, N);
		}

	} // namespace scalar


	namespace matrix 
	{

		/** 
		 * \brief kernel for matrix fused_union_block. 
		 * Low level function not intended for the public API.
		 * */
		template <FusionPolicy P, typename T, typename O1, typename O2>
		inline __attribute__((always_inline)) 
		void
		_fused_union_block(T** A, T** B, T** C, T** D,
			const size_t I, const size_t J,
			const size_t M, const size_t N)
		{
			for(size_t i = 0; i < M; i++)
				for(size_t j = 0; j < N; j++) 
				{
					if constexpr (P == FusionPolicy::UNION_FIRST) 
						D[I+i][J+j] = O2{}(O1{}(A[I+i][J+j], B[I+i][J+j]), C[I+i][J+j]);
					else // FUSION_FIRST
						D[I+i][J+j] = O1{}(A[I+i][J+j], O2{}(B[I+i][J+j], C[I+i][J+j]));
				}
		}

		/** 
		 * \brief kernel for matrix fused_union_block_flat. 
		 * Low level function not intended for the public API.
		 * */
		template <FusionPolicy P, typename T, typename O1, typename O2>
		inline __attribute__((always_inline)) 
		void
		_fused_union_block(T* A, T* B, T* C, T* D,
			const size_t I, const size_t J,
			const size_t M, const size_t N)
		{
			for(size_t i = 0; i < M; i++) 
				for(size_t j = 0; j < N; j++) 
				{
					if constexpr (P == FusionPolicy::UNION_FIRST)
						D[(I+i)*N + (J+j)] = O2{}(O1{}(A[(I+i)*N + (J+j)], B[(I+i)*N + (J+j)]), C[(I+i)*N + (J+j)]);
					else // FUSION_FIRST
						D[(I+i)*N + (J+j)] = O1{}(A[(I+i)*N + (J+j)], O2{}(B[(I+i)*N + (J+j)], C[(I+i)*N + (J+j)]));
				}
		}

		/**
		 * \brief	Element-wise fused union of three 2D pointer matrices using cache-blocked traversal.
		 *
		 * Performs blocked element-wise fused operations on three matrices A, B, and C,
		 * writing the result into matrix D. The operation order is controlled by FusionPolicy:
		 * UNION_FIRST computes OP2(OP1(A,B), C), while FUSION_FIRST computes OP1(A, OP2(B,C)).
		 * Uses cache-optimized tiling for improved memory access patterns.
		 *
		 * \tparam P			Fusion policy controlling operation order
		 * \tparam T			Scalar type (e.g., float or double)
		 * \tparam O1			First binary operator (e.g., std::plus<>)
		 * \tparam O2			Second binary operator (e.g., std::multiplies<>)
		 * \tparam block_size	Tile width for cache-friendly blocking (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Matrix B as array of M row pointers, each of size N
		 * \param C		Matrix C as array of M row pointers, each of size N
		 * \param D		Output matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows in matrices A, B, C, and D
		 * \param N		Number of columns in matrices A, B, C, and D
		 *
		 * \note	All matrices must have the same shape: M × N.
		 * \note	Blocked traversal ensures improved cache locality for large matrices.
		 * \note	Enables efficient ternary operations with configurable precedence.
		 * \note	Supports comprehensive BLAS-style element-wise operations.
		 */
		template <FusionPolicy P, typename T, typename O1, typename O2, const size_t block_size = _block_size, const size_t threads = _threads>	
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union(T** A, T** B, T** C, T** D, const size_t M, const size_t N)
		{
			parallel_for(0, M, block_size,
			[&](size_t i)
			{
				for (size_t j = 0; j < N; j += block_size)
				{ 
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - j);
					_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, m, n);
				}
			}, threads);
		}

		/**
		 * \brief	Element-wise fused union of three flat matrices using cache-blocked traversal.
		 *
		 * Performs blocked element-wise fused operations on flat 1D arrays A, B, and C,
		 * storing the result in array D. Uses row-major layout with cache-optimized blocking.
		 * The operation order is controlled by FusionPolicy for flexible computation patterns.
		 * \tparam P			Fusion policy controlling operation order
		 * \tparam T			Scalar type (e.g., float or double)
		 * \tparam O1			First binary operator (e.g., std::plus<>)
		 * \tparam O2			Second binary operator (e.g., std::multiplies<>)
		 * \tparam block_size	Tile width for cache-friendly blocking (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Matrix B as array of M row pointers, each of size N
		 * \param C		Matrix C as array of M row pointers, each of size N
		 * \param D		Output matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows in matrices A, B, C, and D
		 * \param N		Number of columns in matrices A, B, C, and D
		 *
		 * \note	All matrices must have the same shape: M × N.
		 * \note	Blocked traversal ensures improved cache locality for large matrices.
		 * \note	Enables efficient ternary operations with configurable precedence.
		 * \note	Supports comprehensive BLAS-style element-wise operations.
		 */
		template <FusionPolicy P, typename T, typename O1, typename O2, const size_t block_size = _block_size, const size_t threads = _threads>	
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union(T* A, T* B, T* C, T* D, const size_t M, const size_t N)
		{
			parallel_for(0, M, block_size,
			[&](size_t i)
			{
				for (size_t j = 0; j < N; j += block_size)
				{ 
					size_t m = std::min(block_size, M - i);
					size_t n = std::min(block_size, N - j);
					_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, m, n);
				}
			}, threads);
		}

		// SIMD kernel declarations for matrix operations
		template <FusionPolicy P, typename T, typename O1, typename O2>
		void _fused_union_block_sse(T* A, T* B, T* C, T* D);

		template <FusionPolicy P, typename T, typename O1, typename O2>
		void _fused_union_block_avx(T* A, T* B, T* C, T* D);

		template <FusionPolicy P, typename T, typename O1, typename O2>
		void _fused_union_block_avx512(T* A, T* B, T* C, T* D);

			/**
		 * \brief Element-wise fused union of flattened matrices using SIMD intrinsics.
		 *
		 * Performs SIMD-accelerated fused operations on 1D (flattened row-major) matrices A, B, and C,
		 * storing the result in matrix D. SIMD vectorization is selected via the template parameter S
		 * and processes cache-line sized blocks for optimal throughput and memory bandwidth utilization.
		 *
		 * \tparam P		Fusion policy controlling operation order
		 * \tparam T		Scalar type (float or double)
		 * \tparam O1		First binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam O2		Second binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam S		SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam threads	Number of threads used in parallel execution
		 *
		 * \param A		Pointer to flattened matrix A (row-major order)
		 * \param B		Pointer to flattened matrix B (row-major order)
		 * \param C		Pointer to flattened matrix C (row-major order)
		 * \param D		Pointer to flattened matrix D for storing the result
		 * \param M		Number of rows
		 * \param N		Number of columns
		 *
		 * \note All matrices must have the same dimensions (M×N).
		 * \note Asymmetric matrices (M ≠ N) are supported.
		 * \note Strides not aligned with SIMD block sizes are safely handled.
		 * \note Enables comprehensive element-wise ternary operations with SIMD acceleration.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union_simd(T* A, T* B, T* C, T* D, const size_t M, const size_t N)
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
					_fused_union_block_sse<P, T, O1, O2>(A + offset, B + offset, C + offset, D + offset);
				if constexpr (S == AVX)
					_fused_union_block_avx<P, T, O1, O2>(A + offset, B + offset, C + offset, D + offset);
				if constexpr (S == AVX512)
					_fused_union_block_avx512<P, T, O1, O2>(A + offset, B + offset, C + offset, D + offset);
			});

			// Remainder
			if (const size_t remainder_start = full_blocks * kernel_stride; remainder_start < total_elements)
			{
				const size_t remainder_size = total_elements - remainder_start;
				_fused_union_block<P, T, O1, O2>(A + remainder_start, B + remainder_start, C + remainder_start, D + remainder_start, 0, 0, remainder_size, 1);
			}
		}

		/**
		 * \brief Element-wise fused union of matrices using SIMD intrinsics.
		 *
		 * Performs SIMD-accelerated fused operations on 2D matrices A, B, and C, storing
		 * the result in matrix D. SIMD vectorization is selected at compile time via the
		 * template parameter S. Internally flattens matrices to invoke SIMD routines while
		 * maintaining the 2D pointer interface.
		 *
		 * \tparam P		Fusion policy controlling operation order
		 * \tparam T		Scalar type (float or double)
		 * \tparam O1		First binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam O2		Second binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam S		SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam threads	Number of threads used in parallel execution
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Matrix B as array of M row pointers, each of size N
		 * \param C		Matrix C as array of M row pointers, each of size N
		 * \param D		Output matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows
		 * \param N		Number of columns
		 *
		 * \note Matrices must have the same dimensions (M×N).
		 * \note Asymmetric matrices (M ≠ N) are supported.
		 * \note Strides not aligned with SIMD block sizes are safely handled.
		 * \note 2D interface maintained while leveraging SIMD optimizations internally.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		_fused_union_simd(T** A, T** B, T** C, T** D, const size_t M, const size_t N)
		{
			T* A0 = &A[0][0];
			T* B0 = &B[0][0];
			T* C0 = &C[0][0];
			T* D0 = &D[0][0];

			_fused_union_simd<P, T, O1, O2, S, threads>(A0, B0, C0, D0, M, N);
		}

		/**
		 * \brief Perform optimized element-wise fused union of three matrices using SIMD and blocking algorithms.
		 *
		 * This is the main public interface for matrix-matrix fused union operations using 2D pointer arrays
		 * with row-major layout. It automatically selects between SIMD-optimized and standard blocked
		 * implementations based on the specified SIMD instruction set. The operation order is controlled
		 * by FusionPolicy template parameter.
		 *
		 * The fused operation computes either OP2(OP1(A[i][j], B[i][j]), C[i][j]) for UNION_FIRST
		 * or OP1(A[i][j], OP2(B[i][j], C[i][j])) for FUSION_FIRST, enabling efficient ternary
		 * element-wise operations like Hadamard fusion and compound matrix arithmetic.
		 *
		 * \tparam P			Fusion policy controlling operation precedence
		 * \tparam T			Element type of the matrices (e.g., float, double)
		 * \tparam O1			First binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam O2			Second binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE)
		 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution (default: _threads)
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Matrix B as array of M row pointers, each of size N
		 * \param C		Matrix C as array of M row pointers, each of size N
		 * \param D		Result matrix D as array of M row pointers, each of size N
		 * \param M		Number of rows in matrices A, B, C, and D
		 * \param N		Number of columns in matrices A, B, C, and D
		 *
		 * \note All matrices must be allocated as contiguous memory blocks accessible
		 *       through the 2D pointer interface. Submatrix views are not supported.
		 * \note Non-square matrices and asymmetric dimensions (M ≠ N) are fully supported.
		 * \note When S=NONE, the function uses standard blocked operations without SIMD.
		 * \note SIMD implementations automatically handle non-aligned dimensions with scalar fallback.
		 * \note Enables comprehensive BLAS-style ternary operations with configurable precedence.
		 *
		 * \throws std::invalid_argument if any matrix pointer is null.
		 * \throws std::runtime_error if memory layout validation fails.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S = detect_simd(), 
			const size_t block_size = _block_size, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		fused_union(T** A, T** B, T** C, T** D, const size_t M, const size_t N)
		{
			right<T>("fused_union:", 
				std::make_tuple(A, M, N),
				std::make_tuple(B, M, N),
				std::make_tuple(C, M, N),
				std::make_tuple(D, M, N));

			if constexpr (S == SIMD::NONE)
				_fused_union<P, T, O1, O2, block_size, threads>(A, B, C, D, M, N);
			else
				_fused_union_simd<P, T, O1, O2, S, threads>(A, B, C, D, M, N);
		}

		/**
		 * \brief Perform optimized element-wise fused union of three matrices using SIMD and blocking algorithms.
		 *
		 * This is the main public interface for matrix-matrix fused union operations using flat 1D arrays
		 * in row-major layout. It automatically selects between SIMD-optimized and standard blocked
		 * implementations based on the specified SIMD instruction set. The operation order is controlled
		 * by FusionPolicy template parameter.
		 *
		 * The fused operation computes either OP2(OP1(A[i*N + j], B[i*N + j]), C[i*N + j]) for UNION_FIRST
		 * or OP1(A[i*N + j], OP2(B[i*N + j], C[i*N + j])) for FUSION_FIRST, enabling efficient ternary
		 * element-wise operations with optimal memory access patterns.
		 *
		 * \tparam P			Fusion policy controlling operation precedence
		 * \tparam T			Element type of the matrices (e.g., float, double)
		 * \tparam O1			First binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam O2			Second binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE)
		 * \tparam block_size	Size of square blocks for cache optimization (default: _block_size)
		 * \tparam threads		Number of threads for parallel execution (default: _threads)
		 *
		 * \param A		Matrix A stored as flat array of size M×N in row-major layout
		 * \param B		Matrix B stored as flat array of size M×N in row-major layout
		 * \param C		Matrix C stored as flat array of size M×N in row-major layout
		 * \param D		Result matrix D stored as flat array of size M×N in row-major layout
		 * \param M		Number of rows in matrices A, B, C, and D
		 * \param N		Number of columns in matrices A, B, C, and D
		 *
		 * \note All arrays must be allocated as contiguous memory blocks of appropriate size.
		 * \note Non-square matrices and asymmetric dimensions (M ≠ N) are fully supported.
		 * \note When S=NONE, the function uses standard blocked operations without SIMD.
		 * \note SIMD implementations automatically handle non-aligned dimensions with scalar fallback.
		 * \note This flat array interface is often preferred for interoperability with other
		 *       libraries or when working with pre-allocated buffers.
		 * \note Unlocks comprehensive element-wise ternary operations with configurable precedence.
		 *
		 * \throws std::invalid_argument if any matrix pointer is null.
		 * \throws std::runtime_error if memory layout validation fails.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, SIMD S = detect_simd(), 
			const size_t block_size = _block_size, const size_t threads = _threads>
		requires 
		(
			(std::same_as<O1, std::plus<>> ||
			std::same_as<O1, std::minus<>> ||
			std::same_as<O1, std::multiplies<>> ||
			std::same_as<O1, std::divides<>>) &&
			(std::same_as<O2, std::plus<>> ||
			std::same_as<O2, std::minus<>> ||
			std::same_as<O2, std::multiplies<>> ||
			std::same_as<O2, std::divides<>>)
		)
		inline __attribute__((always_inline))
		void
		fused_union(T* A, T* B, T* C, T* D, const size_t M, const size_t N)
		{
			right<T>("fused_union:", 
				std::make_tuple(A, M, N),
				std::make_tuple(B, M, N),
				std::make_tuple(C, M, N),
				std::make_tuple(D, M, N));

			if constexpr (S == SIMD::NONE)
				_fused_union<P, T, O1, O2, block_size, threads>(A, B, C, D, M, N);
			else
				_fused_union_simd<P, T, O1, O2, S, threads>(A, B, C, D, M, N);
		}

	} // namespace matrix

} //namespace damm
#endif //__FUSED_UNION_H__