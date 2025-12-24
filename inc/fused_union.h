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
#include <simd.h>
#include <damm_kernels.h>
#include <omp.h>


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
 * BLAS patterns like A + Î±*B, (A+B)*Î±, and Î±/(A+B) through strategic operand positioning.
 * The matrix namespace supports full element-wise fusion of three matrices with configurable
 * operation precedence.
 *
 * \note
 * All operations support asymmetric matrices (M â‰  N) and provide SIMD acceleration.
 * Vectorization (SSE/AVX/AVX512) registers are selected based on
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
	 * UNION_FIRST:  (A + B) âŠ™ D, (A - B) + D, (A âŠ™ B) / D
	 * FUSION_FIRST: A + (B âŠ™ D), A - (B âŠ™ D), A âŠ™ (B / D)
	 *
	 * KEY BLAS PATTERNS:
	 * AXPY family:     A + D*B, A - D*B (via signature control)
	 * SCAL variants:   D*(A+B), D*(A-B), (A+B)*D, (A-B)*D  
	 * Hadamard fusion: A + BâŠ™D, A âŠ™ (B+D)
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
		 * \brief	Element-wise fused union of two 2D pointer matrices with scalar using cache-blocked traversal.
		 *
		 * Performs blocked element-wise fused operations combining two binary operators on matrices A, B
		 * and scalar C, writing the result into matrix D. The operation order is controlled by FusionPolicy:
		 * UNION_FIRST computes OP2(OP1(A,B), C), while FUSION_FIRST computes OP1(A, OP2(B,C)).
		 * Uses cache-aware blocking determined by the kernel policy.
		 *
		 * \tparam P	Fusion policy controlling operation order
		 * \tparam T	Scalar type (e.g., float or double)
		 * \tparam O1	First binary operator (e.g., std::plus<>)
		 * \tparam O2	Second binary operator (e.g., std::multiplies<>)
		 * \tparam K	Kernel policy defining cache blocking sizes
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
		template <FusionPolicy P, typename T, typename O1, typename O2, template<typename, typename> class K>	
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
			using kernel = K<T, NONE>;
			using blocking = typename kernel::blocking;
			
			constexpr size_t l2_block = blocking::l2_block;
			constexpr size_t l3_block = blocking::l3_block;
			
			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < M; i += l2_block)
			{
				for (size_t j = 0; j < N; j += l3_block)
				{ 
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l3_block, N - j);
					_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, m, n);
				}
			}
		}

		/**
		 * \brief	Element-wise fused union with different scalar positioning using cache-blocked traversal.
		 *
		 * Performs blocked element-wise fused operations with scalar B positioned between matrices A and C.
		 * The operation order is controlled by FusionPolicy: UNION_FIRST computes OP2(B, OP1(A,C)), 
		 * while FUSION_FIRST computes OP1(A, OP2(B,C)). Enables flexible BLAS patterns through
		 * signature-controlled operand positioning.
		 *
		 * \tparam P	Fusion policy controlling operation order
		 * \tparam T	Scalar type (e.g., float or double)
		 * \tparam O1	First binary operator (e.g., std::plus<>)
		 * \tparam O2	Second binary operator (e.g., std::multiplies<>)
		 * \tparam K	Kernel policy defining cache blocking sizes
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
		template <FusionPolicy P, typename T, typename O1, typename O2, template<typename, typename> class K>	
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
			using kernel = K<T, NONE>;
			using blocking = typename kernel::blocking;
			
			constexpr size_t l2_block = blocking::l2_block;
			constexpr size_t l3_block = blocking::l3_block;
			
			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < M; i += l2_block)
			{
				for (size_t j = 0; j < N; j += l3_block)
				{ 
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l3_block, N - j);
					_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, m, n);
				}
			}
		}


		/**
		 * \brief SIMD block kernel for scalar fused union (A, B, scalar, D) with FMA optimization
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, typename S, 
			template<typename, typename> class K>
		inline __attribute__((always_inline))
		void _fused_union_simd_block(T** A, T** B, const T C, T** D, 
			const size_t row, const size_t col)
		{
			using kernel = K<T, S>;
			using blocking = typename kernel::blocking;
			using register_t = typename S::template register_t<T>;
			
			constexpr size_t M = kernel::row_registers;
			constexpr size_t N = kernel::col_registers;

			// Load A and B into registers
			alignas(S::bytes) register_t a[M][N];
			alignas(S::bytes) register_t b[M][N];
			alignas(S::bytes) register_t d[M][N];
			register_t* a_ptrs[M];
			register_t* b_ptrs[M];
			register_t* d_ptrs[M];
			
			for (size_t i = 0; i < M; ++i)
			{
				a_ptrs[i] = a[i];
				b_ptrs[i] = b[i];
				d_ptrs[i] = d[i];
			}
			
			load<T, S, K>(A, a_ptrs, row, col);
			load<T, S, K>(B, b_ptrs, row, col);

			// Broadcast scalar C to SIMD register
			register_t c_vec = _set1<T, S>(C);

			// Detect and apply fused operation patterns
			if constexpr (P == FusionPolicy::UNION_FIRST && 
						  std::same_as<O1, std::multiplies<>> && 
						  std::same_as<O2, std::plus<>>)
			{
				// D = (A * B) + C = fmadd(A, B, C)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						d[i][j] = _fmadd<T, S>(a[i][j], b[i][j], c_vec);
					});
				});
			}
			else if constexpr (P == FusionPolicy::UNION_FIRST && 
							   std::same_as<O1, std::multiplies<>> && 
							   std::same_as<O2, std::minus<>>)
			{
				// D = (A * B) - C = fmsub(A, B, C)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						d[i][j] = _fmsub<T, S>(a[i][j], b[i][j], c_vec);
					});
				});
			}
			else if constexpr (P == FusionPolicy::FUSION_FIRST && 
							   std::same_as<O2, std::multiplies<>> && 
							   std::same_as<O1, std::plus<>>)
			{
				// D = A + (B * C) = fmadd(B, C, A)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						d[i][j] = _fmadd<T, S>(b[i][j], c_vec, a[i][j]);
					});
				});
			}
			else if constexpr (P == FusionPolicy::FUSION_FIRST && 
							   std::same_as<O2, std::multiplies<>> && 
							   std::same_as<O1, std::minus<>>)
			{
				// D = A - (B * C) = fnmadd(B, C, A)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						d[i][j] = _fnmadd<T, S>(b[i][j], c_vec, a[i][j]);
					});
				});
			}
			else if constexpr (P == FusionPolicy::UNION_FIRST && 
							   std::same_as<O1, std::plus<>> && 
							   std::same_as<O2, std::multiplies<>>)
			{
				// D = (A + B) * C = fmadd(A+B, C, 0)
				// temp = A + B, then D = temp * C
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						register_t temp = _add<T, S>(a[i][j], b[i][j]);
						d[i][j] = _mul<T, S>(temp, c_vec);
					});
				});
			}
			else if constexpr (P == FusionPolicy::UNION_FIRST && 
							   std::same_as<O1, std::minus<>> && 
							   std::same_as<O2, std::multiplies<>>)
			{
				// D = (A - B) * C
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						register_t temp = _sub<T, S>(a[i][j], b[i][j]);
						d[i][j] = _mul<T, S>(temp, c_vec);
					});
				});
			}
			else if constexpr (P == FusionPolicy::FUSION_FIRST && 
							   std::same_as<O2, std::plus<>> && 
							   std::same_as<O1, std::multiplies<>>)
			{
				// D = A * (B + C) = fmadd(A, B+C, 0) - but need temp
				// Actually: temp = B + C, then D = A * temp
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						register_t temp = _add<T, S>(b[i][j], c_vec);
						d[i][j] = _mul<T, S>(a[i][j], temp);
					});
				});
			}
			else if constexpr (P == FusionPolicy::FUSION_FIRST && 
							   std::same_as<O2, std::minus<>> && 
							   std::same_as<O1, std::multiplies<>>)
			{
				// D = A * (B - C)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						register_t temp = _sub<T, S>(b[i][j], c_vec);
						d[i][j] = _mul<T, S>(a[i][j], temp);
					});
				});
			}
			else
			{
				static_for<M>([&]<auto i>() 
				{
					static_for<N>([&]<auto j>() 
					{
						if constexpr (P == FusionPolicy::UNION_FIRST)
						{
							// D = OP2(OP1(A, B), C)
							register_t temp;
							if constexpr (std::same_as<O1, std::plus<>>)
								temp = _add<T, S>(a[i][j], b[i][j]);
							else if constexpr (std::same_as<O1, std::minus<>>)
								temp = _sub<T, S>(a[i][j], b[i][j]);
							else if constexpr (std::same_as<O1, std::multiplies<>>)
								temp = _mul<T, S>(a[i][j], b[i][j]);
							else if constexpr (std::same_as<O1, std::divides<>>)
								temp = _div<T, S>(a[i][j], b[i][j]);
							
							if constexpr (std::same_as<O2, std::plus<>>)
								d[i][j] = _add<T, S>(temp, c_vec);
							else if constexpr (std::same_as<O2, std::minus<>>)
								d[i][j] = _sub<T, S>(temp, c_vec);
							else if constexpr (std::same_as<O2, std::multiplies<>>)
								d[i][j] = _mul<T, S>(temp, c_vec);
							else if constexpr (std::same_as<O2, std::divides<>>)
								d[i][j] = _div<T, S>(temp, c_vec);
						}
						else // FUSION_FIRST
						{
							// D = OP1(A, OP2(B, C))
							register_t temp;
							if constexpr (std::same_as<O2, std::plus<>>)
								temp = _add<T, S>(b[i][j], c_vec);
							else if constexpr (std::same_as<O2, std::minus<>>)
								temp = _sub<T, S>(b[i][j], c_vec);
							else if constexpr (std::same_as<O2, std::multiplies<>>)
								temp = _mul<T, S>(b[i][j], c_vec);
							else if constexpr (std::same_as<O2, std::divides<>>)
								temp = _div<T, S>(b[i][j], c_vec);
							
							if constexpr (std::same_as<O1, std::plus<>>)
								d[i][j] = _add<T, S>(a[i][j], temp);
							else if constexpr (std::same_as<O1, std::minus<>>)
								d[i][j] = _sub<T, S>(a[i][j], temp);
							else if constexpr (std::same_as<O1, std::multiplies<>>)
								d[i][j] = _mul<T, S>(a[i][j], temp);
							else if constexpr (std::same_as<O1, std::divides<>>)
								d[i][j] = _div<T, S>(a[i][j], temp);
						}
					});
				});
			}
			
			store<T, S, K>(D, d_ptrs, row, col);
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
		template<FusionPolicy P, typename T, typename O1, typename O2, typename S, template<typename, typename> class K>
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
			using kernel = K<T, S>;
			using blocking = typename kernel::blocking;
			constexpr size_t tile_rows = kernel::kernel_rows();
			constexpr size_t tile_cols = kernel::kernel_cols();
			
			constexpr size_t l2_block = blocking::l2_block;
			constexpr size_t l3_block = blocking::l3_block;

			const size_t simd_rows = M - (M % tile_rows);
			const size_t simd_cols = N - (N % tile_cols);

			#pragma omp parallel for schedule(static)
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
							_fused_union_simd_block<P, T, O1, O2, S, K>(A, B, C, D, i, j);
						}
					}
				}
			}

			const size_t rem_rows = M % tile_rows;
			const size_t rem_cols = M % tile_cols;

			// Handle remainders with scalar fallback
			if (rem_rows != 0)
			{
				for (size_t i = simd_rows; i < M; ++i)
					for (size_t j = 0; j < simd_cols; j += tile_cols)
						_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, 1, tile_cols);
			}

			if (rem_cols != 0)
			{
				for (size_t i = 0; i < simd_rows; ++i)
					for (size_t j = simd_cols; j < N; ++j)
					{
						if constexpr (P == FusionPolicy::UNION_FIRST)
							D[i][j] = O2{}(O1{}(A[i][j], B[i][j]), C);
						else
							D[i][j] = O1{}(A[i][j], O2{}(B[i][j], C));
					}
			}

			if (rem_rows != 0 && rem_cols != 0)
			{
				for (size_t i = simd_rows; i < M; ++i)
					for (size_t j = simd_cols; j < N; ++j)
					{
						if constexpr (P == FusionPolicy::UNION_FIRST)
							D[i][j] = O2{}(O1{}(A[i][j], B[i][j]), C);
						else
							D[i][j] = O1{}(A[i][j], O2{}(B[i][j], C));
					}
			}
		}

		/**
	 * \brief SIMD block kernel for scalar fused union (A, scalar, C, D) with FMA optimization
	 */
	template<FusionPolicy P, typename T, typename O1, typename O2, typename S, 
		template<typename, typename> class K>
	inline __attribute__((always_inline))
	void _fused_union_simd_block(T** A, const T B, T** C, T** D, 
		const size_t row, const size_t col)
	{
		using kernel = K<T, S>;
			using blocking = typename kernel::blocking;
		using register_t = typename S::template register_t<T>;
		
		constexpr size_t M = kernel::row_registers;
		constexpr size_t N = kernel::col_registers;

		// Load A and C into registers
		alignas(S::bytes) register_t a[M][N];
		alignas(S::bytes) register_t c[M][N];
		alignas(S::bytes) register_t d[M][N];
		register_t* a_ptrs[M];
		register_t* c_ptrs[M];
		register_t* d_ptrs[M];
		
		for (size_t i = 0; i < M; ++i)
		{
			a_ptrs[i] = a[i];
			c_ptrs[i] = c[i];
			d_ptrs[i] = d[i];
		}
		
		load<T, S, K>(A, a_ptrs, row, col);
		load<T, S, K>(C, c_ptrs, row, col);

		// Broadcast scalar B to SIMD register
		register_t b_vec = _set1<T, S>(B);

		// Detect and apply fused operation patterns
		if constexpr (P == FusionPolicy::FUSION_FIRST && 
					  std::same_as<O2, std::multiplies<>> && 
					  std::same_as<O1, std::plus<>>)
		{
			// D = A + (B * C) = fmadd(B, C, A)
			static_for<M>([&]<auto i>() {
				static_for<N>([&]<auto j>() {
					d[i][j] = _fmadd<T, S>(b_vec, c[i][j], a[i][j]);
				});
			});
		}
		else if constexpr (P == FusionPolicy::FUSION_FIRST && 
						   std::same_as<O2, std::multiplies<>> && 
						   std::same_as<O1, std::minus<>>)
		{
			// D = A - (B * C) = fnmadd(B, C, A)
			static_for<M>([&]<auto i>() {
				static_for<N>([&]<auto j>() {
					d[i][j] = _fnmadd<T, S>(b_vec, c[i][j], a[i][j]);
				});
			});
		}
		else if constexpr (P == FusionPolicy::UNION_FIRST && 
						   std::same_as<O1, std::multiplies<>> && 
						   std::same_as<O2, std::multiplies<>>)
		{
			// D = B * (A * C) = (B * A) * C
			static_for<M>([&]<auto i>() {
				static_for<N>([&]<auto j>() {
					register_t temp = _mul<T, S>(a[i][j], c[i][j]);
					d[i][j] = _mul<T, S>(b_vec, temp);
				});
			});
		}
		else if constexpr (P == FusionPolicy::UNION_FIRST && 
						   std::same_as<O1, std::multiplies<>> && 
						   std::same_as<O2, std::plus<>>)
		{
			// D = B + (A * C) = fmadd(A, C, B)
			static_for<M>([&]<auto i>() {
				static_for<N>([&]<auto j>() {
					d[i][j] = _fmadd<T, S>(a[i][j], c[i][j], b_vec);
				});
			});
		}
		else if constexpr (P == FusionPolicy::UNION_FIRST && 
						   std::same_as<O1, std::multiplies<>> && 
						   std::same_as<O2, std::minus<>>)
		{
			// D = B - (A * C) = fnmadd(A, C, B)
			static_for<M>([&]<auto i>() {
				static_for<N>([&]<auto j>() {
					d[i][j] = _fnmadd<T, S>(a[i][j], c[i][j], b_vec);
				});
			});
		}
		else if constexpr (P == FusionPolicy::UNION_FIRST && 
						   std::same_as<O1, std::plus<>> && 
						   std::same_as<O2, std::multiplies<>>)
		{
			// D = B * (A + C) = fmadd(A+C, B, 0)
			static_for<M>([&]<auto i>() {
				static_for<N>([&]<auto j>() {
					register_t temp = _add<T, S>(a[i][j], c[i][j]);
					d[i][j] = _mul<T, S>(b_vec, temp);
				});
			});
		}
		else if constexpr (P == FusionPolicy::UNION_FIRST && 
						   std::same_as<O1, std::minus<>> && 
						   std::same_as<O2, std::multiplies<>>)
		{
			// D = B * (A - C)
			static_for<M>([&]<auto i>() {
				static_for<N>([&]<auto j>() {
					register_t temp = _sub<T, S>(a[i][j], c[i][j]);
					d[i][j] = _mul<T, S>(b_vec, temp);
				});
			});
		}
		else if constexpr (P == FusionPolicy::FUSION_FIRST && 
						   std::same_as<O2, std::plus<>> && 
						   std::same_as<O1, std::multiplies<>>)
		{
			// D = A * (B + C)
			static_for<M>([&]<auto i>() {
				static_for<N>([&]<auto j>() {
					register_t temp = _add<T, S>(b_vec, c[i][j]);
					d[i][j] = _mul<T, S>(a[i][j], temp);
				});
			});
		}
		else if constexpr (P == FusionPolicy::FUSION_FIRST && 
						   std::same_as<O2, std::minus<>> && 
						   std::same_as<O1, std::multiplies<>>)
		{
			// D = A * (B - C)
			static_for<M>([&]<auto i>() {
				static_for<N>([&]<auto j>() {
					register_t temp = _sub<T, S>(b_vec, c[i][j]);
					d[i][j] = _mul<T, S>(a[i][j], temp);
				});
			});
		}
		else
		{
			// General path
			static_for<M>([&]<auto i>() 
			{
				static_for<N>([&]<auto j>() 
				{
					if constexpr (P == FusionPolicy::UNION_FIRST)
					{
						// D = OP2(B, OP1(A, C))
						register_t temp;
						if constexpr (std::same_as<O1, std::plus<>>)
							temp = _add<T, S>(a[i][j], c[i][j]);
						else if constexpr (std::same_as<O1, std::minus<>>)
							temp = _sub<T, S>(a[i][j], c[i][j]);
						else if constexpr (std::same_as<O1, std::multiplies<>>)
							temp = _mul<T, S>(a[i][j], c[i][j]);
						else if constexpr (std::same_as<O1, std::divides<>>)
							temp = _div<T, S>(a[i][j], c[i][j]);
						
						if constexpr (std::same_as<O2, std::plus<>>)
							d[i][j] = _add<T, S>(b_vec, temp);
						else if constexpr (std::same_as<O2, std::minus<>>)
							d[i][j] = _sub<T, S>(b_vec, temp);
						else if constexpr (std::same_as<O2, std::multiplies<>>)
							d[i][j] = _mul<T, S>(b_vec, temp);
						else if constexpr (std::same_as<O2, std::divides<>>)
							d[i][j] = _div<T, S>(b_vec, temp);
					}
					else // FUSION_FIRST
					{
						// D = OP1(A, OP2(B, C))
						register_t temp;
						if constexpr (std::same_as<O2, std::plus<>>)
							temp = _add<T, S>(b_vec, c[i][j]);
						else if constexpr (std::same_as<O2, std::minus<>>)
							temp = _sub<T, S>(b_vec, c[i][j]);
						else if constexpr (std::same_as<O2, std::multiplies<>>)
							temp = _mul<T, S>(b_vec, c[i][j]);
						else if constexpr (std::same_as<O2, std::divides<>>)
							temp = _div<T, S>(b_vec, c[i][j]);
						
						if constexpr (std::same_as<O1, std::plus<>>)
							d[i][j] = _add<T, S>(a[i][j], temp);
						else if constexpr (std::same_as<O1, std::minus<>>)
							d[i][j] = _sub<T, S>(a[i][j], temp);
						else if constexpr (std::same_as<O1, std::multiplies<>>)
							d[i][j] = _mul<T, S>(a[i][j], temp);
						else if constexpr (std::same_as<O1, std::divides<>>)
							d[i][j] = _div<T, S>(a[i][j], temp);
					}
				});
			});
		}
		
		store<T, S, K>(D, d_ptrs, row, col);
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
		template<FusionPolicy P, typename T, typename O1, typename O2, typename S, template <typename, typename> class K>
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

			using kernel = K<T, S>;
			using blocking = typename kernel::blocking;
			constexpr size_t tile_rows = kernel::kernel_rows();
			constexpr size_t tile_cols = kernel::kernel_cols();
			
			constexpr size_t l2_block = blocking::l2_block;
			constexpr size_t l3_block = blocking::l3_block;

			const size_t simd_rows = M - (M % tile_rows);
			const size_t simd_cols = N - (N % tile_cols);

			#pragma omp parallel for schedule(static)
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
							_fused_union_simd_block<P, T, O1, O2, S, K>(A, B, C, D, i, j);
						}
					}
				}
			}

			const size_t rem_rows = M % tile_rows;
			const size_t rem_cols = M % tile_cols;

			// Handle remainders
			if (rem_rows != 0)
			{
				for (size_t i = simd_rows; i < M; ++i)
					for (size_t j = 0; j < simd_cols; j += tile_cols)
						_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, 1, tile_cols);
			}

			if (rem_cols != 0)
			{
				for (size_t i = 0; i < simd_rows; ++i)
					for (size_t j = simd_cols; j < N; ++j)
					{
						if constexpr (P == FusionPolicy::UNION_FIRST)
							D[i][j] = O2{}(B, O1{}(A[i][j], C[i][j]));
						else
							D[i][j] = O1{}(A[i][j], O2{}(B, C[i][j]));
					}
			}

			if (rem_rows != 0 && rem_cols != 0)
			{
				for (size_t i = simd_rows; i < M; ++i)
					for (size_t j = simd_cols; j < N; ++j)
					{
						if constexpr (P == FusionPolicy::UNION_FIRST)
							D[i][j] = O2{}(B, O1{}(A[i][j], C[i][j]));
						else
							D[i][j] = O1{}(A[i][j], O2{}(B, C[i][j]));
					}
			}
		}

		/**
		 * \brief Perform element-wise fused union with signature (A, B, scalar, D) using SIMD and blocking algorithms.
		 *
		 * This is the main public interface for scalar-matrix fused union operations.
		 *
		 * The fused operation computes either OP2(OP1(A[i][j], B[i][j]), C) for UNION_FIRST
		 * or OP1(A[i][j], OP2(B[i][j], C)) for FUSION_FIRST, enabling efficient BLAS-style
		 * patterns like compound operations and scalar broadcasting.
		 *
		 * \tparam P	Fusion policy controlling operation precedence
		 * \tparam T	Element type of the matrices (e.g., float, double)
		 * \tparam O1	First binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam O2	Second binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam S	SIMD instruction set to use (SSE, AVX, AVX512, or NONE)
		 * \tparam K	Kernel policy defining cache blocking sizes
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

		template<FusionPolicy P, typename T, typename O1, typename O2, typename S = decltype(detect_simd()),
			template<typename, typename> class K = fused_union_kernel>
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
			
			if constexpr (std::is_same_v<S, NONE>)
				_fused_union<P, T, O1, O2, K>(A, B, C, D, M, N);
			else
				_fused_union_simd<P, T, O1, O2, S, K>(A, B, C, D, M, N);
		}

		/**
		 * \brief Perform optimized element-wise fused union with signature (A, scalar, B, D) using SIMD and blocking algorithms.
		 *
		 * This is the main public interface for scalar-matrix fused union operations with alternative
		 * operand positioning. The scalar operand is positioned between the
		 * two matrices, enabling different BLAS patterns through signature-controlled operation ordering.
		 *
		 * The fused operation computes either OP2(B, OP1(A[i][j], C[i][j])) for UNION_FIRST
		 * or OP1(A[i][j], OP2(B, C[i][j])) for FUSION_FIRST, unlocking alternative BLAS-style
		 * patterns like A + α*C or A*α + C through strategic operand positioning.
		 *
		 * \tparam P	Fusion policy controlling operation precedence
		 * \tparam T	Element type of the matrices (e.g., float, double)
		 * \tparam O1	First binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam O2	Second binary operator (e.g., std::plus<>, std::multiplies<>)
		 * \tparam S	SIMD instruction set to use (SSE, AVX, AVX512, or NONE)
		 * \tparam K	Kernel policy defining cache blocking sizes
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
			template<FusionPolicy P, typename T, typename O1, typename O2, typename S = decltype(detect_simd()),
			template<typename, typename> class K = fused_union_kernel>
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

			if constexpr (std::is_same_v<S, NONE>)
				_fused_union<P, T, O1, O2, K>(A, B, C, D, M, N);
			else
				_fused_union_simd<P, T, O1, O2, S, K>(A, B, C, D, M, N);
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
		 * \brief	Element-wise fused union of three 2D pointer matrices using cache-blocked traversal.
		 *
		 * Performs blocked element-wise fused operations on three matrices A, B, and C,
		 * writing the result into matrix D. The operation order is controlled by FusionPolicy:
		 * UNION_FIRST computes OP2(OP1(A,B), C), while FUSION_FIRST computes OP1(A, OP2(B,C)).
		 * Uses cache-aware blocking determined by the kernel policy.
		 *
		 * \tparam P	Fusion policy controlling operation order
		 * \tparam T	Scalar type (e.g., float or double)
		 * \tparam O1	First binary operator (e.g., std::plus<>)
		 * \tparam O2	Second binary operator (e.g., std::multiplies<>)
		 * \tparam K	Kernel policy defining cache blocking sizes
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
		template <FusionPolicy P, typename T, typename O1, typename O2, template<typename, typename> class K>	
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
			using kernel = K<T, NONE>;
			using blocking = typename kernel::blocking;
			
			constexpr size_t l2_block = blocking::l2_block;
			constexpr size_t l3_block = blocking::l3_block;
			
			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < M; i += l2_block)
			{
				for (size_t j = 0; j < N; j += l3_block)
				{ 
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l3_block, N - j);
					_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, m, n);
				}
			}
		}

		/**
		 * \brief SIMD block kernel for matrix fused union with FMA optimization
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, typename S, 
			template<typename, typename> class K>
		inline __attribute__((always_inline))
		void _fused_union_simd_block(T** A, T** B, T** C, T** D, 
			const size_t row, const size_t col)
		{
			using kernel = K<T, S>;
			using blocking = typename kernel::blocking;
			using register_t = typename S::template register_t<T>;
			
			constexpr size_t M = kernel::row_registers;
			constexpr size_t N = kernel::col_registers;

			// Load all three matrices
			alignas(S::bytes) register_t a[M][N];
			alignas(S::bytes) register_t b[M][N];
			alignas(S::bytes) register_t c[M][N];
			alignas(S::bytes) register_t d[M][N];
			register_t* a_ptrs[M];
			register_t* b_ptrs[M];
			register_t* c_ptrs[M];
			register_t* d_ptrs[M];
			
			for (size_t i = 0; i < M; ++i)
			{
				a_ptrs[i] = a[i];
				b_ptrs[i] = b[i];
				c_ptrs[i] = c[i];
				d_ptrs[i] = d[i];
			}
			
			load<T, S, K>(A, a_ptrs, row, col);
			load<T, S, K>(B, b_ptrs, row, col);
			load<T, S, K>(C, c_ptrs, row, col);

			// Detect and apply fused operation patterns
			if constexpr (P == FusionPolicy::UNION_FIRST && 
			              std::same_as<O1, std::multiplies<>> && 
			              std::same_as<O2, std::plus<>>)
			{
				// D = (A * B) + C = fmadd(A, B, C)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						d[i][j] = _fmadd<T, S>(a[i][j], b[i][j], c[i][j]);
					});
				});
			}
			else if constexpr (P == FusionPolicy::UNION_FIRST && 
			                   std::same_as<O1, std::multiplies<>> && 
			                   std::same_as<O2, std::minus<>>)
			{
				// D = (A * B) - C = fmsub(A, B, C)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						d[i][j] = _fmsub<T, S>(a[i][j], b[i][j], c[i][j]);
					});
				});
			}
			else if constexpr (P == FusionPolicy::FUSION_FIRST && 
			                   std::same_as<O2, std::multiplies<>> && 
			                   std::same_as<O1, std::plus<>>)
			{
				// D = A + (B * C) = fmadd(B, C, A)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						d[i][j] = _fmadd<T, S>(b[i][j], c[i][j], a[i][j]);
					});
				});
			}
			else if constexpr (P == FusionPolicy::FUSION_FIRST && 
			                   std::same_as<O2, std::multiplies<>> && 
			                   std::same_as<O1, std::minus<>>)
			{
				// D = A - (B * C) = fnmadd(B, C, A)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						d[i][j] = _fnmadd<T, S>(b[i][j], c[i][j], a[i][j]);
					});
				});
			}
			else if constexpr (P == FusionPolicy::UNION_FIRST && 
			                   std::same_as<O1, std::plus<>> && 
			                   std::same_as<O2, std::multiplies<>>)
			{
				// D = (A + B) * C
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						register_t temp = _add<T, S>(a[i][j], b[i][j]);
						d[i][j] = _mul<T, S>(temp, c[i][j]);
					});
				});
			}
			else if constexpr (P == FusionPolicy::UNION_FIRST && 
			                   std::same_as<O1, std::minus<>> && 
			                   std::same_as<O2, std::multiplies<>>)
			{
				// D = (A - B) * C
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						register_t temp = _sub<T, S>(a[i][j], b[i][j]);
						d[i][j] = _mul<T, S>(temp, c[i][j]);
					});
				});
			}
			else if constexpr (P == FusionPolicy::FUSION_FIRST && 
			                   std::same_as<O2, std::plus<>> && 
			                   std::same_as<O1, std::multiplies<>>)
			{
				// D = A * (B + C)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						register_t temp = _add<T, S>(b[i][j], c[i][j]);
						d[i][j] = _mul<T, S>(a[i][j], temp);
					});
				});
			}
			else if constexpr (P == FusionPolicy::FUSION_FIRST && 
			                   std::same_as<O2, std::minus<>> && 
			                   std::same_as<O1, std::multiplies<>>)
			{
				// D = A * (B - C)
				static_for<M>([&]<auto i>() {
					static_for<N>([&]<auto j>() {
						register_t temp = _sub<T, S>(b[i][j], c[i][j]);
						d[i][j] = _mul<T, S>(a[i][j], temp);
					});
				});
			}
			else
			{
				// General path
				static_for<M>([&]<auto i>() 
				{
					static_for<N>([&]<auto j>() 
					{
						if constexpr (P == FusionPolicy::UNION_FIRST)
						{
							// D = OP2(OP1(A, B), C)
							register_t temp;
							if constexpr (std::same_as<O1, std::plus<>>)
								temp = _add<T, S>(a[i][j], b[i][j]);
							else if constexpr (std::same_as<O1, std::minus<>>)
								temp = _sub<T, S>(a[i][j], b[i][j]);
							else if constexpr (std::same_as<O1, std::multiplies<>>)
								temp = _mul<T, S>(a[i][j], b[i][j]);
							else if constexpr (std::same_as<O1, std::divides<>>)
								temp = _div<T, S>(a[i][j], b[i][j]);
							
							if constexpr (std::same_as<O2, std::plus<>>)
								d[i][j] = _add<T, S>(temp, c[i][j]);
							else if constexpr (std::same_as<O2, std::minus<>>)
								d[i][j] = _sub<T, S>(temp, c[i][j]);
							else if constexpr (std::same_as<O2, std::multiplies<>>)
								d[i][j] = _mul<T, S>(temp, c[i][j]);
							else if constexpr (std::same_as<O2, std::divides<>>)
								d[i][j] = _div<T, S>(temp, c[i][j]);
						}
						else // FUSION_FIRST
						{
							// D = OP1(A, OP2(B, C))
							register_t temp;
							if constexpr (std::same_as<O2, std::plus<>>)
								temp = _add<T, S>(b[i][j], c[i][j]);
							else if constexpr (std::same_as<O2, std::minus<>>)
								temp = _sub<T, S>(b[i][j], c[i][j]);
							else if constexpr (std::same_as<O2, std::multiplies<>>)
								temp = _mul<T, S>(b[i][j], c[i][j]);
							else if constexpr (std::same_as<O2, std::divides<>>)
								temp = _div<T, S>(b[i][j], c[i][j]);
							
							if constexpr (std::same_as<O1, std::plus<>>)
								d[i][j] = _add<T, S>(a[i][j], temp);
							else if constexpr (std::same_as<O1, std::minus<>>)
								d[i][j] = _sub<T, S>(a[i][j], temp);
							else if constexpr (std::same_as<O1, std::multiplies<>>)
								d[i][j] = _mul<T, S>(a[i][j], temp);
							else if constexpr (std::same_as<O1, std::divides<>>)
								d[i][j] = _div<T, S>(a[i][j], temp);
						}
					});
				});
			}
			
			store<T, S, K>(D, d_ptrs, row, col);
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
		 * \note Matrices must have the same dimensions (MÃ—N).
		 * \note Asymmetric matrices (M â‰  N) are supported.
		 * \note Strides not aligned with SIMD block sizes are safely handled.
		 * \note 2D interface maintained while leveraging SIMD optimizations internally.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, typename S, template<typename, typename> class K>
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
			using kernel = K<T, S>;
			using blocking = typename kernel::blocking;
	constexpr size_t tile_rows = kernel::kernel_rows();
	constexpr size_t tile_cols = kernel::kernel_cols();
	
	constexpr size_t l2_block = blocking::l2_block;
	constexpr size_t l3_block = blocking::l3_block;

			const size_t simd_rows = M - (M % tile_rows);
			const size_t simd_cols = N - (N % tile_cols);

			#pragma omp parallel for schedule(static)
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
							_fused_union_simd_block<P, T, O1, O2, S, K>(A, B, C, D, i, j);
						}
					}
				}
			}

			const size_t rem_rows = M % tile_rows;
			const size_t rem_cols = M % tile_cols;

			// Handle remainders
			if (rem_rows != 0)
			{
				for (size_t i = simd_rows; i < M; ++i)
					for (size_t j = 0; j < simd_cols; j += tile_cols)
						_fused_union_block<P, T, O1, O2>(A, B, C, D, i, j, 1, tile_cols);
			}

			if (rem_cols != 0)
			{
				for (size_t i = 0; i < simd_rows; ++i)
					for (size_t j = simd_cols; j < N; ++j)
					{
						if constexpr (P == FusionPolicy::UNION_FIRST)
							D[i][j] = O2{}(O1{}(A[i][j], B[i][j]), C[i][j]);
						else
							D[i][j] = O1{}(A[i][j], O2{}(B[i][j], C[i][j]));
					}
			}

			if (rem_rows != 0 && rem_cols != 0)
			{
				for (size_t i = simd_rows; i < M; ++i)
					for (size_t j = simd_cols; j < N; ++j)
					{
						if constexpr (P == FusionPolicy::UNION_FIRST)
							D[i][j] = O2{}(O1{}(A[i][j], B[i][j]), C[i][j]);
						else
							D[i][j] = O1{}(A[i][j], O2{}(B[i][j], C[i][j]));
					}
			}
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
		 * \note Non-square matrices and asymmetric dimensions (M â‰  N) are fully supported.
		 * \note When S=NONE, the function uses standard blocked operations without SIMD.
		 * \note SIMD implementations automatically handle non-aligned dimensions with scalar fallback.
		 * \note Enables comprehensive BLAS-style ternary operations with configurable precedence.
		 *
		 * \throws std::invalid_argument if any matrix pointer is null.
		 * \throws std::runtime_error if memory layout validation fails.
		 */
		template<FusionPolicy P, typename T, typename O1, typename O2, typename S = decltype(detect_simd()),
			template<typename, typename> class K = fused_union_kernel>
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

			if constexpr (std::is_same_v<S, NONE>)
				_fused_union<P, T, O1, O2, K>(A, B, C, D, M, N);
			else
				_fused_union_simd<P, T, O1, O2, S, K>(A, B, C, D, M, N);
		}

	} // namespace matrix

} //namespace damm
#endif //__FUSED_UNION_H__