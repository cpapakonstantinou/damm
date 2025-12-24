#ifndef __UNION_H__
#define __UNION_H__
/**
 * \file union.h
 * \brief definitions for union utilities 
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
 * Computational union.
 *
 * \note 
 * A union in this context refers to a merge over an index domain, where 
 * the merge operation is defined by an arbitrary arithmetic operator 
 * (e.g., addition, subtraction, or pointwise multiplication such as 
 * the Hadamard product).
 *
 * \note
 * This is not a union in the strict set-theoretic sense. Instead, it 
 * reflects a computational interpretation: elements are combined using 
 * a defined arithmetic rule rather than simply accumulated as distinct 
 * members of a set.
 */
namespace damm
{

	namespace scalar 
	{
		/** 
		 * \brief kernel for scalar union_block. 
		 * Low level function not intended for the public API.
		 * */
		template <typename T, typename O>
		inline __attribute__((always_inline)) 
		void
		_union_block(T** A, const T B, T** C, 
			const size_t I, const size_t J,
			const size_t M, const size_t N)
		{
			for(size_t i = 0; i < M; i++)
				for(size_t j = 0; j < N; j++)
					C[I+i][J+j] = O{}(A[I+i][J+j], B);
		}

		/**
		 * \brief	Element-wise union of a 2D pointer matrix with scalar using a binary operation.
		 *
		 * Performs a blocked element-wise operation (union) on matrix A with scalar B,
		 * writing the result into matrix C. The operation is defined by the template
		 * parameter `O`, and must be one of `std::plus<>`, `std::minus<>`,
		 * `std::multiplies<>`, or `std::divides<>`.
		 *
		 * \tparam T	Scalar type (e.g., float or double)
		 * \tparam O	Binary operator (e.g., std::plus<>)
		 * \tparam K	Kernel policy defining cache blocking sizes
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Scalar operand
		 * \param C		Output matrix C as array of M row pointers, each of size N
		 * \param M		Number of rows in A and C
		 * \param N		Number of columns in A and C
		 *
		 * \note	Uses broadcast semantics: scalar B is combined with each element of A.
		 * \note	Blocked traversal ensures improved cache locality.
		 * \note	More efficient than matrix-matrix operations when one operand is scalar.
		 */
		template <typename T, typename O, template<typename, typename> class K>	
		requires 
		(
			std::same_as<O, std::plus<>> ||
			std::same_as<O, std::minus<>> ||
			std::same_as<O, std::multiplies<>> ||
			std::same_as<O, std::divides<>>
		)
		inline __attribute__((always_inline))
		void
		_union(T** A, const T B, T** C, const size_t M, const size_t N)
		{
			using kernel = K<T, NONE>;
			using blocking = typename kernel::blocking;
			
			constexpr size_t l2_block = blocking::l2_block;
			constexpr size_t l3_block = blocking::l3_block;
			
			#pragma omp parallel for schedule(static, l2_block)
			for(size_t i = 0; i < M; i+=l2_block)
			{
				for (size_t j = 0; j < N; j += l3_block)
				{ 
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l3_block, N - j);
					_union_block<T, O>(A, B, C, i, j, m, n);
				}
			}
		}

		template<typename T, typename O, typename S, template<typename, typename> class K>
		requires 
		(
			std::same_as<O, std::plus<>> ||
			std::same_as<O, std::minus<>> ||
			std::same_as<O, std::multiplies<>> ||
			std::same_as<O, std::divides<>>
		) 
		inline __attribute__((always_inline))
		void
		_union_simd_block(T** A, const T B, T** C, const size_t row, const size_t col)
		{
			using kernel = K<T, S>;
			using register_t = typename S::template register_t<T>;
			constexpr size_t rows = kernel::row_registers;
			constexpr size_t cols = kernel::col_registers;
			
			// Allocate 2D register array
			alignas(S::bytes) register_t a[rows][cols];
			alignas(S::bytes) register_t c[rows][cols];
			register_t* aptrs[rows];
			register_t* cptrs[rows];

			for (size_t i = 0; i < rows; ++i)
			{
				aptrs[i] = a[i];
				cptrs[i] = c[i];
			}
			
			load<T, S, K>(A, aptrs, row, col);

			// Process output rows
			static_for<rows>([&]<auto i>() 
			{
				register_t b = _set1<T, S>(B);
				static_for<cols>([&]<auto j>() 
				{
					if constexpr ( std::same_as<O, std::plus<>> )
						cptrs[i][j] = _add<T, S>(a[i][j], b);
					else if constexpr ( std::same_as<O, std::minus<>> )
						cptrs[i][j] = _sub<T, S>(a[i][j], b);
					else if constexpr ( std::same_as<O, std::multiplies<>> )
						cptrs[i][j] = _mul<T, S>(a[i][j], b);
					else if constexpr ( std::same_as<O, std::divides<>> )
						cptrs[i][j] = _div<T, S>(a[i][j], b);
				});
			});
			
			store<T, S, K>(C, cptrs, row, col);
		}


		/**
		 * \brief Element-wise union of matrix with scalar using SIMD intrinsics.
		 *
		 * Performs an element-wise binary operation on 2D matrix A with scalar B, storing
		 * the result in matrix C. SIMD vectorization is selected at compile time via the
		 * template parameter `S`. This is the main entry point for SIMD-based scalar union operations.
		 *
		 * \tparam T Scalar type (float or double)
		 * \tparam O Binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam S SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam K Kernel policy defining cache blocking sizes
		 *
		 * \param A Pointer to a 2D matrix A
		 * \param B Scalar operand to broadcast
		 * \param C Pointer to a 2D matrix C for storing the result
		 * \param M Number of rows
		 * \param N Number of columns
		 *
		 * \note Uses broadcast semantics: scalar B is combined with each element of A.
		 * \note More efficient than matrix-matrix operations when one operand is scalar.
		 */
		template<typename T, typename O, typename S, template<typename, typename> class K>
		requires 
		(
			std::same_as<O, std::plus<>> ||
			std::same_as<O, std::minus<>> ||
			std::same_as<O, std::multiplies<>> ||
			std::same_as<O, std::divides<>>
		) 
		inline __attribute__((always_inline))
		void
		_union_simd(T** A, const T B, T** C, const size_t M, const size_t N)
		{
			using kernel = K<T, S>;
			using blocking = typename kernel::blocking;
			
			constexpr size_t l2_block = blocking::l2_block;
			constexpr size_t l3_block = blocking::l3_block;
			
			constexpr size_t tile_rows = kernel::kernel_rows();
			constexpr size_t tile_cols = kernel::kernel_cols();
		
			const size_t simd_rows = M - (M % tile_rows);
			const size_t simd_cols = N - (N % tile_cols);

			#pragma omp parallel for schedule(static, l2_block)
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
							_union_simd_block<T, O, S, K>(A, B, C, i, j);
						}
					}
				}
			}
			
			const size_t rem_rows = M % tile_rows;
			const size_t rem_cols = N % tile_cols;

			if (rem_rows != 0) 
			{
				for (size_t j = 0; j < simd_cols; j += tile_cols)
					_union_block<T, O>(A, B, C, simd_rows, j, rem_rows, tile_cols);
			}
			
			if (rem_cols != 0)
			{
				for (size_t i = 0; i < simd_rows; i += tile_rows)
					_union_block<T, O>(A, B, C, i, simd_cols, tile_rows, rem_cols);
			}
			
			if (rem_rows != 0 && rem_cols != 0)
			{
				_union_block<T, O>(A, B, C, simd_rows, simd_cols, rem_rows, rem_cols);
			}
		}

		//this function breaks the naming convention due to conflicts with the C keyword "union"
		template<typename T, typename O, typename S = decltype(detect_simd()),
			template<typename, typename> class K = union_kernel>
		requires 
		(
			std::same_as<O, std::plus<>> ||
			std::same_as<O, std::minus<>> ||
			std::same_as<O, std::multiplies<>> ||
			std::same_as<O, std::divides<>>
		) 
		inline __attribute__((always_inline)) 
		void
		unite(T** A, const T B, T** C, const size_t M, const size_t N)
		{
			right<T>("union: ", std::make_tuple(A, M, N), std::make_tuple(C, M, N));

			if constexpr (std::is_same_v<S, NONE>) 
				_union<T, O, K>(A, B, C, M, N);
			else
				_union_simd<T, O, S, K>(A, B, C, M, N);
		}

	} // namespace scalar


	namespace matrix 
	{

		/** 
		 * \brief kernel for matrix union_block. 
		 * Low level function not intended for the public API.
		 * */
		template <typename T, typename O>
		inline __attribute__((always_inline)) 
		void
		_union_block(T** A, T** B, T** C, 
			const size_t I, const size_t J,
			const size_t M, const size_t N)
		{
			for(size_t i = 0; i < M; i++) 
				for(size_t j = 0; j < N; j++)
					C[I+i][J+j] = O{}(A[I+i][J+j], B[I+i][J+j]);
		}

		/**
		 * \brief	Element-wise union of two 2D pointer matrices using a binary operation.
		 *
		 * Performs a blocked element-wise operation (union) on two matrices A and B,
		 * writing the result into matrix C. The operation is defined by the template
		 * parameter `O`, and must be one of `std::plus<>`, `std::minus<>`,
		 * `std::multiplies<>`, or `std::divides<>`.
		 *
		 * \tparam T	Scalar type (e.g., float or double)
		 * \tparam O	Binary operator (e.g., std::plus<>)
		 * \tparam K	Kernel policy defining cache blocking sizes
		 *
		 * \param A		Matrix A as array of M row pointers, each of size N
		 * \param B		Matrix B as array of M row pointers, each of size N
		 * \param C		Output matrix C as array of M row pointers, each of size N
		 * \param M		Number of rows in A, B, and C
		 * \param N		Number of columns in A, B, and C
		 *
		 * \note	Matrices must have the same shape: M × N.
		 * \note	Blocked traversal ensures improved cache locality.
		 * \note	Requires that O is one of the standard arithmetic function objects.
		 * \note	This function operates on a 2D array-of-pointers representation.
		 */
		template <typename T, typename O, template<typename, typename> class K>	
		requires (
			std::same_as<O, std::plus<>> ||
			std::same_as<O, std::minus<>> ||
			std::same_as<O, std::multiplies<>> ||
			std::same_as<O, std::divides<>>
		)
		inline __attribute__((always_inline))
		void
		_union(T** A, T** B, T** C, const size_t M, const size_t N)
		{
			using kernel = K<T, NONE>;
			using blocking = typename kernel::blocking;
			
			constexpr size_t l2_block = blocking::l2_block;
			constexpr size_t l3_block = blocking::l3_block;
			
			#pragma omp parallel for schedule(static, l2_block)
			for(size_t i = 0; i < M; i+=l2_block)
			{
				for (size_t j = 0; j < N; j += l3_block)
				{ 
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l3_block, N - j);
					_union_block<T, O>(A, B, C, i, j, m, n);
				}
			}
		}

		template<typename T, typename O, typename S, template<typename, typename> class K>
		requires 
		(
			std::same_as<O, std::plus<>> ||
			std::same_as<O, std::minus<>> ||
			std::same_as<O, std::multiplies<>> ||
			std::same_as<O, std::divides<>>
		) 
		inline __attribute__((always_inline))
		void
		_union_simd_block(T** A, T** B, T** C, const size_t row, const size_t col)
		{
			using kernel = K<T, S>;
			using register_t = typename S::template register_t<T>;
			constexpr size_t rows = kernel::row_registers;
			constexpr size_t cols = kernel::col_registers;
			
			alignas(S::bytes) register_t a[rows][cols];
			alignas(S::bytes) register_t b[rows][cols];
			alignas(S::bytes) register_t c[rows][cols];
			register_t* aptrs[rows];
			register_t* bptrs[rows];
			register_t* cptrs[rows];

			for (size_t i = 0; i < rows; ++i)
			{
				aptrs[i] = a[i];
				bptrs[i] = b[i];
				cptrs[i] = c[i];
			}
			
			load<T, S, K>(A, aptrs, row, col);
			load<T, S, K>(B, bptrs, row, col);

			// Process output rows
			static_for<rows>([&]<auto i>() 
			{
				static_for<cols>([&]<auto j>() 
				{
					if constexpr ( std::same_as<O, std::plus<>> )
						cptrs[i][j] = _add<T, S>(a[i][j], b[i][j]);
					else if constexpr ( std::same_as<O, std::minus<>> )
						cptrs[i][j] = _sub<T, S>(a[i][j], b[i][j]);
					else if constexpr ( std::same_as<O, std::multiplies<>> )
						cptrs[i][j] = _mul<T, S>(a[i][j], b[i][j]);
					else if constexpr ( std::same_as<O, std::divides<>> )
						cptrs[i][j] = _div<T, S>(a[i][j], b[i][j]);
				});
			});
			
			store<T, S, K>(C, cptrs, row, col);
		}

		/**
		 * \brief Element-wise union of two matrices using SIMD intrinsics.
		 *
		 * Performs an element-wise binary operation on 2D matrices A and B, storing
		 * the result in matrix C. SIMD vectorization is selected at compile time via the
		 * template parameter `S`. This is the main entry point for SIMD-based union operations.
		 *
		 * \tparam T Scalar type (float or double)
		 * \tparam O Binary operator: must be one of std::plus<>, std::minus<>, std::multiplies<>, or std::divides<>
		 * \tparam S SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam K Kernel policy defining cache blocking sizes
		 *
		 * \param A Pointer to a 2D matrix A (outer matrix)
		 * \param B Pointer to a 2D matrix B (inner matrix)
		 * \param C Pointer to a 2D matrix C for storing the result
		 * \param M Number of rows
		 * \param N Number of columns
		 *
		 * \note Matrices must have the same dimensions (M×N).
		 * \note Asymmetric matrices (M ≠ N) are supported.
		 * \note Strides not aligned with SIMD block sizes are safely handled.
		 */
		template<typename T, typename O, typename S, template<typename, typename> class K>
		requires (
			std::same_as<O, std::plus<>> ||
			std::same_as<O, std::minus<>> ||
			std::same_as<O, std::multiplies<>> ||
			std::same_as<O, std::divides<>>
		) 
		inline __attribute__((always_inline))
		void
		_union_simd(T** A, T** B, T** C, const size_t M, const size_t N)
		{
			using kernel = K<T, S>;
			using blocking = typename kernel::blocking;
			
			constexpr size_t l2_block = blocking::l2_block;
			constexpr size_t l3_block = blocking::l3_block;
			
			constexpr size_t tile_rows = kernel::kernel_rows();
			constexpr size_t tile_cols = kernel::kernel_cols();
		
			const size_t simd_rows = M - (M % tile_rows);
			const size_t simd_cols = N - (N % tile_cols);

			#pragma omp parallel for schedule(static, l2_block)
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
							_union_simd_block<T, O, S, K>(A, B, C, i, j);
						}
					}
				}
			}
			
			const size_t rem_rows = M % tile_rows;
			const size_t rem_cols = N % tile_cols;

			if (rem_rows != 0) 
			{
				for (size_t j = 0; j < simd_cols; j += tile_cols)
					_union_block<T, O>(A, B, C, simd_rows, j, rem_rows, tile_cols);
			}
			
			if (rem_cols != 0)
			{
				for (size_t i = 0; i < simd_rows; i += tile_rows)
					_union_block<T, O>(A, B, C, i, simd_cols, tile_rows, rem_cols);
			}
			
			if (rem_rows != 0 && rem_cols != 0)
			{
				_union_block<T, O>(A, B, C, simd_rows, simd_cols, rem_rows, rem_cols);
			}
		}

	//this function breaks the naming convention due to conflicts with the C keyword "union"
		template<typename T, typename O, typename S = decltype(detect_simd()),
			template<typename, typename> class K = union_kernel>
		requires 
		(
			std::same_as<O, std::plus<>> ||
			std::same_as<O, std::minus<>> ||
			std::same_as<O, std::multiplies<>> ||
			std::same_as<O, std::divides<>>
		) 
		inline __attribute__((always_inline))
		void
		unite(T** A, T** B, T** C, const size_t M, const size_t N)
		{
			right<T>("union:", std::make_tuple(A, M, N), std::make_tuple(B, M, N), std::make_tuple(C, M, N));

			if constexpr (std::is_same_v<S, NONE>) 
				_union<T, O, K>(A, B, C, M, N);
			else
				_union_simd<T, O, S, K>(A, B, C, M, N);
		}

	} // namespace matrix

} //namespace damm

#endif //__UNION_H__