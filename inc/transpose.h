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
// IMPLIED, INCLUDING BUT MOT LIMITED TO THE WARRANTIES OF NERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND MONINFRINGEMENT. IN MO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <common.h>
#include <simd.h>
#include <damm_kernels.h>
#include <omp.h>

namespace damm
{
	/** \brief kernal for transpose_block. Low level function not intended for the public API*/
	template <typename T>
	inline __attribute__((always_inline))
	void
	_transpose_block(T** A, T** B, const size_t I, const size_t J, const size_t M, const size_t N)
	{
		for(size_t i=0; i < M; i++) 
			for(size_t j=0; j < N; j++) 
				 B[J + j][I + i] = A[I + i][J + j];
	}
	
	/**
	 * \brief Transpose a matrix using a blocked and parallel strategy.
	 *
	 * This function performs a block-wise transpose of a matrix  A into  B,
	 * optimizing for cache locality and enabling multithreaded execution. It operates
	 * on matrix views that provide 2D access (e.g., row-major pointer-to-pointer layout),
	 * typically created from contiguous memory.
	 *
	 * \tparam T			Element type of the matrices (e.g., float, double).
	 * \tparam K			Kernel policy defining tile size.
	 *
	 * \param A Source matrix view with dimensions M × N.
	 * \param B Destination matrix view with dimensions N × M.
	 * \param M Number of rows in matrix A.
	 * \param N Number of columns in matrix A.
	 *
	 * \note Both  A and  B must be valid 2D views over contiguous row-major memory.
	 *       This function assumes full allocation, not submatrices.
	 * \note The transpose supports asymmetric matrices (M != N) and dimensions
	 *       that are not multiples of the block size.
	 */
	template <typename T, template<typename, typename> class K>
	inline __attribute__((always_inline))
	void
	_transpose(T** A, T** B, const size_t M, const size_t N)
	{
		using kernel = K<T, NONE>;
		using blocking = typename kernel::blocking;
		
		constexpr size_t l1_block = blocking::l1_block;
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;
		
		// L3 blocking (parallel over rows of A)
		#pragma omp parallel for schedule(static, l2_block)
		for (size_t i = 0; i < M; i += l2_block)
		{
			// L2 blocking over columns of A
			for (size_t j = 0; j < N; j += l3_block) 
			{
				// L1 blocking
				for (size_t k = 0; k < std::min(l3_block, N - j); k += l1_block)
				{
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l1_block, std::min(l3_block, N - j) - k);
					_transpose_block(A, B, i, j + k, m, n);
				}
			}
		}
	}

	template<typename T, typename S>
	void __transpose_block_simd(typename S::template register_t<T>* r);

	/**
	 * \brief Transpose a register array by dispatching to specialized implementations.
	 *
	 * \tparam T		The scalar type (float, double, or complex variants).
	 * \tparam S		The SIMD instruction set tag (SSE, AVX, or AVX512).
	 * \param  rows		2D array of SIMD registers to transpose in-place.
	 *
	 * \note This function modifies the register array in-place.
	 */
	template<typename T, typename S, 
		template<typename, typename> class K = transpose_kernel>
	inline __attribute__((always_inline))
	void transpose(typename S::template register_t<T>** rows)
	{
		using kernel = K<T, S>;
		using register_t = typename S::template register_t<T>;
		
		constexpr size_t r = kernel::row_registers;
		constexpr size_t c = kernel::col_registers;
		constexpr size_t e = kernel::register_elements();
		
		static_assert(r % e == 0, "row registers must be divisible by register elements");
		
		if constexpr (r == e && c == 1) 
		{
			__transpose_block_simd<T, S>(rows[0]);
			return;
		}
		
		constexpr size_t row_blocks = r / e; 
		
		static_assert(c * e <= r, "Transpose requires cols * elements <= rows");
		static_assert(row_blocks <= c, "Transpose requires rows * elements <= cols");
		
		alignas(S::bytes) register_t temp[kernel::kernel_cols()][row_blocks];
		
		// Step 1: Transpose each e×e element subtile
		static_for<row_blocks>([&]<auto row_block>()
		{
			static_for<c>([&]<auto col_block>()
			{
				alignas(S::bytes) register_t subtile[e];
				
				static_for<e>([&]<auto i>()
				{
					subtile[i] = rows[row_block * e + i][col_block];
				});
				
				// Transpose the e×e element subtile in place
				__transpose_block_simd<T, S>(subtile);
				
				static_for<e>([&]<auto i>()
				{
					temp[col_block * e + i][row_block] = subtile[i];
				});
			});
		});
		
		// Step 2: Copy back from temp to rows
		static_for<c * e>([&]<auto i>()
		{
			static_for<row_blocks>([&]<auto j>()
			{
				rows[i][j] = temp[i][j];
			});
		});
	}

	/**
	 * \brief Transpose a matrix block using SIMD intrinsics.
	 *
	 * 
	 * \tparam T		The scalar data type of the matrix elements (e.g., float, double).
	 * \tparam S		The SIMD instruction set tag (e.g., SSE, AVX, AVX512).
	 * 
	 * \param A 		Pointer to the input matrix  A stored in row-major order.
	 * \param B			Pointer to the output matrix  B stored in row-major order.
	 * \param row		Row offset in A.
	 * \param col		Column offset in A.
	 * 
	 * \note Asymmetric matrices ( M !=  N) are supported without restrictions.
	 * \note If  M or  N are not multiples of the SIMD register width, the implementation
	 *       handles the remaining edge elements sequentially to ensure correctness.
	 * \note Using strides not aligned to SIMD register sizes may cause unaligned memory
	 *       accesses, potentially incurring performance penalties.
	 */
	template<typename T, typename S, template<typename, typename> class K>
	inline __attribute__((always_inline))
	void
	_transpose_block_simd(T** A, T** B, const size_t row, const size_t col)
	{
		using kernel = K<T, S>;
		using register_t = typename S::template register_t<T>;
		constexpr size_t rows = kernel::row_registers;
		constexpr size_t cols = kernel::col_registers;
		
		// Allocate 2D register array
		alignas(S::bytes) register_t registers[rows][cols];
		register_t* reg_ptrs[rows];

		for (size_t i = 0; i < rows; ++i)
			reg_ptrs[i] = registers[i];
		
		load<T, S, K>(A, reg_ptrs, row, col);
		transpose<T, S, K>(reg_ptrs);
		store<T, S, K>(B, reg_ptrs, col, row);
	}

	/**
	 * \brief Transpose a matrix using SIMD intrinsics.
	 *
	 * This function serves as the main entry point for SIMD-based block transposition.
	 * The SIMD instruction set is specified via the template parameter  S, allowing
	 * selection between SSE, AVX, AVX-512, or other SIMD types as implemented.
	 * 
	 * It efficiently transposes the matrix  A (with row-major layout) into  B, using
	 * blocked operations optimized for the selected SIMD register width.
	 * 
	 * \tparam T		The scalar data type of the matrix elements (e.g., float, double).
	 * \tparam S		The SIMD instruction set tag (e.g., SSE, AVX, AVX512).
	 * \tparam K		Kernel policy defining tile size.
	 * 
	 * \param A			 Pointer to the input matrix  A stored in row-major order.
	 * \param B			 Pointer to the output matrix  B stored in row-major order.
	 * \param M			 The rows of A and columns of B.
	 * \param N			 The columns of A and rows of B.
	 * 
	 * \note Asymmetric matrices ( M !=  N) are supported without restrictions.
	 * \note If  M or  N are not multiples of the SIMD register width, the implementation
	 *       handles the remaining edge elements sequentially to ensure correctness.
	 * \note Using strides not aligned to SIMD register sizes may cause unaligned memory
	 *       accesses, potentially incurring performance penalties.
	 */
	template<typename T, typename S, template<typename, typename> class K> 
	inline __attribute__((always_inline))
	void
	_transpose_simd(T** A, T** B, const size_t M, const size_t N)
	{
		using kernel = K<T, S>;
		using blocking = typename kernel::blocking;
		
		constexpr size_t l1_block = blocking::l1_block;
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;
		
		constexpr size_t tile_rows = kernel::kernel_rows();
		constexpr size_t tile_cols = kernel::kernel_cols();
		
		// Compute SIMD-processable dimensions (must be multiples of tile size)
		const size_t simd_M = M - (M % tile_rows);
		const size_t simd_N = N - (N % tile_cols);
		
		// Three-level cache blocking with micro-kernel tiling
		// - Parallel over I (M dimension): Each thread gets row panel of A
		// - L2 over J (N dimension): Column panel of A
		// - L1: Inner blocking for cache optimization
		
		#pragma omp parallel for schedule(static, l2_block)
		for (size_t i_block = 0; i_block < simd_M; i_block += l2_block)
		{
			size_t i_end = std::min(i_block + l2_block, simd_M);
			
			// L2 blocking: Over columns of A (N dimension)
			for (size_t j_block = 0; j_block < simd_N; j_block += l3_block)
			{
				size_t j_end = std::min(j_block + l3_block, simd_N);
				
				// L1 blocking: Further subdivide for L1 cache
				for (size_t j_l1 = j_block; j_l1 < j_end; j_l1 += l1_block)
				{
					size_t j_l1_end = std::min(j_l1 + l1_block, j_end);
					
					// Iterate over micro-kernel tiles within cache blocks
					for (size_t i = i_block; i < i_end; i += tile_rows)
					{
						for (size_t j = j_l1; j < j_l1_end; j += tile_cols)
						{
							_transpose_block_simd<T, S, K>(A, B, i, j);
						}
					}
				}
			}
		}
		
		// Handle remainder regions using scalar transpose_block
		const size_t rem_rows = M % tile_rows;
		const size_t rem_cols = N % tile_cols;
		
		// Region A: Bottom strip (remaining rows, SIMD-processed columns)
		if (rem_rows != 0 && simd_N > 0)
		{
			_transpose_block(A, B, simd_M, 0, rem_rows, simd_N);
		}
		
		// Region B: Right strip (SIMD-processed rows, remaining columns)
		if (rem_cols != 0 && simd_M > 0)
		{
			_transpose_block(A, B, 0, simd_N, simd_M, rem_cols);
		}
		
		// Region C: Corner (remaining rows and columns)
		if (rem_rows != 0 && rem_cols != 0)
		{
			_transpose_block(A, B, simd_M, simd_N, rem_rows, rem_cols);
		}
	}

	/**
	 * \brief Transpose a matrix using optimized SIMD and blocking algorithms.
	 *
	 * This is the main public interface for matrix transposition. It automatically
	 * selects between SIMD-optimized and standard blocked implementations based on
	 * the specified SIMD instruction set. The function operates on 2D matrix views
	 * (pointer-to-pointer arrays) with row-major layout.
	 *
	 * The transpose operation converts an M×N input matrix A into an N×M output
	 * matrix B, where B[j][i] = A[i][j]. The implementation uses cache-aware
	 * blocking and optional SIMD acceleration for optimal performance.
	 *
	 * \tparam T			Element type of the matrices (e.g., float, double).
	 * \tparam S			SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam K			Kernel policy defining tile size (default: transpose_kernel from simd.h).
	 *
	 * \param A		Source matrix of dimensions M×N in row-major layout.
	 * \param B		Destination matrix of dimensions N×M in row-major layout.
	 * \param M		Number of rows in matrix A (becomes number of columns in B).
	 * \param N		Number of columns in matrix A (becomes number of rows in B).
	 *
	 * \note Both matrices must be allocated as contiguous memory blocks accessible
	 *       through the 2D pointer interface. Submatrix views are not supported.
	 * \note Asymmetric matrices (M ≠ N) are fully supported.
	 * \note When S=NONE, the function uses standard blocked transpose without SIMD.
	 * \note SIMD implementations handle non-aligned dimensions with scalar fallback.
	 *
	 * \throws std::invalid_argument if either A or B is null.
	 * \throws std::runtime_error if memory layout validation fails.
	 */
	template<typename T,  typename S = decltype(detect_simd()), 
		template<typename, typename> class K = transpose_kernel>
	inline 
	void transpose(T** A, T** B, const size_t M, const size_t N)
	{
		right<T>("transpose:", std::make_tuple(A, M, N), std::make_tuple(B, N, M));
		
		if constexpr (std::is_same_v<S, NONE>)
			_transpose<T, K>(A, B, M, N);
		else
			_transpose_simd<T, S, K>(A, B, M, N);
	}

}//namespace damm
#endif //__TRANSPOSE_H__