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
#include <simd.h>
#include <damm_kernels.h>
#include <omp.h>
#include <transpose.h>

namespace damm
{
	/** 
	 * \brief kernal for _multiply. 
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
	 * \brief	Matrix multiplication.
	 *
	 * Performs blocked matrix multiplication of A × B = C using tiling, with inputs
	 * represented as T** (row major). Enables optional use of transposed B for performance.
	 *
	 * \tparam T	Scalar type (e.g., float or double)
	 * \tparam TR	If true, matrix B is assumed to be transposed
	 * \tparam K	Kernel policy definition
	 *
	 * \param A		Pointer to rows of matrix A, of shape M×N
	 * \param B		Pointer to rows of matrix B, of shape N×P (or P×N if TR=true)
	 * \param C		Pointer to rows of output matrix C, of shape M×P
	 * \param M		Number of rows in A and C
	 * \param N		Number of columns in A and rows in B
	 * \param P		Number of columns in B and C
	 *
	 * \note	TR=true enables multiplication with a transposed matrix B for improved memory access patterns.
	 * \note	Supports asymmetric dimensions.
	 */
	template <typename T, bool TR=false, template<typename, typename> class K>
	inline __attribute__((always_inline))
	void
	_multiply(T** A, T** B, T** C, const size_t M, const size_t N, const size_t P)
	{
		using kernel = K<T, NONE>;
		using blocking = typename kernel::blocking;
		
		constexpr size_t l1_block = blocking::l1_block;
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;
		
		#pragma omp parallel for schedule(static, l2_block)
		for (size_t i = 0; i < M; i += l2_block)
		{
			for (size_t j = 0; j < P; j += l3_block) 
			{
				for (size_t k = 0; k < N; k += l1_block)
				{
					size_t m = std::min(l2_block, M - i);
					size_t n = std::min(l1_block, N - k);
					size_t p = std::min(l3_block, P - j);
					_multiply_block<T, TR>(A, B, C, i, j, k, m, n, p);
				}
			}
		}
	}

	/**
	 * \brief SIMD multiply kernel for real types
	 */
	template<typename T, typename S, template<typename, typename> class K>
	requires (!std::is_same_v<T, std::complex<float>> && !std::is_same_v<T, std::complex<double>>)
	inline __attribute__((always_inline))
	void _multiply_block_simd(T** At, T** B, T** C,
		const size_t row, const size_t col, 
		const size_t k_start, const size_t k_end)
	{
		using kernel_t = K<T, S>;
		using register_t = typename S::template register_t<T>;
		
		constexpr size_t row_regs = kernel_t::row_registers;
		constexpr size_t col_regs = kernel_t::col_registers;
		constexpr size_t SIMD_WIDTH = S::template elements<T>();
		
		alignas(S::bytes) register_t c_accum[row_regs][col_regs];
		register_t* c_ptrs[row_regs];
		for (size_t i = 0; i < row_regs; ++i)
			c_ptrs[i] = c_accum[i];
		
		load<T, S, K>(C, c_ptrs, row, col);
		
		for (size_t k = k_start; k < k_end; ++k)
		{
			alignas(S::bytes) register_t b_vecs[col_regs];
			static_for<col_regs>([&]<auto j>() 
			{
				b_vecs[j] = _loadu<T, S>(&B[k][col + j * SIMD_WIDTH]);
			});
			
			static_for<row_regs>([&]<auto i>() 
			{
				register_t a_broadcast = _set1<T, S>(At[k][row + i]);
				
				static_for<col_regs>([&]<auto j>() 
				{
					c_accum[i][j] = _fmadd<T, S>(a_broadcast, b_vecs[j], c_accum[i][j]);
				});
			});
		}
		
		store<T, S, K>(C, c_ptrs, row, col);
	}

	// /**
	//  * \brief SIMD multiply kernel for complex types
	//  */
	template<typename T, typename S, template<typename, typename> class K>
	requires (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
	inline __attribute__((always_inline))
	void _multiply_block_simd(T** packed_A, T** B, T** C,
		const size_t row, const size_t col, 
		const size_t k_start, const size_t k_end)
	{
		using kernel_t = K<T, S>;
		using real_t = typename base<T>::type;
		using register_t = typename S::template register_t<real_t>;
		
		constexpr size_t row_regs = kernel_t::row_registers;
		constexpr size_t col_regs = kernel_t::col_registers;
		constexpr size_t SIMD_WIDTH = S::template elements<real_t>();
		
		auto* packed_A_real = reinterpret_cast<real_t**>(packed_A);
		auto* B_real = reinterpret_cast<real_t**>(B);
		auto* C_real = reinterpret_cast<real_t**>(C);
		
		const size_t col_real = col * 2;
		
		register_t sign_mask = alternating_sign_mask_odd<real_t, S>();
		
		alignas(S::bytes) register_t c_accum[row_regs][col_regs];
		static_for<row_regs>([&]<auto i>()
		{
			static_for<col_regs>([&]<auto j>()
			{
				c_accum[i][j] = _loadu<real_t, S>(&C_real[row + i][col_real + j * SIMD_WIDTH]);
			});
		});
		
		for (size_t k = k_start; k < k_end; ++k)
		{
			alignas(S::bytes) register_t b_vecs[col_regs];
			alignas(S::bytes) register_t b_swapped[col_regs];
			
			static_for<col_regs>([&]<auto j>() 
			{
				b_vecs[j] = _loadu<real_t, S>(&B_real[k][col_real + j * SIMD_WIDTH]);
				b_swapped[j] = swap_adjacent_pairs<real_t, S>(b_vecs[j]);
			});
			
			static_for<row_regs>([&]<auto i>() 
			{
				const size_t a_idx = (row + i) * 2;
				const real_t a_r = packed_A_real[k][a_idx + 0];
				const real_t a_i = packed_A_real[k][a_idx + 1];
				
				register_t a_r_vec = _set1<real_t, S>(a_r);
				register_t a_i_vec = _set1<real_t, S>(a_i);
				register_t a_i_signed = _mul<real_t, S>(a_i_vec, sign_mask);
				
				static_for<col_regs>([&]<auto j>() 
				{
					c_accum[i][j] = _fmadd<real_t, S>(a_r_vec, b_vecs[j], c_accum[i][j]);
					c_accum[i][j] = _fmadd<real_t, S>(a_i_signed, b_swapped[j], c_accum[i][j]);
				});
			});
		}
		
		static_for<row_regs>([&]<auto i>()
		{
			static_for<col_regs>([&]<auto j>()
			{
				_storeu<real_t, S>(&C_real[row + i][col_real + j * SIMD_WIDTH], c_accum[i][j]);
			});
		});
	}

	
	/**
	 * \brief	Matrix multiplication.
	 *
	 * Performs simd matrix multiplication of A × B = C using tiling, with inputs
	 * represented as T** in row major.
	 *
	 * \tparam T	Scalar type (e.g., float or double)
	 * \tparam S	SIMD type
	 * \tparam K	Kernel policy definition
	 *
	 * \param A		Pointer to rows of matrix A, of shape N×M
	 * \param B		Pointer to rows of matrix B, of shape N×P
	 * \param C		Pointer to rows of output matrix C, of shape M×P
	 * \param M		Number of rows in A and C
	 * \param N		Number of columns in A and rows in B
	 * \param P		Number of columns in B and C
	 *
	 * \note    Requires A as the transpose of A to be multiplied 
	 * \note	Supports asymmetric dimensions and non-multiple block sizes.
	 */

	template<typename T, typename S, template<typename, typename> class K> 
	inline __attribute__((always_inline))
	void
	_multiply_simd(T** A, T** B, T** C, const size_t M, const size_t N, const size_t P)
	{
		using kernel_t = K<T, S>;
		using blocking = typename kernel_t::blocking;

		constexpr size_t l1_block = blocking::l1_block;
		constexpr size_t l2_block = blocking::l2_block;
		constexpr size_t l3_block = blocking::l3_block;

		constexpr size_t kernel_rows = kernel_t::kernel_rows();
		constexpr size_t kernel_cols = kernel_t::kernel_cols();

		const size_t simd_M = M - (M % kernel_rows);
		const size_t simd_N = N - (N % kernel_rows);
		const size_t simd_P = P - (P % kernel_cols);
	
		#pragma omp parallel
		{			

			#pragma omp for schedule(static, l3_block) nowait
			for (size_t j_block = 0; j_block < simd_P; j_block += l3_block)
			{
				size_t j_end = std::min(j_block + l3_block, simd_P);
				
				for (size_t i_block = 0; i_block < simd_M; i_block += l2_block)
				{
					size_t i_end = std::min(i_block + l2_block, simd_M);
					
					for (size_t k_block = 0; k_block < simd_N; k_block += l1_block)
					{
						size_t k_end = std::min(k_block + l1_block, simd_N);
						
						const size_t panel_M = i_end - i_block;
						const size_t panel_N = k_end - k_block;

						
						for (size_t i = 0; i < panel_M; i += kernel_rows)
						{
							for (size_t j = 0; j < (j_end - j_block); j += kernel_cols)
							{
								_multiply_block_simd<T, S, K>(
											A, B, C,
											i_block + i,      // row
											j_block + j,      // col
											k_block, k_end    // k_start, k_end
								);
							}
						}
					}
				}
			}
		}

		
		// Handle remainder
		const size_t rem_rows = M % kernel_rows;
		const size_t rem_cols = P % kernel_cols;
		const size_t rem_inner = N % kernel_rows;

		if (rem_inner != 0 && simd_M > 0 && simd_P > 0)
		{
			_multiply_block<T, false>(A, B, C, 0, 0, simd_N, simd_M, rem_inner, simd_P);
		}

		if (rem_cols != 0 && simd_M > 0)
		{
			_multiply_block<T, false>(A, B, C, 0, simd_P, 0, simd_M, N, rem_cols);
		}

		if (rem_rows != 0 && simd_P > 0)
		{
			_multiply_block<T, false>(A, B, C, simd_M, 0, 0, rem_rows, N, simd_P);
		}
		
		if (rem_rows != 0 && rem_cols != 0)
		{
			_multiply_block<T, false>(A, B, C, simd_M, simd_P, 0, rem_rows, N, rem_cols);
		}
	}

	/**
	 * \brief Perform optimized matrix multiplication using SIMD and blocking algorithms.
	 *
	 * This is the main public interface for matrix multiplication A × B = C.
	 *
	 * \tparam T	Element type of the matrices (e.g., float, double).
	 * \tparam S	SIMD instruction set to use (SSE, AVX, AVX512, or NONE).
	 * \tparam K	Kernel policy defining tile size (default: multiply_kernel from simd.h).
	 
	 * \param A		Left operand matrix of dimensions M×N in row-major layout.
	 * \param B		Right operand matrix of dimensions N×P in row-major layout.
	 * \param C		Result matrix of dimensions M×P in row-major layout.
	 * \param M		Number of rows in matrix A (and result matrix C).
	 * \param N		Number of columns in matrix A and rows in matrix B (inner dimension).
	 * \param P		Number of columns in matrix B (and result matrix C).
	 *
	 * \note Matrix C should be zero-initialized before calling this function,
	 *       as the implementation uses += operations internally (accumulation mode).
	 */
	template<typename T, typename S = decltype(detect_simd()), template<typename, typename> class K = multiply_kernel>
	inline 
	void 
	multiply(T** A, T** B, T** C, const size_t M, const size_t N, const size_t P)
	{
		right<T>("multiply:", 
			std::make_tuple(A, M, N), 
			std::make_tuple(B, N, P), 
			std::make_tuple(C, M, P));

		auto At = aligned_alloc_2D<T, S::bytes>(N, M);
		transpose<T, S>(A, At.get(), M, N);

		if constexpr (std::is_same_v<S, NONE>) 
			_multiply<T, true, K>(At.get(), B, C, M, N, P);
		else
			_multiply_simd<T, S, K>(At.get(), B, C, M, N, P);
	}

}//namespace damm

#endif //__MULTIPLY_H__