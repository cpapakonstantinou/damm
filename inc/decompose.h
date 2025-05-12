#ifndef __DECOMPOSE_H__
#define __DECOMPOSE_H__
/**
 * \file decompose.h
 * \brief definitions for matrix decomposition utilities 
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

#include <vector>
#include <algorithm>
#include <ranges>
#include <cmath>
#include <common.h>
#include <fused_union.h>
#include <union.h>
#include <reduce.h>
#include <multiply.h>
#include <transpose.h>
#include <householder.h>

namespace damm
{

/**
 * \brief Decomposition Policy for selecting matrix decomposition method.
 */
enum class DecomposePolicy 
{
	LU,    ///< LU decomposition with partial pivoting
	QR     ///< QR decomposition using Householder reflections
};

/**
 * \brief Matrix Decomposition Operations - Policy-driven SIMD-optimized decompositions
 *
 * Provides efficient matrix decomposition routines using SIMD-optimized operations
 * with configurable decomposition method via DecomposePolicy.
 *
 * \section decompose_policies Decomposition Policies
 *
 * \subsection lu_policy LU Policy
 * **LU Decomposition with Partial Pivoting** - Decomposes A into P*A = L*U
 *
 * **Usage Pattern:**
 * \code{.cpp}
 * std::vector<size_t> P;
 * bool success = decompose_block_simd<DecomposePolicy::LU, double, AVX>(A, P, N);
 * \endcode
 *
 * \subsection qr_policy QR Policy  
 * **QR Decomposition using Householder Reflections** - Decomposes A into A = Q*R
 *
 * **Usage Pattern:**
 * \code{.cpp}
 * auto Q = aligned_alloc_2D<float, AVX>(M, M);
 * auto R = aligned_alloc_2D<float, AVX>(M, N);
 * bool success = decompose_block_simd<DecomposePolicy::QR, float, AVX512>(A, Q.get(), R.get(), M, N);
 * \endcode
 *
 * \note All operations support float/double types and asymmetric matrices (M ≠ N).
 * \note Choose SIMD level (SSE/AVX/AVX512) based on target hardware for optimal performance.
 * \note Uses fused_union operations for optimal SIMD elimination patterns.
 */

namespace lu
{
	/** 
	 * \brief kernal for LU pivot search. 
	 * Low level function not intended for the public API.
	 */
	template <typename T>
	inline __attribute__((always_inline))
	size_t
	_find_pivot(T** A, const size_t k, const size_t N)
	{
		size_t pivot_row = k;
		T max_val = std::abs(A[k][k]);
		
		for (size_t i = k + 1; i < N; ++i) 
		{
			T abs_val = std::abs(A[i][k]);
			if (abs_val > max_val) 
			{
				max_val = abs_val;
				pivot_row = i;
			}
		}
		return pivot_row;
	}

	/** 
	 * \brief kernal for LU elimination step using fused_union. 
	 * Low level function not intended for the public API.
	 */
	template <typename T, SIMD S>
	inline __attribute__((always_inline))
	void
	_eliminate_row(T* current_row, T* pivot_row, const T multiplier, const size_t remaining_cols)
	{
		if (remaining_cols == 0) return;
		
		auto temp_row = aligned_alloc_1D<T, static_cast<size_t>(S)>(1, remaining_cols);
		
		scalar::fused_union<FusionPolicy::FUSION_FIRST, T, 
			std::plus<>, std::multiplies<>, S, _block_size, 1>(
				current_row, pivot_row, -multiplier, temp_row.get(), 1, remaining_cols);
		
		std::copy(temp_row.get(), temp_row.get() + remaining_cols, current_row);
	}

	/**
	 * \brief LU Decomposition with Partial Pivoting using SIMD-optimized operations.
	 *
	 * Performs in-place LU decomposition of matrix A with partial pivoting using SIMD-optimized
	 * elimination steps via fused_union operations. Optimized multiplier computation using
	 * scalar broadcasting for better memory efficiency.
	 *
	 * \tparam T        Scalar type (float or double)
	 * \tparam S        SIMD instruction set (SSE, AVX, AVX512)
	 * \tparam threads  Number of threads for parallel execution
	 *
	 * \param A         Input matrix A (N×N), overwritten with L and U
	 * \param P         Output permutation vector (size N)
	 * \param N         Matrix dimension
	 * 
	 * \return true if decomposition successful, false if matrix is singular
	 */
	template<typename T, SIMD S, const size_t threads = _threads>
	inline bool
	decompose(T** A, size_t* P, const size_t N)
	{		
		right<T>("decompose:", std::make_tuple(A, N, N));
			
		constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;

		std::ranges::iota(P, P + N, 0);
		
		for (size_t k = 0; k < N - 1; ++k) 
		{
			size_t pivot_row = _find_pivot(A, k, N);

			if (std::abs(A[pivot_row][k]) < tolerance) 
				return false; // Matrix is singular
			
			if (pivot_row != k) 
			{
				std::swap(A[k], A[pivot_row]);
				std::swap(P[k], P[pivot_row]);
			}
			
			const T pivot = A[k][k];
			const size_t remaining_rows = N - k - 1;
			const size_t remaining_cols = N - k - 1;
			
			if (remaining_rows == 0 || remaining_cols == 0) 
				continue;
			
			auto column_k = aligned_alloc_2D<T, static_cast<size_t>(S)>(remaining_rows, 1);
			for (size_t i = 0; i < remaining_rows; ++i)
				column_k[i][0] = A[k + 1 + i][k];
			
			auto pivot_matrix = aligned_alloc_2D<T, static_cast<size_t>(S)>(remaining_rows, 1);
			for (size_t i = 0; i < remaining_rows; ++i)
				pivot_matrix[i][0] = pivot;
			
			auto multipliers = aligned_alloc_2D<T, static_cast<size_t>(S)>(remaining_rows, 1);
			matrix::unite<T, std::divides<>, S>(column_k.get(), pivot_matrix.get(), multipliers.get(), remaining_rows, 1);
			
			for (size_t i = 0; i < remaining_rows; ++i)
				A[k + 1 + i][k] = multipliers[i][0];
			
			parallel_for(k + 1, N, 1, 
				[&](size_t i) 
				{
					const T multiplier = A[i][k];
					_eliminate_row<T, S>(&A[i][k + 1], &A[k][k + 1], multiplier, remaining_cols);
				}, threads);
		}
		
		return true;
	}

	/**
	 * \brief LU Decomposition with flat array interface.
	 */
	template<typename T, SIMD S = AVX, const size_t threads = _threads>
	inline bool
	decompose(T* A, size_t* P, const size_t N)
	{
		auto A_view = view_as_2D(A, N, N);
		return decompose_simd<T, S, threads>(A_view.get(), P, N);
	}

} // namespace lu

namespace qr
{
	/**
	 * \brief QR Decomposition using Householder Reflections with SIMD optimization.
	 *
	 * Performs QR decomposition of matrix A using Householder reflections.
	 * Computes A = Q * R where Q is orthogonal and R is upper triangular.
	 * Uses the optimized householder methods from householder.h.
	 *
	 * \tparam T        Scalar type (float or double)
	 * \tparam S        SIMD instruction set (SSE, AVX, AVX512)
	 * \tparam threads  Number of threads for parallel execution
	 *
	 * \param A         Input matrix A (M×N), overwritten with R
	 * \param Q         Output orthogonal matrix Q (M×M)
	 * \param R         Output upper triangular matrix R (M×N)
	 * \param M         Number of rows
	 * \param N         Number of columns
	 * 
	 * \return true if decomposition successful, false if matrix is rank deficient
	 */
	template<typename T, SIMD S, const size_t threads = _threads>
	inline bool
	decompose(T** A, T** Q, T** R, const size_t M, const size_t N)
	{
		right<T>("decompose:", 
			std::make_tuple(A, M, N),
			std::make_tuple(Q, M, M),
			std::make_tuple(R, M, N));
		
		set_identity(Q, M, M);
		
		for (size_t i = 0; i < M; ++i)
			std::copy(A[i], A[i] + N, R[i]);
		
		const size_t min_dim = std::min(M, N);
		auto householder_vector = aligned_alloc_1D<T, static_cast<size_t>(S)>(M, 1);
		
		for (size_t k = 0; k < min_dim; ++k) 
		{
			const size_t remaining_rows = M - k;
			
			if (remaining_rows <= 1)
				continue;
			
			auto column_k = aligned_alloc_1D<T, static_cast<size_t>(S)>(remaining_rows, 1);
			for (size_t i = 0; i < remaining_rows; ++i)
				column_k[i] = R[k + i][k];
			
			T tau, beta;
			make_householder<T, S>(column_k.get(), remaining_rows, householder_vector.get(), tau, beta);
			
			if (std::abs(tau) < std::numeric_limits<T>::epsilon())
				continue; // Skip near-zero reflections
			
			R[k][k] = beta;
			for (size_t i = k + 1; i < M; ++i)
				R[i][k] = T(0);
			
			if (k + 1 < N)
			{
				auto R_sub = aligned_alloc_2D<T, static_cast<size_t>(S)>(remaining_rows, N - k - 1);
				for (size_t i = 0; i < remaining_rows; ++i)
					std::copy(&R[k + i][k + 1], &R[k + i][N], R_sub[i]);
				
				apply_householder_left<T, S>(R_sub.get(), remaining_rows, N - k - 1, 
											householder_vector.get(), tau);
				
				for (size_t i = 0; i < remaining_rows; ++i)
					std::copy(R_sub[i], R_sub[i] + (N - k - 1), &R[k + i][k + 1]);
			}
			
			auto Q_sub = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, remaining_rows);
			for (size_t i = 0; i < M; ++i)
				std::copy(&Q[i][k], &Q[i][M], Q_sub[i]);
			
			apply_householder_right<T, S>(Q_sub.get(), M, remaining_rows, 
										 householder_vector.get(), tau);
			
			for (size_t i = 0; i < M; ++i)
				std::copy(Q_sub[i], Q_sub[i] + remaining_rows, &Q[i][k]);
		}
		
		return true;
	}

	/**
	 * \brief QR Decomposition with flat array interface.
	 */
	template<typename T, SIMD S, const size_t threads = _threads>
	inline bool
	decompose(T* A, T* Q, T* R, const size_t M, const size_t N)
	{
		auto A_view = view_as_2D(A, M, N);
		auto Q_view = view_as_2D(Q, M, M);
		auto R_view = view_as_2D(R, M, N);
		return decompose<T, S, threads>(A_view.get(), Q_view.get(), R_view.get(), M, N);
	}
	
} // namespace qr
} //namespace damm

#endif //__DECOMPOSE_H__