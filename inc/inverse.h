#ifndef __INVERSE_H__
#define __INVERSE_H__
/**
 * \file inverse.h
 * \brief definitions for matrix inversion utilities 
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
#include <cmath>
#include <common.h>
#include <damm_memory.h>
#include <decompose.h>
#include <solve.h>
#include <broadcast.h>
#include <multiply.h>
#include <transpose.h>

namespace damm
{	
	namespace tri
	{
		/**
		 * \brief Triangular matrix inversion.
		 *
		 * \param A         Input triangular matrix (N×N)
		 * \param B         Output inverse matrix (N×N)
		 * \param N         Matrix dimension
		 * \param unit_diag If true, assumes unit diagonal
		 */
		template <typename T, typename S = decltype(detect_simd()), TRIANGULAR UL = TRIANGULAR::UPPER>
		void inverse(T** A, T** B, const size_t N, bool unit_diag = false)
		{
			// Allocate aligned vectors
			auto y_mat = aligned_alloc_2D<T, S::bytes>(1, N);
			auto x_mat = aligned_alloc_2D<T, S::bytes>(1, N);
			T* y = y_mat[0];
			T* x = x_mat[0];

			std::fill(y, y + N, T(0));

			for (size_t col = 0; col < N; ++col)
			{
				y[col] = T(1);

				if constexpr(UL == TRIANGULAR::UPPER)
				{
					// Solve U x = y by backward substitution
					backward_substitution<T, S>(A, y, x, N, unit_diag);
				}
				else if constexpr(UL == TRIANGULAR::LOWER)
				{
					// Solve L x = y by forward substitution
					forward_substitution<T, S>(A, y, x, N, unit_diag);
				}

				for (size_t i = 0; i < N; ++i)
					B[i][col] = x[i];

				y[col] = T(0);
			}
		}
	}


	/**
	 * \brief Matrix Inversion - SIMD aware inversion algorithms.
	 *
	 * Provides efficient matrix inversion routines using SIMD-optimized decomposition
	 * and solve operations. 
	 *
	 */
	namespace lu
	{
		/**
		 * \brief LU-based Matrix Inversion using SIMD-optimized operations.
		 *
		 * Computes the inverse of matrix A using LU decomposition with partial pivoting.
		 * The algorithm solves A * X = I by decomposing A = P * L * U and then solving
		 * L * Y = P and U * X = Y for each column of the identity matrix.
		 *
		 * \tparam T        Scalar type (float or double)
		 * \tparam S        SIMD instruction set (SSE, AVX, AVX512)
		 *
		 * \param A         Input matrix A (N×N), will be overwritten with LU factors
		 * \param A_inv     Output inverse matrix A^(-1) (N×N)
		 * \param N         Matrix dimension
		 * 
		 * \return true if inversion successful, false if matrix is singular
		 *
		 * \note Matrix A is modified during computation (contains LU factors on exit)
		 */
		template<typename T, typename S = decltype(detect_simd())>
		inline bool
		inverse(T** A, T** A_inv, const size_t N)
		{
			right<T>("inverse: ", std::make_tuple(A, N, N), std::make_tuple(A_inv, N, N));
			
			// Perform LU decomposition
			auto P_mat = aligned_alloc_2D<size_t, S::bytes>(1, N);
			size_t* P = P_mat[0];
			
			if (!lu::decompose<T, S>(A, P, N))
				return false; // Matrix is singular
	
			// Column-wise inversion by solving A x = e_col
			for (size_t col = 0; col < N; col++)
			{
				// Allocate aligned vectors for this column solve
				auto b_mat = aligned_alloc_2D<T, S::bytes>(1, N);
				auto y_mat = aligned_alloc_2D<T, S::bytes>(1, N);
				auto x_mat = aligned_alloc_2D<T, S::bytes>(1, N);
				T* b = b_mat[0];
				T* y = y_mat[0];
				T* x = x_mat[0];
				
				std::fill(b, b + N, T(0));
				
				// Set up RHS with permutation
				b[P[col]] = T(1);

				// Solve L * y = b (L has unit diagonal from LU decomposition)
				tri::forward_substitution<T, S>(A, b, y, N, true);

				// Solve U * x = y
				tri::backward_substitution<T, S>(A, y, x, N, false);

				// Store solution in column of A_inv
				for (size_t i = 0; i < N; ++i)
					A_inv[i][col] = x[i];
			}
			
			return true;
		}

	} // namespace lu

	namespace qr
	{
		/**
		 * \brief QR-based Matrix Inversion/Pseudoinverse using SIMD-optimized operations.
		 *
		 * Computes the inverse (for square matrices) or Moore-Penrose pseudoinverse 
		 * (for rectangular matrices) using QR decomposition with Householder reflections.
		 * This method naturally handles overdetermined/underdetermined systems.
		 *
		 * \tparam T        Scalar type (float or double)
		 * \tparam S        SIMD instruction set (SSE, AVX, AVX512, or NONE)
		 *
		 * \param A         Input matrix A (M×N) as 2D pointer array, overwritten with QR factors
		 * \param A_inv     Output inverse/pseudoinverse matrix (N×M) as 2D pointer array
		 * \param M         Number of rows in A
		 * \param N         Number of columns in A
		 * 
		 * \return true if inversion successful, false if matrix is rank deficient
		 *
		 * \note For square matrices (M=N): computes regular inverse A^(-1)
		 * \note For overdetermined (M>N): computes left pseudoinverse A^+ = (A^T*A)^(-1)*A^T
		 * \note For underdetermined (M<N): computes right pseudoinverse A^+ = A^T*(A*A^T)^(-1)
		 * \note Matrix A is modified during computation (contains QR factors on exit)
		 * \note Uses SIMD-optimized QR decomposition from decompose.h
		 * \note Better numerical properties than LU for ill-conditioned matrices
		 *
		 * \warning Fails if matrix is rank deficient (returns false)
		 * \warning More computationally expensive than LU for well-conditioned square matrices
		 */
		template<typename T, typename S = decltype(detect_simd())>
		inline bool
		inverse(T** A, T** A_inv, const size_t M, const size_t N)
		{
			right<T>("inverse:", std::make_tuple(A, M, N), std::make_tuple(A_inv, N, M));
			
			// Allocate Q and R matrices for QR decomposition
			auto Q = aligned_alloc_2D<T, S::bytes>(M, M);
			auto R = aligned_alloc_2D<T, S::bytes>(M, N);
			
			// Perform QR decomposition: A = Q * R
			if (!damm::qr::decompose<T, S>(A, Q.get(), R.get(), M, N))
				return false; // QR decomposition failed
			
			const size_t min_dim = std::min(M, N);
			constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;
			
			// Check for rank deficiency by examining diagonal of R
			for (size_t i = 0; i < min_dim; ++i) 
			{
				if (std::abs(R[i][i]) < tolerance)
					return false; // Rank deficient - cannot invert
			}
			
			if (M >= N) 
			{
				// Overdetermined or square case: A^+ = R^(-1) * Q^T
				
				// Extract the square upper part of R (N×N)
				auto R_square = aligned_alloc_2D<T, S::bytes>(N, N);
				for (size_t i = 0; i < N; ++i)
					std::copy(R[i], R[i] + N, R_square[i]);
				
				// Invert the upper triangular R matrix
				auto R_inv = aligned_alloc_2D<T, S::bytes>(N, N);
				tri::inverse<T, S, TRIANGULAR::UPPER>(R_square.get(), R_inv.get(), N);
				
				// Compute Q^T by transposing first N columns of Q: (M×N) -> (N×M)
				auto Q_sub = aligned_alloc_2D<T, S::bytes>(M, N);
				for (size_t i = 0; i < M; ++i)
					std::copy(Q[i], Q[i] + N, Q_sub[i]);
				
				auto Q_T = aligned_alloc_2D<T, S::bytes>(N, M);
				transpose<T, S>(Q_sub.get(), Q_T.get(), M, N);
				
				// Compute pseudoinverse: A^+ = R^(-1) * Q^T
				// Dimensions: (N×N) × (N×M) = (N×M)
				zeros<T, S>(A_inv, N, M);
				multiply<T, S>(R_inv.get(), Q_T.get(), A_inv, N, N, M);
			} 
			else 
			{
				// Underdetermined case: A^+ = Q * R^(-1)
				
				// Extract the square upper part of R (M×M)
				auto R_square = aligned_alloc_2D<T, S::bytes>(M, M);
				for (size_t i = 0; i < M; ++i)
					std::copy(R[i], R[i] + M, R_square[i]);
				
				// Invert the upper triangular R matrix
				auto R_inv = aligned_alloc_2D<T, S::bytes>(M, M);
				tri::inverse<T, S, TRIANGULAR::UPPER>(R_square.get(), R_inv.get(), M);
				
				// Compute A^+ = Q * R^(-1)
				// Dimensions: (M×M) × (M×M) = (M×M)
				auto temp = aligned_alloc_2D<T, S::bytes>(M, M);
				zeros<T, S>(temp.get(), M, M);
				multiply<T, S>(Q.get(), R_inv.get(), temp.get(), M, M, M);
				
				// Extract first N rows as output: (N×M)
				for (size_t i = 0; i < N; ++i)
					std::copy(temp[i], temp[i] + M, A_inv[i]);
			}
			
			return true;
		}

	} // namespace qr

} //namespace damm

#endif //__INVERSE_H__