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
#include <decompose.h>
#include <solve.h>
#include <broadcast.h>
#include <multiply.h>
#include <transpose.h>


namespace damm
{

	/**
	 * \brief Inversion Policy for selecting matrix inversion method.
	 */
	enum class InversePolicy 
	{
		LU,    ///< LU-based inversion using forward/backward substitution
		QR     ///< QR-based inversion/pseudoinverse using orthogonal decomposition
	};
	
	namespace tri
	{
		template <typename T, SIMD S, TRIANGULAR UL>
		void inverse(T** A, T** B, const size_t N, bool unit_diag = false)
		{
			auto y = aligned_alloc_1D<T, S>(N, 1);
			auto x = aligned_alloc_1D<T, S>(N, 1);

			zeros<T, NONE>(y.get(), N, 1);

			for (size_t col = 0; col < N; ++col)
			{
				y[col] = T(1);

				if constexpr(UL == TRIANGULAR::UPPER)
				{
					// Solve U x = y by backward substitution
					backward_substitution<T, S>(A, y.get(), x.get(), N, unit_diag);
				}
				if constexpr(UL == TRIANGULAR::LOWER)
				{
					// Solve L x = y by forward substitution
					forward_substitution<T, S>(A, y.get(), x.get(), N, unit_diag);
				}

				for (size_t i = 0; i < N; ++i)
					B[i][col] = x[i];

				y[col] = T(0);
			}
		}
	}


	/**
	 * \brief Matrix Inversion Operations - Policy-driven SIMD-optimized matrix inversion
	 *
	 * Provides efficient matrix inversion routines using SIMD-optimized decomposition
	 * and solve operations with configurable inversion method via InversePolicy.
	 *
	 * \section inverse_policies Inversion Policies
	 *
	 * \subsection lu_inverse_policy LU Policy
	 * **LU-based Matrix Inversion** - Computes A^(-1) using LU decomposition
	 *
	 * **Algorithm:**
	 * 1. Decompose A = P * L * U
	 * 2. Solve L * Y = P for each column of identity matrix
	 * 3. Solve U * X = Y for each column to get A^(-1)
	 *
	 * **QR-based Matrix Inversion/Pseudoinverse** - Computes A^(-1) or A^+ using QR decomposition
	 *
	 * **Algorithm:**
	 * 1. Decompose A = Q * R  
	 * 2. For square matrices: A^(-1) = R^(-1) * Q^T
	 * 3. For overdetermined systems: A^+ = R^(-1) * Q^T (pseudoinverse)
	 *
	 * **Usage Pattern:**
	 * \code{.cpp}
	 * auto A_inv = aligned_alloc_2D<double, AVX>(N, N);
	 * bool success = inverse_block_simd<InversePolicy::QR, double, AVX>(A, A_inv.get(), M, N);
	 * \endcode
	 *
	 * \note All operations support float/double types and square matrices only.
	 * \note Choose SIMD level (SSE/AVX/AVX512) based on target hardware for optimal performance.
	 * \note Uses decompose.h LU operations and SIMD-optimized solve routines.
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
		 * \tparam threads  Number of threads for parallel execution
		 *
		 * \param A         Input matrix A (N×N), will be overwritten with LU factors
		 * \param A_inv     Output inverse matrix A^(-1) (N×N)
		 * \param N         Matrix dimension
		 * 
		 * \return true if inversion successful, false if matrix is singular
		 *
		 * \note Matrix ,SA is modified during computation (contains LU factors on exit)
		 * \note Uses parallel computation for multiple right-hand sides
		 */
		template<typename T, SIMD S, const size_t threads = _threads>
		inline bool
		inverse(T** A, T** A_inv, const size_t N)
		{
			right<T>("inverse: ", std::make_tuple(A, N, N), std::make_tuple(A_inv, N, N));
			
			// Perform LU decomposition
			auto P = aligned_alloc_1D<size_t, S>(N, 1);
			
			if (!lu::decompose<T, S, threads>(A, P.get(), N))
				return false; // Matrix is singular
	
			// possible parallel inversion by solving A x = e_col (i.e., column-wise)
			// parallel_for(0, N, 1,
			// 	[&](size_t col)
			// 	{
				for (size_t col = 0; col < N; col++)
				{
					auto b = aligned_alloc_1D<T, S>(N, 1);
					auto y = aligned_alloc_1D<T, S>(N, 1);
					auto x = aligned_alloc_1D<T, S>(N, 1);

					zeros<T, NONE>(b.get(), N, 1);
					
					b[P[col]] = T(1);

					// Solve L * y = b
					tri::forward_substitution<T, S>(A, b.get(), y.get(), N, /*unit_diag=*/true);

					// Solve U * x = y
					tri::backward_substitution<T, S>(A, y.get(), x.get(), N, /*unit_diag=*/false);

					for (size_t i = 0; i < N; ++i)
						A_inv[i][col] = x[i];
				}
				// }, 1);

			
			return true;
		}

		/**
		 * \brief LU-based Matrix Inversion with flat array interface.
		 */
		template<typename T, SIMD S, const size_t threads = _threads>
		inline bool
		inverse(T* A, T* A_inv, const size_t N)
		{
			auto A_view = view_as_2D(A, N, N);
			auto A_inv_view = view_as_2D(A_inv, N, N);
			return inverse<T, S, threads>(A_view.get(), A_inv_view.get(), N);
		}

	} // namespace lu

	namespace qr
	{
	

		/**
		 * \brief QR-based Matrix Inversion/Pseudoinverse using SIMD-optimized operations.
		 *
		 * Computes the inverse (for square matrices) or Moore-Penrose pseudoinverse 
		 * (for rectangular matrices) using QR decomposition. This method provides
		 * better numerical stability for certain matrices and naturally handles
		 * overdetermined systems.
		 *
		 * \tparam T        Scalar type (float or double)
		 * \tparam S        SIMD instruction set (SSE, AVX, AVX512)
		 * \tparam threads  Number of threads for parallel execution
		 *
		 * \param A         Input matrix A (M×N), will be overwritten with QR factors
		 * \param A_inv     Output inverse/pseudoinverse matrix (N×M)
		 * \param M         Number of rows in A
		 * \param N         Number of columns in A
		 * 
		 * \return true if inversion successful, false if matrix is rank deficient
		 *
		 * \note For square matrices (M=N): computes regular inverse A^(-1)
		 * \note For overdetermined systems (M>N): computes pseudoinverse A^+ = (A^T*A)^(-1)*A^T
		 * \note For underdetermined systems (M<N): computes pseudoinverse A^+ = A^T*(A*A^T)^(-1)
		 * \note Matrix A is modified during computation (contains QR factors on exit)
		 */
		template<typename T, SIMD S, const size_t threads = _threads>
		inline bool
		inverse(T** A, T** A_inv, const size_t M, const size_t N)
		{
			right<T>("inverse: ", std::make_tuple(A, M, N), std::make_tuple(A_inv, M, N));
			
			// Allocate Q and R matrices
			auto Q = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, M);
			auto R = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, N);
			
			// Perform QR decomposition
			if (!damm::qr::decompose<T, S, threads>(A, Q.get(), R.get(), M, N))
				return false; // QR decomposition failed
			
			const size_t min_dim = std::min(M, N);
			constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;
			
			// Check for rank deficiency by examining diagonal of R
			for (size_t i = 0; i < min_dim; ++i) 
			{
				if (std::abs(R[i][i]) < tolerance)
					return false; // Rank deficient
			}
			
			if (M >= N) 
			{
				// Overdetermined case: A^+ = R^(-1) * Q^T
				// Extract the square upper part of R
				auto R_inv = aligned_alloc_2D<T, static_cast<size_t>(S)>(N, N);
								
				// Invert the square R matrix
				tri::inverse<T, S, UPPER>(R.get(), R_inv.get(), N);
				
				// Compute A^+ = R^(-1) * Q^T
				// First compute Q^T (transpose first N columns of Q)
				auto Q_T = aligned_alloc_2D<T, static_cast<size_t>(S)>(N, M);
				
				// Extract first N columns of Q and transpose
				auto Q_sub = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, N);
						
				transpose<T, S>(Q.get(), Q_T.get(), M, N);
				
				// Then multiply: A^+ = R^(-1) * Q^T
				multiply<T, S>(R_inv.get(), Q_T.get(), A_inv, N, N, M);
				
			} 
			else 
			{
				// Underdetermined case: A^+ = Q * R^(-1)
				auto R_inv = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, M);
					
				// Invert R
				tri::inverse<T, S, UPPER>(R.get(), R_inv.get(), M);
				
				// Compute A^+ = Q * R^(-1)
				multiply<T, S>(Q.get(), R_inv.get(), A_inv, M, M, M);
			}
			
			return true;
		}

		/**
		 * \brief QR-based Matrix Inversion with flat array interface.
		 */
		template<typename T, SIMD S, const size_t threads = _threads>
		inline bool
		inverse(T* A, T* A_inv, const size_t M, const size_t N)
		{
			auto A_view = view_as_2D(A, M, N);
			auto A_inv_view = view_as_2D(A_inv, N, M);
			return inverse<T, S, threads>(A_view.get(), A_inv_view.get(), M, N);
		}

	} // namespace qr

} //namespace damm

#endif //__INVERSE_H__