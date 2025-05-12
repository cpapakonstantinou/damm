#ifndef __HOUSEHOLDER_H__
#define __HOUSEHOLDER_H__

/**
 * \file householder.h
 * \brief definitions for the householder method 
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

#include <cmath>
#include <cstring>
#include <algorithm>
#include "common.h"
#include "multiply.h"
#include "union.h"
#include "reduce.h"

/**
 * \brief Householder transformation
 *
 * Reflects a vector about a hyperplane. This is commonly used to transform a matrix 
 * to upper triangular form during QR decomposition.
 *
 * Given a reflection vector \p v, the Householder matrix is defined as:
 *     H = I - 2 * (v v^T) / (v^T v)
 *
 * In numerical implementations, the normalized vector is avoided. Instead, we define:
 *     H = I - tau * v * v^T
 * where \p tau = 2 / (v^T v), and \p v is chosen such that v[0] = 1.
 */
namespace damm
{
	/**
	 * \brief Compute the Householder reflector for a vector x
	 *
	 * \param x The input vector of size n (not modified)
	 * \param n The size (dimension) of the vector x
	 * \param v The output Householder vector (length n), caller-allocated
	 *           Convention: v[0] = 1, rest filled in this function
	 * \param tau The output scalar tau such that H = I - tau * v * v^T
	 * \param beta The scalar that replaces x[0] in H * x
	 */
	template<typename T, SIMD S>
	void make_householder_simd(const T** x, size_t N, T* v, T& tau, T& beta) 
	{
		if (N <= 0) {
			tau = T(0);
			beta = T(0);
			return;
		}

		const T x0 = x[0][0];

		if (N == 1) {
			tau = T(0);
			beta = x0;
			v[0] = T(1);
			return;
		}

		// Tail: x[1:] (i.e., x[1][0], ..., x[N-1][0])
		const T* x_tail = &x[1][0];

		// Allocate temporary buffer for squared tail: x_tail^2
		auto x_tail_sq = aligned_alloc_2D<T, static_cast<size_t>(S)>(1, N - 1);  // shape [1][N-1]
		T* x_tail_sq_flat = x_tail_sq[0];

		union_block_simd_flat<T, std::multiplies<>, S>(
			const_cast<T*>(x_tail), const_cast<T*>(x_tail), x_tail_sq_flat, 1, N - 1);

		T norm2 = reduce_block_simd_flat<T, std::plus<>, S>(
			x_tail_sq_flat, T(0), 1, N - 1);

		if (norm2 == T(0) && x0 >= T(0)) {
			tau = T(0);
			beta = x0;
			v[0] = T(1);
			std::fill(v + 1, v + N, T(0));
			return;
		}

		beta = std::sqrt(x0 * x0 + norm2);
		if (x0 >= T(0)) beta = -beta;

		const T u0 = x0 - beta;
		tau = T(2) / (u0 * u0 + norm2);

		v[0] = T(1);

		// Fill temporary vector of u0 values
		auto u0_vec_mem = aligned_alloc_2D<T, static_cast<size_t>(S)>(1, N - 1);
		T* u0_vec = u0_vec_mem[0];
		std::fill(u0_vec, u0_vec + (N - 1), u0);

		// Allocate v_tail buffer
		auto v_tail_mem = aligned_alloc_2D<T, static_cast<size_t>(S)>(1, N - 1);
		T* v_tail = v_tail_mem[0];

		union_block_simd_flat<T, std::divides<>, S>(
			const_cast<T*>(x_tail), u0_vec, v_tail, 1, N - 1);

		for (size_t i = 1; i < N; ++i)
			v[i] = v_tail[i - 1];
	}

		

	/**
	 * \brief Apply the Householder reflector from the left: A ← (I - τvvᵀ)A
	 *
	 * \param A       The input matrix (modified in-place), size m × n, stored row-major
	 * \param M       Rows of A
	 * \param N       Columns of A
	 * \param V       Householder vector of length M (v[0] = 1)
	 * \param tau     The scalar τ from the reflector
	 */
	template<typename T, SIMD S>
	void apply_householder_left_simd(T** A, int M, int N, const T* v, T tau)
	{
		auto v_mat = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, 1);
		auto A_t = aligned_alloc_2D<T, static_cast<size_t>(S)>(N, M);
		auto w_t = aligned_alloc_2D<T, static_cast<size_t>(S)>(N, 1);
		auto w   = aligned_alloc_2D<T, static_cast<size_t>(S)>(1, N);
		auto vw  = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, N);
		auto tau_mat = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, N);
		auto scaled_vw = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, N);

		for (int i = 0; i < M; ++i)
			v_mat[i][0] = v[i];

		transpose_block_simd<T, S>(A, A_t.get(), M, N);
		multiply_block_simd<T, S>(A_t.get(), v_mat.get(), w_t.get(), N, M, 1);
		transpose_block_simd<T, S>(w_t.get(), w.get(), N, 1);
		multiply_block_simd<T, S>(v_mat.get(), w.get(), vw.get(), M, 1, N);

		for (int i = 0; i < M; ++i)
			for (int j = 0; j < N; ++j)
				tau_mat[i][j] = tau;

		union_block_simd<T, std::multiplies<>, S>(tau_mat.get(), vw.get(), scaled_vw.get(), M, N);
		union_block_simd<T, std::minus<>, S>(A, scaled_vw.get(), A, M, N);
	}


	/**
	 * \brief Apply the Householder reflector from the right: A ← A(I - τvvᵀ)
	 *
	 * \param A       The input matrix (modified in-place), size m × n, stored row-major
	 * \param M       Number of rows of A
	 * \param N       Number of columns of A
	 * \param v       Householder vector of length n (v[0] = 1)
	 * \param tau     The scalar τ from the reflector
	 * \param work    Workspace of size at least m (must be allocated by caller)
	 */
	template <typename T, SIMD S>
	void apply_householder_right(T** A, size_t M, size_t N, const T* v, T tau, T* work) 
	{
		auto tmp = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, N);

		for (size_t i = 0; i < M; ++i) 
		{
			union_block_simd_flat<T, std::multiplies<>, S>(A[i], const_cast<T*>(v), tmp[i], 1, N);

			work[i] = reduce_block_simd_flat<T, std::plus<>, S>(tmp[i], T(0), 1, N);
		}

		for (size_t i = 0; i < M; ++i) 
		{
			T w = tau * work[i];
			for (size_t j = 0; j < N; ++j)
				A[i][j] -= w * v[j];
		}
	}

}//namespace damm
#endif //__HOUSEHOLDER_H__
