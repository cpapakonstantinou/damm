#ifndef __HOUSEHOLDER_H__
#define __HOUSEHOLDER_H__

/**
 * \file householder.h
 * \brief Householder method using fused SIMD operations
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
#include <common.h>
#include <multiply.h>
#include <union.h>
#include <reduce.h>
#include <fused_reduce.h>

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
	void make_householder(T* x, size_t N, T* v, T& tau, T& beta) 
	{
		if (N <= 0) 
		{
			tau = T(0);
			beta = T(0);
			return;
		}

		const T x0 = x[0];

		if (N == 1) 
		{
			tau = T(0);
			beta = x0;
			v[0] = T(1);
			return;
		}

		auto x_tail = aligned_alloc_2D<T, static_cast<size_t>(S)>(1, N - 1);
		
		for (size_t i = 1; i < N; ++i)
			x_tail[0][i - 1] = x[i];


		T norm2 = fused_reduce<T, std::multiplies<>, std::plus<>, S>(x_tail[0], x_tail[0], T(0), 1, N - 1);

		if (norm2 == T(0) && x0 >= T(0)) 
		{
			tau = T(0);
			beta = x0;
			v[0] = T(1);
			std::fill(v + 1, v + N, T(0));
			return;
		}

		beta = std::sqrt(x0 * x0 + norm2);
		if (x0 >= T(0))
			beta = -beta;

		const T u0 = x0 - beta;
		const T u_norm_sq = u0 * u0 + norm2;
		tau = T(2) * u0 * u0 / u_norm_sq;

		v[0] = T(1);

		auto v_tail = aligned_alloc_2D<T, static_cast<size_t>(S)>(1, N - 1);
		
		scalar::unite<T, std::divides<>, S>(x_tail[0], u0, v_tail[0], 1, N - 1);

		std::copy(v_tail[0], v_tail[0] + (N - 1), v + 1);
	}

	/**
	 * \brief Apply the Householder reflector from the left: A ← (I - τvvᵀ)A
	 *
	 * Mathematical: (I - τvvᵀ)A = A - τv(vᵀA)
	 *
	 * \param A       The input matrix (modified in-place), size m × n, stored row-major
	 * \param M       Rows of A
	 * \param N       Columns of A
	 * \param V       Householder vector of length M (v[0] = 1)
	 * \param tau     The scalar τ from the reflector
	 */
	template<typename T, SIMD S>
	void apply_householder_left(T** A, int M, int N, const T* v, T tau)
	{
		auto v_mat = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, 1);
		auto vT_mat = aligned_alloc_2D<T, static_cast<size_t>(S)>(1, M);
		auto w = aligned_alloc_2D<T, static_cast<size_t>(S)>(1, N);
		auto outer_prod = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, N);

		std::copy(v, v + M, vT_mat[0]);

		transpose<T, S>(vT_mat.get(), v_mat.get(), 1, M);

		multiply<T, S>(vT_mat.get(), A, w.get(), 1, M, N);

		scalar::unite<T, std::multiplies<>, S>(w.get(), tau, w.get(), 1, N);

		multiply<T, S>(v_mat.get(), w.get(), outer_prod.get(), M, 1, N);

		matrix::unite<T, std::minus<>, S>(A, outer_prod.get(), A, M, N);
	}

	/**
	 * \brief Apply the Householder reflector from the right: A ← A(I - τvvᵀ)
	 *
	 * Mathematical: A(I - τvvᵀ) = A - τ(Av)vᵀ
	 *
	 * \param A       The input matrix (modified in-place), size m × n, stored row-major
	 * \param M       Number of rows of A
	 * \param N       Number of columns of A
	 * \param v       Householder vector of length n (v[0] = 1)
	 * \param tau     The scalar τ from the reflector
	 */
	template <typename T, SIMD S>
	void apply_householder_right(T** A, size_t M, size_t N, const T* v, T tau) 
	{
		auto v_col = aligned_alloc_2D<T, static_cast<size_t>(S)>(N, 1);
		auto v_row = aligned_alloc_2D<T, static_cast<size_t>(S)>(1, N);
		auto w_col = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, 1);
		auto outer_prod = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, N);

		std::copy(v, v + N, v_col[0]);
		std::copy(v, v + N, v_row[0]);

		multiply<T, S>(A, v_col.get(), w_col.get(), M, N, 1);

		scalar::unite<T, std::multiplies<>, S>(w_col.get(), tau, w_col.get(), M, 1);

		multiply<T, S>(w_col.get(), v_row.get(), outer_prod.get(), M, 1, N);

		matrix::unite<T, std::minus<>, S>(A, outer_prod.get(), A, M, N);
	}

}//namespace damm
#endif //__HOUSEHOLDER_H__