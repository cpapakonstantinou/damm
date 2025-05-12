#ifndef __SOLVE_H__
#define __SOLVE_H__
/**
 * \file solve.h
 * \brief definitions for solve utilities 
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

namespace damm
{

	namespace tri
	{
		/**
		 * \brief General forward substitution.
		 *        Solves L * y = b, where L is lower triangular.
		 *
		 * \param L         Lower triangular matrix (NxN).
		 * \param b         Right-hand side vector.
		 * \param y         Output vector (solution).
		 * \param N         Dimension.
		 * \param unit_diag If true, assumes unit diagonal.
		 */
		template <typename T, SIMD S>
		inline __attribute__((always_inline)) 
		void 
		forward_substitution(T** L, const T* b, T* y, const size_t N,
			const bool unit_diag = false)
		{
			for (size_t i = 0; i < N; ++i)
			{
				T sum = (i == 0)
					? T(0)
					: fused_reduce<T, std::multiplies<>, std::plus<>, S>(L[i], y, T(0), i, 1);

				y[i] = unit_diag ? (b[i] - sum)
								 : (b[i] - sum) / L[i][i];
			}
		}

		/**
		 * \brief General backward substitution.
		 *        Solves U * x = y, where U is upper triangular.
		 *
		 * \param U         Upper triangular matrix (NxN).
		 * \param y         Right-hand side vector.
		 * \param x         Output vector (solution).
		 * \param N         Dimension.
		 * \param unit_diag If true, assumes unit diagonal.
		 */
		template <typename T, SIMD S>
		inline __attribute__((always_inline)) 
		void 
		backward_substitution(T** U, const T* y, T* x, const size_t N, 
			const bool unit_diag = false)
		{
			for (size_t i = N; i-- > 0; )
			{
				size_t len = N - i - 1;
				T sum = (len > 0)
					? fused_reduce<T, std::multiplies<>, std::plus<>, S>(&U[i][i + 1], &x[i + 1], T(0), len, 1)
					: T(0);

				x[i] = unit_diag ? (y[i] - sum)
								 : (y[i] - sum) / U[i][i];
			}
		}
	}//namespace tri
}//namespace damm

#endif //__SOLVE_H__