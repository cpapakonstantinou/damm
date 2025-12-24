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
#include <damm_memory.h>
#include <fused_reduce.h>

namespace damm
{
	enum TRIANGULAR
	{
		UPPER,
		LOWER
	};
	
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
		template <typename T, typename S = decltype(detect_simd())>
		inline __attribute__((always_inline)) 
		void 
		forward_substitution(T** L, const T* b, T* y, const size_t N,
			const bool unit_diag = false)
		{
			for (size_t i = 0; i < N; ++i)
			{
				T sum = T(0);
				
				if (i > 0)
				{
					// Need to compute dot product: L[i][0:i] · y[0:i]
					// Wrap L[i] and y as T** (1D views)
					auto L_row = view_as_2D(&L[i][0], 1, i);
					auto y_vec = view_as_2D(const_cast<T*>(y), 1, i);
					
					sum = fused_reduce<T, std::multiplies<>, std::plus<>, S>(
						L_row.get(), y_vec.get(), T(0), 1, i);
				}
				
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
		template <typename T, typename S = decltype(detect_simd())>
		inline __attribute__((always_inline)) 
		void 
		backward_substitution(T** U, const T* y, T* x, const size_t N, 
			const bool unit_diag = false)
		{
			for (size_t i = N; i-- > 0; )
			{
				size_t len = N - i - 1;
				T sum = T(0);
				
				if (len > 0)
				{
					// Need to compute dot product: U[i][i+1:N] · x[i+1:N]
					// Wrap &U[i][i+1] and &x[i+1] as T** (1D views)
					auto U_row = view_as_2D(&U[i][i + 1], 1, len);
					auto x_vec = view_as_2D(&x[i + 1], 1, len);
					
					sum = fused_reduce<T, std::multiplies<>, std::plus<>, S>(
						U_row.get(), x_vec.get(), T(0), 1, len);
				}
				
				x[i] = unit_diag ? (y[i] - sum)
								 : (y[i] - sum) / U[i][i];
			}
		}
	}//namespace tri
}//namespace damm
#endif //__SOLVE_H__