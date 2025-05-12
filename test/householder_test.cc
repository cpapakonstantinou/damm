/**
 * \file householder_test.cc
 * \brief unit test for householder module
 * \author cpapakonstantinou
 * \date 2025
 */
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
#include <iostream>
#include <iomanip>
#include <cmath>
#include "test_utils.h"
#include "householder.h"
#include "carray.h"
#include "oracle.h"
#include "heracles.h"

#define cvector carray<T, 1, 64>

using namespace damm;
using E = int;
using U = std::string_view;

bool oracle::use_syslog = false;
int oracle::log_level = LOG_INFO;

// Test case from Math Stack Exchange: x = (4, 0, 3) -> y = (5, 0, 0)
template<typename T, SIMD S>
std::expected<E, U> 
test_vector_transformation(void* instructions) 
{
	constexpr size_t N = 3;
	constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;

	cvector x(N);
	x[0] = 4.0;
	x[1] = 0.0;
	x[2] = 3.0;
	
	cvector v(N);

	T tau = 0, beta = 0;
	
	make_householder<T, S>(x.get(), N, v.get(), tau, beta);
	
	oracle::Divinate<T, tolerance> expected_beta {-5.0};
	
	if( expected_beta != beta)
		return std::unexpected("beta = "+std::to_string(beta));

	oracle::Divinate<T, tolerance> expected_v0 {T(1)};		

	if( expected_v0 != v[0])
		return std::unexpected("v0 = "+std::to_string(v[0]));
	
	return 0;
}

template<typename T, SIMD S>
std::expected<E, U> 
test_reflection_property(void* instructions) 
{
	constexpr size_t N = 3;
	constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;
	
	cvector x(N);
	x[0] = 4.0;
	x[1] = 0.0;
	x[2] = 3.0;
	
	cvector v(N);
	T tau = 0, beta = 0;
	
	make_householder<T, S>(x.get(), N, v.get(), tau, beta);
	
	carray<T, 2, 64> A(N, N);
	for (size_t i = 0; i < N; ++i) 
		for (size_t j = 0; j < N; ++j) 
		{
			if (j == 0) 
				A[i][j] = x[i];
			else 
				A[i][j] = (i == j) ? T(1) : T(0);
		}
	
	apply_householder_left<T, S>(A.get(), N, N, v.begin(), tau);
	
	oracle::Divinate<T, tolerance> e[3] = { {beta}, {T(0)}, {T{0}} };

	bool reflection_correct = (e[0] == A[0][0]) &&
							  (e[1] == A[1][0]) &&
							  (e[2] == A[2][0]);
	if (!reflection_correct)
	{
		print_matrix(A.get(), N, N, "Matrix A before Householder");
		print_matrix(A.get(), N, N, "Matrix A after Householder");
		std::string result = std::format("expected [{}, {}, {}], got [{}, {}, {}]", 
			beta, T(0), T(0), A[0][0], A[1][0], A[2][0]);
		return std::unexpected{result};
	}

	return 0;

}

template<typename T, SIMD S>
std::expected<E, U> 
test_orthogonality(void* instructions) 
{
	constexpr size_t N = 4;
	constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;

	cvector x(N);
	
	x[0] = 2.0;
	x[1] = -2.0;
	x[2] = 4.0;
	x[3] = -1.0;
	
	cvector v(N);
	T tau = 0, beta = 0;
	
	make_householder<T, S>(x.get(), N, v.get(), tau, beta);
	
	carray<T, 2, 64> I(N, N);
	carray<T, 2, 64> HTH(N, N);
	
	set_identity<T>(I.get(), N, N);
	set_identity<T>(HTH.get(), N, N);
	
	apply_householder_left<T, S>(HTH.get(), N, N, v.get(), tau);	
	apply_householder_right<T, S>(HTH.get(), N, N, v.get(), tau);
	
	T orthogonality_error = matrix_max_error(HTH.get(), I.get(), N, N);
	
	if( orthogonality_error > tolerance)
	{
		print_matrix(&HTH[0], N, N, "H * H^T (should be identity)");
		return std::unexpected("not orthogonal");
	}
	return 0;
}


int main(int argc, char* argv[]) 
{
	using T = double;
	constexpr SIMD S = AVX512;
		
	oracle::Heracles<E, U> heracles{};

	heracles.add_labor(0, "vector_transformation", &test_vector_transformation<T, S>, nullptr);
	heracles.add_labor(1, "reflection_property", &test_reflection_property<T, S>, nullptr);
	heracles.add_labor(2, "orthogonality", &test_orthogonality<T, S>, nullptr);
	
	try
	{
		heracles.perform_labors();
	}
	catch(const std::exception& e)
	{
		std::cerr << "[ EXCEPT ] householder_test:" << e.what() << std::endl;
		return -1;
	}
	
	return 0;
}