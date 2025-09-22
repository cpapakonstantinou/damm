/**
 * \file inverse_test.cc
 * \brief unit test for inverse module
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
#include <random>
#include <chrono>

#include "test_utils.h"
#include "inverse.h"
#include "oracle.h"
#include "heracles.h"

using namespace damm;

using E = int;
using U = std::string_view;

bool oracle::use_syslog = false;
int oracle::log_level = LOG_INFO;

template<typename T, SIMD S>
std::expected<E, U> 
lu_inverse(void* instructions) 
{
	constexpr size_t N = 4;
	constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;
	
	auto A = aligned_alloc_2D<T, S>(N, N);
	auto A_lu = aligned_alloc_2D<T, S>(N, N);
	auto A_inv = aligned_alloc_2D<T, S>(N, N);
	
	T A_data[4][4] = {
		{4, 1, 2, 1},
		{1, 4, 1, 2},
		{2, 1, 4, 1},
		{1, 2, 1, 4}
	};
	
	for (size_t i = 0; i < N; ++i) 
		for (size_t j = 0; j < N; ++j)
		{
			A_lu[i][j] = A_data[i][j];
			A[i][j] = A_data[i][j]; 
		} 
	
	bool lu_success = lu::inverse<T, S>(A_lu.get(), A_inv.get(), N);
		
	if (!lu_success) 
	{
		print_matrix<T>(A.get(), N, N, "Original Matrix A");
		
		print_matrix<T>(A_inv.get(), N, N, "LU Inverse");
		
		return std::unexpected("matrix is singular");
	}

	// Compare accuracy: compute A * A_inv - I for both methods
	auto I = aligned_alloc_2D<T, S>(N, N);
	auto I_lu = aligned_alloc_2D<T, S>(N, N);
	
	set_identity(I.get(), N, N);

	multiply<T>(A.get(), A_inv.get(), I_lu.get(), N, N, N);
	
	T lu_error = matrix_max_error(I.get(), I_lu.get(), N, N);
	
	if (lu_error > tolerance)
	{
		print_matrix<T>(I_lu.get(), N, N, "Reconstructed I:");
		std::string response = std::format("inversion error: {}", lu_error);
		return std::unexpected(response);
	}

	return 0;
}

template<typename T, SIMD S>
std::expected<E, U> 
qr_inverse(void* instructions) 
{
	constexpr size_t N = 4, M = 4;
	constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;
	
	auto A = aligned_alloc_2D<T, S>(M, N);	
	auto A_qr = aligned_alloc_2D<T, S>(M, N);
	auto A_inv = aligned_alloc_2D<T, S>(M, N);
	
	T A_data[M][N] = {
		{4, 1, 2, 1},
		{1, 4, 1, 2},
		{2, 1, 4, 1},
		{1, 2, 1, 4}
	};
	
	for (size_t i = 0; i < M; ++i) 
		for (size_t j = 0; j < N; ++j)
		{
			A_qr[i][j] = A_data[i][j];
			A[i][j] = A_data[i][j]; 
		} 
	
	
	bool qr_success = qr::inverse<T, S>(A_qr.get(), A_inv.get(), M, N);
	
	if (!qr_success) 
	{
		print_matrix(A.get(), M, N, "Original Matrix A");
		print_matrix(A_inv.get(), M, N, "QR Inverse");
		return std::unexpected("matrix is singular");
	}
	// Compare accuracy: compute A * A_inv - I for both methods
	auto I = aligned_alloc_2D<T, S>(N, N);
	auto I_qr = aligned_alloc_2D<T, S>(M, N);
	set_identity(I.get(), N, N);
	
	multiply<T>(A.get(), A_inv.get(), I_qr.get(), M, N, N);
	
	T qr_error = matrix_max_error(I.get(), I_qr.get(), M, N);

	if (qr_error > tolerance)
	{
		print_matrix(I_qr.get(), N, N, "Reconstructed I:");
		std::string response = std::format("inversion error: {}", qr_error);
		return std::unexpected(response);
	}

	return 0;
}

int main(int argc, char* argv[]) 
{
	using T = double;
	constexpr SIMD S = AVX512;
	try 
	{
		oracle::Heracles<E, U> heracles{};

		heracles.add_labor(0, "LU inverse", &lu_inverse<T, S>, nullptr);
		heracles.add_labor(1, "QR inverse", &qr_inverse<T, S>, nullptr);

		heracles.perform_labors();

	} 
	catch (const std::exception& e) 
	{
		std::cerr << "[EXCEPT] inverse_test: " << e.what() << "\n";
		return 1;
	}
	
	return 0;
}