/**
 * \file decompose_test.cc
 * \brief unit test for decompose.h
 * \author cpapakonstantinou
 * \date 2025
 */
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>

#include "test_utils.h"
#include "broadcast.h"
#include "decompose.h"
#include "carray.h"
#include "oracle.h"
#include "heracles.h"

using namespace damm;
using E = int;
using U = std::string_view;

int oracle::log_level = LOG_INFO;
bool oracle::use_syslog = false;

template<typename T, typename S>
std::expected<E, U> 
lu_decomposition(void* instructions) 
{
	constexpr size_t N = 4;
	constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;
	
	auto A = carray<T, 2, S::bytes>(N, N);
	auto L = carray<T, 2, S::bytes>(N, N);
	auto U = carray<T, 2, S::bytes>(N, N);
	auto LU = carray<T, 2, S::bytes>(N, N);
	auto PA = carray<T, 2, S::bytes>(N, N);  // Permuted A matrix
	auto LU_result = carray<T, 2, S::bytes>(N, N);  // L*U result
	
	// Initialize test matrix
	T A_data[N][N] = {
		{4, 1, 2, 1},
		{2, 5, 1, 3},
		{1, 2, 6, 2},
		{3, 1, 1, 4}
	};
	
	// Copy data to matrices
	for (size_t i = 0; i < N; ++i) 
	{
		for (size_t j = 0; j < N; ++j)
		{
			A[i][j] = A_data[i][j];
			LU[i][j] = A_data[i][j];  // LU will be modified in-place from A
		}
	}
	
	size_t P[N];
	
	// Perform LU decomposition
	bool success = lu::decompose<T, S>(LU.get(), P, N);
	
	if (!success) 
	{
		print_matrix(A.get(), N, N, "Original Matrix A");		
		print_matrix(LU.get(), N, N, "LU Factors (L below diagonal, U on and above diagonal)");
		print_vector(P, N, "Permutation vector");	
		return std::unexpected{"matrix may be singular"};
	} 

	// Fill L and U
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
		{
			if (i > j) 
				L[i][j] = LU[i][j];
			else if (i == j) 
			{
				L[i][j] = 1.0;
				U[i][j] = LU[i][j];
			}
			else
				U[i][j] = LU[i][j];
		}
		
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			PA[i][j] = A[P[i]][j];

	multiply<T, S>(L.get(), U.get(), LU_result.get(), N, N, N);
	T max_error = matrix_max_error(A.get(), LU_result.get(), N, N);

	if( max_error > tolerance)
	{
		print_matrix(L.get(), N, N, "Extracted L matrix");
		print_matrix(U.get(), N, N, "Extracted U matrix");
		print_matrix(PA.get(), N, N, "Permuted Matrix PA");
		print_matrix(LU_result.get(), N, N, "L*U Reconstruction");
		std::string response = std::format("Reconstruction error ||PA - L*U||_max = {}", response);
		return std::unexpected{response};
	}

	return 0;
}

template<typename T, typename S>
std::expected<E, U>
qr_decomposition(void* instructions) 
{	
	constexpr size_t M = 4, N = 3;
	constexpr T tolerance = std::is_same_v<T, float> ? 1e-6f : 1e-12;
	
	auto A = carray<T, 2, S::bytes>(M, N);
	auto Q = carray<T, 2, S::bytes>(M, M);
	auto R = carray<T, 2, S::bytes>(M, N);
	
	// Initialize test matrix (overdetermined system)
	T A_data[4][3] = \
	{
		{1, 1, 0},
		{1, 0, 1},
		{0, 1, 1},
		{1, 1, 1}
	};
	
	for(size_t i =0; i < M; ++i)
		for(size_t j = 0; j < N; ++j)
			A[i][j] = A_data[i][j];

	// Use policy-based QR decomposition
	bool success = qr::decompose<T, S>(A.get(), Q.get(), R.get(), M, N);
	
	if (!success) 
	{
		print_matrix(A.get(), M, N, "Original Matrix A");
		print_matrix(Q.get(), M, M, "Orthogonal Matrix Q");
		print_matrix(R.get(), M, N, "Upper Triangular Matrix R");
		return std::unexpected{"decomposition failed"};
	}

	carray<T, 2, S::bytes> QTQ(M, M);
	carray<T, 2, S::bytes> Q_transpose(M, M);
	
	transpose<T, S>(Q.get(), Q_transpose.get(), M, M);
	multiply<T, S>(Q_transpose.get(), Q.get(), QTQ.get(), M, M, M);

	carray<T, 2, S::bytes> I(M, N);
	set_identity(I.get(), M, N);
	T orthogonality_error = matrix_max_error<T>(QTQ.get(), I.get(), M, N);

	if(orthogonality_error > tolerance)
	{
		print_matrix(QTQ.get(), M, N, "Q^T*Q");
		std::string response = std::format("Orthogonality error ||Q^T*Q - I||_max = {}", orthogonality_error);
		return std::unexpected{response};
	}

	carray<T, 2, S::bytes> QR(M, N);
	multiply<T, S>(Q.get(), R.get(), QR.get(), M, M, N);
	T reconstruction_error = matrix_max_error(A.get(), QR.get(), M, N);
	
	if(reconstruction_error > tolerance)
	{
		print_matrix(QR.get(), M, N, "Q*R");
		std::string response = std::format("Reconstruction error ||A - Q*R||_max = {}", reconstruction_error);
		return std::unexpected{response};
	}

	return 0;
}


int main(int argc, char* argv[]) 
{

	using T = double;

	oracle::Heracles<E, U> heracles{};
	heracles.add_labor(0, "lu::decompose", &lu_decomposition<T, AVX512>, nullptr);
	heracles.add_labor(1, "qr::decompose", &qr_decomposition<T, AVX512>, nullptr);

	try 
	{
		heracles.perform_labors();
	} 
	catch (const std::exception& e) 
	{
		std::cerr << "[EXCEPT] decompose_test: " << e.what() << "\n";
		return -1;
	}
	
	return 0;
}