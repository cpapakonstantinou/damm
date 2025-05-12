#include <iostream>
#include <iomanip>
#include "householder.h"
#include "carray.h"

using namespace damm;

template<typename T>
void print_vector(const T* v, size_t n, const std::string& name) {
	std::cout << name << ": [ ";
	for (size_t i = 0; i < n; ++i)
		std::cout << v[i] << " ";
	std::cout << "]\n";
}

template<typename T>
void print_matrix(const T* A, size_t M, size_t N, const std::string& name) {
	std::cout << name << ":\n";
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) 
			std::cout << std::setw(10) << A[i * N + j] << " ";
		std::cout << "\n";
	}
	std::cout << "\n";
}

int main() {
	using T = double;

	constexpr size_t N = 4;
	constexpr size_t M = 4;
	constexpr SIMD S = AVX512;

	// Input vector x to create Householder reflector from
	carray<T, 2, 64> x(N, 1);
	x[0][0] = 4.0;
	x[1][0] = 3.0;
	x[2][0] = 0.0;
	x[3][0] = 0.0;

	// Create pointer array for make_householder_simd
	const T* x_ptrs[N];
	for (size_t i = 0; i < N; ++i)
		x_ptrs[i] = &x[i][0];

	// Allocate Householder vector v and workspace
	carray<T, 2, 64> v(N, 1);
	T tau = 0, beta = 0;

	// Compute Householder reflector
	make_householder_simd<T, S>(x_ptrs, N, &v[0][0], tau, beta);

	print_vector(&v[0][0], N, "Householder vector v");
	std::cout << "tau = " << tau << "\n";
	std::cout << "beta = " << beta << "\n\n";

	// Allocate matrix A for testing
	carray<T, 2, 64> A(M, N);
	// Initialize A with some values
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			A[i][j] = static_cast<T>(i * N + j + 1);

	print_matrix(&A[0][0], M, N, "Original matrix A");

	// Apply Householder from left: A = (I - tau v v^T) A
	apply_householder_left_simd<T, S>(&A[0], M, N, &v[0][0], tau);

	print_matrix(&A[0][0], M, N, "After applying Householder from the left");

	// Reset A for right application test
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			A[i][j] = static_cast<T>(i * N + j + 1);

	// Workspace for right application
	carray<T, 2, 64> work(M, 1);

	// Apply Householder from right: A = A (I - tau v v^T)
	apply_householder_right<T, S>(&A[0], M, N, &v[0][0], tau, &work[0][0]);

	print_matrix(&A[0][0], M, N, "After applying Householder from the right");

	return 0;
}
