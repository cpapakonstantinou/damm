/**
 * \file test_utils.h
 * \brief common functions used in the damm:: unit tests
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
#include <type_traits>
#include <complex>
#include <random>

template<typename T>
void print_vector(const T* v, size_t n, const std::string& name) 
{
	std::cout << name << ": [ ";
	for (size_t i = 0; i < n; ++i)
		std::cout << std::setprecision(6) << v[i] << " ";
	std::cout << "]\n";
}

template<typename T>
void print_matrix(T** A, size_t M, size_t N, const std::string& name) 
{
	std::cout << name << ":\n";
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) 
			std::cout << std::setw(12) << std::setprecision(6) << A[i][j] << " ";
		std::cout << "\n";
	}
	std::cout << "\n";
}

template<typename T, double tol = 1e-4>
bool is_same(const char* name, T** A, T** B, const size_t M, const size_t N)
{
	bool x = true;
	size_t i = 0, j = 0;

	for (i = 0; i < M; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			if constexpr (std::is_floating_point<T>::value)
			{
				if (std::abs(A[i][j] - B[i][j]) > tol)
				{
					x = false;
					break;
				}
			}
			else if constexpr (std::is_same<T, std::complex<float>>::value ||
							   std::is_same<T, std::complex<double>>::value)
			{
				if (std::abs(A[i][j] - B[i][j]) > tol)
				{
					x = false;
					break;
				}
			}
			else
			{
				if (A[i][j] != B[i][j])
				{
					x = false;
					break;
				}
			}
		}
		if (!x) break;
	}

	printf("[%-4s] %s\n", (x ? "OK" : "FAIL"), name);

	if (!x)
	{
		if constexpr (std::is_same<T, std::complex<float>>::value ||
					  std::is_same<T, std::complex<double>>::value)
		{
			printf("Mismatch at [%zu][%zu]: A = (%f, %f), B = (%f, %f)\n",
				   i, j,
				   A[i][j].real(), A[i][j].imag(),
				   B[i][j].real(), B[i][j].imag());
		}
		else
		{
			// Only cast non-complex types to double for printing
			printf("Mismatch at [%zu][%zu]: A = %f, B = %f\n",
				   i, j,
				   static_cast<double>(A[i][j]),
				   static_cast<double>(B[i][j]));
		}
	}

	return x;
}


template <typename T>
bool 
approx_equal(T a, T b, double rel_tol = 1e-3, double abs_tol = 1e-4) 
{
	if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) 
	{
		using ValueType = typename T::value_type;
		return approx_equal(a.real(), b.real(), rel_tol, abs_tol) && 
			   approx_equal(a.imag(), b.imag(), rel_tol, abs_tol);
	} 
	else 
	{
		if (std::abs(a - b) <= abs_tol) 
		{
			return true;
		}
		double rel_diff = std::abs(a - b) / std::max(std::abs(a), std::abs(b));
		return rel_diff <= rel_tol;
	}
}

template<typename T>
T 
matrix_max_error(T** A, T** B, size_t M, size_t N)
{
	T max_error = T(0);
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			max_error = std::max(max_error, std::abs(A[i][j] - B[i][j]));
	
	return max_error;
}


template<typename T>
void
fill_rand(T** A, const size_t M, const size_t N)
{
	// Fill with deterministic test data
	std::mt19937 rng(42);
	
	if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) 
	{
		std::uniform_real_distribution<T> dist(-1.0, 1.0);
		for (size_t i = 0; i < M; ++i) 
		{
			for (size_t j = 0; j < N; ++j) 
			{
				A[i][j] = dist(rng);
			}
		}
	} 
	else if constexpr (std::is_same_v<T, std::complex<float>>) 
	{
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		for (size_t i = 0; i < M; ++i) 
		{
			for (size_t j = 0; j < N; ++j) 
			{
				A[i][j] = std::complex<float>(dist(rng), dist(rng));
			}
		}
	} 
	else if constexpr (std::is_same_v<T, std::complex<double>>) 
	{
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for (size_t i = 0; i < M; ++i) 
		{
			for (size_t j = 0; j < N; ++j) 
			{
				A[i][j] = std::complex<double>(dist(rng), dist(rng));
			}
		}
	}
}	