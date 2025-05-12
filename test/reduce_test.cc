/**
 * \file reduce_test.cc
 * \brief unit test for reduce module
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
#include <cmath>
#include <type_traits>
#include <functional>
#include <tuple>
#include "test_utils.h"
#include "carray.h"
#include "reduce.h"

using namespace damm;


template <typename T, typename O>
T reduce_naive(T** A, T seed, size_t M, size_t N)
{
	T result = seed;
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			result = O()(result, A[i][j]);
	return result;
}

template <typename T, typename O>
void test_op(const char* name, const size_t M, const size_t N)
{
	constexpr size_t ALIGN = 64;
	carray<T, 2, ALIGN> A(M, N);

	//Here we want to generate some series that we know through analytical analysis will converge
	//i.e. a specific geometric series. 
	if constexpr (std::is_same_v<O, std::plus<>>) 
	{
		if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) 
		{
			// Complex geometric series: z_n = (0.8 + 0.1i)^n
			// Converges because |0.8 + 0.1i| = sqrt(0.8^2 + 0.1^2) ≈ 0.806 < 1
			T val(1.0, 0.0);
			T ratio(0.8, 0.1);
			for (auto it = A.begin(); it != A.end(); ++it) 
			{
				*it = val;
				val *= ratio;
			}
		}
		else 
		{
			// Real geometric series
			T val = 1.0;
			for (auto it = A.begin(); it != A.end(); ++it) 
			{
				*it = val;
				val *= 0.9;
			}
		}
	} 

if constexpr (std::is_same_v<O, std::multiplies<>>)
{
	if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) 
	{
		using real_type = typename T::value_type;
		real_type root_val = std::pow(static_cast<real_type>(2.0), static_cast<real_type>(1.0) / static_cast<real_type>(M * N));
		real_type magnitude = root_val;
		real_type angle = static_cast<real_type>(0.01); // Small angle in radians
		
		T val(magnitude * std::cos(angle), magnitude * std::sin(angle));
		
		for (auto it = A.begin(); it != A.end(); ++it)
			*it = val;
	}
	else 
	{
		T val = std::pow(static_cast<T>(2.0), static_cast<T>(1.0) / static_cast<T>(M * N));
		for (auto it = A.begin(); it != A.end(); ++it)
			*it = val;
	}
}

	T seed = seed_left_fold<T, O>();
	T r_ref   = reduce_naive<T, O>(A.get(), seed, M, N);
	// Run all implementations
	T r_none = reduce<T, O, NONE>(A.get(), seed, M, N);
	T r_sse = reduce<T, O, SSE>(A.get(), seed, M, N);
	T r_avx = reduce<T, O, AVX>(A.get(), seed, M, N);
	T r_avx512 = reduce<T, O, AVX512>(A.get(), seed, M, N);

	// Verify results
	bool test_all = true;
	bool test_none = approx_equal<T>(r_ref, r_none, 1e-2);
	bool test_sse = approx_equal<T>(r_ref, r_sse, 1e-2);
	bool test_avx = approx_equal<T>(r_ref, r_avx, 1e-2);
	bool test_avx512 = approx_equal<T>(r_ref, r_avx512, 1e-2);
	test_all = ( test_none && test_sse && test_avx && test_avx512 );

	std::cout << "[" << (test_none ? "OK  " : "FAIL") << "] " << "reduce<NONE>: " << r_ref << std::endl;
	std::cout << "[" << (test_sse ? "OK  " : "FAIL") << "] " << "reduce<SSE>: " << r_sse << std::endl;
	std::cout << "[" << (test_avx ? "OK  " : "FAIL") << "] " << "reduce<AVX>: " << r_avx << std::endl;
	std::cout << "[" << (test_avx512 ? "OK  " : "FAIL") << "] " << "reduce<AVX512>: " << r_avx512 << std::endl;
	std::cout << "[" << (test_all ? "OK  " : "FAIL") << "] " << "reduce: all_tests: " << name << ": " << r_ref << std::endl;
}

void 
test_all_ops(const size_t M, const size_t N) 
{
	test_op<double, std::plus<>>("double: sum(A)", M , N);
	test_op<double, std::multiplies<>>("double: product(A)", M , N);
	test_op<float, std::plus<>>("float: sum(A)", M , N);
	test_op<float, std::multiplies<>>("float: product(A)", M , N);

	test_op<std::complex<double>, std::plus<>>("std::complex<double>: sum(A)", M , N);
	test_op<std::complex<double>, std::multiplies<>>("std::complex<double>: product(A)", M , N);
	test_op<std::complex<float>, std::plus<>>("std::complex<float>: sum(A)", M , N);
	test_op<std::complex<float>, std::multiplies<>>("std::complex<float>: product(A)", M , N);

}

int main(int argc, char* argv[]) 
{
	try
	{
		//tester logic for multiplies limits the max M, N for floats	
		constexpr size_t M = 256, N = 256;
		test_all_ops(M, N);
	}
	catch( std::exception& e )
	{
		std::cerr << "[FAIL] reduce_test: " << e.what() << std::endl;
	}

	return 0;
}

