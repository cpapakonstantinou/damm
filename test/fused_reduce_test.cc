/**
 * \file fused_reduce_test.cc
 * \brief unit test for fused_reduce module
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
#include "fused_reduce.h"
#include "test_utils.h"
#include "carray.h"
#include "broadcast.h"
#include "common.h"

using namespace damm;

template <typename T, typename U, typename R>
T 
fused_reduce_naive(T** A, T** B, T seed, size_t M, size_t N)
{
	T r = seed;
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			r = R{}(r, U{}(A[i][j], B[i][j]));
	return r;
}

template <typename T, typename U, typename R>
void test_op(const char* name, const size_t M, const size_t N) 
{
	carray<T, 2, 64> A(M, N);
	carray<T, 2, 64> B(M, N);
	
	ones<T, NONE>(A.get(), M, N);
	ones<T, NONE>(B.get(), M, N);

	if constexpr (std::is_same_v<R, std::plus<>>) 
	{
		T val = 1.0;
		for (auto it = A.begin(); it != A.end(); ++it) 
		{
			*it = val;
			val *= 0.9;
		}
	} 
	if constexpr (std::is_same_v<R, std::multiplies<>>)
	{
		broadcast<T, NONE>(A.get(), 2E-5, M, N);
		broadcast<T, NONE>(B.get(), 1E-3, M, N);
	}

	if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) 
	{
		using value_t = typename T::value_type;
		
		if constexpr (std::is_same_v<R, std::plus<>>) 
		{
			size_t idx = 0;
			for (auto it = A.begin(); it != A.end(); ++it, ++idx) 
			{
				value_t real_part = 1.0 + static_cast<value_t>(idx % 10) * 0.1;
				value_t imag_part = 0.5 + static_cast<value_t>(idx % 7) * 0.1;
				*it = T(real_part, imag_part);
			}
			
			idx = 0;
			for (auto it = B.begin(); it != B.end(); ++it, ++idx) 
			{
				value_t real_part = 0.8 + static_cast<value_t>(idx % 8) * 0.1;
				value_t imag_part = 0.3 + static_cast<value_t>(idx % 6) * 0.1;
				*it = T(real_part, imag_part);
			}
		}
		else if constexpr (std::is_same_v<R, std::multiplies<>>) 
		{
			value_t real_a = 0.1;
			value_t imag_a = 0.05;
			value_t real_b = 0.2;
			value_t imag_b = 0.08;
			
			for (auto it = A.begin(); it != A.end(); ++it) 
			{
				*it = T(real_a, imag_a);
			}
			
			for (auto it = B.begin(); it != B.end(); ++it) 
			{
				*it = T(real_b, imag_b);
			}
		}
	}

	T seed = seed_left_fold<T, R>();
	
	// Run all implementations
	T r_ref = fused_reduce_naive<T, U, R>(A.get(), B.get(), seed, M, N);
	T r_none = fused_reduce<T, U, R, NONE>(A.get(), B.get(), seed, M, N);
	T r_sse = fused_reduce<T, U, R, SSE>(A.get(), B.get(), seed, M, N);
	T r_avx = fused_reduce<T, U, R, AVX>(A.get(), B.get(), seed, M, N);
	T r_avx512 = fused_reduce<T, U, R, AVX512>(A.get(), B.get(), seed, M, N);
	
	// Verify results
	bool test_all = true;
	bool test_none = approx_equal<T>(r_ref, r_none);
	bool test_sse = approx_equal<T>(r_ref, r_sse);
	bool test_avx = approx_equal<T>(r_ref, r_avx);
	bool test_avx512 = approx_equal<T>(r_ref, r_avx512);
	test_all = ( test_none && test_sse && test_avx && test_avx512 );
	std::cout << "[" << (test_none ? "OK  " : "FAIL") << "] " << "fused_reduce<NONE>: " << r_ref << std::endl;
	std::cout << "[" << (test_sse ? "OK  " : "FAIL") << "] " << "fused_reduce<SSE>: " << r_sse << std::endl;
	std::cout << "[" << (test_avx ? "OK  " : "FAIL") << "] " << "fused_reduce<AVX>: " << r_avx << std::endl;
	std::cout << "[" << (test_avx512 ? "OK  " : "FAIL") << "] " << "fused_reduce<AVX512>: " << r_avx512 << std::endl;
	std::cout << "[" << (test_all ? "OK  " : "FAIL") << "] " << "fused_reduce: all_tests: " << name << ": " << r_ref << std::endl;
}

void test_all_ops(const size_t M, const size_t N)
{	
	test_op<double, std::plus<>, std::plus<>>("double: sum(A + B)",M , N);
	test_op<double, std::minus<>, std::plus<>>("double: sum(A - B)", M, N);
	test_op<double, std::multiplies<>, std::plus<>>("double: sum(A * B)", M, N);
	test_op<double, std::divides<>, std::plus<>>("double: sum(A / B)", M, N);
	test_op<double, std::plus<>, std::multiplies<>>("double: product(A + B)",M , N);
	test_op<double, std::minus<>, std::multiplies<>>("double: product(A - B)", M, N);
	test_op<double, std::multiplies<>, std::multiplies<>>("double: product(A * B)", M, N);
	test_op<double, std::divides<>, std::multiplies<>>("double: product(A / B)", M, N);

	test_op<float, std::plus<>, std::plus<>>("float: sum(A + B)",M , N);
	test_op<float, std::minus<>, std::plus<>>("float: sum(A - B)", M, N);
	test_op<float, std::multiplies<>, std::plus<>>("float: sum(A * B)", M, N);
	test_op<float, std::divides<>, std::plus<>>("float: sum(A / B)", M, N);
	test_op<float, std::plus<>, std::multiplies<>>("float: product(A + B)",M , N);
	test_op<float, std::minus<>, std::multiplies<>>("float: product(A - B)", M, N);
	test_op<float, std::multiplies<>, std::multiplies<>>("float: product(A * B)", M, N);
	test_op<float, std::divides<>, std::multiplies<>>("float: product(A / B)", M, N);

	test_op<std::complex<double>, std::plus<>, std::plus<>>("std::complex<double>: sum(A + B)",M , N);
	test_op<std::complex<double>, std::minus<>, std::plus<>>("std::complex<double>: sum(A - B)", M, N);
	test_op<std::complex<double>, std::multiplies<>, std::plus<>>("std::complex<double>: sum(A * B)", M, N);
	test_op<std::complex<double>, std::divides<>, std::plus<>>("std::complex<double>: sum(A / B)", M, N);
	test_op<std::complex<double>, std::plus<>, std::multiplies<>>("std::complex<double>: product(A + B)",M , N);
	test_op<std::complex<double>, std::minus<>, std::multiplies<>>("std::complex<double>: product(A - B)", M, N);
	test_op<std::complex<double>, std::multiplies<>, std::multiplies<>>("std::complex<double>: product(A * B)", M, N);
	test_op<std::complex<double>, std::divides<>, std::multiplies<>>("std::complex<double>: product(A / B)", M, N);
	
	test_op<std::complex<float>, std::plus<>, std::plus<>>("std::complex<float>: sum(A + B)",M , N);
	test_op<std::complex<float>, std::minus<>, std::plus<>>("std::complex<float>: sum(A - B)", M, N);
	test_op<std::complex<float>, std::multiplies<>, std::plus<>>("std::complex<float>: sum(A * B)", M, N);
	test_op<std::complex<float>, std::divides<>, std::plus<>>("std::complex<float>: sum(A / B)", M, N);
	test_op<std::complex<float>, std::plus<>, std::multiplies<>>("std::complex<float>: product(A + B)",M , N);
	test_op<std::complex<float>, std::minus<>, std::multiplies<>>("std::complex<float>: product(A - B)", M, N);
	test_op<std::complex<float>, std::multiplies<>, std::multiplies<>>("std::complex<float>: product(A * B)", M, N);
	test_op<std::complex<float>, std::divides<>, std::multiplies<>>("std::complex<float>: product(A / B)", M, N);
}	

int main(int argc, char* argv[])
{
	try
	{
		test_all_ops(256, 256);
	}
	catch(const std::exception& e)
	{
		std::cerr << "[EXCEPT] fused_reduce_test: " << e.what() << std::endl;
	}

	return 0;
}

