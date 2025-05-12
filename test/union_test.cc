/**
 * \file union_test.cc
 * \brief unit test for union module
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
#include "test_utils.h"
#include "carray.h"
#include "union.h"
#include <typeinfo>

using namespace damm;

// Naive matrix-matrix union for reference
template<typename T, typename O>
void 
union_naive_matrix(T** A, T** B, T** C, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
	{
		for(size_t j = 0; j < N; ++j)
		{
			C[i][j] = O{}(A[i][j], B[i][j]);
		}
	}
}

// Naive matrix-scalar union for reference  
template<typename T, typename O>
void 
union_naive_scalar(T** A, const T B, T** C, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
	{
		for(size_t j = 0; j < N; ++j)
		{
			C[i][j] = O{}(A[i][j], B);
		}
	}
}

template <typename T, typename O>
void test_matrix_op(const char* name, const size_t M, const size_t N) 
{
	constexpr size_t ALIGN = 64;

	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B(M, N); 
	carray<T, 2, ALIGN> C_ref(M, N);    // Reference result
	carray<T, 2, ALIGN> C_block(M, N);  // union_block result
	carray<T, 2, ALIGN> C_sse(M, N);    // SSE result
	carray<T, 2, ALIGN> C_avx(M, N);    // AVX result
	carray<T, 2, ALIGN> C_avx512(M, N); // AVX512 result

	// Initialize with different values for more thorough testing
	std::fill(A.begin(), A.end(), 3.0);
	std::fill(B.begin(), B.end(), 2.0);

	// Reference naive implementation
	union_naive_matrix<T, O>(A.get(), B.get(), C_ref.get(), M, N);
	
	matrix::unite<T, O, NONE>(A.get(), B.get(), C_block.get(), M, N);
	matrix::unite<T, O, SSE>(A.get(), B.get(), C_sse.get(), M, N);
	matrix::unite<T, O, AVX>(A.get(), B.get(), C_avx.get(), M, N);
	matrix::unite<T, O, AVX512>(A.get(), B.get(), C_avx512.get(), M, N);

	bool test = true;
	test &= is_same<T>("matrix::unite<NONE>: ", C_ref.get(), C_block.get(), M, N);
	test &= is_same<T>("matrix::unite<SSE>: ", C_ref.get(), C_sse.get(), M, N);
	test &= is_same<T>("matrix::unite<AVX>: ", C_ref.get(), C_avx.get(), M, N);
	test &= is_same<T>("matrix::unite<AVX512>: ", C_ref.get(), C_avx512.get(), M, N);
	
	printf("[%-4s] union_test: matrix::unite: %s: %s\n", (test ? "OK" : "FAIL"), typeid(T).name(), name);
}

// Test matrix-scalar operations (scalar:: namespace)
template <typename T, typename O>
void test_scalar_op(const char* name, const size_t M, const size_t N) 
{
	constexpr size_t ALIGN = 64;
	const T scalar_val = 2.5;

	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> C_ref(M, N);    // Reference result
	carray<T, 2, ALIGN> C_block(M, N);  // union_block result
	carray<T, 2, ALIGN> C_sse(M, N);    // SSE result
	carray<T, 2, ALIGN> C_avx(M, N);    // AVX result
	carray<T, 2, ALIGN> C_avx512(M, N); // AVX512 result

	std::fill(A.begin(), A.end(), 4.0);

	// Reference naive implementation
	union_naive_scalar<T, O>(A.get(), scalar_val, C_ref.get(), M, N);
	
	scalar::unite<T, O, NONE>(A.get(), scalar_val, C_block.get(), M, N);
	scalar::unite<T, O, SSE>(A.get(), scalar_val, C_sse.get(), M, N);
	scalar::unite<T, O, AVX>(A.get(), scalar_val, C_avx.get(), M, N);
	scalar::unite<T, O, AVX512>(A.get(), scalar_val, C_avx512.get(), M, N);

	bool test = true;
	test &= is_same<T>("scalar::unite<NONE>", C_ref.get(), C_block.get(), M, N);
	test &= is_same<T>("scalar::unite<SSE>", C_ref.get(), C_sse.get(), M, N);
	test &= is_same<T>("scalar::unite<AVX>", C_ref.get(), C_avx.get(), M, N);
	test &= is_same<T>("scalar::unite<AVX512>", C_ref.get(), C_avx512.get(), M, N);
	
	printf("[%-4s] union_test: scalar::unite: %s: %s\n", (test ? "OK" : "FAIL"), typeid(T).name(), name);
}

template <typename O>
void test_op(const char* name, const size_t M, const size_t N) 
{
	test_matrix_op<double, O>(name, M, N);
	test_matrix_op<float, O>(name, M, N);
	test_scalar_op<double, O>(name, M, N);
	test_scalar_op<float, O>(name, M, N);

	test_matrix_op<std::complex<double>, O>(name, M, N);
	test_matrix_op<std::complex<float>, O>(name, M, N);
	test_scalar_op<std::complex<double>, O>(name, M, N);
	test_scalar_op<std::complex<float>, O>(name, M, N);
}

int main(int argc, char* argv[]) 
{
	try
	{

		const size_t M = 2048, N = 2048;
		auto ops = std::make_tuple
		(
			std::pair<std::plus<>, const char*>{ {}, "plus" },
			std::pair<std::minus<>, const char*>{ {}, "minus" },
			std::pair<std::multiplies<>, const char*>{ {}, "multiply" },
			std::pair<std::divides<>, const char*>{ {}, "divide" }
		);

		std::apply([](auto... pair) 
		{
			((test_op<typename decltype(pair)::first_type>(pair.second, M, N)), ...);
		}, ops);

	}
	catch(const std::exception& e)
	{
		std::cerr << "[EXCEPT] union_test: " << e.what() << std::endl;
	}


	return 0;
}