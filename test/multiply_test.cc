/**
 * \file multiply_test.cc
 * \brief unit test for multiply module
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
#include "multiply.h"
#include <complex>
#include <format>

using namespace damm;

template<typename T>
void 
multiply_naive(T** A, T** B, T**C, const size_t M, const size_t N, const size_t P)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < P ; ++j )
			for(size_t k = 0; k < N; ++k)
				C[i][j] += A[i][k] * B[k][j];
}


template<typename T>
void
test_all_ops(const size_t M, const size_t N, const size_t P)
{
	static constexpr size_t ALIGN = 64;
	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B(N, P);
	carray<T, 2, ALIGN> C_ref(M, P);
	carray<T, 2, ALIGN> C_none(M, P);
	carray<T, 2, ALIGN> C_sse(M, P);
	carray<T, 2, ALIGN> C_avx(M, P);
	carray<T, 2, ALIGN> C_avx512(M, P);

	fill_rand<T>(A.get(), M, N);

	fill_rand<T>(B.get(), M, N);

	multiply_naive<T>(A.get(), B.get(), C_ref.get(), M, N, P);

	multiply<T, NONE>(A.get(), B.get(), C_none.get(), M, N, P);
	
	multiply<T, SSE>(A.get(), B.get(), C_sse.get(), M, N, P);

	multiply<T, AVX>(A.get(), B.get(), C_avx.get(), M, N, P);

	multiply<T, AVX512>(A.get(), B.get(), C_avx512.get(), M, N, P);

	is_same<T>(std::format("multiply<{},{}>:", typeid(T).name(), "NONE").c_str(), C_ref.get(), C_none.get(), M, P);
	
	is_same<T>(std::format("multiply<{},{}>:", typeid(T).name(), "SSE").c_str(), C_ref.get(), C_sse.get(), M, P);

	is_same<T>(std::format("multiply<{},{}>:", typeid(T).name(), "AVX").c_str(), C_ref.get(), C_avx.get(), M, P);

	is_same<T>(std::format("multiply<{},{}>:", typeid(T).name(), "AVX512").c_str(), C_ref.get(), C_avx512.get(), M, P);


	// print_matrix<T>(C_ref.get(), M, P, "ref");
	// print_matrix<T>(C_sse.get(), M, P, "sse");
	// print_matrix<T>(C_avx.get(), M, P, "avx");
	// print_matrix<T>(C_avx512.get(), M, P, "avx512");
}

int main(int argc, char* argv[])
{
	static constexpr size_t M = 8;
	static constexpr size_t N = 8;
	static constexpr size_t P = 8;
	try
	{
		test_all_ops<float>(M, N, P);
		test_all_ops<double>(M, N, P);
		test_all_ops<std::complex<float>>(M, N, P);
		test_all_ops<std::complex<double>>(M, N, P);
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}