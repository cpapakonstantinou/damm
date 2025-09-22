/**
 * \file transpose_test.cc
 * \brief unit test for transpose module
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
#include <numeric>
#include <complex>
#include <format>
#include "test_utils.h"
#include "carray.h"
#include "transpose.h"
using namespace damm; 

template<typename T>
inline void
transpose_naive(T** A, T** B, const size_t N, const size_t M)
{
	for (size_t i = 0; i < N; ++i )
		for (size_t j = 0; j < M; ++j )
			B[j][i] = A[i][j]; 
}

template<typename T>
bool 
is_transposed(const char* name, T** A, T** B, const size_t N, const size_t M, bool verbose = false)
{
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < M; j++)
		{
			if (verbose) {
				std::cout << "(" << i << "," << j << "): ";

				if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
					std::cout << A[i][j].real() << "+" << A[i][j].imag() << "i"
					          << " == "
					          << B[j][i].real() << "+" << B[j][i].imag() << "i\n";
				} else {
					std::cout << A[i][j] << " == " << B[j][i] << "\n";
				}
			}

			if (A[i][j] != B[j][i]) {
				if (verbose) {
					std::cout << "Mismatch at (" << i << "," << j << "): ";
					if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
						std::cout << A[i][j].real() << "+" << A[i][j].imag() << "i"
						          << " != "
						          << B[j][i].real() << "+" << B[j][i].imag() << "i\n";
					} else {
						std::cout << A[i][j] << " != " << B[j][i] << "\n";
					}
				}

				printf("[%s] %s:\n", "FAIL", name);
				return false;
			}
		}
	}
	printf("[ %s ] %s:\n", "OK", name);
	return true;
}
template<typename T>
bool
test_all_ops(const size_t M, const size_t N)
{
	static constexpr size_t ALIGN = 64;
	bool ret = true;

	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B_ref(N, M);
	carray<T, 2, ALIGN> B_none(N, M);
	carray<T, 2, ALIGN> B_sse(N, M);
	carray<T, 2, ALIGN> B_avx(N, M);
	carray<T, 2, ALIGN> B_avx512(N, M);

	fill_rand<T>(A.get(), M , N);

	transpose_naive<T>(A.get(), B_ref.get(), M, N);
	transpose<T, NONE>(A.get(), B_none.get(), M, N);
	transpose<T, SSE>(A.get(), B_sse.get(), M, N);
	transpose<T, AVX>(A.get(), B_avx.get(), M, N);
	transpose<T, AVX512>(A.get(), B_avx512.get(), M, N);

	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "REF").c_str(), A.get(), B_ref.get(), M, N);
	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "NONE").c_str(), A.get(), B_none.get(), M, N);
	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "SSE").c_str(), A.get(), B_sse.get(), M, N);
	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "AVX").c_str(), A.get(), B_avx.get(), M, N);
	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "AVX512").c_str(), A.get(), B_avx512.get(), M, N);
	return ret;
}

int main() 
{

	static constexpr size_t M[] = {8, 32, 64, 128};
	static constexpr size_t N[] = {8, 32, 64, 128};
	try
	{

		for(size_t m = 0; m < sizeof(M)/sizeof(size_t); ++m)
		{
			for(size_t n = 0; n < sizeof(N)/sizeof(size_t); ++n)
			{
				bool all_ops = true;
				all_ops &= test_all_ops<float>(M[m], N[n]);
				all_ops &= test_all_ops<double>(M[m], N[n]);
				all_ops &= test_all_ops<std::complex<float>>(M[m], N[n]);
				all_ops &= test_all_ops<std::complex<double>>(M[m], N[n]);
				std::string report = std::format("[{}] transpose: M={}, N={}", ((all_ops) ? "OK" : "FAIL"), M[m], N[n]);
				std::cout << report << std::endl;
			}
		}
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
