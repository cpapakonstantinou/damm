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
void
test_all_ops(const size_t M, const size_t N)
{
	static constexpr size_t ALIGN = 64;

	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B(N, M);
	carray<T, 2, ALIGN> C(N, M);
	carray<T, 2, ALIGN> D(N, M);
	carray<T, 2, ALIGN> E(N, M);
	carray<T, 2, ALIGN> F(N, M);

	fill_rand<T>(A.get(), M , N);

	transpose_naive<T>(A.get(), B.get(), M, N);
	transpose<T, NONE>(A.get(), C.get(), M, N);
	transpose<T, SSE>(A.get(), D.get(), M, N);
	transpose<T, AVX>(A.get(), E.get(), M, N);
	transpose<T, AVX512>(A.get(), F.get(), M, N);

	is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "REF").c_str(), A.get(), B.get(), M, N);
	is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "NONE").c_str(), A.get(), C.get(), M, N);
	is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "SSE").c_str(), A.get(), D.get(), M, N);
	is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "AVX").c_str(), A.get(), E.get(), M, N);
	is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "AVX512").c_str(), A.get(), F.get(), M, N);
}
int main() 
{
	static constexpr size_t M = 2;
	static constexpr size_t N = 2;

	try
	{
		test_all_ops<float>(M, N);
		test_all_ops<double>(M, N);
		test_all_ops<std::complex<float>>(M, N);
		test_all_ops<std::complex<double>>(M, N);
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
