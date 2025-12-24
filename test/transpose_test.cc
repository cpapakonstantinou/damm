/**
 * \file transpose_test.cc
 * \brief unit test for transpose module
 * \author cpapakonstantinou
 * \date 2025
 */
#include <numeric>
#include <complex>
#include <format>
#include "test_utils.h"
#include "carray.h"
#include "transpose.h"
using namespace damm; 


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

	// print_matrix(A.get(), M, N, "Original");
	// print_matrix(B_sse.get(), M, N, "Transposed (SSE)");
	// print_matrix(B_avx.get(), M, N, "Transposed (AVX)");
	// print_matrix(B_avx512.get(), M, N, "Transposed (AVX512)");

	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "REF").c_str(), A.get(), B_ref.get(), M, N);
	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "NONE").c_str(), A.get(), B_none.get(), M, N);
	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "SSE").c_str(), A.get(), B_sse.get(), M, N);
	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "AVX").c_str(), A.get(), B_avx.get(), M, N);
	ret &= is_transposed<T>(std::format("transpose<{},{}>", typeid(T).name(), "AVX512").c_str(), A.get(), B_avx512.get(), M, N);
	return ret;
}

int main() 
{

	static constexpr size_t M[] = {8, 64, 1024};
	static constexpr size_t N[] = {8, 64, 1024};
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
