/**
 * \file multiply_test.cc
 * \brief unit test for multiply module
 * \author cpapakonstantinou
 * \date 2025
 */
#include "test_utils.h"
#include "carray.h"
#include "multiply.h"
#include <complex>
#include <format>
#include <cstring>
#include <vector>
#include <limits>
#include <cmath>

using namespace damm;


template<typename T>
bool is_accurate(const char* name, T** A, T** B, T** C, const size_t M, const size_t N, const size_t P)
{
	using R = typename base<T>::type;
	using L = std::conditional_t<std::is_floating_point_v<T>, long double, std::complex<long double>>;
	long double max_ref = 0, max_err = 0;
	std::vector<L> ref(M * P, L(0));
	for (size_t i = 0; i < M; ++i)
		for (size_t k = 0; k < N; ++k)
		{
			const L a = L(A[i][k]);
			for (size_t j = 0; j < P; ++j)
				ref[i * P + j] += a * L(B[k][j]);
		}
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < P; ++j)
		{
			max_ref = std::max(max_ref, (long double)std::abs(ref[i * P + j]));
			max_err = std::max(max_err, (long double)std::abs(L(C[i][j]) - ref[i * P + j]));
		}
	const long double u = std::numeric_limits<R>::epsilon() / 2;
	const long double tol = 4.0L * std::sqrt((long double)N) * u * max_ref;
	const bool ok = max_err <= tol;
	printf("[%-4s] %s max|err| = %.3Le (tol %.3Le)\n", ok ? "OK" : "FAIL", name, max_err, tol);
	return ok;
}

template<typename T>
bool
test_all_ops(const size_t M, const size_t N, const size_t P)
{
	static constexpr size_t ALIGN = 64;
	bool ret = true;

	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B(N, P);
	carray<T, 2, ALIGN> C_ref(M, P);
	carray<T, 2, ALIGN> C_none(M, P);
	carray<T, 2, ALIGN> C_sse(M, P);
	carray<T, 2, ALIGN> C_avx(M, P);
	carray<T, 2, ALIGN> C_avx512(M, P);

	fill_rand<T>(A.get(), M, N);
	fill_rand<T>(B.get(), N, P);

	std::memset(C_ref.begin(), 0, M * P * sizeof(T));
	std::memset(C_none.begin(), 0, M * P * sizeof(T));
	std::memset(C_sse.begin(), 0, M * P * sizeof(T));
	std::memset(C_avx.begin(), 0, M * P * sizeof(T));
	std::memset(C_avx512.begin(), 0, M * P * sizeof(T));

	multiply_naive<T>(A.get(), B.get(), C_ref.get(), M, N, P);

	multiply<T, NONE>(A.get(), B.get(), C_none.get(), M, N, P);
		
	multiply<T, SSE>(A.get(), B.get(), C_sse.get(), M, N, P);

	multiply<T, AVX>(A.get(), B.get(), C_avx.get(), M, N, P);

	multiply<T, AVX512>(A.get(), B.get(), C_avx512.get(), M, N, P);

	ret &= is_accurate<T>(std::format("multiply<{},{}>:", typeid(T).name(), "naive").c_str(), A.get(), B.get(), C_ref.get(), M, N, P);
	ret &= is_accurate<T>(std::format("multiply<{},{}>:", typeid(T).name(), "NONE").c_str(), A.get(), B.get(), C_none.get(), M, N, P);
	ret &= is_accurate<T>(std::format("multiply<{},{}>:", typeid(T).name(), "SSE").c_str(), A.get(), B.get(), C_sse.get(), M, N, P);
	ret &= is_accurate<T>(std::format("multiply<{},{}>:", typeid(T).name(), "AVX").c_str(), A.get(), B.get(), C_avx.get(), M, N, P);
	ret &= is_accurate<T>(std::format("multiply<{},{}>:", typeid(T).name(), "AVX512").c_str(), A.get(), B.get(), C_avx512.get(), M, N, P);

	return ret;
}

int main(int argc, char* argv[])
{
	static constexpr size_t M[] = {1024};
	static constexpr size_t N[] = {1024};
	static constexpr size_t P[] = {1024};
	try
	{
		for(size_t m = 0; m < sizeof(M)/sizeof(size_t); ++m)
		{
			for(size_t n = 0; n < sizeof(N)/sizeof(size_t); ++n)
			{
				for(size_t p = 0; p < sizeof(P)/sizeof(size_t); ++p)
				{
					bool all_ops = true;
					all_ops &= test_all_ops<float>(M[m], N[n], P[p]);
					all_ops &= test_all_ops<double>(M[m], N[n], P[p]);
					all_ops &= test_all_ops<std::complex<float>>(M[m], N[n], P[p]);
					all_ops &= test_all_ops<std::complex<double>>(M[m], N[n], P[p]);
					std::string report = std::format("[{}] multiply: M={}, N={}, P={}", ((all_ops) ? "OK" : "FAIL"), M[m], N[n], P[p]);
					std::cout << report << std::endl;
				}
			}
		}
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}