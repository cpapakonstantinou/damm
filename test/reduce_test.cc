#include <iostream>
#include <cmath>
#include <type_traits>
#include <functional>
#include <tuple>
#include <carray.h>
#include <reduce.h>

using namespace damm;

// Operator names
template <typename Op>
constexpr const char* op_name();
template <> constexpr const char* op_name<std::plus<>>()       { return "plus"; }
template <> constexpr const char* op_name<std::minus<>>()      { return "minus"; }
template <> constexpr const char* op_name<std::multiplies<>>() { return "multiply"; }
template <> constexpr const char* op_name<std::divides<>>()    { return "divide"; }

// Operator support trait per SIMD
template <typename Op, SIMD S>
constexpr bool is_supported = false;
// NONE
template <> constexpr bool is_supported<std::plus<>, NONE> = true;
template <> constexpr bool is_supported<std::multiplies<>, NONE> = true;
// SSE
template <> constexpr bool is_supported<std::plus<>, SSE> = true;
template <> constexpr bool is_supported<std::multiplies<>, SSE> = true;
// AVX
template <> constexpr bool is_supported<std::plus<>, AVX> = true;
template <> constexpr bool is_supported<std::multiplies<>, AVX> = true;
// AVX512
template <> constexpr bool is_supported<std::plus<>, AVX512> = true;
template <> constexpr bool is_supported<std::multiplies<>, AVX512> = true;


template <typename T, typename O>
T reduce_naive(T** A, T seed, size_t M, size_t N) 
{
	T result = seed;
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			result = O()(result, A[i][j]);
	return result;
}

// Fuzzy comparison
template <typename T>
bool approx_equal(T a, T b, T tol = static_cast<T>(1e-4)) {
	return std::abs(a - b) <= tol;
}

template <typename T, typename O, SIMD S>
void run_if_supported(const char* simd_name, const size_t M, const size_t N)
{
	if constexpr (is_supported<O, S>) 
	{
		constexpr size_t ALIGN = 64;
		carray<T, 2, ALIGN> A(M, N);


		//Here we want to generate some series that we know through analytical analysis will converge
		//i.e. a specific geometric series. 
		if constexpr (std::is_same_v<O, std::plus<>> || std::is_same_v<O, std::minus<>>) 
		{
			T val = 1.0;
			for (auto it = A.begin(); it != A.end(); ++it) 
			{
				*it = val;
				val *= 0.9;
			}
		} 
		else 
		{
			T val = std::pow(static_cast<T>(2.0), static_cast<T>(1.0) / static_cast<T>(M * N));
			for (auto it = A.begin(); it != A.end(); ++it)
				*it = val;
		}

		T seed = seed_left_fold<T, O>();
		T ref   = reduce_naive<T, O>(&A[0], seed, M, N);
		T result;

		if constexpr (S == NONE)
			result = reduce_block<T, O>(&A[0], seed, M, N);
		else if constexpr (S == SSE)
			result = reduce_block_simd<T, O, SSE>(&A[0], seed, M, N);
		else if constexpr (S == AVX)
			result = reduce_block_simd<T, O, AVX>(&A[0], seed, M, N);
		else if constexpr (S == AVX512)
			result = reduce_block_simd<T, O, AVX512>(&A[0], seed, M, N);

		// Report result
		bool ok = approx_equal(ref, result);
		std::cout << "[" << (ok ? "OK" : "FAIL") << "] "
				  << "T=" << (std::is_same_v<T, float> ? "float" : "double")
				  << " op=" << op_name<O>()
				  << " simd=" << simd_name
				  << " ref=" << ref << " res=" << result << "\n";
	}
}

template <typename... Os>
void test_all_ops() 
{
	//tester logic for multiplies limits the max M, N for floats
	constexpr size_t M = 256, N = 256; 
	((
		run_if_supported<float, Os, NONE>("NONE", M, N),
		run_if_supported<double, Os, NONE>("NONE", M, N),
		run_if_supported<float, Os, SSE>("SSE", M, N),
		run_if_supported<double, Os, SSE>("SSE", M, N),
		run_if_supported<float, Os, AVX>("AVX", M, N),
		run_if_supported<double, Os, AVX>("AVX", M, N),
		run_if_supported<float, Os, AVX512>("AVX512", M, N),
		run_if_supported<double, Os, AVX512>("AVX512", M, N)
	), ...);
}

int main() 
{
	try
	{
		test_all_ops<
			std::plus<>,
			std::multiplies<>,
			std::minus<>,
			std::divides<>
		>();
	}
	catch( ... )
	{
		std::cerr << "[FAIL] Tester" << std::endl;
	}

	return 0;
}

