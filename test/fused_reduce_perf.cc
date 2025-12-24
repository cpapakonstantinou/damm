/**
 * \file fused_reduce_perf.cc
 * \brief Performance test cycling through multiple fused_reduce kernel configurations
 */
#include "test_utils.h"
#include "carray.h"
#include "fused_reduce.h"
using namespace damm;

template<typename T, typename S, size_t R, size_t C, typename U, typename Reduce>
void test_fused_reduce_kernel(T** A, T** B, T ref_result, size_t M, size_t N)
{
	T test_result = T(0);
	
	auto result = test_kernel_compute<T, S, R, C>(
		[&]() 
		{ 
			test_result = fused_reduce<T, U, Reduce, S, kernel_config<R, C>::template kernel>(
				A, B, T(0), M, N); 
		}, 
		nullptr, nullptr, M, N);
	
	// Manual verification for scalar result
	result.verified = approx_equal(ref_result, test_result);
	print_perf_result(result);
}

template<typename T, typename S>
void test_all_kernels(const size_t M, const size_t N)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	
	carray<T, 2, S::bytes> A(M, N);
	carray<T, 2, S::bytes> B(M, N);
	
	fill_geometric<T>(A.get(), M, N, T(1), T(0.99));
	fill_geometric<T>(B.get(), M, N, T(1), T(0.99));
	
	// Compute reference (dot product pattern)
	T ref_result = fused_reduce_naive<T, std::multiplies<>, std::plus<>>(
		A.get(), B.get(), T(0), M, N);
	
	print_perf_header("Fused Reduce (Dot Product)", typeid(T).name(), SIMD_WIDTH, "GFLOPS");
	
	test_fused_reduce_kernel<T, S, 1, 1, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 1, 2, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 1, 4, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 1, 8, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 2, 1, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 2, 2, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 2, 4, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 2, 8, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 4, 1, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 4, 2, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 4, 4, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
	test_fused_reduce_kernel<T, S, 4, 8, std::multiplies<>, std::plus<>>(A.get(), B.get(), ref_result, M, N);
}

int main(int argc, char* argv[])
{
	static constexpr size_t M = 1024;
	static constexpr size_t N = 1024;
	
	std::cout << "\nFused Reduce Kernel Scaling Analysis AVX-512\n";
	try
	{
		test_all_kernels<float, AVX512>(M, N);
		test_all_kernels<double, AVX512>(M, N);
		test_all_kernels<std::complex<float>, AVX512>(M, N);
		test_all_kernels<std::complex<double>, AVX512>(M, N);
	}
	catch(const std::exception& e)
	{
		std::cerr << "[Error]: " << e.what() << std::endl;
		return 1;
	}
	
	return 0;
}