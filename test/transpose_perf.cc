/**
 * \file transpose_perf.cc
 * \brief Performance test cycling through multiple transpose kernel configurations
 */
#include "test_utils.h"
#include "carray.h"
#include "transpose.h"
using namespace damm;

template<typename T, typename S, size_t R, size_t C>
void test_transpose_kernel(T** A, T** B_ref, T** B_test, size_t M, size_t N)
{
	auto result = test_kernel_compute<T, S, R, C>(
		[&]() 
		{ 
			transpose<T, S, kernel_config<R, C>::template kernel>(A, B_test, M, N); 
		}, 
		B_ref, B_test, N, M);  // Note: transposed dimensions
	print_perf_result(result);
}

template<typename T, typename S>
void test_all_kernels(const size_t M, const size_t N)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	
	carray<T, 2, S::bytes> A(M, N);
	carray<T, 2, S::bytes> B_ref(N, M);
	carray<T, 2, S::bytes> B_test(N, M);
	
	fill_rand<T>(A.get(), M, N);
	
	// Compute reference
	transpose_naive<T>(A.get(), B_ref.get(), M, N);
	
	print_perf_header("Transpose", typeid(T).name(), SIMD_WIDTH, "GFLOPS");
	
	test_transpose_kernel<T, S, 1*SIMD_WIDTH, 1>(A.get(), B_ref.get(), B_test.get(), M, N);
	
}

int main(int argc, char* argv[])
{
	static constexpr size_t M = 1024;
	static constexpr size_t N = 1024;
	
	std::cout << "\nTranspose Kernel Scaling Analysis AVX-512\n";
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