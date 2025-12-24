/**
 * \file union_perf.cc
 * \brief Performance test cycling through multiple union kernel configurations
 */
#include "test_utils.h"
#include "carray.h"
#include "union.h"
using namespace damm;

template<typename T, typename S, size_t R, size_t C, typename O>
void test_union_kernel(T** A, T** C_ref, T** C_test, size_t M, size_t N)
{
	auto result = test_kernel_compute<T, S, R, C>(
		[&]() 
		{ 
			scalar::unite<T, O, S, kernel_config<R, C>::template kernel>(A, T(1), C_test, M, N); 
		}, 
		C_ref, C_test, M, N);
	print_perf_result(result);
}

template<typename T, typename S, typename O>
void test_all_kernels(const size_t M, const size_t N)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	
	carray<T, 2, S::bytes> A(M, N);
	carray<T, 2, S::bytes> C_ref(M, N);
	carray<T, 2, S::bytes> C_test(M, N);
	
	fill_rand<T>(A.get(), M, N);
	
	// Compute reference
	union_naive_scalar<T, O>(A.get(), T(1), C_ref.get(), M, N);
	
	std::string op_name = "Union (" + std::string(typeid(O).name()) + ")";
	print_perf_header(op_name, typeid(T).name(), SIMD_WIDTH, "GFLOPS");
	
	test_union_kernel<T, S, 1, 1, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 1, 2, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 1, 4, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 1, 8, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 2, 1, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 2, 2, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 2, 4, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 2, 8, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 4, 1, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 4, 2, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 4, 4, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 4, 8, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 8, 1, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 8, 2, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 8, 4, O>(A.get(), C_ref.get(), C_test.get(), M, N);
	test_union_kernel<T, S, 8, 8, O>(A.get(), C_ref.get(), C_test.get(), M, N);
}

int main(int argc, char* argv[])
{
	static constexpr size_t M = 1024;
	static constexpr size_t N = 1024;
	
	std::cout << "\nUnion Kernel Scaling Analysis AVX-512\n";
	try
	{
		// Test with std::plus
		test_all_kernels<float, AVX512, std::plus<>>(M, N);
		test_all_kernels<double, AVX512, std::plus<>>(M, N);
		test_all_kernels<std::complex<float>, AVX512, std::plus<>>(M, N);
		test_all_kernels<std::complex<double>, AVX512, std::plus<>>(M, N);
		
		// Test with std::multiplies
		test_all_kernels<float, AVX512, std::multiplies<>>(M, N);
		test_all_kernels<double, AVX512, std::multiplies<>>(M, N);
		test_all_kernels<std::complex<float>, AVX512, std::multiplies<>>(M, N);
		test_all_kernels<std::complex<double>, AVX512, std::multiplies<>>(M, N);
	}
	catch(const std::exception& e)
	{
		std::cerr << "[Error]: " << e.what() << std::endl;
		return 1;
	}
	
	return 0;
}