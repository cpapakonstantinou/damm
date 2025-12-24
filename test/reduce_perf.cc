/**
 * \file reduce_perf.cc
 * \brief Performance test cycling through multiple reduce kernel configurations
 */
#include "test_utils.h"
#include "carray.h"
#include "reduce.h"
using namespace damm;

template<typename T, typename S, typename O, size_t R, size_t C>
void test_reduce_kernel(T** A, T ref_result, size_t M, size_t N)
{
	T test_result;
	auto result = test_kernel_reduce<T, S, R, C>(
		[&]() 
		{ 
			test_result = reduce<T, O, S, kernel_config<R, C>::template kernel>(
				A, T(0), M, N); 
		}, 
		ref_result, test_result, M, N);
	print_perf_result(result);
}

template<typename T, typename S, typename O>
void test_all_kernels(const size_t M, const size_t N)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	
	carray<T, 2, S::bytes> A(M, N);
	
	// Use geometric series to prevent overflow
	fill_geometric<T>(A.get(), M, N, T(0.1), T(0.8));
	
	// Compute reference
	T ref_result = reduce_naive<T, O>(A.get(), T(0), M, N);
	
	std::string op_name = std::string("Reduce (") + 
		(std::is_same_v<O, std::plus<>> ? "add" : "mul") + ")";
	print_perf_header(op_name, typeid(T).name(), SIMD_WIDTH, "GFLOPS");
	
	test_reduce_kernel<T, S, O, 1, 1>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 1, 2>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 1, 4>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 1, 8>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 2, 1>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 2, 2>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 2, 4>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 2, 8>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 4, 1>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 4, 2>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 4, 4>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 4, 8>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 8, 1>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 8, 2>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 8, 4>(A.get(), ref_result, M, N);
	test_reduce_kernel<T, S, O, 8, 8>(A.get(), ref_result, M, N);
}

int main(int argc, char* argv[])
{
	static constexpr size_t M = 256;
	static constexpr size_t N = 256;
	
	std::cout << "\nReduce Kernel Scaling Analysis AVX-512\n";
	try
	{
		test_all_kernels<float, AVX512, std::plus<>>(M, N);
		test_all_kernels<double, AVX512, std::plus<>>(M, N);
		test_all_kernels<std::complex<float>, AVX512, std::plus<>>(M, N);
		test_all_kernels<std::complex<double>, AVX512, std::plus<>>(M, N);
		
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