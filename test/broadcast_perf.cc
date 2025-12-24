/**
 * \file broadcast_perf.cc
 * \brief Performance test cycling through multiple broadcast kernel configurations
 */
#include "test_utils.h"
#include "carray.h"
#include "broadcast.h"

using namespace damm;

template<typename T, typename S, size_t R, size_t C>
void test_broadcast_kernel(T** A_ref, T** A_test, size_t M, size_t N)
{
	auto result = test_kernel_compute<T, S, R, C>(
	[&]() 
	{ 
		broadcast<T, S, kernel_config<R, C>::template kernel>(A_test, T(1), M, N); 
	}, A_ref, A_test, M, N);

	print_perf_result(result);
}

template<typename T, typename S>
void test_all_kernels(const size_t M, const size_t N)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	
	carray<T, 2, S::bytes> A_ref(M, N);
	carray<T, 2, S::bytes> A_test(M, N);
	
	broadcast_naive<T>(A_ref.get(), T(1), M, N);
	
	print_perf_header("Broadcast", typeid(T).name(), SIMD_WIDTH, std"GFLOPS");

	test_broadcast_kernel<T, S, 1, 1>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 1, 2>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 1, 4>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 1, 8>(A_ref.get(), A_test.get(), M, N);

	test_broadcast_kernel<T, S, 2, 1>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 2, 2>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 2, 4>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 2, 8>(A_ref.get(), A_test.get(), M, N);

	test_broadcast_kernel<T, S, 4, 1>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 4, 2>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 4, 4>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 4, 8>(A_ref.get(), A_test.get(), M, N);

	test_broadcast_kernel<T, S, 8, 1>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 8, 2>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 8, 4>(A_ref.get(), A_test.get(), M, N);
	test_broadcast_kernel<T, S, 8, 8>(A_ref.get(), A_test.get(), M, N);
	
}

int main(int argc, char* argv[])
{
	static constexpr size_t M = 1024;
	static constexpr size_t N = 1024;
	
	std::cout << "\nBroadcast Kernel Scaling Analysis AVX-512\n";	
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