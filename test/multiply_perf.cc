/**
 * \file multiply_perf.cc
 * \brief Performance test cycling through multiple multiply kernel configurations
 */
#include "test_utils.h"
#include "carray.h"
#include "multiply.h"
#include "broadcast.h"

#include <cstring>
using namespace damm;

template<typename T, typename S, size_t R, size_t C, float L1=0.8f, float L2=0.9f, float L3=0.5f>
void test_multiply_kernel(T** A, T** B, T** C_ref, T** C_test, size_t M, size_t N, size_t P)
{
	auto result = test_kernel_compute<T, S, R, C>(
		[A, B, C_ref, C_test, M, N, P]() 
		{ 
			// Zero out C_test before each run
			broadcast<T, S>(C_test, T(0), M, P);
			multiply<T, S, kernel_config<R, C, L1, L2, L3>::template kernel>(A, B, C_test, M, N, P); 
		}, 
		C_ref, C_test, M, N, P);

	print_perf_result(result);
}

template<typename T, typename S>
void test_all_kernels(const size_t M, const size_t N, const size_t P)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	
	carray<T, 2, S::bytes> A(M, N);
	carray<T, 2, S::bytes> B(N, P);
	carray<T, 2, S::bytes> C_ref(M, P);
	carray<T, 2, S::bytes> C_test(M, P);
	
	fill_rand<T>(A.get(), M, N);
	fill_rand<T>(B.get(), N, P);
	
	// Compute reference
	broadcast<T, S>(C_ref.get(), T(0), M, P);

	multiply_naive<T>(A.get(), B.get(), C_ref.get(), M, N, P);
	
	print_perf_header("Multiply", typeid(T).name(), SIMD_WIDTH, "GFLOPS");
	
	test_multiply_kernel<T, S, 4, 1>(A.get(), B.get(), C_ref.get(), C_test.get(), M, N, P);
	test_multiply_kernel<T, S, 4, 4>(A.get(), B.get(), C_ref.get(), C_test.get(), M, N, P);
	test_multiply_kernel<T, S, 8, 1>(A.get(), B.get(), C_ref.get(), C_test.get(), M, N, P);
	test_multiply_kernel<T, S, 8, 8>(A.get(), B.get(), C_ref.get(), C_test.get(), M, N, P);
	test_multiply_kernel<T, S, 16, 1>(A.get(), B.get(), C_ref.get(), C_test.get(), M, N, P);
	test_multiply_kernel<T, S, 16, 16>(A.get(), B.get(), C_ref.get(), C_test.get(), M, N, P);
}

template<typename T, typename S>
void test_cachesizes(const size_t M, const size_t N, const size_t P)
{

	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	
	carray<T, 2, S::bytes> A(M, N);
	carray<T, 2, S::bytes> B(N, P);
	carray<T, 2, S::bytes> C_ref(M, P);
	carray<T, 2, S::bytes> C_test(M, P);
	
	fill_rand<T>(A.get(), M, N);
	fill_rand<T>(B.get(), N, P);
	
	// Compute reference
	broadcast<T, S>(C_ref.get(), T(0), M, P);

	multiply_naive<T>(A.get(), B.get(), C_ref.get(), M, N, P);
	
	print_perf_header("Multiply", typeid(T).name(), SIMD_WIDTH, "GFLOPS");
	
	static_for<10>([&]<auto i>() 
	{
		static_for<10>([&]<auto j>() 
		{
			static_for<10>([&]<auto k>() 
			{
				constexpr float l1 = (i + 1) * 0.1f;
				constexpr float l2 = (j + 1) * 0.1f;
				constexpr float l3 = (k + 1) * 0.1f;
				
				std::cout << std::format("L1:{:.2f}, L2:{:.2f}, L3:{:.2f}", l1, l2, l3) << std::endl;
					
				test_multiply_kernel<T, S, 4, 4, l1, l2, l3>(A.get(), B.get(), C_ref.get(), C_test.get(), M, N, P);
			});
		});
	});
}

int main(int argc, char* argv[])
{
	static constexpr size_t M = 1024;
	static constexpr size_t N = 1024;
	static constexpr size_t P = 1024;
	
	std::cout << "\nMultiply Kernel Scaling Analysis AVX-512\n";
	try
	{
		test_all_kernels<float, AVX512>(M, N, P);
		test_all_kernels<double, AVX512>(M, N, P);
		test_all_kernels<std::complex<float>, AVX512>(M, N, P);
		test_all_kernels<std::complex<double>, AVX512>(M, N, P);
	
		// test_cachesizes<float, AVX512>(M, N, P);
	}
	catch(const std::exception& e)
	{
		std::cerr << "[Error]: " << e.what() << std::endl;
		return 1;
	}
	
	return 0;
}