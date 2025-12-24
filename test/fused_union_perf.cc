/**
 * \file fused_union_perf.cc
 * \brief Performance test cycling through multiple fused_union kernel configurations
 */
#include "test_utils.h"
#include "carray.h"
#include "fused_union.h"
using namespace damm;

template<typename T, typename O1, typename O2>
void fused_union_naive(T** A, T** B, T** C, T** D, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			D[i][j] = O1{}(A[i][j], O2{}(B[i][j], C[i][j]));
}

template<typename T, typename S, size_t R, size_t _C, typename O1, typename O2>
void test_fused_union_kernel(T** A, T** B, T** C, T** D_ref, T** D_test, size_t M, size_t N)
{
	auto result = test_kernel_compute<T, S, R, _C>(
		[&]() 
		{ 
			matrix::fused_union<FusionPolicy::FUSION_FIRST, T, O1, O2, S, 
				kernel_config<R, _C>::template kernel>(A, B, C, D_test, M, N); 
		}, 
		D_ref, D_test, M, N);
	print_perf_result(result);
}

template<typename T, typename S, typename O1, typename O2>
void test_all_kernels(const size_t M, const size_t N)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	
	carray<T, 2, S::bytes> A(M, N);
	carray<T, 2, S::bytes> B(M, N);
	carray<T, 2, S::bytes> C(M, N);
	carray<T, 2, S::bytes> D_ref(M, N);
	carray<T, 2, S::bytes> D_test(M, N);
	
	fill_rand<T>(A.get(), M, N);
	fill_rand<T>(B.get(), M, N);
	fill_rand<T>(C.get(), M, N);
	
	// Compute reference
	fused_union_naive<T, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), M, N);
	
	std::string op_name = std::string("Fused Union (") + typeid(O1).name() + ", " + typeid(O2).name() + ")";
	print_perf_header(op_name, typeid(T).name(), SIMD_WIDTH, "GFLOPS");
	
	test_fused_union_kernel<T, S, 1, 1, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 1, 2, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 1, 4, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 1, 8, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 2, 1, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 2, 2, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 2, 4, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 2, 8, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 4, 1, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 4, 2, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 4, 4, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
	test_fused_union_kernel<T, S, 8, 8, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), D_test.get(), M, N);
}

int main(int argc, char* argv[])
{
	static constexpr size_t M = 1024;
	static constexpr size_t N = 1024;
	
	std::cout << "\nFused Union Kernel Scaling Analysis AVX-512\n";
	try
	{
		// Test A + (B * C) pattern
		test_all_kernels<float, AVX512, std::plus<>, std::multiplies<>>(M, N);
		test_all_kernels<double, AVX512, std::plus<>, std::multiplies<>>(M, N);
		test_all_kernels<std::complex<float>, AVX512, std::plus<>, std::multiplies<>>(M, N);
		test_all_kernels<std::complex<double>, AVX512, std::plus<>, std::multiplies<>>(M, N);
	}
	catch(const std::exception& e)
	{
		std::cerr << "[Error]: " << e.what() << std::endl;
		return 1;
	}
	
	return 0;
}