/**
 * \file fused_union_test.cc
 * \brief unit test for fused_union module
 * \author cpapakonstantinou
 * \date 2025
 */
#include "test_utils.h"
#include "carray.h"
#include "fused_union.h"

using namespace damm;

template<FusionPolicy P, typename T, typename O1, typename O2>
void 
fused_union_naive_scalar_lhs(T** A, const T B, T** C, T** D, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < N ; ++j )
		{
			if constexpr (P == FusionPolicy::UNION_FIRST) 
				D[i][j] = O2{}(B, O1{}(A[i][j], C[i][j]));
			else // FUSION_FIRST
				D[i][j] = O1{}(A[i][j], O2{}(B, C[i][j]));
		}
}

template<FusionPolicy P, typename T, typename O1, typename O2>
void 
fused_union_naive_scalar_rhs(T** A, T** B, const T C, T** D, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < N ; ++j )
		{
			if constexpr (P == FusionPolicy::UNION_FIRST) 
				D[i][j] = O2{}(O1{}(A[i][j], B[i][j]), C);
			else // FUSION_FIRST
				D[i][j] = O1{}(A[i][j], O2{}(B[i][j], C));
		}
}

template<FusionPolicy P, typename T, typename O1, typename O2>
void 
fused_union_naive_matrix(T** A, T** B, T** C, T** D, const size_t M, const size_t N)
{
	for(size_t i = 0; i < M; i++)
		for(size_t j = 0; j < N; j++)
		{
			if constexpr (P == FusionPolicy::UNION_FIRST) 
				D[i][j] = O2{}(O1{}(A[i][j], B[i][j]), C[i][j]);
			else // FUSION_FIRST
				D[i][j] = O1{}(A[i][j], O2{}(B[i][j], C[i][j]));
		}
}

template <typename T, FusionPolicy P, typename O1, typename O2>
void test_scalar_op_rhs(const char* name, T scalar_value = 3.5) 
{
	constexpr size_t M = 256, N = 256, ALIGN = 64;
	
	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B(M, N);
	carray<T, 2, ALIGN> D_ref(M, N);    // Reference (naive)
	carray<T, 2, ALIGN> D_none(M, N);  // Block implementation
	carray<T, 2, ALIGN> D_sse(M, N);    // SSE SIMD
	carray<T, 2, ALIGN> D_avx(M, N);    // AVX SIMD
	carray<T, 2, ALIGN> D_avx512(M, N); // AVX512 SIMD
	
	// Initialize test data
	std::fill(A.begin(), A.end(), 2.0);
	std::fill(B.begin(), B.end(), 1.5);
	
	// Run all implementations
	fused_union_naive_scalar_rhs<P, T, O1, O2>(A.get(), B.get(), scalar_value, D_ref.get(), M, N);
	scalar::fused_union<P, T, O1, O2, NONE>(A.get(), B.get(), scalar_value, D_none.get(), M, N);
	scalar::fused_union<P, T, O1, O2, SSE>(A.get(), B.get(), scalar_value, D_sse.get(), M, N);
	scalar::fused_union<P, T, O1, O2, AVX>(A.get(), B.get(), scalar_value, D_avx.get(), M, N);
	scalar::fused_union<P, T, O1, O2, AVX512>(A.get(), B.get(), scalar_value, D_avx512.get(), M, N);
	
	// Verify results
	bool test = true;
	test &= is_same<T>("  scalar::fused_union<NONE>:", D_ref.get(), D_none.get(), M, N);
	test &= is_same<T>("  scalar::fused_union<SSE>:", D_ref.get(), D_sse.get(), M, N);
	test &= is_same<T>("  scalar::fused_union<AVX>:", D_ref.get(), D_avx.get(), M, N);
	test &= is_same<T>("  scalar::fused_union<AVX512>:", D_ref.get(), D_avx512.get(), M, N);
	
	const char* policy_name = (P == FusionPolicy::UNION_FIRST) ? "UNION_FIRST" : "FUSION_FIRST";
	printf("[%s] Scalar Test RHS(%s): %s : %s\n", (test ? "OK" : "FAIL"), policy_name, typeid(T).name(), name);
}

template <typename T, FusionPolicy P, typename O1, typename O2>
void test_scalar_op_lhs(const char* name, T scalar_value = 3.5) 
{
	constexpr size_t M = 256, N = 256, ALIGN = 64;
	
	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B(M, N);
	carray<T, 2, ALIGN> D_ref(M, N);    // Reference (naive)
	carray<T, 2, ALIGN> D_none(M, N);  // Block implementation
	carray<T, 2, ALIGN> D_sse(M, N);    // SSE SIMD
	carray<T, 2, ALIGN> D_avx(M, N);    // AVX SIMD
	carray<T, 2, ALIGN> D_avx512(M, N); // AVX512 SIMD
	
	// Initialize test data
	std::fill(A.begin(), A.end(), 2.0);
	std::fill(B.begin(), B.end(), 1.5);
	
	// Run all implementations
	fused_union_naive_scalar_lhs<P, T, O1, O2>(A.get(), scalar_value, B.get(), D_ref.get(), M, N);
	scalar::fused_union<P, T, O1, O2, NONE>(A.get(), scalar_value, B.get(), D_none.get(), M, N);
	scalar::fused_union<P, T, O1, O2, SSE>(A.get(), scalar_value, B.get(), D_sse.get(), M, N);
	scalar::fused_union<P, T, O1, O2, AVX>(A.get(), scalar_value, B.get(), D_avx.get(), M, N);
	scalar::fused_union<P, T, O1, O2, AVX512>(A.get(), scalar_value, B.get(), D_avx512.get(), M, N);
	
	// Verify results
	bool test = true;
	test &= is_same<T>("  scalar::fused_union<NONE>:", D_ref.get(), D_none.get(), M, N);
	test &= is_same<T>("  scalar::fused_union<SSE>:", D_ref.get(), D_sse.get(), M, N);
	test &= is_same<T>("  scalar::fused_union<AVX>:", D_ref.get(), D_avx.get(), M, N);
	test &= is_same<T>("  scalar::fused_union<AVX512>:", D_ref.get(), D_avx512.get(), M, N);
	
	const char* policy_name = (P == FusionPolicy::UNION_FIRST) ? "UNION_FIRST" : "FUSION_FIRST";
	printf("[%s] Scalar Test LHS(%s): %s: %s\n", (test ? "OK" : "FAIL"), policy_name, typeid(T).name(), name);
}

// Test matrix fused union operations
template <typename T, FusionPolicy P, typename O1, typename O2>
void test_matrix_op(const char* name) 
{
	constexpr size_t M = 256, N = 256, ALIGN = 64;
	
	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B(M, N);
	carray<T, 2, ALIGN> C(M, N);        
	carray<T, 2, ALIGN> D_ref(M, N);    // Reference (naive)
	carray<T, 2, ALIGN> D_none(M, N);  // Block implementation
	carray<T, 2, ALIGN> D_sse(M, N);    // SSE SIMD
	carray<T, 2, ALIGN> D_avx(M, N);    // AVX SIMD
	carray<T, 2, ALIGN> D_avx512(M, N); // AVX512 SIMD
	
	// Initialize test data
	std::fill(A.begin(), A.end(), 2.0);
	std::fill(B.begin(), B.end(), 1.5);
	std::fill(C.begin(), C.end(), 0.75);
	
	// Run all implementations
	fused_union_naive_matrix<P, T, O1, O2>(A.get(), B.get(), C.get(), D_ref.get(), M, N);
	matrix::fused_union<P, T, O1, O2, NONE>(A.get(), B.get(), C.get(), D_none.get(), M, N);
	matrix::fused_union<P, T, O1, O2, SSE>(A.get(), B.get(), C.get(), D_sse.get(), M, N);
	matrix::fused_union<P, T, O1, O2, AVX>(A.get(), B.get(), C.get(), D_avx.get(), M, N);
	matrix::fused_union<P, T, O1, O2, AVX512>(A.get(), B.get(), C.get(), D_avx512.get(), M, N);
	
	// Verify results
	bool test = true;
	test &= is_same<T>("  matrix::fused_union<NONE>:", D_ref.get(), D_none.get(), M, N);
	test &= is_same<T>("  matrix::fused_union<SSE>:", D_ref.get(), D_sse.get(), M, N);
	test &= is_same<T>("  matrix::fused_union<AVX>:", D_ref.get(), D_avx.get(), M, N);
	test &= is_same<T>("  matrix::fused_union<AVX512>:", D_ref.get(), D_avx512.get(), M, N);
	
	const char* policy_name = (P == FusionPolicy::UNION_FIRST) ? "UNION_FIRST" : "FUSION_FIRST";
	printf("[%s] Matrix Test (%s): %s: %s\n", (test ? "OK" : "FAIL"), policy_name, typeid(T).name(), name);
}

template<typename T>
void test_all_ops()
{	
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::plus<>, std::plus<>>("(A + B) + D", 0.5);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::plus<>, std::minus<>>("(A + B) - D", 0.5);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::plus<>, std::multiplies<>>("(A + B) * D", 2.0);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::plus<>, std::divides<>>("(A + B) / D", 2.0);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::minus<>, std::plus<>>("(A - B) + D", 3.0);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::minus<>, std::minus<>>("(A - B) - D", 3.0);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::minus<>, std::multiplies<>>("(A - B) * D", 3.0);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::minus<>, std::divides<>>("(A - B) / D", 3.0);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::plus<>>("(A * B) + D", 2.5);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::minus<>>("(A * B) - D", 1.5);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::multiplies<>>("(A * B) * D", 1.5);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::divides<>>("(A * B) / D", 4.0);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::divides<>, std::plus<>>("(A / B) + D", 1.0);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::divides<>, std::minus<>>("(A / B) - D", 1.0);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::divides<>, std::multiplies<>>("(A / B) * D", 2.5);
	test_scalar_op_rhs<T, FusionPolicy::UNION_FIRST, std::divides<>, std::divides<>>("(A / B) / D", 2.5);

	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::plus<>, std::plus<>>("D + (A + B)", 0.5);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::plus<>, std::minus<>>("D - (A + B)", 0.5);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::plus<>, std::multiplies<>>("D * (A + B)", 2.0);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::plus<>, std::divides<>>("D / (A + B)", 2.0);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::minus<>, std::plus<>>("D + (A - B)", 3.0);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::minus<>, std::minus<>>("D - (A - B)", 3.0);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::minus<>, std::multiplies<>>("D * (A - B)", 3.0);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::minus<>, std::divides<>>("D / (A - B)", 3.0);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::plus<>>("D + (A * B)", 2.5);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::minus<>>("D - (A * B)", 1.5);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::multiplies<>>("D * (A * B)", 1.5);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::divides<>>("D / (A * B)", 4.0);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::divides<>, std::plus<>>("D + (A / B)", 1.0);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::divides<>, std::minus<>>("D - (A / B)", 1.0);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::divides<>, std::multiplies<>>("D * (A / B)", 2.5);
	test_scalar_op_lhs<T, FusionPolicy::UNION_FIRST, std::divides<>, std::divides<>>("D / (A / B)", 2.5);

	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::plus<>>("(A + B) + D", 0.5);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::minus<>>("(A + B) - D", 0.5);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::multiplies<>>("(A + B) * D", 2.0);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::divides<>>("(A + B) / D", 2.0);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::plus<>>("(A - B) + D", 3.0);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::minus<>>("(A - B) - D", 3.0);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::multiplies<>>("(A - B) * D", 3.0);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::divides<>>("(A - B) / D", 3.0);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::plus<>>("(A * B) + D", 2.5);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::minus<>>("(A * B) - D", 1.5);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::multiplies<>>("(A * B) * D", 1.5);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::divides<>>("(A * B) / D", 4.0);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::plus<>>("(A / B) + D", 1.0);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::minus<>>("(A / B) - D", 1.0);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::multiplies<>>("(A / B) * D", 2.5);
	test_scalar_op_rhs<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::divides<>>("(A / B) / D", 2.5);

	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::plus<>>("D + (A + B)", 0.5);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::minus<>>("D - (A + B)", 0.5);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::multiplies<>>("D * (A + B)", 2.0);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::divides<>>("D / (A + B)", 2.0);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::plus<>>("D + (A - B)", 3.0);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::minus<>>("D - (A - B)", 3.0);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::multiplies<>>("D * (A - B)", 3.0);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::divides<>>("D / (A - B)", 3.0);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::plus<>>("D + (A * B)", 2.5);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::minus<>>("D - (A * B)", 1.5);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::multiplies<>>("D * (A * B)", 1.5);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::divides<>>("D / (A * B)", 4.0);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::plus<>>("D + (A / B)", 1.0);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::minus<>>("D - (A / B)", 1.0);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::multiplies<>>("D * (A / B)", 2.5);
	test_scalar_op_lhs<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::divides<>>("D / (A / B)", 2.5);
		
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::plus<>, std::plus<>>("(A + B) + D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::plus<>, std::minus<>>("(A + B) - D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::plus<>, std::multiplies<>>("(A + B) * D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::plus<>, std::divides<>>("(A + B) / D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::minus<>, std::plus<>>("(A - B) + D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::minus<>, std::minus<>>("(A - B) - D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::minus<>, std::multiplies<>>("(A - B) * D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::minus<>, std::divides<>>("(A - B) / D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::plus<>>("(A * B) + D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::minus<>>("(A * B) - D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::multiplies<>>("(A * B) * D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::multiplies<>, std::divides<>>("(A * B) / D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::divides<>, std::plus<>>("(A / B) + D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::divides<>, std::minus<>>("(A / B) - D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::divides<>, std::multiplies<>>("(A / B) * D");
	test_matrix_op<T, FusionPolicy::UNION_FIRST, std::divides<>, std::divides<>>("(A / B) / D");

	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::plus<>>("(A + B) + D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::minus<>>("(A + B) - D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::multiplies<>>("(A + B) * D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::plus<>, std::divides<>>("(A + B) / D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::plus<>>("(A - B) + D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::minus<>>("(A - B) - D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::multiplies<>>("(A - B) * D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::minus<>, std::divides<>>("(A - B) / D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::plus<>>("(A * B) + D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::minus<>>("(A * B) - D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::multiplies<>>("(A * B) * D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::multiplies<>, std::divides<>>("(A * B) / D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::plus<>>("(A / B) + D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::minus<>>("(A / B) - D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::multiplies<>>("(A / B) * D");
	test_matrix_op<T, FusionPolicy::FUSION_FIRST, std::divides<>, std::divides<>>("(A / B) / D");
}	

int main(int argc, char* argv[]) 
{
	try
	{
		test_all_ops<float>();
		test_all_ops<double>();
		test_all_ops<std::complex<float>>();
		test_all_ops<std::complex<double>>();
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
	
	return 0;
}