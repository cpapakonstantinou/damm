#include <iostream>
#include <carray.h>
#include <numeric>
#include <transpose.h>

using namespace damm; 

template<typename T>
inline void
transpose_naive(T** A, T** B, const size_t N, const size_t M)
{
	for (size_t i = 0; i < N; ++i )
		for (size_t j = 0; j < M; ++j )
			B[j][i] = A[i][j]; 
}

template<typename T>
bool 
is_transposed(const char* name, T** A, T** B, const size_t N, const size_t M, bool verbose = false)
{
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < M; j++)
		{
			if(verbose)
				printf("(%lu,%lu): %g == %g\n", i, j, A[i][j], B[j][i]);
			
			if (A[i][j] != B[j][i])
			{
				if(verbose)
				{
					printf("Mismatch at (%lu,%lu): %g != %g\n", i, j, A[i][j], B[j][i]);
				}
				printf("%s ? %s\n", name, "false");
				return false;
			}
		}
	}
	printf("%s ? %s\n", name, "true");
	return true;
}

int main() 
{
	using T = float;
	static constexpr size_t ALIGN = 64;
	static constexpr size_t N = 2048;
	static constexpr size_t M = 2048;

	carray<T, 2, ALIGN> A(N, M);
	carray<T, 2, ALIGN> B(M, N);
	carray<T, 2, ALIGN> C(M, N);
	carray<T, 2, ALIGN> D(M, N);
	carray<T, 2, ALIGN> E(M, N);
	carray<T, 2, ALIGN> F(M, N);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			A[i][j] = i * N + j;

	transpose_naive<T>(&A[0], &B[0], N, M);
	transpose_block<T>(&A[0], &C[0], N, M);
	transpose_block_simd<T, SSE>(&A[0], &D[0], N, M);
	transpose_block_simd<T, AVX>(&A[0], &E[0], N, M);
	transpose_block_simd<T, AVX512>(&A[0], &F[0], N, M);

	is_transposed<T>("transpose_naive", &A[0], &B[0], N, M);
	is_transposed<T>("transpose_block", &A[0], &C[0], N, M);
	is_transposed<T>("transpose_block_simd<SSE>", &A[0], &D[0], N, M);
	is_transposed<T>("transpose_block_simd<AVX>", &A[0], &E[0], N, M);
	is_transposed<T>("transpose_block_simd<AVX_512>", &A[0], &F[0], N, M);
	return 0;
}
