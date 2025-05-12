#include <iostream>
#include <carray.h>
#include <multiply.h>

using namespace damm;

template<typename T>
void 
multiply_naive(T** A, T** B, T**C, const size_t M, const size_t N, const size_t P)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < P ; ++j )
			for(size_t k = 0; k < N; ++k)
				C[i][j] += A[i][k] * B[k][j];
}

//compare matrix A with a reference solution B
template<typename T> 
bool
is_multipled(const char* name, T** A, T** B, const size_t M, const size_t P)
{
	bool x = true;
	size_t i=0, j=0;
	for (i = 0; i < M; ++i) 
	{
		for (j = 0; j < P; ++j) 
		{
			if (A[i][j] != B[i][j]) 
			{
				x = false;
				break;
			}
		}
		if (x == false) break;
	}
	printf("[%s] %s\n", (x ? "OK" : "FAIL"), name);
	if (x == false)
		printf("Mismatch at [%zu][%zu]: A = %f, B = %f\n", i, j, (float)A[i][j], (float)B[i][j]);
	return x;
}

template<typename T>
void
print_matrix(const char* name, T**A, const size_t M, const size_t N)
{
	std::cout << name << '\n';
	for (size_t i = 0; i < M; ++i)
	{
		for(size_t j = 0; j < N ; ++j )
			std::cout << A[i][j] << " ";
		std::cout << std::endl;
	}
}
	

int main(int argc, char* argv[])
{

	using T = double;
	#define M 1024
	#define N 1024
	#define P 1024
	#define ALIGN 64
	
	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B(N, P);
	carray<T, 2, ALIGN> C(M, P);
	carray<T, 2, ALIGN> D(M, P);
	carray<T, 2, ALIGN> E(M, P);
	carray<T, 2, ALIGN> F(M, P);
	carray<T, 2, ALIGN> G(M, P);


	std::fill(A.begin(), A.end(), 2);
	std::fill(B.begin(), B.end(), 2);

	multiply_naive<T>(&A[0], &B[0], &C[0], M, N, P);

	multiply_block<T>(&A[0], &B[0], &D[0], M, N, P);
	
	multiply_block_simd<T, SSE>(&A[0], &B[0], &E[0], M, N, P);

	multiply_block_simd<T, AVX>(&A[0], &B[0], &F[0], M, N, P);

	multiply_block_simd<T, AVX512>(&A[0], &B[0], &G[0], M, N, P);

	is_multipled<T>("multiply_block", &C[0], &D[0], M, P);
	
	is_multipled<T>("multiply_block_simd<SSE>", &C[0], &E[0], M, P);

	is_multipled<T>("multiply_block_simd<AVX>", &C[0], &F[0], M, P);

	is_multipled<T>("multiply_block_simd<AVX512>", &C[0], &G[0], M, P);

	// print_matrix<T>("multiply_block", &D[0], M, P);
	// print_matrix<T>("multiply_block_simd<SSE>", &E[0], M, P);
	// print_matrix<T>("multiply_block_simd<AVX>", &F[0], M, P);
	// print_matrix<T>("multiply_block_simd<AVX512>", &G[0], M, P);

	return 0;
}