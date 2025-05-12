#include <iostream>
#include <carray.h>
#include <union.h>

using namespace damm;

template<typename T, typename O>
void 
union_naive(T** A, T** B, T**C, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < N ; ++j )
				C[i][j] = O()(A[i][j],  B[i][j]);
}

//compare matrix A with a reference solution B
template<typename T> 
bool
is_same(const char* name, T** A, T** B, const size_t M, const size_t N)
{
	bool x = true;
	size_t i=0, j=0;
	for (i = 0; i < M; ++i) 
	{
		for (j = 0; j < N; ++j) 
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
print_matrix(const char* name, T** A, const size_t M, const size_t N)
{
	std::cout << name << '\n';
	for (size_t i = 0; i < M; ++i)
	{
		for(size_t j = 0; j < N ; ++j )
			std::cout << A[i][j] << " ";
		std::cout << std::endl;
	}
}

template <typename O>
void test_op(const char* name) 
{

	using T = double;
	constexpr size_t M = 1024, N = 1024, ALIGN = 64;

	carray<T, 2, ALIGN> A(M, N);
	carray<T, 2, ALIGN> B(M, N);
	carray<T, 2, ALIGN> C(M, N);
	carray<T, 2, ALIGN> D(M, N);
	carray<T, 2, ALIGN> E(M, N);
	carray<T, 2, ALIGN> F(M, N);
	carray<T, 2, ALIGN> G(M, N);

	std::fill(A.begin(), A.end(), 2);
	std::fill(B.begin(), B.end(), 2);

	std::cout << "=== Testing " << name << " ===\n";

	union_naive<T, O>(&A[0], &B[0], &C[0], M, N);
	union_block<T, O>(&A[0], &B[0], &D[0], M, N);
	union_block_simd<T, O, SSE>(&A[0], &B[0], &E[0], M, N);
	union_block_simd<T, O, AVX>(&A[0], &B[0], &F[0], M, N);
	union_block_simd<T, O, AVX512>(&A[0], &B[0], &G[0], M, N);

	is_same<T>("union_block:", &C[0], &D[0], M, N);
	is_same<T>("union_block_simd<SSE>:", &C[0], &E[0], M, N);
	is_same<T>("union_block_simd<AVX>:", &C[0], &F[0], M, N);
	is_same<T>("union_block_simd<AVX512>:", &C[0], &G[0], M, N);
	std::cout << "\n";
}

int main(int argc, char* argv[]) 
{
	auto ops = std::make_tuple(
		std::pair<std::plus<>, const char*>{ {}, "plus" },
		std::pair<std::minus<>, const char*>{ {}, "minus" },
		std::pair<std::multiplies<>, const char*>{ {}, "multiply" }
	);

	std::apply( [](auto... pair) 
	{
		((test_op<typename decltype(pair)::first_type>(pair.second)), ...);
	}, ops);

	return 0;
}