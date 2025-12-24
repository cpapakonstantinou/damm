#ifndef __NAIVE_H__
#define __NAIVE_H__

/**
 * \file naive.h
 * \brief Naive implementations of functions for unit and performance test
 * \author cpapakonstantinou
 * \date 2025
 */

template<typename T>
void 
broadcast_naive(T** A, const T B, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < N ; ++j )
			A[i][j] = B;
}

template<typename T>
inline void
transpose_naive(T** A, T** B, const size_t N, const size_t M)
{
	for (size_t i = 0; i < N; ++i )
		for (size_t j = 0; j < M; ++j )
			B[j][i] = A[i][j]; 
}

template<typename T>
void 
multiply_naive(T** A, T** B, T**C, const size_t M, const size_t N, const size_t P)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < P ; ++j )
			for(size_t k = 0; k < N; ++k)
				C[i][j] += A[i][k] * B[k][j];
}

template<typename T, typename O>
void 
union_naive_matrix(T** A, T** B, T** C, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < N; ++j)
			C[i][j] = O{}(A[i][j], B[i][j]);
}

template<typename T, typename O>
void 
union_naive_scalar(T** A, const T B, T** C, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < N; ++j)
			C[i][j] = O{}(A[i][j], B);
}


template <typename T, typename O>
T reduce_naive(T** A, T seed, size_t M, size_t N)
{
	T result = seed;
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			result = O()(result, A[i][j]);
	return result;
}

template <typename T, typename U, typename R>
T 
fused_reduce_naive(T** A, T** B, T seed, size_t M, size_t N)
{
	T r = seed;
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			r = R{}(r, U{}(A[i][j], B[i][j]));
	return r;
}

#endif //__NAIVE_H__