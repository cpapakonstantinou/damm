/**
 * \file broadcast_test.cc
 * \brief unit test for broadcast module
 * \author cpapakonstantinou
 * \date 2025
 */
// Copyright (c) 2025  Constantine Papakonstantinou
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#include <iostream>
#include <format>
#include <carray.h>
#include <broadcast.h>
#include "test_utils.h"

using namespace damm;

template<typename T>
void 
broadcast_naive(T** A, const T B, const size_t M, const size_t N)
{
	for (size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < N ; ++j )
			A[i][j] = B;
}

template<typename T>
void
identity_naive(T** A, const size_t M, const size_t N)
{
	for(size_t i = 0; i < M; ++i)
		for(size_t j = 0; j < N; ++j)
			A[i][j]= ( ( i == j) ? T(1) : T(0) );
}


template<typename T> 
bool
is_identity(const char* name, T** A, const size_t M, const size_t N)
{
	bool x = true;
	size_t i=0, j=0;
	for (i = 0; i < M; ++i) 
	{
		for (j = 0; j < N; ++j) 
		{
			if(i == j)
				x = A[i][j];
			else
				x = !A[i][j];
		}
		if (x == false) break;
	}
	printf("[%-4s] %s\n", (x ? "OK" : "FAIL"), name);
	return x;
}

int main(int argc, char* argv[])
{

	using T = double;
	#define M 2048
	#define N 2048
	#define ALIGN 64

	T B = 1;
	
	carray<T, 2, ALIGN> A_fill(M, N);
	carray<T, 2, ALIGN> A_naive(M, N);
	carray<T, 2, ALIGN> A_none(M, N);
	carray<T, 2, ALIGN> A_sse(M, N);
	carray<T, 2, ALIGN> A_avx(M, N);
	carray<T, 2, ALIGN> A_avx512(M, N);
	carray<T, 2, ALIGN> I(M, N);
	
	std::fill(A_fill.begin(), A_fill.end(), B);	
	
	broadcast_naive<T>(A_naive.get(), B, M, N);	
	broadcast<T, NONE>(A_none.get(), B, M, N);
	broadcast<T, SSE>(A_sse.get(), B, M, N);
	broadcast<T, AVX>(A_avx.get(), B, M, N);
	broadcast<T, AVX512>(A_avx512.get(), B, M, N);

	is_same<T>(std::format("broadcast<{}, NONE>:",typeid(T).name()).c_str(), A_none.get(), A_naive.get(), M, N);
	is_same<T>(std::format("broadcast<{}, SSE>:",typeid(T).name()).c_str(), A_sse.get(), A_naive.get(), M, N);
	is_same<T>(std::format("broadcast<{}, AVX>:",typeid(T).name()).c_str(), A_avx.get(), A_naive.get(), M, N);
	is_same<T>(std::format("broadcast<{}, AVX512>:",typeid(T).name()).c_str(), A_avx512.get(), A_naive.get(), M, N);

	std::complex<T> Bz{1., 2.};
	
	carray<std::complex<T>, 2, ALIGN> Az_fill(M, N);
	carray<std::complex<T>, 2, ALIGN> Az_naive(M, N);
	carray<std::complex<T>, 2, ALIGN> Az_none(M, N);
	carray<std::complex<T>, 2, ALIGN> Az_sse(M, N);
	carray<std::complex<T>, 2, ALIGN> Az_avx(M, N);
	carray<std::complex<T>, 2, ALIGN> Az_avx512(M, N);

	std::fill(Az_fill.begin(), Az_fill.end(), Bz);	
	
	broadcast_naive<std::complex<T>>(Az_naive.get(), Bz, M, N);	
	broadcast<std::complex<T>, NONE>(Az_none.get(), Bz, M, N);
	broadcast<std::complex<T>, SSE>(Az_sse.get(), Bz, M, N);
	broadcast<std::complex<T>, AVX>(Az_avx.get(), Bz, M, N);
	broadcast<std::complex<T>, AVX512>(Az_avx512.get(), Bz, M, N);

	is_same<std::complex<T>>(std::format("broadcast<complex<{}>, NONE>:",typeid(T).name()).c_str(), Az_none.get(), Az_naive.get(), M, N);
	is_same<std::complex<T>>(std::format("broadcast<complex<{}>, SSE>:",typeid(T).name()).c_str(), Az_sse.get(), Az_naive.get(), M, N);
	is_same<std::complex<T>>(std::format("broadcast<complex<{}>, AVX>:",typeid(T).name()).c_str(), Az_avx.get(), Az_naive.get(), M, N);
	is_same<std::complex<T>>(std::format("broadcast<complex<{}>, AVX512>:",typeid(T).name()).c_str(), Az_avx512.get(), Az_naive.get(), M, N);

	identity_naive<T>(A_naive.get(), M, N);
	identity<T, NONE>(A_none.get(), M, N);
	identity<T, SSE>(A_sse.get(), M, N);
	identity<T, AVX>(A_avx.get(), M, N);
	identity<T, AVX512>(A_avx512.get(), M, N);

	is_identity<T>("identity_naive:", A_naive.get(), M, N);
	is_identity<T>(std::format("identity<{} ,NONE>:", typeid(T).name()).c_str(), A_none.get(), M, N);
	is_identity<T>(std::format("identity<{} ,SSE>:", typeid(T).name()).c_str(), A_sse.get(), M, N);
	is_identity<T>(std::format("identity<{} ,AVX>:", typeid(T).name()).c_str(), A_avx.get(), M, N);
	is_identity<T>(std::format("identity<{} ,AVX512>:", typeid(T).name()).c_str(), A_avx512.get(), M, N);

	set_identity(I.get(), M, N);
	is_identity<T>("set_identity:", I.get(), M, N);

	return 0;
}