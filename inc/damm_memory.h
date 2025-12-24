#ifndef __DAMM_MEMORY_H__
#define __DAMM_MEMORY_H__
/**
 * \file damm_memory.h
 * \brief memory allocation definitions for damm 
 * \author cpapakonstantinou
 * \date 2025
 **/
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

#include <memory>

namespace damm
{
	/**
	 \brief Create a 2D row-major view from a flat contiguous memory block.

	 Given a pointer to a flat (1D) contiguous array of size M×N, this function constructs
	 a 2D row-major matrix view using an array of row pointers. The returned view allows 
	 accessing elements with the familiar syntax view[m][n].

	 This is a non-owning utility that does not manage the lifetime of the original data block.

	 \tparam T The data type of the array elements (e.g., float, double).
	 \param data Pointer to a contiguous array of size at least M × N.
	 \param M Number of rows.
	 \param N Number of columns.
	 \return A std::unique_ptr<T*[]> containing pointers to each row of the matrix.

	 \note The returned pointer array is owning, but the underlying data block  data must remain valid.
		This function is especially useful when working with aligned allocations or external data buffers.

	 \warning It is the caller's responsibility to ensure that data points to a valid, properly sized
		and aligned memory block, and that it outlives the returned row-pointer view.
	*/
	template<typename T>
	inline __attribute__((always_inline))
	std::unique_ptr<T*[]>
	view_as_2D(T* data, size_t M, size_t N)
	{
		auto view = std::make_unique<T*[]>(M);
		for (size_t i = 0; i < M; i++)
			view[i] = data + i * N;

		return view;
	}

	/**
	 \brief Allocates an aligned, contiguous 1D array (row-major matrix) with custom deleter.

	 This function allocates a single block of memory large enough to store an M×N matrix of type T,
	 aligned to boundary A. It returns a std::unique_ptr with a custom deleter to safely release
	 the allocated memory using free().

	 \tparam T The data type of the array elements (e.g., float, double).
	 \tparam A The memory alignment in bytes (must be a power of 2 and a multiple of sizeof(void*)).
	 \param M Number of rows.
	 \param N Number of columns.
	 \return A std::unique_ptr<T[]> to the aligned memory block.

	 \throws std::bad_alloc If memory allocation fails.

	 \note This is a simplified version of the `carray` project:
		   see https://github.com/cpapakonstantinou/carray

	 \see aligned_alloc_2D
	*/
	template<typename T, size_t A>
	inline __attribute__((always_inline))
	auto aligned_alloc_1D(size_t M, size_t N)
	{
		T* data = static_cast<T*>(aligned_alloc(A, sizeof(T) * M * N));
		if (!data)
			throw std::bad_alloc();

		auto deleter = [](T* data)
		{
			free(data);
		};

		return std::unique_ptr<T[], decltype(deleter)>(data, std::move(deleter));
	}


	/**
	 \brief Allocates an aligned, contiguous 2D array (matrix view) with row pointers and custom deleter.

	 This function allocates a contiguous aligned memory block large enough to store an M×N matrix of type T,
	 and then builds a 2D view as an array of row pointers. The returned object is a std::unique_ptr<T*[]>
	 with a custom deleter that properly cleans up both the row view and the underlying data block.

	 \tparam T The data type of the matrix elements (e.g., float, double).
	 \tparam A The memory alignment in bytes (must be a power of 2 and a multiple of sizeof(void*)).
	 \param M Number of rows.
	 \param N Number of columns.
	 \return A std::unique_ptr<T*[]> where each pointer in the array points to a row in the matrix.

	 \throws std::bad_alloc If either data or row allocation fails.

	 \note This is a simplified version of the `carray` project:
		   see https://github.com/cpapakonstantinou/carray

	 \see aligned_alloc_1D
	*/
	template<typename T, size_t A>
	inline __attribute__((always_inline))
	auto aligned_alloc_2D(size_t M, size_t N)
	{
		T* data = static_cast<T*>(aligned_alloc(A, sizeof(T) * M * N));
		if (!data)
			throw std::bad_alloc();

		auto deleter = [data](T** view)
		{
			delete[] view;
			free(data);
		};

		T** view = new T*[M];
		for (size_t i = 0; i < M; i++)
			view[i] = data + i * N;

		return std::unique_ptr<T*[], decltype(deleter)>(view, std::move(deleter));
	}

}
#endif //__DAMEMORY_H__