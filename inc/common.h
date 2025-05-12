#ifndef __COMMON_H__
#define __COMMON_H__
/**
 * \file common.h
 * \brief common definitions for libmm 
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

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <ranges>
#include <expected>
#include <complex>
#include <async.h>


namespace damm
{
	#define DAMM_THREADS 1
	#define DAMM_BLOCK_SIZE 64

	static constexpr size_t _threads = DAMM_THREADS; ///< Set to equivalent of hardware_concurrency() at compile time
	static constexpr size_t _block_size = DAMM_BLOCK_SIZE; ///< Set to L1 cache line size at compile time

	/**
	 * \brief The register size in bytes for a given SIMD architecture.
	 * 
	 *  This enumeration is used to instantiate / branch templates for the SIMD architecture.
	 **/ 
	enum SIMD
	{
		NONE = 8,
		SSE = 16,
		AVX = 32,
		AVX512 = 64
	};

	// Compile-time SIMD detection
	consteval SIMD detect_simd() 
	{
		#ifdef __AVX512F__
			return SIMD::AVX512;
		#elif defined(__AVX__)
			return SIMD::AVX;
		#elif defined(__SSE__)
			return SIMD::SSE;
		#else
			return SIMD::NONE;
		#endif
	}

	template <typename T>
	struct value 
	{
		using type = T;
	};

	template <typename T>
	struct value<std::complex<T>> 
	{
		using type = T;
	};

	//not SFINAE, this assumes the alternative is an stl compliant T, with ::value_type
	//add SFINAE to this if needed
	template<typename T>
	consteval std::size_t sizeof_v() 
	{
		if constexpr (!std::is_arithmetic_v<T>) 
			return sizeof(typename T::value_type);
		else return sizeof(T);
	}

	enum TRIANGULAR
	{
		UPPER,
		LOWER
	};

	/**
	 \brief Create a 2D row-major view from a flat contiguous memory block.

	 Given a pointer to a flat (1D) contiguous array of size M×N, this function constructs
	 a 2D row-major matrix view using an array of row pointers. The returned view allows 
	 accessing elements with the familiar syntax \c view[m][n].

	 This is a non-owning utility that does not manage the lifetime of the original data block.

	 \tparam T The data type of the array elements (e.g., float, double).
	 \param data Pointer to a contiguous array of size at least M × N.
	 \param M Number of rows.
	 \param N Number of columns.
	 \return A \c std::unique_ptr<T*[]> containing pointers to each row of the matrix.

	 \note The returned pointer array is owning, but the underlying data block (\c data) must remain valid.
		This function is especially useful when working with aligned allocations or external data buffers.

	 \warning It is the caller's responsibility to ensure that \c data points to a valid, properly sized
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

	 This function allocates a single block of memory large enough to store an M×N matrix of type \c T,
	 aligned to boundary \c A. It returns a \c std::unique_ptr with a custom deleter to safely release
	 the allocated memory using \c free().

	 \tparam T The data type of the array elements (e.g., float, double).
	 \tparam A The memory alignment in bytes (must be a power of 2 and a multiple of \c sizeof(void*)).
	 \param M Number of rows.
	 \param N Number of columns.
	 \return A \c std::unique_ptr<T[]> to the aligned memory block.

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

	 This function allocates a contiguous aligned memory block large enough to store an M×N matrix of type \c T,
	 and then builds a 2D view as an array of row pointers. The returned object is a \c std::unique_ptr<T*[]>
	 with a custom deleter that properly cleans up both the row view and the underlying data block.

	 \tparam T The data type of the matrix elements (e.g., float, double).
	 \tparam A The memory alignment in bytes (must be a power of 2 and a multiple of \c sizeof(void*)).
	 \param M Number of rows.
	 \param N Number of columns.
	 \return A \c std::unique_ptr<T*[]> where each pointer in the array points to a row in the matrix.

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


	/* \brief set identity matrix.
	 * the naive approach would be to implement a dirac-delta using an (if i == j) branch
	 * since the memory layout is known a priori, take advantage of it to skip indices
	 * \note requires a zero filled matrix 
	 */
	template<typename T>
	void
	set_identity(T* I, const size_t M, const size_t N)
	{	
		size_t idx;
		do
		{
			(I + idx) = T(1);
			idx += (N+1); //row-major formulation
		}while(idx < (M*N));
	}

	/* \brief set identity matrix.
	 * the naive approach would be to implement a dirac-delta using an (if i == j) branch
	 * skip some indices by forcing a sequence where i == j 
	 * \note requires a zero filled matrix
	 */
	template<typename T>
	void
	set_identity(T** I, const size_t M, const size_t N)
	{	
		for(size_t i=0, j=0; i < M && j < N; i++, j++)
			I[i][j] = T(1);
	}

	/**
	 \brief Helper variable template that always evaluates to false.

	 This is a utility used in \c static_assert to trigger a compile-time error 
	 in non-instantiated branches of a \c constexpr if or SFINAE context.

	 \tparam T The type to associate with the false condition.
	*/
	template <typename>
	inline constexpr bool always_false = false;

	/**
	 \brief Return the identity element (seed) for a given binary operation.

	 This function provides a neutral element for left-fold operations such as 
	 addition, subtraction, multiplication, or division. It is primarily intended 
	 to initialize an accumulator correctly for fold/reduction operations.

	 \tparam T The type of the value (e.g., float, int, double).
	 \tparam O The binary operation type (e.g., \c std::plus<>, \c std::multiplies<>).

	 \return The identity value of type \c T for the operation \c O.

	 \note The supported operations are:
	 - \c std::plus<> and \c std::minus<> → returns 0
	 - \c std::multiplies<> and \c std::divides<> → returns 1

	 \throws A compile-time error if an unsupported operation is used.

	 \code
	 float acc = seed_left_fold<float, std::plus<>>();  // returns 0.0f
	 int product = seed_left_fold<int, std::multiplies<>>();  // returns 1
	 \endcode

	 \warning This function will fail to compile if used with an unsupported binary operation.
	*/
	template <typename T, typename O>
	constexpr T seed_left_fold() 
	{
		if constexpr (std::same_as<O, std::plus<>> || std::same_as<O, std::minus<>>)
			return T(0);
		else if constexpr (std::same_as<O, std::multiplies<>> || std::same_as<O, std::divides<>>)
			return T(1);
		else
			static_assert(always_false<O>, "Unsupported operation for fold");
	}

	/**
	 * \brief Executes a parallel for-loop over a stepped range [begin, end) with specified step size.
	 * 
	 * This function divides the iteration space into chunks processed asynchronously using multiple threads.
	 * It internally uses C++23 ranges (`std::views::iota` + `std::views::stride`) to lazily generate the sequence
	 * of indices stepped by `step`, and then dispatches the work via `async_for_each`.
	 * 
	 * \tparam F The callable type that will be invoked for each index in the range. 
	 *           Signature should be `void(size_t)`.
	 * \tparam P The type of the progress callback. Defaults to `std::function<void(size_t)>`. 
	 *           It receives the number of completed chunks and can be used for progress reporting.
	 * 
	 * \param begin   The starting index of the iteration range (inclusive).
	 * \param end     The ending index of the iteration range (exclusive).
	 * \param step    The step size between consecutive indices (must be > 0).
	 * \param f       The function to apply to each stepped index.
	 * \param threads The number of concurrent threads to use. Defaults to hardware concurrency.
	 * \param progress Optional progress callback invoked with number of completed chunks (default no-op).
	 * 
	 * \throws std::invalid_argument if step == 0.
	 * \throws Any exceptions thrown by the callable `f` will be proparightd after all threads finish.
	 * 
	 * \note Requires C++23 for `std::views::stride`. Uses `async_for_each` internally for parallelism.
	 */
	template<typename F, typename P = std::function<void(size_t)>>
	void 
	parallel_for(size_t begin, size_t end, size_t step, F&& f, size_t threads = _threads, P&& progress = [](size_t) {})
	{
		auto stepped_range = std::views::iota(begin, end) | std::views::stride(step);

		async::async_for_each(
			std::ranges::begin(stepped_range), std::ranges::end(stepped_range),
			std::forward<F>(f),
			threads,
			std::forward<P>(progress)
		);
	}

	template<typename T>
	std::expected<void, std::string_view> 
	_right(T** ptr, size_t rows, size_t cols)
	{
		if (!ptr)
			return std::unexpected{"null pointer"};
		
		const size_t size = rows * cols;
		if (size < rows || size < cols)
			return std::unexpected{"dimension overflow"};
		
		if (size > std::numeric_limits<size_t>::max() / sizeof(T)) 
			return std::unexpected{"dimensions too large"};
		
		if ((ptr[rows - 1] + cols) - ptr[0] != static_cast<std::ptrdiff_t>(size))
			return std::unexpected{"not contiguous"};
			
		return {};
	}

	template<typename T>
	std::expected<void, std::string_view> 
	_right(T* ptr, size_t rows, size_t cols) 
	{
		if (!ptr) return std::unexpected{"null pointer"};
		
		const size_t size = rows * cols;
		if (size < rows || size < cols) return std::unexpected{"dimension overflow"};
		
		if (size > std::numeric_limits<size_t>::max() / sizeof(T))
			return std::unexpected{"dimensions too large"};

		if (reinterpret_cast<uintptr_t>(ptr + size) < reinterpret_cast<uintptr_t>(ptr))
			return std::unexpected{"memory wraparound"};
			
		return {};
	}

	// Variadic guard that uses expected internally but throws externally
	template<typename T, typename... Args>
	void 
	right(const char* id, const std::tuple<T**, size_t, size_t>& matrix, const Args&... matrices) 
	{
		auto [ptr, rows, cols] = matrix;
		
		auto result = _right(ptr, rows, cols);
		if (!result) [[unlikely]]
			throw std::runtime_error(id + std::string(result.error()));
		
		[[assume(ptr != nullptr)]];
		[[assume(rows > 0)]];
		[[assume(cols > 0)]];  
		[[assume(ptr[0] != nullptr)]];

		if (rows > 1)
			[[assume(ptr[rows-1] != nullptr)]];
		
		if constexpr (sizeof...(matrices) > 0)
			right<T>(id, matrices...);
	}

	template<typename T, typename... Args>
	void 
	right(const char* id, const std::tuple<T*, size_t, size_t>& matrix, const Args&... matrices) 
	{
		auto [ptr, rows, cols] = matrix;
		
		auto result = _right(ptr, rows, cols);
		if (!result) [[unlikely]] 
			throw std::runtime_error(id + std::string(result.error()));
		
		[[assume(ptr != nullptr)]];
		[[assume(rows > 0)]];
		[[assume(cols > 0)]];
		
		const size_t size = rows * cols;
		[[assume(size >= rows)]];
		[[assume(size >= cols)]];
		
		if constexpr (sizeof...(matrices) > 0)
			right<T>(id, matrices...);
	}

}//namespace damm
#endif //__MM_DEFS_H__
