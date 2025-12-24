#ifndef __DAMM_RIGHT_H__
#define __DAMM_RIGHT_H__
/**
 * \file damm_right.h
 * \brief definitions for checking matrix inputs to damm functions
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

#include <expected>
#include <tuple>
#include <string_view>
#include <stdexcept>
#include <cstdint>
#include <limits>

namespace damm
{
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
}
#endif //__DAMM_RIGHT_H__