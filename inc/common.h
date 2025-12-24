#ifndef __COMMON_H__
#define __COMMON_H__
/**
 * \file common.h
 * \brief common definitions for damm 
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

#include <damm_right.h>
#include <damm_memory.h>

#include <cstdint>
#include <ranges>
#include <complex>

namespace damm
{

	template <typename T>
	struct base 
	{
		using type = T;
	};

	template <typename T>
	struct base<std::complex<T>> 
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

	/**
	 \brief Helper variable template that always evaluates to false.

	 This is a utility used in static_assert to trigger a compile-time error 
	 in non-instantiated branches of a constexpr if or SFINAE context.

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
	 \tparam O The binary operation type (e.g., std::plus<>, std::multiplies<>).

	 \return The identity value of type T for the operation O.

	 \note The supported operations are:
	 - std::plus<> and std::minus<> → returns 0
	 - std::multiplies<> and std::divides<> → returns 1

	 \throws A compile-time error if an unsupported operation is used.

	ode
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

	template<auto N, typename F>
	constexpr void
	static_for(F&& func) noexcept 
	{
		[&]<std::size_t... Is>(std::index_sequence<Is...>) 
		{ 
			(func.template operator()<Is>(), ...); 
		}(std::make_index_sequence<N>{});
	}



}//namespace damm
#endif //__COMMON_H__
