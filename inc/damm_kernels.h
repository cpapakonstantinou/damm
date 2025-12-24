#ifndef __DAMM_KERNELS_H__
#define __DAMM_KERNELS_H__
/**
 * \file kernel.h
 * \brief kernel configurations for damm 
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

#include <simd.h>
#include <damm_cache.h>

namespace damm
{
	/**
	 * \brief Policy defining the matrix multiply kernel 
	 */
	template<typename T, typename S>
	struct multiply_kernel
	{
		static consteval size_t register_elements() { return std::max(S::template elements<T>(), size_t(1)); }  
		static constexpr size_t row_registers = 4;
		static constexpr size_t col_registers = 4;
		static consteval size_t kernel_rows() { return row_registers; } 
		static consteval size_t kernel_cols() { return col_registers * register_elements(); }
		using blocking = blocking_policy<T, S, multiply_kernel>;
	}; 

	/**
	 * \brief Policy defining the matrix transpose kernel 
	 */
	template<typename T, typename S>
	struct transpose_kernel
	{
		static consteval size_t register_elements() { return std::max(S::template elements<T>(), size_t(1)); }  
		static constexpr size_t row_registers = 1 * register_elements();
		static constexpr size_t col_registers = 1;
		static consteval size_t kernel_rows() { return row_registers; } 
		static consteval size_t kernel_cols() { return col_registers * register_elements(); }
		using blocking = blocking_policy<T, S, transpose_kernel>;
	};


	/**
	 * \brief Policy defining the matrix broadcast kernel 
	 */
	template<typename T, typename S>
	struct broadcast_kernel 
	{
		static consteval size_t register_elements() { return std::max(S::template elements<T>(), size_t(1)); }  
		static constexpr size_t row_registers = 4;
		static constexpr size_t col_registers = 4;
		static consteval size_t kernel_rows() { return row_registers; } 
		static consteval size_t kernel_cols() { return col_registers * register_elements(); }
		using blocking = blocking_policy<T, S, broadcast_kernel>;
		
	};

	/**
	 * \brief Policy defining the matrix union kernel 
	 */
	template<typename T, typename S>
	struct union_kernel 
	{
		static consteval size_t register_elements() { return std::max(S::template elements<T>(), size_t(1)); }  
		static constexpr size_t row_registers = 4;
		static constexpr size_t col_registers = 2;
		static consteval size_t kernel_rows() { return row_registers; } 
		static consteval size_t kernel_cols() { return col_registers * register_elements(); }
		using blocking = blocking_policy<T, S, union_kernel>;
	};

	/**
	 * \brief Policy defining the matrix reduce kernel 
	 */
	template<typename T, typename S>
	struct reduce_kernel 
	{
		static consteval size_t register_elements() { return std::max(S::template elements<T>(), size_t(1)); }  
		static constexpr size_t row_registers = 4;
		static constexpr size_t col_registers = 4;
		static consteval size_t kernel_rows() { return row_registers; } 
		static consteval size_t kernel_cols() { return col_registers * register_elements(); }
		using blocking = blocking_policy<T, S, reduce_kernel>;
	};

	/**
	 * \brief Policy defining the matrix fused reduce kernel 
	 */
	template<typename T, typename S>
	struct fused_reduce_kernel 
	{
		static consteval size_t register_elements() { return std::max(S::template elements<T>(), size_t(1)); }  
		static constexpr size_t row_registers = 2;
		static constexpr size_t col_registers = 8;
		static consteval size_t kernel_rows() { return row_registers; } 
		static consteval size_t kernel_cols() { return col_registers * register_elements(); }
		using blocking = blocking_policy<T, S, fused_reduce_kernel>;
	};

	/**
	 * \brief Policy defining the matrix fused union kernel 
	 */
	template<typename T, typename S>
	struct fused_union_kernel 
	{
		static consteval size_t register_elements() { return std::max(S::template elements<T>(), size_t(1)); }  
		static constexpr size_t row_registers = 2 ;
		static constexpr size_t col_registers = 4 ;
		static consteval size_t kernel_rows() { return row_registers ; } 
		static consteval size_t kernel_cols() { return col_registers * register_elements(); }
		using blocking = blocking_policy<T, S, fused_union_kernel>;
	};
}
#endif //__DAMM_KERNELS_H__