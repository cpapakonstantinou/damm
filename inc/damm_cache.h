#ifndef __DAMM_CACHE_H__
#define __DAMM_CACHE_H__
/**
 * \file damm_cache.h
 * \brief cache blocking policies for damm 
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
#include <common.h>

namespace damm
{
	#ifndef DAMM_L1_CACHE_SIZE
		#define DAMM_L1_CACHE_SIZE 32*1024
	#endif
	#ifndef DAMM_L2_CACHE_SIZE
		#define DAMM_L2_CACHE_SIZE 256*1024
	#endif
	#ifndef DAMM_L3_CACHE_SIZE
		#define DAMM_L3_CACHE_SIZE 8*1024*1024
	#endif
	#ifndef DAMM_LINE_SIZE
		#define DAMM_LINE_SIZE 64
	#endif

	struct cache_info 
	{
		static constexpr size_t l1_size = DAMM_L1_CACHE_SIZE; ///< 32 KB default
		static constexpr size_t l2_size = DAMM_L2_CACHE_SIZE; ///< 256 KB default
		static constexpr size_t l3_size = DAMM_L3_CACHE_SIZE; ///< 8 MB default
		static constexpr size_t line_size = DAMM_LINE_SIZE; ///< 64 B default
	};
	
	template<typename T, typename S, template<typename, typename> class K>
	class blocking_policy 
	{
		using kernel = K<T, S>;
		static constexpr size_t kernel_rows = kernel::kernel_rows();
		static constexpr size_t kernel_cols = kernel::kernel_cols();

		/**
		 * \brief Machine specific optimization that accounts for associativity in L1 cache
		 */
		static constexpr float l1_fill_factor = []() consteval 
		{
			if constexpr ( requires { kernel::l1_fill_factor; } )
				return kernel::l1_fill_factor;
			else
				return 0.80f;
		}();
		
		/**
		 * \brief Machine specific optimization that accounts for associativity in L2 cache
		 */
		static constexpr float l2_fill_factor = []() consteval 
		{
			if constexpr ( requires { kernel::l2_fill_factor; } )
				return kernel::l2_fill_factor;
			else
				return 0.90f;
		}();
		
		/**
		 * \brief Machine specific optimization that accounts for associativity in L3 cache
		 */
		static constexpr float l3_fill_factor = []() consteval 
		{
			if constexpr ( requires { kernel::l3_fill_factor; } )
				return kernel::l3_fill_factor;
			else
				return 0.50f;
		}();

	private:
		// L1 blocking: Elements to process at once
		static consteval size_t _l1_block() 
		{
			constexpr size_t kernel_elements = kernel_rows * kernel_cols * sizeof(T); // bytes in C tile
			constexpr size_t kernel_chunk = kernel_rows * sizeof(T) + kernel_cols * sizeof(T); // bytes per K iteration
			constexpr size_t l1_fill = cache_info::l1_size*l1_fill_factor;
			
			if constexpr (kernel_chunk == 0 || l1_fill <= kernel_elements) 
				return kernel_rows; // fallback
			
			constexpr size_t l1_block = (l1_fill - kernel_elements) / kernel_chunk;			
			constexpr size_t l1_min = kernel_rows;
			constexpr size_t l1_aligned = (l1_block / kernel_rows) * kernel_rows;
			return (l1_aligned >= l1_min) ? l1_aligned : l1_min;
		}
		
		// L2 blocking:
		static consteval size_t _l2_block() 
		{
			constexpr size_t l1 = _l1_block();
			constexpr size_t l2_fill = cache_info::l2_size*l2_fill_factor;			
			constexpr size_t l2_block = l2_fill / (l1 * sizeof(T));
			constexpr size_t l2_aligned = (l2_block / kernel_rows) * kernel_rows;
			return (l2_aligned >= kernel_rows) ? l2_aligned : kernel_rows;
		}
		
		// L3 blocking:
		static consteval size_t _l3_block() 
		{
			constexpr size_t l1 = _l1_block();
			constexpr size_t l3_fill = cache_info::l3_size*l3_fill_factor;
			constexpr size_t l3_block = l3_fill / (l1 * sizeof(T));
			constexpr size_t l3_aligned = (l3_block / kernel_cols) * kernel_cols;
			return (l3_aligned >= kernel_cols) ? l3_aligned : kernel_cols;
		}
		
	public:
		static constexpr size_t l1_block = _l1_block();
		static constexpr size_t l2_block = _l2_block();
		static constexpr size_t l3_block = _l3_block();
	};
}
#endif //__DAMM_CACHE_H__