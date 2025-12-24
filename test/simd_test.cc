/**
 * \file simd_test.cc
 * \brief unit test for load and store operations
 * \author cpapakonstantinou
 * \date 2025
 */
#include "test_utils.h"
#include "carray.h"
#include "simd.h"
#include <complex>
#include <format>
#include <cstring>

using namespace damm;

template<typename T, typename S, template<typename, typename> class P>
void 
load_store_naive(T** src, T** dst)
{
	constexpr size_t rows = P<T, S>::block_rows();
	constexpr size_t cols = P<T, S>::block_cols();
	
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < cols; ++j)
			dst[i][j] = src[i][j];
}

template<typename T, typename S, template<typename, typename> class P>
bool
test_load_store_aligned(const size_t M, const size_t N)
{ 
	static constexpr size_t ALIGN = 64;
	bool ret = true;
	
	constexpr size_t rows = P<T, S>::row_registers;
	constexpr size_t cols = P<T, S>::col_registers;
	constexpr size_t elements = P<T, S>::register_elements();
	constexpr size_t block_h = P<T, S>::block_rows();
	constexpr size_t block_w = P<T, S>::block_cols();
	
	if (M < block_h || N < block_w) 
	{
		std::cout << std::format("Skipping: M={}, N={} too small for kernel {}x{}\n", 
								 M, N, block_h, block_w);
		return true;
	}
	
	const size_t total_size = M * N;
	
	carray<T, 2, ALIGN> src(M, N);
	carray<T, 2, ALIGN> dst_ref(M, N);
	carray<T, 2, ALIGN> dst_test(M, N);
	
	fill_rand<T>(src.get(), M, N);
	
	std::memset(dst_ref.begin(), 0, total_size * sizeof(T));
	std::memset(dst_test.begin(), 0, total_size * sizeof(T));
	
	// Setup 2D register array
	using register_type = typename S::template register_t<T>;
	register_type reg_storage[rows][cols];
	register_type* registers[rows];
	for (size_t i = 0; i < rows; ++i) {
		registers[i] = reg_storage[i];
	}
	
	// Reference: naive copy
	load_store_naive<T, S, P>(src.get(), dst_ref.get());
	
	// Test: SIMD load and store
	load<T, S, P>(src.get(), registers, 0, 0);
	store<T, S, P>(dst_test.get(), registers, 0, 0);
	
	// Compare only the processed block
	ret &= is_same<T>(
		std::format("load_store<{},{}>:", typeid(T).name(), S::bytes).c_str(), 
		dst_ref.get(), 
		dst_test.get(), 
		block_h,  // Only compare block_h rows
		block_w   // Only compare block_w columns
	);
	
	return ret;
}

template<typename T>
bool
test_all_simd_ops(const size_t M, const size_t N)
{
	bool ret = true;
	ret &= test_load_store_aligned<T, SSE, transpose_kernel>(M, N);
	ret &= test_load_store_aligned<T, AVX, transpose_kernel>(M, N);
	ret &= test_load_store_aligned<T, AVX512, transpose_kernel>(M, N);
	return ret;
}

//TODO: 
//load and store need to check that 
//row_offset, col_offset do not overstep the dimensions of the operand
//this is a guard that can be added for load/store later

int main(int argc, char* argv[])
{
	// Test different element counts
	static constexpr size_t M[] = {128};
	static constexpr size_t N[] = {32, 64};
	
	try
	{
		for(size_t m = 0; m < sizeof(M)/sizeof(size_t); ++m)
		{
			for(size_t n = 0; n < sizeof(N)/sizeof(size_t); ++n)
			{
				bool all_ops = true;
				all_ops &= test_all_simd_ops<float>(M[m], N[n]);
				all_ops &= test_all_simd_ops<double>(M[m], N[n]);
				all_ops &= test_all_simd_ops<std::complex<float>>(M[m], N[n]);
				all_ops &= test_all_simd_ops<std::complex<double>>(M[m], N[n]);
				
				std::string report = std::format(
					"[{}] load_store: M={}, N={}", 
					((all_ops) ? "OK" : "FAIL"), 
					M[m], 
					N[n]
				);
				std::cout << report << std::endl;
			}
		}
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	return 0;
}