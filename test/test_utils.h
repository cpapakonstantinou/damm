/**
 * \file test_utils.h
 * \brief Common utilities and framework for damm performance tests
 * \author cpapakonstantinou
 * \date 2025
 */
#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include <iostream>
#include <iomanip>
#include <format>
#include <type_traits>
#include <complex>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <functional>
#include <damm_kernels.h>
#include "naive.h"

using namespace damm;

/**
 * \brief Generic kernel configuration with templates for overrides
 */
template<size_t R, size_t C, float L1=0.8f, float L2=0.3f, float L3=0.8f>
struct kernel_config
{
	template<typename T, typename S>
	struct kernel
	{
		static consteval size_t register_elements() { 
			return std::max(S::template elements<T>(), size_t(1)); 
		}
		static constexpr size_t row_registers = R;
		static constexpr size_t col_registers = C;
		static consteval size_t kernel_rows() { return row_registers; }
		static consteval size_t kernel_cols() { 
			return col_registers * register_elements(); 
		}
		
		static constexpr float l1_fill_factor = L1;
		static constexpr float l2_fill_factor = L2;
		static constexpr float l3_fill_factor = L3;
		
		using blocking = blocking_policy<T, S, kernel>;
	};
};

/**
 * \brief Compute memory bandwidth in GB/s from bytes accessed
 */
template<typename T>
double compute_bandwidth_w(size_t bytes, double time_ms)
{
	bytes *= sizeof(T);
	return (bytes / (time_ms * 1e-3)) / 1e9;
}

/**
 * \brief Compute memory bandwidth for read+write operations
 */
template<typename T>
double compute_bandwidth_rw(size_t bytes, double time_ms, size_t reads = 2, size_t writes = 1)
{
	bytes *=  sizeof(T) * (reads + writes);
	return (bytes / (time_ms * 1e-3)) / 1e9;
}

/**
 * * \brief Compute GFLOPS
 */
template<typename T>
double compute_gflops(double ops, double time_ms)
{
	if constexpr (std::is_same_v<T, std::complex<float>> || 
				  std::is_same_v<T, std::complex<double>>)
		ops *= 4.0;
	return (ops / 1e9) / (time_ms / 1000.0);
}

/**
 * \brief Generic benchmark runner with configurable warmup and iterations
 * Returns median time in milliseconds
 */
template<typename Func>
double benchmark(Func&& operation, size_t warmup_iters = 2, size_t bench_iters = 5)
{
	std::vector<double> times;
	times.reserve(bench_iters);
	
	// Warmup
	for (size_t i = 0; i < warmup_iters; ++i)
		operation();
	
	// Benchmark
	for (size_t i = 0; i < bench_iters; ++i)
	{
		auto start = std::chrono::high_resolution_clock::now();
		operation();
		auto end = std::chrono::high_resolution_clock::now();
		
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		times.push_back(duration.count() / 1000.0);
	}
	
	// Return median time
	std::sort(times.begin(), times.end());
	return times[times.size() / 2];
}

struct perf_result
{
	size_t tile_rows;
	size_t tile_cols;
	size_t reg_rows;
	size_t reg_cols;
	double time_ms;
	std::string metric_name;
	double metric_value;
	bool verified;
};

void print_perf_header(const std::string& op_name, const std::string& type_name, size_t simd_width, const std::string& metric)
{
	std::cout << "\n" << op_name << "\n";
	std::cout << "Type: " << type_name << " | SIMD_WIDTH: " << simd_width << "\n";
	
	std::cout << \
		std::format("{:<12} {:<12} {:<12} {:<12} {:<8}\n", "Tile", "Registers", "Time(ms)", metric, "Verify");

	std::cout << std::string(60, '-') << "\n";
}

void print_perf_result(const perf_result& r)
{
		std::cout << std::format("{:<12} {:<12} {:<12.3f} {:<12.3} {:<8}\n",
			std::to_string(r.tile_rows) + "x" + std::to_string(r.tile_cols),
			std::to_string(r.reg_rows) + "x" + std::to_string(r.reg_cols),
			r.time_ms, 
			r.metric_value, 
			r.verified ? "PASS" : "FAIL");
}



template<typename T, double tol = 1e-4>
bool is_same(const char* name, T** A, T** B, const size_t M, const size_t N, 
			bool verbose = true)
{
	bool x = true;
	size_t i = 0, j = 0;

	for (i = 0; i < M; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			if constexpr (std::is_floating_point<T>::value)
			{
				if (std::abs(A[i][j] - B[i][j]) > tol)
				{
					x = false;
					break;
				}
			}
			else if constexpr (std::is_same<T, std::complex<float>>::value ||
							   std::is_same<T, std::complex<double>>::value)
			{
				if (std::abs(A[i][j] - B[i][j]) > tol)
				{
					x = false;
					break;
				}
			}
			else
			{
				if (A[i][j] != B[i][j])
				{
					x = false;
					break;
				}
			}
		}
		if (!x) break;
	}

	if (verbose)
		printf("[%-4s] %s\n", (x ? "OK" : "FAIL"), name);

	if (!x && verbose)
	{
		if constexpr (std::is_same<T, std::complex<float>>::value ||
					  std::is_same<T, std::complex<double>>::value)
		{
			printf("Mismatch at [%zu][%zu]: A = (%f, %f), B = (%f, %f)\n",
				   i, j,
				   A[i][j].real(), A[i][j].imag(),
				   B[i][j].real(), B[i][j].imag());
		}
		else
		{
			printf("Mismatch at [%zu][%zu]: A = %f, B = %f\n",
				   i, j,
				   static_cast<double>(A[i][j]),
				   static_cast<double>(B[i][j]));
		}
	}

	return x;
}

template<typename T>
bool approx_equal(T a, T b, double rel_tol = 1e-3, double abs_tol = 1e-4) 
{
	if constexpr (std::is_same_v<T, std::complex<float>> || 
				  std::is_same_v<T, std::complex<double>>) 
	{
		return approx_equal(a.real(), b.real(), rel_tol, abs_tol) && 
			   approx_equal(a.imag(), b.imag(), rel_tol, abs_tol);
	} 
	else 
	{
		if (std::abs(a - b) <= abs_tol) 
			return true;
		double rel_diff = std::abs(a - b) / std::max(std::abs(a), std::abs(b));
		return rel_diff <= rel_tol;
	}
}

template<typename T>
T matrix_max_error(T** A, T** B, size_t M, size_t N)
{
	T max_error = T(0);
	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			max_error = std::max(max_error, std::abs(A[i][j] - B[i][j]));
	return max_error;
}

template<typename T>
void fill_rand(T** A, const size_t M, const size_t N, unsigned seed = 42)
{
	std::mt19937 rng(seed);
	
	if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) 
	{
		std::uniform_real_distribution<T> dist(-1.0, 1.0);
		for (size_t i = 0; i < M; ++i) 
			for (size_t j = 0; j < N; ++j) 
				A[i][j] = dist(rng);
	} 
	else if constexpr (std::is_same_v<T, std::complex<float>>) 
	{
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		for (size_t i = 0; i < M; ++i) 
			for (size_t j = 0; j < N; ++j) 
				A[i][j] = std::complex<float>(dist(rng), dist(rng));
	} 
	else if constexpr (std::is_same_v<T, std::complex<double>>) 
	{
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for (size_t i = 0; i < M; ++i) 
			for (size_t j = 0; j < N; ++j) 
				A[i][j] = std::complex<double>(dist(rng), dist(rng));
	}
}

/**
 * \brief Fill matrix with geometric series to prevent overflow in reductions
 */
template<typename T>
void fill_geometric(T** A, size_t M, size_t N, T initial, T ratio)
{
	T val = initial;
	for (size_t i = 0; i < M; ++i)
	{
		for (size_t j = 0; j < N; ++j)
		{
			A[i][j] = val;
			val *= ratio;
		}
	}
}


template<typename T>
void print_vector(const T* v, size_t n, const std::string& name) 
{
	std::cout << name << ": [ ";
	for (size_t i = 0; i < n; ++i)
		std::cout << std::setprecision(6) << v[i] << " ";
	std::cout << "]\n";
}

template<typename T>
void print_matrix(T** A, size_t M, size_t N, const std::string& name) 
{
	std::cout << name << ":\n";
	for (size_t i = 0; i < M; ++i) 
	{
		for (size_t j = 0; j < N; ++j) 
			std::cout << std::setw(12) << std::setprecision(6) << A[i][j] << " ";
		std::cout << "\n";
	}
	std::cout << "\n";
}


/**
 * \brief Test write bandwidth 
 */
template<typename T, typename S, size_t R, size_t C, typename F>
perf_result test_kernel_write(F&& bench_f, T** A_ref, T** A_test, size_t M, size_t N)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	double time = benchmark(std::forward<F>(bench_f));
	double bandwidth = compute_bandwidth_w<T>(M * N, time);
	bool verified = (A_ref == nullptr) || is_same<T>("", A_ref, A_test, M, N, false);
	
	return perf_result{
		.tile_rows = R, 
		.tile_cols = C * SIMD_WIDTH,
		.reg_rows = R,
		.reg_cols = C,
		.time_ms = time,
		.metric_name = "BW (GB/s)", 
		.metric_value = bandwidth, 
		.verified= verified
	};
}

/**
 * \brief Test compute-bound operations
 */
template<typename T, typename S, size_t R, size_t C, typename F>
perf_result test_kernel_compute(F&& bench_f, T** A_ref, T** A_test, size_t M, size_t N, size_t P = 1)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	double time = benchmark(std::forward<F>(bench_f));
	double gflops = compute_gflops<T>(2*M*N*P, time);
	bool verified = is_same<T, 1e-3>("", A_ref, A_test, M, P, false);
	
	return perf_result{
		.tile_rows = R, 
		.tile_cols = C * SIMD_WIDTH,
		.reg_rows = R,
		.reg_cols = C,
		.time_ms = time,
		.metric_name = "GFLOPS", 
		.metric_value = gflops, 
		.verified= verified
	};
}

/**
 * \brief Test reduction operations
 */
template<typename T, typename S, size_t R, size_t C, typename F>
perf_result test_kernel_reduce(F&& bench_f, T ref_result, T test_result, size_t M, size_t N)
{
	constexpr size_t SIMD_WIDTH = S::template elements<T>();
	double time = benchmark(std::forward<F>(bench_f));
	
	// Throughput in GOPS
	double throughput = compute_gflops<T>(M * N, time);
	
	// Verify scalar result
	bool verified = approx_equal(ref_result, test_result);
	
	return perf_result{
		R * SIMD_WIDTH,
		C * SIMD_WIDTH,
		R, 
		C,
		time, 
		"GFLOPS", 
		throughput, 
		verified
	};
}

#endif // __TEST_UTILS_H__