# Dense Arrayed Matrix Math (DAMM)

DAMM is a high performance numerical computation package that exposes a full SIMD optimized BLAS surface through template metaprogramming in C++

# BLAS Specialization

In addition to standard matrix operators like transpose, multiply and inverse, The BLAS functions are reduced to template set-like operators: 
- union
- fused union
- reduce
- fused reduce

The operators could be specialized to BLAS-named functions:

```c++
	template<typename T, typename S>
	struct it
	{
		using scalar_t = T;
		using vector_t = T*;
		using matrix_t = T**;

	/*  LEVEL 1 BLAS-like operations */
		
		/**
		 * \brief Optimized dot product: x^T * y
		 * Equivalent to: fused_reduce<T, std::multiplies<>, std::plus<>, S>
		 */
		static constexpr auto dot_product = [](vector_t x, vector_t y, const size_t N) -> T 
		{ 
			return fused_reduce<T, std::multiplies<>, std::plus<>, S>(x, y, T(0), 1, N);
		};
		
		/**
		 * \brief Vector 2-norm: ||x||_2 = sqrt(x^T * x)
		 */
		static constexpr auto norm2 = [](vector_t x, const size_t N) -> T 
		{
			T sum_squares = fused_reduce<T, std::multiplies<>, std::plus<>, S>( x, x, T(0), 1, N);
			return std::sqrt(sum_squares);
		};
		
		/**
		 * \brief AXPY operation: y = a*x + y
		 */
		static constexpr auto axpy = [](const T a, vector_t x, vector_t y, const size_t N) -> void 
		{
			scalar::fused_union<FusionPolicy::FUSION_FIRST, T, std::plus<>, std::multiplies<>, S>(y, a, x, y, 1, N);
		};
		
		/**
		 * \brief SCAL operation: x = a*x  
		 */
		static constexpr auto scal = [](const T a, vector_t x, const size_t N) -> void 
		{
			scalar::unite<T, std::multiplies<>, S>(x, a, x, 1, N);
		};
		
		/**
		 * \brief Vector sum: sum(x)
		 */
		static constexpr auto sum = [](vector_t x, const size_t N) -> T 
		{
			return reduce<T, std::plus<>, S>(x, T(0), 1, N);
		};


		
	/*  LEVEL 2 BLAS-like operations */
		
		/**
		 * \brief Matrix-vector product: y = A*x
		 */
		static constexpr auto gemv = [](matrix_t A, vector_t x, vector_t y, const size_t M, const size_t N) -> void 
		{
			auto x_mat = view_as_2D(x, N, 1);
			auto y_mat = view_as_2D(y, M, 1);
			multiply<T, S>(A, x_mat.get(), y_mat.get(), M, N, 1);
		};
		
		/**
		 * \brief Rank-1 update: A = A + alpha*x*y^T
		 */
		static constexpr auto ger = [](const T alpha, vector_t x, vector_t y, matrix_t A,
									 const size_t M, const size_t N) -> void {
			auto x_mat = view_as_2D(x, M, 1);
			auto y_mat = view_as_2D(y, 1, N);
			auto outer_prod = aligned_alloc_2D<T, static_cast<size_t>(S)>(M, N);
			
			multiply<T, S>(x_mat.get(), y_mat.get(), outer_prod.get(), M, 1, N);
			scalar::unite<T, std::multiplies<>, S>(
				outer_prod.get(), alpha, outer_prod.get(), M, N);
			matrix::unite<T, std::plus<>, S>(
				A, outer_prod.get(), A, M, N);
		};
		
	/*  LEVEL 3 BLAS-like operations */
		
		/**
		 * \brief General matrix multiply: C = A*B
		 */
		static constexpr auto gemm = [](matrix_t A, matrix_t B, matrix_t C,
									  const size_t M, const size_t N, const size_t P) -> void {
			multiply<T, S>(A, B, C, M, N, P);
		};
		
		/**
		 * \brief Symmetric matrix multiply: C = A*A^T
		 */
		static constexpr auto syrk = [](matrix_t A, matrix_t C, const size_t M, const size_t N) -> void {
			auto A_T = aligned_alloc_2D<T, static_cast<size_t>(S)>(N, M);
			transpose<T, S>(A, A_T.get(), M, N);
			multiply<T, S>(A, A_T.get(), C, M, N, M);
		};
		
		/**
		 * \brief Matrix transpose: B = A^T
		 */
		static constexpr auto transpose_op = [](matrix_t A, matrix_t B, 
											  const size_t M, const size_t N) -> void {
			transpose<T, S>(A, B, M, N);
		};
		
		/**
		 * \brief Matrix trace: tr(A) = sum(diag(A))
		 */
		static constexpr auto trace = [](matrix_t A, const size_t N) -> T {
			T result = T(0);
			for (size_t i = 0; i < N; ++i)
				result += A[i][i];
			return result;
		};
		
		/**
		 * \brief Frobenius norm: ||A||_F = sqrt(sum(A.*A))
		 */
		static constexpr auto frobenius_norm = [](matrix_t A, const size_t M, const size_t N) -> T {
			T sum_squares = fused_reduce<T, std::multiplies<>, std::plus<>, S>(
				A, A, T(0), M, N);
			return std::sqrt(sum_squares);
		};
		
		/**
		 * \brief Hadamard product: C = A .* B (element-wise multiply)
		 */
		static constexpr auto hadamard = [](matrix_t A, matrix_t B, matrix_t C,
										  const size_t M, const size_t N) -> void {
			matrix::unite<T, std::multiplies<>, S>(A, B, C, M, N);
		};
		
		/**
		 * \brief Element-wise addition: C = A + B
		 */
		static constexpr auto add = [](matrix_t A, matrix_t B, matrix_t C,
									 const size_t M, const size_t N) -> void {
			matrix::unite<T, std::plus<>, S>(A, B, C, M, N);
		};
		
		/**
		 * \brief Element-wise subtraction: C = A - B
		 */
		static constexpr auto subtract = [](matrix_t A, matrix_t B, matrix_t C,
										  const size_t M, const size_t N) -> void {
			matrix::unite<T, std::minus<>, S>(A, B, C, M, N);
		};
		
		/**
		 * \brief LU decomposition: P*A = L*U
		 */
		static constexpr auto lu_decompose = [](matrix_t A, size_t* P, const size_t N) -> bool {
			return lu::decompose<T, S>(A, P, N);
		};
		
		/**
		 * \brief QR decomposition: A = Q*R
		 */
		static constexpr auto qr_decompose = [](matrix_t A, matrix_t Q, matrix_t R,
											   const size_t M, const size_t N) -> bool {
			return qr::decompose<T, S>(A, Q, R, M, N);
		};
		
		/**
		 * \brief Matrix inversion using LU
		 */
		static constexpr auto invert_lu = [](matrix_t A, matrix_t A_inv, const size_t N) -> bool {
			return lu::inverse<T, S>(A, A_inv, N);
		};
		
		/**
		 * \brief Matrix inversion using QR
		 */
		static constexpr auto invert_qr = [](matrix_t A, matrix_t A_inv, 
										   const size_t M, const size_t N) -> bool {
			return qr::inverse<T, S>(A, A_inv, M, N);
		};
		
		/**
		 * \brief Fused multiply-add: D = (A + B) * C
		 */
		static constexpr auto fma_union = [](matrix_t A, matrix_t B, const T C, matrix_t D,
										   const size_t M, const size_t N) -> void {
			scalar::fused_union<FusionPolicy::UNION_FIRST, T, std::plus<>, std::multiplies<>, S,
							  block_size>(A, B, C, D, M, N);
		};
		
		/**
		 * \brief Fused multiply-subtract: D = A - B * C  
		 */
		static constexpr auto fms_fusion = [](matrix_t A, matrix_t B, const T C, matrix_t D,
											const size_t M, const size_t N) -> void {
			scalar::fused_union<FusionPolicy::FUSION_FIRST, T, std::minus<>, std::multiplies<>, S,
							  block_size>(A, B, C, D, M, N);
		};
		
		/**
		 * \brief Fill matrix with zeros
		 */
		static constexpr auto zero_fill = [](matrix_t A, const size_t M, const size_t N) -> void {
			zeros<T, S>(A, M, N);
		};
		
		/**
		 * \brief Fill matrix with ones
		 */
		static constexpr auto one_fill = [](matrix_t A, const size_t M, const size_t N) -> void {
			ones<T, S>(A, M, N);
		};
		
		/**
		 * \brief Set identity matrix
		 */
		static constexpr auto set_identity_matrix = [](matrix_t A, const size_t M, const size_t N) -> void {
			identity<T, S>(A, M, N);
		};
	};
```

## SIMD Specialization

Each operator is SIMD aware and can be specialized for SSE, AVX and AVX512 hardware. 

```c++

/**:
 * \brief damm-blas (dblas) environment. 
 * 
 * Aliases the damm:: functions using BLAS-like semantics.
 * Allows the environment to be easily reconfigured for different targets 
 * 
 * \usage double result = dblas<double>::dot_product(x, y, N);
 * \usage dblas<float>::gemm(A, B, C, M, N, P);
 *
 */
template<typename T> using dblas = damm::it<T, AVX512>;///< BLAS-like environment optimized for target configuration.

```

## Parallel Scaling

DAMM is capable of parallel computation on its operational kernels with OpenMPI. Compile with flags -fopenmp and -lgomp for parallelization or omit the flags to avoid parallel region overheads if operating without parallelization.