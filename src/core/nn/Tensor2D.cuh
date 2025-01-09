#pragma once

#include "Tensor1D.cuh"
#include "ListUtils.cuh"

namespace Flair
{    
    template<int N, int M, bool HasGrad = false>
    struct Tensor2D
    {
        static constexpr int kMaxCUDAThreads = 1024;

    public:
        enum : int
        {
            kN = N,
            kM = M,
            kSize = N * M,

            // The number of columns/ros per thread
            kNPerThread = DivCeil(N, kMaxCUDAThreads / M),
            kMPerThread = DivCeil(M, kMaxCUDAThreads / N),

            // Number of threads per row/column
            kConcurrentN = DivCeil(N, kNPerThread),
            kConcurrentM = DivCeil(M, kMPerThread),

            // The number of concurrent operations (threads) required to vector multiple this tensor
            kMaxConcurrency = CexprMax(M * kConcurrentN, N * kConcurrentM),

            // Whether the elements of this tensor fit into the maximum number of threads
            kFitIntoMaxThreads = kNPerThread == 1 && kMPerThread == 1,

            // Whether to do bounds checking on accessor methods
#if defined(_DEBUG)
            kIsGuarded = 1
#else
            kIsGuarded = 0
#endif
        };

        // NOTE: Tensor stored in column-major order        
    private:
        union
        {
            float data[N * (HasGrad ? 2 : 1)][M];
            float rawData[N * M * (HasGrad ? 2 : 1)];
        };      

    public:
        __host__ __device__ Tensor2D()
        {
#if !defined(__CUDA_ARCH__)
            memset(this, 0, sizeof(Tensor2D));
#endif        
        }

        // Allow tensors to be specified in row-major order (useful for Mathematica testing)
        __host__ __device__ Tensor2D(const float (&d)[M][N])
        {
            for (int n = 0; n < N; ++n)
            {
                for (int m = 0; m < M; ++m)
                {
                    data[n][m] = d[m][n];
                }
            }                 
            ZeroGrad();
        }

        template<typename RNG>
        __host__ void Initialise(RNG& rng)
        {
            for (int i = 0; i < M * N; ++i) { rawData[i] = rng(); }
            ZeroGrad();
        }

#if defined(__CUDA_ARCH__)
        __forceinline__ __device__ void ZeroGrad()
        {
            if(HasGrad && kThreadIdx < N * M) { rawData[N*M + kThreadIdx] = 0; }
        }
#else
        __forceinline__ __host__ void ZeroGrad()
        {
            if (HasGrad) { memset(&rawData[N * M], 0, sizeof(float) * N * M); }
        }
#endif

        __forceinline__ __host__ __device__ static void AssertValidIdx(const int idx)
        {
            CudaAssertFmt(idx < N * M, "Out of bounds index to Tensor2: %i >= %i x %i", idx, N, M);
        }

        __forceinline__ __host__ __device__ static void AssertValidColRow(const int col, const int row)
        {
            CudaAssertFmt(col < N && row < M, "Out of bounds access to Tensor2: [%i, %i] >= [%i, %i]", col, row, N, M)
        }

        __forceinline__ __host__ __device__ float& operator[](const int idx) 
        { 
            if (kIsGuarded) { AssertValidIdx(idx); }
            return rawData[idx]; 
        }

        __forceinline__ __host__ __device__ const float& operator[](const int idx) const 
        { 
            if (kIsGuarded) { AssertValidIdx(idx); }
            return rawData[idx];
        }

        __forceinline__ __host__ __device__ float& operator()(const int col, const int row) 
        { 
            if (kIsGuarded) { AssertValidColRow(col, row); }
            return data[col][row];
        }

        __forceinline__ __host__ __device__ const float& operator()(const int col, const int row) const 
        { 
            if (kIsGuarded) { AssertValidColRow(col, row); }
            return data[col][row];
        }

        __forceinline__ __host__ __device__ float& Grad(const int col, const int row) 
        { 
            static_assert(HasGrad, "This tensor does not have gradients."); 
            if (kIsGuarded) { AssertValidColRow(col, row); }
            return data[N+col][row];
        }

        __forceinline__ __host__ __device__ const float& Grad(const int col, const int row) const 
        { 
            if (kIsGuarded) { AssertValidColRow(col, row); }
            static_assert(HasGrad, "This tensor does not have gradients.");
            return data[N+col][row]; 
        }

        __forceinline__ __host__ __device__ float& Grad(const int idx) 
        { 
            if (kIsGuarded) { AssertValidIdx(idx); }
            static_assert(HasGrad, "This tensor does not have gradients.");
            return rawData[N * M + idx]; 
        } 

        __forceinline__ __host__ __device__ const float& Grad(const int idx) const 
        { 
            if (kIsGuarded) { AssertValidIdx(idx); }
            static_assert(HasGrad, "This tensor does not have gradients.");
            return rawData[N * M + idx]; 
        }

        __forceinline__ __host__ __device__ float* Data() { return rawData; }
        __forceinline__ __host__ __device__ const float* Data() const { return rawData; }

        __host__ __forceinline__ TensorIterator<float> begin() { return TensorIterator<float>(data, 0); }
        __host__ __forceinline__ TensorIterator<const float> begin() const { return TensorIterator<const float>(data, 0); }
        __host__ __forceinline__ TensorIterator<float> end() { return TensorIterator<float>(data, N); }
        __host__ __forceinline__ TensorIterator<const float> end() const { return TensorIterator<const float>(data, N); }

        __host__ __device__ Tensor2D<M, N> Transpose() const
        {
            // Transposes the tensor from size N x M to size M x N
            Tensor2D<M, N> r;
            for (int n = 0; n < N; ++n)
            {
                for (int m = 0; m < M; ++m)
                {
                    r.data[m][n] = data[n][m];
                }
            }
            return r;
        }

        __host__ __device__ void FromRowMajor()
        {
            // Assumes the data are stored in row-major order and need to be converted to column-major
            Tensor2D scratch;
            for (int i = 0; i < kSize; ++i)
            {
                scratch(i % N, i / N) = rawData[i];
            }
            *this = scratch;
        }

        __host__ __device__ void Print(const bool showGrad = false, const bool scientific = true) const
        {
            CudaAssertFmt(!showGrad || HasGrad, "Tensor does not have gradients to print");
            printf("{\n");
            for (int rowIdx = 0; rowIdx < M; ++rowIdx)
            {
                printf(" { ");
                for (int colIdx = 0; colIdx < N; ++colIdx)
                {       
                    printf(scientific ? "%s%.4E" : "%s%.8", colIdx ? ", " : "", data[showGrad ? (N + colIdx) : colIdx][rowIdx]);
                }
                printf(" }%s\n", (rowIdx == M - 1) ? "" : ", ");
            }
            printf("}\n");
        }

        __host__ std::string Format(const bool showGrad = false, const bool scientific = true) const
        {
            CudaAssertFmt(!showGrad || HasGrad, "Tensor does not have gradients to print");
            std::string str = "{\n";
            for (int rowIdx = 0; rowIdx < M; ++rowIdx)
            {
                str += " { ";
                for (int colIdx = 0; colIdx < N; ++colIdx)
                {
                    str += tfm::format(scientific ? "%s%.4E" : "%s%.8", colIdx ? ", " : "", data[showGrad ? (N + colIdx) : colIdx][rowIdx]);
                }
                str += tfm::format(" }%s\n", (rowIdx == M - 1) ? "" : ", ");
            }
            str += "}";
            return str;
        }
    }; 
}
