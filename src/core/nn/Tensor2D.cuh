#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include "Tensor1D.cuh"
#include "thirdparty/tinyformat/tinyformat.h"

namespace Flair
{
    template<int N, int M, bool HasGrad = false>
    struct Tensor2D
    {
        // NOTE: Tensor stored in column-major order        
    private:
        template<int N, bool HasGrad> struct SizeType { enum : int { Value = N * 2 }; };
        template<int N> struct SizeType<N, false> { enum : int { Value = N }; };

        union
        {
            float data[SizeType<N, HasGrad>::Value][M];
            float rawData[SizeType<N, HasGrad>::Value * M];
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

        __forceinline__ __host__ __device__ float& operator[](const int idx) { return rawData[idx]; }
        __forceinline__ __host__ __device__ const float& operator[](const int idx) const { return rawData[idx]; }
        __forceinline__ __host__ __device__ float& operator()(const int col, const int row) { return data[col][row]; }
        __forceinline__ __host__ __device__ const float& operator()(const int col, const int row) const { return data[col][row]; }
        __forceinline__ __host__ __device__ float& Grad(const int col, const int row) { static_assert(HasGrad, "This tensor does not have gradients."); return data[N+col][row]; }
        __forceinline__ __host__ __device__ const float& Grad(const int col, const int row) const { static_assert(HasGrad, "This tensor does not have gradients."); return data[N+col][row]; }
        __forceinline__ __host__ __device__ float& Grad(const int idx) { static_assert(HasGrad, "This tensor does not have gradients."); return rawData[N * M + idx]; }
        __forceinline__ __host__ __device__ const float& Grad(const int idx) const { static_assert(HasGrad, "This tensor does not have gradients."); return rawData[N * M + idx]; }

        __forceinline__ __host__ __device__ float* Data() { return rawData; }
        __forceinline__ __host__ __device__ const float* Data() const { return rawData; }

        __host__ __forceinline__ TensorIterator<float> begin() { return TensorIterator<float>(data, 0); }
        __host__ __forceinline__ TensorIterator<const float> begin() const { return TensorIterator<const float>(data, 0); }
        __host__ __forceinline__ TensorIterator<float> end() { return TensorIterator<float>(data, N); }
        __host__ __forceinline__ TensorIterator<const float> end() const { return TensorIterator<const float>(data, N); }

        __host__ __device__ Tensor2D Transpose() const
        {
            Tensor2D r;
            for (int n = 0; n < N; ++n)
            {
                for (int m = 0; m < M; ++m)
                {
                    r.data[m][n] = data[n][m];
                }
            }
            return r;
        }

        __host__ __device__ void Print(const bool showGrad = false) const
        {
            CudaAssertMsg(!showGrad || HasGrad, "Tensor does not have gradients to print");
            printf("{\n");
            for (int rowIdx = 0; rowIdx < M; ++rowIdx)
            {
                printf(" { ");
                for (int colIdx = 0; colIdx < N; ++colIdx)
                {       
                    printf("%s%.8f", colIdx ? ", " : "", data[showGrad ? (N + colIdx) : colIdx][rowIdx]);
                }
                printf(" }%s\n", (rowIdx == M - 1) ? "" : ", ");
            }
            printf("}\n");
        }

        __host__ std::string Format(const bool showGrad = false) const
        {
            CudaAssertMsg(!showGrad || HasGrad, "Tensor does not have gradients to print");
            std::string str = "{\n";
            for (int rowIdx = 0; rowIdx < M; ++rowIdx)
            {
                str += " { ";
                for (int colIdx = 0; colIdx < N; ++colIdx)
                {
                    str += tfm::format("%s%.8f", colIdx ? ", " : "", data[showGrad ? (N + colIdx) : colIdx][rowIdx]);
                }
                str += tfm::format(" }%s\n", (rowIdx == M - 1) ? "" : ", ");
            }
            str += "}";
            return str;
        }
    }; 
}
