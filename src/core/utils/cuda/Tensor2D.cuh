#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include "Tensor1D.cuh"

namespace Flair
{
    template<int N, int M>
    struct Tensor2D
    {
        // NOTE: Tensor stored in column-major order        
    private:
        union
        {
            float data[N][M];
            float rawData[N*M];
        };

    public:
        union
        {
            float grad[N][M];
            float rawGrad[N*M];
        };

    private:
        __host__ __device__ void PrintImpl(const bool showGrad) const
        {
            printf("{\n");
            for (int rowIdx = 0; rowIdx < M; ++rowIdx)
            {
                printf(" { ");
                for (int colIdx = 0; colIdx < N; ++colIdx)
                {
                    if (showGrad)
                    {
                        printf_yellow("%s%.8f", colIdx ? ", " : "", grad[colIdx][rowIdx]);
                    }
                    else
                    {
                        printf_green("%s%.8f", colIdx ? ", " : "", data[colIdx][rowIdx]);
                    }
                }
                printf(" }%s\n", (rowIdx == M - 1) ? "" : ", ");
            }
            printf("}");
        }

    public:
        __host__ __device__ Tensor2D()
        {
#if !defined(__CUDA_ARCH__)
            memset(this, 0, sizeof(Tensor2D));
#endif        
        }

        __host__ __device__ Tensor2D(const float (&d)[N][M])        
        {
            memcpy(rawData, &d[0][0], sizeof(float) * N * M);
            memset(rawGrad, 0, sizeof(float) * N * M);
        }

        template<typename RNG>
        __host__ void Initialise(RNG& rng)
        {
            for (int i = 0; i < M * N; ++i) { rawData[i] = rng() / N; }
            memset(rawGrad, 0, sizeof(float) * N * M);
        }

        __forceinline__ __device__ void ZeroGrad()
        {
            if (kKernelIdx < N * M) { rawGrad[kKernelIdx] = 0; }
        }

        __forceinline__ __host__ __device__ float& operator[](const int idx) { return rawData[idx]; }
        __forceinline__ __host__ __device__ const float& operator[](const int idx) const { return rawData[idx]; }
        __forceinline__ __host__ __device__ float& operator()(const int col, const int row) { return data[col][row]; }
        __forceinline__ __host__ __device__ const float& operator()(const int col, const int row) const { return data[col][row]; }

        __host__ __device__ __forceinline__ void Print() const { PrintImpl(false); }
        __host__ __device__ __forceinline__ void PrintGrad() const { PrintImpl(true); }
    }; 

    // In-place matrix multiply of an NxN tensor with N tensor. 
    template<bool Transpose, int N>
    __forceinline__ __device__ void Mul(const Tensor2D<N, N>& m, Tensor1D<N>& v, float (&scratch)[N][N])
    {
        // Block must have exactly the same number of threads as elements in the tensor
        CudaAssertDebug(blockDim.x == N * N);

        // Populate the scratch matrix with the products of the N*N tensor and the N tensor
        __syncthreads();
        const int rowIdx = kThreadIdx % N, colIdx = kThreadIdx / N;
        scratch[colIdx][rowIdx] = (Transpose ? m(rowIdx, colIdx) : m[kThreadIdx]) * v[colIdx];

        // Parallel reduce the scratch matrix
        for (int reduceMask = 2; reduceMask < N; reduceMask <<= 1)
        {
            __syncthreads();
            if ((colIdx & (reduceMask - 1)) == 0)
            {
                scratch[colIdx][rowIdx] += scratch[colIdx + (reduceMask >> 1)][rowIdx];
            }
        }
         
        // Final reduction stored in the elements of the N tensor
        __syncthreads();
        if (colIdx == 0)
        {
            v[rowIdx] = scratch[0][rowIdx] + scratch[N >> 1][rowIdx];
        }
    }
}
