#pragma once

#include "Tensor2D.cuh"

namespace Flair
{
    // In-place matrix multiply of an NxN tensor with N vector. 
    /*template<bool Transpose, int N>
    __forceinline__ __device__ void Mul(const Tensor2D<N, N>& m, const Tensor1D<N>& v, Tensor1D<N>& w, float (&scratch)[N][N])
    {        
        // Block must have exactly the same number of threads as elements in the tensor
        CudaAssertDebug(blockDim.x == N * N);

        // Populate the scratch matrix with the products of the N*N tensor and the N tensor
        __syncthreads();
        const int rowIdx = kThreadIdx % N, colIdx = kThreadIdx / N;
        scratch[colIdx][rowIdx] = (Transpose ? m(rowIdx, colIdx) : m[kThreadIdx]) * v[colIdx];

        // Parallel reduce the scratch matrix
        for (int reduceMask = 2; reduceMask <= N + 1; reduceMask <<= 1)
        {
            __syncthreads();
            if ((colIdx & (reduceMask - 1)) == 0)
            {
                scratch[colIdx][rowIdx] += scratch[colIdx + (reduceMask >> 1)][rowIdx];
            }
        }
         
        // Store the output in the vector
        __syncthreads();
        if (colIdx == 0) { w[rowIdx] = scratch[0][rowIdx]; }
    }*/

    // Matrix multiply of an NxM tensor with K-tensor. 
    template<bool Transpose, int N, int M, int V, int W, bool HasGradM, bool HasGradVW>
    __forceinline__ __device__ void MulImpl(const Tensor2D<N, M, HasGradM>& m, const Tensor1D<V, HasGradVW>& v, Tensor1D<W, HasGradVW>& w, float(&scratch)[N][M])
    {
        // Block must have have at least as many threads as the tensor has elements
        CudaAssertDebug(blockDim.x >= N * M);
        // Tensor dimensions must match up
        CudaAssertDebugFmt(Transpose ? (M == V) : (M == W),
            "Tensor dimensions (%ix%i)%s are incompatible with input vector of dimension %i", N, M, Transpose ? "T" : "", V);
        CudaAssertDebugFmt(Transpose ? (N == W) : (N == V),
            "Tensor dimensions (%ix%i)%s are incompatible with output vector of dimension %i", N, M, Transpose ? "T" : "", W);

        // Populate the scratch matrix with the products of the N*N tensor and the N tensor
        // N = cols, M = rows
        __syncthreads();
        const int rowIdx = kThreadIdx % M, colIdx = kThreadIdx / M;
        scratch[colIdx][rowIdx] = m[kThreadIdx] * v[Transpose ? rowIdx : colIdx];

        if (Transpose)
        {
            for (int reduceMask = 2; reduceMask <= M + 1; reduceMask <<= 1)
            {
                __syncthreads();
                if ((rowIdx & (reduceMask - 1)) == 0 && rowIdx + (reduceMask >> 1) < M)
                {
                    scratch[colIdx][rowIdx] += scratch[colIdx][rowIdx + (reduceMask >> 1)];
                }
            }
            __syncthreads();
            if (rowIdx == 0) { w[colIdx] = scratch[colIdx][0]; }
        }
        else
        {
            for (int reduceMask = 2; reduceMask <= N + 1; reduceMask <<= 1)
            {
                __syncthreads();
                if ((colIdx & (reduceMask - 1)) == 0 && colIdx + (reduceMask >> 1) < N)
                {
                    scratch[colIdx][rowIdx] += scratch[colIdx + (reduceMask >> 1)][rowIdx];
                }
            }
            __syncthreads();
            if (colIdx == 0) { w[rowIdx] = scratch[0][rowIdx]; }
        }
    }

    template<int N, int M, bool HasGradM, bool HasGradVW>
    __forceinline__ __device__ void MulT(const Tensor2D<N, M, HasGradM>& m, const Tensor1D<M, HasGradVW>& v, Tensor1D<N, HasGradVW>& w, float(&scratch)[N][M])
    {
        MulImpl<true>(m, v, w, scratch);
    }

    template<int N, int M, bool HasGradM, bool HasGradVW>
    __forceinline__ __device__ void Mul(const Tensor2D<N, M, HasGradM>& m, const Tensor1D<N, HasGradVW>& v, Tensor1D<M, HasGradVW>& w, float(&scratch)[N][M])
    {
        MulImpl<false>(m, v, w, scratch);
    }
}
