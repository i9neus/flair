#pragma once

#include "Tensor2D.cuh"

namespace Flair
{
    template<typename TypeT, int Size>
    class Scratchpad
    {
    private:
        enum : int { kSize = Size };
        using Type = TypeT;
        
        Type data[kSize];

    public:
        Scratchpad() = default;

        template<int N, int M> 
        __forceinline__ __device__ Type& At(int n, int m)
        {
            static_assert(M * N <= kSize, "Presumed scratchpad size exceeds actual size");
            return data[n * M + m];
        }

        template<int N> 
        __forceinline__ __device__ Type& At(int n)
        {
            static_assert(N <= kSize, "Presumed scratchpad size exceeds actual size");
            return data[n];
        }
    };

    // Matrix multiply of an NxM tensor with K-tensor. 
    template<bool Transpose, int N, int M, int V, int W, bool HasGrad, typename ScratchpadT>
    __forceinline__ __device__ void MulImpl(const Tensor2D<N, M, HasGrad>& m, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w, ScratchpadT& scratch)
    {
        // Block must have have at least as many threads as the tensor has elements
        CudaAssertDebug(blockDim.x >= N * M);
        
        static_assert(Transpose ? (M <= V && N <= W) : (M <= W && N <= V), "Vector dimensions must be at least as large as tensor dimensions");
        // Tensor dimensions must match up
        //CudaAssertDebugFmt(Transpose ? (M == V) : (M == W),
        //    "Tensor dimensions (%ix%i)%s are incompatible with input vector of dimension %i", N, M, Transpose ? "T" : "", V);
        //CudaAssertDebugFmt(Transpose ? (N == W) : (N == V),
         //   "Tensor dimensions (%ix%i)%s are incompatible with output vector of dimension %i", N, M, Transpose ? "T" : "", W);

        // Populate the scratch matrix with the products of the N*N tensor and the N tensor
        // N = cols, M = rows
        __syncthreads();
        const int rowIdx = kThreadIdx % M, colIdx = kThreadIdx / M;
        scratch.At<N, M>(colIdx, rowIdx) = m[kThreadIdx] * v[Transpose ? rowIdx : colIdx];

        if (Transpose)
        {
            for (int reduceMask = 2; reduceMask <= M + 1; reduceMask <<= 1)
            {
                __syncthreads();
                if ((rowIdx & (reduceMask - 1)) == 0 && rowIdx + (reduceMask >> 1) < M)
                {
                    scratch.At<N, M>(colIdx, rowIdx) += scratch.At<N, M>(colIdx, rowIdx + (reduceMask >> 1));
                }
            }
            __syncthreads();
            if (rowIdx == 0) { w[colIdx] = scratch.At<N, M>(colIdx, 0); }
        }
        else
        {
            for (int reduceMask = 2; reduceMask <= N + 1; reduceMask <<= 1)
            {
                __syncthreads();
                if ((colIdx & (reduceMask - 1)) == 0 && colIdx + (reduceMask >> 1) < N)
                {
                    scratch.At<N, M>(colIdx, rowIdx) += scratch.At<N, M>(colIdx + (reduceMask >> 1), rowIdx);
                }
            }
            __syncthreads();
            if (colIdx == 0) { w[rowIdx] = scratch.At<N, M>(0, rowIdx); }
        }
    }

    template<int N, int M, int V, int W, bool HasGrad, typename ScratchpadT>
    __forceinline__ __device__ void MulT(const Tensor2D<N, M, HasGrad>& m, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w, ScratchpadT& scratch)
    {
        MulImpl<true>(m, v, w, scratch);
    }

    template<int N, int M, int V, int W, bool HasGrad, typename ScratchpadT>
    __forceinline__ __device__ void Mul(const Tensor2D<N, M, HasGrad>& m, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w, ScratchpadT& scratch)
    {
        MulImpl<false>(m, v, w, scratch);
    }
}
