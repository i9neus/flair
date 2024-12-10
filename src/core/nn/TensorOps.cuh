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

        template<int M> 
        __device__ Type& At(int n, int m)
        {
            return data[n * M + m];
        }

        __device__ Type& At(int n)
        {
            return data[n];
        }
    };

    // Matrix multiply of the transpose of an NxM tensor with K-tensor. 
    template<int N, int M, int V, int W, bool HasGrad, typename ScratchpadT>
    __forceinline__ __device__ void MulTLegacy(const Tensor2D<N, M, HasGrad>& X, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w, ScratchpadT& scratch)
    {
        // Block must have have at least as many threads as the tensor has elements
        CudaAssertDebug(blockDim.x >= N * M);
        static_assert(M <= V && N <= W, "Vector dimensions must be at least as large as tensor dimensions");
      
        // Populate the scratch matrix with the products of the N*N tensor and the N tensor
        // N = cols, M = rows
        __syncthreads();
        const int rowIdx = kThreadIdx % M, colIdx = kThreadIdx / M;
        scratch.At<M>(colIdx, rowIdx) = X[kThreadIdx] * v[rowIdx];

        // Reduce the coefficients. If M is a power of two, the reduce loop can run for one fewer iterations
        constexpr int Shift = ((M & (M - 1)) == 0) ? 0 : 1;
        for (int reduceMask = 2; (reduceMask >> Shift) <= M; reduceMask <<= 1)
        {
            __syncthreads();
            if ((rowIdx & (reduceMask - 1)) == 0 && rowIdx + (reduceMask >> 1) < M)
            {
                scratch.At<M>(colIdx, rowIdx) += scratch.At<M>(colIdx, rowIdx + (reduceMask >> 1));
            }
        }
        __syncthreads();
        if (rowIdx == 0) { w[colIdx] = scratch.At<M>(colIdx, 0); }     
    }

    // Matrix multiply of an NxM tensor with K-tensor. 
    template<int N, int M, int V, int W, bool HasGrad, typename ScratchpadT>
    __forceinline__ __device__ void MulLegacy(const Tensor2D<N, M, HasGrad>& X, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w, ScratchpadT& scratch)
    {
        // Block must have have at least as many threads as the tensor has elements
        CudaAssertDebug(blockDim.x >= N * M);
        static_assert(N <= V && M <= W, "Vector dimensions must be at least as large as tensor dimensions");      

        // Populate the scratch matrix with the products of the N*N tensor and the N tensor
        // N = cols, M = rows
        __syncthreads();
        const int rowIdx = kThreadIdx % M, colIdx = kThreadIdx / M;
        scratch.At<M>(colIdx, rowIdx) = X[kThreadIdx] * v[colIdx];

        // Reduce the coefficients. If N is a power of two, the reduce loop can run for one fewer iterations
        constexpr int Shift = ((N & (N - 1)) == 0) ? 0 : 1; 
        for (int reduceMask = 2; (reduceMask >> Shift) <= N; reduceMask <<= 1)
        {
            __syncthreads();
            if ((colIdx & (reduceMask - 1)) == 0 && colIdx + (reduceMask >> 1) < N)
            {
                scratch.At<M>(colIdx, rowIdx) += scratch.At<M>(colIdx + (reduceMask >> 1), rowIdx);
            }
        }
        __syncthreads();
        if (colIdx == 0) { w[rowIdx] = scratch.At<M>(0, rowIdx); }  
    }

    // Matrix multiply of an NxM tensor with K-tensor. 
    template<int N, int M, int V, int W, bool HasGrad, typename ScratchpadT>
    __forceinline__ __device__ void Mul(const Tensor2D<N, M, HasGrad>& X, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w, ScratchpadT& scratch)
    {
        // Block must have have at least as many threads as the tensor has elements
        static_assert(N <= V && M <= W, "Vector dimensions must be at least as large as tensor dimensions");

        using TensorT = Tensor2D<N, M, HasGrad>;
        const int rowIdx = kThreadIdx % M, colIdx = kThreadIdx / M;
        constexpr int Shift = ((TensorT::kNBlocks & (TensorT::kNBlocks - 1)) == 0) ? 0 : 1;

        // If one thread maps to one tensor element, things are simpler
        if (TensorT::kNPerThread == 1)
        {
            scratch.At<M>(colIdx, rowIdx) = X[kThreadIdx] * v[colIdx];

            // Reduce the coefficients. If N is a power of two, the reduce loop can run for one fewer iterations
            for (int reduceMask = 2; (reduceMask >> Shift) <= N; reduceMask <<= 1)
            {
                __syncthreads();
                if ((colIdx & (reduceMask - 1)) == 0 && colIdx + (reduceMask >> 1) < N)
                {
                    scratch.At<M>(colIdx, rowIdx) += scratch.At<M>(colIdx + (reduceMask >> 1), rowIdx);
                }
            }
        }
        else
        {
            // Iterate over the range 
            __syncthreads();
            if (kKernelIdx < TensorT::kNBlocks * M)
            {
                float& sigma = scratch.At<M>(colIdx, rowIdx);
                sigma = 0;
                for (int k = 0, c = colIdx * TensorT::kNPerThread; k < TensorT::kNPerThread; ++k, ++c)
                {
                    sigma += X(c, rowIdx) * v[c];
                }
            }

            // Reduce the coefficients. 
            for (int reduceMask = 2; (reduceMask >> Shift) <= TensorT::kNBlocks; reduceMask <<= 1)
            {
                __syncthreads();
                if (kKernelIdx < TensorT::kNBlocks * M && (colIdx & (reduceMask - 1)) == 0 && colIdx + (reduceMask >> 1) < N)
                {
                    scratch.At<M>(colIdx, rowIdx) += scratch.At<M>(colIdx + (reduceMask >> 1), rowIdx);
                }
            }
        }

        __syncthreads();
        if (colIdx == 0) { w[rowIdx] = scratch.At<M>(0, rowIdx); }
    }

    // Matrix multiply of an NxM tensor with K-tensor. 
    template<int N, int M, int V, int W, bool HasGrad, typename ScratchpadT>
    __forceinline__ __device__ void MulT(const Tensor2D<N, M, HasGrad>& X, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w, ScratchpadT& scratch)
    {
        // Block must have have at least as many threads as the tensor has elements
        static_assert(M <= V && N <= W, "Vector dimensions must be at least as large as tensor dimensions");

        using TensorT = Tensor2D<N, M, HasGrad>;
        constexpr int Shift = ((TensorT::kMBlocks & (TensorT::kMBlocks - 1)) == 0) ? 0 : 1;
        int colIdx, rowIdx; 

        // If one thread maps to one tensor element, things are simpler
        if (TensorT::kMPerThread == 1)
        {
            // Reduce the coefficients. If M is a power of two, the reduce loop can run for one fewer iterations
            rowIdx = kThreadIdx % M; colIdx = kThreadIdx / M;
            scratch.At<M>(colIdx, rowIdx) = X[kThreadIdx] * v[rowIdx];
            for (int reduceMask = 2; (reduceMask >> Shift) <= M; reduceMask <<= 1)
            {
                __syncthreads();
                if ((rowIdx & (reduceMask - 1)) == 0 && rowIdx + (reduceMask >> 1) < M)
                {
                    scratch.At<M>(colIdx, rowIdx) += scratch.At<M>(colIdx, rowIdx + (reduceMask >> 1));
                }
            }
        }
        else
        {
            // Iterate over the range 
            colIdx = kThreadIdx % N; rowIdx = kThreadIdx / N;
            __syncthreads();
            if (kKernelIdx < N * TensorT::kMBlocks)
            {
                float& sigma = scratch.At<M>(colIdx, rowIdx);
                sigma = 0;
                for (int k = 0, r = rowIdx * TensorT::kMPerThread; k < TensorT::kMPerThread; ++k, ++r)
                {
                    sigma += X(colIdx, r) * v[r];
                }
            }

            // Reduce the coefficients. If N is a power of two, the reduce loop can run for one fewer iterations
            for (int reduceMask = 2; (reduceMask >> Shift) <= TensorT::kMBlocks; reduceMask <<= 1)
            {
                __syncthreads();
                if (kKernelIdx < N * TensorT::kMBlocks && (rowIdx & (reduceMask - 1)) == 0 && rowIdx + (reduceMask >> 1) < M)
                {
                    scratch.At<M>(colIdx, rowIdx) += scratch.At<M>(colIdx, rowIdx + (reduceMask >> 1));
                }
            }
        }

        __syncthreads();
        if (rowIdx == 0) { w[colIdx] = scratch.At<M>(colIdx, 0); }
    }

    // Matrix multiply of the transpose of an NxM tensor with K-tensor. 
    template<int N, int M, int V, int W, bool HasGrad>
    __host__ static void MulT(const Tensor2D<N, M, HasGrad>& X, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w)
    {
        // Block must have have at least as many threads as the tensor has elements
        static_assert(W >= N && V >= M, "Vector dimensions must be at least as large as tensor dimensions");

        for (int n = 0; n < N; ++n)
        {
            w[n] = 0;
            for (int m = 0; m < M; ++m)
            {
                w[n] += X(n, m) * v[m];
            }
        }
    }

    // Matrix multiply of an NxM tensor with K-tensor. 
    template<int N, int M, int V, int W, bool HasGrad>
    __host__ static void Mul(const Tensor2D<N, M, HasGrad>& X, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w)
    {
        AssertFmt(V >= N && W >= M, "Vector dimension %i must be at least as large as tensor dimensions %i x %i");

        for (int m = 0; m < M; ++m)
        {
            w[m] = 0;
            for (int n = 0; n < N; ++n)
            {
                w[m] += X(n, m) * v[n];
            }
        }   
    }
}
