#pragma once

#include "Tensor2D.cuh"

namespace Flair
{
    template<typename Type, int NumElements>
    class Scratchpad
    {
    public:
        using TypeT = Type;
        __device__ constexpr static int Size() { return NumElements; }

    private:        
        Type data[NumElements];

    public:
        Scratchpad() = default;

        template<int M> 
        __device__ Type& At(int n, int m)
        {
            CudaAssertDebugFmt(n * M + m < NumElements, "Scratchpad access out of bounds: %i * %i + %i = %i >= %i", n, M, m, n* M + m, NumElements);
            return data[n * M + m];
        }

        __device__ Type& At(int n)
        {
            CudaAssertDebugFmt(n < NumElements, "Scratchpad access out of bounds: %i >= %i", n, NumElements);
            return data[n];
        }
    };

    // Matrix multiply of an NxM tensor with K-tensor. 
    template<int N, int M, int V, int W, bool HasGrad, typename ScratchpadT>
    __forceinline__ __device__ void Mul(const Tensor2D<N, M, HasGrad>& X, const Tensor1D<V, HasGrad>& v, Tensor1D<W, HasGrad>& w, ScratchpadT& scratch)
    {
        // Block must have have at least as many threads as the tensor has elements
        static_assert(N <= V && M <= W, "Vector dimensions must be at least as large as tensor dimensions");

        using TensorT = Tensor2D<N, M, HasGrad>;
        const int rowIdx = kThreadIdx % M, colIdx = kThreadIdx / M;
        constexpr int Shift = ((TensorT::kConcurrentN & (TensorT::kConcurrentN - 1)) == 0) ? 0 : 1;
        const bool isEvalThread = kThreadIdx < TensorT::kConcurrentN * M;

        // If one thread maps to one tensor element, things are simpler
        if (TensorT::kNPerThread == 1)
        {
            if (isEvalThread)
            {
                scratch.At<M>(colIdx, rowIdx) = X[kThreadIdx] * v[colIdx];
            }

            // Reduce the coefficients. If N is a power of two, the reduce loop can run for one fewer iterations
            for (int reduceMask = 2; (reduceMask >> Shift) <= N; reduceMask <<= 1)
            {
                __syncthreads();
                if (isEvalThread && (colIdx & (reduceMask - 1)) == 0 && colIdx + (reduceMask >> 1) < N)
                {
                    scratch.At<M>(colIdx, rowIdx) += scratch.At<M>(colIdx + (reduceMask >> 1), rowIdx);
                }
            }
        }
        else
        {
            // Iterate over the range 
            __syncthreads();
            if (isEvalThread)
            {
                float& sigma = scratch.At<M>(colIdx, rowIdx);
                sigma = 0;
                for (int k = 0, c = colIdx * TensorT::kNPerThread; c < N && k < TensorT::kNPerThread; ++k, ++c)
                {
                    sigma += X(c, rowIdx) * v[c];
                }
            }

            // Reduce the coefficients. 
            for (int reduceMask = 2; (reduceMask >> Shift) <= TensorT::kConcurrentN; reduceMask <<= 1)
            {
                __syncthreads();
                if (isEvalThread && (colIdx & (reduceMask - 1)) == 0 && colIdx + (reduceMask >> 1) < TensorT::kConcurrentN)
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
        constexpr int Shift = ((TensorT::kConcurrentM & (TensorT::kConcurrentM - 1)) == 0) ? 0 : 1;
        const bool isEvalThread = kThreadIdx < N * TensorT::kConcurrentM;
        int colIdx, rowIdx;

        // If one thread maps to one tensor element, things are simpler
        if (TensorT::kMPerThread == 1)
        {
            // Reduce the coefficients. If M is a power of two, the reduce loop can run for one fewer iterations
            if (isEvalThread)
            {
                rowIdx = kThreadIdx % M; colIdx = kThreadIdx / M;
                scratch.At<M>(colIdx, rowIdx) = X[kThreadIdx] * v[rowIdx];
            }
            for (int reduceMask = 2; (reduceMask >> Shift) <= M; reduceMask <<= 1)
            {
                __syncthreads();
                if (isEvalThread && (rowIdx & (reduceMask - 1)) == 0 && rowIdx + (reduceMask >> 1) < M)
                {
                    scratch.At<M>(colIdx, rowIdx) += scratch.At<M>(colIdx, rowIdx + (reduceMask >> 1));
                }
            }
        }
        else
        {
            // Iterate over the range 
            __syncthreads();
            if (isEvalThread)
            {
                colIdx = kThreadIdx % N; rowIdx = kThreadIdx / N;
                float& sigma = scratch.At<TensorT::kConcurrentM>(colIdx, rowIdx);
                sigma = 0;
                for (int k = 0, r = rowIdx * TensorT::kMPerThread, i = M * colIdx + r;
                    r < M && k < TensorT::kMPerThread;
                    ++k, ++r)
                {
                    //CudaAssertFmt(i + k < N* M, "Matrix out-of-bounds access: %i in %i x %i", i + k, N, M);
                    //CudaAssertFmt(r < M, "Row out-of-bounds access: %i in %i", r, M);
                    sigma += X[i + k] * v[r];
                }
            }

            // Reduce the coefficients. If N is a power of two, the reduce loop can run for one fewer iterations
            for (int reduceMask = 2; reduceMask <= (TensorT::kConcurrentM << Shift); reduceMask <<= 1)
            {
                __syncthreads();
                if (isEvalThread && (rowIdx & (reduceMask - 1)) == 0 && rowIdx + (reduceMask >> 1) < TensorT::kConcurrentM)
                {
                    scratch.At<TensorT::kConcurrentM>(colIdx, rowIdx) += scratch.At<TensorT::kConcurrentM>(colIdx, rowIdx + (reduceMask >> 1));                   
                }
            }
        }

        __syncthreads();
        if (rowIdx == 0 && colIdx < N) { w[colIdx] = scratch.At<TensorT::kConcurrentM>(colIdx, 0); }
    }    

    // Matrix multiply of an NxM tensor with K-tensor. 
    template<int N, int M, int V, bool HasGrad>
    __host__ __device__ static Tensor1D<M, HasGrad> Mul(const Tensor2D<N, M, HasGrad>& X, const Tensor1D<V, HasGrad>& v)
    {
        CudaAssertFmt(V >= N, "Vector dimension must be at least as large as tensor dimensions %i x %i", N, M);

        Tensor1D<M, HasGrad> w;
        for (int m = 0; m < M; ++m)
        {
            w[m] = MapReduceSum<N>([&](int n) -> float { return X(n, m) * v[n]; });
        }
        return w;
    }

    // Matrix multiply of the transpose of an NxM tensor with K-tensor. 
    template<int N, int M, int V, bool HasGrad>
    __host__ __device__ static Tensor1D<N, HasGrad> MulT(const Tensor2D<N, M, HasGrad>& X, const Tensor1D<V, HasGrad>& v)
    {
        // Block must have have at least as many threads as the tensor has elements
        static_assert(V >= M, "Vector dimensions must be at least as large as tensor dimensions");

        Tensor1D<N, HasGrad> w;
        for (int n = 0; n < N; ++n)
        {
            w[n] = MapReduceSum<M>([&](int m) -> float { return X(n, m) * v[m]; });
        }
        return w;
    }
}
