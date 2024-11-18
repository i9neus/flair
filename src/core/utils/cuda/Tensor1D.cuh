#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"

namespace Flair
{
    template<int N>
    struct Tensor1D
    {
    private:
        float data[N];
    public:        
        float grad[N];

    private:
        __host__ void PrintImpl(const bool showGrad) const
        {
            printf("{ ");
            for (int rowIdx = 0; rowIdx < N; ++rowIdx)
            {
                if (showGrad)
                {
                    printf_yellow("%s%.8f", rowIdx ? ", " : "", grad[rowIdx]);
                }
                else
                {
                    printf_green("%s%.8f", rowIdx ? ", " : "", data[rowIdx]);
                }
            }
            printf(" }");
        }

    public:
        __host__ __device__ Tensor1D()
        {
#if !defined(__CUDA_ARCH__)
            memset(this, 0, sizeof(Tensor1D));
#endif
        }

        __host__ __device__ Tensor1D(const float(&d)[N])
        {
            memcpy(data, d, sizeof(float) * N);
            memset(grad, 0, sizeof(float) * N);
        }

        __host__ __device__ Tensor1D(const float(&d)[N][1])
        {
            memcpy(data, &d[0][0], sizeof(float) * N);
            memset(grad, 0, sizeof(float) * N);
        }

        template<typename RNG>
        __host__ void Initialise(RNG& rng)
        {
            for (int i = 0; i < N; ++i) { data[i] = rng() / N; }
            memset(grad, 0, sizeof(float) * N);
        }

        __forceinline__ __device__ void ZeroGrad() 
        { 
            if (kKernelIdx < N) { grad[kKernelIdx] = 0; }
        }

        __forceinline__ __host__ __device__ float& operator[](const int idx) { return data[idx]; }
        __forceinline__ __host__ __device__ const float& operator[](const int idx) const { return data[idx]; }
        __forceinline__ __host__ __device__ float* Data() { return data; }
        __forceinline__ __host__ __device__ const float* Data() const { return data; }

        __host__ __device__ __forceinline__ void Print() const { PrintImpl(false); }
        __host__ __device__ __forceinline__ void PrintGrad() const { PrintImpl(true); }

        __forceinline__ __device__ __host__ Tensor1D& operator+=(const Tensor1D& rhs)
        {
            for (int i = 0; i < N; ++i) { data[i] = rhs.data[i]; }
            return *this;
        }

        __forceinline__ __device__ __host__ Tensor1D& operator-=(const Tensor1D& rhs)
        {
            for (int i = 0; i < N; ++i) { data[i] = rhs.data[i]; }
            return *this;
        }
    };   

    template<int N> __forceinline__ __host__ __device__ Tensor1D<N> operator+(const Tensor1D<N>& lhs, const Tensor1D<N>& rhs)
    {
        Tensor1D<N> r;
        for (int i = 0; i < N; ++i) { r[i] = lhs[i] + rhs[i]; }
        return r;
    }

    template<int N> __forceinline__ __host__ __device__ Tensor1D<N> operator-(const Tensor1D<N>& lhs, const Tensor1D<N>& rhs)
    {
        Tensor1D<N> r;
        for (int i = 0; i < N; ++i) { r[i] = lhs[i] - rhs[i]; }
        return r;
    }

    template<int N> __forceinline__ __host__ __device__ float CwiseMax(const Tensor1D<N>& t)
    {
        float m = t[0];
        for (int i = 1; i < N; ++i) { m = fmaxf(m, t[i]); }
        return m;
    }

    template<int N> __forceinline__ __host__ __device__ float CwiseMin(const Tensor1D<N>& t)
    {
        float m = t[0];
        for (int i = 0; i < N; ++i) { m = fminf(m, t[i]); }
        return m;
    }

    template<int N> __forceinline__ __host__ __device__ Tensor1D<N> Abs(const Tensor1D<N>& t)
    {
        Tensor1D<N> r;
        for (int i = 0; i < N; ++i) { r[i] = fabsf(t[i]); }
        return r;
    }
}
