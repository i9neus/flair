#pragma once

#include "NNUtils.cuh"

namespace Flair
{
    template<int N, bool HasGrad = false>
    struct Tensor1D
    {
    private:
        template<int N, bool HasGrad> struct SizeType { enum : int { Value = N*2 }; };
        template<int N> struct SizeType<N, false> { enum : int { Value = N }; };

        float data[SizeType<N, HasGrad>::Value];

    public:
        enum : int { kN = N, kElements = SizeType<N, HasGrad>::Value };

        __host__ __device__ Tensor1D()
        {
#if !defined(__CUDA_ARCH__)
            memset(this, 0, sizeof(Tensor1D));
#endif
        }

        __host__ __device__ Tensor1D(const float(&d)[N])
        {
            memcpy(data, d, sizeof(float) * N);
            ZeroGrad();
        }

        /*__host__ __device__ Tensor1D(const float(&d)[N][1])
        {
            memcpy(data, &d[0][0], sizeof(float) * N);
            ZeroGrad();
        }*/

        template<typename RNG>
        __host__ void Initialise(RNG& rng)
        {
            for (int i = 0; i < N; ++i) { data[i] = rng(); }
            ZeroGrad();
        }

#if defined(__CUDA_ARCH__)
        __forceinline__ __device__ void ZeroGrad() 
        { 
            if (HasGrad && kThreadIdx < N) { data[N + kThreadIdx] = 0; }
        }
#else
        __forceinline__ __host__ void ZeroGrad()
        {
            if (HasGrad) { memset(&data[N], 0, sizeof(float) * N); }
        }
#endif

        __forceinline__ __host__ __device__ float& operator[](const int idx) { return data[idx]; }
        __forceinline__ __host__ __device__ const float& operator[](const int idx) const { return data[idx]; }
        __forceinline__ __host__ __device__ float& Grad(const int idx) { static_assert(HasGrad, "This tensor does not have gradients."); return data[N + idx]; }
        __forceinline__ __host__ __device__ const float& Grad(const int idx) const { static_assert(HasGrad, "This tensor does not have gradients."); return data[N+idx]; }

        __forceinline__ __host__ __device__ float* Data() { return data; }
        __forceinline__ __host__ __device__ const float* Data() const { return data; }

        __host__ __forceinline__ TensorIterator<float> begin() { return TensorIterator<float>(data, 0); }
        __host__ __forceinline__ TensorIterator<const float> begin() const { return TensorIterator<const float>(data, 0); }
        __host__ __forceinline__ TensorIterator<float> end() { return TensorIterator<float>(data, N); }
        __host__ __forceinline__ TensorIterator<const float> end() const { return TensorIterator<const float>(data, N); }

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

        __host__ __device__ void Print(const bool showGrad = false) const
        {
            CudaAssertMsg(!showGrad || HasGrad, "Tensor does not have gradients to print");
            printf("{ ");
            for (int rowIdx = 0; rowIdx < N; ++rowIdx)
            {
                printf("%s%.8f", rowIdx ? ", " : "", data[showGrad ? (N + rowIdx) : rowIdx]);
            }
            printf(" }\n");
        }

        __host__ std::string Format(const bool showGrad = false) const
        {
            CudaAssertMsg(!showGrad || HasGrad, "Tensor does not have gradients to format");
            std::string str = "{ ";
            for (int rowIdx = 0; rowIdx < N; ++rowIdx)
            {
                str += tfm::format("%s%.8f", rowIdx ? ", " : "", data[showGrad ? (N + rowIdx) : rowIdx]);

            }
            str += " }";
            return str;
        }
    };   

    template<int N, bool HG> __forceinline__ __host__ __device__ Tensor1D<N, HG> operator+(const Tensor1D<N, HG>& lhs, const Tensor1D<N, HG>& rhs)
    {
        Tensor1D<N, HG> r;
        for (int i = 0; i < N; ++i) { r[i] = lhs[i] + rhs[i]; }
        return r;
    }

    template<int N, bool HG> __forceinline__ __host__ __device__ Tensor1D<N, HG> operator-(const Tensor1D<N, HG>& lhs, const Tensor1D<N, HG>& rhs)
    {
        Tensor1D<N, HG> r;
        for (int i = 0; i < N; ++i) { r[i] = lhs[i] - rhs[i]; }
        return r;
    }

    template<int N, bool HG> __forceinline__ __host__ __device__ float CwiseMax(const Tensor1D<N, HG>& t)
    {
        float m = t[0];
        for (int i = 1; i < N; ++i) { m = fmaxf(m, t[i]); }
        return m;
    }

    template<int N, bool HG> __forceinline__ __host__ __device__ float CwiseMin(const Tensor1D<N, HG>& t)
    {
        float m = t[0];
        for (int i = 0; i < N; ++i) { m = fminf(m, t[i]); }
        return m;
    }

    template<int N, bool HG> __forceinline__ __host__ __device__ Tensor1D<N, HG> Abs(const Tensor1D<N, HG>& t)
    {
        Tensor1D<N, HG> r;
        for (int i = 0; i < N; ++i) { r[i] = fabsf(t[i]); }
        return r;
    }
}
