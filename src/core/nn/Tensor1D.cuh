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
        enum : int 
        { 
            kN = N, 
            kSize = SizeType<N, HasGrad>::Value ,

#if defined(_DEBUG)
            kIsGuarded = 1
#else
            kIsGuarded = 0
#endif
        };

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

        __host__ __device__ Tensor1D(const float& value)
        {
            *this = value;
        }

        template<typename RNG>
        __host__ void Initialise(RNG& rng)
        {
            const float norm = kRoot2 * std::sqrt(1.0f / N);
            //const float norm = std::sqrt(1.f / N);
            for (int i = 0; i < N; ++i) 
            { 
                data[i] = rng() * norm;
            }
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

        __forceinline__ __host__ __device__ static void AssertValidIdx(const int idx)
        {
            CudaAssertDebugFmt(idx < N, "Out of bounds index to Tensor1: %i >= %i", idx, N);
        }

        __forceinline__ __host__ __device__ float& operator[](const int idx) 
        {
            if (kIsGuarded) { AssertValidIdx(idx); }
            return data[idx]; 
        }

        __forceinline__ __host__ __device__ const float& operator[](const int idx) const 
        { 
            if (kIsGuarded) { AssertValidIdx(idx); }
            return data[idx];
        }

        // Allows assignment between tensors of unequal lengths
        template<int OtherN, bool OtherHasGrad>
        __forceinline__ __host__ __device__ Tensor1D& operator=(const Tensor1D<OtherN, OtherHasGrad>& rhs)
        {
            for (int i = 0; i < N && i < otherN; ++i) { data[i] = rhs[i]; }
            return *this;
        }

        __forceinline__ __host__ __device__ float& Grad(const int idx) 
        { 
            static_assert(HasGrad, "This tensor does not have gradients.");
            if (kIsGuarded) { AssertValidIdx(idx); }
            return data[N + idx];
        }

        __forceinline__ __host__ __device__ const float& Grad(const int idx) const 
        { 
            static_assert(HasGrad, "This tensor does not have gradients.");
            if (kIsGuarded) { AssertValidIdx(idx); }
            return data[N+idx];
        }

        __forceinline__ __host__ __device__ float* Data() { return data; }
        __forceinline__ __host__ __device__ const float* Data() const { return data; }

        __host__ __forceinline__ TensorIterator<float> begin() { return TensorIterator<float>(data, 0); }
        __host__ __forceinline__ TensorIterator<const float> begin() const { return TensorIterator<const float>(data, 0); }
        __host__ __forceinline__ TensorIterator<float> end() { return TensorIterator<float>(data, N); }
        __host__ __forceinline__ TensorIterator<const float> end() const { return TensorIterator<const float>(data, N); }

#define CwiseTensorUnaryOp(op) \
        __forceinline__ __device__ __host__ Tensor1D& operator##op(const Tensor1D& rhs) \
        { \
            for (int i = 0; i < N; ++i) { data[i] ##op rhs.data[i]; } \
            return *this; \
        }

        CwiseTensorUnaryOp(+=)
        CwiseTensorUnaryOp(-=)
        CwiseTensorUnaryOp(*=)
        CwiseTensorUnaryOp(/=)

#undef CwiseTensorUnaryOp

#define CwiseScalarUnaryOp(op) \
        __forceinline__ __device__ __host__ Tensor1D& operator##op(const float rhs) \
        { \
            for (int i = 0; i < N; ++i) { data[i] ##op rhs; } \
            return *this; \
        }
        
        CwiseScalarUnaryOp(+=)
        CwiseScalarUnaryOp(-=)
        CwiseScalarUnaryOp(*=)
        CwiseScalarUnaryOp(/=)
        CwiseScalarUnaryOp(=)

#undef CwiseScalarUnaryOp

#define CwiseTensorBinaryOp(op) \
        __forceinline__ __device__ __host__ Tensor1D operator##op(const Tensor1D& rhs) \
        { \
            Tensor1D t; \
            for (int i = 0; i < N; ++i) { t[i] = data[i] ##op rhs.data[i]; } \
            return t; \
        }

        CwiseTensorBinaryOp(+)
        CwiseTensorBinaryOp(-)
        CwiseTensorBinaryOp(*)
        CwiseTensorBinaryOp(/)

#undef CwiseTensorBinaryOp

#define CwiseScalarBinaryOp(op) \
        __forceinline__ __device__ __host__ Tensor1D operator##op(const float rhs) \
        { \
            Tensor1D t; \
            for (int i = 0; i < N; ++i) { t[i] = data[i] ##op rhs; } \
            return t; \
        }

        CwiseScalarBinaryOp(+)
        CwiseScalarBinaryOp(-)
        CwiseScalarBinaryOp(*)
        CwiseScalarBinaryOp(/)

#undef CwiseScalarBinaryOp

        __host__ __device__ void Print(const bool showGrad = false, const bool scientific = true) const
        {
            CudaAssertFmt(!showGrad || HasGrad, "Tensor does not have gradients to print");
            printf("{ ");
            for (int rowIdx = 0; rowIdx < N; ++rowIdx)
            {
                printf(scientific ? "%s%.4e" : "%s%.8", rowIdx ? ", " : "", data[showGrad ? (N + rowIdx) : rowIdx]);
            }
            printf(" }\n");
        }

        __host__ std::string Format(const bool showGrad = false, const bool scientific = true) const
        {
            CudaAssertFmt(!showGrad || HasGrad, "Tensor does not have gradients to format");
            std::string str = "{ ";
            for (int rowIdx = 0; rowIdx < N; ++rowIdx)
            {
                str += tfm::format(scientific ? "%s%.4e" : "%s%.8", rowIdx ? ", " : "", data[showGrad ? (N + rowIdx) : rowIdx]);

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
