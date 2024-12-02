#pragma once

#include "NNUtils.cuh"
#include "core/math/MathUtils.h"
#include "Activation.cuh"

namespace Flair
{
    namespace NN
    {
        namespace Loss
        {
            struct L1
            {
                static __forceinline__ __device__ float F(const float& f, const float& t) 
                {
                    return fabsf(f - t);
                }  

                static __forceinline__ __device__ float dF(const float& f, const float& t)
                {
                    return sign(f - t);
                }
            };

            struct L2
            {
                static __forceinline__ __device__ float F(const float& f, const float& t)
                {
                    return sqr(f - t);
                }

                static __forceinline__ __device__ float dF(const float& f, const float& t)
                {
                    return 2 * (f - t);
                }
            };

            struct BinaryCrossEntropy
            {
                static __forceinline__ __device__ float F(float f, float t)
                {
                    Activation::Sigmoid::F(f);
                    Activation::Sigmoid::F(t);
                    return -(t * logf(1e-10 + f) + (1 - t) * logf(1e-10 + 1 - f));
                }

                static __forceinline__ __device__ float dF(float f, float t)
                {
                    const float dSigmoid = Activation::Sigmoid::dF(f);
                    Activation::Sigmoid::F(f);
                    Activation::Sigmoid::F(t);
                    return ((1 - t) / fmaxf(1e-10f, 1 - f) - t / fmaxf(1e-10f, f)) * dSigmoid;
                }
            };
        }
    }
}