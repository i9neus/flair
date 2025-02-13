#pragma once

#include "NNUtils.cuh"
#include "core/math/MathUtils.h"

namespace Flair
{
    namespace NN
    {
        namespace Activation
        {            
            // Constant activation function
            struct Linear
            {
                static __forceinline__ __host__ __device__ float F(const float f) { return f; }                
                static __forceinline__ __host__ __device__ float dF(const float f) { return 1; }
            };

            // Leaky ReLU activation function
            struct LeakyReLU
            {
                static __forceinline__ __host__ __device__ float F(const float f)
                {
                    return (f < 0.) ? (f * 1e-2f) : f;
                }

                static __forceinline__ __host__ __device__ float dF(const float f)
                {
                    return (f < 0.) ? 1e-2f : 1.f;
                }
            };

            // Hyperbolic tangent activation function
            struct TanH
            {
                static __forceinline__ __host__ __device__ float F(const float f)
                {
                    return 2 / (1.f + expf(-2 * f)) - 1;
                }

                static __forceinline__ __host__ __device__ float dF(const float f)
                {
                    const float ef = expf(-2 * f);
                    return (4 * ef) / sqr(1 + ef);
                }
            };

            // Sigmoid activation function
            struct Sigmoid
            {
                static __forceinline__ __host__ __device__ float F(const float f)
                {
                    return 1 / (1 + expf(-f));
                }

                static __forceinline__ __host__ __device__ float dF(const float f)
                {
                    const float ef = expf(-f);
                    return ef / sqr(1. + ef);
                }
            };

            // Sinusoidal activation function
            struct Sine
            {
                static __forceinline__ __host__ __device__ float F(const float f)
                {
                    return sinf(30. * f);
                }

                static __forceinline__ __host__ __device__ float dF(const float f)
                {
                    return 30. * cosf(30. * f);
                }
            };

            // Sinusoidal activation function
            struct SineReLU
            {
                static __forceinline__ __host__ __device__ float F(const float f)
                {
                    return sinf(30. * f) * ((f < 0.) ? 1e-2f : 1.f);
                }

                static __forceinline__ __host__ __device__ float dF(const float f)
                {
                    return 30. * cosf(30. * f) * ((f < 0.) ? 1e-2f : 1.f);
                }
            };
        }
    }
}