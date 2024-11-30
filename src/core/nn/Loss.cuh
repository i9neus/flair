#pragma once

#include "NNUtils.cuh"
#include "core/math/MathUtils.h"

namespace Flair
{
    namespace NN
    {
        namespace Loss
        {
            // Constant activation function
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

            // Leaky ReLU activation function
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

            
        }
    }
}