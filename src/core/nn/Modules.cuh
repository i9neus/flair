#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include "Tensor2D.cuh"

namespace Flair
{
    namespace NN
    {
        // Fully-connected layer with weights and biases
        template<int N, bool HasGrad>
        struct Linear
        {
        public:
            Tensor2D<N, N, HasGrad>   w; 
            Tensor1D<N, HasGrad>      b; 

        public:
            __inline__ __host__ __device__ void ZeroGrad()
            {
                w.ZeroGrad();
                b.ZeroGrad();
            }
        };

        template<int Width, int Depth, bool HasGrad>
        struct SequentialLayers
        {
        public:
            Linear<Width, HasGrad> layers[Depth];       // Fully-connected layer weights and biases

        public:
            template<typename RNG>
            __host__ void Initialise(RNG& rng)
            {
                for (int layerIdx = 0; layerIdx < Depth; ++layerIdx)
                {
                    layers[layerIdx].w.Initialise(rng);
                    layers[layerIdx].b.Initialise(rng);
                }

                ZeroGrad();
            }

            __inline__ __host__ __device__ void ZeroGrad()
            {
                for (int layerIdx = 0; layerIdx < Depth; ++layerIdx)
                {
                    layers[layerIdx].ZeroGrad();
                }
            }
        };
    }
}
