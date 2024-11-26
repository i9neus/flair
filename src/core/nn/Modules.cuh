#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include "Tensor2D.cuh"

namespace Flair
{
    namespace NN
    {
        // Fully-connected layer
        template<int Width>
        struct Linear
        {
        public:
            Tensor2D<Width, Width, true>   w;  // Weights
            Tensor1D<Width, true>          b;  // Biases

        public:
            __inline__ __host__ __device__ void ZeroGrad()
            {
                w.ZeroGrad();
                b.ZeroGrad();
            }
        };

        template<int Width, int Depth>
        struct SequentialLayers
        {
        public:
            Linear<Width> layers[Depth];       // Fully-connected layer weights and biases

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
