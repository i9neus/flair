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
        class Linear
        {
        public: 
            Tensor2D<Width, Width>   w;  // Weights
            Tensor1D<Width>          b;  // Biases

        public:
            Linear() = default;

            __inline__ __host__ __device__ void ZeroGrad()
            {
                w.ZeroGrad();
                b.ZeroGrad();
            }
        };

        template<int Width, int Depth>
        class MLP
        {
        public:
            Linear<Width> layers[Depth];       // Fully-connected layer weights and biases

        public:
            MLP() = default;

            template<typename RNG>
            __host__ void Initialise(RNG& rng)
            {
                for (int layerIdx = 0; layerIdx < kDepth; ++layerIdx)
                {
                    layers[layerIdx].w.Initialise(rng);
                    layers[layerIdx].b.Initialise(rng);
                }
            }

            __inline__ __host__ __device__ void ZeroGrad()
            {
                for (int layerIdx = 0; layerIdx < kDepth; ++layerIdx)
                {
                    layers[layerIdx].ZeroGrad();
                }
            }
        };
    }
}
