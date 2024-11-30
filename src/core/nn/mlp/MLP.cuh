#pragma once

#include "core/utils/cuda/CudaObject.cuh"
#include "core/utils/cuda/CudaVector.cuh"
#include "core/math/MathUtils.h"
#include "../TensorOps.cuh"
#include "../DataLoader.cuh"
#include <memory>

namespace Flair
{
    namespace NN
    {
        template<int, int> struct SequentialLayers;

        class MLP
        {
        public:
            // The number of weight matrices (equal to hidden layers - 1)
            static constexpr int kDepth = 4;

            // The width of the network (number nodes)
            static constexpr int kWidth = 16;

            // The size of the mini batch
            static constexpr int kMiniBatchSize = 64;

            using Model = SequentialLayers<kWidth, kDepth>;
            using Sample = Tensor1D<kWidth, false>;
            using Optimiser = SequentialLayers<kWidth, kDepth>;

        private:
            /*std::unique_ptr<Cuda::Object<Model>>  m_deviceModel;
            std::unique_ptr<Cuda::Object<Sample>> m_deviceSample;
            std::unique_ptr<Cuda::Object<Sample>> m_deviceTarget;*/

        public:
            MLP();

            void Initialise();
            const std::vector<MLP::Sample> Train(const DataLoader<Sample>& data);
        };
    }  
}