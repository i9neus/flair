#pragma once

#include "core/utils/cuda/CudaObject.cuh"
#include "core/utils/cuda/CudaVector.cuh"
#include "core/math/MathUtils.h"
#include "../TensorOps.cuh"
#include <memory>
#include <functional>
#include <vector>

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

            using ReadBatchFunctor = std::function<bool(std::vector<Sample>&, const int)>;
            using WriteBatchFunctor = std::function<void(const std::vector<Sample>&, const int)>;

        private:
            Cuda::Vector<char, kCudaMemMirrored>  m_deviceModelData;

        public:
            MLP();

            void Initialise();
            void Train(const std::vector<Sample>&, const std::vector<Sample>&);
            void Infer(ReadBatchFunctor readBatch, WriteBatchFunctor writeBatch);
        };
    }  
}