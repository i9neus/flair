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
        template<int, int, bool> struct SequentialLayers;

        class MLP
        {
        public:
            using InputSample = Tensor1D<16, false>;
            using OutputSample = Tensor1D<16, false>;
            using ReadBatchFunctor = std::function<bool(std::vector<InputSample>&, const int)>;
            using WriteBatchFunctor = std::function<void(const std::vector<OutputSample>&, const int)>;

        private:
            Cuda::Vector<float>  m_deviceModelData;

        public:
            MLP();

            void Initialise();
            void Train(const std::vector<InputSample>&, const std::vector<OutputSample>&);
            void Infer(ReadBatchFunctor readBatch, WriteBatchFunctor writeBatch);
        };
    }  
}