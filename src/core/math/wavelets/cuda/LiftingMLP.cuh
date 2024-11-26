#pragma once

#include "core/image/Image.h"
#include "core/utils/cuda/MirroredObject.cuh"
#include "core/utils/cuda/MirroredVector.cuh"
#include "core/utils/cuda/RandomDistribution.cuh"
#include "core/math/MathUtils.h"
#include "core/utils/cuda/NN.cuh"

namespace Flair
{
    namespace NN
    {
        // The number of weight matrices (equal to hidden layers - 1)
        static constexpr int kDepth = 3;

        // The width of the network (number nodes)
        static constexpr int kWidth = 4;
         
        // The size of the mini batch
        static constexpr int kMiniBatchSize = 64; 

        using Model = MLP<kWidth, kDepth>;
        using Sample = Tensor1D<kWidth>;
        using Optimiser = MLP<kWidth, kDepth>;
    }
    
    class LiftingMLP
    {    
    public:

    private:
        //const Image1f&          m_inputImage;
        std::vector<float>      m_modelWeights;

        Cuda::MirroredObject<NN::Model> m_deviceModel;
        Cuda::MirroredObject<NN::Sample> m_deviceSample;
        Cuda::MirroredObject<NN::Sample> m_deviceTarget;

    public:
        LiftingMLP();
        //LiftingMLP(const Image1f& inputImage);

        void Initialise();
        void Train();
        void Test();
    };
}