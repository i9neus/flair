#pragma once

#include "core/image/Image.h"
#include "core/utils/cuda/ManagedObject.cuh"
#include "core/utils/cuda/RandomDistribution.cuh"
#include "core/math/MathUtils.h"
#include "core/utils/cuda/NN.cuh"

namespace Flair
{
    namespace NN
    {
        // The number of weight matrices (equal to hidden layers - 1)
        static constexpr uint32_t kDepth = 3;

        // The width of the network (number nodes)
        static constexpr uint32_t kWidth = 4;
 

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

        Cuda::HostDeviceObject<NN::Model> m_deviceModel;
        Cuda::HostDeviceObject<NN::Sample> m_deviceSample;
        Cuda::HostDeviceObject<NN::Sample> m_deviceTarget;

    public:
        LiftingMLP();
        //LiftingMLP(const Image1f& inputImage);

        void Initialise();
        void Train();
        void Test();
    };
}