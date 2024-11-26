#include "MLP.cuh"
#include "tests/cuda/TensorTests.cuh"
#include "core/utils/HighResTimer.h"
#include "../ContinuousRandomVariable.cuh"
#include "MLPKernels.cuh"
#include "../Modules.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace Flair
{
    namespace NN
    {
        MLP::MLP()
        {
            Initialise();
        }

        void MLP::Initialise()
        {
        }

        void MLP::Train(const DataLoader& dataset)
        {
            // Define the model policy
            using Policy = MLPPolicy<kWidth, kDepth, kMiniBatchSize>;
            
            RunTensorTests(false);

            /*UniformDistribution rng(0, 1, 10);
            Cuda::Vector<Model> deviceModels(kMiniBatchSize);
            Cuda::Object<Optimiser> deviceOptimiser;
            Cuda::Vector<Sample> deviceInputSamples(dataset.Size());
            Cuda::Vector<Sample> deviceTargetSamples(dataset.Size());

            // Initialise the mini-batch and optimiser
            for (int i = 0; i < deviceModels.Size(); ++i)
            {
                deviceModels[i].Initialise(UniformDistribution(0, 1, std::hash<int>{}(i)));
            }
            deviceOptimiser->Initialise(Zeros());  

            


            deviceModels.Upload();
            deviceOptimiser.Upload();
            deviceInput.Upload();

           

            KernelData<Policy> kernelData;
            kernelData.mlp = deviceModel.GetDeviceData();
            kernelData.inputVec = deviceInput.GetDeviceData();
            kernelData.outputVec = deviceOutput.GetDeviceData();
            kernelData.targetVec = deviceTarget.GetDeviceData();
            kernelData.loss = deviceLoss.GetDeviceData();
            kernelData.optimiser = deviceOptimiser.GetDeviceData();

            constexpr int kNumEpochs = 1000;
            HighResTimer timer;
            for (int epochIdx = 0; epochIdx < kNumEpochs; ++epochIdx)
            {
                // Estimate the gradients
                EstimateGradients(kernelData);

                // Reduce gradients
                for (int span = kMiniBatchSize >> 1, stride = 2; span >= 2; span >>= 1, stride <<= 1)
                {
                    ReduceGradients(kernelData, span, stride);
                }

                // Optimiser step
                Descend(kernelData);

                IsOk(cudaDeviceSynchronize());

                
            }*/
        }
    }
}