#include "MLP.cuh"
#include "tests/cuda/TensorTests.cuh"
#include "core/utils/HighResTimer.h"
#include "../ContinuousRandomVariable.cuh"
#include "../Indirection.cuh"
#include "MLPKernels.cuh"
#include "../Modules.cuh"

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

        void MLP::Train(const DataLoader<Tensor1D<MLP::kWidth, false>>& dataset)
        {
            // Define the model policy
            using Policy = MLPPolicy<kWidth, kDepth, kMiniBatchSize>;
            
            RunTensorTests(false);

            UniformDistribution rng(0, 1, 10);
            Cuda::Vector<MiniBatchData<Policy>, kCudaMemMirrored> deviceMiniBatch(kMiniBatchSize);
            Cuda::Object<float, kCudaMemMirrored> deviceLoss;
            Cuda::Object<Optimiser, kCudaMemMirrored> deviceOptimiser;
            Cuda::Vector<Sample, kCudaMemDevice> deviceInputSamples(dataset.Size());
            Cuda::Vector<Sample, kCudaMemDevice> deviceTargetSamples(dataset.Size());

            // Determininstically initialise the mini-batch weights and the optimiser 
            for (int i = 0; i < deviceMiniBatch.Size(); ++i)
            {
                deviceMiniBatch[i].mlp.Initialise(UniformDistribution(0, 1, std::hash<int>{}(0)));
            }
            deviceOptimiser->Initialise(Zeros());           
            deviceMiniBatch.Upload();
            deviceOptimiser.Upload();

            // Upload the samples
            auto [inputSamples, targetSamples] = dataset.Data();   
            deviceInputSamples = *inputSamples;
            deviceTargetSamples = *targetSamples;           

            // Create random indirection buffer
            Indirection sampleIdxs(dataset.Size());
            sampleIdxs.Randomise();

            KernelData<Policy> kernelData;
            kernelData.miniBatch = deviceMiniBatch.GetDeviceData();
            kernelData.inputVecs = deviceInputSamples.GetDeviceData();
            kernelData.targetVecs = deviceTargetSamples.GetDeviceData();
            kernelData.optimiser = deviceOptimiser.GetDeviceData();
            kernelData.sampleIdxs = sampleIdxs->GetDeviceData();
            kernelData.loss = deviceLoss.GetDeviceData();

            constexpr int kNumEpochs = 1;
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

                deviceLoss.Download();
                printf("Epoch %i: L1 = %.10f\n", epochIdx, *deviceLoss);

                sampleIdxs.Shuffle();

                IsOk(cudaDeviceSynchronize());                
            }
        }
    }
}