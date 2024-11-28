#include "MLP.cuh"
#include "tests/cuda/TensorTests.cuh"
#include "core/utils/HighResTimer.h"
#include "../ContinuousRandomVariable.cuh"
#include "../Indirection.cuh"
#include "MLPKernels.cuh"
#include "../Modules.cuh"
#include <fstream>

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

            printf_red("TrainingCtx: %i bytes\n", sizeof(TrainingCtx<Policy>));

            UniformDistribution rng(0, 1, 10);
            Cuda::Vector<MiniBatchData<Policy>, kCudaMemMirrored> deviceMiniBatch(kMiniBatchSize);
            Cuda::Object<float, kCudaMemMirrored> deviceLoss;
            Cuda::Object<Optimiser, kCudaMemMirrored> deviceOptimiser;
            Cuda::Vector<Sample, kCudaMemDevice> deviceInputSamples(dataset.Size());
            Cuda::Vector<Sample, kCudaMemDevice> deviceTargetSamples(dataset.Size());

            // Determininstically initialise the mini-batch weights and the optimiser 
            deviceMiniBatch[0].mlp.Initialise(UniformDistribution(0, 1, std::hash<int>{}(0)));            
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

            // Initialise the kernel data structure
            KernelData<Policy> kernelData;
            kernelData.miniBatch = deviceMiniBatch.GetDeviceData();
            kernelData.inputVecs = deviceInputSamples.GetDeviceData();
            kernelData.targetVecs = deviceTargetSamples.GetDeviceData();
            kernelData.optimiser = deviceOptimiser.GetDeviceData();
            kernelData.sampleIdxs = sampleIdxs->GetDeviceData();
            kernelData.loss = deviceLoss.GetDeviceData();
            kernelData.batchSize = dataset.Size();

            std::ofstream file("C:/Unity/SyntheticGS/Assets/HDRI/Loss.dat", std::ios::out);

            constexpr int kNumEpochs = 1000;
            HighResTimer timer;
            for (int epochIdx = 0; epochIdx < kNumEpochs; ++epochIdx)
            {                                
                // Estimate the gradients
                EstimateGradients(kernelData);

                // Reduce gradients 
                ReduceGradients(kernelData);                

                // Optimiser step
                Descend(kernelData);

                deviceLoss.Download();
                if(epochIdx % 100 == 0)
                    printf("Epoch %i: L1 = %.10f\n", epochIdx, *deviceLoss);
                file << tfm::format("%i %f ", epochIdx, *deviceLoss);

                sampleIdxs.Shuffle();

                IsOk(cudaDeviceSynchronize());                
            }
        }
    }
}