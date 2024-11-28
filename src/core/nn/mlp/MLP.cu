#include "MLP.cuh"
#include "tests/cuda/TensorTests.cuh"
#include "core/utils/HighResTimer.h"
#include "../ContinuousRandomVariable.cuh"
#include "../Indirection.cuh"
#include "MLPKernels.cuh"
#include "../Modules.cuh"
#include <fstream>
#include <thread>
#include <chrono>

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

        void MLP::Train(const DataLoader<Tensor1D<16, false>>& dataset)
        {
            //using Activation = LeakyReLU;
            using ActivationFunction = Activation::LeakyReLU;

            using LossFunction = Loss::L1;
            
            // Define the model policy
            using Policy = MLPPolicy<kWidth, kDepth, kMiniBatchSize, ActivationFunction, LossFunction>;
            
            RunTensorTests(false);

            printf_red("TrainingCtx: %i bytes\n", sizeof(TrainingCtx<Policy>));

            UniformDistribution rng(0, 1, 10);
            Cuda::Vector<MiniBatchData<Policy>, kCudaMemMirrored> deviceMiniBatch(kMiniBatchSize);
            Cuda::Object<float> deviceLoss;
            Cuda::Object<Optimiser> deviceOptimiser;
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
            constexpr int kNumMiniBatches = 5000;
            int miniBatchIdx = 0;
            float meanLoss;
            HighResTimer timer;
            for (int epochIdx = 0; epochIdx < kNumEpochs && miniBatchIdx < kNumMiniBatches; ++epochIdx)
            {                                                
                // Reset the kernel data (loss values, etc.) for the new epoch
                PrepareNewEpoch(kernelData);
                
                for (int sampleIdx = 0; sampleIdx < kernelData.batchSize; sampleIdx += Policy::kMiniBatchSize, ++miniBatchIdx)
                {
                    // Estimate the gradients
                    EstimateGradients(kernelData, sampleIdx);

                    // Reduce gradients 
                    ReduceGradients(kernelData, sampleIdx);

                    // Optimiser step
                    Descend(kernelData);

                    IsOk(cudaDeviceSynchronize());

                    meanLoss = deviceLoss.Download();// / std::ceil(kernelData.batchSize / float(Policy::kMiniBatchSize));
                    //printf("Epoch %i: L1 = %.10f\n", epochIdx, meanLoss);
                    file << tfm::format("%i %f ", miniBatchIdx, meanLoss);

                    //using namespace std::chrono_literals;
                    //std::this_thread::sleep_for(100ms);
                }

                // Record the loss
                //meanLoss = deviceLoss.Download();// / std::ceil(kernelData.batchSize / float(Policy::kMiniBatchSize));
                if (epochIdx % 100 == 0) 
                { 
                    printf("Epoch %i: L1 = %.10f\n", epochIdx, meanLoss); 

                    //deviceMiniBatch.Download();
                    //deviceMiniBatch[0].mlp.layers[2].w.Print(true);
                }

                // Shuffle the 
                sampleIdxs.Shuffle();
            }
        }
    }
}