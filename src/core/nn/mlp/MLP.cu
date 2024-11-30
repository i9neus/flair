#include "MLP.cuh"
#include "tests/cuda/TensorTests.cuh"
#include "core/utils/HighResTimer.h"
#include "../ContinuousRandomVariable.cuh"
#include "../Permute.cuh"
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

        const std::vector<MLP::Sample> MLP::Train(const DataLoader<MLP::Sample>& dataset)
        {
            using ActivationFunction = Activation::LeakyReLU;
            using LossFunction = Loss::L1;
            using LearningRate = std::ratio<2, 100>;
            
            // Define the model policy
            using Policy = MLPPolicy<kWidth, kDepth, kMiniBatchSize, ActivationFunction, LossFunction, LearningRate>;
            Policy::AssertValid();
            
            RunTensorTests(false);

            printf_red("TrainingCtx: %i bytes\n", sizeof(TrainingCtx<Policy>));

            UniformDistribution rng(0, 1, 10);
            Cuda::Vector<MiniBatchData<Policy>, kCudaMemMirrored> deviceMiniBatch(kMiniBatchSize);
            Cuda::Object<float> deviceLoss;
            Cuda::Object<Optimiser> deviceOptimiser;
            Cuda::Vector<Sample, kCudaMemDevice> deviceInputSamples(dataset.Size());
            Cuda::Vector<Sample, kCudaMemMirrored> deviceOutputSamples(dataset.Size());
            Cuda::Vector<Sample, kCudaMemDevice> deviceTargetSamples(dataset.Size());

            // Determininstically initialise the mini-batch weights and the optimiser 
            deviceMiniBatch[0].mlp.Initialise(NormalRandomDistribution(0, 0.5f, std::hash<int>{}(0)));            
            deviceOptimiser->Initialise(Zeros());           
            deviceMiniBatch.Upload();
            deviceOptimiser.Upload();

            // Upload the samples
            auto [inputSamples, targetSamples] = dataset.Data();   
            deviceInputSamples <<= *inputSamples;
            deviceTargetSamples <<= *targetSamples;           

            // Create random indirection buffer
            Permutation sampleIdxs(dataset.Size());
            sampleIdxs.Randomise();

            // Initialise the kernel data structure
            KernelData<Policy> kernelData;
            kernelData.miniBatch = deviceMiniBatch.GetDeviceData();
            kernelData.inputVecs = deviceInputSamples.GetDeviceData();
            kernelData.outputVecs = deviceOutputSamples.GetDeviceData();
            kernelData.targetVecs = deviceTargetSamples.GetDeviceData();
            kernelData.optimiser = deviceOptimiser.GetDeviceData();
            kernelData.sampleIdxs = sampleIdxs.GetDeviceData();
            kernelData.loss = deviceLoss.GetDeviceData();
            kernelData.batchSize = dataset.Size();

            constexpr int kMaxEpochs = 100;
            constexpr int kMaxMiniBatches = 2000000;
            int miniBatchIdx = 0;
            HighResTimer timer;

            std::vector<std::pair<int, float>> epochLoss;
            std::vector<float> miniBatchLoss;

            /*printf("Ref:\n");
            (*targetSamples)[0].Print();*/

            for (int epochIdx = 0; epochIdx < kMaxEpochs && miniBatchIdx < kMaxMiniBatches; ++epochIdx)
            {                                                
                const float miniBatchSize = Policy::kMiniBatchSize;// (miniBatchIdx > 5000) ? Policy::kMiniBatchSize : 1;
                
                // Reset the kernel data (loss values, etc.) for the new epoch
                PrepareNewEpoch(kernelData, miniBatchSize);
                
                float meanLoss = 0;
                for (int sampleIdx = 0; sampleIdx < kernelData.batchSize; sampleIdx += miniBatchSize, ++miniBatchIdx)
                {
                    // Estimate the gradients
                    EstimateGradients(kernelData, sampleIdx, miniBatchSize);

                    // Reduce gradients 
                    ReduceGradients(kernelData, sampleIdx, miniBatchSize);

                    // Optimiser step
                    Descend(kernelData);

                    IsOk(cudaDeviceSynchronize());

                    const float loss = deviceLoss.Download();
                    miniBatchLoss.emplace_back(loss);
                    if (miniBatchIdx == 0) { epochLoss.emplace_back(0, loss); }
                    meanLoss += loss;
                    //printf("Epoch %i: L1 = %.10f\n", epochIdx, meanLoss);
                    //file << tfm::format("%i %f ", miniBatchIdx, meanLoss);

                    /*auto& v = deviceOutputSamples.Download();
                    printf("%i: ", miniBatchIdx);
                    for (int i = 0; i < 16; ++i)
                    {
                        printf("%f ", v[0][i]);
                    }
                    printf("\n");*/

                    //using namespace std::chrono_literals;
                    //std::this_thread::sleep_for(100ms);
                }

                // Record the loss
                meanLoss /= std::ceil(kernelData.batchSize / float(miniBatchSize));
                epochLoss.emplace_back(miniBatchIdx, meanLoss);

                //meanLoss = deviceLoss.Download();// / std::ceil(kernelData.batchSize / float(Policy::kMiniBatchSize));
                if (timer.Get() > 0.5)
                { 
                    printf("Epoch %i: L1 = %.10f\n", epochIdx, meanLoss); 
                    timer.Reset();

                    //deviceMiniBatch.Download();
                    //deviceMiniBatch[0].mlp.layers[2].w.Print(true);
                }

                /*if (epochIdx == kMaxEpochs - 1)
                {
                    deviceMiniBatch.Download();
                    for (int i = 0; i < deviceMiniBatch.Size(); ++i)
                    {
                        printf("------------------------------ %i -------------------------------\n", i);
                        for (auto& layer : deviceMiniBatch[i].mlp.layers)
                        {
                            layer.w.Print(true);
                            layer.b.Print(true);
                        }
                        printf("\n\n");
                    }
                }*/

                // Shuffle the indirection indices
                sampleIdxs.Shuffle();
            }

            std::ofstream file("C:/Unity/SyntheticGS/Assets/HDRI/Loss.dat", std::ios::out);
            for(int i = 0; i < miniBatchLoss.size(); ++i)
            {
                file << tfm::format("%i %f ", i, miniBatchLoss[i]);
            }
            file << std::endl;
            for (int i = 0; i < epochLoss.size(); ++i)
            {
                file << tfm::format("%i %f ", epochLoss[i].first, epochLoss[i].second);
            }
            file.close();

            // Inference
            for (int sampleIdx = 0; sampleIdx < kernelData.batchSize; sampleIdx += Policy::kMiniBatchSize, ++miniBatchIdx)
            {
                Infer(kernelData, sampleIdx, Policy::kMiniBatchSize);
            }
            
            return deviceOutputSamples.Download();
        }
    }
}