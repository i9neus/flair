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

        void MLP::Train(const std::vector<Sample>& inputSamples, const std::vector<Sample>& targetSamples)
        {
            Assert(inputSamples.size() == targetSamples.size());
            
            using ActivationFunction = Activation::LeakyReLU;
            using LossFunction = Loss::L1;
            using LearningRate = std::ratio<1, 100>;
            
            // Define the model policy
            using Policy = MLPTrainingPolicy<kWidth, kDepth, kMiniBatchSize, ActivationFunction, LossFunction, LearningRate>;
            Policy::AssertValid();
            
            RunTensorTests(false);

            printf_red("TrainingCtx: %i bytes\n", sizeof(TrainingCtx<Policy>));

            Cuda::Vector<MLPMiniBatchData<Policy>, kCudaMemDevice> deviceMiniBatchData(kMiniBatchSize);
            Cuda::Vector<Sample, kCudaMemDevice> deviceInputSamples(inputSamples.size());
            Cuda::Vector<Sample, kCudaMemMirrored> deviceOutputSamples(inputSamples.size());
            Cuda::Vector<Sample, kCudaMemDevice> deviceTargetSamples(inputSamples.size());
            Cuda::Object<float> deviceLoss;

            // Determininstically initialise the mini-batch weights and the optimiser 
            m_deviceModelData.Resize(sizeof(MLPModel<Policy, true>));
            MLPModel<Policy, true>& masterModel = *reinterpret_cast<MLPModel<Policy, true>*>(&m_deviceModelData[0]);
            masterModel.Initialise(NormalRandomDistribution(0, 0.5f, std::hash<int>{}(0)));

            // Create and initialise the optimiser
            using Optimiser = SequentialLayers<kWidth, kDepth, true>;
            Cuda::Object<Optimiser> deviceOptimiser;
            deviceOptimiser->Initialise(Zeros());           
            deviceOptimiser.Upload();
            m_deviceModelData.Upload();

            // Upload the samples
            deviceInputSamples <<= inputSamples;
            deviceTargetSamples <<= targetSamples;           

            // Create random indirection buffer
            Permutation sampleIdxs(inputSamples.size());
            sampleIdxs.Randomise();

            // Initialise the kernel data structure
            TrainingKernelData<Policy> kernelData;
            kernelData.mlpModelData = reinterpret_cast<MLPModel<Policy, true>*>(m_deviceModelData.GetDeviceData());
            kernelData.mlpMiniBatchData = deviceMiniBatchData.GetDeviceData();
            kernelData.inputVecs = deviceInputSamples.GetDeviceData();
            kernelData.outputVecs = deviceOutputSamples.GetDeviceData();
            kernelData.targetVecs = deviceTargetSamples.GetDeviceData();
            kernelData.optimiser = deviceOptimiser.GetDeviceData();
            kernelData.sampleIdxs = sampleIdxs.GetDeviceData();
            kernelData.loss = deviceLoss.GetDeviceData();
            kernelData.batchSize = inputSamples.size();

            constexpr int kMaxEpochs = 200;
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
        }

        void MLP::Infer(ReadBatchFunctor readBatch, WriteBatchFunctor writeBatch)
        {
            // Define the model policy
            using Policy = MLPInferencePolicy<kWidth, kDepth, kMiniBatchSize, Activation::LeakyReLU>;
            
            Cuda::Vector<Sample, kCudaMemDevice> deviceSamples(Policy::kMiniBatchSize);
            
            // Initialise the kernel data structure
            InferenceKernelData<Policy> kernelData;
            kernelData.mlpModelData = reinterpret_cast<MLPModel<Policy, true>*>(m_deviceModelData.GetDeviceData());
            
            std::vector<Sample> hostSamples;
            int sampleIdx = 0;
            while (readBatch(hostSamples, sampleIdx) && !hostSamples.empty())
            {
                deviceSamples <<= hostSamples;
                kernelData.inOutVecs = deviceSamples.GetDeviceData();
                kernelData.batchSize = hostSamples.size();

                InferBatch(kernelData);

                hostSamples <<= deviceSamples;
                writeBatch(hostSamples, sampleIdx);
                sampleIdx += hostSamples.size();
            }
        }
    }
}