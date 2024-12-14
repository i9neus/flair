#include "MLP.cuh"
#include "core/utils/HighResTimer.h"
#include "../ContinuousRandomVariable.cuh"
#include "../Permute.cuh"
#include "Training.cuh"
#include "Inference.cuh"
#include "Optimiser.cuh"
#include <fstream>
#include <thread>
#include <chrono>

namespace Flair
{
    namespace NN
    {
        // The size of the mini batch
        static constexpr int kMiniBatchSize = 64;

        using LearningRate = std::ratio<1, 1000>;

        using ActivationFunction = Activation::LeakyReLU; 

        using LossFunction = Loss::L1;
        //using LossFunction = Loss::L2;
        //using LossFunction = Loss::BinaryCrossEntropy;

        using OptimiserFunction = Optimiser::Adam<LearningRate>;
        //using OptimiserFunction = Optimiser::SGD<LearningRate>;

        //using Model = LinearSequential<Linear<36, 50>, Linear<50, 50>, Linear<50, 40>, Linear<40, 27>>;
        using Model = LinearSequential<Linear<36, 36>, Linear<36, 36>, Linear<36, 31>, Linear<31, 27>>;
        //using Model = LinearSequential<Linear<16, 16>, Linear<16, 16>, Linear<16, 16>>;

        using Policy = MLPPolicy<Model, HyperParameters<kMiniBatchSize, ActivationFunction, LossFunction, OptimiserFunction>>;
        
        MLP::MLP()
        {
            Initialise();
        }

        void MLP::Initialise()
        {
        }             

        void MLP::Train(const std::vector<InputSample>& inputSamples, const std::vector<OutputSample>& targetSamples)
        {
            Assert(inputSamples.size() == targetSamples.size());

            constexpr size_t kSharedMemorySafeMargin = 1024;
            const size_t ctxSize = sizeof(TrainingCtx<Policy>);
            cudaDeviceProp prop;
            IsOk(cudaGetDeviceProperties(&prop, 0));

            AssertFmt(ctxSize < prop.sharedMemPerBlock - kSharedMemorySafeMargin, "Model context exceeds capacity of shared memory.");

            printf_red("TrainingCtx: %i bytes\n", ctxSize);

            Cuda::Vector<float> deviceGradData(Policy::Hyper::kMiniBatchSize * Policy::Model::kNumParams);
            Cuda::Vector<InputSample> deviceInputSamples(inputSamples.size());
            Cuda::Vector<OutputSample> deviceTargetSamples(inputSamples.size());
            Cuda::Vector<float> deviceSampleLosses(Policy::Hyper::kMiniBatchSize);
            Cuda::Object<float> deviceMiniBatchLoss;

            // Determininstically initialise the mini-batch weights and the optimiser 
            std::vector<float> hostModelData(Model::kNumParams);
            auto rng = NormalRandomDistribution(0, 1.f, std::hash<int>{}(0));
            //auto rng = Ones();
            Model::Initialise(hostModelData, rng);
            m_deviceModelData <<= hostModelData;

            // Create and initialise the optimiser
            Cuda::Vector<float> deviceOptimiserData(Policy::Model::kNumParams * 2);
            deviceOptimiserData.Fill(0);
            deviceGradData.Fill(0);

            /*std::vector<InputSample> tempInput(inputSamples.size(), InputSample(0));
            std::vector<OutputSample> tempTarget(targetSamples.size(), OutputSample(0));
            for (auto& f : tempInput) { f = inputSamples.front(); }
            for (auto& f : tempTarget) { f = targetSamples.front(); }*/

            // Upload the samples
            deviceInputSamples <<= inputSamples;
            deviceTargetSamples <<= targetSamples;

            // Create random indirection buffer
            Permutation sampleIdxs(inputSamples.size());
            sampleIdxs.Randomise();

            // Initialise the kernel data structure
            TrainingKernelData<Policy> kernelData;
            kernelData.mlpModelData = m_deviceModelData.GetDeviceData();
            kernelData.mlpGradData = deviceGradData.GetDeviceData();
            kernelData.inputSamples = deviceInputSamples.GetDeviceData();
            kernelData.targetSamples = deviceTargetSamples.GetDeviceData();
            kernelData.optimiserData = deviceOptimiserData.GetDeviceData();
            kernelData.sampleIdxs = sampleIdxs.GetDeviceData();
            kernelData.sampleLosses = deviceSampleLosses.GetDeviceData();
            kernelData.miniBatchLoss = deviceMiniBatchLoss.GetDeviceData();
            kernelData.batchSize = inputSamples.size();

            constexpr int kMaxEpochs = 100;
            constexpr int kMaxMiniBatches = std::numeric_limits<int>::max();
            int miniBatchIdx = 0;
            HighResTimer kernelTimer, lossTimer;
            double totalTime = 0;

            std::vector<std::pair<int, float>> epochLoss;
            std::vector<float> miniBatchLoss;

            /*printf("Ref:\n");
            (*targetSamples)[0].Print();*/

            for (int epochIdx = 0; epochIdx < kMaxEpochs && miniBatchIdx < kMaxMiniBatches; ++epochIdx)
            {                                                                
                // Reset the kernel data (loss values, etc.) for the new epoch
                PrepareNewEpoch(kernelData);

                float meanLoss = 0;                
                for (int sampleIdx = 0; sampleIdx < kernelData.batchSize && miniBatchIdx < kMaxMiniBatches; sampleIdx += Policy::Hyper::kMiniBatchSize, ++miniBatchIdx)
                {
                    kernelTimer.Reset();

                    // Estimate the gradients
                    EstimateGradients(kernelData, sampleIdx);

                    // Optimiser step
                    Descend(kernelData);

                    IsOk(cudaDeviceSynchronize());
                    totalTime += kernelTimer.Get();

                    const float loss = deviceMiniBatchLoss.Download();
                    miniBatchLoss.emplace_back(loss);
                    if (miniBatchIdx == 0) { epochLoss.emplace_back(0, loss); }
                    //printf("%i %f\n", miniBatchIdx, loss);
                    meanLoss += loss;
                }

                // Record the loss
                meanLoss /= std::ceil(kernelData.batchSize / float(Policy::Hyper::kMiniBatchSize));
                epochLoss.emplace_back(miniBatchIdx, meanLoss);

                if (lossTimer.Get() > 1. / 3)
                { 
                    printf("Epoch %i: L1 = %.10f\n", epochIdx, meanLoss); 
                    lossTimer.Reset();
                }              
                IsOk(cudaDeviceSynchronize());

                // Shuffle the indirection indices
                sampleIdxs.Shuffle();
                
                /*if (epochIdx == 0 || epochIdx == kMaxEpochs - 1)
                {
                    std::vector<float> gradData;
                    gradData <<= deviceGradData;
                    //gradData <<= m_deviceModelData;
                    std::printf("\n\n\n\n%i\n----------------\n%s\n\n", miniBatchIdx, Model::Format(gradData.data()).c_str());

                    // Print optimisers data
                    gradData <<= deviceOptimiserData;
                    for (auto f : gradData)
                    {
                        std::printf("%.3f ", f);
                    }
                    std::printf("\n");
                }*/

                // Print sample indices
                /*std::vector<int>& idxs = sampleIdxs.GetHostData();
                printf("%i: ", epochIdx);
                for (auto& i : idxs) { printf("%i ", i); }
                printf("\n");*/
            }

            printf_green("Total time: %.2f\n", totalTime);

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
            Cuda::Vector<InputSample> deviceInputSamples(Policy::Hyper::kMiniBatchSize);
            Cuda::Vector<OutputSample> deviceOutputSamples(Policy::Hyper::kMiniBatchSize);
            
            // Initialise the kernel data structure
            InferenceKernelData<Policy> kernelData;
            kernelData.mlpModelData = m_deviceModelData.GetDeviceData();

            HighResTimer timer;
            std::vector<InputSample> hostInputSamples;
            std::vector<OutputSample> hostOutputSamples;
            int sampleIdx = 0;
            while (readBatch(hostInputSamples, sampleIdx) && !hostInputSamples.empty())
            {
                deviceOutputSamples.Resize(hostInputSamples.size());
                deviceInputSamples <<= hostInputSamples; 

                kernelData.batchSize = hostInputSamples.size();
                kernelData.inputSamples = deviceInputSamples.GetDeviceData();
                kernelData.outputSamples = deviceOutputSamples.GetDeviceData();

                InferBatch(kernelData);

                hostOutputSamples <<= deviceOutputSamples;
                writeBatch(hostOutputSamples, sampleIdx);
                sampleIdx += hostInputSamples.size();
            }

            printf("Readback took %f\n", timer.Get());
        }
    }
}