#include "MLP.cuh"
#include "core/utils/HighResTimer.h"
#include "core/io/IOUtils.h"
#include "../ContinuousRandomVariable.cuh"
#include "../Permute.cuh"
#include "Training.cuh"
#include "TrainingCPU.cuh"
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

        static constexpr ComputeDevice kComputeDevice = ComputeDevice::kCUDA;
                                                        //ComputeDevice::kCPU;

        using ActivationFunction = Activation::LeakyReLU; 

        using LossFunction = Loss::L1;

        using LearningRate = std::ratio<1, 100>;
        
        using LRDecay = NullDecaySchedule;
        //using LRDecay = Optimiser::ExponentialDecaySchedule<std::ratio<99, 100>>;

        using OptimiserFunction = Adam<LearningRate, LRDecay>;
        //using OptimiserFunction = Optimiser::SGD<LearningRate, LRDecay>;

        //using Model = LinearSequential<Linear<49, 49>, Linear<49, 45>, Linear<45, 41>, Linear<41, 36>>;
        //using Model = LinearSequential<Linear<35, 35>, Linear<35, 32>, Linear<32, 28>, Linear<28, 25>>;
        //using Model = LinearSequential<Linear<25, 25>, Linear<25, 25>, Linear<25, 25>, Linear<25, 25>>;
        using Model = LinearSequential<Linear<49, 49>, Linear<49, 36>, Linear<36, 25>, Linear<25, 9>>;

        using Evaluator = LinearSequentialEvaluator<kComputeDevice, Model>;

        using Policy = MLPPolicy<kComputeDevice, Model, Evaluator, HyperParameters<kMiniBatchSize, ActivationFunction, LossFunction, OptimiserFunction>>;
        
        MLP::MLP() : 
            m_computeModelData(Policy::kComputeDevice)
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

            Cuda::Vector<float> computeGradData(Policy::kComputeDevice, Policy::Hyper::kMiniBatchSize * Policy::Model::kNumParams, 0.f);
            Cuda::Vector<InputSample> computeInputSamples(Policy::kComputeDevice, inputSamples.size());
            Cuda::Vector<OutputSample> computeOutputSamples(Policy::kComputeDevice, inputSamples.size());
            Cuda::Vector<OutputSample> computeTargetSamples(Policy::kComputeDevice, inputSamples.size());
            Cuda::Vector<float> computeSampleLosses(Policy::kComputeDevice, Policy::Hyper::kMiniBatchSize);
            Cuda::Object<float> computeMiniBatchLoss(Policy::kComputeDevice);

            // Determininstically initialise the mini-batch weights and the optimiser 
            std::vector<float> hostModelData(Model::kNumParams);
            //auto rng = NormalRandomDistribution(0, 1.f, std::hash<int>{}(0));
            auto rng = UniformDistribution(-1, 1, std::hash<int>{}(0));
            //auto rng = Ones();
            Model::Initialise(hostModelData, rng);
            
            // Load external weights
            /*Assert(IO::DeserialiseArray(hostModelData, "C:/projects/probenet/src/experiments/flair/weights.dat") > 0);
            Assert(hostModelData.size() == Model::kNumParams);
            Model::Transpose(hostModelData); */
            //printf_yellow("%s\n\n", Model::Format(hostModelData).c_str());

            m_computeModelData <<= hostModelData;

            // Create and initialise the optimiser
            Cuda::Vector<float> computeOptimiserData(Policy::kComputeDevice, Policy::Model::kNumParams * 2, 0.f);

            /*std::vector<InputSample> tempInput(inputSamples.size(), InputSample(0));
            std::vector<OutputSample> tempTarget(targetSamples.size(), OutputSample(0));
            for (auto& f : tempInput) { f = inputSamples.front(); }
            for (auto& f : tempTarget) { f = targetSamples.front(); }*/

            // Upload the samples
            computeInputSamples <<= inputSamples;
            computeTargetSamples <<= targetSamples;
            computeTargetSamples.Resize(targetSamples.size());

            // Create random indirection buffer
            Permutation sampleIdxs(Policy::kComputeDevice, inputSamples.size());
            sampleIdxs.Randomise();
            //sampleIdxs.Sequential();

            // Initialise the kernel data structure
            TrainingKernelData<Policy> kernelData;
            kernelData.mlpModelData = m_computeModelData.GetComputeData();
            kernelData.mlpGradData = computeGradData.GetComputeData();
            kernelData.inputSamples = computeInputSamples.GetComputeData();
            kernelData.outputSamples = computeOutputSamples.GetComputeData();
            kernelData.targetSamples = computeTargetSamples.GetComputeData();
            kernelData.optimiserData = computeOptimiserData.GetComputeData();
            kernelData.sampleIdxs = sampleIdxs.GetComputeData();
            kernelData.sampleLosses = computeSampleLosses.GetComputeData();
            kernelData.miniBatchLoss = computeMiniBatchLoss.GetComputeData();
            kernelData.batchSize = inputSamples.size();

            constexpr int kMaxEpochs = 50;
            constexpr int kMaxMiniBatches = std::numeric_limits<int>::max();
            int miniBatchIdx = 0;
            HighResTimer kernelTimer, lossTimer;
            double totalTime = 0;
            
            using Trainer = MLPTrainer<Policy::kComputeDevice, Policy>;

            std::vector<std::pair<int, float>> epochLoss;
            std::vector<float> miniBatchLoss;

            /*printf("Ref:\n");
            (*targetSamples)[0].Print();*/
            
            for (int epochIdx = 0; epochIdx < kMaxEpochs && miniBatchIdx < kMaxMiniBatches; ++epochIdx)
            {                                                               
                float meanLoss = 0;                
                for (int sampleIdx = 0; sampleIdx < kernelData.batchSize && miniBatchIdx < kMaxMiniBatches; sampleIdx += Policy::Hyper::kMiniBatchSize, ++miniBatchIdx)
                {
                    kernelTimer.Reset();

                    computeGradData.Fill(0.f);

                    // Reset the kernel data (loss values, etc.) for the new epoch
                    Trainer::PrepareNewEpoch(kernelData);

                    // Estimate the gradients
                    Trainer::EstimateGradients(kernelData, sampleIdx);

                    if (false && miniBatchIdx == 1)
                    {
                        //printf_red("\n---------------------------------------------------------\nEPOCH %i\n\n", epochIdx);

                        hostModelData <<= m_computeModelData;
                        printf_yellow("WEIGHTS:\n%s\n\n", Model::Format(hostModelData).c_str());

                    }

                    // Optimiser step
                    Optimiser<Policy::kComputeDevice, Policy>::Descend(kernelData, epochIdx);

                    IsOk(cudaDeviceSynchronize());
                    totalTime += kernelTimer.Get();

                    // State diagnostics
                    //if (kPrintDebug && (epochIdx == 0 || epochIdx == kMaxEpochs - 1))
                    if(false)
                    {
                        //printf_red("\n---------------------------------------------------------\nEPOCH %i\n\n", epochIdx);

                        std::vector<float> gradData;
                        gradData <<= computeGradData;
                        std::printf("GRADIENTS %i: %s\n\n\n", miniBatchIdx, Model::Format(gradData).c_str());

                        std::vector<OutputSample> outputSamples;
                        outputSamples <<= computeOutputSamples;
                        printf("INPUT:\n%s\n", inputSamples[0].Format(false, false).c_str());
                        printf("OUTPUT:\n%s\n", outputSamples[0].Format(false, false).c_str());
                        printf("TARGET:\n%s\n", targetSamples[0].Format(false, false).c_str());

                        printf_red("\n\n\n");
                    }

                    // Print sample losses for mini-batch
                    /*std::vector<float> hostSampleLosses;
                    hostSampleLosses <<= computeSampleLosses;
                    for (auto& f : hostSampleLosses) { printf("%.10f, ", f); }*/

                    const float loss = computeMiniBatchLoss.Download();
                    //miniBatchLoss.emplace_back(loss);
                    //if (miniBatchIdx == 0) { epochLoss.emplace_back(0, loss); }
                    //printf("   Mini batch %i: %.15f\n", miniBatchIdx, loss);
                    meanLoss += loss;

                    //break;
                }

                // Record the loss
                meanLoss /= std::ceil(kernelData.batchSize / float(Policy::Hyper::kMiniBatchSize));
                epochLoss.emplace_back(miniBatchIdx, meanLoss);

                if (epochIdx == 0 || epochIdx == kMaxEpochs - 1 || lossTimer.Get() > 1. / 3)
                { 
                    printf("Epoch %i: L1 = %.10f\n", epochIdx, meanLoss); 
                    lossTimer.Reset();
                }              
                IsOk(cudaDeviceSynchronize());

                // Shuffle the indirection indices
                sampleIdxs.Shuffle();   

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

            /*std::vector<float> gradData;
            gradData <<= computeGradData;
            std::printf("GRADIENTS: %s\n", Model::Format(gradData).c_str());

            hostModelData <<= m_computeModelData;
            printf_yellow("WEIGHTS:\n%s\n\n", Model::Format(hostModelData).c_str());*/

            // Print optimiser data
            /*std::vector<float> adamData;
            adamData <<= computeOptimiserData;
            printf("ADAM:\n");
            for (auto f : adamData)
            {
                std::printf("%.10e ", f);
            }
            std::printf("\n");*/
        }

        void MLP::Infer(ReadBatchFunctor readBatch, WriteBatchFunctor writeBatch)
        {            
            Cuda::Vector<InputSample> computeInputSamples(kComputeDevice, Policy::Hyper::kMiniBatchSize);
            Cuda::Vector<OutputSample> computeOutputSamples(kComputeDevice, Policy::Hyper::kMiniBatchSize);
            
            // Initialise the kernel data structure
            InferenceKernelData<Policy> kernelData;
            kernelData.mlpModelData = m_computeModelData.GetComputeData();

            HighResTimer timer;
            std::vector<InputSample> hostInputSamples;
            std::vector<OutputSample> hostOutputSamples;
            int sampleIdx = 0;
            while (readBatch(hostInputSamples, sampleIdx) && !hostInputSamples.empty())
            {
                computeOutputSamples.Resize(hostInputSamples.size());
                computeInputSamples <<= hostInputSamples; 

                kernelData.batchSize = hostInputSamples.size();
                kernelData.inputSamples = computeInputSamples.GetComputeData();
                kernelData.outputSamples = computeOutputSamples.GetComputeData();

                MLPInferer<Policy::kComputeDevice, Policy>::InferBatch(kernelData);

                hostOutputSamples <<= computeOutputSamples;
                writeBatch(hostOutputSamples, sampleIdx);
                sampleIdx += hostInputSamples.size();
            }

            printf("Readback took %f\n", timer.Get());
        }
    }
}