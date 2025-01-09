#pragma once

#include "core/utils/cuda/CudaObject.cuh"
#include "core/utils/cuda/CudaVector.cuh"
#include "core/math/MathUtils.h"
#include "../TensorOps.cuh"
#include "MLP.cuh"
#include "core/utils/HighResTimer.h"
#include "core/io/IOUtils.h"
#include "../Permute.cuh"
#include "Training.cuh"
#include "TrainingCPU.cuh"
#include "Inference.cuh"
#include "Optimiser.cuh"
#include "../Activation.cuh"

#include <fstream>
#include <thread>
#include <chrono>
#include <memory>
#include <functional>
#include <vector>

namespace Flair
{
    namespace NN
    {
        template<typename Model, typename ModelInitialiser, typename ActivationFunction, typename LossFunction, typename OptimiserFunction, ComputeDevice TargetDevice = ComputeDevice::kCUDA>
        class MLP
        {
        public:
            enum : int
            {
                kMiniBatchSize = 128
            };

            using InputSample = Tensor1D<Model::kInputWidth, false>;
            using OutputSample = Tensor1D<Model::kOutputWidth, false>;
            using Evaluator = LinearSequentialEvaluator<TargetDevice, Model>;
            using Policy = MLPPolicy<TargetDevice, Model, Evaluator, HyperParameters<kMiniBatchSize, ActivationFunction, LossFunction, OptimiserFunction>>;

            using ReadBatchFunctor = std::function<bool(std::vector<InputSample>&, const int)>;
            using WriteBatchFunctor = std::function<void(const std::vector<OutputSample>&, const int)>;

        private:
            Cuda::Vector<float>  m_computeModelData;

        public:
            MLP() : m_computeModelData(TargetDevice) {}

            void Train(const std::vector<InputSample>& inputSamples, const std::vector<OutputSample>& targetSamples, const int numEpochs)
            {
                Assert(inputSamples.size() == targetSamples.size());

                constexpr size_t kSharedMemorySafeMargin = 1024;
                const size_t ctxSize = sizeof(TrainingCtx<Policy>);
                cudaDeviceProp prop;
                IsOk(cudaGetDeviceProperties(&prop, 0));

                AssertFmt(ctxSize < prop.sharedMemPerBlock - kSharedMemorySafeMargin, "Model context exceeds capacity of shared memory.");

                printf_red("TrainingCtx: %i bytes\n", ctxSize);

                Cuda::Vector<float> computeGradData(TargetDevice, Policy::Hyper::kMiniBatchSize * Policy::Model::kNumParams, 0.f);
                Cuda::Vector<InputSample> computeInputSamples(TargetDevice, inputSamples.size());
                Cuda::Vector<OutputSample> computeOutputSamples(TargetDevice, inputSamples.size());
                Cuda::Vector<OutputSample> computeTargetSamples(TargetDevice, inputSamples.size());
                Cuda::Vector<float> computeSampleLosses(TargetDevice, Policy::Hyper::kMiniBatchSize);
                Cuda::Object<float> computeMiniBatchLoss(TargetDevice);

                // Determininstically initialise the mini-batch weights and the optimiser 
                std::vector<float> hostModelData(Model::kNumParams);
                auto rng = ModelInitialiser();
                Model::Initialise(hostModelData, rng);

                // Load external weights
                /*Assert(IO::DeserialiseArray(hostModelData, "C:/projects/probenet/src/experiments/flair/weights.dat") > 0);
                Assert(hostModelData.size() == Model::kNumParams);
                Model::Transpose(hostModelData); */
                //printf_yellow("%s\n\n", Model::Format(hostModelData).c_str());

                m_computeModelData <<= hostModelData;

                // Create and initialise the optimiser
                Cuda::Vector<float> computeOptimiserData(TargetDevice, Policy::Model::kNumParams * 2, 0.f);

                /*std::vector<InputSample> tempInput(inputSamples.size(), InputSample(0));
                std::vector<OutputSample> tempTarget(targetSamples.size(), OutputSample(0));
                for (auto& f : tempInput) { f = inputSamples.front(); }
                for (auto& f : tempTarget) { f = targetSamples.front(); }*/

                // Upload the samples
                computeInputSamples <<= inputSamples;
                computeTargetSamples <<= targetSamples;
                computeTargetSamples.Resize(targetSamples.size());

                // Create random indirection buffer
                Permutation sampleIdxs(TargetDevice, inputSamples.size());
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

                constexpr int kMaxMiniBatches = std::numeric_limits<int>::max();
                int miniBatchIdx = 0;
                HighResTimer kernelTimer, lossTimer;
                double totalTime = 0;

                using Trainer = MLPTrainer<TargetDevice, Policy>;

                std::vector<std::pair<int, float>> epochLoss;
                std::vector<float> miniBatchLoss;

                /*printf("Ref:\n");
                (*targetSamples)[0].Print();*/

                for (int epochIdx = 0; epochIdx < numEpochs && miniBatchIdx < kMaxMiniBatches; ++epochIdx)
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
                        Optimiser<TargetDevice, Policy>::Descend(kernelData, epochIdx, sampleIdx);

                        IsOk(cudaDeviceSynchronize());
                        totalTime += kernelTimer.Get();

                        // State diagnostics
                        //if (kPrintDebug && (epochIdx == 0 || epochIdx == numEpochs - 1))
                        if (false)
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
                        miniBatchLoss.emplace_back(loss);
                        //if (miniBatchIdx == 0) { epochLoss.emplace_back(0, loss); }
                        //printf("   Mini batch %i: %.15f\n", miniBatchIdx, loss);
                        meanLoss += loss;

                        //break;
                    }

                    // Record the loss
                    meanLoss /= std::ceil(kernelData.batchSize / float(Policy::Hyper::kMiniBatchSize));
                    epochLoss.emplace_back(miniBatchIdx, meanLoss);

                    if (epochIdx == 0 || epochIdx == numEpochs - 1 || lossTimer.Get() > 1. / 3)
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
                for (int i = 0; i < miniBatchLoss.size(); ++i)
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

            void Infer(ReadBatchFunctor readBatch, WriteBatchFunctor writeBatch)
            {
                Cuda::Vector<InputSample> computeInputSamples(TargetDevice, Policy::Hyper::kMiniBatchSize);
                Cuda::Vector<OutputSample> computeOutputSamples(TargetDevice, Policy::Hyper::kMiniBatchSize);

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

                    MLPInferer<TargetDevice, Policy>::InferBatch(kernelData);

                    hostOutputSamples <<= computeOutputSamples;
                    writeBatch(hostOutputSamples, sampleIdx);
                    sampleIdx += hostInputSamples.size();
                }

                printf("Readback took %f\n", timer.Get());
            }
        };
    }  
}