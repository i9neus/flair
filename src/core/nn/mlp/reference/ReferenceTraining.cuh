#pragma once

#include "../Training.cuh"

namespace Flair
{
    namespace NN
    {
        template<typename Policy>
        class Trainer<kNNEngineCPU, Policy>
        {
            __host__ static void EstimateGradients(TrainingKernelData<Policy> kernelData, const int miniBatchOffset)
            {
                // Estimate the gradients for each element in the mini-batch
                AssertFmt(Policy::Model::kMaxConcurrency <= 1024, "Exceeded block limit of 1024 threads");
                EstimateGradientsKernel << < Policy::Hyper::kMiniBatchSize, Policy::Model::kMaxConcurrency >> > (kernelData, miniBatchOffset);
                IsOk(cudaGetLastError());

                // Reduce the gradients
                constexpr int kMiniBatchSize = Policy::Hyper::kMiniBatchSize;
                if (kMiniBatchSize > 1)
                {
                    for (int stride = 2; stride <= kMiniBatchSize; stride <<= 1)
                    {
                        constexpr int kNumThreads = 256;
                        const int kNumParams = Policy::Model::kNumParams * kMiniBatchSize / stride;
                        const int kNumBlocks = (kNumParams + kNumThreads - 1) / kNumThreads;

                        ReduceGradientsKernel << <kNumBlocks, kNumThreads >> > (kernelData, stride, miniBatchOffset);
                        IsOk(cudaGetLastError());
                    }
                }
            }

            static __host__ void PrepareNewEpoch(TrainingKernelData<Policy> kernelData)
            {
                kernelData.miniBatchLoss = 0;
                for (int i = 0; i < Policy::Hyper::kMiniBatchSize; ++i)
                {
                    kernelData.sampleLosses[i] = 0;
                }
            }
        };
    }
}