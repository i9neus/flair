#pragma once

#include "LinearSequential.cuh"

namespace Flair
{
    namespace NN
    {
        /**
            Runs an inference pass on the input data and stores it in the same array
        **/
        template<int NumThreads, typename Policy>
        __global__ void InferBatchKernel(InferenceKernelData<Policy> kernelData)
        {
            __shared__ InferenceCtx<Policy> ctx;
            using Model = typename Policy::Model;

            // If the element of the mini-batch overruns the batch size
            if (kBlockIdx >= kernelData.batchSize) { return; }

            // Copy MLP data out of global memory into shared memory. 
            for (int paramIdx = kThreadIdx; paramIdx < Model::kNumParams; ++paramIdx)
            {
                ctx.mlpData[paramIdx] = kernelData.mlpModelData[paramIdx];
            }

            ctx.batchSize = kernelData.batchSize;

            // Progress sample by sample
            for (int sampleIdx = kBlockIdx; sampleIdx < ctx.batchSize; sampleIdx += Policy::Hyper::kMiniBatchSize)
            {
                // Copy input/target samples into memory
                __syncthreads();
                if (kThreadIdx < Model::kInputWidth)
                {
                    ctx.state[kThreadIdx] = kernelData.inputSamples[sampleIdx][kThreadIdx];
                }

                // Feed forward pass
                Model::Forward(ctx);

                __syncthreads();
                if (kThreadIdx < Model::kOutputWidth)
                {
                    kernelData.outputSamples[sampleIdx][kThreadIdx] = ctx.state[kThreadIdx];
                }
            }
        }

        template<typename Policy>
        __forceinline__ __host__ void InferBatch(InferenceKernelData<Policy> kernelData)
        {
            constexpr int kNumThreads = Policy::Model::kMaxConcurrency;
            AssertFmt(kNumThreads <= 1024, "Exceeded block limit of 1024 threads");
            InferBatchKernel<kNumThreads> << < Policy::Hyper::kMiniBatchSize, kNumThreads >> > (kernelData);
            IsOk(cudaGetLastError());
        }
    }
}
