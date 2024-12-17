#pragma once

#include "Training.cuh"
#include "LinearSequentialCPU.cuh"

namespace Flair
{
    namespace NN
    {        
        template<typename Policy>
        struct MLPTrainer<ComputeDevice::kCPU, Policy>
        {
        private:
            __host__ static float EstimateLoss(TrainingCtx<Policy>& ctx)
            {
                float sumLoss = 0;
                for (int m = 0; m < Policy::Model::kOutputWidth; ++m)
                {
                    ctx.error[m] = Policy::Hyper::Loss::dF(ctx.state[m], ctx.target[m]) / Policy::Model::kOutputWidth;
                }

                return MapReduceSum<Policy::Model::kOutputWidth>([&](int m) -> float { return Policy::Hyper::Loss::F(ctx.state[m], ctx.target[m]); }) / Policy::Model::kOutputWidth;
            }

            __host__ static void EstimateGradientsImpl(TrainingCtx<Policy>& ctx, TrainingKernelData<Policy> kernelData, const int sampleIdx, const int miniBatchOffset)
            {
                using Model = typename Policy::Model;
                using Evaluator = typename Policy::Evaluator;

                // Copy the model weights and input/target samples into the training context
                std::memcpy(ctx.mlpData, kernelData.mlpModelData, Policy::Model::kNumParams * sizeof(float));
                ctx.input = kernelData.inputSamples[kernelData.sampleIdxs[miniBatchOffset + sampleIdx]];
                ctx.state = ctx.input;
                ctx.target = kernelData.targetSamples[kernelData.sampleIdxs[miniBatchOffset + sampleIdx]];                

                // Feed forward pass
                Evaluator::Forward(ctx);

                // Store output from forward pass
                for (int m = 0; m < Model::kOutputWidth; ++m)
                {
                    kernelData.outputSamples[kernelData.sampleIdxs[miniBatchOffset + sampleIdx]][m] = ctx.state[m];
                }

                // Calculate the loss and error for the last layer
                ctx.loss = EstimateLoss(ctx);

                // Back propagate error and accumulate gradients
                Evaluator::Backward(ctx);

                // Copy the estimated gradients back into global memory                
                for (int paramIdx = 0; paramIdx < Model::kNumParams; ++paramIdx)
                {
                    kernelData.mlpGradData[paramIdx] += ctx.mlpData[paramIdx];
                }
                kernelData.sampleLosses[sampleIdx] = ctx.loss;               
                *kernelData.miniBatchLoss += ctx.loss;
            }

        public:
            __host__ static void EstimateGradients(TrainingKernelData<Policy> kernelData, const int miniBatchOffset)
            {
                using Model = typename Policy::Model;
                
                TrainingCtx<Policy> ctx;
                std::memset(kernelData.mlpGradData, 0, Policy::Model::kNumParams * sizeof(float)); 
                *kernelData.miniBatchLoss = 0;
                
                // Estimate the gradients accross the batch
                const int miniBatchSize = std::min(int(Policy::Hyper::kMiniBatchSize), kernelData.batchSize - miniBatchOffset);
                for (int sampleIdx = 0; sampleIdx < miniBatchSize; ++sampleIdx)
                {
                    EstimateGradientsImpl(ctx, kernelData, sampleIdx, miniBatchOffset);
                }

                // Normalise the estimated gradients
                for (int paramIdx = 0; paramIdx < Model::kNumParams; ++paramIdx)
                {
                    kernelData.mlpGradData[paramIdx] /= miniBatchSize;
                }
                *kernelData.miniBatchLoss /= miniBatchSize;
            }

            __host__ static void PrepareNewEpoch(TrainingKernelData<Policy> kernelData)
            {
                kernelData.miniBatchLoss = 0;
                for (int sampleIdx = 0; sampleIdx < Policy::Hyper::kMiniBatchSize; ++sampleIdx)
                {
                    kernelData.sampleLosses[sampleIdx] = 0;
                }
            }
        };
    }
}
