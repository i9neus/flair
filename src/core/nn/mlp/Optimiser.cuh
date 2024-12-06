#pragma once

#include "Ctx.cuh"

namespace Flair
{
    namespace NN
    {
        namespace Optimiser
        {
            template<typename LearningRate>
            struct AbstractOptimiser
            {
                static constexpr float kLearningRate = float(LearningRate::num) / float(LearningRate::den);
            };
            
            /**
            * Adam SGD optimiser
            * Updates param based on grad and moments, mo1 and mo2
            **/
            template<typename LearningRate>
            struct Adam : public AbstractOptimiser<LearningRate>
            {   
            private:
                using Base = AbstractOptimiser<LearningRate>;

            public:
                __forceinline__ __device__ static void Step(float& param, const float& grad, const int paramIdx, float* data)
                {                    
                    constexpr float kAlpha = Base::kLearningRate;
                    constexpr float kBeta1 = 0.9;
                    constexpr float kBeta2 = 0.999;
                    constexpr float kEpsilon = 1e-8;

                    float& mo1 = data[paramIdx << 1];
                    float& mo2 = data[(paramIdx << 1) + 1];

                    // Compute biased moments
                    mo1 = kBeta1 * mo1 + (1 - kBeta1) * grad;
                    mo2 = fmaxf(0.f, kBeta2 * mo2 + (1 - kBeta2) * (grad * grad));

                    // Update the parameters
                    param -= kAlpha * (mo1 / (1 - kBeta1)) / (sqrtf(mo2 / (1 - kBeta2)) + kEpsilon);
                }
            };

            /**
            * Naive SGD optimiser
            **/
            template<typename LearningRate>
            struct SGD : public AbstractOptimiser<LearningRate>
            {
            private:
                using Base = AbstractOptimiser<LearningRate>;

            public:
                __forceinline__ __device__ static void Step(float& param, const float& grad, const int, float*)
                {
                    // Update the parameters
                    param -= grad * Base::kLearningRate;
                }
            };
        }    

        template<int NumThreads, typename Policy>
        __global__ void DescendKernel(TrainingKernelData<Policy> kernelData)
        {
            const int paramIdx = blockIdx.x * NumThreads + threadIdx.x;
            if (paramIdx < Policy::Model::kNumParams)
            {
                Policy::Hyper::Optimiser::Step(kernelData.mlpModelData[paramIdx], kernelData.mlpGradData[paramIdx], paramIdx, kernelData.optimiserData);
            }
        }

        template<typename Policy>
        __forceinline__ __host__ void Descend(TrainingKernelData<Policy> kernelData)
        {
            constexpr int kNumThreads = 256;
            constexpr int kNumBlocks = (Policy::Model::kNumParams + (kNumThreads - 1)) / kNumThreads;
            DescendKernel<kNumThreads> << < kNumBlocks, kNumThreads >> > (kernelData);
        }    
    }
}
