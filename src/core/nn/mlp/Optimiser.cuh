#pragma once

#include "Ctx.cuh"

namespace Flair
{
    namespace NN
    {
        struct NullDecaySchedule
        {
            __device__  static constexpr float Decay(const int) { return 1.; }
        };

        template<typename DecayRate>
        struct ExponentialDecaySchedule
        {
            static constexpr float kDecayExponent = float(DecayRate::num) / float(DecayRate::den);

            __device__ static float Decay(const int epochIdx)
            {
                return powf(kDecayExponent, float(epochIdx));
            }
        };

        template<typename LearningRate, typename LRScheduleT = NullDecaySchedule>
        struct AbstractOptimiser
        {
            using LRSchedule = LRScheduleT;
            static constexpr float kLearningRate = float(LearningRate::num) / float(LearningRate::den);
        };

        /**
        * Adam SGD optimiser
        * Updates param based on grad and moments, mo1 and mo2
        **/
        template<typename LearningRate, typename LRScheduleT = NullDecaySchedule>
        struct Adam : public AbstractOptimiser<LearningRate, LRScheduleT>
        {
        private:
            using Base = AbstractOptimiser<LearningRate>;

        public:
            __forceinline__ __host__ __device__ static void Step(float& param, float grad, const int paramIdx, const int epochIdx, float* data)
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
                param -= kAlpha * LRSchedule::Decay(epochIdx) * (mo1 / (1 - kBeta1)) / (sqrtf(mo2 / (1 - kBeta2)) + kEpsilon);
            }
        };

        /**
        * Naive SGD optimiser
        **/
        template<typename LearningRate, typename LRScheduleT = NullDecaySchedule>
        struct SGD : public AbstractOptimiser<LearningRate, LRScheduleT>
        {
        private:
            using Base = AbstractOptimiser<LearningRate>;

        public:
            __forceinline__ __host__ __device__ static void Step(float& param, const float& grad, const int, const int epochIdx, float*)
            {
                // Update the parameters
                param -= grad * Base::kLearningRate * LRSchedule::Decay(epochIdx);
            }
        };       

        template<typename Policy>
        __global__ void DescendKernel(TrainingKernelData<Policy> kernelData, const int epochIdx)
        {
            const int paramIdx = kKernelIdx;
            if (paramIdx < Policy::Model::kNumParams)
            {
                Policy::Hyper::Optimiser::Step(kernelData.mlpModelData[paramIdx], kernelData.mlpGradData[paramIdx], paramIdx, epochIdx, kernelData.optimiserData);
            }
        }

        template<ComputeDevice targetDevice, typename Policy>
        struct Optimiser {};

        template<typename Policy>
        struct Optimiser<ComputeDevice::kCUDA, Policy>
        {
            __host__ static void Descend(TrainingKernelData<Policy> kernelData, const int epochIdx)
            {
                constexpr int kNumThreads = 256;
                constexpr int kNumBlocks = (Policy::Model::kNumParams + (kNumThreads - 1)) / kNumThreads;
                DescendKernel << < kNumBlocks, kNumThreads >> > (kernelData, epochIdx);
            }
        };

        template<typename Policy>
        struct Optimiser<ComputeDevice::kCPU, Policy>
        {
            __host__ static void Descend(TrainingKernelData<Policy> kernelData, const int epochIdx)
            {
                for (int paramIdx = 0; paramIdx < Policy::Model::kNumParams; ++paramIdx)
                {
                    Policy::Hyper::Optimiser::Step(kernelData.mlpModelData[paramIdx], kernelData.mlpGradData[paramIdx], paramIdx, epochIdx, kernelData.optimiserData);
                }
            }
        };
    }
}
