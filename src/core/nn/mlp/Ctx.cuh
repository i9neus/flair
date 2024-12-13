#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include "../Tensor2D.cuh"
#include "../TensorOps.cuh"
#include <ratio>
#include <type_traits>

namespace Flair
{    
    namespace NN
    {         
        template<int MiniBatchSize, typename ActivationT, typename LossT, typename OptimiserT>
        struct HyperParameters
        {
            using Activation = ActivationT;
            using Loss = LossT;
            using Optimiser = OptimiserT;

            enum : int 
            { 
                kMiniBatchSize = MiniBatchSize 
            };

            __host__ static void AssertValid()
            {
                static_assert((kMiniBatchSize & (kMiniBatchSize - 1)) == 0, "MiniBatchSize must be a power of two.");
            }
        };

        template<typename ModelT, typename HyperT>
        struct MLPPolicy
        {
            using Hyper = HyperT;
            using Model = ModelT;
        }; 
        
        template<typename Policy>
        struct TrainingKernelData
        {           
            float*                                  mlpModelData = nullptr;
            float*                                  mlpGradData = nullptr;
            Tensor1D<Policy::Model::kInputWidth>*   inputSamples = nullptr;
            Tensor1D<Policy::Model::kOutputWidth>*  targetSamples = nullptr;
            float*                                  optimiserData = nullptr;
            int*                                    sampleIdxs = nullptr;
            float*                                  sampleLosses = nullptr;
            float*                                  miniBatchLoss = nullptr;
            int                                     batchSize = 0;
        };

        template<typename Policy>
        struct InferenceKernelData
        {           
            float*                                  mlpModelData = nullptr;
            Tensor1D<Policy::Model::kInputWidth>*   inputSamples = nullptr;
            Tensor1D<Policy::Model::kOutputWidth>*  outputSamples = nullptr;
            int                                     batchSize = 0;
        };

        template<typename PolicyT>
        struct TrainingCtx
        {
            using Policy = PolicyT;
            
            __device__ TrainingCtx() {}
            __device__ TrainingCtx(const TrainingCtx&) = delete;

            float                                   mlpData[Policy::Model::kNumParams]; // The model weights and biases
            Tensor1D<Policy::Model::kInputWidth>    input;                          // The input sample for this eval
            Tensor1D<Policy::Model::kOutputWidth>   target;                         // The target sample for this eval
            Tensor1D<Policy::Model::kMaxWidth>      state;                          // The intermediate state of the activations in the forward/backward pass
            Tensor1D<Policy::Model::kMaxWidth>      error;                          // The propagated error during the backward pass
            Tensor1D<Policy::Model::kMaxWidth>      acts[Policy::Model::kDepth];    // Cached per-layer activations required during the forward/backward passes
            float                                   loss;   

            Scratchpad<float, Policy::Model::kMaxConcurrency> scratch;               // Scratch memory for accumulating values during tensor multiplication
        };

        template<typename PolicyT>
        struct InferenceCtx
        {
            using Policy = PolicyT;

            __host__ __device__ InferenceCtx() {}

            float                                           mlpData[Policy::Model::kNumParams];
            Tensor1D<Policy::Model::kMaxWidth, false>       state, error;
            Scratchpad<float, Policy::Model::kMaxConcurrency>  scratch;
            int                                             batchSize;
        };      
    }
}
