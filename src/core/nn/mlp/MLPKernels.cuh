#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include "../TensorOps.cuh"
#include "../Modules.cuh"
#include "../Activation.cuh"
#include "../Loss.cuh"
#include <ratio>
#include <type_traits>

namespace Flair
{
    namespace NN
    {
        template<int Width, int Depth, int MiniBatchSize, typename ActivationT, typename LossT, typename LearningRateT>
        struct MLPPolicy
        {
            __host__ static void AssertValid()
            {
                static_assert((MiniBatchSize & (MiniBatchSize - 1)) == 0, "MiniBatchSize must be a power of two.");
                //static_assert(std::is_same(LearningRateT, typename std::ratio<decltype(LearningRateT::Num), decltype(LearningRateT::Denom)>::value, "LearningRateT must be of type std::ratio");
                using T = decltype(LearningRateT::num);
            }

            using Activation = ActivationT;
            using Loss = LossT;

            static constexpr float kLearningRate = float(LearningRateT::num) / float(LearningRateT::den);

            enum : int
            {
                kWidth = Width,
                kDepth = Depth,
                kMiniBatchSize = MiniBatchSize
            };
        };

        template<typename PolicyT>
        struct MiniBatchData
        {
            SequentialLayers<PolicyT::kWidth, PolicyT::kDepth>  mlp;
            float                                               loss;
        };
        
        template<typename PolicyT>
        struct KernelData
        {           
            MiniBatchData<PolicyT>*                                 miniBatch = nullptr;
            Tensor1D<PolicyT::kWidth, false>*                       inputVecs = nullptr;
            Tensor1D<PolicyT::kWidth, false>*                       targetVecs = nullptr;
            Tensor1D<PolicyT::kWidth, false>*                       outputVecs = nullptr;
            SequentialLayers<PolicyT::kWidth, PolicyT::kDepth>*     optimiser = nullptr;
            int*                                                    sampleIdxs = nullptr;
            float*                                                  loss = nullptr;
            int                                                     batchSize = 0;
        };

        template<typename PolicyT>
        struct TrainingCtx
        {
            using Policy = PolicyT;
            enum : int { kWidth = PolicyT::kWidth, kDepth = PolicyT::kDepth };
            
            __device__ TrainingCtx() {}
            __device__ TrainingCtx(const TrainingCtx&) = delete;

            SequentialLayers<kWidth, kDepth> mlp;                        // The model weights, biases and gradients
            Tensor1D<kWidth, false>         input;                      // The input sample for this eval
            Tensor1D<kWidth, false>         target;                     // The target sample for this eval
            union
            {
                Tensor1D<kWidth, false>     state;                 // The intermediate state of the activations in the forward pass
                Tensor1D<kWidth, false>     error;                 // The propagated error during the backward pass
            };
            Tensor1D<kWidth, false>         acts[kDepth];               // Cached per-layer activations required during the forward/backward passes
            union
            {
                float                   scratch2D[kWidth][kWidth];    // Scratch memory for accumulating values during tensor multiplication
                float                   scratch1D[kWidth * kWidth];
            };
            float loss;
        };

        template<typename PolicyT>
        struct InferenceCtx
        {
            using Policy = PolicyT;
            enum : int { kWidth = PolicyT::kWidth, kDepth = PolicyT::kDepth };

            __host__ __device__ InferenceCtx() {}

            SequentialLayers<kWidth, kDepth>    mlp;
            Tensor1D<kWidth, false>             state;
            float                               scratch2D[kWidth][kWidth];
        };

        template<typename Ctx>
        __forceinline__ __device__ void CacheActivations(const int, Ctx&) { }

        template<typename PolicyT>
        __forceinline__ __device__ void CacheActivations(const int layerIdx, TrainingCtx<PolicyT>& ctx)
        {
            ctx.acts[layerIdx][kThreadIdx] = ctx.state[kThreadIdx];
        }

        template<typename Ctx>
        __inline__ __device__ void Forward(Ctx& ctx, int maxLayer = 100)
        {
            for (int layerIdx = 0; layerIdx < Ctx::kDepth && layerIdx < maxLayer; ++layerIdx)
            {
                __syncthreads();

                // Multiply the state by the layer weights
                Mul(ctx.mlp.layers[layerIdx].w, ctx.state, ctx.state, ctx.scratch2D);

                if (kThreadIdx < Ctx::kWidth)
                {
                    // Add the bias
                    ctx.state[kThreadIdx] += ctx.mlp.layers[layerIdx].b[kThreadIdx];

                    // Apply leaky ReLU activation, except on the last layer
                    if (layerIdx != Ctx::kDepth - 1)
                    {
                        Ctx::Policy::Activation::F(ctx.state[kThreadIdx]);
                    }

                    // Cache the feed-forward intermediate activations in this layer for use during backprop
                    CacheActivations(layerIdx, ctx);
                }
            }
        }

        template<typename Ctx>
        __inline__ __device__ void Backward(Ctx& ctx)
        {
            // Col -> source neuron. Row -> destination neuron.
            const int rowIdx = kThreadIdx % Ctx::kWidth;
            const int colIdx = kThreadIdx / Ctx::kWidth;

            for (int layerIdx = Ctx::kDepth - 1; layerIdx >= 0; --layerIdx)
            {
                __syncthreads();
                if (kThreadIdx < Ctx::kWidth && layerIdx != Ctx::kDepth - 1)
                {
                    // Derivative of activation at this layer (except last layer)
                    ctx.error[kThreadIdx] *= Ctx::Policy::Activation::dF(ctx.acts[layerIdx][kThreadIdx]);
                }

                // Accumulate derivative of the weights...
                __syncthreads();
                auto& layer = ctx.mlp.layers[layerIdx];
                layer.w.Grad(colIdx, rowIdx) += ctx.error[rowIdx] * ((layerIdx == 0) ? ctx.input[colIdx] : ctx.acts[layerIdx - 1][colIdx]);
                if (kThreadIdx < Ctx::kWidth)
                {
                    layer.b.Grad(kThreadIdx) += ctx.error[kThreadIdx];
                }

                __syncthreads();
                if (layerIdx != 0)
                {
                    // Multiply the error by the transpose of the weight matrix                 
                    MulT(ctx.mlp.layers[layerIdx].w, ctx.error, ctx.error, ctx.scratch2D);
                }
            }
        }

        /**
        *  Computes the loss from the output and broadcast it back to the error tensor in the last layer
        */
        template<typename Ctx>
        __inline__ __device__ float EstimateLoss(Ctx& ctx)
        {
            // L1 loss
            __syncthreads();
            if (kThreadIdx < Ctx::kWidth)
            {
                ctx.scratch1D[kThreadIdx] = Ctx::Policy::Loss::F(ctx.state[kThreadIdx], ctx.target[kThreadIdx]);
            }

            // Reduce
            for (int reduceMask = 2; reduceMask <= Ctx::kWidth; reduceMask <<= 1)
            {
                __syncthreads();
                if (kThreadIdx < Ctx::kWidth && (kThreadIdx & (reduceMask - 1)) == 0)
                {
                    ctx.scratch1D[kThreadIdx] += ctx.scratch1D[kThreadIdx + (reduceMask >> 1)];
                }
            }

            // Average
            __syncthreads();
            if (kThreadIdx == 0)
            {
                ctx.scratch1D[0] /= Ctx::kWidth;
            }

            // Broadcast
            __syncthreads();
            if (kThreadIdx < Ctx::kWidth)
            {
                ctx.error[kThreadIdx] = ctx.scratch1D[0] * Ctx::Policy::Loss::dF(ctx.state[kThreadIdx], ctx.target[kThreadIdx]);
            }

            return ctx.scratch1D[0];
        }

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<typename PolicyT>
        __global__ void EstimateGradientsKernel(KernelData<PolicyT> kernelData, const int miniBatchOffset)
        {
            __shared__ TrainingCtx<PolicyT> ctx;

            // Clear the gradients ready for accumulation
            ctx.mlp.ZeroGrad();

            // If the element of the mini-batch overruns the batch size
            if (miniBatchOffset + kBlockIdx >= kernelData.batchSize) { return; }
            
            // Copy MLP data out of global memory into shared memory. 
            // NOTE: the data are pulled from the first copy in the mini-batch which serves as the master network for gradient descent
            auto& masterMlp = kernelData.miniBatch[0].mlp;
            for (int layerIdx = 0; layerIdx < PolicyT::kDepth; ++layerIdx)
            {
                ctx.mlp.layers[layerIdx].w[kThreadIdx] = masterMlp.layers[layerIdx].w[kThreadIdx];
                if (kThreadIdx < PolicyT::kWidth)
                {
                    ctx.mlp.layers[layerIdx].b[kThreadIdx] = masterMlp.layers[layerIdx].b[kThreadIdx];
                }
            }
            if (kThreadIdx == 0) { ctx.loss = 0; }

            __syncthreads();

            // Progress sample by sample
            //int numSamples = 0;
            //for (int sampleIdx = kBlockIdx; sampleIdx < ctx.batchSize; sampleIdx += PolicyT::kMiniBatchSize, ++numSamples)
            {
                // Copy input/target samples into memory
                if (kThreadIdx < PolicyT::kWidth)
                {
                    const int indirect = kernelData.sampleIdxs[miniBatchOffset + kBlockIdx];
                    ctx.input[kThreadIdx] = kernelData.inputVecs[indirect][kThreadIdx];
                    ctx.target[kThreadIdx] = kernelData.targetVecs[indirect][kThreadIdx];
                    ctx.state[kThreadIdx] = ctx.input[kThreadIdx];
                }

                // Feed forward pass
                Forward(ctx);

                /*__syncthreads();
                if (debug)
                {
                    if (kKernelIdx == 0)
                    {
                        printf("State:\n");
                        ctx.state.Print(false);
                        printf("Target:\n");
                        ctx.target.Print(false);
                    }
                }*/

                // Calculate the loss and error for the last layer
                ctx.loss = EstimateLoss(ctx);

                // Back propagate error and accumulate gradients
                Backward(ctx);
            }
            //numSamples = max(1, numSamples);

            // Calculate the mean of the accumulated gradients and copy them back into global memory                
            __syncthreads();
            auto& batchElement = kernelData.miniBatch[kBlockIdx];
            for (int layerIdx = 0; layerIdx < PolicyT::kDepth; ++layerIdx)
            {
                batchElement.mlp.layers[layerIdx].w.Grad(kThreadIdx) = ctx.mlp.layers[layerIdx].w.Grad(kThreadIdx);// / numSamples;
                if (kThreadIdx < PolicyT::kWidth)
                {
                    batchElement.mlp.layers[layerIdx].b.Grad(kThreadIdx) = ctx.mlp.layers[layerIdx].b.Grad(kThreadIdx);// / numSamples;
                }
            }
            if (kThreadIdx == 0) { batchElement.loss = ctx.loss/* / numSamples*/; };
        }

        template<typename PolicyT>
        __forceinline__ __host__ void EstimateGradients(KernelData<PolicyT> kernelData, const int miniBatchOffset, const int miniBatchSize)
        {
            constexpr int kThreadsPerBlock = PolicyT::kWidth * PolicyT::kWidth;
            AssertFmt(kThreadsPerBlock <= 1024, "Exceeded block limit of 1024 threads");
            EstimateGradientsKernel << < miniBatchSize, kThreadsPerBlock >> > (kernelData, miniBatchOffset);
        }

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<typename PolicyT>
        __global__ void InferKernel(KernelData<PolicyT> kernelData, const int miniBatchOffset)
        {
            __shared__ TrainingCtx<PolicyT> ctx;

            // If the element of the mini-batch overruns the batch size
            if (miniBatchOffset + kBlockIdx >= kernelData.batchSize) { return; }

            // Copy MLP data out of global memory into shared memory. 
            // NOTE: the data are pulled from the first copy in the mini-batch which serves as the master network for gradient descent
            auto& masterMlp = kernelData.miniBatch[0].mlp;
            for (int layerIdx = 0; layerIdx < PolicyT::kDepth; ++layerIdx)
            {
                ctx.mlp.layers[layerIdx].w[kThreadIdx] = masterMlp.layers[layerIdx].w[kThreadIdx];
                if (kThreadIdx < PolicyT::kWidth)
                {
                    ctx.mlp.layers[layerIdx].b[kThreadIdx] = masterMlp.layers[layerIdx].b[kThreadIdx];
                }
            }

            // Copy input/target samples into memory
            if (kThreadIdx < PolicyT::kWidth)
            {
                ctx.state[kThreadIdx] = kernelData.inputVecs[miniBatchOffset + kBlockIdx][kThreadIdx];
            }

            // Feed forward pass
            Forward(ctx); 

            // Calculate the loss and error for the last layer
            //ctx.loss = EstimateLoss(ctx);
           
            __syncthreads();
            if (kThreadIdx < PolicyT::kWidth)
            {
                kernelData.outputVecs[miniBatchOffset + kBlockIdx][kThreadIdx] = ctx.state[kThreadIdx];// -kernelData.targetVecs[miniBatchOffset + kBlockIdx][kThreadIdx];;
            }
        }

        template<typename PolicyT>
        __forceinline__ __host__ void Infer(KernelData<PolicyT> kernelData, const int miniBatchOffset, const int miniBatchSize)
        {
            constexpr int kThreadsPerBlock = PolicyT::kWidth * PolicyT::kWidth;
            AssertFmt(kThreadsPerBlock <= 1024, "Exceeded block limit of 1024 threads");
            InferKernel << < miniBatchSize, kThreadsPerBlock >> > (kernelData, miniBatchOffset);
        }

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<typename PolicyT>
        __global__ void ReduceGradientsKernel(KernelData<PolicyT> kernelData, const int stride, const int miniBatchOffset, const int miniBatchSize)
        {            
            const int otherIdx = kBlockIdx * stride + (stride >> 1);
            auto& thisElement = kernelData.miniBatch[kBlockIdx * stride];

            if (otherIdx < miniBatchSize && miniBatchOffset + otherIdx < kernelData.batchSize)
            {
                const auto& otherElement = kernelData.miniBatch[otherIdx];

                for (int layerIdx = 0; layerIdx < PolicyT::kDepth; ++layerIdx)
                {
                    thisElement.mlp.layers[layerIdx].w.Grad(kThreadIdx) += otherElement.mlp.layers[layerIdx].w.Grad(kThreadIdx);
                    if (kThreadIdx < PolicyT::kWidth)
                    {
                        thisElement.mlp.layers[layerIdx].b.Grad(kThreadIdx) += otherElement.mlp.layers[layerIdx].b.Grad(kThreadIdx);
                    }
                }
                if (kThreadIdx == 0)
                {
                    thisElement.loss += otherElement.loss;
                }
            }

            __syncthreads();

            // On the last reduce, average the accumulated gradients.
            if (stride == miniBatchSize)
            {
                const int N = min(miniBatchSize, kernelData.batchSize - miniBatchOffset);
                for (int layerIdx = 0; layerIdx < PolicyT::kDepth; ++layerIdx)
                {
                    thisElement.mlp.layers[layerIdx].w.Grad(kThreadIdx) /= N;
                    if (kThreadIdx < PolicyT::kWidth)
                    {
                        thisElement.mlp.layers[layerIdx].b.Grad(kThreadIdx) /= N;
                    }
                }
                if (kThreadIdx == 0)
                {
                    thisElement.loss /= N;
                    
                    //thisElement.mlp.layers[2].b.Print(true);
                    //printf("\n");
                }
            }
        }

        template<typename PolicyT>
        __forceinline__ __host__ void ReduceGradients(KernelData<PolicyT> kernelData, const int miniBatchOffset, const int miniBatchSize)
        {
            if (miniBatchSize > 1)
            {
                constexpr int kThreadsPerBlock = PolicyT::kWidth * PolicyT::kWidth;
                AssertFmt(kThreadsPerBlock <= 1024, "Exceeded block limit of 1024 threads");

                for (int stride = 2; stride <= miniBatchSize; stride <<= 1)
                {
                    const int kNumBlocks = miniBatchSize / stride;
                    ReduceGradientsKernel << <kNumBlocks, kThreadsPerBlock >> > (kernelData, stride, miniBatchOffset, miniBatchSize);
                }
            }
        }

        /**
        * Adam SGD optimiser
        * Updates param based on grad and moments, mo1 and mo2
        **/
        template<typename PolicyT>
        __forceinline__ __device__ void Adam(float& param, const float& grad, float& mo1, float& mo2)
        {
            constexpr float kAlpha = PolicyT::kLearningRate;
            constexpr float kBeta1 = 0.9;
            constexpr float kBeta2 = 0.999;
            constexpr float kEpsilon = 1e-8;

            // Compute biased moments
            mo1 = kBeta1 * mo1 + (1 - kBeta1) * grad;
            mo2 = fmaxf(.0f, kBeta2 * mo2 + (1 - kBeta2) * (grad * grad));

            // Update the parameters
            param -= kAlpha * (mo1 / (1 - kBeta1)) / (sqrtf(mo2 / (1 - kBeta2)) + kEpsilon);
        }

        template<typename PolicyT>
        __forceinline__ __device__ void SGD(float& param, const float& grad)
        {
            // Update the parameters
            param -= grad * PolicyT::kLearningRate;
        }

        template<typename PolicyT>
        __global__ void DescendKernel(KernelData<PolicyT> kernelData)
        {
            auto& mlpLayer = kernelData.miniBatch[0].mlp.layers[kBlockIdx];
            auto& adamLayer = kernelData.optimiser->layers[kBlockIdx];

            if (kThreadIdx < PolicyT::kWidth * PolicyT::kWidth)
            {
                // Update the weights
                Adam<PolicyT>(mlpLayer.w[kThreadIdx], mlpLayer.w.Grad(kThreadIdx), adamLayer.w[kThreadIdx], adamLayer.w.Grad(kThreadIdx));
                //SGD<PolicyT>(mlpLayer.w[kThreadIdx], mlpLayer.w.Grad(kThreadIdx));
            }
            else if (kThreadIdx < PolicyT::kWidth * (1 + PolicyT::kWidth))
            {
                // Update the biases
                const int biasIdx = kThreadIdx - PolicyT::kWidth * PolicyT::kWidth;
                Adam<PolicyT>(mlpLayer.b[biasIdx], mlpLayer.b.Grad(biasIdx), adamLayer.b[biasIdx], adamLayer.b.Grad(biasIdx));
                //SGD<PolicyT>(mlpLayer.b[biasIdx], mlpLayer.b.Grad(biasIdx));
            }

            if (kKernelIdx == 0)
            {
                *kernelData.loss = kernelData.miniBatch[0].loss;
            }
        }


        template<typename PolicyT>
        __forceinline__ __host__ void Descend(KernelData<PolicyT> kernelData)
        {
            constexpr int kThreadsPerBlock = PolicyT::kWidth * (PolicyT::kWidth + 1);
            constexpr int kNumBlocks = PolicyT::kDepth;
            AssertFmt(kThreadsPerBlock <= 1024, "Exceeded block limit of 1024 threads");
            DescendKernel << < kNumBlocks, kThreadsPerBlock >> > (kernelData);
        }    

        template<typename PolicyT>
        __global__ void PrepareNewEpochKernel(KernelData<PolicyT> kernelData)
        {
            if (kThreadIdx == 0)
            {
                kernelData.loss = 0;
            }
            kernelData.miniBatch[kThreadIdx].loss = 0;
        }

        template<typename PolicyT>
        __forceinline__ __host__ void PrepareNewEpoch(KernelData<PolicyT> kernelData, const int miniBatchSize)
        {
            AssertFmt(miniBatchSize <= 1024, "Exceeded block limit of 1024 threads");
            PrepareNewEpochKernel << < 1, miniBatchSize >> > (kernelData);
        }
    }
}
