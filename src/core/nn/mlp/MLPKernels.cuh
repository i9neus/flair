#pragma once

#include "core/utils/cuda/CudaUtils.cuh"
#include "core/utils/ConsoleUtils.h"
#include "../TensorOps.cuh"
#include "../Modules.cuh"

namespace Flair
{
    namespace NN
    {
        template<int Width, int Depth, int MiniBatchSize>
        struct MLPPolicy
        {
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
            int                                                     batchSize = 0;
            float*                                                  loss = nullptr;
        };

        template<typename PolicyT>
        struct TrainingCtx
        {
            enum : int { kWidth = PolicyT::kWidth, kDepth = PolicyT::kDepth };
            
            __host__ __device__ TrainingCtx() {}

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
            int batchSize;
        };

        template<typename PolicyT>
        struct InferenceCtx
        {
            enum : int { kWidth = PolicyT::kWidth, kDepth = PolicyT::Depth };

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
        __inline__ __device__ void Forward(Ctx& ctx)
        {
            __syncthreads();

            for (int layerIdx = 0; layerIdx < Ctx::kDepth; ++layerIdx)
            {
                // Multiply the state by the layer weights
                Mul(ctx.mlp.layers[layerIdx].w, ctx.state, ctx.state, ctx.scratch2D);

                if (kThreadIdx < Ctx::kWidth)
                {
                    // Add the bias
                    ctx.state[kThreadIdx] += ctx.mlp.layers[layerIdx].b[kThreadIdx];

                    // Apply leaky ReLU activation, except on the last layer
                    if (layerIdx != Ctx::kDepth - 1)
                    {
                        if (ctx.state[kThreadIdx] < 0)
                        {
                            ctx.state[kThreadIdx] *= 1e-2f;
                        }
                    }

                    // Cache the feed-forward intermediate activations in this layer for use during backprop
                    CacheActivations(layerIdx, ctx);
                }
            }
        }

        template<typename Ctx>
        __inline__ __device__ void Backward(Ctx& ctx)
        {
            // Row -> source neuron. Col -> destination neuron.1
            const int rowIdx = threadIdx.x / Ctx::kWidth, colIdx = threadIdx.x % Ctx::kWidth;

            for (int layerIdx = Ctx::kDepth - 1; layerIdx >= 0; --layerIdx)
            {
                __syncthreads();
                if (kThreadIdx < Ctx::kWidth && layerIdx != Ctx::kDepth - 1)
                {
                    // Derivative of activation at this layer (except last layer)
                    ctx.error[kThreadIdx] *= (ctx.acts[layerIdx][kThreadIdx] >= 0) ? 1.f : 1e-3f;
                }

                // Accumulate derivative of the weights...
                __syncthreads();
                auto& layer = ctx.mlp.layers[layerIdx];
                layer.w.Grad(colIdx, rowIdx) += ctx.error[colIdx] * ((layerIdx == 0) ? ctx.input[rowIdx] : ctx.acts[layerIdx - 1][rowIdx]);
                if (rowIdx == 0)
                {
                    layer.b.Grad(colIdx) += ctx.error[colIdx];
                }

                if (layerIdx != 0)
                {
                    // Multiply the error by the transpose of the weight matrix 
                    MulT(ctx.mlp.layers[layerIdx].w, ctx.error, ctx.error, ctx.scratch2D);
                }
            }
        }

        /**
        *  Computes the loss from the output and broadcasts it back to the error tensor in the last layer
        */
        template<typename Ctx>
        __inline__ __device__ float Loss(Ctx& ctx)
        {
            // L1 loss
            __syncthreads();
            if (kThreadIdx < Ctx::kWidth)
            {
                ctx.scratch1D[kThreadIdx] = fabsf(ctx.state[kThreadIdx] - ctx.target[kThreadIdx]);
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
            ctx.scratch1D[0] /= Ctx::kWidth;

            // Broadcast
            __syncthreads();
            if (kThreadIdx < Ctx::kWidth)
            {
                ctx.error[kThreadIdx] = ctx.scratch1D[0] * sign(ctx.state[kThreadIdx] - ctx.target[kThreadIdx]);
            }

            return ctx.scratch1D[0];
        }

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<typename PolicyT>
        __global__ void EstimateGradientsKernel(KernelData<PolicyT> kernelData)
        {
            __shared__ TrainingCtx<PolicyT> ctx;

            // Copy MLP data out of global memory into shared memory. 
            // NOTE: the data are pulled from the first copy in the mini-batch which serves as the master network for gradient descent
            auto& masterMlp = kernelData.miniBatch[0].mlp;
            for (int layerIdx = 0; layerIdx < PolicyT::kDepth; ++layerIdx)
            {
                ctx.mlp.layers[layerIdx].w.Grad(kThreadIdx) = masterMlp.layers[layerIdx].w.Grad(kThreadIdx);
                if (kThreadIdx < PolicyT::kWidth)
                {
                    ctx.mlp.layers[layerIdx].b.Grad(kThreadIdx) = masterMlp.layers[layerIdx].b.Grad(kThreadIdx);
                }
            }

            // Clear the gradients ready for accumulation
            ctx.mlp.ZeroGrad();

            if (kThreadIdx == 0)
            {
                ctx.loss = 0;
                ctx.batchSize = kernelData.batchSize;
            }

            __syncthreads();

            // Progress sample by sample
            int numSamples = 0;
            for (int sampleIdx = kBlockIdx; sampleIdx < ctx.batchSize; sampleIdx += PolicyT::kMiniBatchSize, ++numSamples)
            {
                // Copy input/target samples into memory
                if (kThreadIdx == 0)
                {
                    const int indirect = kernelData.sampleIdxs[sampleIdx];
                    ctx.state = ctx.input = kernelData.inputVecs[indirect];
                    ctx.target = kernelData.targetVecs[indirect];
                }

                // Feed forward pass
                Forward(ctx);

                // Calculate the loss and error for the last layer
                ctx.loss += Loss(ctx);

                // Back propagate error and accumulate gradients
                Backward(ctx);
            }

            // Calculate the mean of the accumulated gradients and copy them back into global memory                
            __syncthreads();
            auto& batchMlp = kernelData.miniBatch[kBlockIdx].mlp;
            for (int layerIdx = 0; layerIdx < PolicyT::kDepth; ++layerIdx)
            {
                batchMlp.layers[layerIdx].w.Grad(kThreadIdx) = ctx.mlp.layers[layerIdx].w.Grad(kThreadIdx) / numSamples;
                if (kThreadIdx < PolicyT::kWidth)
                {
                    batchMlp.layers[layerIdx].b.Grad(kThreadIdx) = ctx.mlp.layers[layerIdx].b.Grad(kThreadIdx) / numSamples;
                }
            }
        }

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<typename PolicyT>
        __global__ void ReduceGradientsKernel(KernelData<PolicyT> kernelData, const bool span, const int stride)
        {
            if (kBlockIdx * (stride + 1) >= PolicyT::kMiniBatchSize) { return; }

            // Reduce the MLP gradients
            auto& thisMlp = kernelData.miniBatch[kBlockIdx * stride].mlp;
            const auto& otherMlp = kernelData.miniBatch[kBlockIdx * (stride + 1)].mlp;
            for (int layerIdx = 0; layerIdx < PolicyT::kDepth; ++layerIdx)
            {
                thisMlp.layers[layerIdx].w.Grad(kThreadIdx) += otherMlp.layers[layerIdx].w.Grad(kThreadIdx);
                if (kThreadIdx < PolicyT::kWidth)
                {
                    thisMlp.layers[layerIdx].b.Grad(kThreadIdx) += otherMlp.layers[layerIdx].b.Grad(kThreadIdx);
                }
            }

            // Reduce the accumulated loss
            kernelData.miniBatch[kBlockIdx * stride].loss += kernelData.miniBatch[kBlockIdx * (stride + 1)].loss;

            __syncthreads();

            // On the last reduce, average the accumulated gradients
            if (span == PolicyT::kMiniBatchSize >> 1)
            {
                for (int layerIdx = 0; layerIdx < PolicyT::kDepth; ++layerIdx)
                {
                    thisMlp.layers[layerIdx].w.Grad(kThreadIdx) /= PolicyT::kMiniBatchSize;
                    if (kThreadIdx < PolicyT::kWidth)
                    {
                        thisMlp.layers[layerIdx].b.Grad(kThreadIdx) /= PolicyT::kMiniBatchSize;
                    }
                }
            }
        }

        /**
        * Adam SGD optimiser
        * Updates param based on grad and moments, mo1 and mo2
        **/
        __forceinline__ __device__ void Adam(float& param, const float& grad, float& mo1, float& mo2)
        {
            constexpr float kAlpha = 1e-2;
            constexpr float kBeta1 = 0.9;
            constexpr float kBeta2 = 0.999;
            constexpr float kEpsilon = 1e-8;

            // Compute biased moments
            mo1 = kBeta1 * mo1 + (1 - kBeta1) * grad;
            mo2 = fmaxf(0.f, kBeta2 * mo2 + (1 - kBeta2) * (grad * grad));

            // Update the parameters
            param -= kAlpha * (mo1 / (1 - kBeta1)) / (sqrtf(mo2 / (1 - kBeta2)) + kEpsilon);
        }

        template<typename PolicyT>
        __global__ void DescendKernel(KernelData<PolicyT> kernelData)
        {
            auto& mlpLayer = kernelData.miniBatch[0].mlp.layers[kBlockIdx];
            auto& adamLayer = kernelData.optimiser->layers[kBlockIdx];

            // Update the gradients
            if (kThreadIdx < PolicyT::kWidth * PolicyT::kWidth)
            {
                Adam(mlpLayer.w[kThreadIdx], mlpLayer.w.Grad(kThreadIdx), adamLayer.w[kThreadIdx], adamLayer.w.Grad(kThreadIdx));
            }
            else if (kThreadIdx < PolicyT::kWidth * (1 + PolicyT::kWidth))
            {
                // Update the biases
                const int biasIdx = kThreadIdx - PolicyT::kWidth * PolicyT::kWidth;
                Adam(mlpLayer.b[biasIdx], mlpLayer.b.Grad(biasIdx), adamLayer.b[biasIdx], adamLayer.b.Grad(biasIdx));
            }
        }

        template<typename PolicyT>
        __forceinline__ __host__ void EstimateGradients(KernelData<PolicyT> kernelData)
        {
            constexpr int kThreadsPerBlock = PolicyT::kWidth * PolicyT::kWidth;
            AssertFmt(kThreadsPerBlock <= 1024, "Exceeded block limit of 1024 threads");
            EstimateGradientsKernel << < PolicyT::kMiniBatchSize, kThreadsPerBlock >> > (kernelData);
        }

        template<typename PolicyT>
        __forceinline__ __host__ void ReduceGradients(KernelData<PolicyT> kernelData, const int span, const int stride)
        {
            constexpr int kThreadsPerBlock = PolicyT::kWidth * PolicyT::kWidth;
            AssertFmt(kThreadsPerBlock <= 1024, "Exceeded block limit of 1024 threads");
            ReduceGradientsKernel << <span, kThreadsPerBlock >> > (kernelData, span, stride);
        }

        template<typename PolicyT>
        __forceinline__ __host__ void Descend(KernelData<PolicyT> kernelData)
        {
            constexpr int kThreadsPerBlock = PolicyT::kWidth * (PolicyT::kWidth + 1);
            constexpr int kNumBlocks = PolicyT::kDepth;
            AssertFmt(kThreadsPerBlock <= 1024, "Exceeded block limit of 1024 threads");
            DescendKernel << < kNumBlocks, kThreadsPerBlock >> > (kernelData);
        }
    }
}
