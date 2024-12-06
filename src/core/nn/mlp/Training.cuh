#pragma once

#include "LinearSequential.cuh"
#include "../Loss.cuh"

namespace Flair
{
    namespace NN
    {
        /**
        *  Computes the loss from the output and broadcast it back to the error tensor in the last layer
        */
        template<typename Policy>
        __inline__ __device__ float EstimateLoss(TrainingCtx<Policy>& ctx)
        {
            constexpr int kOutputWidth = Policy::Model::kOutputWidth;
            using LossFunction = typename Policy::Hyper::Loss;
            
            // L1 loss
            __syncthreads();
            if (kThreadIdx < kOutputWidth)
            {
                ctx.scratch.At<kOutputWidth>(kThreadIdx) = LossFunction::F(ctx.state[kThreadIdx], ctx.target[kThreadIdx]);
            }

            // Reduce
            for (int reduceMask = 2; reduceMask <= kOutputWidth; reduceMask <<= 1)
            {
                __syncthreads();
                if (kThreadIdx < kOutputWidth && (kThreadIdx & (reduceMask - 1)) == 0)
                {
                    ctx.scratch.At<kOutputWidth>(kThreadIdx) += ctx.scratch.At<kOutputWidth>(kThreadIdx + (reduceMask >> 1));
                }
            }

            // Average
            __syncthreads();
            if (kThreadIdx == 0)
            {
                ctx.scratch.At<kOutputWidth>(0) /= kOutputWidth;
            }

            // Broadcast
            __syncthreads();
            if (kThreadIdx < kOutputWidth)
            {
                ctx.error[kThreadIdx] = ctx.scratch.At<kOutputWidth>(0) * LossFunction::dF(ctx.state[kThreadIdx], ctx.target[kThreadIdx]);
            }

            return ctx.scratch.At<kOutputWidth>(0);
        }

        template<int N, typename ScratchpadT>
        __inline__ __device__ float NormaliseTensor(Tensor1D<N>& tensor, ScratchpadT& scratch)
        {
            __syncthreads();
            if (kThreadIdx < N)
            {
                scratch.At<N>(kThreadIdx) = fabsf(tensor[kThreadIdx]);
            }

            // Reduce
            for (int reduceMask = 2; reduceMask <= N; reduceMask <<= 1)
            {
                __syncthreads();
                if (kThreadIdx < N && (kThreadIdx & (reduceMask - 1)) == 0)
                {
                    scratch.At<N>(kThreadIdx) = max(scratch.At<N>(kThreadIdx), scratch.At<N>(kThreadIdx + (reduceMask >> 1)));
                }
            }

            scratch.At<N>(0) = fmaxf(1e-3f, scratch.At<N>(0));

            __syncthreads();
            if (kThreadIdx < N)
            {
                tensor[kThreadIdx] /= scratch.At<N>(0);
            }

            return scratch.At<N>(0);
        }

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<int NumThreads, typename Policy>
        __global__ void EstimateGradientsKernel(TrainingKernelData<Policy> kernelData, const int miniBatchOffset)
        {
            __shared__ TrainingCtx<Policy> ctx;            
            using Model = typename Policy::Model;

            // If the element of the mini-batch overruns the batch size
            if (miniBatchOffset + kBlockIdx >= kernelData.batchSize) { return; }
            
            // Copy MLP data out of global memory into shared memory. 
            for (int paramIdx = kThreadIdx; paramIdx < Model::kNumParams; paramIdx += NumThreads)
            {
                ctx.mlpData[paramIdx] = kernelData.mlpModelData[paramIdx];   
            }
            ctx.loss = 0;

            //if (kThreadIdx == 0) printf("%f\n", ctx.mlpData[0]);

            __syncthreads();

            // Copy input/target samples into memory
            if (kThreadIdx < Model::kInputWidth)
            {
                ctx.input[kThreadIdx] = kernelData.inputSamples[kernelData.sampleIdxs[miniBatchOffset + kBlockIdx]][kThreadIdx];
                ctx.state[kThreadIdx] = ctx.input[kThreadIdx];
            }
            if (kThreadIdx < Model::kOutputWidth)
            {
                ctx.target[kThreadIdx] = kernelData.targetSamples[kernelData.sampleIdxs[miniBatchOffset + kBlockIdx]][kThreadIdx];
            }

            // Feed forward pass
            Model::Forward(ctx);        

            // Calculate the loss and error for the last layer
            ctx.loss = EstimateLoss(ctx);

            // Back propagate error and accumulate gradients
            Model::Backward(ctx);

            // Copy the estimated gradients back into global memory                
            __syncthreads();
            for (int paramIdx = kThreadIdx; paramIdx < Model::kNumParams; paramIdx += NumThreads)
            {
                kernelData.mlpGradData[kBlockIdx * Model::kNumParams + paramIdx] = ctx.mlpData[paramIdx];
            }
            kernelData.sampleLosses[kBlockIdx] = ctx.loss;
        }

        template<typename Policy>
        __forceinline__ __host__ void EstimateGradients(TrainingKernelData<Policy> kernelData, const int miniBatchOffset)
        {
            constexpr int kNumThreads = Policy::Model::kConcurrency;
            AssertFmt(kNumThreads <= 1024, "Exceeded block limit of 1024 threads");
            EstimateGradientsKernel<kNumThreads> << < Policy::Hyper::kMiniBatchSize, kNumThreads >> > (kernelData, miniBatchOffset);
        }

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<int NumThreads, typename Policy>
        __global__ void ReduceGradientsKernel(TrainingKernelData<Policy> kernelData, const int stride, const int miniBatchOffset)
        {
            constexpr int kMiniBatchSize = Policy::Hyper::kMiniBatchSize;
            const int kernelIdx = blockIdx.x * NumThreads + threadIdx.x;
            const int destIdx = (kernelIdx / Policy::Model::kNumParams) * stride;
            const int srcIdx = destIdx + (stride >> 1);

            if (destIdx < kMiniBatchSize && srcIdx < kMiniBatchSize && miniBatchOffset + srcIdx < kernelData.batchSize)
            {
                const int paramIdx = kernelIdx % Policy::Model::kNumParams;
                kernelData.mlpGradData[destIdx * Policy::Model::kNumParams + paramIdx] += kernelData.mlpGradData[srcIdx * Policy::Model::kNumParams + paramIdx];

                if (paramIdx == 0)
                {
                    kernelData.sampleLosses[destIdx] += kernelData.sampleLosses[srcIdx];
                }

                // On the last reduce, average the accumulated gradients.
                if (stride == kMiniBatchSize) 
                { 
                    const auto N = min(kMiniBatchSize, kernelData.batchSize - miniBatchOffset);
                    kernelData.mlpGradData[destIdx * Policy::Model::kNumParams + paramIdx] /= N; 

                    if (paramIdx == 0)
                    {
                        *kernelData.miniBatchLoss = kernelData.sampleLosses[0] / N;
                    }
                }              
                 
                //if(paramIdx == 0) printf("%i: %f\n", stride, kernelData.mlpGradData[destIdx * Policy::Model::kNumParams + paramIdx]);
            }
        }

        template<typename Policy>
        __forceinline__ __host__ void ReduceGradients(TrainingKernelData<Policy> kernelData, const int miniBatchOffset)
        {
            constexpr int kMiniBatchSize = Policy::Hyper::kMiniBatchSize;
            if (kMiniBatchSize > 1)
            {
                for (int stride = 2; stride <= kMiniBatchSize; stride <<= 1)
                {
                    constexpr int kNumThreads = 256;
                    const int kNumParams = Policy::Model::kNumParams * kMiniBatchSize / stride;
                    const int kNumBlocks = (kNumParams + kNumThreads - 1) / kNumThreads;
                    
                    ReduceGradientsKernel<kNumThreads> << <kNumBlocks, kNumThreads >> > (kernelData, stride, miniBatchOffset);
                }
            }
        }         

        template<typename Policy>
        __global__ void PrepareNewEpochKernel(TrainingKernelData<Policy> kernelData)
        {
            kernelData.miniBatchLoss = 0;
            kernelData.sampleLosses[kThreadIdx] = 0;
        }

        template<typename Policy>
        __forceinline__ __host__ void PrepareNewEpoch(TrainingKernelData<Policy> kernelData)
        {
            AssertFmt(Policy::Hyper::kMiniBatchSize <= 1024, "Exceeded block limit of 1024 threads");
            PrepareNewEpochKernel << < 1, Policy::Hyper::kMiniBatchSize >> > (kernelData);
        }
    }
}
