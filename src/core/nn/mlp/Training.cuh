#pragma once

#include "LinearSequential.cuh"
#include "../Loss.cuh"
#include "core/utils/cuda/CudaVector.cuh"

namespace Flair
{
    namespace NN
    {
        /**
        *  Computes the loss from the output and broadcast it back to the error tensor in the last layer
        */
        template<typename Policy>
        __inline__ __device__ void EstimateLoss(TrainingCtx<Policy>& ctx)
        {
            constexpr int kOutputWidth = Policy::Model::kOutputWidth;
            using LossFunction = typename Policy::Hyper::Loss;
            
            // L1 loss and its derivative
            if (kThreadIdx < kOutputWidth)
            {
                ctx.scratch.At(kThreadIdx) = LossFunction::F(ctx.state[kThreadIdx], ctx.target[kThreadIdx]);
                ctx.error[kThreadIdx] = LossFunction::dF(ctx.state[kThreadIdx], ctx.target[kThreadIdx]) / kOutputWidth;
            }

            // Reduce
            constexpr int Shift = ((kOutputWidth & (kOutputWidth - 1)) == 0) ? 0 : 1;
            for (int reduceMask = 2; reduceMask <= (kOutputWidth << Shift); reduceMask <<= 1)
            {
                __syncthreads();
                if ((kThreadIdx & (reduceMask - 1)) == 0 && kThreadIdx + (reduceMask >> 1) < kOutputWidth)
                {
                    ctx.scratch.At(kThreadIdx) += ctx.scratch.At(kThreadIdx + (reduceMask >> 1));
                }
            }          

            __syncthreads();
            if (kThreadIdx == 0) { ctx.loss = ctx.scratch.At(0) / kOutputWidth; }
        }

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<typename Policy>
        __global__ void EstimateGradientsKernel(TrainingKernelData<Policy> kernelData, const int miniBatchOffset)//, TrainingCtx<Policy>* ctxData)
        {
            __shared__ TrainingCtx<Policy> ctx;
            //TrainingCtx<Policy>& ctx = ctxData[kBlockIdx];

            using Model = typename Policy::Model;
            using Evaluator = typename Policy::Evaluator;

            // If the element of the mini-batch overruns the batch size
            if (miniBatchOffset + kBlockIdx >= kernelData.batchSize) { return; }
            
            // Copy MLP data out of global memory into shared memory. 
            for (int paramIdx = kThreadIdx; paramIdx < Model::kNumParams; paramIdx += kBlockDim)
            {
                ctx.mlpData[paramIdx] = kernelData.mlpModelData[paramIdx];   
            }

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
            Evaluator::Forward(ctx);

            // Store output from forward pass
            __syncthreads();
            if (kThreadIdx < Model::kOutputWidth)
            {
                kernelData.outputSamples[kernelData.sampleIdxs[miniBatchOffset + kBlockIdx]][kThreadIdx] = ctx.state[kThreadIdx];
            }

            // Calculate the loss and error for the last layer
            EstimateLoss(ctx);

            // Back propagate error and accumulate gradients
            Evaluator::Backward(ctx);

            // Copy the estimated gradients back into global memory                
            __syncthreads();
            for (int paramIdx = kThreadIdx; paramIdx < Model::kNumParams; paramIdx += kBlockDim)
            {
                kernelData.mlpGradData[kBlockIdx * Model::kNumParams + paramIdx] = ctx.mlpData[paramIdx];
            }
            if (kThreadIdx == 0) { kernelData.sampleLosses[kBlockIdx] = ctx.loss; }    
        }

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<typename Policy>
        __global__ void ReduceGradientsKernel(TrainingKernelData<Policy> kernelData, const int stride, const int miniBatchOffset)
        {
            constexpr int kMiniBatchSize = Policy::Hyper::kMiniBatchSize;
            const int destIdx = (kKernelIdx / Policy::Model::kNumParams) * stride;
            const int srcIdx = destIdx + (stride >> 1);

            if (destIdx < kMiniBatchSize && srcIdx < kMiniBatchSize && miniBatchOffset + srcIdx < kernelData.batchSize)
            {
                const int paramIdx = kKernelIdx % Policy::Model::kNumParams;
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

        /**
            Reduces accumulated gradients and loss values over the mini batch and stores them in the 0th layer
        **/
        template<typename Policy>
        __global__ void ReduceLossKernel(TrainingKernelData<Policy> kernelData)
        {
            __shared__ float scratch[Policy::Hyper::kMiniBatchSize];

            // Reduce the loss accross the batch
            scratch[kThreadIdx] = kernelData.sampleLosses[kThreadIdx];
            for (int stride = 2; stride <= Policy::Hyper::kMiniBatchSize; stride <<= 1)
            {
                __syncthreads();
                if (kThreadIdx + (stride >> 1) < kBlockDim && (kThreadIdx & (stride - 1)) == 0)
                {
                    scratch[kThreadIdx] += scratch[kThreadIdx + (stride >> 1)];
                }
            }
            __syncthreads();
            if (kKernelIdx == 0) { *kernelData.miniBatchLoss = scratch[0] / kBlockDim; }
        }

        template<typename Policy>
        __global__ void PrepareNewEpochKernel(TrainingKernelData<Policy> kernelData)
        {
            kernelData.miniBatchLoss = 0;
            kernelData.sampleLosses[kThreadIdx] = 0;
        }

        template<ComputeDevice targetDevice, typename Policy>
        struct MLPTrainer {};

        template<typename Policy>
        struct MLPTrainer<ComputeDevice::kCUDA, Policy>
        {
            __host__ static void EstimateGradients(TrainingKernelData<Policy> kernelData, const int miniBatchOffset)
            {
                //Cuda::Vector<TrainingCtx<Policy>> ctx(ComputeDevice::kCUDA, Policy::Hyper::kMiniBatchSize);
                
                // Estimate the gradients for each element in the mini-batch
                AssertFmt(Policy::Model::kMaxConcurrency <= 1024, "Exceeded block limit of 1024 threads");
                EstimateGradientsKernel << < Policy::Hyper::kMiniBatchSize, Policy::Model::kMaxConcurrency >> > (kernelData, miniBatchOffset);// , ctx.GetComputeData());
                //IsOk(cudaGetLastError());

                const int kMiniBatchSize = std::min(int(Policy::Hyper::kMiniBatchSize), kernelData.batchSize - miniBatchOffset);
                ReduceLossKernel << < 1, kMiniBatchSize >> > (kernelData);

                // Reduce the gradients
                /*constexpr int kMiniBatchSize = Policy::Hyper::kMiniBatchSize;
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
                }*/
            }

            __host__ static void PrepareNewEpoch(TrainingKernelData<Policy> kernelData)
            {
                AssertFmt(Policy::Hyper::kMiniBatchSize <= 1024, "Exceeded block limit of 1024 threads");
                PrepareNewEpochKernel << < 1, Policy::Hyper::kMiniBatchSize >> > (kernelData);
                IsOk(cudaGetLastError());
            }
        };
    }
}
