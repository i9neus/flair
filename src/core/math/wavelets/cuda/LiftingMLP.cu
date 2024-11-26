#include "LiftingMLP.cuh"
#include "tests/cuda/TensorTests.cuh"
#include "core/utils/HighResTimer.h"
#include "core/utils/cuda/TensorOps.cuh"

namespace Flair
{      
    struct KernelData
    {
        NN::Model* mlp = nullptr;
        NN::Sample* inputVec = nullptr;
        NN::Sample* targetVec = nullptr;
        NN::Sample* outputVec = nullptr;
        NN::Optimiser* optimiser = nullptr;
        int* sampleIdxs = nullptr;
        int batchSize = 0;
        float* loss = nullptr;
    };

    struct TrainingCtx
    {   
        __host__ __device__ TrainingCtx() {}
        
        NN::Model                mlp;                        // The model weights, biases and gradients
        Tensor1D<NN::kWidth>     input;                      // The input sample for this eval
        Tensor1D<NN::kWidth>     target;                     // The target sample for this eval
        union
        {
            Tensor1D<NN::kWidth>     state;                 // The intermediate state of the activations in the forward pass
            Tensor1D<NN::kWidth>     error;                 // The propagated error during the backward pass
        };
        Tensor1D<NN::kWidth>    acts[NN::kDepth];               // Cached per-layer activations required during the forward/backward passes
        union
        {
            float                   scratch2D[NN::kWidth][NN::kWidth];    // Scratch memory for accumulating values during tensor multiplication
            float                   scratch1D[NN::kWidth * NN::kWidth];   
        };
        float loss;
        int batchSize;
    };

    struct InferenceCtx
    {
        NN::Model               mlp;                       
        Tensor1D<NN::kWidth>    state;                 
        float                   scratch2D[NN::kWidth][NN::kWidth];    
    };

    template<typename CtxType>
    __forceinline__ __device__ void CacheActivations(const int, CtxType&) { }

    template<>
    __forceinline__ __device__ void CacheActivations(const int layerIdx, TrainingCtx& ctx)
    {
        ctx.acts[layerIdx][kThreadIdx] = ctx.state[kThreadIdx];
    }

    template<typename CtxType>
    __inline__ __device__ void Forward(CtxType& ctx)
    {
        __syncthreads();
        
        for (int layerIdx = 0; layerIdx < NN::kDepth; ++layerIdx)
        {
            // Multiply the state by the layer weights
            Mul(ctx.mlp.layers[layerIdx].w, ctx.state, ctx.state, ctx.scratch2D);

            if (kThreadIdx < NN::kWidth)
            {
                // Add the bias
                ctx.state[kThreadIdx] += ctx.mlp.layers[layerIdx].b[kThreadIdx];

                // Apply leaky ReLU activation, except on the last layer
                if (layerIdx != NN::kDepth - 1)
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

    __inline__ __device__ void Backward(TrainingCtx& ctx)
    {
        // Row -> source neuron. Col -> destination neuron.1
        const int rowIdx = threadIdx.x / NN::kWidth, colIdx = threadIdx.x % NN::kWidth;

        for (int layerIdx = NN::kDepth - 1; layerIdx >= 0; --layerIdx)
        {
            __syncthreads();
            if (kThreadIdx < NN::kWidth && layerIdx != NN::kDepth - 1)
            {
                // Derivative of activation at this layer (except last layer)
                ctx.error[kThreadIdx] *= (ctx.acts[layerIdx][kThreadIdx] >= 0) ? 1.f : 1e-3f;
            }

            // Accumulate derivative of the weights...
            __syncthreads();
            auto& layer = ctx.mlp.layers[layerIdx];
            layer.w.grad[colIdx][rowIdx] += ctx.error[colIdx] * ((layerIdx == 0) ? ctx.input[rowIdx] : ctx.acts[layerIdx - 1][rowIdx]);
            if (rowIdx == 0)
            {
                layer.b.grad[colIdx] += ctx.error[colIdx];
            }

            if (layerIdx != 0)
            {
                // Multiply the error by the transpose of the weight matrix 
                MulT(ctx.mlp.layers[layerIdx].w, ctx.error, ctx.error, ctx.scratch2D);
            }
        }
    }

    __inline__ __device__ float L1(TrainingCtx& ctx)
    {
        // L1 loss
        __syncthreads();
        if (kThreadIdx < NN::kWidth)
        {
            ctx.scratch1D[kThreadIdx] = fabsf(ctx.state[kThreadIdx] - ctx.target[kThreadIdx]);
        }

        // Reduce
        for (int reduceMask = 2; reduceMask <= NN::kWidth; reduceMask <<= 1)
        {
            __syncthreads();
            if (kThreadIdx < NN::kWidth && (kThreadIdx & (reduceMask - 1)) == 0)
            {
                ctx.scratch1D[kThreadIdx] += ctx.scratch1D[kThreadIdx + (reduceMask >> 1)];
            }
        }

        // Average
        __syncthreads();
        ctx.scratch1D[0] /= NN::kWidth;

        // Broadcast
        __syncthreads();
        if (kThreadIdx < NN::kWidth) 
        { 
            ctx.error[kThreadIdx] = ctx.scratch1D[0] * sign(ctx.state[kThreadIdx] - ctx.target[kThreadIdx]);
        }

        return ctx.scratch1D[0];
    }

    __global__ void EstimateGradients(KernelData kernelData)
    {
        __shared__ TrainingCtx ctx;

        // Copy MLP data out of global memory into shared memory. 
        // NOTE: the data are pulled from the first copy in the mini-batch which serves as the master network for gradient descent
        for (int layerIdx = 0; layerIdx < NN::kDepth; ++layerIdx)
        {
            ctx.mlp.layers[layerIdx].w.rawGrad[kThreadIdx] = kernelData.mlp[0].layers[layerIdx].w.rawGrad[kThreadIdx];
            if (kThreadIdx < NN::kWidth)
            {
                ctx.mlp.layers[layerIdx].b.grad[kThreadIdx] = kernelData.mlp[0].layers[layerIdx].b.grad[kThreadIdx];
            }
        }

        ctx.mlp.ZeroGrad();

        if (kThreadIdx == 0)
        {
            ctx.loss = 0;
            ctx.batchSize = kernelData.batchSize;
        }

        __syncthreads();

        int numSamples = 0;
        for (int sampleIdx = kBlockIdx; sampleIdx < ctx.batchSize; sampleIdx += NN::kMiniBatchSize, ++numSamples)
        {
            // Copy input/target samples into memory
            if (kThreadIdx == 0)
            {
                const int indirect = kernelData.sampleIdxs[sampleIdx];
                ctx.state = ctx.input = kernelData.inputVec[indirect];
                ctx.target = kernelData.targetVec[indirect];
            }
            
            // Feed forward pass
            Forward(ctx);

            // Calculate the loss and error for the last layer
            ctx.loss += L1(ctx);

            // Back propagate error and accumulate gradients
            Backward(ctx);           
        }

        // Copy the mean of the accumulated gradients back into global memory                
        __syncthreads();
        for (int layerIdx = 0; layerIdx < NN::kDepth; ++layerIdx)
        {
            kernelData.mlp[kBlockIdx].layers[layerIdx].w.rawGrad[kThreadIdx] = ctx.mlp.layers[layerIdx].w.rawGrad[kThreadIdx] / numSamples;
            if (kThreadIdx < NN::kWidth)
            {
                kernelData.mlp[kBlockIdx].layers[layerIdx].b.grad[kThreadIdx] = ctx.mlp.layers[layerIdx].b.grad[kThreadIdx] / numSamples;
            }
        }

        //if(kKernelIdx 
        //kernelData.loss[kBlockIdx]
    }

    __global__ void ReduceGradients(KernelData kernelData, const bool span, const int stride)
    {
        if (kBlockIdx * (stride + 1) >= NN::kMiniBatchSize) { return; }
        
        // Reduce the MLP gradients
        auto& thisNet = kernelData.mlp[kBlockIdx * stride];
        const auto& otherNet = kernelData.mlp[kBlockIdx * (stride + 1)];
        for (int layerIdx = 0; layerIdx < NN::kDepth; ++layerIdx)
        {
            thisNet.layers[layerIdx].w.rawGrad[kThreadIdx] += otherNet.layers[layerIdx].w.rawGrad[kThreadIdx];
            if (kThreadIdx < NN::kWidth)
            {
                thisNet.layers[layerIdx].b.grad[kThreadIdx] += otherNet.layers[layerIdx].b.grad[kThreadIdx];
            }
        }

        // Reduce the accumulated loss
        kernelData.loss[kBlockIdx * stride] += kernelData.loss[kBlockIdx * (stride + 1)];

        __syncthreads();            

        // On the last reduce, average the accumulated gradients
        if (span == NN::kMiniBatchSize >> 1)
        {
            for (int layerIdx = 0; layerIdx < NN::kDepth; ++layerIdx)
            {
                thisNet.layers[layerIdx].w.rawGrad[kThreadIdx] /= NN::kMiniBatchSize;
                if (kThreadIdx < NN::kWidth)
                {
                    thisNet.layers[layerIdx].b.grad[kThreadIdx] /= NN::kMiniBatchSize;
                }
            }
        }
    }

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

    __global__ void Descend(KernelData kernelData)
    {
        auto& mlpLayer = kernelData.mlp[0].layers[kBlockIdx];
        auto& adamLayer = kernelData.optimiser->layers[kBlockIdx];
        
        // Update the gradients
        if (kThreadIdx < NN::kWidth * NN::kWidth)
        {
            Adam(mlpLayer.w[kThreadIdx], mlpLayer.w.rawGrad[kThreadIdx], adamLayer.w[kThreadIdx], adamLayer.w.rawGrad[kThreadIdx]);
        }
        else if(kThreadIdx < NN::kWidth * (1 + NN::kWidth))
        {
            // Update the biases
            const int biasIdx = kThreadIdx - NN::kWidth * NN::kWidth;
            Adam(mlpLayer.b[biasIdx], mlpLayer.b.grad[biasIdx], adamLayer.b[biasIdx], adamLayer.b.grad[biasIdx]);
        }
    }

    /*LiftingMLP::LiftingMLP(const Image1f& inputImage) :
        m_inputImage(inputImage)
    {
        Initialise();
    }*/

    LiftingMLP::LiftingMLP()
    {
        Initialise();
    }

    void LiftingMLP::Initialise()
    {
    }

    void LiftingMLP::Train()
    {

    }

    void LiftingMLP::Test()
    {        
        RunTensorTests(false);

        constexpr int kBatchSize = 1024;
        
        //NN::Ones rng;
        UniformDistribution rng(0, 1, 10);
        //NormalDistribution rng(0.1f, 1.0f, 10);
        Cuda::MirroredObject<NN::Model> deviceModel;
        Cuda::MirroredObject<NN::Optimiser> deviceOptimiser;
        Cuda::MirroredObject<float> deviceLoss;
        Cuda::MirroredObject<NN::Sample> deviceInput;
        Cuda::MirroredObject<NN::Sample> deviceOutput;
        Cuda::MirroredObject<NN::Sample> deviceTarget = NN::Sample({ 0.f, 0.0f, 1.f, 1.0f});

        // Allocate vectors for the samples
        Cuda::MirroredVector<float> deviceSamples(kBatchSize * NN::kWidth);
        Cuda::MirroredVector<int> deviceIndirect(kBatchSize);
        
        deviceModel->Initialise(rng);
        deviceInput->Initialise(rng);
        deviceOptimiser.Upload();
        deviceInput.Upload();        
        deviceModel.Upload();

        KernelData kernelData;
        kernelData.mlp = deviceModel.GetDeviceData();
        kernelData.inputVec = deviceInput.GetDeviceData();
        kernelData.outputVec = deviceOutput.GetDeviceData();
        kernelData.targetVec = deviceTarget.GetDeviceData();
        kernelData.loss = deviceLoss.GetDeviceData();
        kernelData.optimiser = deviceOptimiser.GetDeviceData();

        constexpr int kNumEpochs = 1000;
        HighResTimer timer;
        for (int epochIdx = 0; epochIdx < kNumEpochs; ++epochIdx)
        {
            // Estimate the gradients
            constexpr int kThreadsPerEstimateBlock = NN::kWidth * NN::kWidth;
            EstimateGradients << <NN::kMiniBatchSize, kThreadsPerEstimateBlock >> > (kernelData);

            // Reduce gradients
            for (int span = NN::kMiniBatchSize >> 1, stride = 2; span >= 2; span >>= 1, stride <<= 1)
            {
                ReduceGradients << <span, kThreadsPerEstimateBlock >> > (kernelData, span, stride);
            }

            // Optimiser step
            constexpr int kNumStepThreads = NN::kWidth * (NN::kWidth + 1);
            constexpr int kNumStepBlocks = NN::kDepth;
            Descend << < kNumStepBlocks, kNumStepThreads >> > (kernelData);
            
            IsOk(cudaDeviceSynchronize());

            //if (timer.Get() > 0.5 || epochIdx % 100 == 0)
            {
                /*deviceLoss.Download();
                std::printf("\n************************************************\n%i: Loss: %.10f\n", epochIdx, *deviceLoss);
                timer.Reset();

                deviceOutput->Print();
                NL();

                deviceOutput.Download();
                deviceModel.Download();
                deviceOptimiser.Download();*/

                /*for (int layerIdx = 0; layerIdx < NN::kDepth; ++layerIdx)
                {
                    std::printf("%i:\n", layerIdx);
                    deviceModel->layers[layerIdx].w.Print();
                    deviceModel->layers[layerIdx].b.Print();
                    deviceModel->layers[layerIdx].w.PrintGrad();
                    deviceModel->layers[layerIdx].b.PrintGrad();
                    std::printf("\n-----\n");
                    deviceOptimiser->layers[layerIdx].w.Print();
                    deviceOptimiser->layers[layerIdx].b.Print();
                    deviceOptimiser->layers[layerIdx].w.PrintGrad();
                    deviceOptimiser->layers[layerIdx].b.PrintGrad();
                    std::printf("\n-----\n");
                }*/

                /*deviceModel->layers[2].w.Print();
                deviceModel->layers[2].b.Print();
                std::printf("\n");
                deviceModel->layers[2].w.PrintGrad();
                deviceModel->layers[2].b.PrintGrad();
                std::printf("\n");
                std::printf("\n");
                deviceOptimiser->layers[2].w.Print();
                deviceOptimiser->layers[2].b.Print();
                std::printf("\n");
                deviceOptimiser->layers[2].w.PrintGrad();
                deviceOptimiser->layers[2].b.PrintGrad();
                std::printf("\n");
                std::printf("\n");
                deviceOutput->Print();
                std::printf("\n");
                std::printf("\n");
                deviceTarget->Print();
                std::printf("\n");
                std::printf("\n");*/
            }
        }       
    }
}