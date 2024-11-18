#include "LiftingMLP.cuh"
#include "tests/cuda/TensorTests.cuh"
#include "core/utils/HighResTimer.h"

namespace Flair
{      
    struct KernelData
    {
        NN::Model* mlp = nullptr;
        NN::Sample* inputVec = nullptr;
        NN::Sample* targetVec = nullptr;
        NN::Sample* outputVec = nullptr;
        NN::Optimiser* optimiser = nullptr;
        float* loss = nullptr;
    };

    struct IterateCtx
    {
        __host__ __device__ IterateCtx() {}
        
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
            float                   scratch1D[NN::kWidth * NN::kWidth];    // Scratch memory for accumulating values during tensor multiplication
        };
    };

    struct StepCtx
    {
        NN::Optimiser optimiser;
    };

    __inline__ __device__ void Forward(IterateCtx& ctx)
    {
        for (int layerIdx = 0; layerIdx < NN::kDepth; ++layerIdx)
        {
            // Multiply the state by the layer weights
            Mul<false>(ctx.mlp.layers[layerIdx].w, ctx.state, ctx.scratch2D);

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
                ctx.acts[layerIdx][kThreadIdx] = ctx.state[kThreadIdx];
            }
        }
    }

    __inline__ __device__ void Backward(IterateCtx& ctx)
    {
        // Row -> source neuron. Col -> destination neuron.
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
            layer.w.grad[colIdx][rowIdx] = ctx.error[colIdx] * ((layerIdx == 0) ? ctx.input[rowIdx] : ctx.acts[layerIdx - 1][rowIdx]);
            if (rowIdx == 0)
            {
                layer.b.grad[colIdx] = ctx.error[colIdx];
            }

            if (layerIdx != 0)
            {
                // Multiply the error by the transpose of the weight matrix 
                Mul<true>(ctx.mlp.layers[layerIdx].w, ctx.error, ctx.scratch2D);
            }
        }
    }

    __inline__ __device__ float L1(IterateCtx& ctx)
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

    // Zeroes network gradients. 
    // NOTE: We assume each block has the same number of threads as kWidth^2
    __global__ void ZeroGrad(KernelData kernelData)
    {
        kernelData.mlp[kBlockIdx / NN::kDepth].layers[kBlockIdx % NN::kDepth].ZeroGrad();
    }

    __global__ void Iterate(KernelData kernelData)
    {
        __shared__ IterateCtx ctx;

        // Copy MLP data out of global memory into shared memory
        if (kThreadIdx == 0)
        {
            CudaAssertDebug(kernelData.mlp);
            CudaAssertDebug(kernelData.targetVec);
            CudaAssertDebug(kernelData.inputVec);

            ctx.mlp = kernelData.mlp[blockIdx.x];
            ctx.target = kernelData.targetVec[blockIdx.x];
            ctx.state = ctx.input = kernelData.inputVec[blockIdx.x];
        }

        __syncthreads();

        // Feed forward pass
        Forward(ctx);

        kernelData.outputVec[blockIdx.x] = ctx.state;

        // Calculate the loss and error for the last layer
        kernelData.loss[blockIdx.x] = L1(ctx);

        // Back propagate error and accumulate gradients
        Backward(ctx);

        // Copy the updated gradients back to global memory                
        __syncthreads();
        for (int layerIdx = 0; layerIdx < NN::kDepth; ++layerIdx)
        {
            kernelData.mlp[blockIdx.x].layers[layerIdx].w.rawGrad[kThreadIdx] = ctx.mlp.layers[layerIdx].w.rawGrad[kThreadIdx];
            if (kThreadIdx < NN::kWidth)
            {
                kernelData.mlp[blockIdx.x].layers[layerIdx].b.grad[kThreadIdx] = ctx.mlp.layers[layerIdx].b.grad[kThreadIdx];
            }
        }
    }

    __forceinline__ __device__ void Adam(float& param, const float& grad, float& mo1, float& mo2)
    {
        constexpr float kAlpha = 1e-2;
        constexpr float kBeta1 = 0.9;
        constexpr float kBeta2 = 0.999;
        constexpr float kEpsilon = 1e-8;

        CudaAssert(mo2 >= 0);

        // Compute biased moments
        mo1 = kBeta1 * mo1 + (1 - kBeta1) * grad;
        mo2 = fmaxf(0.f, kBeta2 * mo2 + (1 - kBeta2) * (grad * grad));

        CudaAssert(mo2 >= 0);

        // Update the parameter
        param -= kAlpha * (mo1 / (1 - kBeta1)) / (sqrtf(mo2 / (1 - kBeta2)) + kEpsilon);
    }

    __global__ void Step(KernelData kernelData)
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
        RunTensorTests();
        printf_green("Tests passed!\n");
        
        //NN::Ones rng;
        UniformDistribution rng(0, 1, 10);
        //NormalDistribution rng(0.1f, 1.0f, 10);
        Cuda::HostDeviceObject<NN::Model> deviceModel;
        Cuda::HostDeviceObject<NN::Optimiser> deviceOptimiser;
        Cuda::HostDeviceObject<float> deviceLoss;
        Cuda::HostDeviceObject<NN::Sample> deviceInput;
        Cuda::HostDeviceObject<NN::Sample> deviceOutput;
        Cuda::HostDeviceObject<NN::Sample> deviceTarget = NN::Sample({ 0.f, 0.0f, 1.f, 1.0f});
        
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
            // Forward-back propagate
            constexpr int kThreadsPerIterateBlock = NN::kWidth * NN::kWidth;
            Iterate << <1, kThreadsPerIterateBlock >> > (kernelData);
            IsOk(cudaDeviceSynchronize());

            // Optimiser step
            constexpr int kThreadsPerStepBlock = NN::kWidth * (NN::kWidth + 1);
            Step << <NN::kDepth, kThreadsPerStepBlock >> > (kernelData);
            IsOk(cudaDeviceSynchronize());

            //if (timer.Get() > 0.5 || epochIdx % 100 == 0)
            {
                deviceLoss.Download();
                std::printf("\n************************************************\n%i: Loss: %.10f\n", epochIdx, *deviceLoss);
                timer.Reset();

                deviceOutput->Print();
                NL();

                deviceOutput.Download();
                deviceModel.Download();
                deviceOptimiser.Download();

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