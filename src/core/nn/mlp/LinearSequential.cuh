#pragma once

#include "Ctx.cuh"
#include "../Activation.cuh"
#include "core/utils/TemplateUtils.h"
#include <functional>
#include "../ParameterInitialiser.cuh"

namespace Flair
{
    namespace NN
    {
        // Fully-connected layer with weights and biases
        template<int N, int M, bool HasGrad = false>
        struct Linear
        {
        public:
            using WeightsT = Tensor2D<N, M, HasGrad>;
            using BiasesT = Tensor1D<M, HasGrad>;

            WeightsT  w;
            BiasesT   b;

            enum : int
            {
                kHasGrad = HasGrad,
                kN = N,
                kM = M,
                kMaxDim = (N < M) ? M : N,
                kMaxConcurrency = Tensor2D<N, M, HasGrad>::kMaxConcurrency,
                kNumParams = N * M + M
            };

        public:

            __host__ std::string Format() const
            {
                return w.Format(false, false) + "\n" + b.Format(false, false) + "\n";
            }

            __inline__ __host__ __device__ void ZeroGrad()
            {
                w.ZeroGrad();
                b.ZeroGrad();
            }
        };

        template<typename... Layers>
        struct LinearSequential
        {
        protected:
            enum class Terminator {};

            __host__ __device__ static constexpr int GetMaxWidth()
            {
                int maxWidth = 0;
                ([&](int width) { maxWidth = (width > maxWidth) ? width : maxWidth; }(Layers::kMaxDim), ...);
                return maxWidth;
            }

            __host__ __device__ static constexpr int GetMaxConcurrency()
            {
                int maxCon = 0;
                ([&](int con) { maxCon = (con > maxCon) ? con : maxCon; }(Layers::kMaxConcurrency), ...);
                return maxCon;
            }

            template<typename LayerN> __host__ __device__ constexpr static bool VerifyLayerConnectivityRecursor() { return true; }

            template<typename Layer1, typename Layer2, typename... Next>
            __host__ __device__ constexpr static bool VerifyLayerConnectivityRecursor()
            {
                static_assert(Layer1::kM == Layer2::kN, "LinearSequential connectivity is invalid.");
                return VerifyLayerConnectivityRecursor<Layer2, Next...>();
            }

        public:
            using InputLayer = FirstOf<Layers...>::Type;
            using OutputLayer = LastOf<Layers...>::Type;

            enum : int
            {
                // Depth of the model
                kDepth = sizeof...(Layers),

                // Width of the input layer
                kInputWidth = InputLayer::kN,

                // Width of the output layer
                kOutputWidth = OutputLayer::kM,

                // Maximum width of the span of all layers
                kMaxWidth = GetMaxWidth(),

                // Maximum number of concurrent threads per block for concurrent evaluation of this model
                kMaxConcurrency = GetMaxConcurrency(),

                // Check to make sure the number of the parameters output by the last layer match that the of next layer
                kIsValidConnected = VerifyLayerConnectivityRecursor<Layers...>(),

                // Total number of parameters in the model
                // FIXME: Due to a bug in the MSVC compiler, we can't use a fold to query Layers::kNumParams direct. 
                // The number of parameters must therefore be deduced from the size of the pack, however this might not be correct (e.g. if gradients are enabled)
                kNumParams = SizeOfPack<Layers...>::kValue / sizeof(float), 

                // Whether or not the activation function is applied on the last layer (important when using ReLU)
                kActivateLastLayer = 0
            };

        protected:

            template<int LayerIdx, typename Layer, typename... Next>
            struct InitialiseRecursor
            {
                __host__ static void F(float* data, ParameterInitialiser& rng)
                {
                    Layer& layer = *reinterpret_cast<Layer*>(data);

                    // Initialise the weights...
                    for (int m = 0; m < Layer::kM; ++m)
                    {
                        for (int n = 0; n < Layer::kN; ++n)
                        {
                            layer.w(n, m) = rng(LayerIdx, Layer::kN, Layer::kM);
                        }
                    }
                    // ...and the biases.
                    for (int m = 0; m < Layer::kM; ++m)
                    {
                        layer.b[m] = rng(LayerIdx, 1, Layer::kM);
                    }

                    // Clear the gradients
                    if (Layer::kHasGrad)
                    {
                        layer.w.ZeroGrad();
                        layer.b.ZeroGrad();
                    }

                    // Recurse to the next layer
                    InitialiseRecursor<LayerIdx + 1, Next...>::F(data + Layer::kNumParams, rng);
                }

            };
            template<int LayerIdx> struct InitialiseRecursor<LayerIdx, Terminator> { __host__ static void F(float* data, ParameterInitialiser& rng) {} };

            template<typename Layer, typename... Next>
            struct FormatRecursor
            {
                __inline__ __host__ static std::string F(const float* data)
                {
                    const Layer& layer = *reinterpret_cast<const Layer*>(data);
                    return "{\n" +
                        layer.Format() +
                        "}\n" +
                        FormatRecursor<Next...>::F(data + sizeof(Layer) / sizeof(float));
                }
            };
            template<> struct FormatRecursor<Terminator> { __inline__ __host__ static std::string F(const float*) { return ""; } };

            // Transpose the weight matrices (useful for switching between row- and column-major order)
            template<typename Layer, typename... Next>
            struct TransposeRecursor
            {
                __inline__ __host__ static void F(float* data)
                {
                    reinterpret_cast<Layer*>(data)->w.FromRowMajor();
                    TransposeRecursor<Next...>::F(data + sizeof(Layer) / sizeof(float));
                }
            };
            template<> struct TransposeRecursor<Terminator> { __inline__ __host__ static void F(float*) { } };

        public:
            __inline__ __host__ static void Initialise(std::vector<float>& data, ParameterInitialiser& rng)
            {
                AssertFmt(data.size() >= kNumParams, "Param data size %i does not match model size %i.", data.size(), kNumParams);
                InitialiseRecursor<sizeof...(Layers), Layers..., Terminator>::F(data.data(), rng);
            }

            __inline__ __host__ static void Transpose(std::vector<float>& data)
            {
                AssertFmt(data.size() >= kNumParams, "Param data size %i does not match model size %i.", data.size(), kNumParams);
                TransposeRecursor<Layers..., Terminator>::F(data.data());
            }

            __inline__ __host__ static std::string Format(const std::vector<float>& data)
            {
                AssertFmt(data.size() >= kNumParams, "Param data size %i does not match model size %i.", data.size(), kNumParams);
                return FormatRecursor<Layers..., Terminator>::F(data.data());
            }
        };

        template<ComputeDevice TargetDevice, typename Model>
        struct LinearSequentialEvaluator : public Model {};

        template<template<typename...> class Model, typename... Layers>
        struct LinearSequentialEvaluator<ComputeDevice::kCUDA, Model<Layers...>> : public LinearSequential<Layers...>
        {
            using Super = LinearSequential<Layers...>;

        protected:
            //***************************** Forward *****************************

            template<int LayerIdx, typename Ctx>
            __forceinline__ __device__ static void CacheActivations(Ctx&) { }

            template<int LayerIdx, typename PolicyT>
            __forceinline__ __device__ static void CacheActivations(TrainingCtx<PolicyT>& ctx)
            {
                ctx.acts[LayerIdx][kThreadIdx] = ctx.state[kThreadIdx];
            }

            template<typename Ctx, int LayerIdx, typename Layer, typename... Next>
            struct ForwardRecursor
            {
                __forceinline__ __device__ static void F(Ctx& ctx, const float* data)
                {
                    __syncthreads();

                    const Layer& layer = *reinterpret_cast<const Layer*>(data);

                    //if (kThreadIdx < Layer::kN) ctx.error[kThreadIdx] = ctx.state[kThreadIdx];
                    //if (kThreadIdx < ctx.scratch.Size()) ctx.scratch.At(kThreadIdx) = 0;

                    // Multiply the state by the layer weights
                    Mul(layer.w, ctx.state, ctx.state, ctx.scratch);

                    __syncthreads();
                    if (kThreadIdx < Layer::kM)
                    {
                        // Add the bias
                        ctx.state[kThreadIdx] += layer.b[kThreadIdx];

                        // Apply leaky ReLU activation, except on the last layer
                        if (kActivateLastLayer || LayerIdx != kDepth - 1)
                        {
                            Ctx::Policy::Hyper::Activation::F(ctx.state[kThreadIdx]);
                        }

                        // Cache the feed-forward intermediate activations in this layer for use during backprop
                        CacheActivations<LayerIdx>(ctx);
                    }

                    //PrintActs<Layer, LayerIdx>(ctx);

                    // Recursor to the next layer
                    ForwardRecursor<Ctx, LayerIdx + 1, Next...>::F(ctx, data + sizeof(Layer) / sizeof(float));
                }
            };

            template<typename Ctx, int LayerIdx>
            struct ForwardRecursor<Ctx, LayerIdx, Super::Terminator>
            {
                __forceinline__ __device__ static void F(Ctx&, const float*) {}
            };

            //***************************** Backward *****************************

            template<typename Ctx, int LayerIdx, typename Layer, typename... Next>
            struct BackwardRecursor
            {
                __forceinline__ __device__ static void F(Ctx& ctx, float* data)
                {
                    // Work in reverse from the last layer
                    BackwardRecursor<Ctx, LayerIdx + 1, Next...>::F(ctx, data + sizeof(Layer) / sizeof(float));

                    // Col -> source neuron. Row -> destination neuron.

                    Layer& layer = *reinterpret_cast<Layer*>(data);
                    using WeightsT = typename Layer::WeightsT;
                    constexpr int kN = WeightsT::kN, kM = WeightsT::kM, kMPerThread = WeightsT::kMPerThread;

                    __syncthreads();
                    if (kThreadIdx < kM && (kActivateLastLayer || LayerIdx != kDepth - 1))
                    {
                        // Derivative of activation at this layer (except last layer)
                        ctx.error[kThreadIdx] *= Ctx::Policy::Hyper::Activation::dF(ctx.acts[LayerIdx][kThreadIdx]);
                    }

                    // Backpropagate the error by the transpose of the weight matrix and cache as a temporary state
                    __syncthreads();
                    if (kActivateLastLayer || LayerIdx != 0) { MulT(layer.w, ctx.error, ctx.state, ctx.scratch); }

                    // Repurpose the memory used to store the weights with the gradients of the weights
                    __syncthreads();
                    if (kThreadIdx < kN * WeightsT::kConcurrentM)
                    {
                        const int colIdx = kThreadIdx % kN, rowIdx = kThreadIdx / kN;
                        for (int k = 0, r = rowIdx * kMPerThread, i = kM * colIdx + r;
                            r < kM && k < kMPerThread;
                            ++k, ++r, ++i)
                        {
                            layer.w[i] = ctx.error[r] * ((LayerIdx == 0) ? ctx.input[colIdx] : ctx.acts[LayerIdx - 1][colIdx]);
                        }
                    }

                    // Update the biases
                    if (kThreadIdx < kM) { layer.b[kThreadIdx] = ctx.error[kThreadIdx]; }

                    // Update the error to its backpropagated derivative
                    __syncthreads();
                    if (kThreadIdx < kN) { ctx.error[kThreadIdx] = ctx.state[kThreadIdx]; }
                }
            };
            template<typename Ctx, int LayerIdx> struct BackwardRecursor<Ctx, LayerIdx, Super::Terminator>
            {
                __forceinline__ __device__ static void F(Ctx&, float*) { }
            };

         public:
             template<typename Ctx>
             __forceinline__ __device__ static void Forward(Ctx& ctx)
             {
                 ForwardRecursor<Ctx, 0, Layers..., Terminator>::F(ctx, ctx.mlpData);
             }

             template<typename Ctx>
             __forceinline__ __device__ static void Backward(Ctx& ctx)
             {
                 BackwardRecursor<Ctx, 0, Layers..., Terminator>::F(ctx, ctx.mlpData);
             }


        };
    }
}
