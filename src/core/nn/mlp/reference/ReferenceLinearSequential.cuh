#pragma once

#include "Ctx.cuh"
#include "../Activation.cuh"
#include "core/utils/TemplateUtils.h"
#include "../LinearSequential.cuh"

namespace Flair
{
    namespace NN
    {
        template<typename... Layers>
        struct ReferenceLinearSequential : public LinearSequential<Layers...>
        {
            Super = NN::LinearSequential<Layers...>;

        private:

            //***************************** Forward *****************************

            template<typename Ctx, int LayerIdx, typename Layer, typename... Next>
            struct ForwardRecursor
            {
                __host__ static void F(Ctx& ctx, const float* data)
                {
                    __syncthreads();

                    const Layer& layer = *reinterpret_cast<const Layer*>(data);

                    MulT(

                    if (kThreadIdx < Layer::kN) ctx.error[kThreadIdx] = ctx.state[kThreadIdx];

                    if (kThreadIdx < ctx.scratch.Size()) ctx.scratch.At(kThreadIdx) = 0;

                    // Multiply the state by the layer weights
                    Mul(layer.w, ctx.state, ctx.state, ctx.scratch);

                    __syncthreads();
                    if (kThreadIdx < Layer::kM)
                    {
                        // Add the bias
                        ctx.state[kThreadIdx] += layer.b[kThreadIdx];

                        // Apply leaky ReLU activation, except on the last layer
                        if (LayerIdx != kDepth - 1)
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
            struct ForwardRecursor<Ctx, LayerIdx, Terminator>
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
                    if (kThreadIdx < kM && LayerIdx != kDepth - 1)
                    {
                        // Derivative of activation at this layer (except last layer)
                        ctx.error[kThreadIdx] *= Ctx::Policy::Hyper::Activation::dF(ctx.acts[LayerIdx][kThreadIdx]);
                    }

                    // Backpropagate the error by the transpose of the weight matrix and cache as a temporary state
                    __syncthreads();
                    if (LayerIdx != 0) { MulT(layer.w, ctx.error, ctx.state, ctx.scratch); }

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
            template<typename Ctx, int LayerIdx> struct BackwardRecursor<Ctx, LayerIdx, Terminator>
            {
                __forceinline__ __device__ static void F(Ctx&, float*) { }
            };

            template<typename RNG, typename Layer, typename... Next>
            struct InitialiseRecursor
            {
                __host__ static void F(float* data, RNG& rng)
                {
                    Layer& layer = *reinterpret_cast<Layer*>(data);

                    layer.w.Initialise(rng);
                    layer.b.Initialise(rng);
                    if (Layer::kHasGrad)
                    {
                        layer.w.ZeroGrad();
                        layer.b.ZeroGrad();
                    }

                    InitialiseRecursor<RNG, Next...>::F(data + Layer::kNumParams, rng);
                }
            };
            template<typename RNG> struct InitialiseRecursor<RNG, Terminator> { __host__ static void F(float* data, RNG& rng) {} };

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
                    auto& w = reinterpret_cast<Layer*>(data)->w;
                    w = w.Transpose();
                    TransposeRecursor<Next...>::F(data + sizeof(Layer) / sizeof(float));
                }
            };
            template<> struct TransposeRecursor<Terminator> { __inline__ __host__ static void F(float*) { } };

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
