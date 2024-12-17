#pragma once

#include "LinearSequential.cuh"

namespace Flair
{
    namespace NN    
    {
        template<template<typename...> class Model, typename... Layers>
        struct LinearSequentialEvaluator<ComputeDevice::kCPU, Model<Layers...>> : public LinearSequential<Layers...>
        {
            using Super = LinearSequential<Layers...>;

            //***************************** Forward *****************************

            template<int LayerIdx, typename Ctx>
            __forceinline__ __host__ static void CacheActivations(Ctx&, const int) { }

            template<int LayerIdx, typename PolicyT>
            __forceinline__ __host__ static void CacheActivations(TrainingCtx<PolicyT>& ctx, const int m)
            {
                ctx.acts[LayerIdx][m] = ctx.state[m];
            }

            template<typename Ctx, int LayerIdx, typename Layer, typename... Next>
            struct ForwardRecursor
            {
                __forceinline__ __host__ static void F(Ctx& ctx, const float* data)
                {
                    const Layer& layer = *reinterpret_cast<const Layer*>(data);

                    ctx.state = Mul(layer.w, ctx.state);

                    for (int m = 0; m < Layer::kM; ++m) 
                    { 
                        // Add the bias
                        ctx.state[m] += layer.b[m]; 

                        // Apply leaky ReLU activation, except on the last layer
                        if (LayerIdx != kDepth - 1)
                        {
                            Ctx::Policy::Hyper::Activation::F(ctx.state[m]);
                        }                    

                        // Cache the feed-forward intermediate activations in this layer for use during backprop
                        CacheActivations<LayerIdx>(ctx, m);
                    }

                    // Recursor to the next layer
                    ForwardRecursor<Ctx, LayerIdx + 1, Next...>::F(ctx, data + sizeof(Layer) / sizeof(float));
                }
            };

            template<typename Ctx, int LayerIdx>
            struct ForwardRecursor<Ctx, LayerIdx, Super::Terminator>
            {
                __forceinline__ __host__ static void F(Ctx&, const float*) {}
            };

            //***************************** Backward *****************************

            template<typename Ctx, int LayerIdx, typename Layer, typename... Next>
            struct BackwardRecursor
            {
                __forceinline__ __host__ static void F(Ctx& ctx, float* data)
                {
                    // Work in reverse from the last layer
                    BackwardRecursor<Ctx, LayerIdx + 1, Next...>::F(ctx, data + sizeof(Layer) / sizeof(float));       

                    Layer& layer = *reinterpret_cast<Layer*>(data);

                    // Derivative of activation at this layer (except last layer)
                    if (LayerIdx != kDepth - 1)
                    {
                        for (int m = 0; m < Layer::kM; ++m)
                        {
                            ctx.error[m] *= Ctx::Policy::Hyper::Activation::dF(ctx.acts[LayerIdx][m]);
                        }
                    }

                    // Backpropagate the error by the transpose of the weight matrix and cache as a temporary state
                    if (LayerIdx != 0) { ctx.state = MulT(layer.w, ctx.error); }

                    // Update the weights
                    for (int n = 0, i = 0; n < Layer::kN; ++n)
                    {
                        for (int m = 0; m < Layer::kM; ++m, ++i)
                        {
                            layer.w[i] = ctx.error[m] * ((LayerIdx == 0) ? ctx.input[n] : ctx.acts[LayerIdx - 1][n]);
                        }
                    }

                    // Update the biases
                    layer.b = ctx.error;

                    // Update the error to its backpropagated derivative
                    ctx.error = ctx.state;
                }
            };
            template<typename Ctx, int LayerIdx> struct BackwardRecursor<Ctx, LayerIdx, Super::Terminator>
            {
                __forceinline__ __host__ static void F(Ctx&, float*) { }
            };

         public:
             template<typename Ctx>
             __forceinline__ __host__ static void Forward(Ctx& ctx)
             {
                 ForwardRecursor<Ctx, 0, Layers..., Terminator>::F(ctx, ctx.mlpData);
             }

             template<typename Ctx>
             __forceinline__ __host__ static void Backward(Ctx& ctx)
             {
                 BackwardRecursor<Ctx, 0, Layers..., Terminator>::F(ctx, ctx.mlpData);
             }


        };
    }
}
