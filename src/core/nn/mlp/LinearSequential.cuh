#pragma once

#include "Ctx.cuh"
#include "../Activation.cuh"
#include "core/utils/TemplateUtils.h"

namespace Flair
{
    namespace NN
    {
        // Fully-connected layer with weights and biases
        template<int N, int M, bool HasGrad = false>
        struct Linear
        {
        public:
            enum : int
            {
                kHasGrad = HasGrad,
                kN = N,
                kM = M,
                kMaxDim = (N < M) ? M : N,
                kConcurrency = M * N,
                kNumParams = N*M + M
            };
            

            Tensor2D<N, M, HasGrad>   w; 
            Tensor1D<M, HasGrad>      b; 

        public:

            __inline__ __host__ __device__ void ZeroGrad()
            {
                w.ZeroGrad();
                b.ZeroGrad();
            }
        };        

        template<typename... Layers>
        struct LinearSequential
        {
        private:
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
                ([&](int con) { maxCon = (con > maxCon) ? con : maxCon; }(Layers::kConcurrency), ...);
                return maxCon;
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
                kConcurrency = GetMaxConcurrency(),

                // Total number of parameters in the model
                // FIXME: Due to a bug in the MSVC compiler, we can't use a fold to query Layers::kNumParams direct. 
                // The number of parameters must therefore be deduced from the size of the pack, however this might not be correct (e.g. if gradients are enabled)
                kNumParams = SizeOfPack<Layers...>::kValue / sizeof(float)
            };  

        private:

            //***************************** Forward *****************************

            template<int LayerIdx, typename Ctx> 
            __forceinline__ __device__ static void CacheActivations(Ctx&) { }

            template<int LayerIdx, typename PolicyT>
            __forceinline__ __device__ static void CacheActivations(TrainingCtx<PolicyT>& ctx)
            {
                ctx.acts[LayerIdx][kThreadIdx] = ctx.state[kThreadIdx];
            }

            template<typename Ctx, int LayerIdx, typename Layer, typename... Next>
            struct ForwardRecurse
            {
                __forceinline__ __device__ static void F(Ctx& ctx, const float* data)
                {
                    __syncthreads();

                    const Layer& layer = *reinterpret_cast<const Layer*>(data);
                    using Policy = typename Ctx::Policy;

                    // Multiply the state by the layer weights
                    Mul(layer.w, ctx.state, ctx.state, ctx.scratch);

                    if (kThreadIdx < Layer::kM)
                    {
                        // Add the bias
                        ctx.state[kThreadIdx] += layer.b[kThreadIdx];

                        // Apply leaky ReLU activation, except on the last layer
                        if (LayerIdx != kDepth - 1)
                        {
                            Policy::Hyper::Activation::F(ctx.state[kThreadIdx]);
                        }

                        // Cache the feed-forward intermediate activations in this layer for use during backprop
                        CacheActivations<LayerIdx>(ctx);
                    }

                    // Recurse to the next layer
                    ForwardRecurse<Ctx, LayerIdx + 1, Next...>::F(ctx, data + sizeof(Layer) / sizeof(float));
                }
            };

            template<typename Ctx, int LayerIdx>
            struct ForwardRecurse<Ctx, LayerIdx, Terminator>
            {
                __forceinline__ __device__ static void F(Ctx&, const float*) {}
            };

            //***************************** Backward *****************************

            template<typename Ctx, int LayerIdx, typename Layer, typename... Next>
            struct BackwardRecurse
            {
                __forceinline__ __device__ static void F(Ctx& ctx, float* data)
                {
                    // Work in reverse from the last layer
                    BackwardRecurse<Ctx, LayerIdx + 1, Next...>::F(ctx, data + sizeof(Layer) / sizeof(float));

                    // Col -> source neuron. Row -> destination neuron.
                    const int rowIdx = kThreadIdx % Layer::kM;
                    const int colIdx = kThreadIdx / Layer::kM;
                    Layer& layer = *reinterpret_cast<Layer*>(data);
                    using Policy = typename Ctx::Policy;

                    __syncthreads();
                    if (kThreadIdx < Layer::kM && LayerIdx != kDepth - 1)
                    {
                        // Derivative of activation at this layer (except last layer)
                        ctx.error[kThreadIdx] *= Policy::Hyper::Activation::dF(ctx.acts[LayerIdx][kThreadIdx]);
                    }

                    // Backpropagate the error by the transpose of the weight matrix and cache as a temporary state
                    __syncthreads();
                    if (LayerIdx != 0) { MulT(layer.w, ctx.error, ctx.state, ctx.scratch); }

                    // Repurpose the memory used to store the weights with the gradients of the weights
                    __syncthreads();
                    layer.w(colIdx, rowIdx) = ctx.error[rowIdx] * ((LayerIdx == 0) ? ctx.input[colIdx] : ctx.acts[LayerIdx - 1][colIdx]);
                    if (kThreadIdx < Layer::kN)
                    {
                        layer.b[kThreadIdx] = ctx.error[kThreadIdx];
                    }

                    // Update the error to its backpropagated derivative
                    __syncthreads();
                    if (kThreadIdx < Layer::kN) { ctx.error[kThreadIdx] = ctx.state[kThreadIdx]; }

                    /*if (kBlockIdx == 0)
                    {
                        printf("%f ", layer.w(colIdx, rowIdx));
                    }
                    __syncthreads();
                    if(kBlockIdx == 0 && kThreadIdx == 0) printf("\n\n");*/
                }
            };

            template<typename Ctx, int LayerIdx>
            struct BackwardRecurse<Ctx, LayerIdx, Terminator>
            {
                __forceinline__ __device__ static void F(Ctx&, float*) { }
            };

            template<typename RNG, typename Layer, typename... Next>
            struct InitialiseRecurse
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

                    InitialiseRecurse<RNG, Next...>::F(data + Layer::kNumParams, rng);
                }
            };

            template<typename RNG>
            struct InitialiseRecurse<RNG, Terminator>
            {
                __host__ static void F(float* data, RNG& rng) {}
            };

            __host__ void GetLayerOffsetsImpl(std::vector<int>&) {}
            
            template<typename Layer, typename... Next>
            __host__ void GetLayerOffsetsImpl(std::vector<int>& offsets)
            {
                offsets.push_back(sizeof(Layer) / sizeof(float));
                GetLayerOffsetsImpl<Next...>(offsets);
            }

         public:
             template<typename RNG>
             __inline__ __host__ static void Initialise(std::vector<float>& data, RNG& rng)
             {
                 InitialiseRecurse<RNG, Layers..., Terminator>::F(data.data(), rng);
             }

             template<typename Ctx>
             __forceinline__ __device__ static void Forward(Ctx& ctx)
             {
                 ForwardRecurse<Ctx, 0, Layers..., Terminator>::F(ctx, ctx.mlpData);
             }

             template<typename Ctx>
             __forceinline__ __device__ static void Backward(Ctx& ctx)
             {
                 BackwardRecurse<Ctx, 0, Layers..., Terminator>::F(ctx, ctx.mlpData);
             }

        };

    }
}
