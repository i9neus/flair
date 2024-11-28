#pragma once

//#include "NNUtils.cuh"
#include "core/utils/cuda/CudaVector.cuh"
#include "core/math/MathUtils.h"
#include <random>
#include <set>

namespace Flair
{
    namespace NN
    {
        __global__ void FillSequentialKernel(int* data, const int dataSize)
        {
            if (kKernelIdx < dataSize)
            {
                data[kKernelIdx] = kKernelIdx;
            }
        }       

        __global__ void ShuffleKernel(int* dest, const int* src, const int dataSize, uint32_t offset)
        {
            if (kKernelIdx < dataSize)
            {
                dest[kKernelIdx] = (src[src[kKernelIdx]] + offset) % dataSize;
            }
        }

        __global__ void BijectiveShuffleKernel(int* data, const int dataSize, const uint32_t W, const uint32_t Q, uint32_t offset)
        {
            const uint32_t k = kKernelIdx / Q;
            const uint32_t i = k & ((1 << W) - 1);
            const uint32_t j = RadicalInverse(i) >> (31 - W);
            const uint32_t p = (k & ~((1 << W) - 1)) << 1;

            const uint32_t r = kKernelIdx % Q;
            const int i0 = (p + i) * Q + r;
            const int i1 = (p + j) * Q + r;

            if (i0 < dataSize && i1 < dataSize)
            {
                Swap(data[(i0 + offset) % dataSize], data[(i1 + offset) % dataSize]);
            }
        }        

        class Indirection
        {
        private:
            Cuda::Vector<int, kCudaMemMirrored>         m_indices;
            Cuda::Vector<int, kCudaMemMirrored>         m_swap;
            std::mt19937                                m_mt;
            std::uniform_int_distribution<int>          m_rng;

        public:
            Indirection(const int size, const uint32_t seed = 0) : 
                m_mt(seed),
                m_indices(size),
                m_swap(size)
            {}

            __host__ Cuda::Vector<int, kCudaMemMirrored>& operator*() { return m_indices; }
            __host__ Cuda::Vector<int, kCudaMemMirrored>* operator->() { return &m_indices; }

            // Check that the each index in the vector maps to one and only one other element 
            __host__ void CheckBijective()
            {
                m_indices.Download();
                std::set<int> numbers;
                for (auto& i : m_indices)
                {
                    AssertMsg(numbers.find(i) == numbers.end(), "Map is not bijective!");
                    numbers.emplace(i);
                }
                printf_green("Map is bijective!\n");
            }

            __host__ void Randomise()
            {
                // Fill array with sequential integers
                for (int i = 0; i < m_indices.Size(); ++i) { m_indices[i] = i; }

                // Step through the array and place an element from the sorted part into a random position in the unsorted part
                for (int i = 0; i < m_indices.Size(); ++i)
                {
                    const int j = (i < m_indices.Size() / 2) ? (i + m_rng(m_mt) % (m_indices.Size() - i)) : (m_rng(m_mt) % (1 + i));
                    Swap(m_indices[i], m_indices[j]);
                }

                //CheckBijective(m_indices);
                m_indices.Upload();
            }

            __host__ void Sequential()
            {
                const int kNumBlocks = (m_indices.Size() + 255) / 256;
                FillSequentialKernel << <kNumBlocks, 256 >> > (m_indices.GetDeviceData(), m_indices.Size());
                IsOk(cudaDeviceSynchronize());
            }

            __host__ void Shuffle()
            {
                const int offset = m_rng(m_mt) % m_indices.Size();
                const int kNumBlocks = (m_indices.Size() + 255) / 256;
                ShuffleKernel << < kNumBlocks, 256 >> > (m_swap.GetDeviceData(), m_indices.GetDeviceData(), m_indices.Size(), offset);
                IsOk(cudaDeviceSynchronize());
                
                Swap(m_swap, m_indices);
            }
        };
    }
}